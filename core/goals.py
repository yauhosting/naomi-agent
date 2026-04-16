"""
NAOMI Agent - Autonomous Goal Tree

Hierarchical goal management with LLM-powered decomposition.
Goals are stored in SQLite and organized as a tree (parent_id references).
The system supports priority-based scheduling, automatic cascading
completion, and LLM-assisted task decomposition.
"""
import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("naomi.goals")

# Goal statuses
STATUS_ACTIVE = "active"
STATUS_PAUSED = "paused"
STATUS_COMPLETED = "completed"
STATUS_ABANDONED = "abandoned"

VALID_STATUSES = {STATUS_ACTIVE, STATUS_PAUSED, STATUS_COMPLETED, STATUS_ABANDONED}

# Priority range
MIN_PRIORITY = 1
MAX_PRIORITY = 10


@dataclass(frozen=True)
class Goal:
    """Immutable representation of a goal."""
    id: str
    title: str
    parent_id: Optional[str]
    priority: int
    status: str
    created_at: float
    completed_at: Optional[float]


class GoalTree:
    """
    Hierarchical goal manager backed by SQLite.

    Goals form a tree via parent_id. Leaf goals represent actionable
    tasks; parent goals complete automatically when all children are done.
    """

    def __init__(self, db_path: str = "data/naomi_memory.db") -> None:
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else "data", exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_table()

    def _init_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                parent_id TEXT,
                priority INTEGER DEFAULT 5 CHECK(priority >= 1 AND priority <= 10),
                status TEXT DEFAULT 'active'
                    CHECK(status IN ('active', 'paused', 'completed', 'abandoned')),
                created_at REAL NOT NULL,
                completed_at REAL,
                FOREIGN KEY (parent_id) REFERENCES goals(id)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_goals_parent ON goals(parent_id)
        """)
        self._conn.commit()

    # -- Core operations --

    def add_goal(self, title: str, parent_id: Optional[str] = None,
                 priority: int = 5) -> Goal:
        """
        Add a goal (or subgoal if parent_id is provided).

        Returns the created Goal.
        """
        if not title or not title.strip():
            raise ValueError("Goal title cannot be empty")

        priority = max(MIN_PRIORITY, min(MAX_PRIORITY, priority))

        if parent_id is not None:
            parent = self._get_goal_row(parent_id)
            if parent is None:
                raise ValueError(f"Parent goal '{parent_id}' not found")

        goal_id = uuid.uuid4().hex[:12]
        now = time.time()

        self._conn.execute(
            "INSERT INTO goals (id, title, parent_id, priority, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (goal_id, title.strip(), parent_id, priority, STATUS_ACTIVE, now),
        )
        self._conn.commit()

        created = Goal(
            id=goal_id, title=title.strip(), parent_id=parent_id,
            priority=priority, status=STATUS_ACTIVE,
            created_at=now, completed_at=None,
        )
        logger.info("Goal added: [%s] %s (priority=%d, parent=%s)",
                     goal_id, title.strip(), priority, parent_id)
        return created

    def complete_goal(self, goal_id: str) -> List[str]:
        """
        Mark a goal as completed. If all siblings under the same parent
        are now completed, cascade completion to the parent.

        Returns a list of all goal IDs that were completed (including cascaded).
        """
        row = self._get_goal_row(goal_id)
        if row is None:
            raise ValueError(f"Goal '{goal_id}' not found")
        if row["status"] == STATUS_COMPLETED:
            return [goal_id]

        now = time.time()
        completed_ids = []

        # Complete this goal
        self._conn.execute(
            "UPDATE goals SET status = ?, completed_at = ? WHERE id = ?",
            (STATUS_COMPLETED, now, goal_id),
        )
        completed_ids.append(goal_id)

        # Also complete any active children (cascading down)
        children = self._get_children(goal_id)
        for child in children:
            if child["status"] in (STATUS_ACTIVE, STATUS_PAUSED):
                self._conn.execute(
                    "UPDATE goals SET status = ?, completed_at = ? WHERE id = ?",
                    (STATUS_COMPLETED, now, child["id"]),
                )
                completed_ids.append(child["id"])

        # Cascade up: check if parent should be completed
        parent_id = row["parent_id"]
        while parent_id is not None:
            siblings = self._get_children(parent_id)
            all_done = all(
                s["status"] in (STATUS_COMPLETED, STATUS_ABANDONED)
                for s in siblings
            )
            if all_done:
                self._conn.execute(
                    "UPDATE goals SET status = ?, completed_at = ? WHERE id = ?",
                    (STATUS_COMPLETED, now, parent_id),
                )
                completed_ids.append(parent_id)
                parent_row = self._get_goal_row(parent_id)
                parent_id = parent_row["parent_id"] if parent_row else None
            else:
                break

        self._conn.commit()
        logger.info("Completed goals: %s", completed_ids)
        return completed_ids

    def get_active_goals(self, limit: int = 5) -> List[Goal]:
        """
        Return the highest-priority incomplete leaf goals.

        A leaf goal has no active children. These represent concrete
        tasks that can be worked on.
        """
        # Leaf goals: active goals with no active children
        rows = self._conn.execute("""
            SELECT g.* FROM goals g
            WHERE g.status = 'active'
              AND NOT EXISTS (
                  SELECT 1 FROM goals c
                  WHERE c.parent_id = g.id AND c.status = 'active'
              )
            ORDER BY g.priority DESC, g.created_at ASC
            LIMIT ?
        """, (limit,)).fetchall()

        return [self._row_to_goal(r) for r in rows]

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a single goal by ID."""
        row = self._get_goal_row(goal_id)
        return self._row_to_goal(row) if row else None

    def update_goal(self, goal_id: str, title: Optional[str] = None,
                    priority: Optional[int] = None,
                    status: Optional[str] = None) -> Goal:
        """Update goal fields. Returns the updated Goal."""
        row = self._get_goal_row(goal_id)
        if row is None:
            raise ValueError(f"Goal '{goal_id}' not found")

        updates = []
        params: list = []

        if title is not None:
            updates.append("title = ?")
            params.append(title.strip())

        if priority is not None:
            priority = max(MIN_PRIORITY, min(MAX_PRIORITY, priority))
            updates.append("priority = ?")
            params.append(priority)

        if status is not None:
            if status not in VALID_STATUSES:
                raise ValueError(f"Invalid status: {status}")
            updates.append("status = ?")
            params.append(status)
            if status == STATUS_COMPLETED:
                updates.append("completed_at = ?")
                params.append(time.time())

        if not updates:
            return self._row_to_goal(row)

        params.append(goal_id)
        self._conn.execute(
            f"UPDATE goals SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        self._conn.commit()

        updated_row = self._get_goal_row(goal_id)
        return self._row_to_goal(updated_row)

    # -- LLM-assisted operations --

    async def decompose(self, goal_id: str, brain: Any) -> List[Goal]:
        """
        Ask the LLM to break a goal into subgoals.

        Calls brain._think() to generate a decomposition, then creates
        subgoals in the tree.
        """
        goal = self.get_goal(goal_id)
        if goal is None:
            raise ValueError(f"Goal '{goal_id}' not found")

        # Build context: the goal and its ancestors
        context_parts = [f"Goal to decompose: {goal.title} (priority: {goal.priority})"]
        ancestors = self._get_ancestors(goal_id)
        if ancestors:
            context_parts.append("Parent goals: " + " > ".join(
                a["title"] for a in reversed(ancestors)
            ))

        # Existing children
        existing = self._get_children(goal_id)
        if existing:
            context_parts.append("Existing subgoals: " + ", ".join(
                c["title"] for c in existing
            ))

        prompt = (
            "Break down the following goal into 3-5 concrete, actionable subgoals.\n"
            "Each subgoal should be specific enough to work on directly.\n\n"
            + "\n".join(context_parts) + "\n\n"
            "Respond in valid JSON only:\n"
            '{"subgoals": [{"title": "...", "priority": N}, ...]}\n'
            "Priority 1-10 (10 = most urgent). Respond in the same language as the goal."
        )

        system = "You are a task decomposition assistant. Output JSON only."
        response = brain._think(prompt, system)

        # Parse the LLM response
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            data = json.loads(response.strip())
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse decomposition response: %s", response[:200])
            return []

        subgoals_data = data.get("subgoals", [])
        created: List[Goal] = []
        for sg in subgoals_data:
            title = sg.get("title", "").strip()
            pri = sg.get("priority", goal.priority)
            if title:
                new_goal = self.add_goal(title, parent_id=goal_id, priority=pri)
                created.append(new_goal)

        logger.info("Decomposed '%s' into %d subgoals", goal.title, len(created))
        return created

    async def suggest_next_task(self, brain: Any) -> Optional[Dict[str, Any]]:
        """
        Use the LLM to pick the best next task from active leaf goals.

        Returns a dict with the chosen goal and reasoning.
        """
        active = self.get_active_goals(limit=10)
        if not active:
            return None

        if len(active) == 1:
            return {
                "goal": self._goal_to_dict(active[0]),
                "reasoning": "Only one active task available.",
            }

        goals_text = "\n".join(
            f"- [{g.id}] {g.title} (priority: {g.priority})"
            for g in active
        )

        prompt = (
            "Given these active tasks, pick the single best one to work on next.\n"
            "Consider priority, urgency, and dependencies.\n\n"
            f"{goals_text}\n\n"
            "Respond in valid JSON only:\n"
            '{"chosen_id": "...", "reasoning": "..."}'
        )

        system = "You are a task prioritization assistant. Output JSON only."
        response = brain._think(prompt, system)

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            data = json.loads(response.strip())
        except (json.JSONDecodeError, IndexError):
            # Fallback: just pick the highest priority
            best = active[0]
            return {
                "goal": self._goal_to_dict(best),
                "reasoning": f"Highest priority task (priority={best.priority}).",
            }

        chosen_id = data.get("chosen_id", "")
        reasoning = data.get("reasoning", "")

        # Find the chosen goal
        chosen = next((g for g in active if g.id == chosen_id), None)
        if chosen is None:
            chosen = active[0]
            reasoning = f"LLM choice '{chosen_id}' not found; defaulting to highest priority."

        return {
            "goal": self._goal_to_dict(chosen),
            "reasoning": reasoning,
        }

    # -- Display --

    def get_tree(self) -> List[Dict[str, Any]]:
        """
        Return the full goal tree as a nested structure.

        Each node has: id, title, priority, status, created_at,
        completed_at, children: [...]
        """
        # Get all root goals (no parent)
        roots = self._conn.execute(
            "SELECT * FROM goals WHERE parent_id IS NULL ORDER BY priority DESC, created_at ASC"
        ).fetchall()

        return [self._build_subtree(dict(r)) for r in roots]

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics about the goal tree."""
        rows = self._conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'paused' THEN 1 ELSE 0 END) as paused,
                SUM(CASE WHEN status = 'abandoned' THEN 1 ELSE 0 END) as abandoned
            FROM goals
        """).fetchone()

        total = rows["total"] or 0
        completed = rows["completed"] or 0

        return {
            "total": total,
            "active": rows["active"] or 0,
            "completed": completed,
            "paused": rows["paused"] or 0,
            "abandoned": rows["abandoned"] or 0,
            "completion_rate": round(completed / max(total, 1) * 100, 1),
        }

    # -- Internal helpers --

    def _get_goal_row(self, goal_id: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM goals WHERE id = ?", (goal_id,)
        ).fetchone()

    def _get_children(self, parent_id: str) -> List[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM goals WHERE parent_id = ? ORDER BY priority DESC, created_at ASC",
            (parent_id,),
        ).fetchall()

    def _get_ancestors(self, goal_id: str) -> List[Dict[str, Any]]:
        """Walk up the tree and collect ancestor goals."""
        ancestors: List[Dict[str, Any]] = []
        current = self._get_goal_row(goal_id)
        while current and current["parent_id"]:
            parent = self._get_goal_row(current["parent_id"])
            if parent is None:
                break
            ancestors.append(dict(parent))
            current = parent
        return ancestors

    def _build_subtree(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively build a nested tree from a goal node."""
        children_rows = self._get_children(node["id"])
        children = [self._build_subtree(dict(c)) for c in children_rows]

        return {
            "id": node["id"],
            "title": node["title"],
            "priority": node["priority"],
            "status": node["status"],
            "created_at": node["created_at"],
            "completed_at": node["completed_at"],
            "children": children,
        }

    @staticmethod
    def _row_to_goal(row: sqlite3.Row) -> Goal:
        return Goal(
            id=row["id"],
            title=row["title"],
            parent_id=row["parent_id"],
            priority=row["priority"],
            status=row["status"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )

    @staticmethod
    def _goal_to_dict(goal: Goal) -> Dict[str, Any]:
        return {
            "id": goal.id,
            "title": goal.title,
            "parent_id": goal.parent_id,
            "priority": goal.priority,
            "status": goal.status,
            "created_at": goal.created_at,
            "completed_at": goal.completed_at,
        }
