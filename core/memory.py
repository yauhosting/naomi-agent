"""
NAOMI Agent - Memory System v2
Inspired by Claude Code's memory architecture:
- 3-tier memory: short-term, long-term, session context
- Progressive context compression (rule-based -> LLM summary)
- Semantic retrieval with freshness decay
- Structured 9-section summaries
- Deduplication interlock
"""
import sqlite3
import json
import time
import os
import math
from typing import Optional, List, Dict, Any

# Approximate token count (rough: 1 token ~ 4 chars for English, ~2 for CJK)
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text) // 3  # Conservative estimate


class Memory:
    # Context budget limits (in estimated tokens)
    MAX_CONTEXT_TOKENS = 8000
    MAX_SHORT_TERM = 50
    FRESHNESS_DECAY_DAYS = 7  # Memories older than this get lower weight

    def __init__(self, db_path="data/naomi_memory.db"):
        # Accept both string path and config dict
        if isinstance(db_path, dict):
            db_path = db_path.get("db_path", "data/naomi_memory.db")
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else "data", exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_tables()
        self._extraction_lock = False  # Interlock: prevent duplicate extractions

    def _init_tables(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS short_term (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            timestamp REAL NOT NULL,
            expires_at REAL
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS long_term (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            importance INTEGER DEFAULT 5,
            access_count INTEGER DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            tags TEXT DEFAULT '[]'
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            tool_command TEXT,
            install_command TEXT,
            success_count INTEGER DEFAULT 0,
            fail_count INTEGER DEFAULT 0,
            last_used REAL,
            learned_at REAL NOT NULL
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS persona (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            value TEXT NOT NULL,
            updated_at REAL NOT NULL
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS task_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            result TEXT,
            started_at REAL,
            completed_at REAL,
            duration REAL,
            brain_mode TEXT DEFAULT 'left'
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp REAL NOT NULL,
            persona TEXT DEFAULT 'naomi'
        )''')
        # Migrate: add persona column if missing (existing DBs)
        try:
            c.execute("SELECT persona FROM conversations LIMIT 1")
        except sqlite3.OperationalError:
            c.execute("ALTER TABLE conversations ADD COLUMN persona TEXT DEFAULT 'naomi'")
        # v2: Compressed session summaries
        c.execute('''CREATE TABLE IF NOT EXISTS session_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary TEXT NOT NULL,
            message_count INTEGER DEFAULT 0,
            created_at REAL NOT NULL
        )''')
        # v3: Feedback signals (reactions, implicit)
        c.execute('''CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            signal TEXT NOT NULL,
            source TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            created_at REAL NOT NULL
        )''')
        # v3: Persona drift history
        c.execute('''CREATE TABLE IF NOT EXISTS persona_drift (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persona TEXT NOT NULL DEFAULT 'naomi',
            version INTEGER NOT NULL,
            style_overrides TEXT NOT NULL,
            trigger_reason TEXT,
            created_at REAL NOT NULL
        )''')
        # Migrate: add session_id column if missing
        try:
            c.execute("SELECT session_id FROM conversations LIMIT 1")
        except sqlite3.OperationalError:
            c.execute("ALTER TABLE conversations ADD COLUMN session_id TEXT DEFAULT 'default'")
        # v4: Persistent multi-turn tool context (session messages for agent loop)
        c.execute("""CREATE TABLE IF NOT EXISTS session_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_session_msg ON session_messages(session_id)")
        # v4: Observability metrics
        c.execute("""CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            backend TEXT NOT NULL,
            tokens_in INTEGER DEFAULT 0,
            tokens_out INTEGER DEFAULT 0,
            latency_ms INTEGER DEFAULT 0,
            success INTEGER DEFAULT 1,
            error TEXT DEFAULT ''
        )""")
        self.conn.commit()

    # === Short-term Memory ===
    def remember_short(self, content: str, category: str = "general", ttl: int = 3600):
        now = time.time()
        # Dedup: don't store identical content within 60 seconds
        existing = self.conn.execute(
            "SELECT id FROM short_term WHERE content=? AND timestamp > ?",
            (content, now - 60)
        ).fetchone()
        if existing:
            return

        self.conn.execute(
            "INSERT INTO short_term (content, category, timestamp, expires_at) VALUES (?, ?, ?, ?)",
            (content, category, now, now + ttl)
        )
        self.conn.commit()
        self._cleanup_short_term()

    def recall_short(self, category: str = None, limit: int = 20) -> List[Dict]:
        self._cleanup_short_term()
        if category:
            rows = self.conn.execute(
                "SELECT * FROM short_term WHERE category=? ORDER BY timestamp DESC LIMIT ?",
                (category, limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM short_term ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def _cleanup_short_term(self):
        now = time.time()
        self.conn.execute("DELETE FROM short_term WHERE expires_at < ?", (now,))
        # Keep max N entries
        self.conn.execute(
            "DELETE FROM short_term WHERE id NOT IN (SELECT id FROM short_term ORDER BY timestamp DESC LIMIT ?)",
            (self.MAX_SHORT_TERM,)
        )
        self.conn.commit()

    # === Long-term Memory with Freshness Decay ===
    def remember_long(self, title: str, content: str, category: str = "general",
                      importance: int = 5, tags: List[str] = None):
        now = time.time()
        # Dedup interlock
        if self._extraction_lock:
            return
        # Check for near-duplicate titles
        existing = self.conn.execute(
            "SELECT id, content FROM long_term WHERE title=?", (title,)
        ).fetchone()
        if existing:
            # Update if new content is substantially different
            if existing['content'][:100] != content[:100]:
                self.conn.execute(
                    "UPDATE long_term SET content=?, importance=?, updated_at=?, tags=? WHERE id=?",
                    (content, importance, now, json.dumps(tags or []), existing['id'])
                )
        else:
            self.conn.execute(
                "INSERT INTO long_term (title, content, category, importance, created_at, updated_at, tags) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (title, content, category, importance, now, now, json.dumps(tags or []))
            )
        self.conn.commit()

    def recall_long(self, query: str = None, category: str = None, limit: int = 10) -> List[Dict]:
        """Retrieve memories with freshness-weighted scoring."""
        if query:
            rows = self.conn.execute(
                "SELECT * FROM long_term WHERE title LIKE ? OR content LIKE ? ORDER BY importance DESC, updated_at DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit * 2)  # Fetch extra for scoring
            ).fetchall()
        elif category:
            rows = self.conn.execute(
                "SELECT * FROM long_term WHERE category=? ORDER BY importance DESC LIMIT ?",
                (category, limit * 2)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM long_term ORDER BY importance DESC, updated_at DESC LIMIT ?", (limit * 2,)
            ).fetchall()

        # Apply freshness decay scoring
        now = time.time()
        scored = []
        for r in rows:
            age_days = (now - r['updated_at']) / 86400
            freshness = math.exp(-age_days / self.FRESHNESS_DECAY_DAYS)  # Exponential decay
            score = r['importance'] * freshness + r['access_count'] * 0.1
            scored.append((score, dict(r)))

        # Sort by score, return top N
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item[1] for item in scored[:limit]]

        # Update access counts
        for r in results:
            self.conn.execute("UPDATE long_term SET access_count=access_count+1 WHERE id=?", (r['id'],))
        self.conn.commit()
        return results

    # === Semantic Search (keyword-based for now, upgradeable to embeddings) ===
    def semantic_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memories by relevance. Top-5 cap like Claude Code."""
        words = query.lower().split()[:5]  # Use top 5 keywords
        conditions = " OR ".join(["(LOWER(title) LIKE ? OR LOWER(content) LIKE ?)"] * len(words))
        params = []
        for w in words:
            params.extend([f"%{w}%", f"%{w}%"])

        if not conditions:
            return []

        rows = self.conn.execute(
            f"SELECT * FROM long_term WHERE {conditions} ORDER BY importance DESC LIMIT ?",
            params + [limit]
        ).fetchall()
        return [dict(r) for r in rows]

    # === Skill Memory ===
    def learn_skill(self, name: str, description: str, tool_command: str = "",
                    install_command: str = ""):
        now = time.time()
        self.conn.execute(
            "INSERT OR REPLACE INTO skills (name, description, tool_command, install_command, learned_at) VALUES (?, ?, ?, ?, ?)",
            (name, description, tool_command, install_command, now)
        )
        self.conn.commit()

    def recall_skill(self, name: str = None):
        if name:
            row = self.conn.execute("SELECT * FROM skills WHERE name=?", (name,)).fetchone()
            return dict(row) if row else None
        rows = self.conn.execute("SELECT * FROM skills ORDER BY success_count DESC").fetchall()
        return [dict(r) for r in rows]

    def skill_used(self, name: str, success: bool = True):
        if success:
            self.conn.execute(
                "UPDATE skills SET success_count=success_count+1, last_used=? WHERE name=?",
                (time.time(), name)
            )
        else:
            self.conn.execute(
                "UPDATE skills SET fail_count=fail_count+1, last_used=? WHERE name=?",
                (time.time(), name)
            )
        self.conn.commit()

    # === Persona Memory ===
    def set_persona(self, key: str, value: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO persona (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, time.time())
        )
        self.conn.commit()

    def get_persona(self, key: str = None):
        if key:
            row = self.conn.execute("SELECT value FROM persona WHERE key=?", (key,)).fetchone()
            return row['value'] if row else None
        rows = self.conn.execute("SELECT * FROM persona").fetchall()
        return {r['key']: r['value'] for r in rows}

    # === Task History ===
    MAX_TASK_HISTORY = 200

    def add_task(self, task: str, brain_mode: str = "left") -> int:
        now = time.time()
        c = self.conn.execute(
            "INSERT INTO task_history (task, status, started_at, brain_mode) VALUES (?, 'running', ?, ?)",
            (task, now, brain_mode)
        )
        # Trim old task history
        self.conn.execute(
            "DELETE FROM task_history WHERE id NOT IN "
            "(SELECT id FROM task_history ORDER BY id DESC LIMIT ?)",
            (self.MAX_TASK_HISTORY,)
        )
        self.conn.commit()
        return c.lastrowid

    def complete_task(self, task_id: int, result: str, status: str = "completed"):
        now = time.time()
        row = self.conn.execute("SELECT started_at FROM task_history WHERE id=?", (task_id,)).fetchone()
        duration = now - row['started_at'] if row else 0
        self.conn.execute(
            "UPDATE task_history SET status=?, result=?, completed_at=?, duration=? WHERE id=?",
            (status, result, now, duration, task_id)
        )
        self.conn.commit()

    def get_pending_tasks(self) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM task_history WHERE status='pending' ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_tasks(self, limit: int = 20) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM task_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # === Conversation Log ===
    MAX_CONVERSATIONS = 500

    def log_conversation(self, role: str, content: str, persona: str = "naomi",
                         session_id: str = "default"):
        self.conn.execute(
            "INSERT INTO conversations (role, content, timestamp, persona, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (role, content, time.time(), persona, session_id),
        )
        self.conn.execute(
            "DELETE FROM conversations WHERE id NOT IN "
            "(SELECT id FROM conversations ORDER BY timestamp DESC LIMIT ?)",
            (self.MAX_CONVERSATIONS,),
        )
        self.conn.commit()

    def get_conversations(self, limit: int = 50, persona: str = None,
                          session_id: str = None) -> List[Dict]:
        """Get conversations, optionally filtered by persona and/or session."""
        clauses: List[str] = []
        params: list = []
        if persona:
            clauses.append("persona=?")
            params.append(persona)
        if session_id:
            clauses.append("session_id=?")
            params.append(session_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM conversations{where} ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_last_conversation(self, persona: str = None) -> Optional[Dict]:
        """Return the most recent conversation entry for a persona."""
        if persona:
            row = self.conn.execute(
                "SELECT * FROM conversations WHERE persona=? ORDER BY timestamp DESC LIMIT 1",
                (persona,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM conversations ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def get_conversation_count_since(self, since: float, persona: str = "naomi") -> int:
        """Count conversations since a given timestamp."""
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM conversations WHERE persona=? AND timestamp>?",
            (persona, since),
        ).fetchone()
        return row["cnt"] if row else 0

    def list_sessions(self, persona: str = "naomi", limit: int = 10) -> List[Dict]:
        """List distinct sessions with first-message preview."""
        rows = self.conn.execute(
            "SELECT session_id, MIN(timestamp) as started, MAX(timestamp) as last_active, "
            "COUNT(*) as msg_count "
            "FROM conversations WHERE persona=? AND session_id != 'default' "
            "GROUP BY session_id ORDER BY last_active DESC LIMIT ?",
            (persona, limit),
        ).fetchall()
        sessions = []
        for r in rows:
            preview_row = self.conn.execute(
                "SELECT content FROM conversations WHERE session_id=? AND role='user' "
                "ORDER BY timestamp ASC LIMIT 1",
                (r["session_id"],),
            ).fetchone()
            sessions.append({
                "session_id": r["session_id"],
                "started": r["started"],
                "last_active": r["last_active"],
                "msg_count": r["msg_count"],
                "preview": preview_row["content"][:80] if preview_row else "",
            })
        return sessions

    # === Feedback ===
    def log_feedback(self, signal: str, source: str, weight: float = 1.0,
                     conversation_id: int = None):
        self.conn.execute(
            "INSERT INTO feedback (conversation_id, signal, source, weight, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, signal, source, weight, time.time()),
        )
        self.conn.commit()

    def get_feedback_summary(self, limit: int = 50) -> Dict[str, Any]:
        """Summarize recent feedback signals."""
        rows = self.conn.execute(
            "SELECT signal, weight FROM feedback ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        if not rows:
            return {"total": 0, "positive": 0, "negative": 0, "neutral": 0, "score": 0.5}
        pos = sum(r["weight"] for r in rows if r["signal"] == "positive")
        neg = sum(r["weight"] for r in rows if r["signal"] == "negative")
        neu = sum(r["weight"] for r in rows if r["signal"] == "neutral")
        total = pos + neg + neu
        score = (pos / total) if total > 0 else 0.5
        return {"total": len(rows), "positive": pos, "negative": neg, "neutral": neu, "score": round(score, 2)}

    # === Persona Drift ===
    def save_drift(self, persona: str, version: int, style_overrides: str,
                   trigger_reason: str = ""):
        self.conn.execute(
            "INSERT INTO persona_drift (persona, version, style_overrides, trigger_reason, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (persona, version, style_overrides, trigger_reason, time.time()),
        )
        self.conn.commit()

    def get_latest_drift(self, persona: str = "naomi") -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM persona_drift WHERE persona=? ORDER BY version DESC LIMIT 1",
            (persona,),
        ).fetchone()
        return dict(row) if row else None

    def get_drift_history(self, persona: str = "naomi", limit: int = 5) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM persona_drift WHERE persona=? ORDER BY version DESC LIMIT ?",
            (persona, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # === v2: Progressive Context Compression ===
    def compress_context(self, brain=None) -> str:
        """
        Progressive compression inspired by Claude Code:
        Layer 1: Rule-based trimming (cheap)
        Layer 2: Structured LLM summary (expensive, only if needed)
        """
        conversations = self.get_conversations(limit=100)
        if not conversations:
            return ""

        total_tokens = sum(estimate_tokens(c['content']) for c in conversations)

        if total_tokens <= self.MAX_CONTEXT_TOKENS:
            # No compression needed
            return "\n".join(f"[{c['role']}] {c['content']}" for c in conversations)

        # Layer 1: Rule-based trimming - keep recent, truncate old
        recent = conversations[-20:]  # Always keep last 20 messages
        older = conversations[:-20]

        # Trim older messages to first 100 chars each
        trimmed_older = "\n".join(
            f"[{c['role']}] {c['content'][:100]}..." for c in older[-10:]
        )

        recent_text = "\n".join(f"[{c['role']}] {c['content']}" for c in recent)
        combined = trimmed_older + "\n---\n" + recent_text

        if estimate_tokens(combined) <= self.MAX_CONTEXT_TOKENS:
            return combined

        # Layer 2: LLM-based structured summary (only if brain available)
        if brain and older:
            older_text = "\n".join(f"[{c['role']}] {c['content'][:200]}" for c in older)
            summary = brain._think(
                "Compress this conversation history into a structured summary. "
                "Use this 9-section format:\n"
                "1. USER INTENT: What the user wants\n"
                "2. KEY CONCEPTS: Important topics discussed\n"
                "3. DECISIONS MADE: What was decided\n"
                "4. ERRORS & FIXES: Problems encountered and solutions\n"
                "5. CURRENT STATE: Where things stand now\n"
                "6. PENDING TASKS: What still needs to be done\n"
                "7. USER PREFERENCES: Learned preferences\n"
                "8. IMPORTANT FACTS: Key facts to remember\n"
                "9. NEXT STEPS: What to do next\n\n"
                f"Conversation:\n{older_text[:3000]}"
            )
            # Store summary
            self.conn.execute(
                "INSERT INTO session_summaries (summary, message_count, created_at) VALUES (?, ?, ?)",
                (summary, len(older), time.time())
            )
            self.conn.commit()
            return f"=== Session Summary ===\n{summary}\n\n=== Recent ===\n{recent_text}"

        return recent_text  # Fallback: just recent messages

    # === v2: Enhanced Context Builder ===
    def build_context(self, brain=None, query: str = None) -> str:
        """Build context with token budget management."""
        parts = []
        token_budget = self.MAX_CONTEXT_TOKENS
        tokens_used = 0

        # 1. Persona (highest priority, smallest)
        persona = self.get_persona()
        if persona:
            persona_text = "=== Identity ===\n" + "\n".join(f"- {k}: {v}" for k, v in persona.items())
            parts.append(persona_text)
            tokens_used += estimate_tokens(persona_text)

        # 2. Relevant memories (semantic search if query provided)
        if query:
            relevant = self.semantic_search(query, limit=5)
        else:
            relevant = self.recall_long(limit=5)

        if relevant:
            mem_text = "=== Relevant Memories ===\n" + "\n".join(
                f"- {m['title']}: {m['content'][:150]}" for m in relevant
            )
            if tokens_used + estimate_tokens(mem_text) < token_budget:
                parts.append(mem_text)
                tokens_used += estimate_tokens(mem_text)

        # 3. Recent tasks
        tasks = self.get_recent_tasks(5)
        if tasks:
            task_text = "=== Recent Tasks ===\n" + "\n".join(
                f"- [{t['status']}] {t['task'][:80]}" for t in tasks
            )
            if tokens_used + estimate_tokens(task_text) < token_budget:
                parts.append(task_text)
                tokens_used += estimate_tokens(task_text)

        # 4. Short-term context
        short = self.recall_short(limit=5)
        if short:
            short_text = "=== Recent Activity ===\n" + "\n".join(
                f"- [{s['category']}] {s['content'][:100]}" for s in short
            )
            if tokens_used + estimate_tokens(short_text) < token_budget:
                parts.append(short_text)
                tokens_used += estimate_tokens(short_text)

        # 5. Compressed conversation history (fills remaining budget)
        remaining = token_budget - tokens_used
        if remaining > 500:
            conv = self.compress_context(brain)
            if conv:
                # Truncate to fit budget
                while estimate_tokens(conv) > remaining and len(conv) > 200:
                    nl = conv.find('\n', 100)
                    if nl == -1:
                        break
                    conv = conv[nl+1:]  # Remove oldest line
                if conv:
                    parts.append("=== Conversation ===\n" + conv)

        return "\n\n".join(parts)

    # === v2: Extraction Interlock ===
    def lock_extraction(self):
        """Prevent duplicate memory extraction this turn."""
        self._extraction_lock = True

    def unlock_extraction(self):
        self._extraction_lock = False

    # === v2: Memory Consolidation ===
    def consolidate(self, brain=None):
        """Merge similar memories, remove stale ones."""
        now = time.time()
        # Remove very old, low-importance, never-accessed memories
        self.conn.execute(
            "DELETE FROM long_term WHERE importance <= 3 AND access_count = 0 AND updated_at < ?",
            (now - 86400 * 30,)  # 30 days old
        )
        self.conn.commit()

        # Remove duplicate short-term entries
        self.conn.execute("""
            DELETE FROM short_term WHERE id NOT IN (
                SELECT MIN(id) FROM short_term GROUP BY content
            )
        """)
        self.conn.commit()

    # === v4: Session Messages (Persistent Multi-Turn Tool Context) ===

    def save_session_messages(self, session_id: str, messages: list) -> None:
        """Save agent loop messages for later resumption."""
        now = time.time()
        rows = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, default=str)
            rows.append((session_id, role, content, now))
        self.conn.executemany(
            "INSERT INTO session_messages (session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def load_session_messages(self, session_id: str) -> List[Dict]:
        """Load previous session messages."""
        rows = self.conn.execute(
            "SELECT role, content, created_at FROM session_messages "
            "WHERE session_id=? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        results = []
        for r in rows:
            content = r["content"]
            # Try to deserialize JSON content back to its original form
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                pass
            results.append({
                "role": r["role"],
                "content": content,
                "created_at": r["created_at"],
            })
        return results

    def clear_session_messages(self, session_id: str) -> None:
        """Clear a session's messages."""
        self.conn.execute(
            "DELETE FROM session_messages WHERE session_id=?",
            (session_id,),
        )
        self.conn.commit()

    # === v4: Observability Metrics ===

    def log_metric(self, backend: str, tokens_in: int = 0, tokens_out: int = 0,
                   latency_ms: int = 0, success: bool = True, error: str = "") -> None:
        """Log an API call metric."""
        self.conn.execute(
            "INSERT INTO metrics (timestamp, backend, tokens_in, tokens_out, "
            "latency_ms, success, error) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time.time(), backend, tokens_in, tokens_out, latency_ms,
             1 if success else 0, error),
        )
        self.conn.commit()

    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Return aggregated metrics for the last N hours."""
        since = time.time() - hours * 3600
        rows = self.conn.execute(
            "SELECT backend, COUNT(*) as calls, "
            "SUM(tokens_in) as total_in, SUM(tokens_out) as total_out, "
            "AVG(latency_ms) as avg_latency, "
            "SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as successes, "
            "SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as failures "
            "FROM metrics WHERE timestamp > ? GROUP BY backend",
            (since,),
        ).fetchall()

        total_calls = 0
        total_success = 0
        total_latency = 0
        by_backend: Dict[str, Any] = {}
        for r in rows:
            backend = r["backend"]
            calls = r["calls"]
            total_calls += calls
            total_success += r["successes"]
            total_latency += (r["avg_latency"] or 0) * calls
            by_backend[backend] = {
                "calls": calls,
                "tokens_in": r["total_in"] or 0,
                "tokens_out": r["total_out"] or 0,
                "avg_latency_ms": round(r["avg_latency"] or 0),
                "success_rate": round(r["successes"] / max(calls, 1) * 100, 1),
            }

        success_rate = round(total_success / max(total_calls, 1) * 100, 1)
        avg_latency = round(total_latency / max(total_calls, 1))

        return {
            "hours": hours,
            "total_calls": total_calls,
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "by_backend": by_backend,
        }

    def close(self):
        self.conn.close()
