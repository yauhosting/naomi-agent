"""
NAOMI Agent - Knowledge Graph (Entity Relationship Storage)

Stores entity-relationship triples (subject, predicate, object) in SQLite
and supports path-finding between entities and LLM-based triple extraction.

Works alongside core/memory.py using the same SQLite database.
"""
import sqlite3
import json
import time
import os
import logging
from typing import Optional, List, Dict, Any, Tuple
from collections import deque

logger = logging.getLogger("naomi.knowledge_graph")


class KnowledgeGraph:
    """Entity-relationship triple store backed by SQLite."""

    def __init__(self, db_path: str = "data/naomi_memory.db") -> None:
        if isinstance(db_path, dict):
            db_path = db_path.get("db_path", "data/naomi_memory.db")
        os.makedirs(
            os.path.dirname(db_path) if os.path.dirname(db_path) else "data",
            exist_ok=True,
        )
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_tables()

    def _init_tables(self) -> None:
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS knowledge_triples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            confidence REAL DEFAULT 0.8,
            source TEXT DEFAULT '',
            created_at REAL NOT NULL
        )""")
        # Indexes for fast lookup by subject or object
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_triples_subject "
            "ON knowledge_triples (subject)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_triples_object "
            "ON knowledge_triples (object)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_triples_predicate "
            "ON knowledge_triples (predicate)"
        )
        self.conn.commit()
        logger.info("Knowledge graph tables initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_triple(
        self,
        subject: str,
        predicate: str,
        object_: str,
        confidence: float = 0.8,
        source: str = "",
    ) -> int:
        """Store a (subject, predicate, object) triple.

        Deduplicates exact matches — if an identical triple exists, its
        confidence and timestamp are updated instead of inserting a new row.

        Returns the row id.
        """
        subject = subject.strip()
        predicate = predicate.strip()
        object_ = object_.strip()

        if not subject or not predicate or not object_:
            logger.warning("Skipping empty triple: (%s, %s, %s)", subject, predicate, object_)
            return -1

        # Deduplicate
        existing = self.conn.execute(
            "SELECT id FROM knowledge_triples "
            "WHERE subject = ? AND predicate = ? AND object = ?",
            (subject, predicate, object_),
        ).fetchone()

        now = time.time()
        if existing:
            # Update confidence (take the higher) and timestamp
            self.conn.execute(
                "UPDATE knowledge_triples SET confidence = MAX(confidence, ?), "
                "source = CASE WHEN source = '' THEN ? ELSE source END, "
                "created_at = ? WHERE id = ?",
                (confidence, source, now, existing["id"]),
            )
            self.conn.commit()
            return existing["id"]

        c = self.conn.cursor()
        c.execute(
            "INSERT INTO knowledge_triples "
            "(subject, predicate, object, confidence, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (subject, predicate, object_, confidence, source, now),
        )
        self.conn.commit()
        row_id = c.lastrowid
        logger.debug("Added triple id=%s: (%s, %s, %s)", row_id, subject, predicate, object_)
        return row_id

    def query(self, entity: str) -> List[Dict[str, Any]]:
        """Return all triples where *entity* appears as subject OR object."""
        rows = self.conn.execute(
            "SELECT * FROM knowledge_triples "
            "WHERE subject = ? OR object = ? "
            "ORDER BY confidence DESC, created_at DESC",
            (entity, entity),
        ).fetchall()
        return [dict(r) for r in rows]

    def query_subject(self, subject: str) -> List[Dict[str, Any]]:
        """Return all triples where *subject* is the subject."""
        rows = self.conn.execute(
            "SELECT * FROM knowledge_triples WHERE subject = ? "
            "ORDER BY confidence DESC",
            (subject,),
        ).fetchall()
        return [dict(r) for r in rows]

    def query_predicate(self, predicate: str) -> List[Dict[str, Any]]:
        """Return all triples with the given predicate."""
        rows = self.conn.execute(
            "SELECT * FROM knowledge_triples WHERE predicate = ? "
            "ORDER BY confidence DESC",
            (predicate,),
        ).fetchall()
        return [dict(r) for r in rows]

    def find_path(
        self,
        entity_a: str,
        entity_b: str,
        max_depth: int = 3,
    ) -> List[List[Dict[str, Any]]]:
        """Find relationship paths from *entity_a* to *entity_b* using BFS.

        Returns a list of paths. Each path is a list of triple dicts forming
        a chain from entity_a to entity_b. Returns an empty list if no path
        exists within *max_depth* hops.
        """
        if entity_a == entity_b:
            return [[]]

        # BFS: each queue entry is (current_entity, path_so_far)
        queue: deque[Tuple[str, List[Dict[str, Any]]]] = deque()
        queue.append((entity_a, []))
        visited: set = {entity_a}
        found_paths: List[List[Dict[str, Any]]] = []

        while queue:
            current, path = queue.popleft()
            if len(path) >= max_depth:
                continue

            # Get all triples involving the current entity
            neighbors = self.conn.execute(
                "SELECT * FROM knowledge_triples "
                "WHERE subject = ? OR object = ? "
                "ORDER BY confidence DESC",
                (current, current),
            ).fetchall()

            for row in neighbors:
                triple = dict(row)
                # Determine the "next" entity along this edge
                if triple["subject"] == current:
                    next_entity = triple["object"]
                else:
                    next_entity = triple["subject"]

                new_path = path + [triple]

                if next_entity == entity_b:
                    found_paths.append(new_path)
                    continue

                if next_entity not in visited:
                    visited.add(next_entity)
                    queue.append((next_entity, new_path))

        return found_paths

    def extract_from_text(self, text: str, brain: Any = None) -> List[Dict[str, Any]]:
        """Use an LLM (via the Brain instance) to extract entity-relationship triples.

        The *brain* parameter should be the Brain class instance that provides
        a `_think(prompt)` method. If brain is None, this is a no-op.

        Returns the list of triples that were stored.
        """
        if brain is None:
            logger.warning("extract_from_text called without brain — skipping")
            return []

        prompt = (
            "Extract entity-relationship triples from the following text. "
            "Return ONLY a JSON array of objects, each with keys: "
            '"subject", "predicate", "object", "confidence" (0.0-1.0).\n'
            "Rules:\n"
            "- Entities should be proper nouns, concepts, or specific things\n"
            "- Predicates should be short verb phrases (e.g. 'is', 'works_at', 'likes')\n"
            "- Confidence reflects how certain the relationship is\n"
            "- Return at most 10 triples\n"
            "- Return [] if no clear relationships found\n\n"
            f"Text:\n{text[:2000]}\n\n"
            "JSON array:"
        )

        try:
            response = brain._think(prompt)
        except Exception as exc:
            logger.error("LLM extraction failed: %s", exc)
            return []

        # Parse the JSON response
        triples = _parse_triples_json(response)
        stored: List[Dict[str, Any]] = []
        for t in triples:
            subject = t.get("subject", "").strip()
            predicate = t.get("predicate", "").strip()
            object_ = t.get("object", "").strip()
            confidence = float(t.get("confidence", 0.7))

            if not subject or not predicate or not object_:
                continue

            row_id = self.add_triple(
                subject, predicate, object_,
                confidence=confidence, source="llm_extraction",
            )
            stored.append({
                "id": row_id,
                "subject": subject,
                "predicate": predicate,
                "object": object_,
                "confidence": confidence,
            })

        logger.info("Extracted %d triples from text (%d chars)", len(stored), len(text))
        return stored

    def get_context(self, entity: str, limit: int = 10) -> str:
        """Return a formatted context string about *entity* for LLM injection.

        Gathers triples involving the entity and formats them as natural
        language statements suitable for inclusion in an LLM prompt.
        """
        triples = self.query(entity)[:limit]
        if not triples:
            return ""

        lines: List[str] = [f"Known facts about '{entity}':"]
        for t in triples:
            conf = t["confidence"]
            marker = "" if conf >= 0.7 else " (uncertain)"
            lines.append(
                f"  - {t['subject']} {t['predicate']} {t['object']}{marker}"
            )
        return "\n".join(lines)

    def get_related_entities(self, entity: str, max_hops: int = 1) -> List[str]:
        """Return entities connected to *entity* within *max_hops*."""
        current_level: set = {entity}
        all_related: set = set()

        for _ in range(max_hops):
            next_level: set = set()
            for ent in current_level:
                rows = self.conn.execute(
                    "SELECT subject, object FROM knowledge_triples "
                    "WHERE subject = ? OR object = ?",
                    (ent, ent),
                ).fetchall()
                for r in rows:
                    if r["subject"] != entity:
                        next_level.add(r["subject"])
                    if r["object"] != entity:
                        next_level.add(r["object"])
            all_related.update(next_level)
            current_level = next_level - {entity}

        return sorted(all_related)

    def delete_triple(self, triple_id: int) -> bool:
        """Delete a triple by id."""
        self.conn.execute("DELETE FROM knowledge_triples WHERE id = ?", (triple_id,))
        self.conn.commit()
        return True

    def count(self) -> int:
        """Return total number of stored triples."""
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM knowledge_triples"
        ).fetchone()
        return row["cnt"] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_triples_json(response: str) -> List[Dict[str, Any]]:
    """Best-effort parse of LLM JSON response containing triples."""
    if not response:
        return []

    # Try direct parse first
    text = response.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return []
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from markdown code block
    import re

    match = re.search(r"```(?:json)?\s*(\[.*?])\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find a bare JSON array in the text
    match = re.search(r"\[.*]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse triples JSON from LLM response")
    return []


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def create_knowledge_graph(db_path: str = "data/naomi_memory.db") -> KnowledgeGraph:
    """Create a KnowledgeGraph instance with the given database path."""
    return KnowledgeGraph(db_path=db_path)
