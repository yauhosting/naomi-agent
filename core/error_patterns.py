"""
NAOMI Agent - Error Pattern Learning

Stores error signatures, tracks frequency, and retrieves known resolutions
for recurring errors. Learns from past failures to speed up debugging.
"""
import hashlib
import sqlite3
import time
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("naomi.error_patterns")


def _compute_signature(error_type: str, error_msg: str) -> str:
    """Compute a stable hash from error type + first meaningful line.

    The signature groups similar errors together so that resolutions
    can be looked up even when the full traceback differs.
    """
    first_line = ""
    for line in error_msg.strip().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("Traceback") and not stripped.startswith("File "):
            first_line = stripped
            break
    if not first_line:
        first_line = error_msg.strip()[:200]

    key = f"{error_type}:{first_line}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _extract_error_type(error_msg: str) -> str:
    """Extract the error class name from an error message.

    Looks for patterns like 'ModuleNotFoundError: ...' or 'TypeError: ...'.
    Falls back to 'UnknownError' if no pattern matches.
    """
    for line in error_msg.strip().splitlines():
        stripped = line.strip()
        if "Error:" in stripped or "Exception:" in stripped:
            colon = stripped.find(":")
            if colon > 0:
                candidate = stripped[:colon].strip()
                # Only accept if it looks like a class name
                parts = candidate.split(".")
                last = parts[-1]
                if last and last[0].isupper() and last.isidentifier():
                    return last
        # Also match "error[E0xxx]" style (Rust, etc.)
        if stripped.lower().startswith("error"):
            return stripped.split(":")[0].split("[")[0].strip()
    return "UnknownError"


class ErrorPatternDB:
    """SQLite-backed error pattern database for learning from failures."""

    def __init__(self, db_path: str = "data/naomi_memory.db") -> None:
        if isinstance(db_path, dict):
            db_path = db_path.get("db_path", "data/naomi_memory.db")
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else "data", exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_table()

    def _init_table(self) -> None:
        self.conn.execute('''CREATE TABLE IF NOT EXISTS error_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signature TEXT NOT NULL,
            error_type TEXT NOT NULL,
            error_msg TEXT NOT NULL,
            context TEXT DEFAULT '',
            resolution TEXT DEFAULT '',
            count INTEGER DEFAULT 1,
            first_seen REAL NOT NULL,
            last_seen REAL NOT NULL
        )''')
        # Index for fast signature lookup
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_error_sig ON error_patterns(signature)"
        )
        # Index for substring search on error_msg
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_error_type ON error_patterns(error_type)"
        )
        self.conn.commit()

    def record_error(
        self,
        task_type: str,
        error_msg: str,
        context: str = "",
    ) -> str:
        """Store an error occurrence. Returns the error signature.

        On duplicate signature, increments count and updates last_seen.
        """
        error_type = _extract_error_type(error_msg)
        signature = _compute_signature(error_type, error_msg)
        now = time.time()

        existing = self.conn.execute(
            "SELECT id, count FROM error_patterns WHERE signature=?",
            (signature,),
        ).fetchone()

        if existing:
            new_count = existing["count"] + 1
            self.conn.execute(
                "UPDATE error_patterns SET count=?, last_seen=?, context=? WHERE id=?",
                (new_count, now, context[:2000] if context else "", existing["id"]),
            )
            logger.debug(
                "Error pattern updated: sig=%s type=%s count=%d",
                signature, error_type, new_count,
            )
        else:
            self.conn.execute(
                "INSERT INTO error_patterns "
                "(signature, error_type, error_msg, context, resolution, count, first_seen, last_seen) "
                "VALUES (?, ?, ?, ?, '', 1, ?, ?)",
                (signature, error_type, error_msg[:5000], context[:2000], now, now),
            )
            logger.info("New error pattern: sig=%s type=%s", signature, error_type)

        self.conn.commit()
        return signature

    def record_resolution(self, error_signature: str, resolution: str) -> bool:
        """Store how an error was resolved. Returns True if the signature was found."""
        row = self.conn.execute(
            "SELECT id FROM error_patterns WHERE signature=?",
            (error_signature,),
        ).fetchone()

        if not row:
            logger.warning("No error pattern found for signature: %s", error_signature)
            return False

        self.conn.execute(
            "UPDATE error_patterns SET resolution=? WHERE id=?",
            (resolution[:5000], row["id"]),
        )
        self.conn.commit()
        logger.info("Resolution recorded for sig=%s", error_signature)
        return True

    def find_resolution(self, error_msg: str) -> Optional[Dict[str, Any]]:
        """Look up known resolutions for a similar error.

        Checks exact signature match first, then falls back to substring
        similarity on error_type and error message keywords.
        """
        # Step 1: Exact signature match
        error_type = _extract_error_type(error_msg)
        signature = _compute_signature(error_type, error_msg)

        row = self.conn.execute(
            "SELECT * FROM error_patterns WHERE signature=? AND resolution != ''",
            (signature,),
        ).fetchone()
        if row:
            return {
                "match_type": "exact",
                "signature": row["signature"],
                "error_type": row["error_type"],
                "error_msg": row["error_msg"][:500],
                "resolution": row["resolution"],
                "count": row["count"],
                "last_seen": row["last_seen"],
            }

        # Step 2: Same error type with resolution
        rows = self.conn.execute(
            "SELECT * FROM error_patterns WHERE error_type=? AND resolution != '' "
            "ORDER BY count DESC LIMIT 5",
            (error_type,),
        ).fetchall()

        if rows:
            # Score by substring overlap with the error message
            first_line = ""
            for line in error_msg.strip().splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("Traceback"):
                    first_line = stripped.lower()
                    break

            best = None
            best_score = 0
            for r in rows:
                stored_msg = r["error_msg"].lower()
                # Simple word overlap scoring
                words = set(first_line.split())
                stored_words = set(stored_msg.split())
                if not words:
                    continue
                overlap = len(words & stored_words)
                score = overlap / max(len(words), 1)
                if score > best_score:
                    best_score = score
                    best = r

            if best and best_score > 0.3:
                return {
                    "match_type": "similar",
                    "similarity": round(best_score, 2),
                    "signature": best["signature"],
                    "error_type": best["error_type"],
                    "error_msg": best["error_msg"][:500],
                    "resolution": best["resolution"],
                    "count": best["count"],
                    "last_seen": best["last_seen"],
                }

        # Step 3: Broad substring search across all resolved errors
        keywords = [
            w for w in error_msg.strip().split()[:8]
            if len(w) > 3 and w.isalnum()
        ]
        if keywords:
            conditions = " OR ".join(["error_msg LIKE ?"] * len(keywords))
            params = [f"%{kw}%" for kw in keywords]
            row = self.conn.execute(
                f"SELECT * FROM error_patterns WHERE resolution != '' AND ({conditions}) "
                "ORDER BY count DESC LIMIT 1",
                params,
            ).fetchone()
            if row:
                return {
                    "match_type": "keyword",
                    "signature": row["signature"],
                    "error_type": row["error_type"],
                    "error_msg": row["error_msg"][:500],
                    "resolution": row["resolution"],
                    "count": row["count"],
                    "last_seen": row["last_seen"],
                }

        return None

    def get_patterns(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return most frequent error patterns."""
        rows = self.conn.execute(
            "SELECT * FROM error_patterns ORDER BY count DESC, last_seen DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "signature": r["signature"],
                "error_type": r["error_type"],
                "error_msg": r["error_msg"][:200],
                "resolution": r["resolution"][:200] if r["resolution"] else "",
                "count": r["count"],
                "first_seen": r["first_seen"],
                "last_seen": r["last_seen"],
                "has_resolution": bool(r["resolution"]),
            }
            for r in rows
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics."""
        total = self.conn.execute("SELECT COUNT(*) as c FROM error_patterns").fetchone()["c"]
        resolved = self.conn.execute(
            "SELECT COUNT(*) as c FROM error_patterns WHERE resolution != ''"
        ).fetchone()["c"]
        top_types = self.conn.execute(
            "SELECT error_type, SUM(count) as total FROM error_patterns "
            "GROUP BY error_type ORDER BY total DESC LIMIT 5"
        ).fetchall()
        return {
            "total_patterns": total,
            "resolved": resolved,
            "unresolved": total - resolved,
            "resolution_rate": round(resolved / max(total, 1) * 100, 1),
            "top_error_types": [
                {"type": r["error_type"], "count": r["total"]} for r in top_types
            ],
        }

    def close(self) -> None:
        self.conn.close()
