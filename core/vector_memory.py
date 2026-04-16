"""
NAOMI Agent - Vector Memory (Semantic Search)

Adds semantic vector search to NAOMI's memory system using:
- sqlite-vec for vector storage (SQLite extension)
- sentence-transformers with all-MiniLM-L6-v2 for embeddings (384 dims, CPU-fast)

Complements core/memory.py's keyword-based search with true semantic similarity.
"""
import sqlite3
import json
import time
import os
import logging
import threading
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger("naomi.vector_memory")

# Embedding dimension for all-MiniLM-L6-v2
EMBEDDING_DIM = 384
MODEL_NAME = "all-MiniLM-L6-v2"


class VectorMemory:
    """Semantic vector search over NAOMI's memory using sqlite-vec."""

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

        self._model = None
        self._model_lock = threading.Lock()

        self._load_sqlite_vec()
        self._init_tables()

    def _load_sqlite_vec(self) -> None:
        """Load the sqlite-vec extension. Degrades gracefully if unavailable."""
        try:
            import sqlite_vec  # noqa: F811

            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            self._vec_available = True
            logger.info("sqlite-vec extension loaded")
        except Exception as exc:
            self._vec_available = False
            logger.warning(
                "sqlite-vec not available (%s) — vector search disabled. "
                "Install with: pip install sqlite-vec", exc
            )

    def _init_tables(self) -> None:
        """Create metadata and virtual vector tables."""
        c = self.conn.cursor()
        # Metadata table — stores text, category, and timestamps alongside the vector id
        c.execute("""CREATE TABLE IF NOT EXISTS memory_vectors_meta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            metadata TEXT DEFAULT '{}',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )""")
        # sqlite-vec virtual table for vector similarity search
        if self._vec_available:
            c.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors "
                f"USING vec0(id INTEGER PRIMARY KEY, embedding float[{EMBEDDING_DIM}])"
            )
        self.conn.commit()
        logger.info("Vector memory tables initialized (vec=%s)", self._vec_available)

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _get_model(self) -> Any:
        """Lazy-load the sentence-transformer model (thread-safe)."""
        if self._model is not None:
            return self._model
        with self._model_lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return self._model
            logger.info("Loading embedding model %s ...", MODEL_NAME)
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(MODEL_NAME)
                logger.info("Embedding model loaded")
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for VectorMemory. "
                    "Install with: pip install sentence-transformers"
                ) from exc
            return self._model

    def _embed(self, text: str) -> bytes:
        """Encode text into a float32 vector and return as raw bytes for sqlite-vec."""
        import struct

        model = self._get_model()
        vec = model.encode(text, normalize_embeddings=True)
        return struct.pack(f"{EMBEDDING_DIM}f", *vec.tolist())

    def _embed_batch(self, texts: List[str]) -> List[bytes]:
        """Encode a batch of texts into raw-byte vectors."""
        import struct

        model = self._get_model()
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
        results: List[bytes] = []
        for v in vecs:
            results.append(struct.pack(f"{EMBEDDING_DIM}f", *v.tolist()))
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Embed *text* and store in sqlite-vec.

        Returns the row id of the newly inserted memory.
        """
        now = time.time()
        meta_json = json.dumps(metadata or {})

        # Deduplicate: skip if identical text was stored in the last 60 seconds
        existing = self.conn.execute(
            "SELECT id FROM memory_vectors_meta WHERE text = ? AND created_at > ?",
            (text, now - 60),
        ).fetchone()
        if existing:
            logger.debug("Skipping duplicate vector memory (id=%s)", existing["id"])
            return existing["id"]

        embedding = self._embed(text)

        c = self.conn.cursor()
        c.execute(
            "INSERT INTO memory_vectors_meta (text, category, metadata, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (text, category, meta_json, now, now),
        )
        row_id = c.lastrowid

        c.execute(
            "INSERT INTO memory_vectors (id, embedding) VALUES (?, ?)",
            (row_id, embedding),
        )
        self.conn.commit()
        logger.debug("Stored vector memory id=%s category=%s", row_id, category)
        return row_id

    def search(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.5,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic similarity search.

        Returns up to *limit* results whose cosine similarity >= *threshold*.
        Each result dict contains: id, text, category, metadata, similarity, created_at.
        """
        query_vec = self._embed(query)

        rows = self.conn.execute(
            """
            SELECT v.id, v.distance, m.text, m.category, m.metadata, m.created_at
            FROM memory_vectors v
            JOIN memory_vectors_meta m ON m.id = v.id
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
            """,
            (query_vec, limit * 3),  # fetch extra to allow filtering
        ).fetchall()

        results: List[Dict[str, Any]] = []
        for r in rows:
            # sqlite-vec returns cosine *distance* (1 - similarity)
            similarity = 1.0 - r["distance"]
            if similarity < threshold:
                continue
            if category and r["category"] != category:
                continue
            results.append({
                "id": r["id"],
                "text": r["text"],
                "category": r["category"],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "similarity": round(similarity, 4),
                "created_at": r["created_at"],
            })
            if len(results) >= limit:
                break

        return results

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Combine vector similarity and keyword search results.

        Results are scored as:
            final_score = vector_weight * similarity + keyword_weight * keyword_match
        """
        # 1. Vector results (fetch extra for merging)
        vec_results = self.search(query, limit=limit * 2, threshold=0.3)

        # 2. Keyword results from the same metadata table
        words = query.lower().split()[:5]
        keyword_hits: Dict[int, float] = {}
        if words:
            conditions = " OR ".join(
                ["LOWER(text) LIKE ?"] * len(words)
            )
            params = [f"%{w}%" for w in words]
            kw_rows = self.conn.execute(
                f"SELECT id, text FROM memory_vectors_meta WHERE {conditions}",
                params,
            ).fetchall()
            for kr in kw_rows:
                # Simple keyword score: fraction of query words matched
                matched = sum(1 for w in words if w in kr["text"].lower())
                keyword_hits[kr["id"]] = matched / len(words)

        # 3. Merge and score
        scored: Dict[int, Dict[str, Any]] = {}
        for vr in vec_results:
            vid = vr["id"]
            kw_score = keyword_hits.get(vid, 0.0)
            final = vector_weight * vr["similarity"] + keyword_weight * kw_score
            scored[vid] = {**vr, "score": round(final, 4)}

        # Add keyword-only hits not already in vector results
        for kid, kw_score in keyword_hits.items():
            if kid not in scored:
                row = self.conn.execute(
                    "SELECT * FROM memory_vectors_meta WHERE id = ?", (kid,)
                ).fetchone()
                if row:
                    scored[kid] = {
                        "id": kid,
                        "text": row["text"],
                        "category": row["category"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                        "similarity": 0.0,
                        "created_at": row["created_at"],
                        "score": round(keyword_weight * kw_score, 4),
                    }

        ranked = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:limit]

    def delete(self, row_id: int) -> bool:
        """Remove a vector memory entry by id."""
        self.conn.execute("DELETE FROM memory_vectors WHERE id = ?", (row_id,))
        self.conn.execute("DELETE FROM memory_vectors_meta WHERE id = ?", (row_id,))
        self.conn.commit()
        return True

    def count(self) -> int:
        """Return total number of stored vectors."""
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM memory_vectors_meta").fetchone()
        return row["cnt"] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def create_vector_memory(db_path: str = "data/naomi_memory.db") -> VectorMemory:
    """Create a VectorMemory instance with the given database path."""
    return VectorMemory(db_path=db_path)
