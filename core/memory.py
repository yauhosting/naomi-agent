"""
NAOMI Agent - Memory System
SQLite-based persistent memory with short-term, long-term, skill, and persona memories.
"""
import sqlite3
import json
import time
import os
from typing import Optional, List, Dict, Any


class Memory:
    def __init__(self, db_path: str = "data/naomi_memory.db"):
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else "data", exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

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
            timestamp REAL NOT NULL
        )''')
        self.conn.commit()

    # === Short-term Memory ===
    def remember_short(self, content: str, category: str = "general", ttl: int = 3600):
        now = time.time()
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
        self.conn.execute("DELETE FROM short_term WHERE expires_at < ?", (time.time(),))
        self.conn.commit()

    # === Long-term Memory ===
    def remember_long(self, title: str, content: str, category: str = "general",
                      importance: int = 5, tags: List[str] = None):
        now = time.time()
        existing = self.conn.execute("SELECT id FROM long_term WHERE title=?", (title,)).fetchone()
        if existing:
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
        if query:
            rows = self.conn.execute(
                "SELECT * FROM long_term WHERE title LIKE ? OR content LIKE ? ORDER BY importance DESC, updated_at DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit)
            ).fetchall()
        elif category:
            rows = self.conn.execute(
                "SELECT * FROM long_term WHERE category=? ORDER BY importance DESC LIMIT ?",
                (category, limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM long_term ORDER BY importance DESC, updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        for r in rows:
            self.conn.execute("UPDATE long_term SET access_count=access_count+1 WHERE id=?", (r['id'],))
        self.conn.commit()
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
        field = "success_count" if success else "fail_count"
        self.conn.execute(
            f"UPDATE skills SET {field}={field}+1, last_used=? WHERE name=?",
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
    def add_task(self, task: str, brain_mode: str = "left") -> int:
        now = time.time()
        c = self.conn.execute(
            "INSERT INTO task_history (task, status, started_at, brain_mode) VALUES (?, 'running', ?, ?)",
            (task, now, brain_mode)
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
    def log_conversation(self, role: str, content: str):
        self.conn.execute(
            "INSERT INTO conversations (role, content, timestamp) VALUES (?, ?, ?)",
            (role, content, time.time())
        )
        self.conn.commit()

    def get_conversations(self, limit: int = 50) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM conversations ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # === Context Builder ===
    def build_context(self) -> str:
        parts = []
        persona = self.get_persona()
        if persona:
            parts.append("=== User Profile ===")
            for k, v in persona.items():
                parts.append(f"- {k}: {v}")
        tasks = self.get_recent_tasks(5)
        if tasks:
            parts.append("\n=== Recent Tasks ===")
            for t in tasks:
                parts.append(f"- [{t['status']}] {t['task']}")
        short = self.recall_short(limit=10)
        if short:
            parts.append("\n=== Recent Context ===")
            for s in short:
                parts.append(f"- {s['content']}")
        important = self.recall_long(limit=5)
        if important:
            parts.append("\n=== Key Knowledge ===")
            for m in important:
                parts.append(f"- {m['title']}: {m['content'][:200]}")
        return "\n".join(parts)

    def close(self):
        self.conn.close()
