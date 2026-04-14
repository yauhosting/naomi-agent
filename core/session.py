"""
NAOMI Agent - Session Manager v1
Time-based auto-split + manual session control.
Gap > 30 min between messages = new session.
"""
import time
import uuid
import logging
from typing import Optional, List, Dict

logger = logging.getLogger("naomi.session")

# 30 minutes in seconds
SESSION_GAP_SECONDS = 30 * 60


class SessionManager:
    """Manages conversation sessions with auto-split on inactivity."""

    def __init__(self, memory):
        self.memory = memory
        # Track active session per persona
        self._active_sessions: Dict[str, str] = {}

    def get_or_create_session(self, persona: str = "naomi") -> str:
        """Get current session or create new one if gap > 30 min."""
        last_conv = self.memory.get_last_conversation(persona=persona)

        if last_conv:
            gap = time.time() - last_conv["timestamp"]
            current_session = last_conv.get("session_id", "default")

            if gap < SESSION_GAP_SECONDS:
                self._active_sessions[persona] = current_session
                return current_session

            # Gap exceeded — auto-split
            new_id = self._generate_session_id()
            logger.info(
                "Session auto-split for %s: gap=%.0fs > %ds → %s",
                persona, gap, SESSION_GAP_SECONDS, new_id,
            )
            self._active_sessions[persona] = new_id
            return new_id

        # No history — first session
        new_id = self._generate_session_id()
        self._active_sessions[persona] = new_id
        return new_id

    def create_session(self, persona: str = "naomi") -> str:
        """Force-create a new session."""
        new_id = self._generate_session_id()
        self._active_sessions[persona] = new_id
        logger.info("New session created for %s: %s", persona, new_id)
        return new_id

    def get_active_session(self, persona: str = "naomi") -> Optional[str]:
        """Return cached active session without DB lookup."""
        return self._active_sessions.get(persona)

    def list_sessions(self, persona: str = "naomi", limit: int = 10) -> List[Dict]:
        """List recent sessions with first-message preview."""
        return self.memory.list_sessions(persona=persona, limit=limit)

    @staticmethod
    def _generate_session_id() -> str:
        return uuid.uuid4().hex[:12]
