"""
NAOMI Agent - Memory Extraction Sub-Agent
Runs independently after each conversation turn to extract knowledge.
Inspired by Claude Code's forked memory extraction agent.

Rules:
- Read-only access to conversation, write-only to memory
- Skips if main agent already wrote memory this turn (interlock)
- Extracts: user facts, feedback, project info, references
- Max 5 new memories per extraction
- Semantic dedup against existing memories
"""
import json
import time
import logging
import asyncio
from typing import Dict, List, Optional

logger = logging.getLogger("naomi.memory_agent")

EXTRACTION_PROMPT = """You are NAOMI's memory extraction sub-agent.
Analyze the recent conversation and extract knowledge worth remembering.

Rules:
- Only extract from the last {msg_count} messages
- Max 5 memories per extraction
- Skip trivial/obvious information
- Check existing memories to avoid duplicates
- Each memory needs: title, content, category, importance (1-10)

Categories:
- user_insight: Facts about Master (preferences, goals, skills, interests)
- feedback: How Master wants NAOMI to behave (corrections, confirmations)
- project: Ongoing work, deadlines, decisions
- reference: External resources, links, tools mentioned
- technical: Technical knowledge learned during tasks

Existing memories (DO NOT duplicate these):
{existing_memories}

Recent conversation:
{conversation}

Extract memories as JSON array. If nothing worth saving, return [].
[{{"title": "...", "content": "...", "category": "...", "importance": 5}}]
Return ONLY the JSON array, nothing else."""


class MemoryExtractionAgent:
    """Independent sub-agent that extracts knowledge from conversations."""

    def __init__(self, brain, memory):
        self.brain = brain
        self.memory = memory
        self._running = False
        self._last_extraction = 0
        self._min_interval = 30  # Min seconds between extractions
        self._messages_since_last = 0

    async def on_conversation_turn(self, user_msg: str, naomi_response: str):
        """Called after each conversation turn. Decides whether to extract."""
        self._messages_since_last += 1

        # Skip if too recent
        if time.time() - self._last_extraction < self._min_interval:
            return

        # Skip if interlock is on (main agent wrote memory this turn)
        if self.memory._extraction_lock:
            logger.debug("Extraction skipped: interlock active")
            return

        # Extract every 3 messages or if message seems important
        should_extract = (
            self._messages_since_last >= 3 or
            self._is_important(user_msg)
        )

        if should_extract:
            asyncio.create_task(self._extract(user_msg, naomi_response))

    def _is_important(self, msg: str) -> bool:
        """Quick heuristic: is this message worth extracting from?"""
        important_signals = [
            "remember", "don't forget", "記住", "記得",  # Explicit memory requests
            "i prefer", "i like", "i want", "我喜歡", "我想要", "我偏好",  # Preferences
            "project", "deadline", "目標", "項目", "計劃",  # Project info
            "feedback", "don't do", "stop doing", "不要", "別",  # Feedback
            "important", "key", "critical", "重要", "關鍵",  # Importance markers
        ]
        msg_lower = msg.lower()
        return any(signal in msg_lower for signal in important_signals)

    async def _extract(self, user_msg: str, naomi_response: str):
        """Run extraction in background."""
        if self._running:
            return
        self._running = True

        try:
            logger.info("Memory extraction starting...")

            # Get recent conversations
            recent = self.memory.get_conversations(limit=self._messages_since_last * 2)
            conv_text = "\n".join(
                f"[{c['role']}] {c['content'][:300]}" for c in recent[-10:]
            )

            # Get existing memories for dedup
            existing = self.memory.recall_long(limit=20)
            existing_text = "\n".join(
                f"- {m['title']}" for m in existing
            ) if existing else "(no existing memories)"

            # Build prompt
            prompt = EXTRACTION_PROMPT.format(
                msg_count=min(self._messages_since_last * 2, 10),
                existing_memories=existing_text,
                conversation=conv_text[:3000]
            )

            # Ask brain to extract
            response = self.brain._think(prompt)

            # Parse response
            memories = self._parse_memories(response)

            # Save extracted memories
            saved = 0
            for mem in memories[:5]:  # Max 5
                title = mem.get("title", "")
                content = mem.get("content", "")
                category = mem.get("category", "general")
                importance = min(max(mem.get("importance", 5), 1), 10)

                if not title or not content:
                    continue

                # Dedup check
                existing_check = self.memory.semantic_search(title, limit=1)
                if existing_check and existing_check[0]['title'] == title:
                    continue

                self.memory.remember_long(title, content, category, importance)
                saved += 1
                logger.info(f"Extracted memory: [{category}] {title}")

            self._messages_since_last = 0
            self._last_extraction = time.time()
            logger.info(f"Memory extraction complete: {saved} memories saved")

        except Exception as e:
            logger.error(f"Memory extraction error: {e}")
        finally:
            self._running = False

    def _parse_memories(self, response: str) -> List[Dict]:
        """Parse LLM response into memory list."""
        try:
            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            # Try to find JSON array
            response = response.strip()
            if not response.startswith("["):
                # Try to find array in response
                start = response.find("[")
                end = response.rfind("]") + 1
                if start >= 0 and end > start:
                    response = response[start:end]

            result = json.loads(response)
            if isinstance(result, list):
                return result
            return []
        except (json.JSONDecodeError, IndexError):
            return []

    def get_status(self) -> Dict:
        return {
            "running": self._running,
            "messages_since_last": self._messages_since_last,
            "last_extraction": self._last_extraction,
        }
