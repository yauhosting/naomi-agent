"""
NAOMI Agent - Context Compaction Engine
Inspired by Claude Code's 3-tier compaction system:

Tier 1: Micro-compaction (free) — clear old tool results, keep conversation
Tier 2: Session memory compact (cheap) — reuse existing summaries
Tier 3: Full compact (expensive) — LLM generates structured 9-section summary

Safety:
- Circuit breaker: stop after 3 consecutive failures
- Anti-recursion guard: no compact during compact
- 13K token buffer before triggering
"""
import time
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger("naomi.compaction")

# Tool result placeholder after clearing
CLEARED_PLACEHOLDER = "[Old tool result cleared to save context space]"

# Token budget constants
CONTEXT_WINDOW = 128000  # MiniMax M2.7 context window
AUTO_COMPACT_BUFFER = 13000  # Trigger compaction when this close to limit
WARNING_BUFFER = 20000
BLOCKING_BUFFER = 3000

# Micro-compact: keep N most recent tool results
KEEP_RECENT_TOOLS = 5

# Structured summary template
NINE_SECTION_TEMPLATE = """Compress this conversation into a structured summary.
Use EXACTLY this format:

## 1. USER INTENT
What the user wants to accomplish (1-2 sentences)

## 2. KEY CONCEPTS
Important technical concepts, tools, and topics discussed (bullet points)

## 3. FILES & CODE
Specific files mentioned or modified, with relevant code snippets

## 4. DECISIONS MADE
What was decided, chosen, or agreed upon

## 5. ERRORS & FIXES
Problems encountered and how they were resolved

## 6. CURRENT STATE
Where things stand right now

## 7. PENDING TASKS
What still needs to be done

## 8. USER PREFERENCES
Learned preferences, communication style, priorities

## 9. NEXT STEPS
What to do next (specific, actionable)

Conversation to compress:
{conversation}"""


class CompactionEngine:
    """3-tier context compaction with safety mechanisms."""

    def __init__(self, memory, brain=None):
        self.memory = memory
        self.brain = brain
        self._compacting = False  # Anti-recursion guard
        self._failure_count = 0   # Circuit breaker counter
        self._max_failures = 3    # Circuit breaker threshold
        self._last_compact_time = 0
        self._compact_cooldown = 60  # Min seconds between compactions

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text) // 3

    def should_compact(self, total_tokens: int) -> bool:
        """Check if compaction should trigger (Claude Code: 13K buffer)."""
        if self._compacting:
            return False  # Anti-recursion
        if self._failure_count >= self._max_failures:
            logger.warning("Circuit breaker: compaction disabled after %d failures", self._max_failures)
            return False
        if time.time() - self._last_compact_time < self._compact_cooldown:
            return False
        threshold = CONTEXT_WINDOW - AUTO_COMPACT_BUFFER
        return total_tokens > threshold

    def compact(self, conversations: List[Dict], tool_results: List[Dict] = None) -> Dict[str, Any]:
        """
        Run compaction with 3-tier fallback:
        Tier 1: Micro-compact (clear old tool results)
        Tier 2: Session memory (reuse existing summaries)
        Tier 3: Full compact (LLM summary)
        """
        if self._compacting:
            return {"action": "skipped", "reason": "anti-recursion"}

        self._compacting = True
        self._last_compact_time = time.time()

        try:
            # Calculate current size
            total_tokens = sum(self.estimate_tokens(c.get("content", "")) for c in conversations)
            logger.info(f"Compaction triggered: {total_tokens} tokens")

            # Tier 1: Micro-compaction
            result = self._micro_compact(conversations, tool_results)
            if result["tokens_saved"] > 0:
                new_total = total_tokens - result["tokens_saved"]
                if new_total < CONTEXT_WINDOW - AUTO_COMPACT_BUFFER:
                    logger.info(f"Tier 1 micro-compact saved {result['tokens_saved']} tokens")
                    self._failure_count = 0
                    return {"tier": 1, "action": "micro_compact", **result}

            # Tier 2: Session memory reuse
            result2 = self._session_memory_compact(conversations)
            if result2:
                logger.info("Tier 2 session memory compact succeeded")
                self._failure_count = 0
                return {"tier": 2, "action": "session_memory", **result2}

            # Tier 3: Full LLM compact
            if self.brain:
                result3 = self._full_compact(conversations)
                if result3:
                    logger.info("Tier 3 full compact succeeded")
                    self._failure_count = 0
                    return {"tier": 3, "action": "full_compact", **result3}

            # All tiers failed
            self._failure_count += 1
            logger.error(f"All compaction tiers failed (failure {self._failure_count}/{self._max_failures})")
            return {"action": "failed", "failures": self._failure_count}

        except Exception as e:
            self._failure_count += 1
            logger.error(f"Compaction error: {e}")
            return {"action": "error", "error": str(e)}
        finally:
            self._compacting = False

    def _micro_compact(self, conversations: List[Dict], tool_results: List[Dict] = None) -> Dict:
        """
        Tier 1: Clear old tool results without touching conversation.
        Cheapest operation - no LLM calls needed.
        Keep N most recent tool results, clear the rest.
        """
        tokens_saved = 0
        cleared_count = 0

        # Find tool-result-like entries in conversations
        tool_indices = []
        for i, conv in enumerate(conversations):
            content = conv.get("content", "")
            # Detect tool results by patterns
            if any(marker in content for marker in [
                "{'success':", '{"success":', "Shell:", "output:", "returncode:",
                "results:", "Error:", "Traceback"
            ]):
                tool_indices.append(i)

        # Keep the most recent N, clear the rest
        if len(tool_indices) > KEEP_RECENT_TOOLS:
            to_clear = tool_indices[:-KEEP_RECENT_TOOLS]
            for idx in to_clear:
                old_content = conversations[idx].get("content", "")
                old_tokens = self.estimate_tokens(old_content)
                new_tokens = self.estimate_tokens(CLEARED_PLACEHOLDER)
                tokens_saved += old_tokens - new_tokens
                conversations[idx]["content"] = CLEARED_PLACEHOLDER
                cleared_count += 1

        return {
            "tokens_saved": tokens_saved,
            "cleared_count": cleared_count,
        }

    def _session_memory_compact(self, conversations: List[Dict]) -> Optional[Dict]:
        """
        Tier 2: Reuse existing session summary if available.
        Cheap - just reads from DB, no LLM call.
        """
        # Check for existing session summaries
        rows = self.memory.conn.execute(
            "SELECT summary, message_count FROM session_summaries ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        if not rows:
            return None

        existing_summary = rows["summary"]
        summarized_count = rows["message_count"]

        # Use existing summary + keep recent messages
        recent = conversations[-20:]  # Always keep last 20
        total_conv = len(conversations)

        if total_conv <= 20:
            return None  # Not enough to compact

        compressed = f"=== Previous Context (summarized from {summarized_count} messages) ===\n{existing_summary}\n\n=== Recent Messages ===\n"
        for c in recent:
            compressed += f"[{c.get('role', '?')}] {c.get('content', '')[:300]}\n"

        return {
            "compressed_text": compressed,
            "original_count": total_conv,
            "kept_recent": len(recent),
            "tokens_estimate": self.estimate_tokens(compressed),
        }

    def _full_compact(self, conversations: List[Dict]) -> Optional[Dict]:
        """
        Tier 3: Full LLM-based compression with 9-section summary.
        Most expensive - requires brain/LLM call.
        """
        if not self.brain:
            return None

        # Split: older messages get summarized, recent kept verbatim
        split_point = max(len(conversations) - 20, 0)
        older = conversations[:split_point]
        recent = conversations[split_point:]

        if not older:
            return None

        # Build conversation text for summarization
        conv_text = "\n".join(
            f"[{c.get('role', '?')}] {c.get('content', '')[:200]}"
            for c in older
        )

        # Generate 9-section summary
        prompt = NINE_SECTION_TEMPLATE.format(conversation=conv_text[:4000])
        summary = self.brain._think(prompt)

        if not summary or summary.startswith("[Brain"):
            return None

        # Store summary
        self.memory.conn.execute(
            "INSERT INTO session_summaries (summary, message_count, created_at) VALUES (?, ?, ?)",
            (summary, len(older), time.time())
        )
        self.memory.conn.commit()

        # Build compressed output
        compressed = f"=== Session Summary ({len(older)} messages compressed) ===\n{summary}\n\n=== Recent ({len(recent)} messages) ===\n"
        for c in recent:
            compressed += f"[{c.get('role', '?')}] {c.get('content', '')}\n"

        return {
            "compressed_text": compressed,
            "original_count": len(conversations),
            "summarized_count": len(older),
            "kept_recent": len(recent),
            "tokens_estimate": self.estimate_tokens(compressed),
        }

    def reset_circuit_breaker(self):
        """Manual reset of circuit breaker."""
        self._failure_count = 0
        logger.info("Circuit breaker reset")

    def get_status(self) -> Dict:
        return {
            "compacting": self._compacting,
            "failures": self._failure_count,
            "max_failures": self._max_failures,
            "circuit_breaker_tripped": self._failure_count >= self._max_failures,
            "last_compact": self._last_compact_time,
        }
