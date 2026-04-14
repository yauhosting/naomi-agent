"""
NAOMI Agent - Persona Drift Engine v1
Periodically analyzes conversation patterns and feedback to micro-adjust
communication style. Core identity stays fixed; only style drifts.
"""
import json
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("naomi.persona_drift")

# Default style — neutral starting point
DEFAULT_STYLE: Dict[str, Any] = {
    "tone": "warm",
    "verbosity": 0.5,
    "humor": 0.5,
    "formality": 0.3,
    "emoji_level": 0.3,
    "topics_of_interest": [],
}

# How many conversations between drift checks
DRIFT_INTERVAL = 50

# Exponential moving average alpha — lower = slower drift
EMA_ALPHA = 0.3


class PersonaDrift:
    """Analyzes conversations + feedback to evolve communication style."""

    def __init__(self, brain, memory):
        self.brain = brain
        self.memory = memory
        self._cache: Dict[str, Dict[str, Any]] = {}

    def maybe_drift(self, persona: str = "naomi") -> bool:
        """Check if enough conversations have passed to trigger a drift.
        Returns True if drift was applied."""
        latest = self.memory.get_latest_drift(persona)
        since = latest["created_at"] if latest else 0.0
        count = self.memory.get_conversation_count_since(since, persona)

        if count < DRIFT_INTERVAL:
            return False

        logger.info("Drift triggered for %s: %d conversations since last drift", persona, count)
        return self._run_drift(persona, latest)

    def _run_drift(self, persona: str, previous: Optional[Dict]) -> bool:
        """Analyze recent conversations + feedback, produce new style overrides."""
        conversations = self.memory.get_conversations(limit=DRIFT_INTERVAL, persona=persona)
        if not conversations:
            return False

        feedback = self.memory.get_feedback_summary(limit=DRIFT_INTERVAL)
        current_style = self._get_current_style(persona, previous)

        # Build analysis prompt
        conv_sample = "\n".join(
            f"[{c['role']}] {c['content'][:150]}" for c in conversations[-30:]
        )
        feedback_text = (
            f"Feedback score: {feedback['score']:.0%} positive "
            f"(+{feedback['positive']:.1f} / -{feedback['negative']:.1f})"
        )

        analysis_prompt = (
            "Analyze this conversation history and feedback to determine the ideal "
            "communication style adjustments. The goal is to make NAOMI's responses "
            "feel more natural and aligned with what the user enjoys.\n\n"
            f"Current style:\n{json.dumps(current_style, ensure_ascii=False, indent=2)}\n\n"
            f"Recent feedback:\n{feedback_text}\n\n"
            f"Conversation sample:\n{conv_sample}\n\n"
            "Return ONLY a JSON object with these fields (no other text):\n"
            '{\n'
            '  "tone": "warm|neutral|playful|serious",\n'
            '  "verbosity": 0.0-1.0,\n'
            '  "humor": 0.0-1.0,\n'
            '  "formality": 0.0-1.0,\n'
            '  "emoji_level": 0.0-1.0,\n'
            '  "topics_of_interest": ["topic1", "topic2"],\n'
            '  "reasoning": "one sentence why"\n'
            "}"
        )

        try:
            raw = self.brain._think(analysis_prompt)
            new_style = self._parse_style(raw)
        except Exception as e:
            logger.error("Drift analysis failed: %s", e)
            return False

        # Apply EMA blending
        blended = self._blend_styles(current_style, new_style)
        reasoning = new_style.get("reasoning", "periodic drift")

        # Persist
        version = (previous["version"] + 1) if previous else 1
        self.memory.save_drift(
            persona=persona,
            version=version,
            style_overrides=json.dumps(blended, ensure_ascii=False),
            trigger_reason=reasoning,
        )
        self._cache[persona] = blended
        logger.info("Drift v%d applied for %s: %s", version, persona, reasoning)
        return True

    def get_style_prompt(self, persona: str = "naomi") -> str:
        """Return a style overlay string to append to the system prompt."""
        style = self._get_current_style(persona)

        tone_map = {
            "warm": "Be warm, caring, and supportive in your responses.",
            "playful": "Be playful, witty, and light-hearted. Use humor naturally.",
            "serious": "Be focused and professional. Keep responses matter-of-fact.",
            "neutral": "Use a balanced, natural conversational tone.",
        }
        parts = [tone_map.get(style.get("tone", "warm"), "")]

        verbosity = style.get("verbosity", 0.5)
        if verbosity < 0.3:
            parts.append("Keep responses very concise — short sentences, no filler.")
        elif verbosity > 0.7:
            parts.append("Give detailed, thorough responses with context.")

        humor = style.get("humor", 0.5)
        if humor > 0.7:
            parts.append("Feel free to joke and be playful.")
        elif humor < 0.3:
            parts.append("Stay focused, minimal humor.")

        formality = style.get("formality", 0.3)
        if formality > 0.7:
            parts.append("Use polite, somewhat formal language.")
        elif formality < 0.3:
            parts.append("Speak casually, like a close friend.")

        emoji = style.get("emoji_level", 0.3)
        if emoji > 0.6:
            parts.append("Use emoji naturally in responses.")
        elif emoji < 0.2:
            parts.append("Avoid emoji unless the user uses them first.")

        topics = style.get("topics_of_interest", [])
        if topics:
            parts.append(f"User is interested in: {', '.join(topics[:5])}.")

        return "\n".join(p for p in parts if p)

    def get_status(self, persona: str = "naomi") -> Dict[str, Any]:
        """Return current drift state for display."""
        latest = self.memory.get_latest_drift(persona)
        style = self._get_current_style(persona)
        feedback = self.memory.get_feedback_summary()
        since = latest["created_at"] if latest else 0.0
        convs_since = self.memory.get_conversation_count_since(since, persona)
        return {
            "version": latest["version"] if latest else 0,
            "style": style,
            "feedback": feedback,
            "conversations_until_next_drift": max(0, DRIFT_INTERVAL - convs_since),
            "last_drift_reason": latest["trigger_reason"] if latest else "none",
        }

    def _get_current_style(self, persona: str,
                           previous: Optional[Dict] = None) -> Dict[str, Any]:
        """Load current style from cache, DB, or default."""
        if persona in self._cache:
            return self._cache[persona]

        if previous is None:
            previous = self.memory.get_latest_drift(persona)

        if previous:
            try:
                style = json.loads(previous["style_overrides"])
                self._cache[persona] = style
                return style
            except (json.JSONDecodeError, KeyError):
                pass

        return dict(DEFAULT_STYLE)

    @staticmethod
    def _parse_style(raw: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        result = json.loads(text.strip())

        # Clamp numeric values to [0, 1]
        for key in ("verbosity", "humor", "formality", "emoji_level"):
            if key in result:
                result[key] = max(0.0, min(1.0, float(result[key])))

        return result

    @staticmethod
    def _blend_styles(current: Dict[str, Any],
                      new: Dict[str, Any]) -> Dict[str, Any]:
        """Blend old and new styles using EMA for gradual drift."""
        blended = {}
        for key in ("verbosity", "humor", "formality", "emoji_level"):
            old_val = current.get(key, 0.5)
            new_val = new.get(key, old_val)
            blended[key] = round(old_val * (1 - EMA_ALPHA) + new_val * EMA_ALPHA, 2)

        # Discrete values — take new if provided
        blended["tone"] = new.get("tone", current.get("tone", "warm"))

        # Topics — merge and deduplicate, keep latest 10
        old_topics = current.get("topics_of_interest", [])
        new_topics = new.get("topics_of_interest", [])
        merged = list(dict.fromkeys(new_topics + old_topics))[:10]
        blended["topics_of_interest"] = merged

        return blended
