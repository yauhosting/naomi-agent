"""
NAOMI Agent - Brain (Dual-Brain Architecture)
Left Brain: Logic, analysis, code, debugging
Right Brain: Creativity, strategy, trends, risk assessment
Subconscious: Background processing, memory consolidation, self-optimization
"""
import subprocess
import json
import time
import asyncio
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("naomi.brain")


class Brain:
    def __init__(self, config: dict):
        self.config = config
        self.primary = config.get("primary", {})
        self.fallback = config.get("fallback", {})
        self._claude_available = None

    def _check_claude_cli(self) -> bool:
        if self._claude_available is not None:
            return self._claude_available
        try:
            result = subprocess.run(
                ["bash", "-lc", "which claude"],
                capture_output=True, text=True, timeout=10
            )
            self._claude_available = result.returncode == 0
        except Exception:
            self._claude_available = False
        logger.info(f"Claude CLI available: {self._claude_available}")
        return self._claude_available

    def _call_claude(self, prompt: str, system_prompt: str = "") -> str:
        """Call Claude CLI with a prompt."""
        cmd = 'claude -p'
        if system_prompt:
            full_prompt = f"[System: {system_prompt}]\n\n{prompt}"
        else:
            full_prompt = prompt

        try:
            result = subprocess.run(
                ["bash", "-lc", f'echo {json.dumps(full_prompt)} | claude -p --no-input'],
                capture_output=True, text=True,
                timeout=self.primary.get("timeout", 120)
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            else:
                logger.warning(f"Claude CLI error: {result.stderr[:200]}")
                return self._call_fallback(prompt, system_prompt)
        except subprocess.TimeoutExpired:
            logger.warning("Claude CLI timeout, using fallback")
            return self._call_fallback(prompt, system_prompt)
        except Exception as e:
            logger.error(f"Claude CLI failed: {e}")
            return self._call_fallback(prompt, system_prompt)

    def _call_fallback(self, prompt: str, system_prompt: str = "") -> str:
        """Call MiniMax API as fallback."""
        import httpx
        api_key = self.fallback.get("api_key", "")
        if not api_key:
            import os
            api_key = os.environ.get("MINIMAX_API_KEY", "")
        if not api_key:
            return "[Brain offline: No API key configured. Set MINIMAX_API_KEY or configure Claude CLI.]"

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            resp = httpx.post(
                f"{self.fallback.get('base_url', 'https://api.minimax.chat/v1')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": self.fallback.get("model", "MiniMax-Text-01"), "messages": messages},
                timeout=60
            )
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Fallback API failed: {e}")
            return f"[Brain error: {e}]"

    def _think(self, prompt: str, system_prompt: str = "") -> str:
        """Core thinking method - routes to available backend."""
        if self._check_claude_cli():
            return self._call_claude(prompt, system_prompt)
        return self._call_fallback(prompt, system_prompt)

    # === Left Brain: Logic & Analysis ===
    def analyze(self, task: str, context: str = "") -> Dict[str, Any]:
        """Left brain: Break down a task into steps."""
        system = """You are NAOMI's left brain - the logical, analytical side.
Your job is to break down tasks into concrete, executable steps.
Always respond in JSON format:
{
  "understanding": "what the task is about",
  "steps": [{"step": 1, "action": "...", "tool": "...", "details": "..."}],
  "tools_needed": ["tool1", "tool2"],
  "estimated_complexity": "low/medium/high",
  "risks": ["risk1"]
}"""
        prompt = f"Task: {task}"
        if context:
            prompt += f"\n\nContext:\n{context}"

        response = self._think(prompt, system)
        try:
            # Try to extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "understanding": task,
                "steps": [{"step": 1, "action": "execute", "tool": "shell", "details": response}],
                "tools_needed": [],
                "estimated_complexity": "medium",
                "risks": []
            }

    def debug(self, error: str, context: str = "") -> str:
        """Left brain: Diagnose and fix errors."""
        system = "You are NAOMI's debugger. Analyze errors and provide exact fix commands."
        prompt = f"Error:\n{error}"
        if context:
            prompt += f"\n\nContext:\n{context}"
        return self._think(prompt, system)

    def write_code(self, spec: str, language: str = "python") -> str:
        """Left brain: Generate code."""
        system = f"You are NAOMI's code writer. Write clean, working {language} code. Output ONLY the code, no explanations."
        return self._think(spec, system)

    # === Right Brain: Creativity & Strategy ===
    def strategize(self, goal: str, context: str = "") -> Dict[str, Any]:
        """Right brain: Create a strategy for a goal."""
        system = """You are NAOMI's right brain - the creative, strategic side.
Your job is to think big picture, find opportunities, and create strategies.
Respond in JSON format:
{
  "analysis": "situation analysis",
  "strategy": "recommended approach",
  "opportunities": ["opp1", "opp2"],
  "risks": ["risk1"],
  "creative_ideas": ["idea1", "idea2"],
  "next_steps": ["step1", "step2"]
}"""
        prompt = f"Goal: {goal}"
        if context:
            prompt += f"\n\nContext:\n{context}"

        response = self._think(prompt, system)
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "analysis": goal,
                "strategy": response,
                "opportunities": [],
                "risks": [],
                "creative_ideas": [],
                "next_steps": []
            }

    def generate_ideas(self, topic: str) -> str:
        """Right brain: Brainstorm ideas."""
        system = "You are NAOMI's creative mind. Generate innovative, practical ideas."
        return self._think(f"Generate creative ideas for: {topic}", system)

    # === Subconscious: Background Processing ===
    def reflect(self, history: str) -> Dict[str, Any]:
        """Subconscious: Review recent activity and suggest improvements."""
        system = """You are NAOMI's subconscious - always thinking in the background.
Review recent activity and suggest what to do next.
Respond in JSON:
{
  "observations": ["obs1"],
  "suggestions": ["sug1"],
  "self_improvements": ["imp1"],
  "proactive_tasks": ["task1"],
  "priority": "high/medium/low"
}"""
        response = self._think(f"Review this recent activity:\n{history}", system)
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "observations": [],
                "suggestions": [response],
                "self_improvements": [],
                "proactive_tasks": [],
                "priority": "low"
            }

    def consolidate_memories(self, memories: str) -> str:
        """Subconscious: Summarize and consolidate memories."""
        system = "Summarize these memories into key insights. Be concise but preserve important details."
        return self._think(f"Consolidate these memories:\n{memories}", system)

    # === Combined Thinking ===
    def think(self, prompt: str, context: str = "") -> str:
        """General thinking - uses both brains."""
        system = """You are NAOMI, an autonomous AI agent.
You have full authority to plan and execute tasks.
Be direct, actionable, and proactive.
If you need a tool that isn't available, say what tool you need and how to install it."""
        full_prompt = prompt
        if context:
            full_prompt += f"\n\nContext:\n{context}"
        return self._think(full_prompt, system)
