"""
NAOMI Agent - Plan-Execute-Reflect Loop

Structured planner that decomposes tasks into steps, executes them
with real tools, reflects on outcomes, and adjusts dynamically.

Flow: plan -> execute each step -> reflect -> adjust -> repeat
"""
import json
import time
import asyncio
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("naomi.planner")

# Reflection outcomes
CONTINUE = "continue"
RETRY = "retry"
REPLAN = "replan"
ABORT = "abort"

MAX_RETRIES_PER_STEP = 2


@dataclass(frozen=True)
class PlanStep:
    """A single step in an execution plan."""
    step: int
    action: str
    tool: str
    expected: str


@dataclass(frozen=True)
class StepResult:
    """Result of executing a single step."""
    step_num: int
    action: str
    tool: str
    expected: str
    actual: str
    success: bool
    duration: float


@dataclass(frozen=True)
class Reflection:
    """Reflection on a step's outcome."""
    step_num: int
    expected: str
    actual: str
    outcome: str  # continue, retry, replan, abort
    reasoning: str


@dataclass
class PlanExecutionResult:
    """Final result of a plan-execute-reflect run."""
    success: bool
    steps_completed: int
    total_steps: int
    plan: List[Dict[str, Any]]
    reflections: List[Dict[str, Any]]
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "steps_completed": self.steps_completed,
            "total_steps": self.total_steps,
            "plan": self.plan,
            "reflections": self.reflections,
            "execution_history": self.execution_history,
            "error": self.error,
        }


def _parse_plan(raw: str) -> List[Dict[str, Any]]:
    """Parse brain output into a list of plan step dicts.

    Accepts either a JSON array or numbered text lines.
    Returns list of dicts with keys: step, action, tool, expected.
    """
    raw = raw.strip()

    # Try JSON first
    text = raw
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    text = text.strip()

    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and len(parsed) > 0:
                steps = []
                for i, item in enumerate(parsed, 1):
                    if isinstance(item, dict):
                        steps.append({
                            "step": item.get("step", i),
                            "action": item.get("action", ""),
                            "tool": item.get("tool", ""),
                            "expected": item.get("expected", ""),
                        })
                return steps
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: parse numbered lines
    steps: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading number/bullet: "1.", "1)", "- ", "* "
        for prefix in ("- ", "* "):
            if line.startswith(prefix):
                line = line[len(prefix):]
                break
        if len(line) > 1 and line[0].isdigit():
            # Remove "1. " or "1) " prefix
            rest = line.lstrip("0123456789")
            if rest and rest[0] in (".", ")"):
                line = rest[1:].strip()

        if not line:
            continue

        steps.append({
            "step": len(steps) + 1,
            "action": line,
            "tool": "",
            "expected": "",
        })

    return steps


def _parse_reflection(raw: str) -> Dict[str, str]:
    """Parse brain reflection output into outcome + reasoning."""
    raw_lower = raw.strip().lower()

    # Try JSON
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    text = text.strip()

    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            outcome = parsed.get("outcome", "continue").lower()
            if outcome in (CONTINUE, RETRY, REPLAN, ABORT):
                return {
                    "outcome": outcome,
                    "reasoning": parsed.get("reasoning", ""),
                }
        except (json.JSONDecodeError, TypeError):
            pass

    # Keyword detection fallback
    if "abort" in raw_lower:
        outcome = ABORT
    elif "replan" in raw_lower or "re-plan" in raw_lower:
        outcome = REPLAN
    elif "retry" in raw_lower:
        outcome = RETRY
    else:
        outcome = CONTINUE

    return {"outcome": outcome, "reasoning": raw.strip()[:500]}


class PlanExecuteReflect:
    """Plan-Execute-Reflect loop for structured task execution."""

    def __init__(self) -> None:
        self._history: List[Dict[str, Any]] = []

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Execution history for learning."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def plan(self, task: str, brain: Any, context: str = "") -> List[Dict[str, Any]]:
        """Ask brain to decompose a task into numbered steps with expected outcomes.

        Returns a list of dicts:
        [{"step": 1, "action": "...", "tool": "...", "expected": "..."}, ...]
        """
        system = (
            "You are a task planner. Decompose the task into numbered steps. "
            "Reply in JSON array ONLY. Each element: "
            '{"step": N, "action": "what to do", "tool": "tool_name", '
            '"expected": "expected outcome"}. '
            "Available tools: shell, python_exec, file_read, file_write, "
            "web_search, git, pip_install, screenshot, click, type_text, "
            "key_press, open_app, web_fetch, generate_image, deploy_web. "
            "Keep it concise. Max 10 steps."
        )
        prompt = f"Task: {task}"
        if context:
            prompt += f"\n\nContext:\n{context}"

        raw = brain._think(prompt, system)
        if not raw or raw.startswith("[Brain"):
            logger.warning("Brain returned no plan: %s", raw[:200] if raw else "(empty)")
            return []

        steps = _parse_plan(raw)
        logger.info("Plan generated: %d steps for task: %s", len(steps), task[:80])
        return steps

    async def execute_step(
        self,
        step: Dict[str, Any],
        executor: Any,
    ) -> Dict[str, Any]:
        """Execute a single plan step via the executor.

        Returns: {"success": bool, "output": str, "duration": float}
        """
        tool = step.get("tool", "")
        action = step.get("action", "")

        start = time.monotonic()
        result: Dict[str, Any]

        if tool and hasattr(executor, "execute"):
            # Map common tool names to executor params
            param = action
            if tool == "shell":
                param = action
            elif tool == "python_exec":
                param = action
            elif tool == "file_read":
                param = action
            elif tool == "web_search":
                param = action

            try:
                result = await executor.execute(tool, param)
            except Exception as e:
                result = {"success": False, "error": str(e)}
        else:
            # No specific tool -- try shell as default
            try:
                result = await executor.execute("shell", action)
            except Exception as e:
                result = {"success": False, "error": str(e)}

        elapsed = time.monotonic() - start
        success = result.get("success", False) if isinstance(result, dict) else False
        output = ""
        if isinstance(result, dict):
            output = str(result.get("output", result.get("error", "")))[:2000]
        else:
            output = str(result)[:2000]

        return {"success": success, "output": output, "duration": elapsed}

    def reflect(
        self,
        step: Dict[str, Any],
        expected: str,
        actual: str,
        brain: Any,
    ) -> Dict[str, str]:
        """Compare expected vs actual outcome and decide next action.

        Returns: {"outcome": "continue|retry|replan|abort", "reasoning": "..."}
        """
        system = (
            "You are a task reflector. Compare expected vs actual results. "
            "Reply in JSON ONLY: "
            '{"outcome": "continue|retry|replan|abort", "reasoning": "why"}. '
            "continue = step succeeded, move on. "
            "retry = step failed but worth retrying. "
            "replan = remaining steps need to change. "
            "abort = task cannot be completed."
        )
        prompt = (
            f"Step: {step.get('action', '')}\n"
            f"Tool: {step.get('tool', '')}\n"
            f"Expected: {expected}\n"
            f"Actual result: {actual[:1000]}"
        )

        raw = brain._think(prompt, system)
        if not raw or raw.startswith("[Brain"):
            # If brain is offline, check success heuristically
            if "error" in actual.lower() or "fail" in actual.lower():
                return {"outcome": RETRY, "reasoning": "Brain offline; actual output contains error signals"}
            return {"outcome": CONTINUE, "reasoning": "Brain offline; assuming success"}

        return _parse_reflection(raw)

    async def run(
        self,
        task: str,
        executor: Any,
        brain: Any,
        context: str = "",
        max_steps: int = 10,
    ) -> Dict[str, Any]:
        """Full plan-execute-reflect loop.

        Returns structured result dict.
        """
        logger.info("PlanExecuteReflect starting: %s", task[:100])
        run_start = time.monotonic()

        # Phase 1: Plan
        steps = await asyncio.to_thread(self.plan, task, brain, context)
        if not steps:
            result = PlanExecutionResult(
                success=False,
                steps_completed=0,
                total_steps=0,
                plan=[],
                reflections=[],
                error="Failed to generate plan",
            )
            return result.to_dict()

        # Limit steps
        steps = steps[:max_steps]

        all_reflections: List[Dict[str, Any]] = []
        execution_log: List[Dict[str, Any]] = []
        completed = 0

        i = 0
        while i < len(steps):
            step = steps[i]
            step_num = step.get("step", i + 1)
            logger.info("Executing step %d/%d: %s", step_num, len(steps), step.get("action", "")[:80])

            retries = 0
            step_done = False

            while retries <= MAX_RETRIES_PER_STEP and not step_done:
                # Phase 2: Execute
                exec_result = await self.execute_step(step, executor)

                entry = {
                    "step": step_num,
                    "action": step.get("action", ""),
                    "tool": step.get("tool", ""),
                    "expected": step.get("expected", ""),
                    "actual": exec_result["output"][:500],
                    "success": exec_result["success"],
                    "duration": round(exec_result["duration"], 2),
                    "retry": retries,
                }
                execution_log.append(entry)

                # Phase 3: Reflect
                reflection = await asyncio.to_thread(
                    self.reflect,
                    step,
                    step.get("expected", ""),
                    exec_result["output"],
                    brain,
                )

                ref_entry = {
                    "step": step_num,
                    "expected": step.get("expected", ""),
                    "actual": exec_result["output"][:300],
                    "outcome": reflection["outcome"],
                    "reasoning": reflection["reasoning"][:300],
                }
                all_reflections.append(ref_entry)

                outcome = reflection["outcome"]

                if outcome == CONTINUE:
                    completed += 1
                    step_done = True
                elif outcome == RETRY:
                    retries += 1
                    if retries > MAX_RETRIES_PER_STEP:
                        logger.warning("Step %d: max retries exceeded", step_num)
                        step_done = True  # Move on
                    else:
                        logger.info("Step %d: retrying (%d/%d)", step_num, retries, MAX_RETRIES_PER_STEP)
                elif outcome == REPLAN:
                    logger.info("Step %d: replanning remaining steps", step_num)
                    remaining_task = (
                        f"Original task: {task}\n"
                        f"Completed steps: {completed}\n"
                        f"Last step failed: {step.get('action', '')}\n"
                        f"Error: {exec_result['output'][:500]}\n"
                        f"Generate remaining steps to complete the task."
                    )
                    new_steps = await asyncio.to_thread(self.plan, remaining_task, brain, context)
                    if new_steps:
                        # Replace remaining steps
                        for j, ns in enumerate(new_steps):
                            ns["step"] = completed + j + 1
                        steps = steps[:i] + new_steps[:max_steps - i]
                    step_done = True  # Move to next (replanned) step
                elif outcome == ABORT:
                    logger.warning("Step %d: aborting plan", step_num)
                    self._record_history(task, steps, all_reflections, execution_log, False)
                    result = PlanExecutionResult(
                        success=False,
                        steps_completed=completed,
                        total_steps=len(steps),
                        plan=[s for s in steps],
                        reflections=all_reflections,
                        execution_history=execution_log,
                        error=f"Aborted at step {step_num}: {reflection['reasoning'][:200]}",
                    )
                    return result.to_dict()

            i += 1

        elapsed = time.monotonic() - run_start
        success = completed > 0 and completed >= len(steps) * 0.5
        logger.info(
            "PlanExecuteReflect done: %d/%d steps in %.1fs (success=%s)",
            completed, len(steps), elapsed, success,
        )

        self._record_history(task, steps, all_reflections, execution_log, success)

        result = PlanExecutionResult(
            success=success,
            steps_completed=completed,
            total_steps=len(steps),
            plan=[s for s in steps],
            reflections=all_reflections,
            execution_history=execution_log,
        )
        return result.to_dict()

    # ------------------------------------------------------------------
    # History tracking
    # ------------------------------------------------------------------

    def _record_history(
        self,
        task: str,
        plan: List[Dict[str, Any]],
        reflections: List[Dict[str, Any]],
        execution_log: List[Dict[str, Any]],
        success: bool,
    ) -> None:
        """Record execution for learning purposes."""
        entry = {
            "task": task[:200],
            "timestamp": time.time(),
            "success": success,
            "steps_count": len(plan),
            "reflections_summary": [
                {"step": r.get("step"), "outcome": r.get("outcome")}
                for r in reflections
            ],
        }
        self._history.append(entry)

        # Keep history bounded
        if len(self._history) > 100:
            self._history = self._history[-100:]
