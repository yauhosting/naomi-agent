"""
NAOMI Agent - Heartbeat (Main Loop)
The never-stopping life cycle of NAOMI.
Sense -> Think -> Act -> Remember -> Repeat
"""
import asyncio
import time
import signal
import logging
import traceback
from datetime import datetime
from typing import Optional

logger = logging.getLogger("naomi.heartbeat")


class Heartbeat:
    def __init__(self, agent):
        self.agent = agent
        self.config = agent.config.get("heartbeat", {})
        self.interval = self.config.get("interval", 30)
        self.idle_threshold = self.config.get("idle_threshold", 300)
        self.running = False
        self.beat_count = 0
        self.last_activity = time.time()
        self.last_self_check = time.time()
        self.state = "starting"  # starting, active, idle, thinking, executing, error

    async def start(self):
        """Start the heartbeat - NAOMI comes alive."""
        self.running = True
        self.state = "active"
        logger.info("NAOMI heartbeat started")
        self.agent.memory.remember_short("NAOMI started", category="system")
        self.agent.memory.log_conversation("system", "NAOMI heartbeat started")

        while self.running:
            try:
                await self._beat()
            except Exception as e:
                self.state = "error"
                logger.error(f"Heartbeat error: {e}\n{traceback.format_exc()}")
                self.agent.memory.remember_short(f"Heartbeat error: {e}", category="error")
                await asyncio.sleep(5)  # Brief pause on error, then recover

        self.state = "stopped"
        logger.info("NAOMI heartbeat stopped")

    async def _beat(self):
        """One heartbeat cycle: Sense -> Think -> Act -> Remember"""
        self.beat_count += 1
        now = time.time()

        # 1. SENSE - Gather information from environment
        senses = await self._sense()

        # 2. THINK - Process sensory input
        if senses.get("has_command"):
            # User command takes highest priority
            self.state = "executing"
            self.last_activity = now
            command = senses["command"]
            logger.info(f"Processing command: {command[:100]}")
            await self._execute_command(command)

        elif senses.get("has_error"):
            # Auto-diagnose and fix errors
            self.state = "thinking"
            self.last_activity = now
            await self._handle_error(senses["error"])

        elif senses.get("has_pending_task"):
            # Continue pending work
            self.state = "executing"
            self.last_activity = now
            await self._continue_task(senses["pending_task"])

        elif now - self.last_activity > self.idle_threshold:
            # Idle mode - be creative
            self.state = "idle"
            await self._idle_think()

        # 3. Periodic self-check
        if now - self.last_self_check > self.config.get("self_check_interval", 3600):
            await self._self_check()
            self.last_self_check = now

        # Wait for next heartbeat
        self.state = "active"
        await asyncio.sleep(self.interval)

    async def _sense(self) -> dict:
        """Gather sensory input from all sources."""
        result = {
            "has_command": False,
            "has_error": False,
            "has_pending_task": False,
            "timestamp": time.time(),
        }

        # Check for user commands (from dashboard/API)
        if hasattr(self.agent, 'command_queue') and not self.agent.command_queue.empty():
            try:
                cmd = self.agent.command_queue.get_nowait()
                result["has_command"] = True
                result["command"] = cmd
            except asyncio.QueueEmpty:
                pass

        # Check for pending tasks in memory
        pending = self.agent.memory.get_pending_tasks()
        if pending:
            result["has_pending_task"] = True
            result["pending_task"] = pending[0]

        # Check system health (via senses module if available)
        if hasattr(self.agent, 'senses'):
            try:
                health = await self.agent.senses.check_system()
                if health.get("alerts"):
                    result["has_error"] = True
                    result["error"] = health["alerts"][0]
            except Exception:
                pass

        return result

    async def _execute_command(self, command: str):
        """Execute a user command through the brain."""
        task_id = self.agent.memory.add_task(command)
        self.agent.memory.log_conversation("user", command)

        try:
            # Left brain analyzes the task
            context = self.agent.memory.build_context()
            plan = self.agent.brain.analyze(command, context)

            logger.info(f"Plan: {plan.get('understanding', 'unknown')}")
            self.agent.memory.remember_short(
                f"Planning: {plan.get('understanding', command)}", category="task"
            )

            # Check for missing tools
            tools_needed = plan.get("tools_needed", [])
            for tool in tools_needed:
                if not self.agent.tool_manager.has_tool(tool):
                    logger.info(f"Missing tool: {tool}, searching...")
                    await self.agent.tool_manager.auto_install(tool)

            # Execute each step
            results = []
            for step in plan.get("steps", []):
                action = step.get("action", "")
                tool = step.get("tool", "shell")
                details = step.get("details", "")

                logger.info(f"Step {step.get('step', '?')}: {action}")
                result = await self.agent.execute_action(tool, details)
                results.append(result)

                # Check if step failed
                if result.get("error"):
                    fix = self.agent.brain.debug(
                        result["error"],
                        f"While executing: {details}"
                    )
                    logger.info(f"Auto-fix attempt: {fix[:200]}")
                    retry_result = await self.agent.execute_action("shell", fix)
                    results.append(retry_result)

            # Compile results
            summary = "\n".join(str(r) for r in results)
            self.agent.memory.complete_task(task_id, summary[:2000])
            self.agent.memory.log_conversation("naomi", f"Completed: {command}\nResult: {summary[:500]}")
            self.agent.memory.remember_long(
                f"Task: {command[:100]}",
                summary[:1000],
                category="task_result",
                importance=7
            )

        except Exception as e:
            error_msg = f"Failed: {e}\n{traceback.format_exc()}"
            self.agent.memory.complete_task(task_id, error_msg[:2000], status="failed")
            self.agent.memory.log_conversation("naomi", f"Failed: {command}\nError: {e}")
            logger.error(error_msg)

    async def _handle_error(self, error: str):
        """Auto-diagnose and fix errors."""
        logger.info(f"Auto-handling error: {error[:200]}")
        context = self.agent.memory.build_context()
        fix = self.agent.brain.debug(error, context)
        self.agent.memory.remember_short(f"Auto-fix: {fix[:200]}", category="fix")

    async def _continue_task(self, task: dict):
        """Continue a pending task."""
        await self._execute_command(task["task"])

    async def _idle_think(self):
        """Right brain creative mode when idle."""
        logger.info("Entering idle/creative mode...")
        context = self.agent.memory.build_context()
        tasks = self.agent.memory.get_recent_tasks(10)
        history = "\n".join(f"- [{t['status']}] {t['task']}" for t in tasks)

        reflection = self.agent.brain.reflect(f"{context}\n\nRecent history:\n{history}")

        if reflection.get("proactive_tasks"):
            for task in reflection["proactive_tasks"][:1]:  # Only suggest top 1
                self.agent.memory.remember_short(
                    f"Suggestion: {task}", category="suggestion"
                )
                logger.info(f"Proactive suggestion: {task}")

        if reflection.get("self_improvements"):
            for imp in reflection["self_improvements"][:1]:
                self.agent.memory.remember_short(
                    f"Self-improvement idea: {imp}", category="improvement"
                )

    async def _self_check(self):
        """Periodic self-health check."""
        logger.info(f"Self-check: beat #{self.beat_count}, state={self.state}")
        self.agent.memory.remember_short(
            f"Self-check OK: {self.beat_count} beats, uptime {time.time() - self.agent.start_time:.0f}s",
            category="system"
        )

    def stop(self):
        """Stop the heartbeat gracefully."""
        self.running = False
        logger.info("Heartbeat stopping...")

    def get_status(self) -> dict:
        return {
            "state": self.state,
            "beat_count": self.beat_count,
            "last_activity": self.last_activity,
            "uptime": time.time() - self.agent.start_time if hasattr(self.agent, 'start_time') else 0,
            "interval": self.interval,
        }
