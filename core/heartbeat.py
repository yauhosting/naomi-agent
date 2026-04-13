"""
NAOMI Agent - Heartbeat (Main Loop) v2
The never-stopping life cycle of NAOMI.
Sense -> Think -> Act -> Remember -> Repeat

v2: Smart task routing - brain classifies tasks before execution
"""
import asyncio
import time
import logging
import traceback
import json
from typing import Optional

logger = logging.getLogger("naomi.heartbeat")

# Available action types that map to real executor methods
AVAILABLE_ACTIONS = {
    "shell": "Execute a shell/terminal command",
    "python": "Write and execute Python code",
    "file_read": "Read a file (param: file path)",
    "file_write": "Write a file (param: path|||content)",
    "git": "Execute a git command",
    "pip_install": "Install a Python package",
    "web_search": "Search the web via DuckDuckGo",
}


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
        self.state = "starting"

    async def start(self):
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
                await asyncio.sleep(5)

        self.state = "stopped"
        logger.info("NAOMI heartbeat stopped")

    async def _beat(self):
        self.beat_count += 1
        now = time.time()
        senses = await self._sense()

        if senses.get("has_command"):
            self.state = "executing"
            self.last_activity = now
            await self._execute_command(senses["command"])
        elif senses.get("has_error"):
            self.state = "thinking"
            self.last_activity = now
            await self._handle_error(senses["error"])
        elif senses.get("has_pending_task"):
            self.state = "executing"
            self.last_activity = now
            await self._continue_task(senses["pending_task"])
        elif now - self.last_activity > self.idle_threshold:
            self.state = "idle"
            await self._idle_think()

        if now - self.last_self_check > self.config.get("self_check_interval", 3600):
            await self._self_check()
            self.last_self_check = now

        self.state = "active"
        await asyncio.sleep(self.interval)

    async def _sense(self) -> dict:
        result = {"has_command": False, "has_error": False, "has_pending_task": False}

        if hasattr(self.agent, 'command_queue') and not self.agent.command_queue.empty():
            try:
                cmd = self.agent.command_queue.get_nowait()
                result["has_command"] = True
                result["command"] = cmd
            except asyncio.QueueEmpty:
                pass

        pending = self.agent.memory.get_pending_tasks()
        if pending:
            result["has_pending_task"] = True
            result["pending_task"] = pending[0]

        return result

    async def _execute_command(self, command: str):
        """Smart command execution with task classification."""
        task_id = self.agent.memory.add_task(command)
        self.agent.memory.log_conversation("user", command)

        try:
            context = self.agent.memory.build_context()

            # Step 1: Classify the task
            task_type = self._classify_task(command)
            logger.info(f"Task classified as: {task_type}")

            if task_type == "think":
                # Pure thinking task - just let the brain answer
                result = await self._handle_think_task(command, context)
            elif task_type == "search":
                # Web search task
                result = await self._handle_search_task(command, context)
            elif task_type == "code":
                # Code generation + execution
                result = await self._handle_code_task(command, context)
            elif task_type == "execute":
                # Shell command execution
                result = await self._handle_execute_task(command, context)
            elif task_type == "project":
                # Multi-step project
                result = await self._handle_project_task(command, context)
            else:
                result = await self._handle_think_task(command, context)

            self.agent.memory.complete_task(task_id, str(result)[:2000])
            self.agent.memory.log_conversation("naomi", str(result)[:1000])
            self.agent.memory.remember_long(
                f"Task: {command[:100]}", str(result)[:1000],
                category="task_result", importance=7
            )

        except Exception as e:
            error_msg = f"Failed: {e}\n{traceback.format_exc()}"
            self.agent.memory.complete_task(task_id, error_msg[:2000], status="failed")
            self.agent.memory.log_conversation("naomi", f"Failed: {command}\nError: {e}")
            logger.error(error_msg)

    def _classify_task(self, command: str) -> str:
        """Classify task type based on keywords."""
        cmd_lower = command.lower()

        # Search tasks
        search_keywords = ["search", "find", "look up", "搜尋", "搜索", "查找", "查詢", "google"]
        if any(k in cmd_lower for k in search_keywords):
            return "search"

        # Code tasks
        code_keywords = ["write code", "create script", "寫代碼", "寫程式", "build", "implement",
                         "develop", "create a", "make a", "建立", "開發"]
        if any(k in cmd_lower for k in code_keywords):
            return "code"

        # Execute tasks (direct commands)
        exec_keywords = ["run", "execute", "install", "delete", "remove", "kill", "restart",
                         "執行", "安裝", "刪除", "重啟", "update", "upgrade"]
        if any(k in cmd_lower for k in exec_keywords):
            return "execute"

        # Project tasks (multi-step)
        project_keywords = ["project", "automate", "automation", "自動化", "項目", "系統",
                           "setup", "configure", "deploy"]
        if any(k in cmd_lower for k in project_keywords):
            return "project"

        # Default: think
        return "think"

    async def _handle_think_task(self, command: str, context: str) -> str:
        """Let the brain think and return the answer directly."""
        logger.info("Handling as THINK task")
        response = self.agent.brain.think(command, context)
        logger.info(f"Brain response: {response[:200]}")
        return response

    async def _handle_search_task(self, command: str, context: str) -> dict:
        """Perform web search and summarize results."""
        logger.info("Handling as SEARCH task")

        # Extract search query from command
        query_prompt = f"Extract the search query from this command. Reply with ONLY the search query, nothing else:\n{command}"
        search_query = self.agent.brain._think(query_prompt)
        # Clean up - remove quotes and extra text
        search_query = search_query.strip().strip('"').strip("'")
        if len(search_query) > 200:
            search_query = command  # Fallback to original command

        logger.info(f"Search query: {search_query}")

        # Execute web search
        search_result = await self.agent.execute_action("web_search", search_query)

        if search_result.get("success") and search_result.get("results"):
            results = search_result["results"]
            # Format results
            formatted = []
            for i, r in enumerate(results[:5], 1):
                formatted.append(f"{i}. {r.get('title', 'No title')}\n   {r.get('href', '')}\n   {r.get('body', '')[:150]}")
            results_text = "\n\n".join(formatted)

            # Let brain summarize
            summary = self.agent.brain.think(
                f"Based on these search results, answer the original question: {command}\n\nResults:\n{results_text}"
            )

            return {
                "type": "search",
                "query": search_query,
                "results_count": len(results),
                "summary": summary,
                "raw_results": results[:5],
            }
        else:
            # Search failed, let brain answer from knowledge
            fallback = self.agent.brain.think(command, context)
            return {
                "type": "search_fallback",
                "query": search_query,
                "answer": fallback,
                "note": "Web search returned no results, answered from knowledge",
            }

    async def _handle_code_task(self, command: str, context: str) -> dict:
        """Generate code and optionally execute it."""
        logger.info("Handling as CODE task")

        # Ask brain to generate executable code
        code_prompt = f"""Task: {command}

Generate a complete, self-contained Python script to accomplish this task.
The script should:
1. Be executable directly with python3
2. Print results to stdout
3. Handle errors gracefully
4. Install any needed packages with subprocess

Output ONLY the Python code, no explanations."""

        code = self.agent.brain.write_code(code_prompt)

        # Clean up code
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        code = code.strip()

        logger.info(f"Generated code ({len(code)} chars)")

        # Execute the code
        exec_result = await self.agent.execute_action("python", code)

        return {
            "type": "code",
            "code_length": len(code),
            "execution": exec_result,
            "code_preview": code[:500],
        }

    async def _handle_execute_task(self, command: str, context: str) -> dict:
        """Generate and execute shell commands."""
        logger.info("Handling as EXECUTE task")

        # Ask brain for the exact shell command
        cmd_prompt = f"""I need to execute this task on macOS: {command}

Available tools: python3, git, curl, pip3, node, npm, ffmpeg, yt-dlp
Working directory: /Users/yokowai/Projects/naomi-agent

Reply with ONLY the exact shell command(s) to run, one per line. No explanations.
If multiple commands, separate with &&."""

        shell_cmd = self.agent.brain._think(cmd_prompt)

        # Clean up - remove markdown formatting
        if "```" in shell_cmd:
            shell_cmd = shell_cmd.split("```")[1] if "```bash" not in shell_cmd else shell_cmd.split("```bash")[1]
            shell_cmd = shell_cmd.split("```")[0]
        shell_cmd = shell_cmd.strip()

        # Safety check - don't run dangerous commands
        dangerous = ["rm -rf /", "mkfs", "dd if=", "> /dev/sd"]
        if any(d in shell_cmd for d in dangerous):
            return {"type": "execute", "error": "Dangerous command blocked", "command": shell_cmd}

        logger.info(f"Executing: {shell_cmd[:200]}")
        result = await self.agent.execute_action("shell", shell_cmd)

        return {
            "type": "execute",
            "command": shell_cmd,
            "result": result,
        }

    async def _handle_project_task(self, command: str, context: str) -> dict:
        """Handle multi-step project tasks."""
        logger.info("Handling as PROJECT task")

        # Use left brain to create a concrete plan with real actions
        plan_prompt = f"""Task: {command}

Create an actionable plan. For each step, specify:
- action_type: one of [shell, python, web_search, file_write, pip_install, git]
- command: the exact command or code to execute

Respond in JSON:
{{
  "understanding": "...",
  "steps": [
    {{"step": 1, "description": "...", "action_type": "shell", "command": "..."}}
  ]
}}"""

        plan_response = self.agent.brain._think(plan_prompt)

        # Parse plan
        try:
            if "```json" in plan_response:
                plan_response = plan_response.split("```json")[1].split("```")[0]
            elif "```" in plan_response:
                plan_response = plan_response.split("```")[1].split("```")[0]
            plan = json.loads(plan_response.strip())
        except (json.JSONDecodeError, IndexError):
            # Fallback: treat as think task
            return await self._handle_think_task(command, context)

        # Execute each step
        results = []
        for step in plan.get("steps", [])[:10]:  # Max 10 steps
            action_type = step.get("action_type", "shell")
            cmd = step.get("command", "")
            desc = step.get("description", "")

            if action_type not in AVAILABLE_ACTIONS:
                logger.warning(f"Unknown action_type: {action_type}, skipping")
                results.append({"step": step.get("step"), "skipped": True, "reason": f"Unknown action: {action_type}"})
                continue

            logger.info(f"Project step {step.get('step', '?')}: {desc[:100]}")
            result = await self.agent.execute_action(action_type, cmd)
            results.append({"step": step.get("step"), "description": desc, "result": result})

            # If step failed, ask brain to fix
            if result.get("error") and not result.get("success"):
                logger.warning(f"Step failed: {result['error'][:200]}")
                fix = self.agent.brain.debug(result["error"], f"Step: {desc}\nCommand: {cmd}")
                # Try to extract a fix command
                if fix and not fix.startswith("[Brain"):
                    fix_result = await self.agent.execute_action("shell", fix.strip()[:500])
                    results.append({"step": f"{step.get('step')}_fix", "result": fix_result})

        return {
            "type": "project",
            "understanding": plan.get("understanding", ""),
            "total_steps": len(plan.get("steps", [])),
            "results": results,
        }

    async def _handle_error(self, error: str):
        logger.info(f"Auto-handling error: {error[:200]}")
        context = self.agent.memory.build_context()
        fix = self.agent.brain.debug(error, context)
        self.agent.memory.remember_short(f"Auto-fix: {fix[:200]}", category="fix")

    async def _continue_task(self, task: dict):
        await self._execute_command(task["task"])

    async def _idle_think(self):
        logger.info("Entering idle/creative mode...")
        context = self.agent.memory.build_context()
        tasks = self.agent.memory.get_recent_tasks(10)
        history = "\n".join(f"- [{t['status']}] {t['task']}" for t in tasks)

        reflection = self.agent.brain.reflect(f"{context}\n\nRecent history:\n{history}")

        if reflection.get("proactive_tasks"):
            for task in reflection["proactive_tasks"][:1]:
                self.agent.memory.remember_short(f"Suggestion: {task}", category="suggestion")
                logger.info(f"Proactive suggestion: {task}")

        if reflection.get("self_improvements"):
            for imp in reflection["self_improvements"][:1]:
                self.agent.memory.remember_short(f"Self-improvement: {imp}", category="improvement")

    async def _self_check(self):
        logger.info(f"Self-check: beat #{self.beat_count}, state={self.state}")
        self.agent.memory.remember_short(
            f"Self-check OK: {self.beat_count} beats, uptime {time.time() - self.agent.start_time:.0f}s",
            category="system"
        )

        # Trigger evolution cycle every self-check (hourly)
        try:
            logger.info("Triggering auto-evolution cycle...")
            result = self.agent.evolution.evolution_cycle()
            if result.get("bugs_found", 0) > 0:
                self.agent.memory.remember_short(
                    f"Evolution: found {result[bugs_found]} bugs, fixed {result.get(fixes_attempted, 0)}",
                    category="evolution"
                )
                # Notify via Telegram if available
                if hasattr(self.agent, "telegram"):
                    import asyncio
                    await self.agent.telegram.send_message(
                        f"Self-evolution complete: {result[bugs_found]} bugs found, "
                        f"{result.get(fixes_attempted, 0)} fixes attempted"
                    )
        except Exception as e:
            logger.error(f"Evolution cycle error: {e}")

    def stop(self):
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
