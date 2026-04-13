"""
NAOMI Agent - Heartbeat (Main Loop) v3
The never-stopping life cycle of NAOMI.
Sense -> Think -> Act -> Verify -> Remember -> Repeat

v3: Anti-hallucination — every action task MUST produce verifiable evidence.
    Brain is never trusted to self-report completion.
    Result validation enforces real execution vs. narrative fabrication.
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
    "screenshot": "Take a screenshot of the screen",
    "click": "Click at x,y coordinates",
    "type_text": "Type text via keyboard",
    "key_press": "Press a key or key combo",
    "open_app": "Open an application",
    "look_and_act": "Vision-action loop: screenshot → analyze → act",
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

        # Periodic cleanup (every 10 minutes)
        if self.beat_count % 20 == 0:
            self._periodic_cleanup()

        # Check scheduled tasks
        await self._check_scheduled_tasks()

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
        """Smart command execution with anti-hallucination verification."""
        task_id = self.agent.memory.add_task(command)
        self.agent.memory.log_conversation("user", command)

        try:
            context = self.agent.memory.build_context()

            # Step 1: Classify the task
            task_type = self._classify_task(command)
            logger.info(f"Task classified as: {task_type}")

            if task_type == "think":
                result = await self._handle_think_task(command, context)
            elif task_type == "search":
                result = await self._handle_search_task(command, context)
            elif task_type == "code":
                result = await self._handle_code_task(command, context)
            elif task_type == "execute":
                result = await self._handle_execute_task(command, context)
            elif task_type == "project":
                result = await self._handle_project_task(command, context)
            elif task_type == "computer":
                result = await self._handle_computer_task(command, context)
            elif task_type == "action":
                result = await self._handle_action_task(command, context)
            else:
                result = await self._handle_action_task(command, context)

            # Step 2: Validate result — anti-hallucination check
            validated = self._validate_result(task_type, result)

            if validated["honest"]:
                self.agent.memory.complete_task(task_id, str(result)[:2000])
                self.agent.memory.log_conversation("naomi", str(result)[:1000])
                self.agent.memory.remember_long(
                    f"Task: {command[:100]}", str(result)[:1000],
                    category="task_result", importance=7
                )

                # Self-learning: extract skill from completed complex tasks
                if isinstance(result, dict) and hasattr(self.agent, 'skills'):
                    steps = result.get("steps", [])
                    if len(steps) >= 3 and result.get("success"):
                        try:
                            skill_result = self.agent.skills.extract_skill_from_task(
                                command, steps, str(result.get("result", ""))
                            )
                            if skill_result and skill_result.get("success"):
                                logger.info(f"Skill learned: {skill_result['name']}")
                                await self._notify_master(
                                    f"💡 新技能學會: {skill_result['name']}\n"
                                    f"{skill_result.get('description', '')}"
                                )
                        except Exception as e:
                            logger.debug(f"Skill extraction error: {e}")
            else:
                # Result is suspicious — mark as failed, not completed
                logger.warning(f"Result validation FAILED: {validated['reason']}")
                self.agent.memory.complete_task(
                    task_id,
                    f"[UNVERIFIED] {validated['reason']}\nRaw: {str(result)[:1500]}",
                    status="failed"
                )
                self.agent.memory.log_conversation(
                    "naomi",
                    f"[UNVERIFIED — no real action taken] {str(result)[:500]}"
                )

        except Exception as e:
            error_msg = f"Failed: {e}\n{traceback.format_exc()}"
            self.agent.memory.complete_task(task_id, error_msg[:2000], status="failed")
            self.agent.memory.log_conversation("naomi", f"Failed: {command}\nError: {e}")
            logger.error(error_msg)

            # Auto-resolve: discover and install missing capabilities, then retry once
            if hasattr(self.agent, 'discovery'):
                try:
                    resolve = await self.agent.discovery.auto_resolve(command, str(e))
                    if resolve.get("resolved", 0) > 0 and resolve.get("can_retry"):
                        logger.info(f"Auto-resolved {resolve['resolved']} capabilities, retrying task...")
                        self.agent.memory.remember_short(
                            f"Auto-installed {resolve['resolved']} capabilities for: {command[:80]}",
                            category="discovery",
                        )
                        # Retry the task once
                        retry_id = self.agent.memory.add_task(f"[Retry] {command}")
                        try:
                            retry_result = await self._handle_think_task(command, self.agent.memory.build_context())
                            self.agent.memory.complete_task(retry_id, str(retry_result)[:2000])
                        except Exception as retry_e:
                            self.agent.memory.complete_task(retry_id, str(retry_e)[:2000], status="failed")
                except Exception as disc_e:
                    logger.warning(f"Discovery auto-resolve error: {disc_e}")

    def _classify_task(self, command: str) -> str:
        """Classify task: does this need REAL actions or is it a pure question?"""
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

        # Computer control tasks (GUI, browser, screen)
        computer_keywords = [
            "click", "screenshot", "screen", "browser", "safari", "chrome",
            "window", "scroll", "type in", "type into",
            "打開瀏覽器", "點擊", "截圖", "螢幕", "視窗", "滾動", "輸入",
            "open safari", "open chrome", "open browser", "open app",
            "go to website", "navigate to", "visit",
        ]
        if any(k in cmd_lower for k in computer_keywords):
            return "computer"

        # Action-required tasks — anything that implies doing something real
        action_keywords = [
            "check", "verify", "look at", "download", "upload", "send", "post",
            "save", "write", "read", "open", "close", "start", "stop", "monitor",
            "test", "debug", "fix", "modify", "change", "move", "copy",
            "幫我", "幫忙", "做", "改", "查", "看看", "檢查", "確認", "下載",
            "上傳", "寄", "發", "存", "寫", "讀", "打開", "關閉", "啟動",
            "測試", "修", "修改", "移動", "複製", "監控",
        ]
        if any(k in cmd_lower for k in action_keywords):
            return "action"  # New type: needs real execution, not just brain text

        # Default: think (pure question / conversation)
        return "think"

    async def _handle_think_task(self, command: str, context: str) -> str:
        """Pure Q&A — brain answers a question. No action taken."""
        logger.info("Handling as THINK task (pure Q&A, no action)")
        response = self.agent.brain.think(command, context)
        logger.info(f"Brain response: {response[:200]}")
        return response

    async def _handle_action_task(self, command: str, context: str) -> dict:
        """
        Action task — uses Anthropic native tool_use agent loop.
        Claude decides which tools to call, executor runs them, results fed back.
        Anti-hallucination: every step is a real tool execution with real output.
        """
        logger.info("Handling as ACTION task (native tool_use agent loop)")

        # Inject relevant skills into context
        skill_context = ""
        if hasattr(self.agent, 'skills'):
            skill_context = self.agent.skills.get_skill_context(command)

        system = (
            "You are NAOMI, an autonomous AI agent on macOS. "
            f"Context:\n{context[:1500]}\n\n"
            f"{skill_context}\n\n" if skill_context else
            "You are NAOMI, an autonomous AI agent on macOS. "
            f"Context:\n{context[:1500]}\n\n"
        ) + "Use tools to complete the task. Do NOT describe — execute. " \
            "When finished, call the task_complete tool with a summary."

        result = await self.agent.brain.agent_loop(
            task=command,
            executor=self.agent.actions,
            system_prompt=system,
            max_iterations=15,
        )

        result["type"] = "action"
        return result

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

        # Full permissions granted by Master — no command restrictions
        logger.info(f"Executing: {shell_cmd[:200]}")
        result = await self.agent.execute_action("shell", shell_cmd)

        return {
            "type": "execute",
            "command": shell_cmd,
            "result": result,
        }

    async def _handle_project_task(self, command: str, context: str) -> dict:
        """Handle multi-step project tasks via agent loop."""
        logger.info("Handling as PROJECT task (agent loop)")

        system = (
            "You are NAOMI, an autonomous AI agent on macOS. "
            f"Context:\n{context[:1500]}\n\n"
            "This is a multi-step project. Break it down and execute step by step. "
            "Use tools to do real work. Call task_complete when finished."
        )

        result = await self.agent.brain.agent_loop(
            task=command,
            executor=self.agent.actions,
            system_prompt=system,
            max_iterations=15,
        )

        result["type"] = "project"
        return result

    async def _handle_computer_task(self, command: str, context: str) -> dict:
        """Handle GUI/screen control tasks via agent loop with screenshot tools."""
        logger.info("Handling as COMPUTER task (agent loop with GUI tools)")

        system = (
            "You are NAOMI, controlling a macOS computer. "
            "Use screenshot, click, type_text, key_press, open_app tools to interact with the GUI. "
            "Workflow: screenshot first → analyze what you see → click/type → screenshot to verify. "
            "When the task is done, call task_complete."
        )

        result = await self.agent.brain.agent_loop(
            task=command,
            executor=self.agent.actions,
            system_prompt=system,
            max_iterations=15,
        )

        result["type"] = "computer"
        return result

    def _validate_result(self, task_type: str, result) -> dict:
        """
        Anti-hallucination: verify that tasks which require real actions
        actually produced real execution evidence, not just brain-generated text.
        """
        # Pure think tasks are always honest (they're just Q&A)
        if task_type == "think":
            return {"honest": True, "reason": "Q&A task, no action required"}

        # String results from action-type tasks are suspicious
        # (brain just generated text without executing anything)
        if isinstance(result, str):
            # Check for hallucination signals in text
            hallucination_signals = [
                "I would", "I will", "I can", "let me", "here's what",
                "I'll", "I'd suggest", "you can", "you should",
                "我會", "我可以", "讓我", "建議你", "你可以",
            ]
            result_lower = result.lower()
            if any(sig in result_lower for sig in hallucination_signals):
                return {
                    "honest": False,
                    "reason": "Brain produced narrative text instead of executing real actions",
                }
            # Short string responses from action tasks are suspicious
            if task_type in ("action", "execute", "code", "project") and len(result) < 200:
                return {
                    "honest": False,
                    "reason": "Action task returned suspiciously short text response",
                }

        # Dict results: check for real execution evidence
        if isinstance(result, dict):
            rtype = result.get("type", "")

            # Action tasks must have executed steps
            if rtype == "action":
                steps = result.get("steps", [])
                executed = len([s for s in steps if s.get("tool")])
                if executed == 0:
                    return {"honest": False, "reason": "Action plan had 0 executed steps"}
                return {"honest": True, "reason": f"Executed {executed} real actions"}

            if rtype == "action_failed":
                return {"honest": True, "reason": "Honestly reported action failure"}

            if rtype == "think_downgrade":
                return {"honest": True, "reason": "Brain correctly identified as Q&A"}

            # Search must have real results or honest fallback note
            if rtype == "search":
                if result.get("results_count", 0) > 0:
                    return {"honest": True, "reason": "Search returned real results"}
                return {"honest": False, "reason": "Search claimed success but no results"}

            if rtype == "search_fallback":
                return {"honest": True, "reason": "Honestly reported search failure, used knowledge"}

            # Code tasks must have execution output
            if rtype == "code":
                exec_data = result.get("execution", {})
                if isinstance(exec_data, dict) and "output" in exec_data:
                    return {"honest": True, "reason": "Code was executed with real output"}
                return {"honest": False, "reason": "Code task has no execution evidence"}

            # Execute tasks must have shell output
            if rtype == "execute":
                exec_data = result.get("result", {})
                if isinstance(exec_data, dict) and ("output" in exec_data or "returncode" in exec_data):
                    return {"honest": True, "reason": "Shell command was really executed"}
                return {"honest": False, "reason": "Execute task has no shell output"}

            # Project tasks must have step results
            if rtype == "project":
                step_results = result.get("results", [])
                if step_results:
                    return {"honest": True, "reason": f"Project executed {len(step_results)} steps"}
                return {"honest": False, "reason": "Project plan had no executed steps"}

            # Computer tasks must have look_and_act steps
            if rtype == "computer":
                steps = result.get("steps_taken", 0)
                if steps > 0:
                    return {"honest": True, "reason": f"Computer control executed {steps} vision-action steps"}
                return {"honest": False, "reason": "Computer task had no executed steps"}

        # Default: pass through (benefit of the doubt for unknown types)
        return {"honest": True, "reason": "Unclassified result type"}

    def _periodic_cleanup(self):
        """Periodic maintenance: screenshot cleanup, memory consolidation."""
        # Cleanup old screenshots
        try:
            computer = self.agent.actions._get_computer()
            result = computer.cleanup_screenshots(keep_latest=20)
            if result.get("deleted", 0) > 0:
                logger.info(f"Cleaned up {result['deleted']} old screenshots")
        except Exception as e:
            logger.debug(f"Screenshot cleanup: {e}")

        # Memory consolidation
        try:
            self.agent.memory.consolidate()
        except Exception as e:
            logger.debug(f"Memory consolidation: {e}")

    async def _check_scheduled_tasks(self):
        """Check and execute due scheduled tasks."""
        if not hasattr(self.agent, 'scheduler'):
            return
        try:
            due_jobs = self.agent.scheduler.get_due_jobs()
            for job in due_jobs:
                job_id = job["id"]
                command = job["command"]
                logger.info(f"Scheduled job due: [{job_id}] {command[:80]}")
                await self._notify_master(f"⏰ 排程執行: {job['name']}\n{command[:100]}")
                await self._execute_command(command)
                self.agent.scheduler.mark_completed(job_id)
        except Exception as e:
            logger.debug(f"Scheduled tasks check: {e}")

    async def _notify_master(self, message: str):
        """Send proactive notification to Master via Telegram."""
        if hasattr(self.agent, 'telegram'):
            try:
                await self.agent.telegram.send_message(message)
            except Exception as e:
                logger.error(f"Failed to notify master: {e}")

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

        # Idle capability discovery — check if recent failures need new tools
        if hasattr(self.agent, 'discovery'):
            try:
                disc_result = self.agent.discovery.idle_discover()
                if disc_result.get("action") == "installed":
                    logger.info(f"Idle discovery installed: {disc_result.get('details', [])}")
                elif disc_result.get("action") == "suggested":
                    logger.info(f"Idle discovery suggested: {disc_result.get('suggestions', {})}")
            except Exception as e:
                logger.warning(f"Idle discovery error: {e}")

    async def _self_check(self):
        uptime = time.time() - self.agent.start_time
        logger.info(f"Self-check: beat #{self.beat_count}, uptime {uptime:.0f}s")
        self.agent.memory.remember_short(
            f"Self-check OK: {self.beat_count} beats, uptime {uptime:.0f}s",
            category="system"
        )

        # Check for accumulated errors and alert Master
        recent_errors = self.agent.memory.recall_short(category="error", limit=10)
        if len(recent_errors) >= 5:
            await self._notify_master(
                f"⚠️ 系統警告：最近有 {len(recent_errors)} 個錯誤\n"
                + "\n".join(f"- {e['content'][:60]}" for e in recent_errors[:3])
            )

        # Trigger evolution cycle in background thread (doesn't block event loop)
        import threading
        def _run_evolution():
            try:
                logger.info("Triggering auto-evolution cycle (background)...")
                result = self.agent.evolution.evolution_cycle()
                if result.get("bugs_found", 0) > 0:
                    bugs = result.get("bugs_found", 0)
                    fixes = result.get("fixes_attempted", 0)
                    self.agent.memory.remember_short(
                        f"Evolution: found {bugs} bugs, fixed {fixes}",
                        category="evolution"
                    )
                    # Schedule notification (can't await from thread)
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self._notify_master(f"🔧 Self-evolution: {bugs} bugs, {fixes} fixes"),
                            loop,
                        )
            except Exception as e:
                logger.error(f"Evolution cycle error: {e}")

        threading.Thread(target=_run_evolution, daemon=True).start()

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
