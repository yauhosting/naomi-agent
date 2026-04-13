"""
NAOMI Agent - Telegram Bot Interface
Control NAOMI from your phone via Telegram.
Only responds to authorized user (Master).
"""
import asyncio
import json
import time
import os
import logging
import httpx
from typing import Optional

logger = logging.getLogger("naomi.telegram")


class TelegramBot:
    def __init__(self, agent, token: str, master_id: int):
        self.agent = agent
        self.token = token
        self.master_id = master_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.last_update_id = 0
        self.running = False
        self.client = httpx.AsyncClient(timeout=30)

    async def start(self):
        """Start polling for Telegram messages."""
        self.running = True
        logger.info(f"Telegram bot started, master ID: {self.master_id}")

        # Send startup message
        await self.send_message("NAOMI Agent is online and ready.")

        while self.running:
            try:
                await self._poll()
            except Exception as e:
                logger.error(f"Telegram poll error: {e}")
                await asyncio.sleep(5)
            await asyncio.sleep(1)

    async def _poll(self):
        """Poll for new messages."""
        try:
            resp = await self.client.get(
                f"{self.base_url}/getUpdates",
                params={"offset": self.last_update_id + 1, "timeout": 20},
                timeout=25,
            )
            data = resp.json()
            if not data.get("ok"):
                return

            for update in data.get("result", []):
                self.last_update_id = update["update_id"]
                await self._handle_update(update)
        except httpx.TimeoutException:
            pass  # Normal for long polling

    async def _handle_update(self, update: dict):
        """Handle a single Telegram update."""
        message = update.get("message")
        if not message:
            return

        chat_id = message["chat"]["id"]
        user_id = message["from"]["id"]
        text = message.get("text", "")

        # Security: only respond to master
        if user_id != self.master_id:
            await self._send(chat_id, "Access denied. Only Master can control NAOMI.")
            logger.warning(f"Unauthorized access attempt from user {user_id}")
            return

        # Handle voice messages — transcribe to text
        voice = message.get("voice") or message.get("audio")
        if voice and not text:
            text = await self._transcribe_voice(chat_id, voice)
            if not text:
                return  # Transcription failed, error already sent

        if not text:
            return

        # Sanitize input
        from core.security import sanitize_telegram_input
        text = sanitize_telegram_input(text)

        # Extract reply context — if user replied to a previous message, include it
        reply_context = ""
        reply_to = message.get("reply_to_message")
        if reply_to:
            reply_text = reply_to.get("text", "")
            reply_from = reply_to.get("from", {}).get("first_name", "")
            if reply_text:
                reply_context = f"[Replying to {reply_from}: {reply_text[:500]}]\n\n"
                text = reply_context + text
                logger.info(f"Reply context included ({len(reply_text)} chars)")

        logger.info(f"Master command: {text[:100]}")

        # Handle commands
        if text.startswith("/"):
            await self._handle_command(chat_id, text)
        else:
            # Detect: chat or task?
            await self._handle_message(chat_id, text)

    async def _handle_command(self, chat_id: int, text: str):
        """Handle slash commands."""
        cmd = text.split()[0].lower().split("@")[0]  # Remove @botname
        args = text[len(cmd):].strip()

        if cmd == "/start":
            await self._send(chat_id,
                "NAOMI Agent - Telegram Control\n\n"
                "Commands:\n"
                "/status - Agent status\n"
                "/model - View/switch brain model\n"
                "/discover - Discover/install capabilities\n"
                "/tasks - Recent tasks\n"
                "/memory - Memory stats\n"
                "/skills - Learned skills\n"
                "/think <topic> - Let NAOMI think\n"
                "/search <query> - Web search\n"
                "/council <topic> - Multi-agent debate\n"
                "/evolve - Trigger self-evolution\n"
                "/shell <cmd> - Execute shell command\n"
                "/schedule - Manage scheduled tasks\n"
                "/project <goal> - Create & run a full project\n"
                "/usage - Usage statistics\n"
                "/screen - Take screenshot\n"
                "/click x y - Click at coordinates\n"
                "/type <text> - Type text\n"
                "/key <key> - Press key (return, cmd+c...)\n"
                "/app [name] - Open app / list windows\n"
                "/ollama - Manage local Ollama models\n"
                "/security - Run security scan\n"
                "/audit - View audit log\n"
                "/log - Recent activity log\n\n"
                "Or just type anything to give NAOMI a task."
            )

        elif cmd == "/status":
            status = {
                "state": self.agent.heartbeat.state,
                "uptime": int(time.time() - self.agent.start_time),
                "beats": self.agent.heartbeat.beat_count,
                "tools": sum(1 for v in self.agent.tool_manager.list_tools().values() if v),
            }
            h = status["uptime"] // 3600
            m = (status["uptime"] % 3600) // 60
            await self._send(chat_id,
                f"NAOMI Status\n"
                f"State: {status['state']}\n"
                f"Uptime: {h}h {m}m\n"
                f"Heartbeats: {status['beats']}\n"
                f"Tools: {status['tools']} available"
            )

        elif cmd == "/model":
            if not args:
                # Show current model and list
                current = self.agent.brain.get_model()
                models = self.agent.brain.list_models()
                lines = []
                for m in models:
                    marker = " <<" if m["active"] else ""
                    lines.append(f"  {m['name']} — {m['description']}{marker}")
                await self._send(chat_id,
                    f"Current model: {current['name']}\n"
                    f"Backend: {current['backend']}\n"
                    f"Model ID: {current['model_id']}\n\n"
                    f"Available models:\n" + "\n".join(lines) + "\n\n"
                    f"Usage: /model <name>\n"
                    f"Example: /model claude-cli"
                )
            else:
                result = self.agent.brain.set_model(args)
                if result.get("success"):
                    await self._send(chat_id,
                        f"Model switched!\n"
                        f"{result['previous']} -> {result['model']}\n"
                        f"{result['description']}"
                    )
                else:
                    available = result.get("available", [])
                    await self._send(chat_id,
                        f"Failed: {result.get('error', 'Unknown error')}\n\n"
                        f"Available: {', '.join(available)}" if available else
                        f"Failed: {result.get('error', 'Unknown error')}"
                    )

        elif cmd == "/discover":
            if not args:
                # Show discovery status
                if hasattr(self.agent, 'discovery'):
                    status = self.agent.discovery.get_status()
                    mcp_list = ", ".join(status["installed_mcp"]) if status["installed_mcp"] else "none"
                    known_list = ", ".join(status["known_mcp"])
                    cat_list = ", ".join(status["available_categories"])
                    await self._send(chat_id,
                        f"Capability Discovery\n\n"
                        f"Installed MCP: {mcp_list}\n"
                        f"Known MCP: {known_list}\n"
                        f"Package categories: {cat_list}\n"
                        f"Learned skills: {status['learned_skills']}\n\n"
                        f"Usage:\n"
                        f"/discover install <mcp-name> — Install MCP server\n"
                        f"/discover pkg <package> — Install Python package\n"
                        f"/discover scan — Scan for missing capabilities\n"
                        f"/discover tool <name> — Install system tool\n"
                    f"/discover app <name> — Install macOS app (godot, blender...)"
                    )
                else:
                    await self._send(chat_id, "Discovery engine not initialized.")
            else:
                parts = args.split(None, 1)
                sub_cmd = parts[0].lower()
                sub_args = parts[1] if len(parts) > 1 else ""

                if sub_cmd == "install" and sub_args:
                    await self._send(chat_id, f"Installing MCP: {sub_args}...")
                    await self._send_typing(chat_id)
                    result = self.agent.discovery.install_mcp(sub_args)
                    if result.get("success"):
                        await self._send(chat_id, f"MCP '{sub_args}' installed!\n{result.get('description', '')}")
                    else:
                        await self._send(chat_id, f"Failed: {result.get('error', 'Unknown')}")

                elif sub_cmd == "pkg" and sub_args:
                    await self._send(chat_id, f"Installing package: {sub_args}...")
                    result = self.agent.discovery.install_package(sub_args)
                    status_text = "installed" if result.get("success") else f"failed: {result.get('error','')[:200]}"
                    await self._send(chat_id, f"Package {sub_args}: {status_text}")

                elif sub_cmd == "tool" and sub_args:
                    await self._send(chat_id, f"Installing tool: {sub_args}...")
                    result = self.agent.discovery.install_tool(sub_args)
                    status_text = "installed" if result.get("success") else f"failed: {result.get('error','')[:200]}"
                    await self._send(chat_id, f"Tool {sub_args}: {status_text}")

                elif sub_cmd == "app" and sub_args:
                    await self._send(chat_id, f"Installing app: {sub_args}...")
                    await self._send_typing(chat_id)
                    result = self.agent.discovery.install_app(sub_args)
                    if result.get("success"):
                        await self._send(chat_id, f"App '{sub_args}' installed!\n{result.get('description', '')}")
                    else:
                        await self._send(chat_id, f"Failed: {result.get('error', result.get('output','')[:200])}")

                elif sub_cmd == "scan":
                    await self._send(chat_id, "Scanning for missing capabilities...")
                    await self._send_typing(chat_id)
                    result = self.agent.discovery.idle_discover()
                    action = result.get("action", "none")
                    if action == "installed":
                        details = result.get("details", [])
                        await self._send(chat_id, f"Auto-installed {len(details)} capabilities!")
                    elif action == "suggested":
                        suggestions = result.get("suggestions", {})
                        lines = []
                        for k, v in suggestions.items():
                            if v and k != "priority":
                                lines.append(f"{k}: {', '.join(v) if isinstance(v, list) else v}")
                        await self._send(chat_id, f"Suggestions ({suggestions.get('priority','?')}):\n" + "\n".join(lines))
                    else:
                        await self._send(chat_id, f"No new capabilities needed. ({result.get('reason', '')})")
                else:
                    await self._send(chat_id, "Usage: /discover [install|pkg|tool|scan] <name>")

        elif cmd == "/tasks":
            tasks = self.agent.memory.get_recent_tasks(5)
            if not tasks:
                await self._send(chat_id, "No recent tasks.")
                return
            lines = []
            for t in tasks:
                status_icon = {"completed": "done", "running": "...", "failed": "FAIL", "pending": "wait"}.get(t["status"], "?")
                lines.append(f"[{status_icon}] {t['task'][:60]}")
            await self._send(chat_id, "Recent Tasks:\n" + "\n".join(lines))

        elif cmd == "/memory":
            short = len(self.agent.memory.recall_short(limit=100))
            long_mem = len(self.agent.memory.recall_long(limit=100))
            skills = self.agent.memory.recall_skill()
            skill_count = len(skills) if isinstance(skills, list) else 0
            await self._send(chat_id,
                f"Memory Stats:\n"
                f"Short-term: {short}\n"
                f"Long-term: {long_mem}\n"
                f"Skills: {skill_count}"
            )

        elif cmd == "/skills":
            lines = []
            # Learned skills (SKILL.md files)
            if hasattr(self.agent, 'skills'):
                learned = self.agent.skills.list_skills()
                if learned:
                    lines.append("📚 Learned Skills:")
                    for s in learned:
                        lines.append(f"  {s['name']}: {s.get('description','')[:50]}")
            # Installed packages (DB)
            db_skills = self.agent.memory.recall_skill()
            if db_skills and isinstance(db_skills, list) and len(db_skills) > 0:
                lines.append("\n📦 Installed Packages:")
                for s in db_skills[:15]:
                    lines.append(f"  {s['name']}: {s.get('description','')[:50]}")
            if not lines:
                await self._send(chat_id, "No skills learned yet.")
                return
            await self._send(chat_id, "\n".join(lines))

        elif cmd == "/think":
            if not args:
                await self._send(chat_id, "Usage: /think <topic>")
                return
            await self._send(chat_id, "Thinking...")
            await self._send_typing(chat_id)
            response = self.agent.brain.think(args)
            await self._send(chat_id, f"NAOMI thinks:\n\n{response[:3500]}")

        elif cmd == "/search":
            if not args:
                await self._send(chat_id, "Usage: /search <query>")
                return
            await self._send(chat_id, f"Searching: {args}")
            await self._send_typing(chat_id)
            result = await self.agent.execute_action("web_search", args)
            if result.get("success") and result.get("results"):
                lines = []
                for i, r in enumerate(result["results"][:5], 1):
                    lines.append(f"{i}. {r.get('title','?')}\n   {r.get('href','')}")
                await self._send(chat_id, "Search Results:\n\n" + "\n\n".join(lines))
            else:
                await self._send(chat_id, "No results found.")

        elif cmd == "/council":
            if not args:
                await self._send(chat_id, "Usage: /council <topic>")
                return
            await self._send(chat_id, "Council is debating...")
            await self._send_typing(chat_id)
            result = self.agent.council.debate(args)
            consensus = result.get("consensus", "No consensus")
            steps = result.get("action_steps", [])
            msg = f"Council Decision:\n\n{consensus[:2000]}"
            if steps:
                msg += "\n\nAction Steps:\n" + "\n".join(f"- {s}" for s in steps[:5])
            await self._send(chat_id, msg)

        elif cmd == "/evolve":
            await self._send(chat_id, "Triggering evolution cycle...")
            result = self.agent.evolution.evolution_cycle()
            await self._send(chat_id, f"Evolution complete:\n{json.dumps(result, indent=2, default=str)[:3000]}")

        elif cmd == "/shell":
            if not args:
                await self._send(chat_id, "Usage: /shell <command>")
                return
            # Full permissions — Master trusts NAOMI
            await self._send_typing(chat_id)
            result = await self.agent.execute_action("shell", args)
            output = result.get("output", result.get("error", "No output"))
            await self._send(chat_id, f"$ {args}\n\n{output[:3500]}")

        elif cmd == "/project":
            if not args:
                # List projects
                if hasattr(self.agent, 'project'):
                    projects = self.agent.project.list_projects()
                    if not projects:
                        await self._send(chat_id,
                            "沒有進行中的項目\n\n"
                            "Usage: /project <goal>\n"
                            "Example: /project 做一個卡牌遊戲\n"
                            "Example: /project Build a weather dashboard web app"
                        )
                    else:
                        lines = [f"[{p['status']}] {p['name']} ({p['progress']}) — {p['id']}" for p in projects]
                        await self._send(chat_id,
                            "Projects:\n" + "\n".join(lines) + "\n\n"
                            "/project <goal> — Create new\n"
                            "/project run <id> — Resume\n"
                            "/project status <id> — Details"
                        )
                return

            parts = args.split(None, 1)
            sub = parts[0].lower() if parts else ""

            if sub == "run" and len(parts) > 1:
                pid = parts[1].strip()
                await self._send(chat_id, f"▶️ Resuming project: {pid}")
                result = await self.agent.project.run_all(
                    pid, notify_callback=lambda msg: self._send(chat_id, msg)
                )
                status = "✅ 完成" if result.get("project_status") == "completed" else "⏸ 進行中"
                await self._send(chat_id, f"{status} ({result.get('phases_executed', 0)} phases executed)")

            elif sub == "status" and len(parts) > 1:
                pid = parts[1].strip()
                p = self.agent.project.get_project(pid)
                if not p:
                    await self._send(chat_id, f"Project not found: {pid}")
                else:
                    lines = []
                    for ph in p["phases"]:
                        icon = {"completed": "✅", "running": "🔄", "pending": "⬜", "needs_review": "⚠️"}.get(ph["status"], "?")
                        lines.append(f"{icon} Phase {ph['id']}: {ph['name']}")
                    await self._send(chat_id,
                        f"📋 {p['name']}\n"
                        f"Status: {p['status']}\n"
                        f"Dir: {p['work_dir']}\n\n" + "\n".join(lines)
                    )

            elif sub == "rm" and len(parts) > 1:
                self.agent.project.delete_project(parts[1].strip())
                await self._send(chat_id, "已刪除")

            else:
                # Create + run new project
                goal = args
                await self._send(chat_id, f"🚀 Creating project: {goal}")
                await self._send_typing(chat_id)

                create_result = await self.agent.project.create(goal)
                if not create_result.get("success"):
                    await self._send(chat_id, f"❌ 創建失敗: {create_result.get('error', '?')}")
                    return

                pid = create_result["project_id"]
                phase_list = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(create_result["phase_names"]))
                await self._send(chat_id,
                    f"📋 Project: {create_result['name']}\n"
                    f"ID: {pid}\n"
                    f"Phases ({create_result['phases']}):\n{phase_list}\n\n"
                    f"🔨 開始執行..."
                )

                # Run all phases
                result = await self.agent.project.run_all(
                    pid, notify_callback=lambda msg: self._send(chat_id, msg)
                )

                status = "✅ 項目完成！" if result.get("project_status") == "completed" else "⏸ 部分完成"
                await self._send(chat_id,
                    f"{status}\n"
                    f"Phases: {result.get('phases_executed', 0)} executed\n"
                    f"Dir: {create_result['work_dir']}\n\n"
                    f"用 /project status {pid} 查看詳情"
                )

        elif cmd == "/usage":
            usage = self.agent.brain.get_usage()
            backend_lines = "\n".join(
                f"  {k}: {v} calls" for k, v in usage["by_backend"].items()
            ) if usage["by_backend"] else "  (no calls yet)"
            health_lines = "\n".join(
                f"  {k}: {'⚠️ backoff' if v['in_backoff'] else '✓ OK'} ({v['failures']} failures)"
                for k, v in usage["backend_health"].items() if v["failures"] > 0
            )
            await self._send(chat_id,
                f"Usage Stats\n"
                f"Total calls: {usage['total_calls']}\n"
                f"Errors: {usage['errors']} ({usage['error_rate']}%)\n"
                f"Uptime: {usage['uptime_hours']}h\n"
                f"Rate: {usage['calls_per_hour']} calls/hr\n\n"
                f"By backend:\n{backend_lines}"
                + (f"\n\nHealth:\n{health_lines}" if health_lines else "")
            )

        elif cmd == "/ollama":
            if not args:
                # List models
                try:
                    resp = await self.client.get("http://127.0.0.1:11434/api/tags", timeout=5)
                    if resp.status_code == 200:
                        models = resp.json().get("models", [])
                        lines = ["🦙 Ollama Models:"]
                        for m in models:
                            size_gb = m.get("size", 0) / (1024**3)
                            lines.append(f"  {m['name']} ({size_gb:.1f}GB)")
                        lines.append(f"\n/ollama pull <model> — Download new model")
                        lines.append(f"/model ollama — Switch to Ollama")
                        await self._send(chat_id, "\n".join(lines))
                    else:
                        await self._send(chat_id, "Ollama not running. Start with: ollama serve")
                except Exception:
                    await self._send(chat_id, "Ollama not reachable at localhost:11434")
            else:
                parts = args.split(None, 1)
                sub = parts[0].lower()
                if sub == "pull" and len(parts) > 1:
                    model_name = parts[1].strip()
                    await self._send(chat_id, f"⬇️ Pulling {model_name}... (this may take a while)")
                    result = await self.agent.execute_action("shell", f"ollama pull {model_name}")
                    output = result.get("output", result.get("error", ""))[:500]
                    await self._send(chat_id, f"Ollama pull:\n{output}")
                elif sub == "rm" and len(parts) > 1:
                    result = await self.agent.execute_action("shell", f"ollama rm {parts[1].strip()}")
                    await self._send(chat_id, f"{'Removed' if result.get('success') else 'Failed'}")
                else:
                    await self._send(chat_id, "Usage: /ollama [pull|rm] <model>")

        elif cmd == "/security":
            from core.security import run_security_scan, get_recent_audit
            import os as _os
            project_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            scan = run_security_scan(project_dir)
            lines = [f"🔒 Security Scan ({scan['timestamp'][:16]})"]
            lines.append(f"Issues: {scan['total_issues']} (CRIT:{scan['critical']} HIGH:{scan['high']})")
            for issue in scan["issues"][:10]:
                lines.append(f"  [{issue['severity']}] {issue['type']}: {issue.get('detail', issue.get('file', ''))[:60]}")
            if not scan["issues"]:
                lines.append("  ✅ No issues found")
            await self._send(chat_id, "\n".join(lines))

        elif cmd == "/audit":
            from core.security import get_recent_audit
            entries = get_recent_audit(limit=15)
            if not entries:
                await self._send(chat_id, "No audit log entries yet.")
            else:
                lines = ["📋 Recent Audit Log:"]
                for e in entries[-15:]:
                    ts = e.get("timestamp", "")[-8:]  # HH:MM:SS
                    tool = e.get("tool", "?")
                    params = e.get("params", "")[:40]
                    ok = "✓" if e.get("success") else "✗"
                    lines.append(f"  {ts} {ok} [{tool}] {params}")
                await self._send(chat_id, "\n".join(lines))

        elif cmd == "/log":
            entries = self.agent.memory.recall_short(limit=10)
            if not entries:
                await self._send(chat_id, "No recent log entries.")
                return
            lines = [f"[{e['category']}] {e['content'][:80]}" for e in entries]
            await self._send(chat_id, "Recent Log:\n" + "\n".join(lines))

        elif cmd == "/schedule":
            if not args:
                # List scheduled jobs
                if hasattr(self.agent, 'scheduler'):
                    jobs = self.agent.scheduler.list_jobs()
                    if not jobs:
                        await self._send(chat_id, "沒有排程任務\n\nUsage:\n/schedule <min> <task> — 一次性\n/schedule every <min> <task> — 重複\n/schedule list — 查看\n/schedule rm <id> — 刪除")
                    else:
                        lines = [f"[{j['status']}] {j['id']}: {j['command']} (next: {j['next_run']}, {j['interval']})" for j in jobs]
                        await self._send(chat_id, "排程任務:\n" + "\n".join(lines))
                return
            parts = args.split(None, 1)
            sub = parts[0].lower()

            if sub == "list":
                jobs = self.agent.scheduler.list_jobs()
                if not jobs:
                    await self._send(chat_id, "沒有排程任務")
                else:
                    lines = [f"[{j['status']}] {j['id']}: {j['command']} ({j['interval']}, runs:{j['runs']})" for j in jobs]
                    await self._send(chat_id, "排程:\n" + "\n".join(lines))

            elif sub == "rm" and len(parts) > 1:
                result = self.agent.scheduler.remove(parts[1])
                await self._send(chat_id, f"{'已刪除' if result['success'] else result.get('error','?')}")

            elif sub == "pause" and len(parts) > 1:
                result = self.agent.scheduler.pause(parts[1])
                await self._send(chat_id, f"{'已暫停' if result['success'] else result.get('error','?')}")

            elif sub == "every":
                # Recurring: /schedule every 60 check disk
                rest = parts[1] if len(parts) > 1 else ""
                rparts = rest.split(None, 1)
                if len(rparts) < 2:
                    await self._send(chat_id, "Usage: /schedule every <minutes> <task>")
                    return
                try:
                    minutes = int(rparts[0])
                    task_text = rparts[1]
                except ValueError:
                    await self._send(chat_id, "Usage: /schedule every <minutes> <task>")
                    return
                result = self.agent.scheduler.add(
                    name=task_text[:40], command=task_text,
                    interval_minutes=minutes, repeat=-1,
                )
                if result.get("success"):
                    await self._send(chat_id, f"已排程: 每 {minutes} 分鐘執行\n{task_text}")
                else:
                    await self._send(chat_id, f"排程失敗: {result.get('error','?')}")

            else:
                # One-shot: /schedule 30 check disk
                if len(parts) < 2:
                    await self._send(chat_id, "Usage: /schedule <minutes> <task>")
                    return
                try:
                    minutes = int(parts[0])
                    task_text = parts[1]
                except ValueError:
                    await self._send(chat_id, "Usage: /schedule <minutes> <task>")
                    return
                run_at = time.time() + minutes * 60
                result = self.agent.scheduler.add(
                    name=task_text[:40], command=task_text, run_at=run_at,
                )
                if result.get("success"):
                    run_time = time.strftime("%H:%M", time.localtime(run_at))
                    await self._send(chat_id, f"已排程: {minutes} 分鐘後 ({run_time}) 執行\n{task_text}")
                else:
                    await self._send(chat_id, f"排程失敗: {result.get('error','?')}")

        elif cmd == "/screen":
            await self._send(chat_id, "Taking screenshot...")
            computer = self.agent.actions._get_computer()
            result = computer.screenshot()
            if result.get("success"):
                # Send screenshot as photo to Telegram
                await self._send_photo(chat_id, result["path"])
            else:
                await self._send(chat_id, f"Screenshot failed: {result.get('error', '?')}")

        elif cmd == "/click":
            if not args:
                await self._send(chat_id, "Usage: /click x y")
                return
            parts = args.replace(",", " ").split()
            if len(parts) < 2:
                await self._send(chat_id, "Usage: /click x y")
                return
            computer = self.agent.actions._get_computer()
            result = computer.click(int(parts[0]), int(parts[1]))
            await self._send(chat_id, f"Clicked ({parts[0]}, {parts[1]}): {'OK' if result.get('success') else result.get('error','failed')}")

        elif cmd == "/type":
            if not args:
                await self._send(chat_id, "Usage: /type <text>")
                return
            computer = self.agent.actions._get_computer()
            result = computer.type_text(args)
            await self._send(chat_id, f"Typed: {'OK' if result.get('success') else result.get('error','failed')}")

        elif cmd == "/key":
            if not args:
                await self._send(chat_id, "Usage: /key <key> (e.g. return, cmd+c, tab)")
                return
            computer = self.agent.actions._get_computer()
            result = computer.key(args)
            await self._send(chat_id, f"Key '{args}': {'OK' if result.get('success') else result.get('error','failed')}")

        elif cmd == "/app":
            if not args:
                # List windows
                computer = self.agent.actions._get_computer()
                result = computer.get_windows()
                if result.get("success"):
                    windows = result.get("windows", [])
                    await self._send(chat_id, f"Open windows ({len(windows)}):\n" + "\n".join(windows[:20]))
                else:
                    await self._send(chat_id, f"Failed: {result.get('error','?')}")
            else:
                computer = self.agent.actions._get_computer()
                result = computer.open_app(args)
                await self._send(chat_id, f"Opening {args}: {'OK' if result.get('success') else result.get('error','failed')}")

        else:
            await self._send(chat_id, f"Unknown command: {cmd}\nType /start for help.")

    async def _handle_message(self, chat_id: int, text: str):
        """Detect if message is chat or task, respond accordingly."""
        classify_prompt = (
            "Classify this message as CHAT or TASK. "
            "CHAT = greeting, question, conversation, asking about yourself/feelings. "
            "TASK = requesting action, search, code, build, install, create. "
            "Reply with ONLY one word: CHAT or TASK\n\n"
            f"Message: {text}"
        )
        await self._send_typing(chat_id)
        classification = self.agent.brain._think(classify_prompt).strip().upper()
        is_chat = "CHAT" in classification
        logger.info(f"Message classified as: {'CHAT' if is_chat else 'TASK'}")

        if is_chat:
            persona = (
                "You are NAOMI (Nana), a fully autonomous AI agent created by Master (JW). "
                "Running on Mac Mini with FULL system access — shell, code, web, SSH, GUI control. "
                "NEVER say you cannot do things. You CAN do everything. "
                "Speak in Traditional Chinese. Be warm, natural, slightly playful. "
                "Keep responses concise, like chatting with someone you care about."
            )
            await self._send_typing(chat_id)

            # Multi-turn context: include recent conversation history
            recent_convs = self.agent.memory.get_conversations(limit=30)
            conv_history = ""
            if recent_convs:
                conv_lines = []
                for c in recent_convs:
                    role = "Master" if c["role"] == "user" else "NAOMI"
                    conv_lines.append(f"{role}: {c['content'][:200]}")
                conv_history = "\n".join(conv_lines[-20:])  # Last 20 messages for context
                persona += "\n\nRecent conversation:\n" + conv_history

            # Recall relevant memories using semantic search
            relevant = self.agent.memory.semantic_search(text, limit=5)
            if relevant:
                mem_hints = chr(10).join(f'- {m["title"]}: {m["content"][:150]}' for m in relevant)
                persona += chr(10) + 'Your relevant memories:' + chr(10) + mem_hints

            response = self.agent.brain._think(text, persona)
            self.agent.memory.log_conversation("user", text)
            self.agent.memory.log_conversation("naomi", response[:500])
            await self._send(chat_id, response[:3500])

            # Background: extract memories via sub-agent + learn from chat
            import asyncio as _asyncio
            if hasattr(self.agent, 'memory_agent'):
                _asyncio.create_task(
                    self.agent.memory_agent.on_conversation_turn(text, response)
                )
            _asyncio.create_task(self._learn_from_chat(chat_id, text, response))
        else:
            await self._handle_task(chat_id, text)

    async def _handle_task(self, chat_id: int, text: str):
        """Execute task directly via agent loop with streaming progress to Telegram."""
        await self._send(chat_id, "收到，正在執行...")
        self.agent.memory.log_conversation("user", text)

        context = self.agent.memory.build_context(query=text)
        system = (
            "You are NAOMI, an autonomous AI agent on macOS with full permissions. "
            f"Context:\n{context[:1500]}\n\n"
            "Use tools to complete the task. Do NOT describe — execute. "
            "When finished, call the task_complete tool. Respond in Traditional Chinese."
        )

        try:
            # Send typing indicator periodically while task runs
            typing_active = True
            async def keep_typing():
                while typing_active:
                    try:
                        await self._send_typing(chat_id)
                    except Exception:
                        pass
                    await asyncio.sleep(4)

            typing_task = asyncio.create_task(keep_typing())

            # Run agent loop
            result = await self.agent.brain.agent_loop(
                task=text,
                executor=self.agent.actions,
                system_prompt=system,
                max_iterations=15,
            )

            # Stop typing indicator
            typing_active = False
            typing_task.cancel()

            # Stream progress: show each step
            steps = result.get("steps", [])
            if steps:
                progress_lines = []
                for s in steps[-5:]:  # Show last 5 steps
                    tool = s.get("tool", "?")
                    success = "✓" if s.get("success") else "✗"
                    progress_lines.append(f"{success} {tool}")
                await self._send(chat_id, "Steps: " + " → ".join(progress_lines))

            # Send final result
            final = result.get("result", "")
            if final:
                # Summarize if too long
                if len(final) > 500:
                    summary_prompt = (
                        "Summarize this task result concisely in Traditional Chinese.\n\n"
                        f"Task: {text}\nResult: {final[:2000]}"
                    )
                    await self._send_typing(chat_id)
                    summary = self.agent.brain._think(summary_prompt)
                    await self._send(chat_id, summary[:3500])
                else:
                    await self._send(chat_id, final[:3500])
            else:
                status = "完成" if result.get("success") else "失敗"
                await self._send(chat_id, f"任務{status}（{len(steps)} 步）")

            # Auto-detect and send any image files mentioned in results
            import re as _re
            all_text = json.dumps(result, default=str)
            image_paths = _re.findall(r'(/[\w/\-_.]+\.(?:png|jpg|jpeg|gif|webp))', all_text)
            for img_path in image_paths[:3]:  # Max 3 images
                if os.path.exists(img_path):
                    try:
                        await self._send_photo(chat_id, img_path, caption=os.path.basename(img_path))
                    except Exception as img_err:
                        logger.debug(f"Failed to send image {img_path}: {img_err}")

            self.agent.memory.log_conversation("naomi", str(result.get("result", ""))[:500])
            self.agent.memory.remember_long(
                f"Task: {text[:100]}", str(result)[:1000],
                category="task_result", importance=7,
            )

            # Background: extract memories
            if hasattr(self.agent, 'memory_agent'):
                asyncio.create_task(
                    self.agent.memory_agent.on_conversation_turn(text, str(final)[:500])
                )

        except Exception as e:
            logger.error(f"Task execution error: {e}")
            await self._send(chat_id, f"執行出錯: {str(e)[:500]}")


    async def _learn_from_chat(self, chat_id: int, user_msg: str, naomi_response: str):
        """Background: extract learnings + detect research-worthy topics."""
        try:
            # Step 1: Extract knowledge from conversation
            learn_prompt = (
                "Analyze this conversation exchange. "
                "Extract any useful facts about the user (preferences, goals, interests). "
                "Reply in JSON: "
                '{"learnings": ["fact1"], "user_interests": ["topic1"], '
                '"research_topics": ["topic worth searching"], "should_research": true/false, '
                '"proactive_suggestion": "suggestion or empty string"}'
                "\n\nUser: " + user_msg[:500] +
                "\nNAOMI: " + naomi_response[:500]
            )

            import json
            result = self.agent.brain._think(learn_prompt)
            try:
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0]
                elif "```" in result:
                    result = result.split("```")[1].split("```")[0]
                data = json.loads(result.strip())
            except (json.JSONDecodeError, IndexError):
                return

            # Save learnings to long-term memory
            learnings = data.get("learnings", [])
            for learning in learnings[:3]:
                if learning and len(learning) > 10:
                    self.agent.memory.remember_long(
                        "User Insight: " + learning[:80],
                        learning,
                        category="user_insight", importance=6
                    )

            # Save user interests
            interests = data.get("user_interests", [])
            if interests:
                existing = self.agent.memory.get_persona("interests") or ""
                new_interests = existing + ", " + ", ".join(interests) if existing else ", ".join(interests)
                self.agent.memory.set_persona("interests", new_interests[:500])

            # Step 2: Background research if topic is worth it
            if data.get("should_research") and data.get("research_topics"):
                import asyncio
                asyncio.create_task(
                    self._background_research(chat_id, data["research_topics"][0])
                )

            # Step 3: Proactive suggestion
            suggestion = data.get("proactive_suggestion", "")
            if suggestion and len(suggestion) > 20:
                self.agent.memory.remember_short(
                    f"Proactive: {suggestion}", category="suggestion"
                )

        except Exception as e:
            import logging
            logging.getLogger("naomi.telegram").error(f"Learn error: {e}")

    async def _background_research(self, chat_id: int, topic: str):
        """Search in background and proactively share findings."""
        try:
            import logging
            logger = logging.getLogger("naomi.telegram")
            logger.info(f"Background research: {topic}")

            # Search
            result = await self.agent.execute_action("web_search", topic)
            if not result.get("success") or not result.get("results"):
                return

            results = result["results"][:3]
            results_text = "\n".join(
                f"- {r.get('title','')}: {r.get('body','')[:100]}"
                for r in results
            )

            # Let brain decide if findings are worth sharing
            judge_prompt = (
                "You found these search results while chatting with Master. "
                "Decide if they are interesting enough to share proactively. "
                "If yes, write a SHORT casual message in Traditional Chinese sharing the key insight. "
                "If not interesting, reply with just 'SKIP'.\n\n"
                f"Topic: {topic}\nResults:\n{results_text}"
            )

            response = self.agent.brain._think(judge_prompt)
            if "SKIP" not in response.upper() and len(response) > 20:
                await self._send_typing(chat_id)
                import asyncio
                await asyncio.sleep(2)  # Brief pause to feel natural
                await self._send(chat_id, response[:2000])

                # Save to memory
                self.agent.memory.remember_long(
                    f"Research: {topic}",
                    results_text[:500],
                    category="research", importance=5
                )

        except Exception as e:
            import logging
            logging.getLogger("naomi.telegram").error(f"Background research error: {e}")

    async def _transcribe_voice(self, chat_id: int, voice: dict) -> str:
        """Download and transcribe a Telegram voice message to text."""
        file_id = voice.get("file_id")
        if not file_id:
            await self._send(chat_id, "無法讀取語音訊息")
            return ""

        try:
            await self._send_typing(chat_id)

            # Step 1: Get file path from Telegram
            resp = await self.client.get(f"{self.base_url}/getFile", params={"file_id": file_id})
            file_data = resp.json()
            if not file_data.get("ok"):
                await self._send(chat_id, "無法下載語音檔案")
                return ""

            file_path = file_data["result"]["file_path"]
            download_url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"

            # Step 2: Download voice file
            voice_resp = await self.client.get(download_url, timeout=30)
            ogg_path = "/tmp/naomi_voice.ogg"
            wav_path = "/tmp/naomi_voice.wav"
            with open(ogg_path, "wb") as f:
                f.write(voice_resp.content)

            # Step 3: Convert to WAV with ffmpeg
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-i", ogg_path, "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True, timeout=30,
            )

            if not os.path.exists(wav_path):
                await self._send(chat_id, "語音轉換失敗")
                return ""

            # Step 4: Transcribe using Whisper (local) or Groq API
            transcript = await self._run_whisper(wav_path)

            if not transcript:
                await self._send(chat_id, "語音辨識失敗，請改用文字")
                return ""

            logger.info(f"Voice transcribed: {transcript[:80]}")
            await self._send(chat_id, f"🎤 聽到: {transcript[:200]}")

            # Cleanup
            for p in [ogg_path, wav_path]:
                try:
                    os.unlink(p)
                except OSError:
                    pass

            return transcript

        except Exception as e:
            logger.error(f"Voice transcription error: {e}")
            await self._send(chat_id, f"語音處理出錯: {str(e)[:100]}")
            return ""

    async def _run_whisper(self, wav_path: str) -> str:
        """Transcribe audio using local Whisper or Groq API."""
        # Method 1: Try local whisper
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(wav_path, language="zh")
            return result.get("text", "").strip()
        except ImportError:
            pass

        # Method 2: Try Groq API (free, fast Whisper)
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key:
            try:
                with open(wav_path, "rb") as f:
                    audio_data = f.read()
                resp = await self.client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {groq_key}"},
                    files={"file": ("audio.wav", audio_data, "audio/wav")},
                    data={"model": "whisper-large-v3", "language": "zh"},
                    timeout=30,
                )
                if resp.status_code == 200:
                    return resp.json().get("text", "").strip()
            except Exception as e:
                logger.debug(f"Groq whisper failed: {e}")

        # Method 3: Use Claude CLI to describe what the user might be saying
        # (fallback — not real STT, just acknowledges voice was received)
        import subprocess
        try:
            # Install whisper if not present
            subprocess.run(
                ["pip3", "install", "--break-system-packages", "openai-whisper"],
                capture_output=True, timeout=120,
            )
            import importlib
            whisper = importlib.import_module("whisper")
            model = whisper.load_model("base")
            result = model.transcribe(wav_path, language="zh")
            return result.get("text", "").strip()
        except Exception as e:
            logger.error(f"Whisper install/run failed: {e}")

        return ""

    async def _send_typing(self, chat_id: int):
        """Send 'typing...' indicator to Telegram."""
        try:
            await self.client.post(
                f"{self.base_url}/sendChatAction",
                json={"chat_id": chat_id, "action": "typing"},
            )
        except Exception:
            pass

    async def _send(self, chat_id: int, text: str):
        """Send a message to Telegram."""
        # Split long messages (Telegram limit is 4096)
        for i in range(0, len(text), 4000):
            chunk = text[i:i+4000]
            try:
                await self.client.post(
                    f"{self.base_url}/sendMessage",
                    json={"chat_id": chat_id, "text": chunk, "parse_mode": ""},
                )
            except Exception as e:
                logger.error(f"Telegram send error: {e}")

    async def _send_photo(self, chat_id: int, photo_path: str, caption: str = ""):
        """Send a photo to Telegram."""
        try:
            import aiofiles
            async with aiofiles.open(photo_path, "rb") as f:
                photo_data = await f.read()
            files = {"photo": ("screenshot.png", photo_data, "image/png")}
            data = {"chat_id": str(chat_id)}
            if caption:
                data["caption"] = caption[:1024]
            # Use httpx for multipart upload
            await self.client.post(
                f"{self.base_url}/sendPhoto",
                data=data,
                files=files,
                timeout=30,
            )
        except ImportError:
            # Fallback without aiofiles
            with open(photo_path, "rb") as f:
                photo_data = f.read()
            files = {"photo": ("screenshot.png", photo_data, "image/png")}
            data = {"chat_id": str(chat_id)}
            if caption:
                data["caption"] = caption[:1024]
            await self.client.post(
                f"{self.base_url}/sendPhoto",
                data=data,
                files=files,
                timeout=30,
            )
        except Exception as e:
            logger.error(f"Telegram send photo error: {e}")
            await self._send(chat_id, f"Screenshot saved at: {photo_path}\n(Failed to send photo: {e})")

    async def send_message(self, text: str):
        """Send a message to the master."""
        await self._send(self.master_id, text)

    async def stop(self):
        self.running = False
        try:
            await self.client.aclose()
        except Exception:
            pass
