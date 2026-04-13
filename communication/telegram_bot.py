"""
NAOMI Agent - Telegram Bot Interface
Control NAOMI from your phone via Telegram.
Only responds to authorized user (Master).
"""
import asyncio
import json
import time
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

        if not text:
            return

        logger.info(f"Master command: {text[:100]}")

        # Handle commands
        if text.startswith("/"):
            await self._handle_command(chat_id, text)
        else:
            # Treat as a task for NAOMI
            await self._handle_task(chat_id, text)

    async def _handle_command(self, chat_id: int, text: str):
        """Handle slash commands."""
        cmd = text.split()[0].lower().split("@")[0]  # Remove @botname
        args = text[len(cmd):].strip()

        if cmd == "/start":
            await self._send(chat_id,
                "NAOMI Agent - Telegram Control\n\n"
                "Commands:\n"
                "/status - Agent status\n"
                "/tasks - Recent tasks\n"
                "/memory - Memory stats\n"
                "/skills - Learned skills\n"
                "/think <topic> - Let NAOMI think\n"
                "/search <query> - Web search\n"
                "/council <topic> - Multi-agent debate\n"
                "/evolve - Trigger self-evolution\n"
                "/shell <cmd> - Execute shell command\n"
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
            skills = self.agent.memory.recall_skill()
            if not skills or (isinstance(skills, list) and len(skills) == 0):
                await self._send(chat_id, "No skills learned yet.")
                return
            lines = [f"- {s['name']}: {s.get('description','')[:50]}" for s in (skills if isinstance(skills, list) else [])]
            await self._send(chat_id, "Learned Skills:\n" + "\n".join(lines[:20]))

        elif cmd == "/think":
            if not args:
                await self._send(chat_id, "Usage: /think <topic>")
                return
            await self._send(chat_id, "Thinking...")
            response = self.agent.brain.think(args)
            await self._send(chat_id, f"NAOMI thinks:\n\n{response[:3500]}")

        elif cmd == "/search":
            if not args:
                await self._send(chat_id, "Usage: /search <query>")
                return
            await self._send(chat_id, f"Searching: {args}")
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
            # Safety check
            dangerous = ["rm -rf /", "mkfs", "dd if=", "> /dev/sd", "passwd", "sudo rm"]
            if any(d in args for d in dangerous):
                await self._send(chat_id, "Dangerous command blocked.")
                return
            result = await self.agent.execute_action("shell", args)
            output = result.get("output", result.get("error", "No output"))
            await self._send(chat_id, f"$ {args}\n\n{output[:3500]}")

        elif cmd == "/log":
            entries = self.agent.memory.recall_short(limit=10)
            if not entries:
                await self._send(chat_id, "No recent log entries.")
                return
            lines = [f"[{e['category']}] {e['content'][:80]}" for e in entries]
            await self._send(chat_id, "Recent Log:\n" + "\n".join(lines))

        else:
            await self._send(chat_id, f"Unknown command: {cmd}\nType /start for help.")

    async def _handle_task(self, chat_id: int, text: str):
        """Submit a task to NAOMI."""
        await self._send(chat_id, f"Task queued: {text[:100]}")
        await self.agent.submit_command(text)

        # Wait for completion (poll with timeout)
        for _ in range(60):  # Max 5 minutes
            await asyncio.sleep(5)
            tasks = self.agent.memory.get_recent_tasks(1)
            if tasks and tasks[0]["task"] == text:
                if tasks[0]["status"] in ("completed", "failed"):
                    result = tasks[0].get("result", "No result")
                    status = tasks[0]["status"]
                    icon = "done" if status == "completed" else "FAILED"
                    await self._send(chat_id, f"[{icon}] {text[:60]}\n\n{result[:3500]}")
                    return

        await self._send(chat_id, "Task is still running. Check /tasks later.")

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

    async def send_message(self, text: str):
        """Send a message to the master."""
        await self._send(self.master_id, text)

    def stop(self):
        self.running = False
