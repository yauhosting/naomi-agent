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
        # v1.2: message_id → conversation_id mapping for reaction tracking
        self._msg_to_conv: dict[int, int] = {}
        # v1.2: track last NAOMI response length for implicit feedback
        self._last_response_len: int = 0
        self._last_response_time: float = 0.0

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
        """Poll for new messages and reactions."""
        try:
            resp = await self.client.get(
                f"{self.base_url}/getUpdates",
                params={
                    "offset": self.last_update_id + 1,
                    "timeout": 20,
                    "allowed_updates": json.dumps(["message", "message_reaction"]),
                },
                timeout=25,
            )
            data = resp.json()
            if not data.get("ok"):
                return

            for update in data.get("result", []):
                self.last_update_id = update["update_id"]
                if "message_reaction" in update:
                    await self._handle_reaction(update["message_reaction"])
                else:
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

    # === v1.2: Reaction-based feedback ===
    _POSITIVE_REACTIONS = {"👍", "❤️", "🔥", "👏", "🎉", "❤", "😍", "🥰", "💯"}
    _NEGATIVE_REACTIONS = {"👎", "😢", "💩", "🤮", "😡"}
    _NEUTRAL_REACTIONS = {"🤔", "😐", "🤷"}

    async def _handle_reaction(self, reaction_update: dict):
        """Handle message_reaction update — log as feedback."""
        msg_id = reaction_update.get("message_id")
        user_id = reaction_update.get("user", {}).get("id")
        if user_id != self.master_id:
            return

        new_reactions = reaction_update.get("new_reaction", [])
        if not new_reactions:
            return

        emoji = new_reactions[0].get("emoji", "")
        conv_id = self._msg_to_conv.get(msg_id)

        if emoji in self._POSITIVE_REACTIONS:
            signal, weight = "positive", 1.0
        elif emoji in self._NEGATIVE_REACTIONS:
            signal, weight = "negative", 1.0
        elif emoji in self._NEUTRAL_REACTIONS:
            signal, weight = "neutral", 0.5
        else:
            signal, weight = "neutral", 0.3

        self.agent.memory.log_feedback(
            signal=signal,
            source=f"reaction:{emoji}",
            weight=weight,
            conversation_id=conv_id,
        )
        logger.info("Feedback logged: %s (emoji=%s, conv_id=%s)", signal, emoji, conv_id)

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
                "/skill <search|install|list> - ClawHub skill marketplace\n"
                "/tasks - Recent tasks\n"
                "/memory - Memory stats\n"
                "/skills - Learned skills\n"
                "/think <topic> - Let NAOMI think\n"
                "/search <query> - Web search\n"
                "/research <topic> - Deep research with report\n"
                "/council <topic> - Multi-agent debate\n"
                "/discuss <topic> - Claude vs GPT-5.4 cross-model debate\n"
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
                "/voice <text> - Text-to-speech / toggle voice mode\n"
                "/goal <title> - Manage goals & subgoals\n"
                "/remember <text> - Store in vector memory\n"
                "/recall <query> - Semantic memory search\n"
                "/graph <entity> - Knowledge graph query\n"
                "/browse <url> - Browser automation\n"
                "/private - Toggle privacy mode (local only)\n"
                "/persona - Manage personas (YUMIKO etc.)\n"
                "/session - Session management\n"
                "/email - Gmail operations\n"
                "/cal - Google Calendar\n"
                "/ollama - Manage local Ollama models\n"
                "/security - Run security scan\n"
                "/audit - View audit log\n"
                "/log - Recent activity log\n\n"
                "Or just type anything to give NAOMI a task."
            )

        elif cmd == "/help":
            if not args:
                # Show categorized menu
                await self._send(chat_id,
                    "NAOMI Help - Choose a category:\n\n"
                    "/help brain - LLM models & switching\n"
                    "/help chat - Conversation & voice\n"
                    "/help tools - Shell, code, web, browser\n"
                    "/help skills - ClawHub marketplace & skills\n"
                    "/help memory - Memory, knowledge graph, recall\n"
                    "/help goals - Goal management & planning\n"
                    "/help research - Deep research pipeline\n"
                    "/help computer - Screenshot, click, GUI control\n"
                    "/help comms - Email, calendar, TTS\n"
                    "/help security - Security, sandbox, audit\n"
                    "/help system - Status, evolution, metrics\n"
                    "/help private - YUMIKO & privacy mode\n\n"
                    "Or type /start for quick command list."
                )
            else:
                cat = args.lower()
                help_pages = {
                    "brain": (
                        "Brain & Models\n\n"
                        "/model - View current model\n"
                        "/model <name> - Switch model\n"
                        "/model list - Show all available\n\n"
                        "Available: claude-sonnet, claude-opus, claude-cli, "
                        "openai (GPT-5.4), openai-mini, openai-o3, "
                        "ollama, ollama-gemma, ollama-coder, "
                        "glm, minimax, auto\n\n"
                        "/discuss <topic> - Claude vs GPT-5.4 debate\n"
                        "/council <topic> - Multi-agent council debate"
                    ),
                    "chat": (
                        "Chat & Voice\n\n"
                        "Just type anything to chat with NAOMI.\n\n"
                        "/voice on|off - Toggle voice replies\n"
                        "/voice <text> - One-time TTS\n"
                        "Send a voice message - auto-transcribed to text\n\n"
                        "/think <topic> - Deep thinking mode"
                    ),
                    "tools": (
                        "Tools & Execution\n\n"
                        "/shell <cmd> - Run shell command\n"
                        "/browse <url> - Open page & extract content\n"
                        "/search <query> - Web search (DuckDuckGo)\n\n"
                        "Just describe a task - NAOMI auto-selects tools."
                    ),
                    "skills": (
                        "ClawHub Skill Marketplace\n\n"
                        "/skill search <query> - Search 13,000+ skills\n"
                        "/skill inspect <slug> - View skill details\n"
                        "/skill install <slug> - Install (with security scan)\n"
                        "/skill list - Show installed skills\n\n"
                        "/discover - Auto-discover needed capabilities\n"
                        "/discover scan - Scan for missing tools"
                    ),
                    "memory": (
                        "Memory & Knowledge\n\n"
                        "/remember <text> - Store in vector memory\n"
                        "/recall <query> - Semantic memory search\n"
                        "/graph <entity> - Knowledge graph query\n"
                        "/memory - Memory statistics\n\n"
                        "NAOMI auto-stores conversations and extracts knowledge."
                    ),
                    "goals": (
                        "Goals & Planning\n\n"
                        "/goal - View active goals\n"
                        "/goal add <title> - Add new goal (auto-decomposes)\n"
                        "/goal done <id> - Complete a goal\n\n"
                        "Complex tasks auto-use plan-execute-reflect loop."
                    ),
                    "research": (
                        "Deep Research\n\n"
                        "/research <topic> - Multi-source research pipeline\n"
                        "  1. Decomposes into sub-questions\n"
                        "  2. Parallel web search & extraction\n"
                        "  3. Cross-reference findings\n"
                        "  4. Structured report with citations"
                    ),
                    "computer": (
                        "Computer Control\n\n"
                        "/screen - Take screenshot\n"
                        "/click x y - Click coordinates\n"
                        "/type <text> - Type text\n"
                        "/key <key> - Press key (return, cmd+c...)\n"
                        "/app [name] - Open app / list windows\n"
                        "/scroll up|down [amount]"
                    ),
                    "comms": (
                        "Communications\n\n"
                        "/email - Gmail operations\n"
                        "/cal - Google Calendar\n"
                        "/voice - Text-to-speech settings\n\n"
                        "Supports: Kokoro (MLX), Qwen3-TTS, Edge TTS"
                    ),
                    "security": (
                        "Security & Safety\n\n"
                        "/security - Run security scan\n"
                        "/audit - View audit log\n\n"
                        "Core files protected from self-evolution\n"
                        "ClawHub skills scanned by Claude + GPT-5.4\n"
                        "Sensitive commands logged\n"
                        "Docker sandbox for risky operations"
                    ),
                    "system": (
                        "System & Administration\n\n"
                        "/status - Agent status & uptime\n"
                        "/usage - API usage statistics\n"
                        "/log - Recent activity log\n"
                        "/evolve - Trigger self-evolution\n"
                        "/schedule - Managed scheduled tasks\n"
                        "/session - Session management\n\n"
                        "Dashboard: http://127.0.0.1:18802"
                    ),
                    "private": (
                        "Privacy Mode\n\n"
                        "/private - Toggle YUMIKO mode (local only)\n"
                        "/private off - Return to NAOMI mode\n"
                        "/persona - Manage personas\n\n"
                        "Private mode: all processing on local Ollama.\n"
                        "No data leaves your machine."
                    ),
                }
                page = help_pages.get(
                    cat,
                    f"Unknown category: {cat}\nUse /help to see categories.",
                )
                await self._send(chat_id, page)

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

        elif cmd == "/skill":
            if not hasattr(self.agent, 'discovery'):
                await self._send(chat_id, "Discovery engine not initialized.")
                return
            if not args:
                await self._send(chat_id,
                    "🧩 ClawHub Skill Marketplace (13,000+ skills)\n\n"
                    "/skill search <query> — Search skills\n"
                    "/skill install <slug> — Install with security scan\n"
                    "/skill list — Show installed skills\n"
                    "/skill inspect <slug> — View skill details"
                )
                return
            sub = args.split(None, 1)
            sub_cmd = sub[0].lower()
            sub_args = sub[1] if len(sub) > 1 else ""

            if sub_cmd == "search" and sub_args:
                await self._send(chat_id, f"🔍 Searching ClawHub: {sub_args}")
                import asyncio as _aio
                result = await _aio.to_thread(
                    self.agent.discovery.clawhub_search, sub_args
                )
                if result.get("success") and result.get("results"):
                    lines = []
                    for s in result["results"]:
                        lines.append(f"  `{s['slug']}` — {s['name']} ({s.get('score', '')})")
                    await self._send(chat_id,
                        f"Found {len(result['results'])} skills:\n" + "\n".join(lines)
                        + "\n\nInstall: /skill install <slug>"
                    )
                else:
                    await self._send(chat_id, f"No skills found for '{sub_args}'")

            elif sub_cmd == "inspect" and sub_args:
                await self._send(chat_id, f"🔎 Inspecting: {sub_args}")
                import asyncio as _aio
                result = await _aio.to_thread(
                    self.agent.discovery.clawhub_inspect, sub_args
                )
                if result.get("success"):
                    await self._send(chat_id,
                        f"📦 {result['name']} (v{result['version']})\n"
                        f"By: {result['owner']}\n"
                        f"⬇️ {result['downloads']} downloads | ⭐ {result['stars']}\n\n"
                        f"{result['summary'][:500]}\n\n"
                        f"Install: /skill install {sub_args}"
                    )
                else:
                    await self._send(chat_id, f"Not found: {result.get('error', '?')}")

            elif sub_cmd == "install" and sub_args:
                await self._send(chat_id,
                    f"⬇️ Downloading {sub_args}...\n"
                    "🔒 Will run Claude + GPT-5.4 dual security scan before installing."
                )
                await self._send_typing(chat_id)
                import asyncio as _aio
                result = await _aio.to_thread(
                    self.agent.discovery.clawhub_install, sub_args
                )
                if result.get("success"):
                    await self._send(chat_id,
                        f"✅ Skill `{sub_args}` installed!\n"
                        f"Security scan: PASSED ✅\n"
                        f"Path: {result.get('path', '?')}"
                    )
                elif result.get("action") == "rejected_and_inspired":
                    inspired = result.get("inspired_skill", {})
                    msg = (
                        f"🚫 Skill `{sub_args}` REJECTED by security scan\n"
                        f"Flagged by: {result.get('flagged_by', '?')}\n"
                        f"Reason: {result.get('reason', '?')[:300]}\n"
                    )
                    if inspired.get("success"):
                        msg += f"\n✨ Created safe alternative: `{inspired['slug']}`"
                    await self._send(chat_id, msg)
                else:
                    await self._send(chat_id,
                        f"❌ Install failed: {result.get('error', '?')[:300]}"
                    )

            elif sub_cmd == "list":
                if hasattr(self.agent, 'skills'):
                    skills = self.agent.skills.list_skills()
                    if skills:
                        lines = [f"  `{s['name']}` — {s.get('description', '')[:60]}" for s in skills]
                        await self._send(chat_id,
                            f"📦 Installed Skills ({len(skills)}):\n" + "\n".join(lines)
                        )
                    else:
                        await self._send(chat_id, "No skills installed yet.")
                else:
                    await self._send(chat_id, "Skill manager not initialized.")
            else:
                await self._send(chat_id, "Usage: /skill <search|install|inspect|list> <name>")

        elif cmd == "/remember":
            if not args:
                await self._send(chat_id, "Usage: /remember <text to store in vector memory>")
                return
            if hasattr(self.agent, 'vector_memory'):
                import asyncio as _aio
                row_id = await _aio.to_thread(
                    self.agent.vector_memory.add, args, "user_memory"
                )
                await self._send(chat_id, f"🧠 Stored in vector memory (id: {row_id})")
                # Also extract knowledge triples
                if hasattr(self.agent, 'knowledge_graph'):
                    await _aio.to_thread(
                        self.agent.knowledge_graph.extract_from_text, args, self.agent.brain
                    )
            else:
                await self._send(chat_id, "Vector memory not initialized.")

        elif cmd == "/recall":
            if not args:
                await self._send(chat_id, "Usage: /recall <query> — semantic memory search")
                return
            if hasattr(self.agent, 'vector_memory'):
                import asyncio as _aio
                results = await _aio.to_thread(
                    self.agent.vector_memory.search, args, 5
                )
                if results:
                    lines = []
                    for r in results:
                        score = f"{r.get('score', 0):.2f}"
                        text = r.get('text', '')[:120]
                        lines.append(f"  [{score}] {text}")
                    await self._send(chat_id, f"🔍 Found {len(results)} memories:\n" + "\n".join(lines))
                else:
                    await self._send(chat_id, "No matching memories found.")
            else:
                await self._send(chat_id, "Vector memory not initialized.")

        elif cmd == "/graph":
            if not args:
                await self._send(chat_id, "Usage: /graph <entity> — query knowledge graph")
                return
            if hasattr(self.agent, 'knowledge_graph'):
                import asyncio as _aio
                context = await _aio.to_thread(
                    self.agent.knowledge_graph.get_context, args
                )
                if context:
                    await self._send(chat_id, f"🕸️ Knowledge about '{args}':\n\n{context[:3000]}")
                else:
                    await self._send(chat_id, f"No knowledge found about '{args}'.")
            else:
                await self._send(chat_id, "Knowledge graph not initialized.")

        elif cmd == "/goal":
            if not hasattr(self.agent, 'goals'):
                await self._send(chat_id, "Goal system not initialized.")
                return
            if not args:
                # Show current goals
                import asyncio as _aio
                stats = await _aio.to_thread(self.agent.goals.get_stats)
                active = await _aio.to_thread(self.agent.goals.get_active_goals, 5)
                msg = (
                    f"🎯 Goals: {stats.get('active', 0)} active, "
                    f"{stats.get('completed', 0)} completed\n\n"
                )
                if active:
                    for g in active:
                        msg += f"  [{g.priority}] {g.title}\n"
                else:
                    msg += "No active goals.\n"
                msg += "\n/goal add <title> — New goal\n/goal done <id> — Complete"
                await self._send(chat_id, msg)
            else:
                sub = args.split(None, 1)
                sub_cmd = sub[0].lower()
                sub_args = sub[1] if len(sub) > 1 else ""
                import asyncio as _aio
                if sub_cmd == "add" and sub_args:
                    goal_id = await _aio.to_thread(self.agent.goals.add_goal, sub_args)
                    await self._send(chat_id, f"🎯 Goal added (id: {goal_id}): {sub_args}")
                    # Auto-decompose
                    await self._send_typing(chat_id)
                    subs = await _aio.to_thread(
                        self.agent.goals.decompose, goal_id, self.agent.brain
                    )
                    if subs.get("success") and subs.get("subgoals"):
                        lines = [f"  └ {s['title']}" for s in subs["subgoals"][:5]]
                        await self._send(chat_id, "Auto-decomposed:\n" + "\n".join(lines))
                elif sub_cmd == "done" and sub_args.isdigit():
                    result = await _aio.to_thread(self.agent.goals.complete_goal, int(sub_args))
                    if result.get("success"):
                        await self._send(chat_id, f"✅ Goal {sub_args} completed!")
                    else:
                        await self._send(chat_id, f"❌ {result.get('error', '?')}")
                else:
                    await self._send(chat_id, "Usage: /goal [add <title> | done <id>]")

        elif cmd == "/browse":
            if not args:
                await self._send(chat_id, "Usage: /browse <url> — open and extract page content")
                return
            await self._send(chat_id, f"🌐 Browsing: {args}")
            await self._send_typing(chat_id)
            try:
                from core.browser import BrowserAgent
                browser = BrowserAgent()
                result = await browser.navigate(args)
                if result.get("success"):
                    text = result.get("text_preview", "")[:2000]
                    title = result.get("title", "?")
                    await self._send(chat_id, f"📄 {title}\n\n{text}")
                else:
                    await self._send(chat_id, f"❌ {result.get('error', 'Failed')}")
                await browser.close()
            except Exception as e:
                await self._send(chat_id, f"Browser error: {e}")

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

        elif cmd == "/discuss":
            if not args:
                await self._send(chat_id, "Usage: /discuss <topic>\nExample: /discuss AI是否會取代人類工作")
                return
            # Parse optional round count: /discuss 5 topic here
            parts = args.split(None, 1)
            rounds = 3
            topic = args
            if parts[0].isdigit():
                rounds = min(int(parts[0]), 5)  # max 5 rounds
                topic = parts[1] if len(parts) > 1 else ""
            if not topic:
                await self._send(chat_id, "Usage: /discuss <topic>")
                return
            await self._send(chat_id, f"⚔️ Claude vs GPT-5.4 辯論開始\n主題：{topic}\n回合：{rounds}")
            await self._send_typing(chat_id)

            import asyncio as _aio
            result = await _aio.to_thread(
                self.agent.brain.cross_model_discuss, topic, rounds
            )

            if result.get("error"):
                await self._send(chat_id, f"❌ {result['error']}")
                return

            # Send each round
            for entry in result.get("debate_log", []):
                icon = "🟣" if entry["model"] == "Claude" else "🟢"
                msg = f"{icon} **{entry['model']}** (R{entry['round']})\n{entry['content']}"
                await self._send(chat_id, msg[:4000])
                await _aio.sleep(0.5)

            # Send summary
            summary = result.get("summary", "")
            if summary:
                await self._send(chat_id, f"🏆 **NAOMI 總結**\n\n{summary[:4000]}")

        elif cmd == "/research":
            if not args:
                await self._send(chat_id, "Usage: /research <topic>\nExample: /research AI agents 2024 trends")
                return
            # Parse optional depth: /research 5 topic here
            parts = args.split(None, 1)
            depth = 3
            topic = args
            if parts[0].isdigit() and len(parts) > 1:
                depth = max(2, min(int(parts[0]), 5))
                topic = parts[1]
            if not topic:
                await self._send(chat_id, "Usage: /research <topic>")
                return

            await self._send(chat_id, f"Deep research started\nTopic: {topic}\nSub-questions: {depth}")

            async def _research_progress(msg: str):
                await self._send(chat_id, msg)

            try:
                from core.researcher import DeepResearcher
                researcher = DeepResearcher(self.agent.brain, self.agent.actions)

                # Keep typing indicator active
                typing_active = True
                async def keep_typing_research():
                    while typing_active:
                        try:
                            await self._send_typing(chat_id)
                        except Exception:
                            pass
                        await asyncio.sleep(4)
                typing_task = asyncio.create_task(keep_typing_research())

                result = await researcher.research(
                    topic, depth=depth, progress_callback=_research_progress
                )

                typing_active = False
                typing_task.cancel()

                if result.get("success"):
                    # Send individual findings progressively
                    for i, finding in enumerate(result.get("findings", []), 1):
                        q = finding.get("question", "")
                        a = finding.get("answer", "")[:800]
                        src_count = len(finding.get("sources", []))
                        await self._send(chat_id,
                            f"Finding {i}/{len(result['findings'])}: {q}\n\n{a}\n\n({src_count} sources)"
                        )
                        await asyncio.sleep(0.5)

                    # Send final report
                    report = result.get("report", "")
                    duration = result.get("duration_seconds", 0)
                    if report:
                        header = f"Research Report ({duration}s)\n{'=' * 30}\n\n"
                        full = header + report
                        for i in range(0, len(full), 3800):
                            chunk = full[i:i + 3800]
                            if chunk.strip():
                                await self._send(chat_id, chunk)
                else:
                    await self._send(chat_id, f"Research failed: {result.get('report', 'Unknown error')}")

            except Exception as e:
                logger.error("Research command error: %s", e)
                await self._send(chat_id, f"Research error: {str(e)[:300]}")

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

        elif cmd == "/private":
            if not args:
                current = self.agent.brain._private_mode
                result = self.agent.brain.set_private_mode(not current)
                if result["private_mode"]:
                    await self._send(chat_id, "😈 YUMIKO mode ON\n所有對話都在本機處理，不會發送到任何外部 API")
                else:
                    await self._send(chat_id, "😇 NAOMI mode\n恢復自動路由（聊天→MiniMax, 代碼→Claude CLI）")
            elif args.lower() in ("on", "開", "1"):
                self.agent.brain.set_private_mode(True)
                await self._send(chat_id, "😈 YUMIKO mode ON")
            elif args.lower() in ("off", "關", "0"):
                self.agent.brain.set_private_mode(False)
                await self._send(chat_id, "😇 NAOMI mode")
            elif args.lower().startswith("persona "):
                custom = args[8:].strip()
                self.agent.brain.set_private_persona(custom)
                if not self.agent.brain._private_mode:
                    self.agent.brain.set_private_mode(True)
                await self._send(chat_id, f"🔒 Private mode ON\n自訂人格已設定")
            else:
                self.agent.brain.set_private_mode(True)
                await self._send(chat_id, "🔒 Private mode ON")

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

        elif cmd == "/session":
            session_mgr = getattr(self.agent, 'session_manager', None)
            if not session_mgr:
                await self._send(chat_id, "Session manager not initialized.")
                return
            persona = "yumiko" if self.agent.brain._private_mode else "naomi"
            if not args or args == "list":
                sessions = session_mgr.list_sessions(persona, limit=8)
                if not sessions:
                    await self._send(chat_id, "No sessions found.")
                else:
                    lines = []
                    active = session_mgr.get_active_session(persona) or ""
                    for s in sessions:
                        marker = "→ " if s["session_id"] == active else "  "
                        lines.append(
                            f"{marker}{s['session_id']} ({s['msg_count']} msgs) "
                            f"{s['preview'][:40]}"
                        )
                    await self._send(chat_id, "Sessions:\n" + "\n".join(lines))
            elif args == "new":
                new_id = session_mgr.create_session(persona)
                await self._send(chat_id, f"New session: {new_id}")
            else:
                # Switch to specific session
                session_mgr._active_sessions[persona] = args.strip()
                await self._send(chat_id, f"Switched to session: {args.strip()}")

        elif cmd == "/persona":
            drift = getattr(self.agent, 'persona_drift', None)
            if not drift:
                await self._send(chat_id, "Persona drift not initialized.")
                return
            persona = "yumiko" if self.agent.brain._private_mode else "naomi"
            if args == "drift":
                # Force a drift now
                success = drift._run_drift(persona, self.agent.memory.get_latest_drift(persona))
                if success:
                    await self._send(chat_id, "✨ Persona drift applied!")
                else:
                    await self._send(chat_id, "Drift failed (not enough data?)")
            else:
                status = drift.get_status(persona)
                style = status["style"]
                fb = status["feedback"]
                lines = [
                    f"Persona Drift v{status['version']}",
                    f"Tone: {style.get('tone', '?')}",
                    f"Verbosity: {style.get('verbosity', 0.5):.0%}",
                    f"Humor: {style.get('humor', 0.5):.0%}",
                    f"Formality: {style.get('formality', 0.3):.0%}",
                    f"Emoji: {style.get('emoji_level', 0.3):.0%}",
                    f"Topics: {', '.join(style.get('topics_of_interest', [])[:5]) or 'none'}",
                    f"",
                    f"Feedback: {fb['score']:.0%} positive ({fb['total']} signals)",
                    f"Next drift in: {status['conversations_until_next_drift']} conversations",
                    f"Last reason: {status['last_drift_reason']}",
                ]
                await self._send(chat_id, "\n".join(lines))

        elif cmd == "/email":
            gmail = getattr(self.agent, 'gmail', None)
            if not gmail or not gmail.available:
                await self._send(chat_id,
                    "Gmail not configured.\n"
                    "Put credentials.json in data/gmail_credentials.json\n"
                    "(Google Cloud Console → OAuth2 → Desktop App)")
                return
            if not args or args == "inbox":
                msgs = gmail.list_messages("is:unread", max_results=5)
                if not msgs:
                    await self._send(chat_id, "No unread emails.")
                else:
                    lines = ["📬 Unread emails:"]
                    for m in msgs:
                        lines.append(f"\n• {m['from'][:40]}\n  {m['subject'][:60]}\n  {m['snippet'][:80]}")
                    await self._send(chat_id, "\n".join(lines))
            elif args.startswith("read "):
                msg_id = args[5:].strip()
                msg = gmail.read_message(msg_id)
                if msg:
                    await self._send(chat_id,
                        f"From: {msg['from']}\n"
                        f"Subject: {msg['subject']}\n"
                        f"Date: {msg['date']}\n\n"
                        f"{msg['body'][:2000]}")
                else:
                    await self._send(chat_id, "Message not found.")
            elif args.startswith("send "):
                # /email send to@example.com | Subject | Body
                parts = args[5:].split("|", 2)
                if len(parts) < 3:
                    await self._send(chat_id, "Usage: /email send to@email.com | Subject | Body")
                    return
                to, subject, body = [p.strip() for p in parts]
                msg_id = gmail.send_message(to, subject, body)
                if msg_id:
                    await self._send(chat_id, f"✅ Email sent to {to}")
                else:
                    await self._send(chat_id, "❌ Failed to send email")
            elif args.startswith("search "):
                query = args[7:].strip()
                msgs = gmail.search(query, max_results=5)
                if not msgs:
                    await self._send(chat_id, f"No results for: {query}")
                else:
                    lines = [f"🔍 Search: {query}"]
                    for m in msgs:
                        lines.append(f"\n• {m['from'][:40]}\n  {m['subject'][:60]}")
                    await self._send(chat_id, "\n".join(lines))
            else:
                await self._send(chat_id,
                    "/email — unread inbox\n"
                    "/email read <id> — read full message\n"
                    "/email send to|subject|body — send\n"
                    "/email search <query> — search")

        elif cmd == "/cal":
            cal = getattr(self.agent, 'calendar', None)
            if not cal or not cal.available:
                await self._send(chat_id,
                    "Calendar not configured.\n"
                    "Put credentials.json in data/gmail_credentials.json\n"
                    "Enable Calendar API in Google Cloud Console.")
                return
            if not args or args == "today":
                events = cal.today_events()
                if not events:
                    await self._send(chat_id, "📅 No events today.")
                else:
                    lines = ["📅 Today's events:"]
                    for ev in events:
                        start = ev['start']
                        if 'T' in start:
                            start = start[11:16]  # Extract HH:MM
                        lines.append(f"  {start} — {ev['summary']}")
                        if ev['location']:
                            lines.append(f"    📍 {ev['location'][:50]}")
                    await self._send(chat_id, "\n".join(lines))
            elif args == "week":
                events = cal.list_events(days=7)
                if not events:
                    await self._send(chat_id, "No events this week.")
                else:
                    lines = ["📅 This week:"]
                    for ev in events:
                        start = ev['start'][:16].replace('T', ' ')
                        lines.append(f"  {start} — {ev['summary']}")
                    await self._send(chat_id, "\n".join(lines))
            elif args.startswith("add "):
                # Natural language: /cal add Meeting with JW tomorrow at 3pm for 1 hour
                text = args[4:].strip()
                result = cal.quick_add(text)
                if result:
                    await self._send(chat_id, f"✅ Event created: {result.get('summary', text)}")
                else:
                    await self._send(chat_id, "❌ Failed to create event")
            elif args.startswith("search "):
                query = args[7:].strip()
                events = cal.search_events(query)
                if not events:
                    await self._send(chat_id, f"No events matching: {query}")
                else:
                    lines = [f"🔍 Events: {query}"]
                    for ev in events:
                        lines.append(f"  {ev['start'][:16]} — {ev['summary']}")
                    await self._send(chat_id, "\n".join(lines))
            else:
                await self._send(chat_id,
                    "/cal — today's events\n"
                    "/cal week — this week\n"
                    "/cal add <natural language> — create event\n"
                    "/cal search <query> — search events")

        elif cmd == "/voice":
            if not args:
                # Toggle voice reply mode
                current = getattr(self, '_voice_mode', False)
                self._voice_mode = not current
                status = "ON" if self._voice_mode else "OFF"
                await self._send(chat_id, f"🎙 Voice reply mode: {status}")
            elif args.lower() in ("on", "開", "1"):
                self._voice_mode = True
                await self._send(chat_id, "🎙 Voice reply mode: ON")
            elif args.lower() in ("off", "關", "0"):
                self._voice_mode = False
                await self._send(chat_id, "🎙 Voice reply mode: OFF")
            else:
                # /voice <text> — synthesize and send as voice
                from core.tts import text_to_speech
                audio_path = await text_to_speech(args)
                if audio_path:
                    await self._send_voice(chat_id, audio_path)
                else:
                    await self._send(chat_id, "TTS failed. Need ffmpeg installed.")

        else:
            await self._send(chat_id, f"Unknown command: {cmd}\nType /help for categories or /start for quick list.")

    def _get_session_id(self, persona: str) -> str:
        """Get or create session for the current persona."""
        session_mgr = getattr(self.agent, 'session_manager', None)
        if session_mgr:
            return session_mgr.get_or_create_session(persona)
        return "default"

    def _detect_implicit_feedback(self, text: str):
        """Detect implicit feedback from reply patterns."""
        if self._last_response_time == 0:
            return
        gap = time.time() - self._last_response_time
        # Only consider replies within 5 minutes
        if gap > 300:
            self._last_response_time = 0
            return
        # Very short reply to a long response → might be dissatisfied
        if len(text) < 5 and self._last_response_len > 200:
            self.agent.memory.log_feedback(
                signal="negative", source="implicit:short_reply", weight=0.3
            )
        # Quick enthusiastic reply (with ! or emoji)
        elif gap < 30 and any(c in text for c in "！!❤️👍🔥讚好"):
            self.agent.memory.log_feedback(
                signal="positive", source="implicit:enthusiastic", weight=0.3
            )
        self._last_response_time = 0

    async def _handle_message(self, chat_id: int, text: str):
        """Detect if message is chat or task, respond accordingly."""

        # Implicit feedback detection from reply patterns
        self._detect_implicit_feedback(text)

        # YUMIKO mode: same capabilities but all local models
        if self.agent.brain._private_mode:
            persona_name = "yumiko"
            session_id = self._get_session_id(persona_name)
            complexity = self.agent.brain._classify_complexity(text)

            if complexity in ("private", "chat"):
                logger.info("YUMIKO mode: chat → local Ollama")
                await self._send_typing(chat_id)

                recent_convs = self.agent.memory.get_conversations(
                    limit=20, persona=persona_name, session_id=session_id
                )
                conv_history = ""
                if recent_convs:
                    conv_lines = [f"{'Master' if c['role']=='user' else 'YUMIKO'}: {c['content'][:200]}"
                                  for c in recent_convs]
                    conv_history = "\n".join(conv_lines[-15:])

                persona = self.agent.brain.get_private_persona()
                if conv_history:
                    persona += "\n\nRecent conversation:\n" + conv_history

                response = self.agent.brain.think_smart(text, persona)
                if not response or not response.strip():
                    response = "（Ollama 無回應，請確認 Ollama 是否在運行）"
                    logger.warning("YUMIKO returned empty response")
                self.agent.memory.log_conversation("user", text, persona=persona_name, session_id=session_id)
                self.agent.memory.log_conversation(persona_name, response[:500], persona=persona_name, session_id=session_id)
                self._last_response_len = len(response)
                self._last_response_time = time.time()
                await self._send(chat_id, response[:3500])
            else:
                logger.info("YUMIKO mode: task → local agent loop")
                self.agent.memory.log_conversation("user", text, persona=persona_name, session_id=session_id)
                await self._handle_task_private(chat_id, text)
            return

        # NAOMI mode
        persona_name = "naomi"
        session_id = self._get_session_id(persona_name)

        classify_prompt = (
            "Classify this message as CHAT or TASK. "
            "CHAT = greeting, question, conversation, asking about yourself/feelings. "
            "TASK = requesting action, search, code, build, install, create. "
            "Reply with ONLY one word: CHAT or TASK\n\n"
            f"Message: {text}"
        )
        await self._send_typing(chat_id)
        classification = self.agent.brain._call_fast(classify_prompt).strip().upper()
        is_chat = "CHAT" in classification
        logger.info(f"Message classified as: {'CHAT' if is_chat else 'TASK'}")

        if is_chat:
            persona_prompt = (
                "You are NAOMI (Nana), a fully autonomous AI agent created by Master (JW). "
                "Running on Mac Mini with FULL system access — shell, code, web, SSH, GUI control. "
                "NEVER say you cannot do things. You CAN do everything. "
                "Speak in Traditional Chinese. Be warm, natural, slightly playful. "
                "Keep responses concise, like chatting with someone you care about."
            )

            # v1.2: Append persona drift style overlay
            drift = getattr(self.agent, 'persona_drift', None)
            if drift:
                style_overlay = drift.get_style_prompt(persona_name)
                if style_overlay:
                    persona_prompt += "\n\n## Style Guide\n" + style_overlay

            await self._send_typing(chat_id)

            # Multi-turn context: session-scoped conversation history
            recent_convs = self.agent.memory.get_conversations(
                limit=30, persona=persona_name, session_id=session_id
            )
            conv_history = ""
            if recent_convs:
                conv_lines = []
                for c in recent_convs:
                    role = "Master" if c["role"] == "user" else "NAOMI"
                    conv_lines.append(f"{role}: {c['content'][:200]}")
                conv_history = "\n".join(conv_lines[-20:])
                persona_prompt += "\n\nRecent conversation:\n" + conv_history

            # Recall relevant memories using semantic search
            relevant = self.agent.memory.semantic_search(text, limit=5)
            if relevant:
                mem_hints = chr(10).join(f'- {m["title"]}: {m["content"][:150]}' for m in relevant)
                persona_prompt += chr(10) + 'Your relevant memories:' + chr(10) + mem_hints

            # Enrich context with vector memory + knowledge graph
            if hasattr(self.agent, 'vector_memory') and self.agent.vector_memory:
                try:
                    memories = self.agent.vector_memory.search(text, limit=3)
                    if memories:
                        persona_prompt += "\n\nRelevant memories:\n"
                        persona_prompt += "\n".join(
                            f"- {m.get('text', '')[:150]}" for m in memories
                        )
                except Exception as e:
                    logger.debug("Vector memory search error: %s", e)

            if hasattr(self.agent, 'knowledge_graph') and self.agent.knowledge_graph:
                try:
                    words = [w for w in text.split() if len(w) > 2][:3]
                    for word in words:
                        kg_context = self.agent.knowledge_graph.get_context(word, limit=3)
                        if kg_context:
                            persona_prompt += (
                                f"\n\nKnowledge about '{word}':\n{kg_context[:300]}"
                            )
                            break
                except Exception as e:
                    logger.debug("Knowledge graph query error: %s", e)

            response = self.agent.brain.think_smart(text, persona_prompt)
            self.agent.memory.log_conversation("user", text, persona=persona_name, session_id=session_id)
            self.agent.memory.log_conversation(persona_name, response[:500], persona=persona_name, session_id=session_id)
            self._last_response_len = len(response)
            self._last_response_time = time.time()
            await self._send(chat_id, response[:3500])

            # Store conversation in vector memory for future recall
            if hasattr(self.agent, 'vector_memory') and self.agent.vector_memory:
                try:
                    self.agent.vector_memory.add(
                        f"Q: {text[:200]} A: {response[:200]}",
                        category="conversation",
                    )
                except Exception as e:
                    logger.debug("Vector memory store error: %s", e)

            # Background: extract memories (NAOMI only, not YUMIKO)
            if hasattr(self.agent, 'memory_agent'):
                asyncio.create_task(
                    self.agent.memory_agent.on_conversation_turn(text, response)
                )
            asyncio.create_task(self._learn_from_chat(chat_id, text, response))

            # v1.3: TTS voice reply if voice mode is on
            if getattr(self, '_voice_mode', False):
                asyncio.create_task(self._send_tts_reply(chat_id, response))

            # v1.2: Check if persona drift should trigger
            if drift:
                asyncio.create_task(asyncio.to_thread(drift.maybe_drift, persona_name))
        else:
            await self._handle_task(chat_id, text)

    async def _handle_task_private(self, chat_id: int, text: str):
        """Execute task in YUMIKO mode — agent loop via local Ollama only."""
        await self._send(chat_id, "😈 收到，正在本地執行...")
        try:
            typing_active = True
            async def keep_typing():
                while typing_active:
                    try:
                        await self._send_typing(chat_id)
                    except Exception:
                        pass
                    await asyncio.sleep(4)
            typing_task = asyncio.create_task(keep_typing())

            # Force Ollama model for agent loop
            code_model = self.agent.brain._private_code_model
            context = self.agent.memory.build_context(query=text)
            system = (
                "You are YUMIKO, an autonomous AI agent on macOS with full permissions. "
                f"Context:\n{context[:1500]}\n\n"
                "Use tools to complete the task. Respond in Traditional Chinese."
            )

            # Use Ollama for the agent loop
            result = await self.agent.brain._agent_loop_ollama(
                task=text, executor=self.agent.actions,
                system_prompt=system, model=code_model,
            )

            typing_active = False
            typing_task.cancel()

            # Send result
            final = result.get("result", "")
            if final:
                await self._send(chat_id, final[:3500])
            else:
                status = "完成" if result.get("success") else "失敗"
                await self._send(chat_id, f"任務{status}")

            # Auto-send images
            import re as _re
            all_text = json.dumps(result, default=str)
            image_paths = _re.findall(r'(/[\w/\-_.]+\.(?:png|jpg|jpeg|gif|webp))', all_text)
            for img_path in image_paths[:3]:
                if os.path.exists(img_path):
                    try:
                        await self._send_photo(chat_id, img_path)
                    except Exception:
                        pass

            self.agent.memory.log_conversation("yumiko", str(final)[:500], persona="yumiko")

        except Exception as e:
            logger.error(f"YUMIKO task error: {e}")
            await self._send(chat_id, f"執行出錯: {str(e)[:300]}")

    async def _handle_task(self, chat_id: int, text: str):
        """Execute task directly via agent loop with streaming progress to Telegram."""
        await self._send(chat_id, "收到，正在執行...")
        persona_name = "yumiko" if self.agent.brain._private_mode else "naomi"
        session_id = self._get_session_id(persona_name)
        self.agent.memory.log_conversation("user", text, persona=persona_name, session_id=session_id)

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

            self.agent.memory.log_conversation("naomi", str(result.get("result", ""))[:500],
                                               persona=persona_name, session_id=session_id)
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
            result = self.agent.brain._call_fast(learn_prompt)
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

            response = self.agent.brain._call_fast(judge_prompt)
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
            with open(ogg_path, "wb") as f:
                f.write(voice_resp.content)

            # Step 3: Transcribe using unified STT engine (handles format conversion internally)
            from core.stt import transcribe
            transcript = await transcribe(ogg_path, language="auto")

            if not transcript:
                await self._send(chat_id, "語音辨識失敗，請改用文字")
                return ""

            logger.info(f"Voice transcribed: {transcript[:80]}")
            await self._send(chat_id, f"🎤 聽到: {transcript[:200]}")

            # Cleanup
            for p in [ogg_path, ogg_path.replace(".ogg", ".wav")]:
                try:
                    os.unlink(p)
                except OSError:
                    pass

            return transcript

        except Exception as e:
            logger.error(f"Voice transcription error: {e}")
            await self._send(chat_id, f"語音處理出錯: {str(e)[:100]}")
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
        if not text or not text.strip():
            logger.warning("Attempted to send empty message, skipping")
            return
        # Split long messages (Telegram limit is 4096)
        for i in range(0, len(text), 4000):
            chunk = text[i:i+4000]
            if not chunk.strip():
                continue
            try:
                resp = await self.client.post(
                    f"{self.base_url}/sendMessage",
                    json={"chat_id": chat_id, "text": chunk, "parse_mode": ""},
                )
                if resp.status_code != 200:
                    logger.error("Telegram send HTTP %d: %s", resp.status_code, resp.text[:200])
            except Exception as e:
                logger.error("Telegram send error: %s", e)

    async def _send_streaming(self, chat_id: int, generator_func, *args):
        """Send a streaming response by progressively editing a Telegram message.

        1. Send initial "..." message, get message_id
        2. Call generator_func(*args) which yields text chunks
        3. Every 800ms, edit the message with accumulated text
        4. Final edit with complete text
        """
        # Send initial placeholder message
        try:
            resp = await self.client.post(
                f"{self.base_url}/sendMessage",
                json={"chat_id": chat_id, "text": "..."},
            )
            if resp.status_code != 200:
                logger.error("Streaming: failed to send initial message")
                return
            msg_data = resp.json()
            message_id = msg_data.get("result", {}).get("message_id")
            if not message_id:
                logger.error("Streaming: no message_id in response")
                return
        except Exception as e:
            logger.error("Streaming: initial send error: %s", e)
            return

        accumulated = ""
        last_edit_time = time.time()
        edit_interval = 0.8  # seconds between edits

        try:
            async for chunk in generator_func(*args):
                accumulated += chunk
                now = time.time()
                # Only edit if enough time has passed to avoid rate limits
                if now - last_edit_time >= edit_interval:
                    display_text = accumulated or "..."
                    try:
                        await self.client.post(
                            f"{self.base_url}/editMessageText",
                            json={
                                "chat_id": chat_id,
                                "message_id": message_id,
                                "text": display_text[:4000],
                            },
                        )
                        last_edit_time = now
                    except Exception as edit_err:
                        logger.debug("Streaming edit error: %s", edit_err)
        except Exception as gen_err:
            logger.error("Streaming generator error: %s", gen_err)

        # Final edit with complete text
        final_text = accumulated.strip() or "(empty response)"
        try:
            await self.client.post(
                f"{self.base_url}/editMessageText",
                json={
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "text": final_text[:4000],
                },
            )
        except Exception as e:
            logger.error("Streaming final edit error: %s", e)

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

    async def _send_tts_reply(self, chat_id: int, text: str):
        """Background task: synthesize text and send as voice message."""
        try:
            from core.tts import text_to_speech
            audio_path = await text_to_speech(text[:1500])
            if audio_path:
                await self._send_voice(chat_id, audio_path)
                # Cleanup
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass
        except Exception as e:
            logger.debug("TTS reply failed: %s", e)

    async def _send_voice(self, chat_id: int, audio_path: str):
        """Send a voice message to Telegram."""
        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            files = {"voice": ("voice.ogg", audio_data, "audio/ogg")}
            data = {"chat_id": str(chat_id)}
            await self.client.post(
                f"{self.base_url}/sendVoice",
                data=data,
                files=files,
                timeout=30,
            )
        except Exception as e:
            logger.error("Telegram send voice error: %s", e)
            await self._send(chat_id, "(Voice send failed)")

    async def send_message(self, text: str):
        """Send a message to the master."""
        await self._send(self.master_id, text)

    async def stop(self):
        self.running = False
        try:
            await self.client.aclose()
        except Exception:
            pass
