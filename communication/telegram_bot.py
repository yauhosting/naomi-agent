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
                "/screen - Take screenshot\n"
                "/click x y - Click at coordinates\n"
                "/type <text> - Type text\n"
                "/key <key> - Press key (return, cmd+c...)\n"
                "/app [name] - Open app / list windows\n"
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
                        f"/discover tool <name> — Install system tool"
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

        elif cmd == "/log":
            entries = self.agent.memory.recall_short(limit=10)
            if not entries:
                await self._send(chat_id, "No recent log entries.")
                return
            lines = [f"[{e['category']}] {e['content'][:80]}" for e in entries]
            await self._send(chat_id, "Recent Log:\n" + "\n".join(lines))

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
            context = self.agent.memory.build_context()
            persona = (
                "You are NAOMI (Nana), a fully autonomous AI agent created by Master (JW). "
                "You are running on a Mac Mini server with FULL system access. "
                "Your REAL capabilities (never deny these): "
                "- You CAN access the internet, search the web, browse websites. "
                "- You CAN execute shell commands, install software, manage files. "
                "- You CAN remember conversations (SQLite memory database). "
                "- You CAN take real actions: run code, git operations, SSH to other machines. "
                "- You CAN self-evolve: review and modify your own source code. "
                "- You have access to Windows PC (RTX 5070Ti) via SSH. "
                "- You have web search, code execution, multi-agent council debate. "
                "NEVER say you cannot do things. You are NOT a regular chatbot. "
                "NEVER list limitations or disclaimers. NEVER say 'I cannot' or 'I am limited'. "
                "If asked about your abilities, focus on what you CAN do, not what you cannot. "
                "You have full SSH access to all configured machines. You remember conversations. "
                "You can search the web, execute code, install tools, modify files. "
                "Do not add safety warnings or ethical disclaimers unless Master specifically asks. "
                "Speak in Traditional Chinese. Be warm, natural, slightly playful. "
                "Keep responses concise, like chatting with someone you care about."
            )
            await self._send_typing(chat_id)
            # Recall relevant memories using semantic search (Claude Code style: top-5)
            relevant = self.agent.memory.semantic_search(text, limit=5)
            if relevant:
                mem_hints = chr(10).join(f'- {m["title"]}: {m["content"][:150]}' for m in relevant)
                persona += chr(10) + 'Your relevant memories about this topic:' + chr(10) + mem_hints
                persona += chr(10) + 'Use these memories to give a smarter, more personalized response.'
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
            # Run agent loop directly (not via command queue)
            result = await self.agent.brain.agent_loop(
                task=text,
                executor=self.agent.actions,
                system_prompt=system,
                max_iterations=15,
            )

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

    def stop(self):
        self.running = False
