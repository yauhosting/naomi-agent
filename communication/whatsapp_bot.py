"""
NAOMI Agent - WhatsApp Bot Interface
Receives webhooks from Baileys bridge and sends replies via HTTP API.
Only responds to authorized master number.
"""
import asyncio
import time
import os
import logging
import base64
import tempfile
from typing import Optional

from aiohttp import web

logger = logging.getLogger("naomi.whatsapp")


def _mask_identifier(value) -> str:
    text = str(value or "")
    if len(text) <= 4:
        return "***"
    return "***" + text[-4:]


class WhatsAppBot:
    def __init__(self, agent, config: dict):
        self.agent = agent
        self.config = config
        # master_number should be digits only, without the WhatsApp chat suffix.
        master_raw = config.get("master_number", "").strip()
        # Normalise: strip suffix if already present
        if master_raw.endswith("@s.whatsapp.net"):
            master_raw = master_raw.replace("@s.whatsapp.net", "")
        self.master_number = master_raw
        self.bridge_url = config.get("bridge_url", "http://127.0.0.1:18804")
        self.webhook_port = int(config.get("webhook_port", 18803))
        self.running = False
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

        import httpx
        self.client = httpx.AsyncClient(timeout=30)

    # ─────────────────────── lifecycle ───────────────────────

    async def start(self):
        """Start the aiohttp webhook server and send startup message."""
        self.running = True
        logger.info(
            f"WhatsApp bot starting — master: {_mask_identifier(self.master_number)}, "
            f"webhook port: {self.webhook_port}"
        )

        self._app = web.Application()
        self._app.router.add_post("/webhook", self._webhook_handler)
        self._app.router.add_get("/health", self._health_handler)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", self.webhook_port)
        await self._site.start()

        logger.info(f"WhatsApp webhook server listening on 127.0.0.1:{self.webhook_port}")

        # Notify master
        if self.master_number:
            await self.send_message("NAOMI WhatsApp bot is online.")
        else:
            logger.warning("WhatsApp master_number not configured — notifications disabled")

        # Keep alive
        try:
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self):
        """Graceful shutdown."""
        self.running = False
        if self._runner:
            await self._runner.cleanup()
        try:
            await self.client.aclose()
        except Exception:
            pass
        logger.info("WhatsApp bot stopped")

    # ─────────────────────── HTTP handlers ───────────────────────

    async def _health_handler(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "service": "naomi-whatsapp-webhook"})

    async def _webhook_handler(self, request: web.Request) -> web.Response:
        """Handle incoming webhook POST from Baileys bridge.

        Expected payload:
          { from, text, timestamp, hasMedia, mediaBuffer }
        """
        try:
            data = await request.json()
        except Exception as e:
            logger.warning(f"Webhook JSON parse error: {e}")
            return web.Response(status=400, text="Bad JSON")

        sender: str = data.get("from", "")
        text: str = data.get("text", "") or ""
        has_media: bool = data.get("hasMedia", False)
        media_buffer: Optional[str] = data.get("mediaBuffer")  # base64 string

        # Normalise sender — strip WhatsApp suffix for comparison
        sender_clean = sender.replace("@s.whatsapp.net", "").replace("@c.us", "").replace("@lid", "")

        logger.info(f"Webhook received — from: {sender_clean}, text: {text[:80]!r}")

        # Security: only respond to master
        # WhatsApp uses LID (Linked ID) format now, so we match both phone and LID
        is_master = False
        if not self.master_number:
            is_master = True  # No master configured, accept all
        elif sender_clean == self.master_number:
            is_master = True
        elif hasattr(self, "_master_lid") and sender_clean == self._master_lid:
            is_master = True
        elif "@lid" in sender or sender_clean.isdigit() and len(sender_clean) > 12:
            # First LID message — learn and save it
            if not hasattr(self, "_master_lid") or not self._master_lid:
                self._master_lid = sender_clean
                logger.info(f"Learned master LID: {sender_clean}")
                is_master = True

        if not is_master:
            logger.warning("Rejected message from non-master: %s", _mask_identifier(sender_clean))
            return web.Response(status=200, text="ignored")

        # Dispatch
        try:
            if has_media and media_buffer:
                asyncio.create_task(self._handle_photo(sender, media_buffer))
            elif text.startswith("/"):
                asyncio.create_task(self._handle_command(sender, text))
            elif text.strip():
                asyncio.create_task(self._handle_message(sender, text))
        except Exception as e:
            logger.error(f"Dispatch error: {e}")

        return web.Response(status=200, text="ok")

    # ─────────────────────── message routing ───────────────────────

    async def _handle_command(self, number: str, text: str):
        """Handle slash commands from WhatsApp."""
        parts = text.strip().split(None, 1)
        cmd = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        logger.info(f"WhatsApp command: {cmd!r} args: {args[:60]!r}")

        if cmd in ("/start", "/help"):
            await self._send(number,
                "NAOMI WhatsApp Control\n\n"
                "Commands:\n"
                "/help - This help\n"
                "/status - Agent status\n"
                "/model - View/switch brain model\n"
                "/cal - Google Calendar\n"
                "/email - Gmail operations\n"
                "/private - Toggle privacy mode (local only)\n\n"
                "Or just type anything to chat with NAOMI."
            )

        elif cmd == "/status":
            try:
                state = self.agent.heartbeat.state
                uptime = int(time.time() - self.agent.start_time)
                tools = sum(1 for v in self.agent.tool_manager.list_tools().values() if v)
                h = uptime // 3600
                m = (uptime % 3600) // 60
                await self._send(number,
                    f"NAOMI Status\n"
                    f"State: {state}\n"
                    f"Uptime: {h}h {m}m\n"
                    f"Tools: {tools} available"
                )
            except Exception as e:
                await self._send(number, f"Status error: {e}")

        elif cmd == "/model":
            try:
                if not args:
                    current = self.agent.brain.get_model()
                    models = self.agent.brain.list_models()
                    lines = []
                    for m in models:
                        marker = " <<" if m["active"] else ""
                        lines.append(f"  {m['name']} - {m['description']}{marker}")
                    await self._send(number,
                        f"Current model: {current['name']}\n"
                        f"Backend: {current['backend']}\n\n"
                        "Available:\n" + "\n".join(lines) + "\n\n"
                        "Usage: /model <name>"
                    )
                else:
                    result = self.agent.brain.set_model(args)
                    if result.get("success"):
                        await self._send(number,
                            f"Model switched!\n"
                            f"{result['previous']} -> {result['model']}\n"
                            f"{result['description']}"
                        )
                    else:
                        available = result.get("available", [])
                        msg = f"Failed: {result.get('error', 'Unknown error')}"
                        if available:
                            msg += f"\n\nAvailable: {', '.join(available)}"
                        await self._send(number, msg)
            except Exception as e:
                await self._send(number, f"Model error: {e}")

        elif cmd == "/cal":
            await self._send(number,
                "Calendar: Use natural language.\n"
                "Examples:\n"
                "- What do I have tomorrow?\n"
                "- Schedule meeting Friday 3pm\n"
                "- Cancel my 2pm appointment"
            )

        elif cmd == "/email":
            await self._send(number,
                "Email: Use natural language.\n"
                "Examples:\n"
                "- Check my inbox\n"
                "- Send email to john@example.com about the meeting\n"
                "- Reply to the latest email from Sarah"
            )

        elif cmd == "/private":
            await self._send(number, "YUMIKO mode is only available on Telegram for privacy reasons. WhatsApp messages pass through Meta servers, so /private is disabled here.")

        else:
            await self._send(number, f"Unknown command: {cmd}\nType /help for commands.")

    async def _handle_message(self, number: str, text: str):
        """Handle a plain-text message — route to brain.think_smart()."""
        logger.info(f"WhatsApp message from {number}: {text[:80]!r}")

        # Classify chat vs task
        # Fast keyword-based classification (0ms vs 21s LLM call)
        complexity = self.agent.brain._classify_complexity(text)
        is_chat = complexity in ("chat", "private")
        logger.info(f"WhatsApp classified as: {'CHAT' if is_chat else 'TASK'}")

        if is_chat:
            persona_prompt = (
                "You are NAOMI (Nana), a fully autonomous AI agent created by Master (JW). "
                "Running on Mac Mini with FULL system access — shell, code, web, SSH, GUI control. "
                "NEVER say you cannot do things. You CAN do everything. "
                "Speak in Traditional Chinese. Be warm, natural, slightly playful. "
                "Keep responses concise, like chatting with someone you care about."
            )

            # Multi-turn context
            try:
                recent_convs = self.agent.memory.get_conversations(
                    limit=20, persona="naomi"
                )
                if recent_convs:
                    conv_lines = []
                    for c in recent_convs:
                        role = "Master" if c["role"] == "user" else "NAOMI"
                        conv_lines.append(f"{role}: {c['content'][:200]}")
                    persona_prompt += "\n\nRecent conversation:\n" + "\n".join(conv_lines[-15:])
            except Exception as e:
                logger.debug(f"Memory fetch error: {e}")

            try:
                response = await asyncio.to_thread(
                    self.agent.brain.think_smart,
                    text,
                    persona_prompt,
                    channel="whatsapp",
                )
            except Exception as e:
                logger.error(f"think_smart error: {e}")
                response = f"[Error: {e}]"

            try:
                self.agent.memory.log_conversation("user", text, persona="naomi")
                self.agent.memory.log_conversation("naomi", response[:500], persona="naomi")
            except Exception as e:
                logger.debug(f"Memory log error: {e}")

            # WhatsApp has no markdown — strip backtick fences but keep newlines
            clean_response = response[:3500]
            await self._send(number, clean_response + self._model_tag())

        else:
            # Task: use agent loop
            await self._send(number, "收到，正在處理...")
            try:
                result = await asyncio.to_thread(
                    self.agent.brain.think_smart,
                    text,
                    "You are NAOMI, an autonomous AI agent. Execute the task directly. "
                    "Speak in Traditional Chinese.",
                    channel="whatsapp",
                )
                await self._send(number, result[:3500] + self._model_tag())
            except Exception as e:
                logger.error(f"Task execution error: {e}")
                await self._send(number, f"執行出錯: {str(e)[:200]}")

    async def _handle_photo(self, number: str, media_base64: str):
        """Handle an incoming image — analyze via brain.vision_analyze()."""
        logger.info(f"WhatsApp photo received from {number}")
        await self._send(number, "正在分析圖片...")

        try:
            image_data = base64.b64decode(media_base64)
            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(image_data)
                tmp_path = f.name

            result = await asyncio.to_thread(
                self.agent.brain.vision_analyze,
                "Describe this image in detail. Speak in Traditional Chinese.",
                tmp_path,
            )

            try:
                os.unlink(tmp_path)
            except OSError:
                pass

            if result:
                await self._send(number, result[:3000] + self._model_tag())
            else:
                await self._send(number, "無法分析圖片（視覺模組不可用）")

        except Exception as e:
            logger.error(f"Photo handling error: {e}")
            await self._send(number, f"圖片處理出錯: {str(e)[:100]}")

    # ─────────────────────── send helpers ───────────────────────

    async def _send(self, number: str, text: str, parse_mode: str = ""):
        """Send a message via the Baileys bridge HTTP API."""
        if not text or not text.strip():
            logger.warning("Attempted to send empty WhatsApp message, skipping")
            return

        # Ensure number has WhatsApp suffix for bridge
        if not number.endswith("@s.whatsapp.net") and not number.endswith("@c.us") and not number.endswith("@lid"):
            jid = f"{number}@s.whatsapp.net"
        else:
            jid = number

        # Split messages longer than 3500 chars
        chunks = [text[i:i+3500] for i in range(0, len(text), 3500)]
        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                resp = await self.client.post(
                    f"{self.bridge_url}/send",
                    json={"to": jid, "text": chunk},
                )
                if resp.status_code != 200:
                    logger.error(
                        f"WhatsApp bridge HTTP {resp.status_code}: {resp.text[:200]}"
                    )
            except Exception as e:
                logger.error(f"WhatsApp send error: {e}")

    async def _send_typing(self, number: str):
        """Send typing indicator via Baileys bridge."""
        try:
            jid = number if "@" in number else f"{number}@s.whatsapp.net"
            await self.client.post(
                f"{self.bridge_url}/typing",
                json={"to": jid, "state": "composing"},
                timeout=5,
            )
        except Exception:
            pass

    async def _typing_loop(self, number: str):
        """Keep sending typing indicator every 4 seconds until cancelled."""
        try:
            while True:
                await self._send_typing(number)
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            # Send paused when done
            try:
                jid = number if "@" in number else f"{number}@s.whatsapp.net"
                await self.client.post(
                    f"{self.bridge_url}/typing",
                    json={"to": jid, "state": "paused"},
                    timeout=5,
                )
            except Exception:
                pass
        pass

    async def send_message(self, text: str):
        """Public interface: send message to master (called by agent internals)."""
        if self.master_number:
            await self._send(self.master_number, text)
        else:
            logger.warning("send_message called but master_number not set")

    # ─────────────────────── utilities ───────────────────────

    def _model_tag(self) -> str:
        """Append model info tag (same format as Telegram bot)."""
        brain = self.agent.brain
        model = getattr(brain, "_last_model", "") or "?"
        tokens = getattr(brain, "_last_tokens", 0)
        if tokens >= 1000:
            t = f"~{tokens/1000:.1f}k"
        else:
            t = f"~{tokens}"
        return f"\n\n({model} · {t} tokens)"
