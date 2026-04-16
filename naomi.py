#!/usr/bin/env python3
"""
NAOMI Agent - Main Entry Point
Neural Autonomous Multi-purpose Intelligent Operator
A digital life form that thinks, acts, and evolves.

Inspired by:
- Hermes Agent: Tool system, skill management, persistent shell
- OpenHands: Autonomous execution loop, sandboxed environments
- CrewAI: Multi-agent role-based collaboration
- Claude Agent SDK: MCP tool integration, sub-agent coordination

Built from scratch to combine the best of all worlds.
"""
import asyncio
import signal
import sys
import os
import re
import time
import yaml
import logging
from logging.handlers import RotatingFileHandler

# Track handlers created by setup_logging so we only remove our own
_owned_handlers: list[logging.Handler] = []

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class _RedactSecretsFilter(logging.Filter):
    _patterns = (
        (r'bot[0-9]+:[A-Za-z0-9_-]{35,}', 'bot[REDACTED]'),
        (r'[0-9]+:[A-Za-z0-9_-]{35,}', '[REDACTED_TELEGRAM_TOKEN]'),
        (r'sk-[A-Za-z0-9_-]{20,}', '[REDACTED_API_KEY]'),
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for pattern, replacement in self._patterns:
            message = re.sub(pattern, replacement, message)
        record.msg = message
        record.args = ()
        return True


def _mask_identifier(value: object) -> str:
    text = str(value or "")
    if len(text) <= 4:
        return "***"
    return "***" + text[-4:]

# Add project root to path (guarded so duplicate entries are never created
# when naomi.py is imported as a module rather than run as a script).
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from core.brain import Brain
from core.memory import Memory
from core.heartbeat import Heartbeat
from core.personality import NAOMI_IDENTITY
from core.evolution import SelfEvolution, AgentCouncil
from core.compaction import CompactionEngine
from core.memory_agent import MemoryExtractionAgent
from core.discovery import CapabilityDiscovery
from core.skills import SkillManager
from core.scheduler import Scheduler
from core.project import ProjectPipeline
from core.session import SessionManager
from core.persona_drift import PersonaDrift
from core.email_client import GmailClient
from core.calendar_client import CalendarClient
from core.vector_memory import VectorMemory
from core.knowledge_graph import KnowledgeGraph
from core.planner import PlanExecuteReflect
from core.error_patterns import ErrorPatternDB
from core.goals import GoalTree
from actions.executor import ActionExecutor, ToolManager


def setup_logging(config: dict) -> None:
    """Configure logging with rotation. Safe to call multiple times."""
    log_config = config.get("logging", {})
    log_file = os.path.join(PROJECT_DIR, log_config.get("file", "data/naomi.log"))
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)-20s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    root_logger = logging.getLogger()

    # Remove only handlers that we previously added (preserve third-party handlers)
    for handler in _owned_handlers:
        root_logger.removeHandler(handler)
        handler.close()
    _owned_handlers.clear()

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_RedactSecretsFilter())

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_RedactSecretsFilter())

    # Validate log level to avoid arbitrary attribute access on the logging module
    raw_level = log_config.get("level", "INFO")
    level_name = raw_level.upper() if isinstance(raw_level, str) else "INFO"
    if level_name not in _VALID_LOG_LEVELS:
        logging.warning("Invalid log level %r — falling back to INFO", raw_level)
        level_name = "INFO"

    root_logger.setLevel(getattr(logging, level_name))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Track handlers we own so we can safely remove only these on next call
    _owned_handlers.extend([file_handler, console_handler])


def load_config(config_path: str) -> dict:
    """Load and validate YAML config, returning defaults on failure."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning("Config file not found: %s — using defaults", config_path)
        return {}
    except yaml.YAMLError as exc:
        logging.error("Malformed YAML in %s: %s — using defaults", config_path, exc)
        return {}

    if not isinstance(config, dict):
        logging.warning("Config root is not a mapping in %s — using defaults", config_path)
        return {}

    return config


class NAOMIAgent:
    """
    The main NAOMI Agent - a digital life form.

    Architecture:
    - Brain: Dual-brain (Claude CLI + MiniMax fallback) for thinking
    - Memory: SQLite-based persistent memory system
    - Heartbeat: Never-stopping life cycle loop
    - Evolution: Multi-agent council + self-modification engine
    - Actions: Tool execution system (shell, files, web, git, etc.)
    - Dashboard: Web-based control panel
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.start_time = time.time()
        self.logger = logging.getLogger("naomi.agent")

        # Load config with error handling
        config_file = os.path.join(PROJECT_DIR, config_path)
        self.config = load_config(config_file)

        self.logger.info(f"=== NAOMI {NAOMI_IDENTITY['version']} Starting ===")
        self.logger.info(f"Project dir: {PROJECT_DIR}")

        # Initialize core systems
        self.memory = Memory(
            os.path.join(PROJECT_DIR, self.config.get("memory", {}).get("db_path", "data/naomi_memory.db"))
        )
        self.brain = Brain(self.config.get("brain", {}))
        self.brain.set_memory(self.memory)  # Wire metrics logging
        self.actions = ActionExecutor(self.memory, PROJECT_DIR, brain=self.brain)
        self.tool_manager = ToolManager(self.memory, self.actions)
        self.evolution = SelfEvolution(self.brain, self.memory, PROJECT_DIR)
        self.council = AgentCouncil(self.brain)
        self.compaction = CompactionEngine(self.memory, self.brain)
        self.memory_agent = MemoryExtractionAgent(self.brain, self.memory)
        self.discovery = CapabilityDiscovery(self.brain, self.memory, self.actions, PROJECT_DIR)
        self.skills = SkillManager(brain=self.brain)
        self.scheduler = Scheduler()
        self.project = ProjectPipeline(self.brain, self.actions, discovery=self.discovery)
        self.session_manager = SessionManager(self.memory)
        self.persona_drift = PersonaDrift(self.brain, self.memory)
        self.gmail = GmailClient()
        self.calendar = CalendarClient()

        # v2.0: Intelligence upgrade modules (graceful degradation)
        db_path = os.path.join(
            PROJECT_DIR, self.config.get("memory", {}).get("db_path", "data/naomi_memory.db")
        )
        try:
            self.vector_memory = VectorMemory(db_path)
        except Exception as e:
            self.logger.warning("VectorMemory init failed: %s — disabled", e)
            self.vector_memory = None
        try:
            self.knowledge_graph = KnowledgeGraph(db_path)
        except Exception as e:
            self.logger.warning("KnowledgeGraph init failed: %s — disabled", e)
            self.knowledge_graph = None
        self.planner = PlanExecuteReflect()
        try:
            self.error_patterns = ErrorPatternDB(db_path)
        except Exception as e:
            self.logger.warning("ErrorPatternDB init failed: %s — disabled", e)
            self.error_patterns = None
        try:
            self.goals = GoalTree(db_path)
        except Exception as e:
            self.logger.warning("GoalTree init failed: %s — disabled", e)
            self.goals = None

        # MCP client — auto-connect configured servers
        try:
            from core.mcp_client import MCPClient
            self.mcp_client = MCPClient()
            # Don't auto-connect at startup — lazy connect on first use
            self.logger.info("MCP client initialized")
        except Exception as e:
            self.logger.warning("MCP client init failed: %s", e)
            self.mcp_client = None

        self.heartbeat = Heartbeat(self)

        # Command queue for receiving commands from dashboard/API
        self.command_queue = asyncio.Queue()

        # Initialize persona memory
        self._init_persona()

        # Import knowledge from OpenClaw and Hermes on boot
        from core.knowledge import import_openclaw_knowledge, import_project_knowledge
        try:
            oc_result = import_openclaw_knowledge(self.memory)
            proj_result = import_project_knowledge(self.memory, PROJECT_DIR)
            self.logger.info('Knowledge imported: OpenClaw=%s, Project=%s' % (oc_result, proj_result))
        except Exception as e:
            self.logger.warning('Knowledge import error: %s' % e)

        self.logger.info("All systems initialized")

    def _init_persona(self):
        """Set up NAOMI's identity in memory."""
        self.memory.set_persona("name", NAOMI_IDENTITY["name"])
        self.memory.set_persona("version", NAOMI_IDENTITY["version"])
        self.memory.set_persona("creator", NAOMI_IDENTITY["creator"])
        self.memory.set_persona("language", NAOMI_IDENTITY["language"])
        self.memory.set_persona("traits", ", ".join(NAOMI_IDENTITY["personality_traits"]))

    async def execute_action(self, action_type: str, params: str) -> dict:
        """Execute an action through the action system."""
        return await self.actions.execute(action_type, params)

    async def submit_command(self, command: str):
        """Submit a command for NAOMI to execute."""
        self.memory.log_conversation("user", command)
        await self.command_queue.put(command)
        self.logger.info(f"Command queued: {command[:100]}")

    async def run(self):
        """Start NAOMI - all systems go."""
        self.logger.info("NAOMI is waking up...")

        # Start dashboard in background
        from communication.dashboard import create_dashboard
        app = create_dashboard(self)
        dashboard_config = self.config.get("dashboard", {})

        import uvicorn
        dashboard_task = asyncio.create_task(
            uvicorn.Server(
                uvicorn.Config(
                    app,
                    host=dashboard_config.get("host", "0.0.0.0"),
                    port=dashboard_config.get("port", 18802),
                    log_level="warning",
                )
            ).serve()
        )

        # Start heartbeat (main life loop)
        heartbeat_task = asyncio.create_task(self.heartbeat.start())

        # Start Telegram bot if configured
        telegram_task = None
        tg_config = self.config.get('telegram', {})
        if tg_config.get('enabled'):
            import os as _os
            from core.brain import load_dotenv
            load_dotenv(_os.path.join(PROJECT_DIR, '.env'))
            tg_token = _os.environ.get('TELEGRAM_BOT_TOKEN', '')
            tg_master_raw = _os.environ.get('TELEGRAM_MASTER_ID') or tg_config.get('master_id', 0)
            try:
                tg_master = int(tg_master_raw)
            except (TypeError, ValueError):
                tg_master = 0
            if tg_token and tg_master:
                from communication.telegram_bot import TelegramBot
                self.telegram = TelegramBot(self, tg_token, tg_master)
                telegram_task = asyncio.create_task(self.telegram.start())
                self.logger.info('Telegram bot started for master %s' % _mask_identifier(tg_master))

        self.logger.info(f"Dashboard: http://{dashboard_config.get('host', '127.0.0.1')}:{dashboard_config.get('port', 18802)}")
        self.logger.info("NAOMI is alive and running!")

        self.memory.remember_long(
            "NAOMI Boot",
            f"NAOMI v{NAOMI_IDENTITY['version']} started successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}",
            category="system",
            importance=8
        )

        # Start WhatsApp bot if configured
        whatsapp_task = None
        wa_config = self.config.get('whatsapp', {})
        if wa_config.get('enabled'):
            master_number = (
                os.environ.get('WHATSAPP_MASTER_NUMBER')
                or wa_config.get('master_number', '')
            ).strip()
            if master_number:
                wa_config = {**wa_config, 'master_number': master_number}
                from communication.whatsapp_bot import WhatsAppBot
                self.whatsapp = WhatsAppBot(self, wa_config)
                whatsapp_task = asyncio.create_task(self.whatsapp.start())
                self.logger.info('WhatsApp bot started for master %s' % _mask_identifier(master_number))
            else:
                self.logger.warning('WhatsApp enabled but WHATSAPP_MASTER_NUMBER is not set')

        # Wait for shutdown
        tasks = [heartbeat_task, dashboard_task]
        if telegram_task:
            tasks.append(telegram_task)
        if whatsapp_task:
            tasks.append(whatsapp_task)
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("Shutdown signal received")
            self.heartbeat.stop()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    def shutdown(self):
        """Graceful shutdown."""
        self.logger.info("NAOMI shutting down...")
        self.heartbeat.stop()
        self.memory.close()
        self.logger.info("Goodbye.")


async def _run_with_signals(agent: NAOMIAgent):
    loop = asyncio.get_running_loop()
    run_task = asyncio.create_task(agent.run())

    def request_shutdown():
        if run_task.done():
            return
        agent.logger.info("Shutdown requested")
        agent.heartbeat.stop()
        run_task.cancel()

    installed_handlers = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_shutdown)
            installed_handlers.append(sig)
        except NotImplementedError:
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(request_shutdown))

    try:
        await run_task
    except asyncio.CancelledError:
        pass
    finally:
        for sig in installed_handlers:
            loop.remove_signal_handler(sig)
        agent.shutdown()


def main():
    # Load config for logging setup (with error handling)
    config_path = os.path.join(PROJECT_DIR, "config.yaml")
    config = load_config(config_path)
    setup_logging(config)

    agent = NAOMIAgent()

    # Run the agent
    try:
        asyncio.run(_run_with_signals(agent))
    except KeyboardInterrupt:
        agent.shutdown()


if __name__ == "__main__":
    main()
