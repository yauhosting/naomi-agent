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
import time
import yaml
import logging
from logging.handlers import RotatingFileHandler

# Track handlers created by setup_logging so we only remove our own
_owned_handlers: list[logging.Handler] = []

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

# Add project root to path (guarded so duplicate entries are never created
# when naomi.py is imported as a module rather than run as a script).
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from core.brain import Brain
from core.memory import Memory
from core.heartbeat import Heartbeat
from core.personality import NAOMI_IDENTITY, SYSTEM_PROMPT
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

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Validate log level to avoid arbitrary attribute access on the logging module
    raw_level = log_config.get("level", "INFO")
    level_name = raw_level.upper() if isinstance(raw_level, str) else "INFO"
    if level_name not in _VALID_LOG_LEVELS:
        logging.warning("Invalid log level %r — falling back to INFO", raw_level)
        level_name = "INFO"

    root_logger.setLevel(getattr(logging, level_name))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

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


def get_default_config() -> dict:
    """Return default configuration values."""
    return {
        "logging": {
            "level": "INFO",
            "file": "data/naomi.log"
        },
        "brain": {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 4096
        },
        "memory": {
            "max_size_mb": 100,
            "retention_days": 30
        },
        "heartbeat": {
            "interval_seconds": 60
        },
        "evolution": {
            "enabled": True,
            "interval_hours": 24
        },
        "compaction": {
            "enabled": True,
            "threshold_mb": 80
        },
        "scheduler": {
            "max_concurrent_tasks": 5
        },
        "session": {
            "auto_save_interval_seconds": 300
        },
        "email": {
            "enabled": False
        },
        "calendar": {
            "enabled": False
        }
    }


def merge_config(defaults: dict, overrides: dict) -> dict:
    """Deep merge overrides into defaults, returning a new dict."""
    result = defaults.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


class NAOMI:
    """Main NAOMI agent class orchestrating all subsystems."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("NAOMI")
        self.running = False
        self.shutdown_event = asyncio.Event()

        # Initialize core systems
        self.logger.info("Initializing NAOMI core systems...")
        self.memory = Memory(config.get("memory", {}))
        self.brain = Brain(config.get("brain", {}), self.memory)
        self.heartbeat = Heartbeat(config.get("heartbeat", {}), self)
        self.skill_manager = SkillManager(self.memory)
        self.tool_manager = ToolManager()
        self.action_executor = ActionExecutor(self.brain, self.tool_manager, self.memory)
        self.scheduler = Scheduler(config.get("scheduler", {}))
        self.session_manager = SessionManager(config.get("session", {}))
        self.persona_drift = PersonaDrift(config.get("persona_drift", {}))

        # Initialize advanced systems
        self.evolution = SelfEvolution(config.get("evolution", {}), self) if config.get("evolution", {}).get("enabled", True) else None
        self.agent_council = AgentCouncil(config.get("council", {}), self) if config.get("council", {}).get("enabled", False) else None
        self.compaction = CompactionEngine(config.get("compaction", {}), self.memory) if config.get("compaction", {}).get("enabled", True) else None
        self.memory_agent = MemoryExtractionAgent(self.brain, self.memory)
        self.capability_discovery = CapabilityDiscovery(self)

        # Optional integrations
        self.email_client = GmailClient(config.get("email", {})) if config.get("email", {}).get("enabled", False) else None
        self.calendar_client = CalendarClient(config.get("calendar", {})) if config.get("calendar", {}).get("enabled", False) else None

        # Project pipeline
        self.project_pipeline = ProjectPipeline(self.brain, self.memory, self.scheduler)

        self.logger.info("NAOMI core systems initialized successfully")

    async def start(self) -> None:
        """Start NAOMI and all subsystems."""
        self.running = True
        self.logger.info("Starting NAOMI...")

        # Load previous session state
        await self.session_manager.load_session(self)

        # Start heartbeat
        if self.heartbeat:
            self.heartbeat.start()

        # Start scheduler
        self.scheduler.start()

        # Start evolution if enabled
        if self.evolution:
            self.evolution.start()

        # Discover capabilities
        await self.capability_discovery.discover()

        # Register signal handlers
        self._setup_signal_handlers()

        self.logger.info("NAOMI started successfully")
        self.logger.info("Identity: %s", NAOMI_IDENTITY)

        # Main event loop
        try:
            await self.run_interactive()
        except asyncio.CancelledError:
            self.logger.info("NAOMI cancelled")
        finally:
            await self.shutdown()

    def _setup_signal_handlers(self) -> None:
        """Setup handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info("Received signal %d, initiating shutdown...", signum)
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def run_interactive(self) -> None:
        """Run NAOMI in interactive mode, processing user input."""
        self.logger.info("Entering interactive mode (type 'exit' to quit, 'help' for commands)")

        while self.running and not self.shutdown_event.is_set():
            try:
                # Check for scheduled tasks
                task = self.scheduler.get_next_task()
                if task:
                    await self.process_task(task)
                    continue

                # Read user input with timeout to allow periodic checks
                try:
                    user_input = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, lambda: input("\n[You] ").strip()
                        ),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                if not user_input:
                    continue

                # Process commands
                if user_input.lower() in ('exit', 'quit', 'bye'):
                    self.logger.info("Exit command received")
                    break
                elif user_input.lower() == 'help':
                    await self.show_help()
                elif user_input.lower() == 'status':
                    await self.show_status()
                elif user_input.lower() == 'memory':
                    await self.show_memory()
                elif user_input.lower() == 'skills':
                    await self.show_skills()
                elif user_input.lower() == 'plan':
                    await self.project_pipeline.list_projects()
                elif user_input.lower().startswith('project '):
                    project_name = user_input[8:].strip()
                    await self.project_pipeline.show_project(project_name)
                elif user_input.lower() == 'think':
                    await self.evolution.analyze_and_evolve() if self.evolution else self.logger.warning("Evolution not enabled")
                elif user_input.lower() == 'compact':
                    await self.compaction.run() if self.compaction else self.logger.warning("Compaction not enabled")
                elif user_input.lower() == 'save':
                    await self.session_manager.save_session(self)
                else:
                    # Process as general query
                    await self.process_input(user_input)

            except EOFError:
                self.logger.info("EOF received, shutting down...")
                break
            except Exception as e:
                self.logger.error("Error in interactive loop: %s", e, exc_info=True)

    async def show_help(self) -> None:
        """Display available commands."""
        help_text = """
NAOMI Interactive Commands:
===========================
  help              - Show this help message
  status            - Show NAOMI status and statistics
  memory            - Show memory usage and recent memories
  skills            - List available skills
  plan              - List active projects
  project <name>    - Show details of a specific project
  think             - Trigger evolution/learning cycle
  compact           - Run memory compaction
  save              - Save current session state
  exit/quit/bye     - Exit NAOMI

Any other input will be processed as a general query or task.
"""
        print(help_text)

    async def show_status(self) -> None:
        """Display NAOMI status."""
        uptime = time.time() - getattr(self, '_start_time', time.time())
        status = f"""
NAOMI Status:
============
  Uptime:      {uptime:.1f} seconds
  Running:     {self.running}
  Memory:      {len(self.memory.memories)} memories stored
  Skills:      {self.skill_manager.count()} skills registered
  Tasks:       {self.scheduler.pending_count()} pending tasks
  Evolution:   {'Enabled' if self.evolution else 'Disabled'}
  Compaction:  {'Enabled' if self.compaction else 'Disabled'}
  Session:     {self.session_manager.current_session_id or 'None'}
"""
        print(status)

    async def show_memory(self) -> None:
        """Display memory statistics."""
        stats = self.memory.get_stats()
        print(f"""
Memory Statistics:
==================
  Total Memories: {stats.get('total', 0)}
  Memory Size:    {stats.get('size_mb', 0):.2f} MB
  Categories:     {', '.join(stats.get('categories', []))}
""")

    async def show_skills(self) -> None:
        """Display available skills."""
        skills = self.skill_manager.list_skills()
        if not skills:
            print("No skills registered.")
            return
        print("\nAvailable Skills:")
        print("=================")
        for skill in skills:
            print(f"  - {skill['name']}: {skill.get('description', 'No description')}")

    async def process_input(self, user_input: str) -> None:
        """Process user input and generate response."""
        self.logger.debug("Processing input: %s", user_input[:100])

        # Store in memory
        self.memory.add(f"User said: {user_input}", category="interaction")

        # Execute actions
        result = await self.action_executor.execute(user_input)

        # Display result
        if result.get('success'):
            response = result.get('response', 'Done.')
            print(f"\n[NAOMI] {response}")

            # Store response in memory
            self.memory.add(f"NAOMI responded: {response}", category="interaction")

            # Handle any actions taken
            actions = result.get('actions', [])
            if actions:
                self.logger.info("Executed %d actions", len(actions))
        else:
            error = result.get('error', 'Unknown error')
            print(f"\n[NAOMI] Error: {error}")
            self.logger.error("Action execution failed: %s", error)

    async def process_task(self, task: dict) -> None:
        """Process a scheduled task."""
        self.logger.info("Processing scheduled task: %s", task.get('name', 'unnamed'))

        try:
            task_type = task.get('type', 'general')
            if task_type == 'project':
                await self.project_pipeline.execute_project(task.get('project'))
            elif task_type == 'skill':
                await self.skill_manager.execute_skill(task.get('skill'), task.get('params', {}))
            else:
                await self.process_input(task.get('description', ''))
        except Exception as e:
            self.logger.error("Task execution failed: %s", e)

    async def shutdown(self) -> None:
        """Gracefully shutdown NAOMI and all subsystems."""
        if not self.running:
            return

        self.logger.info("Shutting down NAOMI...")
        self.running = False
        self.shutdown_event.set()

        # Stop heartbeat
        if self.heartbeat:
            self.heartbeat.stop()

        # Stop scheduler
        self.scheduler.stop()

        # Stop evolution
        if self.evolution:
            self.evolution.stop()

        # Run compaction if enabled
        if self.compaction:
            await self.compaction.run()

        # Save session state
        await self.session_manager.save_session(self)

        # Close optional integrations
        if self.email_client:
            await self.email_client.close()
        if self.calendar_client:
            await self.calendar_client.close()

        self.logger.info("NAOMI shutdown complete")


async def async_main(config: dict) -> int:
    """Async main entry point."""
    naomi = NAOMI(config)
    naomi._start_time = time.time()

    try:
        await naomi.start()
        return 0
    except Exception as e:
        logging.error("Fatal error in NAOMI: %s", e, exc_info=True)
        return 1


def main() -> int:
    """Main entry point."""
    # Determine config path
    config_path = os.environ.get('NAOMI_CONFIG', 'config/naomi.yaml')

    # Load configuration
    config = load_config(config_path)

    # Merge with defaults
    defaults = get_default_config()
    config = merge_config(defaults, config)

    # Setup logging
    setup_logging(config)

    # Log startup
    logger = logging.getLogger("NAOMI")
    logger.info("=" * 60)
    logger.info("NAOMI Agent starting...")
    logger.info("Version: 1.0.0")
    logger.info("Project directory: %s", PROJECT_DIR)
    logger.info("Config path: %s", config_path)
    logger.info("=" * 60)

    # Run async main
    try:
        return asyncio.run(async_main(config))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error("Unhandled exception: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())