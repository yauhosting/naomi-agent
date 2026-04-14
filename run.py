#!/usr/bin/env python3
"""
NAOMI Agent Launcher — Hot Reload + Process Supervisor
Watches .py files for changes → auto-restarts.
Restarts on crash with exponential backoff.
"""
import os
import sys
import time
import signal
import subprocess
import logging
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print(
        "Missing required dependency: watchdog\n"
        "Install it with: pip install watchdog",
        file=sys.stderr,
    )
    sys.exit(1)

PROJECT_DIR = Path(__file__).parent
LOG_FMT = "[%(asctime)s] %(levelname)-7s %(message)s"
logging.basicConfig(format=LOG_FMT, datefmt="%H:%M:%S", level=logging.INFO)
logger = logging.getLogger("launcher")

# Backoff limits
MIN_BACKOFF = 2
MAX_BACKOFF = 60
# If the process runs longer than this (seconds), reset backoff
STABLE_THRESHOLD = 30


class CodeChangeHandler(FileSystemEventHandler):
    """Detect .py file changes and signal reload."""

    def __init__(self):
        self.changed = False
        self.last_event = 0.0

    def on_modified(self, event):
        if not event.src_path.endswith(".py"):
            return
        # Debounce: ignore events within 1 second
        now = time.time()
        if now - self.last_event < 1.0:
            return
        self.last_event = now
        rel = os.path.relpath(event.src_path, PROJECT_DIR)
        # Skip data/ and __pycache__
        if rel.startswith("data/") or "__pycache__" in rel:
            return
        logger.info("File changed: %s -> scheduling reload", rel)
        self.changed = True


def start_agent() -> subprocess.Popen:
    """Start naomi.py as a subprocess."""
    logger.info("Starting NAOMI agent...")
    return subprocess.Popen(
        [sys.executable, str(PROJECT_DIR / "naomi.py")],
        cwd=str(PROJECT_DIR),
    )


def _terminate_process(proc: subprocess.Popen) -> None:
    """Gracefully terminate a process, killing it if it doesn't stop in time."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main():
    handler = CodeChangeHandler()
    observer = Observer()
    # Watch core/, communication/, actions/, senses/, and root .py files
    for watch_dir in ["core", "communication", "actions", "senses", "navigation"]:
        path = PROJECT_DIR / watch_dir
        if path.exists():
            observer.schedule(handler, str(path), recursive=True)
    observer.schedule(handler, str(PROJECT_DIR), recursive=False)
    observer.start()

    # Use a mutable container so the shutdown closure always sees the current process
    proc_holder = [start_agent()]
    backoff = MIN_BACKOFF
    consecutive_crashes = 0
    shutting_down = False

    def shutdown(signum, _frame):
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        logger.info("Shutdown signal received (signal %d)", signum)
        _terminate_process(proc_holder[0])
        observer.stop()
        observer.join()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("Launcher ready — watching for file changes + supervising process")

    stable_since = time.time()

    try:
        while True:
            # Check for code changes -> hot reload
            if handler.changed:
                handler.changed = False
                logger.info("Hot reload: restarting NAOMI...")
                _terminate_process(proc_holder[0])
                proc_holder[0] = start_agent()
                backoff = MIN_BACKOFF
                consecutive_crashes = 0
                stable_since = time.time()
                continue

            # Check if process crashed
            retcode = proc_holder[0].poll()
            if retcode is not None:
                if retcode == 0:
                    logger.info("NAOMI exited cleanly (code 0)")
                    break

                elapsed = time.time() - stable_since
                consecutive_crashes += 1
                logger.warning(
                    "NAOMI crashed (exit code %d, crash #%d, uptime %.1fs)",
                    retcode,
                    consecutive_crashes,
                    elapsed,
                )

                # Reset backoff if the process was stable long enough
                if elapsed >= STABLE_THRESHOLD:
                    backoff = MIN_BACKOFF
                    consecutive_crashes = 1

                logger.info("Restarting in %d seconds (backoff)...", backoff)
                time.sleep(backoff)

                # Exponential backoff capped at MAX_BACKOFF
                backoff = min(backoff * 2, MAX_BACKOFF)

                proc_holder[0] = start_agent()
                stable_since = time.time()
                continue

            time.sleep(0.5)
    except SystemExit:
        raise
    except Exception:
        logger.exception("Launcher encountered an unexpected error")
        _terminate_process(proc_holder[0])
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()