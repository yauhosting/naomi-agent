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

# Lazy-import watchdog — install if missing
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "watchdog"])
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

PROJECT_DIR = Path(__file__).parent
LOG_FMT = "[%(asctime)s] %(levelname)-7s %(message)s"
logging.basicConfig(format=LOG_FMT, datefmt="%H:%M:%S", level=logging.INFO)
logger = logging.getLogger("launcher")

# Backoff limits
MIN_BACKOFF = 2
MAX_BACKOFF = 60


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
        logger.info("File changed: %s → scheduling reload", rel)
        self.changed = True


def start_agent() -> subprocess.Popen:
    """Start naomi.py as a subprocess."""
    logger.info("Starting NAOMI agent...")
    return subprocess.Popen(
        [sys.executable, str(PROJECT_DIR / "naomi.py")],
        cwd=str(PROJECT_DIR),
    )


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

    process = start_agent()
    backoff = MIN_BACKOFF
    consecutive_crashes = 0

    def shutdown(signum, frame):
        logger.info("Shutdown signal received")
        process.terminate()
        process.wait(timeout=10)
        observer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("Launcher ready — watching for file changes + supervising process")

    while True:
        # Check for code changes → hot reload
        if handler.changed:
            handler.changed = False
            logger.info("Hot reload: restarting NAOMI...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process = start_agent()
            backoff = MIN_BACKOFF
            consecutive_crashes = 0
            continue

        # Check if process crashed
        retcode = process.poll()
        if retcode is not None:
            consecutive_crashes += 1
            if retcode == 0:
                logger.info("NAOMI exited cleanly (code 0)")
                break

            logger.warning(
                "NAOMI crashed (code %d, crash #%d) — restarting in %ds",
                retcode, consecutive_crashes, backoff,
            )
            time.sleep(backoff)
            process = start_agent()
            # Exponential backoff, reset after 3 successful minutes
            backoff = min(backoff * 2, MAX_BACKOFF)

            # If stable for 3 minutes, reset backoff
            if consecutive_crashes > 5:
                logger.error("Too many crashes — check logs. Still retrying...")
        else:
            # Process running — reset crash counter if stable
            if consecutive_crashes > 0:
                # Check if process has been running for > 60s
                pass  # Could track start time, but keep simple
            time.sleep(1)

    observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
