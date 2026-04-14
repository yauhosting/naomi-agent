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
import threading
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
        self._lock = threading.Lock()
        self._changed = False
        self._last_event = 0.0

    @property
    def changed(self) -> bool:
        with self._lock:
            return self._changed

    @changed.setter
    def changed(self, value: bool) -> None:
        with self._lock:
            self._changed = value

    def on_modified(self, event):
        if not event.src_path.endswith(".py"):
            return
        rel = os.path.relpath(event.src_path, PROJECT_DIR)
        # Skip data/ and __pycache__
        if rel.startswith("data/") or "__pycache__" in rel:
            return
        now = time.time()
        with self._lock:
            # Debounce: ignore events within 1 second
            if now - self._last_event < 1.0:
                return
            self._last_event = now
            self._changed = True
        logger.info("File changed: %s -> scheduling reload", rel)


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
    # Watch core/, communication/, actions/, senses/, navigation/ and root .py files
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

            # Check if process has exited (crash or normal exit)
            ret = proc_holder[0].poll()
            if ret is not None:
                elapsed = time.time() - stable_since
                if elapsed > STABLE_THRESHOLD:
                    # Process ran long enough — reset backoff
                    backoff = MIN_BACKOFF
                    consecutive_crashes = 0

                consecutive_crashes += 1
                logger.warning(
                    "NAOMI exited with code %d (crash #%d). "
                    "Restarting in %ds...",
                    ret,
                    consecutive_crashes,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
                proc_holder[0] = start_agent()
                stable_since = time.time()

            time.sleep(0.5)
    except SystemExit:
        raise
    except Exception:
        logger.exception("Launcher encountered an unexpected error")
        _terminate_process(proc_holder[0])
        observer.stop()
        observer.join()
        sys.exit(1)


if __name__ == "__main__":
    main()