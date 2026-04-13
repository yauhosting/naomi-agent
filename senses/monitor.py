"""
NAOMI Agent - Senses System (Eyes)
Monitors files, system resources, web trends, and project state.
"""
import os
import time
import psutil
import logging
import subprocess
from typing import Dict, List, Any

logger = logging.getLogger("naomi.senses")


class SystemSenses:
    """Monitor system state - CPU, RAM, disk, processes."""

    def __init__(self):
        self.last_check = {}

    async def check_system(self) -> Dict[str, Any]:
        alerts = []
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        if cpu > 90:
            alerts.append(f"High CPU usage: {cpu}%")
        if ram.percent > 90:
            alerts.append(f"High RAM usage: {ram.percent}%")
        if disk.percent > 90:
            alerts.append(f"Low disk space: {disk.percent}% used")

        result = {
            "cpu_percent": cpu,
            "ram_percent": ram.percent,
            "ram_available_gb": round(ram.available / (1024**3), 1),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 1),
            "alerts": alerts,
            "timestamp": time.time(),
        }

        self.last_check = result
        return result

    def get_running_processes(self, keyword: str = None) -> List[Dict]:
        procs = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                info = proc.info
                if keyword and keyword.lower() not in info['name'].lower():
                    continue
                procs.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return sorted(procs, key=lambda x: x.get('cpu_percent', 0), reverse=True)[:20]


class ProjectSenses:
    """Monitor project directories for changes."""

    def __init__(self, watch_dirs: List[str] = None):
        self.watch_dirs = watch_dirs or []
        self.file_states = {}  # path -> mtime

    def scan_changes(self) -> List[Dict]:
        changes = []
        for watch_dir in self.watch_dirs:
            if not os.path.exists(watch_dir):
                continue
            for root, dirs, files in os.walk(watch_dir):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != 'node_modules']
                for f in files:
                    if f.startswith('.') or f.endswith('.pyc'):
                        continue
                    path = os.path.join(root, f)
                    try:
                        mtime = os.path.getmtime(path)
                        if path in self.file_states:
                            if mtime > self.file_states[path]:
                                changes.append({
                                    "type": "modified",
                                    "path": path,
                                    "mtime": mtime,
                                })
                        else:
                            self.file_states[path] = mtime
                            # Don't report initial scan as changes
                        self.file_states[path] = mtime
                    except OSError:
                        pass
        return changes

    def check_git_status(self, repo_dir: str) -> Dict[str, Any]:
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=repo_dir
            )
            log = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                capture_output=True, text=True, cwd=repo_dir
            )
            return {
                "changes": status.stdout.strip().split('\n') if status.stdout.strip() else [],
                "recent_commits": log.stdout.strip().split('\n') if log.stdout.strip() else [],
                "has_uncommitted": bool(status.stdout.strip()),
            }
        except Exception as e:
            return {"error": str(e)}
