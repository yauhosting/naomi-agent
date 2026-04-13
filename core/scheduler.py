"""
NAOMI Agent - Persistent Task Scheduler
Jobs survive restarts via JSON file storage.

Supports:
- One-shot: run once at a specific time
- Recurring: cron-like repeat (every N minutes/hours)
- Persistent: saved to data/scheduled_jobs.json
"""
import os
import json
import time
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("naomi.scheduler")

JOBS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "scheduled_jobs.json")


class Scheduler:
    """Persistent job scheduler with JSON file storage."""

    def __init__(self):
        self._jobs = {}
        self._load()

    def _load(self):
        """Load jobs from disk."""
        if os.path.exists(JOBS_FILE):
            try:
                with open(JOBS_FILE, 'r') as f:
                    self._jobs = json.load(f)
                logger.info(f"Loaded {len(self._jobs)} scheduled jobs")
            except Exception as e:
                logger.warning(f"Failed to load jobs: {e}")
                self._jobs = {}

    def _save(self):
        """Save jobs to disk."""
        os.makedirs(os.path.dirname(JOBS_FILE), exist_ok=True)
        with open(JOBS_FILE, 'w') as f:
            json.dump(self._jobs, f, indent=2, ensure_ascii=False)

    def add(self, name: str, command: str, run_at: float = None,
            interval_minutes: int = None, repeat: int = 1) -> Dict:
        """
        Add a scheduled job.
        - run_at: unix timestamp for first run (default: now + interval)
        - interval_minutes: repeat interval (None = one-shot)
        - repeat: number of times to repeat (-1 = forever)
        """
        job_id = name.lower().replace(" ", "-")

        if run_at is None:
            if interval_minutes:
                run_at = time.time() + interval_minutes * 60
            else:
                return {"success": False, "error": "Need run_at or interval_minutes"}

        self._jobs[job_id] = {
            "name": name,
            "command": command,
            "run_at": run_at,
            "interval_minutes": interval_minutes,
            "repeat": repeat,
            "runs_completed": 0,
            "last_run": None,
            "status": "active",
            "created_at": time.time(),
        }
        self._save()
        logger.info(f"Job added: {job_id} (run at {time.strftime('%H:%M', time.localtime(run_at))})")
        return {"success": True, "id": job_id, "run_at": run_at}

    def remove(self, job_id: str) -> Dict:
        """Remove a job."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            self._save()
            return {"success": True, "id": job_id}
        return {"success": False, "error": f"Job not found: {job_id}"}

    def pause(self, job_id: str) -> Dict:
        """Pause a job."""
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = "paused"
            self._save()
            return {"success": True}
        return {"success": False, "error": "Job not found"}

    def resume(self, job_id: str) -> Dict:
        """Resume a paused job."""
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = "active"
            self._save()
            return {"success": True}
        return {"success": False, "error": "Job not found"}

    def list_jobs(self) -> List[Dict]:
        """List all jobs."""
        jobs = []
        for job_id, job in self._jobs.items():
            next_run = time.strftime("%m-%d %H:%M", time.localtime(job["run_at"])) if job["run_at"] else "?"
            jobs.append({
                "id": job_id,
                "name": job["name"],
                "command": job["command"][:80],
                "next_run": next_run,
                "interval": f"{job['interval_minutes']}m" if job.get("interval_minutes") else "once",
                "runs": job["runs_completed"],
                "repeat": job["repeat"],
                "status": job["status"],
            })
        return jobs

    def get_due_jobs(self) -> List[Dict]:
        """Get jobs that are due for execution."""
        now = time.time()
        due = []
        for job_id, job in list(self._jobs.items()):
            if job["status"] != "active":
                continue
            if job["run_at"] <= now:
                due.append({"id": job_id, **job})
        return due

    def mark_completed(self, job_id: str):
        """Mark a job run as completed. Schedule next run or remove if done."""
        if job_id not in self._jobs:
            return

        job = self._jobs[job_id]
        job["runs_completed"] += 1
        job["last_run"] = time.time()

        # Check if we should schedule next run
        if job.get("interval_minutes") and (job["repeat"] == -1 or job["runs_completed"] < job["repeat"]):
            # Schedule next run
            job["run_at"] = time.time() + job["interval_minutes"] * 60
            logger.info(f"Job {job_id}: next run at {time.strftime('%H:%M', time.localtime(job['run_at']))}")
        else:
            # One-shot or exhausted repeats — remove
            job["status"] = "completed"
            logger.info(f"Job {job_id}: completed ({job['runs_completed']} runs)")

        self._save()

    def get_status(self) -> Dict:
        active = sum(1 for j in self._jobs.values() if j["status"] == "active")
        return {
            "total_jobs": len(self._jobs),
            "active_jobs": active,
            "jobs": self.list_jobs(),
        }
