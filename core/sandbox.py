"""NAOMI Agent - Docker Sandbox for Safe Command Execution
Runs dangerous commands in isolated Docker containers.
"""
import subprocess
import logging
import os
import time
from typing import Dict, Any

logger = logging.getLogger("naomi.sandbox")


class DockerSandbox:
    """Run commands in Docker containers for safety."""

    IMAGE = "python:3.12-slim"  # Default sandbox image

    def __init__(self, project_dir: str = "") -> None:
        self.project_dir = project_dir
        self._docker_available = self._check_docker()

    def _check_docker(self) -> bool:
        """Check if Docker is available and the daemon is running."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            available = result.returncode == 0
            if available:
                logger.info("Docker sandbox available")
            else:
                logger.debug("Docker not available: %s", result.stderr[:200])
            return available
        except FileNotFoundError:
            logger.debug("Docker binary not found")
            return False
        except subprocess.TimeoutExpired:
            logger.debug("Docker info timed out")
            return False
        except Exception as e:
            logger.debug("Docker check failed: %s", e)
            return False

    def execute(
        self,
        command: str,
        image: str | None = None,
        timeout: int = 60,
        mount_dir: str = "",
    ) -> Dict[str, Any]:
        """Execute a command inside a Docker container.

        Args:
            command: Shell command to run
            image: Docker image (default: python:3.12-slim)
            timeout: Max seconds
            mount_dir: Optional directory to mount as /workspace (read-only)
        Returns:
            {"success": bool, "output": str, "exit_code": int, "sandboxed": True}
        """
        if not self._docker_available:
            logger.warning("Docker not available, cannot sandbox command")
            return {
                "success": False,
                "output": "Docker not available for sandboxed execution",
                "exit_code": -1,
                "sandboxed": False,
            }

        image = image or self.IMAGE
        start_time = time.time()

        docker_cmd = [
            "docker", "run",
            "--rm",                    # Auto-remove container after exit
            "--network", "none",       # Network isolation
            "--memory", "512m",        # Memory limit
            "--cpus", "1.0",           # CPU limit
            "--pids-limit", "100",     # Process limit
            "--read-only",             # Read-only root filesystem
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=100m",  # Writable /tmp
        ]

        # Mount directory read-only if specified
        effective_mount = mount_dir or self.project_dir
        if effective_mount and os.path.isdir(effective_mount):
            abs_mount = os.path.abspath(effective_mount)
            docker_cmd.extend(["-v", f"{abs_mount}:/workspace:ro"])
            docker_cmd.extend(["-w", "/workspace"])

        docker_cmd.extend([image, "sh", "-c", command])

        logger.info("Sandbox exec: %s (image=%s, timeout=%ds)", command[:100], image, timeout)

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = (result.stdout + result.stderr).strip()
            elapsed = round(time.time() - start_time, 1)

            return {
                "success": result.returncode == 0,
                "output": output[:5000],
                "exit_code": result.returncode,
                "sandboxed": True,
                "image": image,
                "elapsed_seconds": elapsed,
            }
        except subprocess.TimeoutExpired:
            elapsed = round(time.time() - start_time, 1)
            logger.warning("Sandbox command timed out after %ds: %s", timeout, command[:100])
            return {
                "success": False,
                "output": f"Command timed out after {timeout}s (sandboxed)",
                "exit_code": -1,
                "sandboxed": True,
                "elapsed_seconds": elapsed,
            }
        except Exception as e:
            logger.error("Sandbox execution error: %s", e)
            return {
                "success": False,
                "output": f"Sandbox error: {e}",
                "exit_code": -1,
                "sandboxed": True,
            }

    def is_available(self) -> bool:
        """Check if Docker sandbox is available."""
        return self._docker_available

    def refresh_availability(self) -> bool:
        """Re-check Docker availability (e.g. after Docker daemon starts)."""
        self._docker_available = self._check_docker()
        return self._docker_available
