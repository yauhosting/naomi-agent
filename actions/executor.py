"""
NAOMI Agent - Action System (Hands)
Executes commands, manages files, runs code, handles Git operations.
"""
import os
import subprocess
import logging
import json
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("naomi.actions")


class ActionExecutor:
    """Unified action execution system."""

    def __init__(self, memory, project_dir: str):
        self.memory = memory
        self.project_dir = project_dir
        self.actions = {
            "shell": self.execute_shell,
            "python": self.execute_python,
            "file_read": self.read_file,
            "file_write": self.write_file,
            "file_append": self.append_file,
            "git": self.git_operation,
            "pip_install": self.pip_install,
            "web_search": self.web_search,
        }

    async def execute(self, action_type: str, params: str) -> Dict[str, Any]:
        """Execute an action by type."""
        handler = self.actions.get(action_type)
        if not handler:
            return {"error": f"Unknown action type: {action_type}", "success": False}

        try:
            result = handler(params)
            self.memory.remember_short(
                f"Action [{action_type}]: {params[:100]} -> success",
                category="action"
            )
            return result
        except Exception as e:
            error_msg = str(e)
            self.memory.remember_short(
                f"Action [{action_type}]: {params[:100]} -> FAILED: {error_msg[:200]}",
                category="error"
            )
            return {"error": error_msg, "success": False}

    def execute_shell(self, command: str) -> Dict[str, Any]:
        """Execute a shell command."""
        logger.info(f"Shell: {command[:200]}")
        try:
            result = subprocess.run(
                ["bash", "-lc", command],
                capture_output=True, text=True, timeout=300,
                cwd=self.project_dir
            )
            output = result.stdout + result.stderr
            return {
                "success": result.returncode == 0,
                "output": output[:5000],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out (300s)", "success": False}

    def execute_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code."""
        logger.info(f"Python exec: {code[:100]}")
        # Write to temp file and execute
        tmp_file = "/tmp/naomi_exec.py"
        with open(tmp_file, 'w') as f:
            f.write(code)
        return self.execute_shell(f"python3 {tmp_file}")

    def read_file(self, path: str) -> Dict[str, Any]:
        """Read a file."""
        full_path = path if os.path.isabs(path) else os.path.join(self.project_dir, path)
        if not os.path.exists(full_path):
            return {"error": f"File not found: {full_path}", "success": False}
        with open(full_path, 'r') as f:
            content = f.read()
        return {"success": True, "content": content, "path": full_path}

    def write_file(self, params: str) -> Dict[str, Any]:
        """Write to a file. Format: path|||content"""
        parts = params.split("|||", 1)
        if len(parts) != 2:
            return {"error": "Format: path|||content", "success": False}
        path, content = parts
        full_path = path.strip() if os.path.isabs(path.strip()) else os.path.join(self.project_dir, path.strip())
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
        return {"success": True, "path": full_path, "size": len(content)}

    def append_file(self, params: str) -> Dict[str, Any]:
        """Append to a file. Format: path|||content"""
        parts = params.split("|||", 1)
        if len(parts) != 2:
            return {"error": "Format: path|||content", "success": False}
        path, content = parts
        full_path = path.strip() if os.path.isabs(path.strip()) else os.path.join(self.project_dir, path.strip())
        with open(full_path, 'a') as f:
            f.write(content)
        return {"success": True, "path": full_path}

    def git_operation(self, command: str) -> Dict[str, Any]:
        """Execute a git command."""
        return self.execute_shell(f"git {command}")

    def pip_install(self, package: str) -> Dict[str, Any]:
        """Install a Python package."""
        logger.info(f"Installing: {package}")
        result = self.execute_shell(f"pip3 install {package}")
        if result.get("success"):
            self.memory.learn_skill(
                name=package,
                description=f"Python package: {package}",
                tool_command=f"import {package}",
                install_command=f"pip3 install {package}"
            )
        return result

    def web_search(self, query: str) -> Dict[str, Any]:
        """Search the web using DuckDuckGo."""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            return {
                "success": True,
                "results": results,
                "query": query,
            }
        except ImportError:
            # Auto-install duckduckgo-search
            self.pip_install("duckduckgo-search")
            return self.web_search(query)  # Retry
        except Exception as e:
            return {"error": str(e), "success": False}


class ToolManager:
    """Discovers, installs, and manages tools."""

    def __init__(self, memory, actions: ActionExecutor):
        self.memory = memory
        self.actions = actions
        self.available_tools = {}
        self._scan_tools()

    def _scan_tools(self):
        """Scan for available tools on the system."""
        common_tools = {
            "python3": "python3 --version",
            "git": "git --version",
            "curl": "curl --version",
            "pip3": "pip3 --version",
            "node": "bash -lc 'node --version'",
            "npm": "bash -lc 'npm --version'",
            "ffmpeg": "ffmpeg -version",
            "yt-dlp": "yt-dlp --version",
            "docker": "docker --version",
        }
        for tool, check_cmd in common_tools.items():
            result = self.actions.execute_shell(check_cmd)
            self.available_tools[tool] = result.get("success", False)

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is available."""
        # Check cached status
        if tool_name in self.available_tools:
            return self.available_tools[tool_name]
        # Check if it's a known Python package
        skill = self.memory.recall_skill(tool_name)
        if skill:
            return True
        return False

    async def auto_install(self, tool_name: str) -> Dict[str, Any]:
        """Automatically find and install a missing tool."""
        logger.info(f"Auto-installing tool: {tool_name}")

        # Check if it's a Python package first
        result = self.actions.pip_install(tool_name)
        if result.get("success"):
            self.available_tools[tool_name] = True
            return result

        # Try common install methods
        install_methods = [
            f"pip3 install {tool_name}",
            f"bash -lc 'npm install -g {tool_name}'",
        ]

        for method in install_methods:
            result = self.actions.execute_shell(method)
            if result.get("success"):
                self.available_tools[tool_name] = True
                self.memory.learn_skill(
                    name=tool_name,
                    description=f"Auto-installed tool: {tool_name}",
                    install_command=method,
                )
                return result

        # Search the web for installation instructions
        search_result = self.actions.web_search(f"install {tool_name} macOS")
        return {
            "success": False,
            "message": f"Could not auto-install {tool_name}",
            "search_results": search_result.get("results", []),
        }

    def list_tools(self) -> Dict[str, bool]:
        """List all known tools and their availability."""
        return self.available_tools
