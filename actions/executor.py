"""
NAOMI Agent - Action System (Hands)
Executes commands, manages files, runs code, handles Git operations.
"""
import os
import asyncio
import subprocess
import logging
import time
import tempfile
import shlex
from typing import Dict, Any

logger = logging.getLogger("naomi.actions")


class ActionExecutor:
    """Unified action execution system."""

    def __init__(self, memory, project_dir: str, brain=None):
        self.memory = memory
        self.project_dir = project_dir
        self._computer = None  # Lazy init
        self._sandbox = None   # Lazy init Docker sandbox

        self.actions = {
            "shell": self.execute_shell,
            "python": self.execute_python,
            "file_read": self.read_file,
            "file_write": self.write_file,
            "file_append": self.append_file,
            "git": self.git_operation,
            "pip_install": self.pip_install,
            "web_search": self.web_search,
            "screenshot": self._do_screenshot,
            "click": self._do_click,
            "type_text": self._do_type,
            "key_press": self._do_key,
            "open_app": self._do_open_app,
            "look_and_act": self._do_look_and_act,
            "web_fetch": self.web_fetch,
            "scroll": self._do_scroll,
            "generate_image": self.generate_image,
            "deploy_web": self.deploy_web,
        }
        self._brain = brain

    def set_brain(self, brain):
        """Set brain reference (called after brain is initialized)."""
        self._brain = brain

    def _get_computer(self):
        """Lazy-init ComputerControl."""
        if self._computer is None:
            from actions.computer import ComputerControl
            self._computer = ComputerControl(brain=self._brain)
        return self._computer

    # Computer control action wrappers
    def _do_screenshot(self, params: str) -> Dict[str, Any]:
        return self._get_computer().screenshot(params if params else None)

    def _do_click(self, params: str) -> Dict[str, Any]:
        parts = params.replace(",", " ").split()
        if len(parts) < 2:
            return {"success": False, "error": "Format: x y"}
        return self._get_computer().click(int(parts[0]), int(parts[1]))

    def _do_type(self, params: str) -> Dict[str, Any]:
        return self._get_computer().type_text(params)

    def _do_key(self, params: str) -> Dict[str, Any]:
        return self._get_computer().key(params)

    def _do_open_app(self, params: str) -> Dict[str, Any]:
        return self._get_computer().open_app(params)

    def _do_look_and_act(self, params: str) -> Dict[str, Any]:
        return self._get_computer().look_and_act(params)

    async def execute(self, action_type: str, params: str) -> Dict[str, Any]:
        """Execute an action by type with audit logging."""
        from core.security import audit_log, check_sensitive_command

        handler = self.actions.get(action_type)
        if not handler:
            return {"error": f"Unknown action type: {action_type}", "success": False}

        # Check for sensitive operations (shell commands)
        if action_type in ("shell", "python"):
            sensitive = check_sensitive_command(params)
            if sensitive:
                logger.warning(f"Sensitive operation detected: {sensitive['description']} — {params[:100]}")
                audit_log("SENSITIVE_OP", action_type, params, sensitive["description"], success=True)
                # Log but allow (Master granted full permissions)

        try:
            result = await asyncio.to_thread(handler, params)

            # Audit log every tool call
            result_summary = str(result.get("output", result.get("error", "")))[:200] if isinstance(result, dict) else str(result)[:200]
            audit_log("execute", action_type, params, result_summary, success=result.get("success", True) if isinstance(result, dict) else True)

            self.memory.remember_short(
                f"Action [{action_type}]: {params[:100]} -> success",
                category="action"
            )
            return result
        except Exception as e:
            error_msg = str(e)
            audit_log("execute", action_type, params, error_msg, success=False)
            self.memory.remember_short(
                f"Action [{action_type}]: {params[:100]} -> FAILED: {error_msg[:200]}",
                category="error"
            )
            return {"error": error_msg, "success": False}

    def _get_sandbox(self):
        """Lazy-init DockerSandbox."""
        if self._sandbox is None:
            from core.sandbox import DockerSandbox
            self._sandbox = DockerSandbox(project_dir=self.project_dir)
        return self._sandbox

    def execute_shell(self, command: str) -> Dict[str, Any]:
        """Execute a shell command. Routes sensitive commands through Docker sandbox if available."""
        from core.security import check_sensitive_command

        logger.info(f"Shell: {command[:200]}")

        # Check if command is sensitive and sandbox is available
        sensitive = check_sensitive_command(command)
        if sensitive:
            sandbox = self._get_sandbox()
            if sandbox.is_available():
                logger.info("Routing sensitive command through Docker sandbox: %s", sensitive.get("description", ""))
                return sandbox.execute(command, mount_dir=self.project_dir)
            else:
                logger.warning("Sensitive command but Docker not available, executing directly: %s", command[:100])

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
        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".py", prefix="naomi_exec_", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                tmp_file = f.name
            return self.execute_shell(f"python3 {shlex.quote(tmp_file)}")
        finally:
            if tmp_file:
                try:
                    os.unlink(tmp_file)
                except OSError:
                    pass

    def read_file(self, path: str) -> Dict[str, Any]:
        """Read a file."""
        full_path = path if os.path.isabs(path) else os.path.join(self.project_dir, path)
        if not os.path.exists(full_path):
            return {"error": f"File not found: {full_path}", "success": False}
        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
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
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"success": True, "path": full_path, "size": len(content)}

    def append_file(self, params: str) -> Dict[str, Any]:
        """Append to a file. Format: path|||content"""
        parts = params.split("|||", 1)
        if len(parts) != 2:
            return {"error": "Format: path|||content", "success": False}
        path, content = parts
        full_path = path.strip() if os.path.isabs(path.strip()) else os.path.join(self.project_dir, path.strip())
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'a', encoding='utf-8') as f:
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
        """Search the web using DuckDuckGo with content sanitization."""
        from core.security import sanitize_external_content
        try:
            from duckduckgo_search import DDGS
            results = list(DDGS().text(query, max_results=5))
            # Sanitize search result content
            for r in results:
                if "body" in r:
                    r["body"] = sanitize_external_content(r["body"], "web_search")[:300]
            if not results:
                return {"success": False, "results": [], "query": query,
                        "error": "Search returned no results"}
            return {
                "success": True,
                "results": results,
                "query": query,
            }
        except ImportError:
            install_result = self.pip_install("duckduckgo-search")
            if install_result.get("success"):
                try:
                    from duckduckgo_search import DDGS
                    results = list(DDGS().text(query, max_results=5))
                    return {"success": bool(results), "results": results, "query": query,
                            "error": "" if results else "No results"}
                except Exception as e2:
                    return {"error": f"ddgs installed but failed: {e2}", "success": False, "results": []}
            return {"error": "Could not install ddgs", "success": False, "results": []}
        except Exception as e:
            return {"error": str(e), "success": False, "results": []}

    def web_fetch(self, url: str) -> Dict[str, Any]:
        """Fetch a web page with content sanitization against prompt injection."""
        from core.security import sanitize_external_content
        logger.info(f"Fetching: {url[:100]}")
        try:
            import httpx
            from bs4 import BeautifulSoup

            headers = {"User-Agent": "NAOMI-Agent/0.4 (compatible; Bot)"}
            resp = httpx.get(url.strip(), headers=headers, timeout=30, follow_redirects=True)

            if resp.status_code != 200:
                return {"success": False, "error": f"HTTP {resp.status_code}", "url": url}

            content_type = resp.headers.get("content-type", "")

            if "text/html" in content_type or "text/plain" in content_type:
                soup = BeautifulSoup(resp.text, "html.parser")
                # Remove script/style tags
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                # Get title
                title = soup.title.string if soup.title else ""
                return {
                    "success": True,
                    "url": url,
                    "title": title,
                    "content": sanitize_external_content(text[:10000], f"web:{url[:50]}"),
                    "length": len(text),
                }
            elif "application/json" in content_type:
                return {"success": True, "url": url, "content": resp.text[:10000], "type": "json"}
            else:
                return {"success": True, "url": url, "content": resp.text[:5000],
                        "type": content_type, "length": len(resp.text)}

        except ImportError:
            self.pip_install("beautifulsoup4")
            try:
                from bs4 import BeautifulSoup
                return self.web_fetch(url)  # One retry after install
            except ImportError:
                return {"error": "Could not install beautifulsoup4", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False, "url": url}

    def _do_scroll(self, params: str) -> Dict[str, Any]:
        parts = params.split()
        direction = parts[0] if parts else "down"
        amount = int(parts[1]) if len(parts) > 1 else 3
        return self._get_computer().scroll(direction, amount)

    def generate_image(self, params: str) -> Dict[str, Any]:
        """Generate an image using ComfyUI on Windows PC or local tools.
        Format: prompt|||output_path  or just prompt (saves to /tmp/)
        """
        parts = params.split("|||", 1)
        prompt = parts[0].strip()
        output_path = parts[1].strip() if len(parts) > 1 else f"/tmp/naomi_gen_{int(time.time())}.png"

        logger.info(f"Generating image: {prompt[:80]}")

        # Try ComfyUI via OpenClaw API (Windows PC with GPU)
        try:
            import httpx
            resp = httpx.post(
                "http://127.0.0.1:18801/api/generate",
                json={"prompt": prompt, "width": 1024, "height": 1024},
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("image_url"):
                    # Download the image
                    img_resp = httpx.get(data["image_url"], timeout=30)
                    if img_resp.status_code == 200:
                        with open(output_path, "wb") as f:
                            f.write(img_resp.content)
                        return {"success": True, "path": output_path,
                                "method": "comfyui", "prompt": prompt}
        except Exception as e:
            logger.debug(f"ComfyUI generate failed: {e}")

        # Fallback: use Python Pillow to create placeholder
        try:
            code = f"""
from PIL import Image, ImageDraw, ImageFont
img = Image.new('RGB', (1024, 1024), '#1a1a2e')
draw = ImageDraw.Draw(img)
draw.text((50, 480), '''{prompt[:60]}''', fill='#00d4aa')
draw.text((50, 520), '[Placeholder — connect ComfyUI for real AI art]', fill='#888')
img.save('{output_path}')
print('saved')
"""
            result = self.execute_python(code)
            if result.get("success"):
                return {"success": True, "path": output_path,
                        "method": "placeholder", "prompt": prompt}
        except Exception:
            pass

        return {"success": False, "error": "No image generation backend available"}

    def deploy_web(self, params: str) -> Dict[str, Any]:
        """Deploy a web project. Format: project_dir|||method
        Methods: vercel, netlify, gh-pages, local (python http.server)
        """
        parts = params.split("|||", 1)
        project_dir = parts[0].strip()
        method = parts[1].strip() if len(parts) > 1 else "local"

        if not os.path.isdir(project_dir):
            return {"success": False, "error": f"Directory not found: {project_dir}"}

        logger.info(f"Deploying {project_dir} via {method}")

        if method == "vercel":
            result = self.execute_shell(f"cd '{project_dir}' && npx vercel --yes 2>&1")
            return {**result, "method": "vercel"}

        elif method == "netlify":
            result = self.execute_shell(f"cd '{project_dir}' && npx netlify deploy --prod 2>&1")
            return {**result, "method": "netlify"}

        elif method == "gh-pages":
            result = self.execute_shell(
                f"cd '{project_dir}' && git init && git add -A && git commit -m 'deploy' && "
                f"npx gh-pages -d . 2>&1"
            )
            return {**result, "method": "gh-pages"}

        elif method == "local":
            # Start a simple HTTP server in background
            port = 18900
            result = self.execute_shell(
                f"cd '{project_dir}' && nohup python3 -m http.server {port} > /dev/null 2>&1 &"
                f" && echo 'Server started on port {port}'"
            )
            return {**result, "method": "local", "url": f"http://localhost:{port}"}

        return {"success": False, "error": f"Unknown deploy method: {method}"}


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
        result = await asyncio.to_thread(self.actions.pip_install, tool_name)
        if result.get("success"):
            self.available_tools[tool_name] = True
            return result

        # Try common install methods
        install_methods = [
            f"pip3 install {tool_name}",
            f"bash -lc 'npm install -g {tool_name}'",
        ]

        for method in install_methods:
            result = await asyncio.to_thread(self.actions.execute_shell, method)
            if result.get("success"):
                self.available_tools[tool_name] = True
                self.memory.learn_skill(
                    name=tool_name,
                    description=f"Auto-installed tool: {tool_name}",
                    install_command=method,
                )
                return result

        # Search the web for installation instructions
        search_result = await asyncio.to_thread(self.actions.web_search, f"install {tool_name} macOS")
        return {
            "success": False,
            "message": f"Could not auto-install {tool_name}",
            "search_results": search_result.get("results", []),
        }

    def list_tools(self) -> Dict[str, bool]:
        """List all known tools and their availability."""
        return self.available_tools
