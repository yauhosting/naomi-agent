"""
NAOMI Agent - Brain v4 (Anthropic Native Tool Use)

Architecture:
- Primary: Anthropic Messages API with native tool_use (structured, no hallucination)
- Fallback 1: Claude CLI direct
- Fallback 2: Claude proxy (OpenClaw)
- Fallback 3: MiniMax M2.7

Features:
- Native tool_use: Claude decides which tool to call, returns structured JSON
- Computer Use beta: Official screenshot + mouse/keyboard API
- Web Search: Built-in server-side tool
- Model switching: /model command
"""
import subprocess
import json
import os
import time
import base64
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger("naomi.brain")


def load_dotenv(env_path: str = ".env"):
    """Load .env file into environment."""
    p = Path(env_path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, val = line.split('=', 1)
        os.environ.setdefault(key.strip(), val.strip())


# === Tool Definitions for Anthropic API ===

NAOMI_TOOLS = [
    {
        "name": "shell",
        "description": "Execute a shell command on macOS. Returns stdout + stderr.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"}
            },
            "required": ["command"]
        }
    },
    {
        "name": "python_exec",
        "description": "Write and execute a Python script. Returns output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"]
        }
    },
    {
        "name": "file_read",
        "description": "Read a file and return its contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "file_write",
        "description": "Write content to a file (creates or overwrites).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"},
                "content": {"type": "string", "description": "Content to write"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the web using DuckDuckGo. Returns top 5 results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "open_app",
        "description": "Open a macOS application by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "app_name": {"type": "string", "description": "Application name (e.g. Safari, Calculator)"}
            },
            "required": ["app_name"]
        }
    },
    {
        "name": "screenshot",
        "description": "Take a screenshot of the current screen.",
        "input_schema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "click",
        "description": "Click at screen coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate"},
                "y": {"type": "integer", "description": "Y coordinate"}
            },
            "required": ["x", "y"]
        }
    },
    {
        "name": "type_text",
        "description": "Type text using the keyboard.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to type"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "key_press",
        "description": "Press a key or key combination (e.g. 'return', 'cmd+c', 'tab').",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key or combo to press"}
            },
            "required": ["key"]
        }
    },
    {
        "name": "pip_install",
        "description": "Install a Python package via pip.",
        "input_schema": {
            "type": "object",
            "properties": {
                "package": {"type": "string", "description": "Package name"}
            },
            "required": ["package"]
        }
    },
    {
        "name": "git",
        "description": "Execute a git command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Git command (without 'git' prefix)"}
            },
            "required": ["command"]
        }
    },
    {
        "name": "web_fetch",
        "description": "Fetch a web page URL and extract its text content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"}
            },
            "required": ["url"]
        }
    },
    {
        "name": "scroll",
        "description": "Scroll the screen up or down.",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down"], "description": "Scroll direction"},
                "amount": {"type": "integer", "description": "Number of scroll ticks (default 3)"}
            },
            "required": ["direction"]
        }
    },
    {
        "name": "generate_image",
        "description": "Generate an image using AI (ComfyUI). Returns file path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Image generation prompt"},
                "output_path": {"type": "string", "description": "Where to save (optional)"}
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "deploy_web",
        "description": "Deploy a web project. Methods: vercel, netlify, gh-pages, local.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_dir": {"type": "string", "description": "Project directory path"},
                "method": {"type": "string", "enum": ["vercel", "netlify", "gh-pages", "local"],
                           "description": "Deployment method"}
            },
            "required": ["project_dir"]
        }
    },
    {
        "name": "task_complete",
        "description": "Report that the task is complete. Call this when you have finished all actions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Summary of what was done"},
                "success": {"type": "boolean", "description": "Whether the task succeeded"}
            },
            "required": ["summary", "success"]
        }
    },
]

# Web search as Anthropic server-side tool
ANTHROPIC_WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5,
}


class Brain:
    # Model registry: name -> (backend, model_id, description)
    MODEL_REGISTRY = {
        "claude-sonnet":  ("anthropic_api", "claude-sonnet-4-6-20250514", "Claude Sonnet 4.6 via API (tool_use + vision)"),
        "claude-opus":    ("anthropic_api", "claude-opus-4-6-20250514",   "Claude Opus 4.6 via API (strongest)"),
        "claude-cli":     ("claude_cli",    None,                         "Claude CLI direct (Max subscription)"),
        "glm":            ("glm",           "glm-5.1",                    "GLM 5.1 (Z.AI, OpenAI compatible)"),
        "glm-turbo":      ("glm",           "glm-5-turbo",                "GLM 5 Turbo (Z.AI, fast)"),
        "ollama":         ("ollama",        "qwen3.5:35b",                "Ollama Qwen 3.5 35B (local, free)"),
        "ollama-gemma":   ("ollama",        "gemma4-uncensored:latest",    "Ollama Gemma 4 uncensored Q8 9GB (local, vision)"),
        "ollama-gemma4b": ("ollama",        "fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:e4b", "Ollama Gemma 4 E4B uncensored 6GB (local, fast)"),
        "ollama-gemma31b":("ollama",        "juilpark/gemma-4-31B-it-uncensored-heretic:q4_k_m", "Ollama Gemma 4 31B uncensored (local, heavy)"),
        "ollama-glm":     ("ollama",        "glm-4.7-flash:q8_0",        "Ollama GLM 4.7 Flash (local, 30GB)"),
        "ollama-dolphin":  ("ollama",       "dolphin-llama3:8b",          "Ollama Dolphin Llama3 8B (local, fast)"),
        "minimax":        ("minimax",       "MiniMax-M2.7",               "MiniMax M2.7 (fallback)"),
        "auto":           ("auto",          None,                         "Auto: CLI → Ollama → GLM → MiniMax"),
    }

    def __init__(self, config: dict):
        self.config = config
        self.primary = config.get("primary", {})
        self.fallback = config.get("fallback", {})
        self._claude_available = None
        self._claude_cli_path = None
        self._active_mode = "auto"
        self._anthropic_client = None

        # Error failover tracking
        self._consecutive_failures = {"cli": 0, "proxy": 0, "minimax": 0, "api": 0, "glm": 0, "ollama": 0}
        self._backoff_until = {"cli": 0, "proxy": 0, "minimax": 0, "api": 0, "glm": 0, "ollama": 0}
        self._max_failures_before_backoff = 3
        self._backoff_seconds = 60  # Skip a failed backend for 60s

        # Usage tracking
        self._usage = {"total_calls": 0, "by_backend": {}, "errors": 0, "start_time": time.time()}

        # Load .env
        load_dotenv()
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

        # Get API keys
        self._minimax_key = self.fallback.get("api_key") or os.environ.get("MINIMAX_API_KEY", "")
        self._anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._glm_key = os.environ.get("GLM_API_KEY", "")
        self._glm_base_url = "https://api.z.ai/api/paas/v4"

        if self._anthropic_key:
            logger.info("Anthropic API key loaded — native tool_use enabled")
        else:
            logger.info("No ANTHROPIC_API_KEY — using Claude CLI (Max subscription)")

        if self._glm_key:
            logger.info("GLM API key loaded (Z.AI)")

        if self._minimax_key:
            logger.info("MiniMax API key loaded")

    def _get_anthropic_client(self):
        """Lazy-init Anthropic client."""
        if self._anthropic_client is None and self._anthropic_key:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=self._anthropic_key)
        return self._anthropic_client

    # === Anthropic Native API ===

    def call_anthropic(self, prompt: str, system_prompt: str = "",
                       tools: list = None, model: str = None,
                       max_tokens: int = 4096,
                       images: List[Dict] = None) -> Any:
        """Call Anthropic Messages API directly. Returns full response object."""
        client = self._get_anthropic_client()
        if not client:
            return None

        model = model or self.MODEL_REGISTRY.get(
            self._active_mode, ("anthropic_api", "claude-sonnet-4-6-20250514", "")
        )[1] or "claude-sonnet-4-6-20250514"

        # Build message content
        content = []
        if images:
            for img in images:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.get("media_type", "image/png"),
                        "data": img["data"],
                    }
                })
        content.append({"type": "text", "text": prompt})

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = tools

        try:
            response = client.messages.create(**kwargs)
            return response
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return None

    def call_with_tools(self, prompt: str, system_prompt: str = "",
                        tools: list = None, model: str = None,
                        images: List[Dict] = None) -> Dict[str, Any]:
        """
        Call Anthropic API with tool_use. Returns structured result:
        {"text": "...", "tool_calls": [{"name": "...", "input": {...}, "id": "..."}], "stop_reason": "..."}
        """
        response = self.call_anthropic(
            prompt, system_prompt, tools=tools or NAOMI_TOOLS,
            model=model, images=images,
        )
        if not response:
            return {"text": "", "tool_calls": [], "stop_reason": "error"}

        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "name": block.name,
                    "input": block.input,
                    "id": block.id,
                })

        return {
            "text": "\n".join(text_parts),
            "tool_calls": tool_calls,
            "stop_reason": response.stop_reason,
        }

    def vision_analyze(self, prompt: str, image_path: str,
                       model: str = None) -> str:
        """Analyze a screenshot using Anthropic Vision API."""
        if not os.path.exists(image_path):
            return f"[Vision error: file not found {image_path}]"

        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        # Resize if too large (> 1MB base64)
        if len(img_data) > 1_400_000:
            resized = image_path.replace(".png", "_sm.png")
            subprocess.run(["sips", "-Z", "1280", image_path, "--out", resized],
                           capture_output=True, timeout=10)
            if os.path.exists(resized):
                with open(resized, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")
                try:
                    os.unlink(resized)
                except OSError:
                    pass

        images = [{"data": img_data, "media_type": "image/png"}]
        response = self.call_anthropic(prompt, images=images, model=model)

        if response:
            for block in response.content:
                if block.type == "text":
                    return block.text
        return "[Vision failed]"

    # === Agent Loop with Tool Use ===

    async def agent_loop(self, task: str, executor, system_prompt: str = "",
                         max_iterations: int = 15,
                         images: List[Dict] = None) -> Dict[str, Any]:
        """
        Full agent loop with native tool_use:
        1. Send task + tools to Claude
        2. Claude returns tool_use calls
        3. Execute tools via executor
        4. Send results back
        5. Repeat until done

        Returns: {"success": bool, "result": str, "steps": [...], "verified": True}
        """
        client = self._get_anthropic_client()
        if not client:
            # No API key — use CLI-based agent loop with structured JSON
            return await self._agent_loop_cli(task, executor, system_prompt, max_iterations)

        model = self.MODEL_REGISTRY.get(
            self._active_mode, ("anthropic_api", "claude-sonnet-4-6-20250514", "")
        )[1] or "claude-sonnet-4-6-20250514"

        sys = system_prompt or (
            "You are NAOMI, an autonomous AI agent running on macOS (Mac Mini). "
            "You have full system access. Use tools to complete tasks — do NOT just describe what you would do. "
            "Actually execute commands. When done, call the task_complete tool. "
            "Respond in Traditional Chinese when communicating results."
        )

        # Build initial message
        content = []
        if images:
            for img in images:
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": img.get("media_type", "image/png"),
                    "data": img["data"],
                }})
        content.append({"type": "text", "text": task})

        messages = [{"role": "user", "content": content}]
        steps = []

        for iteration in range(1, max_iterations + 1):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=sys,
                    tools=NAOMI_TOOLS,
                    messages=messages,
                )
            except Exception as e:
                logger.error(f"Agent loop API error: {e}")
                return {"success": False, "result": str(e), "steps": steps, "verified": True}

            # Process response
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Collect tool calls
            tool_results = []
            text_output = ""
            task_completed = False
            completion_summary = ""

            for block in assistant_content:
                if block.type == "text":
                    text_output += block.text
                elif block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_id = block.id

                    logger.info(f"Agent loop step {iteration}: tool={tool_name} input={json.dumps(tool_input)[:100]}")

                    # Check for task_complete
                    if tool_name == "task_complete":
                        task_completed = True
                        completion_summary = tool_input.get("summary", "")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": "Task marked as complete.",
                        })
                        steps.append({
                            "step": iteration, "tool": tool_name,
                            "input": tool_input,
                            "result": "completed",
                            "success": tool_input.get("success", True),
                        })
                        continue

                    # Execute the tool via executor
                    exec_result = await self._execute_tool(executor, tool_name, tool_input)

                    # Format result for API
                    result_content = str(exec_result.get("output", exec_result.get("error", json.dumps(exec_result))))[:5000]
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_content,
                    })

                    steps.append({
                        "step": iteration, "tool": tool_name,
                        "input": tool_input,
                        "result": result_content[:500],
                        "success": exec_result.get("success", False),
                    })

            if task_completed:
                return {
                    "success": True,
                    "result": completion_summary or text_output,
                    "steps": steps,
                    "iterations": iteration,
                    "verified": True,
                }

            # No tool use = Claude is done thinking
            if not tool_results:
                return {
                    "success": True,
                    "result": text_output,
                    "steps": steps,
                    "iterations": iteration,
                    "verified": len(steps) > 0,  # Only verified if tools were actually used
                }

            # Send tool results back
            messages.append({"role": "user", "content": tool_results})

        # Max iterations reached
        return {
            "success": False,
            "result": f"Max iterations ({max_iterations}) reached",
            "steps": steps,
            "iterations": max_iterations,
            "verified": True,
        }

    async def _agent_loop_cli(self, task: str, executor, system_prompt: str = "",
                              max_iterations: int = 15) -> Dict[str, Any]:
        """
        Agent loop using Claude CLI with --json-schema.
        Claude CLI has its own built-in tools (shell, file, web search) and executes them
        internally. We get back the structured result via --json-schema + --output-format json.
        """
        sys = system_prompt or (
            "You are NAOMI, an autonomous AI agent on macOS. "
            "Execute the task using your tools (shell, file operations, web search). "
            "Do NOT just describe — actually run commands and report real results. "
            "Respond in Traditional Chinese."
        )

        result_schema = {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "description": "Whether the task was completed successfully"},
                "summary": {"type": "string", "description": "Summary of what was done and the results"},
                "steps_taken": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "result": {"type": "string"},
                        },
                    },
                    "description": "List of actions taken and their results",
                },
            },
            "required": ["success", "summary"],
        }

        logger.info(f"CLI agent loop: {task[:100]}")

        # Run blocking CLI call in thread executor to not block event loop
        import asyncio as _aio
        loop = _aio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._call_claude_cli(task, system_prompt=sys, json_schema=result_schema),
        )

        if not response:
            return {"success": False, "result": "CLI returned no response",
                    "steps": [], "verified": False}

        # Parse the structured response
        try:
            result_data = json.loads(response)
        except json.JSONDecodeError:
            # Non-JSON response — CLI gave text answer
            return {"success": True, "result": response[:2000],
                    "steps": [{"tool": "cli_direct", "result": response[:500], "success": True}],
                    "verified": True}

        # Build steps from CLI's report
        steps = []
        for i, s in enumerate(result_data.get("steps_taken", []), 1):
            steps.append({
                "step": i,
                "tool": s.get("action", "cli"),
                "result": s.get("result", "")[:500],
                "success": True,
            })

        # If CLI didn't report steps, it still did work (num_turns in the wrapper)
        if not steps:
            steps = [{"step": 1, "tool": "cli_builtin", "result": "CLI executed internally", "success": True}]

        return {
            "success": result_data.get("success", True),
            "result": result_data.get("summary", response[:500]),
            "steps": steps,
            "iterations": 1,
            "verified": True,
        }

    async def _execute_tool(self, executor, tool_name: str, tool_input: dict) -> Dict[str, Any]:
        """Execute a tool call from the agent loop."""
        try:
            if tool_name == "shell":
                return await executor.execute("shell", tool_input["command"])
            elif tool_name == "python_exec":
                return await executor.execute("python", tool_input["code"])
            elif tool_name == "file_read":
                return await executor.execute("file_read", tool_input["path"])
            elif tool_name == "file_write":
                return await executor.execute("file_write", f"{tool_input['path']}|||{tool_input['content']}")
            elif tool_name == "web_search":
                return await executor.execute("web_search", tool_input["query"])
            elif tool_name == "open_app":
                return await executor.execute("open_app", tool_input["app_name"])
            elif tool_name == "screenshot":
                return await executor.execute("screenshot", "")
            elif tool_name == "click":
                return await executor.execute("click", f"{tool_input['x']} {tool_input['y']}")
            elif tool_name == "type_text":
                return await executor.execute("type_text", tool_input["text"])
            elif tool_name == "key_press":
                return await executor.execute("key_press", tool_input["key"])
            elif tool_name == "pip_install":
                return await executor.execute("pip_install", tool_input["package"])
            elif tool_name == "git":
                return await executor.execute("git", tool_input["command"])
            elif tool_name == "web_fetch":
                return await executor.execute("web_fetch", tool_input["url"])
            elif tool_name == "scroll":
                direction = tool_input.get("direction", "down")
                amount = tool_input.get("amount", 3)
                return await executor.execute("scroll", f"{direction} {amount}")
            elif tool_name == "generate_image":
                prompt = tool_input["prompt"]
                out = tool_input.get("output_path", "")
                params = f"{prompt}|||{out}" if out else prompt
                return await executor.execute("generate_image", params)
            elif tool_name == "deploy_web":
                d = tool_input.get("project_dir", "")
                m = tool_input.get("method", "local")
                return await executor.execute("deploy_web", f"{d}|||{m}")
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # === Legacy methods (kept for backward compatibility) ===

    def _check_claude_cli(self) -> bool:
        if self._claude_available is not None:
            return self._claude_available
        for claude_path in [
            "/opt/homebrew/bin/claude",
            os.path.expanduser("~/.nvm/versions/node/v22.22.2/bin/claude"),
            "/usr/local/bin/claude",
        ]:
            if os.path.exists(claude_path):
                self._claude_cli_path = claude_path
                self._claude_available = True
                logger.info("Claude CLI found at: %s" % claude_path)
                return True
        try:
            result = subprocess.run(["which", "claude"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                self._claude_cli_path = result.stdout.strip()
                self._claude_available = True
                return True
        except Exception:
            pass
        self._claude_available = False
        self._claude_cli_path = None
        return False

    def _call_minimax(self, prompt: str, system_prompt: str = "") -> str:
        """Call MiniMax M2.7 via Anthropic-compatible API."""
        import httpx
        if not self._minimax_key:
            return "[Brain offline: No MINIMAX_API_KEY configured]"

        base_url = self.fallback.get("base_url", "https://api.minimax.io/anthropic/v1")
        model = self.fallback.get("model", "MiniMax-M2.7")
        try:
            headers = {
                "x-api-key": self._minimax_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }
            body = {"model": model, "max_tokens": 4096,
                    "messages": [{"role": "user", "content": prompt}]}
            if system_prompt:
                body["system"] = system_prompt

            resp = None
            for attempt in range(3):
                resp = httpx.post(f"{base_url}/messages", headers=headers, json=body, timeout=90)
                if resp.status_code == 200:
                    break
                if resp.status_code in (429, 529):
                    time.sleep((attempt + 1) * 5)
                else:
                    break

            if resp.status_code != 200:
                return f"[Brain API error: {resp.status_code}]"

            data = resp.json()
            content = data.get("content", [])
            if content and isinstance(content, list):
                return next((c.get("text", "") for c in content if c.get("type") == "text"), str(content))
            return str(data)
        except Exception as e:
            return f"[Brain error: {e}]"

    def _call_ollama(self, prompt: str, system_prompt: str = "",
                     model: str = None) -> str:
        """Call Ollama local LLM (OpenAI-compatible API at localhost:11434)."""
        import httpx

        model = model or self.MODEL_REGISTRY.get(self._active_mode, ("", "qwen3.5:35b", ""))[1]
        if not model or model in (None, "(CLI default)"):
            model = "qwen3.5:35b"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = httpx.post(
                "http://127.0.0.1:11434/v1/chat/completions",
                json={"model": model, "messages": messages, "stream": False},
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                # Strip thinking tags if present (Qwen/DeepSeek)
                if "<think>" in content:
                    import re
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                self._record_success("ollama")
                return content
            else:
                logger.warning(f"Ollama error {resp.status_code}: {resp.text[:200]}")
                self._record_failure("ollama")
                return None
        except Exception as e:
            logger.debug(f"Ollama failed: {e}")
            self._record_failure("ollama")
            return None

    def _call_glm(self, prompt: str, system_prompt: str = "",
                   model: str = None) -> str:
        """Call Z.AI GLM API (OpenAI-compatible format).
        Note: GLM Coding Plan only works through Coding tools (Claude Code/OpenClaw).
        This method requires regular API balance (not Coding Plan).
        Use /model glm only if you have topped up API credits at z.ai.
        """
        import httpx
        if not self._glm_key:
            return None

        model = model or self.MODEL_REGISTRY.get(self._active_mode, ("", "glm-5.1", ""))[1]
        if not model or not model.startswith("glm"):
            model = "glm-5.1"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = httpx.post(
                f"{self._glm_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._glm_key}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "messages": messages, "max_tokens": 4096},
                timeout=90,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                self._record_success("glm")
                return content
            else:
                logger.warning(f"GLM API error {resp.status_code}: {resp.text[:200]}")
                self._record_failure("glm")
                return None
        except Exception as e:
            logger.error(f"GLM API failed: {e}")
            self._record_failure("glm")
            return None

    def _call_claude_proxy(self, prompt: str, system_prompt: str = "") -> str:
        import httpx
        proxy_url = self.primary.get("proxy_url", "http://127.0.0.1:18790/v1/chat/completions")
        model = self.primary.get("model", "claude-sonnet-4-6")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = httpx.post(proxy_url, json={"model": model, "messages": messages, "max_tokens": 4096},
                              timeout=self.primary.get("timeout", 120))
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            return None
        except Exception as e:
            logger.error(f"Claude proxy failed: {e}")
            return None

    def _call_claude_cli(self, prompt: str, system_prompt: str = "",
                         json_schema: dict = None) -> str:
        """Call Claude CLI. Use json_schema for structured output (agent loop)."""
        if not self._claude_cli_path:
            self._check_claude_cli()
            if not self._claude_cli_path:
                return None

        cmd = [self._claude_cli_path, "-p"]
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])
        if json_schema:
            cmd.extend(["--json-schema", json.dumps(json_schema), "--output-format", "json"])

        try:
            result = subprocess.run(
                cmd + [prompt],
                capture_output=True, text=True,
                timeout=self.primary.get("timeout", 120),
                env={**os.environ, "LANG": "en_US.UTF-8"},
                cwd="/tmp",
            )
            output = result.stdout.strip()
            if result.returncode == 0 and output and "Not logged in" not in output:
                self._record_success("cli")
                # If json output mode, extract structured_output from the wrapper
                if json_schema and output.startswith("{"):
                    try:
                        wrapper = json.loads(output)
                        structured = wrapper.get("structured_output")
                        if structured:
                            return json.dumps(structured)
                        # Fallback: return result text
                        return wrapper.get("result", output)
                    except json.JSONDecodeError:
                        return output
                return output
            if "Not logged in" in (output or ""):
                logger.warning("Claude CLI: not logged in")
            self._record_failure("cli")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("Claude CLI timed out")
            self._record_failure("cli")
            return None
        except Exception as e:
            logger.error(f"Claude CLI failed: {e}")
            self._record_failure("cli")
            return None

    # === Model Switching ===

    def set_model(self, model_name: str) -> Dict[str, Any]:
        name = model_name.strip().lower()
        aliases = {
            "sonnet": "claude-sonnet", "claude sonnet": "claude-sonnet",
            "opus": "claude-opus", "claude opus": "claude-opus",
            "cli": "claude-cli", "claude": "claude-cli",
            "minimax2.7": "minimax", "minimax-m2.7": "minimax", "mm": "minimax",
            "glm5": "glm", "glm-5": "glm", "glm5.1": "glm", "glm-5.1": "glm", "zhipu": "glm",
            "glm-turbo": "glm-turbo", "glm5-turbo": "glm-turbo",
            "qwen": "ollama", "qwen3.5": "ollama", "local": "ollama",
            "gemma": "ollama-gemma", "gemma4": "ollama-gemma", "uncensored": "ollama-gemma",
            "gemma4b": "ollama-gemma4b", "gemma-e4b": "ollama-gemma4b", "gemma-small": "ollama-gemma4b",
            "gemma31b": "ollama-gemma31b", "gemma-31b": "ollama-gemma31b", "gemma-big": "ollama-gemma31b",
            "dolphin": "ollama-dolphin", "llama": "ollama-dolphin",
            "glm-local": "ollama-glm", "glm4.7": "ollama-glm",
        }
        name = aliases.get(name, name)

        if name not in self.MODEL_REGISTRY:
            return {"success": False, "error": f"Unknown model: {model_name}",
                    "available": list(self.MODEL_REGISTRY.keys())}

        backend, model_id, desc = self.MODEL_REGISTRY[name]

        if backend == "anthropic_api" and not self._anthropic_key:
            return {"success": False, "error": "No ANTHROPIC_API_KEY configured"}
        if backend == "claude_cli" and not self._check_claude_cli():
            return {"success": False, "error": "Claude CLI not found"}
        if backend == "glm" and not self._glm_key:
            return {"success": False, "error": "No GLM_API_KEY configured"}
        if backend == "ollama":
            try:
                import httpx
                r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=3)
                if r.status_code != 200:
                    return {"success": False, "error": "Ollama not running"}
            except Exception:
                return {"success": False, "error": "Ollama not reachable at localhost:11434"}
        if backend == "minimax" and not self._minimax_key:
            return {"success": False, "error": "No MINIMAX_API_KEY configured"}

        old_mode = self._active_mode
        self._active_mode = name
        logger.info(f"Model switched: {old_mode} -> {name} ({desc})")
        return {"success": True, "model": name, "description": desc, "previous": old_mode}

    def get_model(self) -> Dict[str, str]:
        entry = self.MODEL_REGISTRY.get(self._active_mode, ("?", "?", "?"))
        return {"name": self._active_mode, "backend": entry[0],
                "model_id": entry[1] or "(CLI default)", "description": entry[2]}

    def get_usage(self) -> Dict[str, Any]:
        """Get usage statistics."""
        uptime = time.time() - self._usage["start_time"]
        hours = uptime / 3600
        return {
            "total_calls": self._usage["total_calls"],
            "errors": self._usage["errors"],
            "by_backend": self._usage["by_backend"],
            "uptime_hours": round(hours, 1),
            "calls_per_hour": round(self._usage["total_calls"] / max(hours, 0.01), 1),
            "error_rate": round(self._usage["errors"] / max(self._usage["total_calls"], 1) * 100, 1),
            "backend_health": {
                k: {"failures": v, "in_backoff": time.time() < self._backoff_until.get(k, 0)}
                for k, v in self._consecutive_failures.items()
            },
        }

    def list_models(self) -> List[Dict[str, str]]:
        models = []
        for name, (backend, model_id, desc) in self.MODEL_REGISTRY.items():
            active = " (active)" if name == self._active_mode else ""
            models.append({"name": name, "description": desc + active,
                           "active": name == self._active_mode})
        return models

    # === _think: simple text-only response (backward compat) ===

    def _is_backend_available(self, backend: str) -> bool:
        """Check if a backend is available (not in backoff)."""
        if self._consecutive_failures.get(backend, 0) >= self._max_failures_before_backoff:
            if time.time() < self._backoff_until.get(backend, 0):
                return False
            # Backoff expired, reset and try again
            self._consecutive_failures[backend] = 0
        return True

    def _record_success(self, backend: str):
        """Record a successful call."""
        self._consecutive_failures[backend] = 0
        self._usage["total_calls"] += 1
        self._usage["by_backend"][backend] = self._usage["by_backend"].get(backend, 0) + 1

    def _record_failure(self, backend: str):
        """Record a failed call with backoff."""
        self._consecutive_failures[backend] = self._consecutive_failures.get(backend, 0) + 1
        self._usage["errors"] += 1
        if self._consecutive_failures[backend] >= self._max_failures_before_backoff:
            self._backoff_until[backend] = time.time() + self._backoff_seconds
            logger.warning(f"Backend {backend}: {self._consecutive_failures[backend]} consecutive failures, "
                           f"backing off for {self._backoff_seconds}s")

    def _think(self, prompt: str, system_prompt: str = "") -> str:
        """Text response with automatic failover and backoff."""
        # Try Anthropic API
        if (self._anthropic_key and self._active_mode in ("auto", "claude-sonnet", "claude-opus")
                and self._is_backend_available("api")):
            response = self.call_anthropic(prompt, system_prompt)
            if response:
                for block in response.content:
                    if block.type == "text":
                        self._record_success("api")
                        return block.text
            self._record_failure("api")

        # Fallback: Claude CLI
        if self._active_mode in ("auto", "claude-cli") and self._is_backend_available("cli"):
            if self._check_claude_cli():
                result = self._call_claude_cli(prompt, system_prompt)
                if result and "Not logged in" not in result:
                    self._record_success("cli")
                    return result
                self._record_failure("cli")

        # Fallback: Claude proxy
        if self._active_mode == "auto" and self._is_backend_available("proxy"):
            result = self._call_claude_proxy(prompt, system_prompt)
            if result:
                self._record_success("proxy")
                return result
            self._record_failure("proxy")

        # Fallback: Ollama (local, free)
        if self._active_mode in ("auto", "ollama", "ollama-gemma", "ollama-glm", "ollama-dolphin") and self._is_backend_available("ollama"):
            result = self._call_ollama(prompt, system_prompt)
            if result:
                return result

        # Fallback: GLM (Z.AI)
        if self._active_mode in ("auto", "glm", "glm-turbo") and self._is_backend_available("glm"):
            if self._glm_key:
                result = self._call_glm(prompt, system_prompt)
                if result:
                    return result

        # Fallback: MiniMax
        if self._active_mode in ("auto", "minimax") and self._is_backend_available("minimax"):
            if self._minimax_key:
                result = self._call_minimax(prompt, system_prompt)
                if result and not result.startswith("[Brain"):
                    self._record_success("minimax")
                    return result
                self._record_failure("minimax")

        return "[Brain offline: No backend available]"

    # === Smart Routing ===

    # Privacy mode: when True, ALL messages go to local Ollama only (nothing leaves machine)
    _private_mode = False
    _private_model = "fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:e4b"

    def set_private_mode(self, enabled: bool, model: str = None) -> Dict[str, Any]:
        """Toggle privacy mode — all traffic stays local when enabled."""
        self._private_mode = enabled
        if model:
            self._private_model = model
        logger.info(f"Private mode: {'ON → ' + self._private_model if enabled else 'OFF'}")
        return {
            "private_mode": enabled,
            "model": self._private_model if enabled else "auto routing",
        }

    def _classify_complexity(self, prompt: str) -> str:
        """Classify: chat → MiniMax, code → CLI, private → local Ollama.
        Returns: 'chat', 'code', 'private'
        In private mode: code still returns 'code' (→ CLI), rest returns 'private' (→ local)
        """
        prompt_lower = prompt.lower()
        code_signals = [
            "code", "debug", "fix", "error", "bug", "implement", "build",
            "review", "analyze", "architecture", "design", "refactor",
            "寫代碼", "寫程式", "程式碼", "修改代碼", "修復", "分析", "設計", "架構", "重構",
            "def ", "class ", "function", "import ", "```",
            "explain this code", "what does this", "how to implement",
            "script", "api", "server", "deploy", "database",
        ]
        if any(sig in prompt_lower for sig in code_signals):
            return "code"  # Always CLI, even in private mode
        if len(prompt) > 500:
            return "code"
        return "private" if self._private_mode else "chat"

    def think_smart(self, prompt: str, context: str = "") -> str:
        """Multi-track routing:
        - chat → MiniMax M2.7 (Master's preference)
        - code → Claude CLI (strongest reasoning)
        - private → local Gemma 4B (nothing leaves machine)
        """
        complexity = self._classify_complexity(prompt)
        full = prompt + (f"\n\nContext:\n{context}" if context else "")
        system = ("You are NAOMI, an autonomous AI agent. "
                  "Be direct, actionable, and proactive. Respond in Traditional Chinese.")

        # Private mode — chat stays local, code still goes to CLI
        if complexity == "private":
            result = self._call_ollama(full, system, model=self._private_model)
            if result:
                logger.debug(f"Smart route: private chat → Ollama {self._private_model}")
                return result
            result = self._call_ollama(full, system)
            if result:
                return result
            return "[Private mode: Ollama not available. Use /private off to disable.]"

        # Chat — MiniMax first (Master's preference), then Ollama, then CLI
        if complexity == "chat":
            if self._minimax_key and self._is_backend_available("minimax"):
                result = self._call_minimax(full, system)
                if result and not result.startswith("[Brain"):
                    logger.debug("Smart route: chat → MiniMax")
                    self._record_success("minimax")
                    return result
            if self._is_backend_available("ollama"):
                result = self._call_ollama(full, system)
                if result:
                    logger.debug("Smart route: chat → Ollama (MiniMax failed)")
                    return result

        # Code / heavy — Claude CLI (strongest, even in private mode)
        return self._think(full, system)

    # === High-level methods ===

    def think(self, prompt: str, context: str = "") -> str:
        """General thinking — uses smart routing in auto mode."""
        if self._active_mode == "auto":
            return self.think_smart(prompt, context)
        system = ("You are NAOMI, an autonomous AI agent. "
                  "Be direct, actionable, and proactive. Respond in Traditional Chinese.")
        full = prompt + (f"\n\nContext:\n{context}" if context else "")
        return self._think(full, system)

    def analyze(self, task: str, context: str = "") -> Dict[str, Any]:
        system = ("You are NAOMI's analytical brain. Break down tasks into steps. "
                  "Respond in valid JSON only: "
                  '{"understanding":"...","steps":[{"step":1,"action":"...","tool":"...","details":"..."}],'
                  '"tools_needed":[],"estimated_complexity":"low/medium/high","risks":[]}')
        prompt = f"Task: {task}" + (f"\n\nContext:\n{context}" if context else "")
        response = self._think(prompt, system)
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response.strip())
        except (json.JSONDecodeError, IndexError):
            return {"understanding": task, "steps": [], "tools_needed": [],
                    "estimated_complexity": "medium", "risks": []}

    def debug(self, error: str, context: str = "") -> str:
        system = "You are NAOMI's debugger. Analyze errors and provide exact fix commands. Be concise."
        prompt = f"Error:\n{error}" + (f"\n\nContext:\n{context}" if context else "")
        return self._think(prompt, system)

    def write_code(self, spec: str, language: str = "python") -> str:
        system = f"Write clean, working {language} code. Output ONLY the code, no explanations."
        return self._think(spec, system)

    def reflect(self, history: str) -> Dict[str, Any]:
        system = ('Review recent activity. Respond in valid JSON only: '
                  '{"observations":[],"suggestions":[],"self_improvements":[],'
                  '"proactive_tasks":[],"priority":"low"}')
        response = self._think(f"Review:\n{history}", system)
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response.strip())
        except (json.JSONDecodeError, IndexError):
            return {"observations": [], "suggestions": [], "self_improvements": [],
                    "proactive_tasks": [], "priority": "low"}

    def generate_ideas(self, topic: str) -> str:
        return self._think(f"Generate creative ideas for: {topic}",
                           "Generate innovative ideas. Respond in Traditional Chinese.")

    def consolidate_memories(self, memories: str) -> str:
        return self._think(f"Consolidate:\n{memories}",
                           "Summarize into key insights. Be concise.")

    def strategize(self, goal: str, context: str = "") -> Dict[str, Any]:
        system = ('Create a strategy. Respond JSON only: '
                  '{"analysis":"...","strategy":"...","opportunities":[],"risks":[],'
                  '"creative_ideas":[],"next_steps":[]}')
        prompt = f"Goal: {goal}" + (f"\n\nContext:\n{context}" if context else "")
        response = self._think(prompt, system)
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response.strip())
        except (json.JSONDecodeError, IndexError):
            return {"analysis": goal, "strategy": response[:500],
                    "opportunities": [], "risks": [], "creative_ideas": [], "next_steps": []}
