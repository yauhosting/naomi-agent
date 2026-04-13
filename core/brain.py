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
        "minimax":        ("minimax",       "MiniMax-M2.7",               "MiniMax M2.7 (free fallback)"),
        "auto":           ("auto",          None,                         "Auto: API → CLI → MiniMax fallback"),
    }

    def __init__(self, config: dict):
        self.config = config
        self.primary = config.get("primary", {})
        self.fallback = config.get("fallback", {})
        self._claude_available = None
        self._claude_cli_path = None
        self._active_mode = "auto"
        self._anthropic_client = None

        # Load .env
        load_dotenv()
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

        # Get API keys
        self._minimax_key = self.fallback.get("api_key") or os.environ.get("MINIMAX_API_KEY", "")
        self._anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

        if self._anthropic_key:
            logger.info("Anthropic API key loaded — native tool_use enabled")
        else:
            logger.info("No ANTHROPIC_API_KEY — using Claude CLI (Max subscription)")

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
        Agent loop using Claude CLI (no API key needed, uses Max subscription).
        Claude returns JSON tool calls, NAOMI executes them.
        """
        tool_names = [t["name"] for t in NAOMI_TOOLS]
        tool_descriptions = "\n".join(
            f"- {t['name']}: {t['description']} | params: {list(t['input_schema'].get('properties', {}).keys())}"
            for t in NAOMI_TOOLS
        )

        sys = system_prompt or "You are NAOMI, an autonomous AI agent on macOS."
        steps = []
        history = ""  # Accumulate results for context

        for iteration in range(1, max_iterations + 1):
            prompt = f"""{sys}

Available tools:
{tool_descriptions}

Task: {task}

{"Previous steps and results:" + chr(10) + history if history else ""}

Decide the next action. Reply in JSON ONLY:
{{"tool": "tool_name", "input": {{"param": "value"}}, "reasoning": "why"}}

If the task is complete, use:
{{"tool": "task_complete", "input": {{"summary": "what was done", "success": true}}}}

RULES:
- Use exactly ONE tool per response
- Do NOT describe what you would do — call the tool
- Do NOT make up results — wait for real output"""

            response = self._think(prompt)

            # Parse JSON tool call
            try:
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0]
                raw = response.strip()
                if not raw.startswith("{"):
                    start = raw.find("{")
                    end = raw.rfind("}") + 1
                    if start >= 0 and end > start:
                        raw = raw[start:end]
                call = json.loads(raw)
            except (json.JSONDecodeError, IndexError):
                logger.warning(f"CLI agent loop: could not parse response at step {iteration}")
                steps.append({"step": iteration, "error": "Could not parse tool call",
                              "raw": response[:300]})
                # Give it one more try with simpler prompt
                continue

            tool_name = call.get("tool", "")
            tool_input = call.get("input", {})
            reasoning = call.get("reasoning", "")

            logger.info(f"CLI agent step {iteration}: {tool_name} — {reasoning[:80]}")

            # Check for task_complete
            if tool_name == "task_complete":
                steps.append({"step": iteration, "tool": "task_complete",
                              "input": tool_input, "success": tool_input.get("success", True)})
                return {
                    "success": tool_input.get("success", True),
                    "result": tool_input.get("summary", ""),
                    "steps": steps,
                    "iterations": iteration,
                    "verified": True,
                }

            if tool_name not in tool_names:
                steps.append({"step": iteration, "error": f"Unknown tool: {tool_name}"})
                history += f"\nStep {iteration}: ERROR — unknown tool '{tool_name}'\n"
                continue

            # Execute the tool
            exec_result = await self._execute_tool(executor, tool_name, tool_input)
            output = str(exec_result.get("output", exec_result.get("error", json.dumps(exec_result))))[:1000]
            success = exec_result.get("success", False)

            steps.append({
                "step": iteration, "tool": tool_name, "input": tool_input,
                "result": output[:500], "success": success,
            })

            history += f"\nStep {iteration}: [{tool_name}] {'OK' if success else 'FAILED'}\nOutput: {output[:500]}\n"

        return {
            "success": False,
            "result": f"Max iterations ({max_iterations}) reached",
            "steps": steps,
            "iterations": max_iterations,
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

    def _call_claude_cli(self, prompt: str, system_prompt: str = "") -> str:
        if not self._claude_cli_path:
            return None
        full_prompt = f"[System: {system_prompt}]\n\n{prompt}" if system_prompt else prompt
        try:
            result = subprocess.run(
                [self._claude_cli_path, "-p", full_prompt],
                capture_output=True, text=True,
                timeout=self.primary.get("timeout", 120),
                env={**os.environ, "LANG": "en_US.UTF-8"},
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            return None
        except Exception as e:
            logger.error(f"Claude CLI failed: {e}")
            return None

    # === Model Switching ===

    def set_model(self, model_name: str) -> Dict[str, Any]:
        name = model_name.strip().lower()
        aliases = {
            "sonnet": "claude-sonnet", "claude sonnet": "claude-sonnet",
            "opus": "claude-opus", "claude opus": "claude-opus",
            "cli": "claude-cli", "claude": "claude-cli",
            "minimax2.7": "minimax", "minimax-m2.7": "minimax", "mm": "minimax",
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

    def list_models(self) -> List[Dict[str, str]]:
        models = []
        for name, (backend, model_id, desc) in self.MODEL_REGISTRY.items():
            active = " (active)" if name == self._active_mode else ""
            models.append({"name": name, "description": desc + active,
                           "active": name == self._active_mode})
        return models

    # === _think: simple text-only response (backward compat) ===

    def _think(self, prompt: str, system_prompt: str = "") -> str:
        """Simple text response — tries Anthropic API first, then fallbacks."""
        # Try Anthropic API (no tools, just text)
        if self._anthropic_key and self._active_mode in ("auto", "claude-sonnet", "claude-opus"):
            response = self.call_anthropic(prompt, system_prompt)
            if response:
                for block in response.content:
                    if block.type == "text":
                        return block.text

        # Fallback: Claude CLI
        if self._active_mode in ("auto", "claude-cli"):
            if self._check_claude_cli():
                result = self._call_claude_cli(prompt, system_prompt)
                if result and "Not logged in" not in result:
                    return result

        # Fallback: Claude proxy
        if self._active_mode == "auto":
            result = self._call_claude_proxy(prompt, system_prompt)
            if result:
                return result

        # Fallback: MiniMax
        if self._active_mode in ("auto", "minimax"):
            if self._minimax_key:
                result = self._call_minimax(prompt, system_prompt)
                if result and not result.startswith("[Brain"):
                    return result

        return "[Brain offline: No backend available]"

    # === High-level methods (backward compat) ===

    def think(self, prompt: str, context: str = "") -> str:
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
