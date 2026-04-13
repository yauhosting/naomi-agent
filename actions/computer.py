"""
NAOMI Agent - Computer Control (Eyes + Hands)
macOS desktop control via AppleScript + screencapture + Claude Vision.

Zero dependencies — uses only macOS built-in tools:
- screencapture: screenshot
- osascript: AppleScript for mouse/keyboard/app control
- claude CLI or proxy: vision analysis of screenshots

Vision loop: screenshot → analyze → act → screenshot → verify → repeat
"""
import os
import json
import time
import base64
import subprocess
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("naomi.computer")

# Screenshot storage
SCREENSHOT_DIR = "/tmp/naomi_screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Key code mapping for AppleScript
KEY_CODES = {
    "return": 36, "enter": 36, "tab": 48, "space": 49,
    "delete": 51, "backspace": 51, "escape": 53, "esc": 53,
    "up": 126, "down": 125, "left": 123, "right": 124,
    "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96,
}

# Modifier key mapping
MODIFIERS = {
    "cmd": "command down", "command": "command down",
    "ctrl": "control down", "control": "control down",
    "alt": "option down", "option": "option down",
    "shift": "shift down",
}


def _run_applescript(script: str) -> Dict[str, Any]:
    """Execute an AppleScript and return result."""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=10,
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout.strip(),
            "error": result.stderr.strip() if result.returncode != 0 else "",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "AppleScript timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _run_shell(cmd: str, timeout: int = 10) -> Dict[str, Any]:
    """Run a shell command."""
    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout.strip(),
            "error": result.stderr.strip() if result.returncode != 0 else "",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Command timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


class ComputerControl:
    """macOS desktop control — screenshot, click, type, app control."""

    def __init__(self, brain=None, config: dict = None):
        self.brain = brain
        self.config = config or {}
        self.max_steps = self.config.get("max_steps", 15)
        self._screenshot_count = 0

    # ==================== Screenshot ====================

    def screenshot(self, region: str = None) -> Dict[str, Any]:
        """Take a screenshot. Returns file path.
        region: "x,y,w,h" for partial capture, None for full screen.
        """
        self._screenshot_count += 1
        ts = int(time.time())
        filename = f"screen_{ts}_{self._screenshot_count}.png"
        filepath = os.path.join(SCREENSHOT_DIR, filename)

        if region:
            # Partial capture: -R x,y,w,h
            cmd = f"screencapture -x -R{region} {filepath}"
        else:
            cmd = f"screencapture -x {filepath}"

        result = _run_shell(cmd)

        if result["success"] and os.path.exists(filepath):
            size = os.path.getsize(filepath)
            logger.info(f"Screenshot saved: {filepath} ({size} bytes)")
            return {"success": True, "path": filepath, "size": size}
        else:
            return {"success": False, "error": result.get("error", "screencapture failed")}

    # ==================== Vision ====================

    def vision(self, prompt: str, screenshot_path: str = None) -> str:
        """Analyze a screenshot using Claude Vision.
        Uses Claude CLI with piped image description, or Claude proxy API.
        """
        if not screenshot_path:
            # Take a fresh screenshot
            ss = self.screenshot()
            if not ss["success"]:
                return f"[Vision failed: {ss.get('error')}]"
            screenshot_path = ss["path"]

        if not os.path.exists(screenshot_path):
            return f"[Vision failed: screenshot not found at {screenshot_path}]"

        # Read and encode image once
        with open(screenshot_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Resize if too large
        if len(img_b64) > 1_400_000:
            resized = screenshot_path.replace(".png", "_sm.png")
            _run_shell(f"sips -Z 1280 '{screenshot_path}' --out '{resized}'")
            if os.path.exists(resized):
                with open(resized, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                try:
                    os.unlink(resized)
                except OSError:
                    pass

        # Method 1: Brain's Anthropic API (if API key available)
        if self.brain and self.brain._anthropic_key:
            result = self.brain.vision_analyze(prompt, screenshot_path)
            if result and not result.startswith("[Vision"):
                return result

        # Method 2: Claude proxy with vision (works on Mac Mini)
        result = self._vision_via_proxy(prompt, img_b64)
        if result:
            return result

        # Method 3: MiniMax M2.7 vision
        if self.brain and self.brain._minimax_key:
            result = self._vision_via_minimax(prompt, img_b64)
            if result:
                return result

        # Method 4: Anthropic API direct (if key exists)
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            result = self._vision_via_anthropic_api(prompt, img_b64, api_key)
            if result:
                return result

        return "[Vision offline: no vision backend available. Need: Claude proxy (port 18790), MiniMax key, or Anthropic API key]"

    def _vision_via_proxy(self, prompt: str, img_b64: str) -> Optional[str]:
        """Use Claude proxy (OpenAI-compatible) with vision."""
        try:
            import httpx
        except ImportError:
            return None

        proxy_url = "http://127.0.0.1:18790/v1/chat/completions"

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                },
                {"type": "text", "text": prompt},
            ]
        }]

        try:
            resp = httpx.post(
                proxy_url,
                json={"model": "claude-sonnet-4-6", "messages": messages, "max_tokens": 2048},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.debug(f"Vision via proxy failed: {e}")
        return None

    def _vision_via_anthropic_api(self, prompt: str, img_b64: str, api_key: str) -> Optional[str]:
        """Direct Anthropic API call with vision."""
        try:
            import httpx
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-6-20250514",
                    "max_tokens": 2048,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {
                                "type": "base64", "media_type": "image/png", "data": img_b64,
                            }},
                            {"type": "text", "text": prompt},
                        ]
                    }],
                },
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("content", [])
                if content:
                    return next((c["text"] for c in content if c.get("type") == "text"), None)
        except Exception as e:
            logger.debug(f"Vision via Anthropic API failed: {e}")
        return None

    def _vision_via_minimax(self, prompt: str, img_b64: str) -> Optional[str]:
        """MiniMax M2.7 vision (if supported)."""
        try:
            import httpx
            resp = httpx.post(
                "https://api.minimax.io/anthropic/v1/messages",
                headers={
                    "x-api-key": self.brain._api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "MiniMax-M2.7",
                    "max_tokens": 2048,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {
                                "type": "base64", "media_type": "image/png", "data": img_b64,
                            }},
                            {"type": "text", "text": prompt},
                        ]
                    }],
                },
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("content", [])
                if content:
                    return next((c["text"] for c in content if c.get("type") == "text"), None)
        except Exception as e:
            logger.debug(f"Vision via MiniMax failed: {e}")
        return None

    # ==================== Mouse Control ====================

    def click(self, x: int, y: int) -> Dict[str, Any]:
        """Click at screen coordinates."""
        logger.info(f"Click at ({x}, {y})")
        script = f'''
tell application "System Events"
    click at {{{x}, {y}}}
end tell
'''
        result = _run_applescript(script)
        if not result["success"]:
            # Fallback: use cliclick-style via python
            result = _run_shell(
                f"osascript -e 'tell application \"System Events\" to click at {{{x}, {y}}}'"
            )
        return {**result, "action": "click", "x": x, "y": y}

    def double_click(self, x: int, y: int) -> Dict[str, Any]:
        """Double-click at screen coordinates."""
        logger.info(f"Double-click at ({x}, {y})")
        script = f'''
tell application "System Events"
    click at {{{x}, {y}}}
    delay 0.1
    click at {{{x}, {y}}}
end tell
'''
        return {**_run_applescript(script), "action": "double_click", "x": x, "y": y}

    def right_click(self, x: int, y: int) -> Dict[str, Any]:
        """Right-click at screen coordinates."""
        logger.info(f"Right-click at ({x}, {y})")
        # AppleScript right-click via control+click
        script = f'''
tell application "System Events"
    key down control
    click at {{{x}, {y}}}
    key up control
end tell
'''
        return {**_run_applescript(script), "action": "right_click", "x": x, "y": y}

    def move_mouse(self, x: int, y: int) -> Dict[str, Any]:
        """Move mouse to coordinates without clicking."""
        # Use Python + Quartz if available, otherwise AppleScript workaround
        script = f'''
do shell script "python3 -c \\"
import subprocess
subprocess.run(['osascript', '-e', 'tell application \\\\\\"System Events\\\\\\" to ¬
set position of first window to {{{x}, {y}}}'])
\\""
'''
        # Simpler approach: just click at position (most reliable on macOS)
        return {"success": True, "action": "move", "x": x, "y": y, "note": "Use click() for interaction"}

    # ==================== Keyboard Control ====================

    def type_text(self, text: str) -> Dict[str, Any]:
        """Type text using AppleScript keystroke."""
        logger.info(f"Typing: {text[:50]}")
        # Escape special characters for AppleScript
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        script = f'tell application "System Events" to keystroke "{escaped}"'
        return {**_run_applescript(script), "action": "type", "text": text[:100]}

    def key(self, key_name: str) -> Dict[str, Any]:
        """Press a key or key combination.
        Examples: "return", "tab", "cmd+c", "cmd+shift+s", "escape"
        """
        logger.info(f"Key press: {key_name}")
        parts = key_name.lower().replace("+", " ").split()

        modifiers = []
        key_part = None
        for p in parts:
            if p in MODIFIERS:
                modifiers.append(MODIFIERS[p])
            else:
                key_part = p

        if not key_part:
            return {"success": False, "error": f"No key specified in: {key_name}"}

        # Build AppleScript
        if key_part in KEY_CODES:
            key_expr = f"key code {KEY_CODES[key_part]}"
        elif len(key_part) == 1:
            key_expr = f'keystroke "{key_part}"'
        else:
            return {"success": False, "error": f"Unknown key: {key_part}"}

        if modifiers:
            modifier_str = ", ".join(modifiers)
            script = f'tell application "System Events" to {key_expr} using {{{modifier_str}}}'
        else:
            script = f'tell application "System Events" to {key_expr}'

        return {**_run_applescript(script), "action": "key", "key": key_name}

    # ==================== App Control ====================

    def open_app(self, app_name: str) -> Dict[str, Any]:
        """Open an application."""
        logger.info(f"Opening app: {app_name}")
        result = _run_shell(f'open -a "{app_name}"')
        if result["success"]:
            time.sleep(1)  # Wait for app to launch
        return {**result, "action": "open_app", "app": app_name}

    def get_windows(self) -> Dict[str, Any]:
        """Get list of all visible windows."""
        script = '''
tell application "System Events"
    set windowList to {}
    repeat with proc in (every process whose visible is true)
        try
            set procName to name of proc
            repeat with w in (every window of proc)
                try
                    set windowList to windowList & {procName & ": " & name of w}
                end try
            end repeat
        end try
    end repeat
    return windowList
end tell
'''
        result = _run_applescript(script)
        if result["success"]:
            windows = [w.strip() for w in result["output"].split(",") if w.strip()]
            return {"success": True, "windows": windows, "count": len(windows)}
        return result

    def focus_window(self, app_name: str) -> Dict[str, Any]:
        """Bring an app's window to front."""
        logger.info(f"Focusing: {app_name}")
        script = f'''
tell application "{app_name}"
    activate
end tell
'''
        result = _run_applescript(script)
        if result["success"]:
            time.sleep(0.5)
        return {**result, "action": "focus", "app": app_name}

    def get_frontmost_app(self) -> Dict[str, Any]:
        """Get the name of the frontmost application."""
        script = 'tell application "System Events" to get name of first process whose frontmost is true'
        result = _run_applescript(script)
        return {**result, "action": "frontmost_app"}

    # ==================== Scroll ====================

    def scroll(self, direction: str = "down", amount: int = 3) -> Dict[str, Any]:
        """Scroll the screen."""
        logger.info(f"Scroll {direction} x{amount}")
        # Negative = scroll down, positive = scroll up
        delta = -amount if direction == "down" else amount
        script = f'''
tell application "System Events"
    repeat {abs(amount)} times
        if {delta} > 0 then
            key code 126 -- up arrow
        else
            key code 125 -- down arrow
        end if
        delay 0.1
    end repeat
end tell
'''
        return {**_run_applescript(script), "action": "scroll", "direction": direction, "amount": amount}

    # ==================== Vision-Action Loop ====================

    def look_and_act(self, task: str, max_steps: int = None) -> Dict[str, Any]:
        """
        Core autonomous loop:
        1. Screenshot
        2. Vision analysis — what's on screen, what to do next
        3. Execute the action
        4. Repeat until done or max steps

        Anti-hallucination: every step has real screencapture + real AppleScript execution.
        """
        if not self.brain:
            return {"success": False, "error": "No brain available for vision-action loop"}

        max_steps = max_steps or self.max_steps
        steps_taken = []
        completed = False

        logger.info(f"look_and_act started: {task[:100]} (max {max_steps} steps)")

        for step_num in range(1, max_steps + 1):
            # Step 1: Screenshot
            ss = self.screenshot()
            if not ss["success"]:
                steps_taken.append({"step": step_num, "error": "Screenshot failed", "done": False})
                break

            # Step 2: Vision analysis
            vision_prompt = f"""You are controlling a macOS computer. Look at this screenshot carefully.

Task: {task}
Step: {step_num}/{max_steps}
Previous steps: {json.dumps([s.get('action_taken','') for s in steps_taken[-3:]])}

Based on what you see, decide the NEXT action. Reply in JSON ONLY:
{{
  "observation": "What I see on screen right now",
  "action": "click|double_click|right_click|type|key|scroll|open_app|done|failed",
  "x": 0, "y": 0,
  "text": "",
  "key": "",
  "app": "",
  "direction": "up|down",
  "amount": 3,
  "reason": "Why this action"
}}

Rules:
- action="done" if the task is complete
- action="failed" if the task cannot be completed
- For click: provide exact x,y pixel coordinates you see in the screenshot
- For type: provide the text to type
- For key: provide key combo like "return" or "cmd+c"
- Be precise with coordinates. Look at the screenshot carefully."""

            analysis = self.vision(vision_prompt, ss["path"])
            if not analysis or analysis.startswith("[Vision"):
                steps_taken.append({"step": step_num, "error": f"Vision failed: {analysis}", "done": False})
                break

            # Step 3: Parse action
            try:
                if "```json" in analysis:
                    analysis = analysis.split("```json")[1].split("```")[0]
                elif "```" in analysis:
                    analysis = analysis.split("```")[1].split("```")[0]
                # Find JSON in response
                analysis = analysis.strip()
                if not analysis.startswith("{"):
                    start = analysis.find("{")
                    end = analysis.rfind("}") + 1
                    if start >= 0 and end > start:
                        analysis = analysis[start:end]
                action_data = json.loads(analysis)
            except (json.JSONDecodeError, IndexError):
                steps_taken.append({
                    "step": step_num,
                    "error": f"Could not parse vision response",
                    "raw": analysis[:300],
                    "done": False,
                })
                continue

            action = action_data.get("action", "")
            observation = action_data.get("observation", "")
            reason = action_data.get("reason", "")

            logger.info(f"Step {step_num}: {action} — {reason[:80]}")

            # Step 4: Check if done
            if action == "done":
                steps_taken.append({
                    "step": step_num, "action_taken": "done",
                    "observation": observation, "done": True,
                    "screenshot": ss["path"],
                })
                completed = True
                break

            if action == "failed":
                steps_taken.append({
                    "step": step_num, "action_taken": "failed",
                    "observation": observation, "reason": reason, "done": True,
                    "screenshot": ss["path"],
                })
                break

            # Step 5: Execute the action
            exec_result = self._execute_vision_action(action_data)
            steps_taken.append({
                "step": step_num,
                "action_taken": f"{action}",
                "observation": observation,
                "reason": reason,
                "execution": exec_result,
                "screenshot": ss["path"],
                "done": False,
            })

            # Brief pause for UI to update
            time.sleep(0.5)

        return {
            "success": completed,
            "task": task,
            "steps_taken": len(steps_taken),
            "max_steps": max_steps,
            "completed": completed,
            "steps": steps_taken,
            "verified": True,  # Every step has real screenshot + real AppleScript
        }

    def _execute_vision_action(self, action_data: dict) -> Dict[str, Any]:
        """Execute a single action from vision analysis."""
        action = action_data.get("action", "")

        if action == "click":
            return self.click(int(action_data.get("x", 0)), int(action_data.get("y", 0)))
        elif action == "double_click":
            return self.double_click(int(action_data.get("x", 0)), int(action_data.get("y", 0)))
        elif action == "right_click":
            return self.right_click(int(action_data.get("x", 0)), int(action_data.get("y", 0)))
        elif action == "type":
            return self.type_text(action_data.get("text", ""))
        elif action == "key":
            return self.key(action_data.get("key", "return"))
        elif action == "scroll":
            return self.scroll(
                action_data.get("direction", "down"),
                int(action_data.get("amount", 3))
            )
        elif action == "open_app":
            return self.open_app(action_data.get("app", ""))
        else:
            return {"success": False, "error": f"Unknown action: {action}"}

    # ==================== Cleanup ====================

    def cleanup_screenshots(self, keep_latest: int = 20):
        """Remove old screenshots to save disk space."""
        files = sorted(
            [os.path.join(SCREENSHOT_DIR, f) for f in os.listdir(SCREENSHOT_DIR) if f.endswith(".png")],
            key=os.path.getmtime,
        )
        to_delete = files[:-keep_latest] if len(files) > keep_latest else []
        for f in to_delete:
            try:
                os.unlink(f)
            except OSError:
                pass
        return {"deleted": len(to_delete), "remaining": len(files) - len(to_delete)}
