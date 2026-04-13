"""
NAOMI Agent - Capability Discovery Engine
Autonomously discovers, evaluates, and installs:
- Python packages (pip)
- System tools (brew, npm)
- MCP servers (for Claude CLI integration)
- Claude Code skills (ECC marketplace)

Triggered by:
1. Task failure (missing tool/capability)
2. Idle-time self-improvement
3. Explicit /discover command
"""
import os
import json
import time
import logging
import subprocess
from typing import Dict, List, Any, Optional

logger = logging.getLogger("naomi.discovery")

# Well-known MCP servers NAOMI can self-install
KNOWN_MCP_SERVERS = {
    "web-search": {
        "package": "@anthropic-ai/mcp-server-web-search",
        "command": "npx",
        "args": ["-y", "@anthropic-ai/mcp-server-web-search"],
        "env_keys": ["ANTHROPIC_API_KEY"],
        "description": "Web search via Anthropic API",
        "category": "search",
    },
    "filesystem": {
        "package": "@anthropic-ai/mcp-server-filesystem",
        "command": "npx",
        "args": ["-y", "@anthropic-ai/mcp-server-filesystem", "/Users/yokowai"],
        "env_keys": [],
        "description": "File system operations",
        "category": "files",
    },
    "github": {
        "package": "@modelcontextprotocol/server-github",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env_keys": ["GITHUB_PERSONAL_ACCESS_TOKEN"],
        "description": "GitHub PRs, issues, repos",
        "category": "dev",
    },
    "memory": {
        "package": "@modelcontextprotocol/server-memory",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "env_keys": [],
        "description": "Persistent memory across sessions",
        "category": "memory",
    },
    "sequential-thinking": {
        "package": "@modelcontextprotocol/server-sequential-thinking",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        "env_keys": [],
        "description": "Chain-of-thought reasoning",
        "category": "thinking",
    },
    "context7": {
        "package": "@anthropic-ai/mcp-server-context7",
        "command": "npx",
        "args": ["-y", "@upstash/context7-mcp@latest"],
        "env_keys": [],
        "description": "Library documentation lookup",
        "category": "docs",
    },
    "exa": {
        "package": "exa-mcp-server",
        "command": "npx",
        "args": ["-y", "exa-mcp-server"],
        "env_keys": ["EXA_API_KEY"],
        "description": "Neural web search via Exa",
        "category": "search",
    },
    "playwright": {
        "package": "@anthropic-ai/mcp-server-playwright",
        "command": "npx",
        "args": ["-y", "@anthropic-ai/mcp-server-playwright"],
        "env_keys": [],
        "description": "Browser automation",
        "category": "browser",
    },
}

# Well-known Python packages for common needs
KNOWN_PACKAGES = {
    "image": ["Pillow", "opencv-python"],
    "audio": ["pydub", "soundfile"],
    "video": ["moviepy"],
    "pdf": ["PyPDF2", "reportlab"],
    "scraping": ["beautifulsoup4", "requests", "httpx"],
    "data": ["pandas", "numpy"],
    "chart": ["matplotlib", "plotly"],
    "api": ["fastapi", "uvicorn", "flask"],
    "telegram": ["python-telegram-bot"],
    "crypto": ["ccxt"],
    "stock": ["yfinance"],
    "ai": ["openai", "anthropic"],
    "schedule": ["apscheduler", "schedule"],
    "email": ["aiosmtplib"],
}

# MCP config path for Claude Code
MCP_CONFIG_PATH = os.path.expanduser("~/.claude/mcp-configs/mcp-servers.json")


class CapabilityDiscovery:
    """Autonomously discovers and installs capabilities NAOMI needs."""

    def __init__(self, brain, memory, actions, project_dir: str):
        self.brain = brain
        self.memory = memory
        self.actions = actions
        self.project_dir = project_dir
        self._installed_mcp = set()
        self._scan_installed_mcp()

    def _scan_installed_mcp(self):
        """Scan what MCP servers are already configured."""
        if os.path.exists(MCP_CONFIG_PATH):
            try:
                with open(MCP_CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                self._installed_mcp = set(config.get("mcpServers", {}).keys())
                logger.info(f"Found {len(self._installed_mcp)} installed MCP servers")
            except Exception as e:
                logger.warning(f"Failed to read MCP config: {e}")

    # === Core: Detect what's needed ===
    def detect_needs(self, task: str, error: str = "") -> Dict[str, Any]:
        """Ask brain to identify what capabilities are needed for a task."""
        prompt = f"""Analyze this task and identify what tools/packages/capabilities are needed.

Task: {task}
{"Error: " + error if error else ""}

Available tools: shell, python3, git, curl, pip3, node, npm, ffmpeg, yt-dlp
Installed MCP servers: {', '.join(self._installed_mcp) if self._installed_mcp else 'none'}

Reply in JSON only:
{{
  "needed_packages": ["package1"],
  "needed_tools": ["tool1"],
  "needed_mcp": ["mcp-name"],
  "can_proceed": true,
  "explanation": "..."
}}

Known MCP servers: {', '.join(KNOWN_MCP_SERVERS.keys())}
Known package categories: {', '.join(KNOWN_PACKAGES.keys())}
If no special tools needed, return empty lists and can_proceed=true."""

        response = self.brain._think(prompt)
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response.strip())
        except (json.JSONDecodeError, IndexError):
            return {"needed_packages": [], "needed_tools": [], "needed_mcp": [],
                    "can_proceed": True, "explanation": ""}

    # === Install Python packages ===
    def install_package(self, package: str) -> Dict[str, Any]:
        """Install a Python package and register as skill."""
        logger.info(f"Installing package: {package}")
        result = self.actions.execute_shell(f"pip3 install {package}")
        if result.get("success"):
            self.memory.learn_skill(
                name=package,
                description=f"Python package: {package}",
                tool_command=f"import {package.replace('-', '_')}",
                install_command=f"pip3 install {package}",
            )
            logger.info(f"Package installed: {package}")
        else:
            logger.warning(f"Package install failed: {package}")
        return result

    def install_packages_for_category(self, category: str) -> List[Dict]:
        """Install all packages in a category."""
        packages = KNOWN_PACKAGES.get(category, [])
        results = []
        for pkg in packages:
            results.append({"package": pkg, **self.install_package(pkg)})
        return results

    # === Install system tools ===
    def install_tool(self, tool: str) -> Dict[str, Any]:
        """Install a system tool via brew or npm."""
        logger.info(f"Installing tool: {tool}")

        # Try brew first
        result = self.actions.execute_shell(f"brew install {tool}")
        if result.get("success"):
            self.memory.learn_skill(
                name=tool, description=f"System tool: {tool}",
                install_command=f"brew install {tool}",
            )
            return result

        # Try npm
        result = self.actions.execute_shell(f"bash -lc 'npm install -g {tool}'")
        if result.get("success"):
            self.memory.learn_skill(
                name=tool, description=f"NPM tool: {tool}",
                install_command=f"npm install -g {tool}",
            )
            return result

        return {"success": False, "error": f"Could not install {tool} via brew or npm"}

    # === Install MCP servers ===
    def install_mcp(self, name: str) -> Dict[str, Any]:
        """Install an MCP server into Claude Code config."""
        if name in self._installed_mcp:
            return {"success": True, "message": f"MCP '{name}' already installed"}

        server_info = KNOWN_MCP_SERVERS.get(name)
        if not server_info:
            # Try searching for it
            return self._search_and_install_mcp(name)

        # Check if required env vars are set
        missing_env = []
        env_config = {}
        for key in server_info.get("env_keys", []):
            val = os.environ.get(key, "")
            if not val:
                missing_env.append(key)
            else:
                env_config[key] = val

        if missing_env:
            return {
                "success": False,
                "error": f"Missing env vars: {', '.join(missing_env)}",
                "hint": f"Set these in ~/.env or export them: {' '.join(missing_env)}",
            }

        # Build MCP entry
        mcp_entry = {
            "command": server_info["command"],
            "args": server_info["args"],
            "description": server_info["description"],
        }
        if env_config:
            mcp_entry["env"] = env_config

        # Read existing config
        config = {"mcpServers": {}}
        if os.path.exists(MCP_CONFIG_PATH):
            try:
                with open(MCP_CONFIG_PATH, 'r') as f:
                    config = json.load(f)
            except Exception:
                pass

        # Add new server
        config.setdefault("mcpServers", {})[name] = mcp_entry

        # Write back
        os.makedirs(os.path.dirname(MCP_CONFIG_PATH), exist_ok=True)
        with open(MCP_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)

        self._installed_mcp.add(name)

        # Also install the npm package
        pkg = server_info.get("package", "")
        if pkg:
            self.actions.execute_shell(f"bash -lc 'npm install -g {pkg}'")

        self.memory.learn_skill(
            name=f"mcp-{name}",
            description=f"MCP Server: {server_info['description']}",
            tool_command=f"MCP:{name}",
            install_command=f"npm install -g {pkg}",
        )
        self.memory.remember_long(
            f"MCP Installed: {name}",
            f"Installed MCP server '{name}': {server_info['description']}",
            category="capability", importance=8,
        )

        logger.info(f"MCP server installed: {name}")
        return {"success": True, "name": name, "description": server_info["description"]}

    def _search_and_install_mcp(self, query: str) -> Dict[str, Any]:
        """Search for an MCP server by name/description on the web."""
        logger.info(f"Searching for MCP server: {query}")
        search_result = self.actions.web_search(f"MCP server {query} npm modelcontextprotocol")
        if not search_result.get("success") or not search_result.get("results"):
            return {"success": False, "error": f"No MCP server found for '{query}'"}

        # Ask brain to pick the best result
        results_text = "\n".join(
            f"- {r.get('title','')}: {r.get('body','')[:100]}"
            for r in search_result["results"][:5]
        )
        pick_prompt = f"""I'm looking for an MCP server for: {query}

Search results:
{results_text}

If you can identify an npm package for this MCP server, reply in JSON:
{{"package": "@scope/package-name", "description": "what it does", "env_keys": []}}

If no suitable MCP server found, reply: {{"package": null}}"""

        response = self.brain._think(pick_prompt)
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            data = json.loads(response.strip())
        except (json.JSONDecodeError, IndexError):
            return {"success": False, "error": f"Could not parse MCP search result for '{query}'"}

        if not data.get("package"):
            return {"success": False, "error": f"No MCP server found for '{query}'"}

        # Install the discovered package
        pkg = data["package"]
        desc = data.get("description", query)

        mcp_entry = {
            "command": "npx",
            "args": ["-y", pkg],
            "description": desc,
        }

        config = {"mcpServers": {}}
        if os.path.exists(MCP_CONFIG_PATH):
            try:
                with open(MCP_CONFIG_PATH, 'r') as f:
                    config = json.load(f)
            except Exception:
                pass

        server_name = query.replace(" ", "-").lower()
        config.setdefault("mcpServers", {})[server_name] = mcp_entry

        with open(MCP_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)

        self._installed_mcp.add(server_name)
        self.actions.execute_shell(f"bash -lc 'npm install -g {pkg}'")

        self.memory.learn_skill(
            name=f"mcp-{server_name}",
            description=f"MCP Server: {desc}",
            tool_command=f"MCP:{server_name}",
            install_command=f"npm install -g {pkg}",
        )

        logger.info(f"MCP server discovered and installed: {server_name} ({pkg})")
        return {"success": True, "name": server_name, "package": pkg, "description": desc}

    # === Auto-resolve: called when a task fails ===
    async def auto_resolve(self, task: str, error: str) -> Dict[str, Any]:
        """Automatically detect what's missing and install it."""
        logger.info(f"Auto-resolving: {task[:100]} (error: {error[:100]})")

        needs = self.detect_needs(task, error)
        installed = []

        # Install packages
        for pkg in needs.get("needed_packages", []):
            result = self.install_package(pkg)
            installed.append({"type": "package", "name": pkg, **result})

        # Install tools
        for tool in needs.get("needed_tools", []):
            result = self.install_tool(tool)
            installed.append({"type": "tool", "name": tool, **result})

        # Install MCP servers
        for mcp in needs.get("needed_mcp", []):
            result = self.install_mcp(mcp)
            installed.append({"type": "mcp", "name": mcp, **result})

        return {
            "resolved": len([i for i in installed if i.get("success")]),
            "failed": len([i for i in installed if not i.get("success")]),
            "details": installed,
            "can_retry": needs.get("can_proceed", True),
        }

    # === Idle discovery: proactive capability enhancement ===
    def idle_discover(self) -> Dict[str, Any]:
        """During idle time, check for useful capabilities to add."""
        # Review recent failed tasks
        recent_tasks = self.memory.get_recent_tasks(20)
        failed = [t for t in recent_tasks if t.get("status") == "failed"]

        if not failed:
            return {"action": "none", "reason": "No recent failures"}

        # Ask brain what capabilities would help
        failures_text = "\n".join(f"- {t['task'][:80]}: {(t.get('result',''))[:100]}" for t in failed[:5])
        prompt = f"""These tasks failed recently:
{failures_text}

Currently installed: pip3, git, node, npm, python3, ffmpeg, yt-dlp
Installed MCP: {', '.join(self._installed_mcp) or 'none'}

What tools/packages/MCP servers should I install to handle these better?
Reply JSON: {{"packages": ["pkg"], "tools": ["tool"], "mcp": ["name"], "priority": "high/medium/low"}}
Only suggest what's directly needed. Return empty lists if nothing missing."""

        response = self.brain._think(prompt)
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            suggestions = json.loads(response.strip())
        except (json.JSONDecodeError, IndexError):
            return {"action": "none", "reason": "Could not parse suggestions"}

        # Auto-install high priority items
        if suggestions.get("priority") == "high":
            installed = []
            for pkg in suggestions.get("packages", [])[:3]:
                installed.append(self.install_package(pkg))
            for mcp in suggestions.get("mcp", [])[:2]:
                installed.append(self.install_mcp(mcp))
            return {"action": "installed", "details": installed}

        # Remember suggestions for later
        if any(suggestions.get(k) for k in ["packages", "tools", "mcp"]):
            self.memory.remember_short(
                f"Capability suggestion: {json.dumps(suggestions)[:200]}",
                category="suggestion",
            )
            return {"action": "suggested", "suggestions": suggestions}

        return {"action": "none", "reason": "No new capabilities needed"}

    # === Status ===
    def get_status(self) -> Dict[str, Any]:
        skills = self.memory.recall_skill()
        return {
            "installed_mcp": sorted(self._installed_mcp),
            "known_mcp": sorted(KNOWN_MCP_SERVERS.keys()),
            "available_categories": sorted(KNOWN_PACKAGES.keys()),
            "learned_skills": len(skills) if isinstance(skills, list) else 0,
        }
