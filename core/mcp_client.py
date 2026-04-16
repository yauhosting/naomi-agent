"""
NAOMI Agent - MCP (Model Context Protocol) Client

Connects to external MCP servers (Notion, Slack, databases, etc.)
via JSON-RPC over stdio transport. Each server runs as a subprocess
with stdin/stdout pipes for communication.
"""
import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("naomi.mcp")

# Protocol constants
MCP_PROTOCOL_VERSION = "2024-11-05"
MCP_CLIENT_NAME = "naomi"
MCP_CLIENT_VERSION = "0.5.0"
DEFAULT_TIMEOUT = 30.0


@dataclass
class MCPTool:
    """Represents a tool exposed by an MCP server."""
    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class MCPServer:
    """A connected MCP server subprocess."""
    name: str
    process: asyncio.subprocess.Process
    tools: List[MCPTool] = field(default_factory=list)
    request_id: int = 0
    connected_at: float = field(default_factory=time.time)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def next_id(self) -> int:
        self.request_id += 1
        return self.request_id


class MCPClientError(Exception):
    """Base exception for MCP client errors."""


class MCPTimeoutError(MCPClientError):
    """Server did not respond within timeout."""


class MCPServerCrashError(MCPClientError):
    """Server process terminated unexpectedly."""


class MCPClient:
    """
    Client for connecting to MCP servers via stdio transport.

    Each server runs as a child process. Communication uses JSON-RPC 2.0
    messages over stdin (requests) and stdout (responses).
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._servers: Dict[str, MCPServer] = {}
        self._timeout = timeout

    # -- Internal protocol helpers --

    @staticmethod
    def _build_request(method: str, params: Optional[Dict[str, Any]] = None,
                       request_id: int = 1) -> bytes:
        """Build a JSON-RPC 2.0 request message."""
        msg: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            msg["params"] = params
        return (json.dumps(msg) + "\n").encode("utf-8")

    async def _send_and_receive(self, server: MCPServer, method: str,
                                params: Optional[Dict[str, Any]] = None,
                                timeout: Optional[float] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request and wait for the matching response."""
        timeout = timeout or self._timeout

        if server.process.returncode is not None:
            raise MCPServerCrashError(
                f"Server '{server.name}' process exited with code {server.process.returncode}"
            )

        async with server._lock:
            req_id = server.next_id()
            request = self._build_request(method, params, req_id)

            try:
                server.process.stdin.write(request)
                await server.process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise MCPServerCrashError(
                    f"Server '{server.name}' pipe broken: {exc}"
                ) from exc

            # Read lines until we find our response (skip notifications)
            try:
                response = await asyncio.wait_for(
                    self._read_response(server, req_id),
                    timeout=timeout,
                )
            except asyncio.TimeoutError as exc:
                raise MCPTimeoutError(
                    f"Server '{server.name}' did not respond to '{method}' "
                    f"within {timeout}s"
                ) from exc

        return response

    async def _read_response(self, server: MCPServer,
                             expected_id: int) -> Dict[str, Any]:
        """Read stdout lines until a JSON-RPC response with the expected id arrives."""
        while True:
            if server.process.returncode is not None:
                raise MCPServerCrashError(
                    f"Server '{server.name}' exited while waiting for response"
                )

            raw = await server.process.stdout.readline()
            if not raw:
                raise MCPServerCrashError(
                    f"Server '{server.name}' closed stdout"
                )

            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Non-JSON line from '%s': %s", server.name, line[:200])
                continue

            # Skip notifications (no id field)
            if "id" not in msg:
                logger.debug("Notification from '%s': %s", server.name, msg.get("method", ""))
                continue

            if msg.get("id") == expected_id:
                if "error" in msg:
                    err = msg["error"]
                    raise MCPClientError(
                        f"Server '{server.name}' returned error: "
                        f"[{err.get('code', '?')}] {err.get('message', str(err))}"
                    )
                return msg.get("result", {})

            # Response for a different id -- discard and keep reading
            logger.debug(
                "Discarding response id=%s (expected %s) from '%s'",
                msg.get("id"), expected_id, server.name,
            )

    # -- Public API --

    async def connect(self, server_name: str, command: str,
                      args: Optional[List[str]] = None,
                      env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Start an MCP server subprocess and perform the initialization handshake.

        Returns the server's capabilities from the initialize response.
        """
        if server_name in self._servers:
            logger.warning("Server '%s' already connected -- disconnecting first", server_name)
            await self.disconnect(server_name)

        merged_env = {**os.environ, **(env or {})}
        full_args = [command] + (args or [])

        logger.info("Starting MCP server '%s': %s", server_name, " ".join(full_args))

        try:
            process = await asyncio.create_subprocess_exec(
                *full_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=merged_env,
            )
        except FileNotFoundError as exc:
            raise MCPClientError(
                f"Command not found for server '{server_name}': {command}"
            ) from exc
        except OSError as exc:
            raise MCPClientError(
                f"Failed to start server '{server_name}': {exc}"
            ) from exc

        server = MCPServer(name=server_name, process=process)
        self._servers[server_name] = server

        # Initialize handshake
        try:
            init_result = await self._send_and_receive(server, "initialize", {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": MCP_CLIENT_NAME,
                    "version": MCP_CLIENT_VERSION,
                },
            })
        except (MCPClientError, MCPTimeoutError):
            await self._kill_process(process)
            del self._servers[server_name]
            raise

        # Send initialized notification (no response expected)
        notif = (json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }) + "\n").encode("utf-8")
        try:
            process.stdin.write(notif)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass

        # Auto-discover tools
        try:
            await self._refresh_tools(server)
        except MCPClientError as exc:
            logger.warning("Could not list tools for '%s': %s", server_name, exc)

        tool_count = len(server.tools)
        logger.info(
            "MCP server '%s' connected (%d tools available)",
            server_name, tool_count,
        )

        return {
            "server_name": server_name,
            "capabilities": init_result,
            "tools_count": tool_count,
            "tools": [t.name for t in server.tools],
        }

    async def _refresh_tools(self, server: MCPServer) -> None:
        """Fetch the tool list from a server and cache it."""
        result = await self._send_and_receive(server, "tools/list")
        tools_raw = result.get("tools", [])
        server.tools = [
            MCPTool(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", t.get("input_schema", {})),
            )
            for t in tools_raw
        ]

    async def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """List available tools from a connected server."""
        server = self._get_server(server_name)
        # Refresh on demand
        await self._refresh_tools(server)
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in server.tools
        ]

    async def call_tool(self, server_name: str, tool_name: str,
                        arguments: Optional[Dict[str, Any]] = None,
                        timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Call a tool on a connected server and return the result.

        Returns the full result dict from the server's response.
        """
        server = self._get_server(server_name)

        logger.info("Calling tool '%s/%s' with %s", server_name, tool_name,
                     json.dumps(arguments or {})[:200])

        result = await self._send_and_receive(
            server, "tools/call",
            {"name": tool_name, "arguments": arguments or {}},
            timeout=timeout,
        )

        logger.debug("Tool '%s/%s' result: %s", server_name, tool_name,
                      json.dumps(result)[:300])
        return result

    async def disconnect(self, server_name: str) -> None:
        """Stop a connected MCP server."""
        server = self._servers.pop(server_name, None)
        if server is None:
            logger.warning("Server '%s' not found -- nothing to disconnect", server_name)
            return

        logger.info("Disconnecting MCP server '%s'", server_name)
        await self._kill_process(server.process)

    def list_servers(self) -> List[Dict[str, Any]]:
        """List all connected servers and their tools."""
        result = []
        for name, srv in self._servers.items():
            alive = srv.process.returncode is None
            result.append({
                "name": name,
                "alive": alive,
                "tools": [t.name for t in srv.tools],
                "tools_count": len(srv.tools),
                "connected_at": srv.connected_at,
                "uptime_seconds": round(time.time() - srv.connected_at, 1),
            })
        return result

    async def disconnect_all(self) -> None:
        """Disconnect all servers. Call on shutdown."""
        names = list(self._servers.keys())
        for name in names:
            try:
                await self.disconnect(name)
            except Exception as exc:
                logger.error("Error disconnecting '%s': %s", name, exc)

    async def auto_connect_from_config(
        self, config_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Read an MCP config file and connect to all defined servers.

        Searches in order:
        1. Provided config_path
        2. ~/.naomi/mcp_servers.json
        3. ~/.claude/claude_desktop_config.json

        Config format (same as Claude Desktop):
        {
          "mcpServers": {
            "server-name": {
              "command": "npx",
              "args": ["-y", "@some/mcp-server"],
              "env": {"API_KEY": "..."}
            }
          }
        }
        """
        search_paths = []
        if config_path:
            search_paths.append(Path(config_path))
        search_paths.extend([
            Path.home() / ".naomi" / "mcp_servers.json",
            Path.home() / ".claude" / "claude_desktop_config.json",
        ])

        config_data = None
        used_path = None
        for p in search_paths:
            if p.exists():
                try:
                    config_data = json.loads(p.read_text(encoding="utf-8"))
                    used_path = p
                    break
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Failed to read config '%s': %s", p, exc)

        if config_data is None:
            logger.info("No MCP config found in: %s", [str(p) for p in search_paths])
            return []

        servers_config = config_data.get("mcpServers", {})
        if not servers_config:
            logger.info("No mcpServers defined in %s", used_path)
            return []

        logger.info("Loading %d MCP servers from %s", len(servers_config), used_path)

        results = []
        for name, cfg in servers_config.items():
            command = cfg.get("command", "")
            args = cfg.get("args", [])
            env = cfg.get("env", {})

            if not command:
                logger.warning("Server '%s' has no command -- skipping", name)
                continue

            try:
                result = await self.connect(name, command, args, env)
                results.append(result)
            except MCPClientError as exc:
                logger.error("Failed to connect server '%s': %s", name, exc)
                results.append({
                    "server_name": name,
                    "error": str(exc),
                    "tools_count": 0,
                    "tools": [],
                })

        return results

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        Get a flat list of all tools from all connected servers.
        Useful for presenting to the LLM as available tool definitions.
        """
        all_tools = []
        for name, srv in self._servers.items():
            if srv.process.returncode is not None:
                continue
            for tool in srv.tools:
                all_tools.append({
                    "server": name,
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                })
        return all_tools

    def get_anthropic_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Convert all MCP tools to Anthropic tool_use format for direct
        injection into Brain.call_with_tools().
        """
        definitions = []
        for name, srv in self._servers.items():
            if srv.process.returncode is not None:
                continue
            for tool in srv.tools:
                definitions.append({
                    "name": f"mcp__{name}__{tool.name}",
                    "description": f"[MCP:{name}] {tool.description}",
                    "input_schema": tool.input_schema or {
                        "type": "object", "properties": {},
                    },
                })
        return definitions

    # -- Helpers --

    def _get_server(self, server_name: str) -> MCPServer:
        """Get a connected server or raise."""
        server = self._servers.get(server_name)
        if server is None:
            raise MCPClientError(f"Server '{server_name}' is not connected")
        if server.process.returncode is not None:
            raise MCPServerCrashError(
                f"Server '{server_name}' process exited with code "
                f"{server.process.returncode}"
            )
        return server

    @staticmethod
    async def _kill_process(process: asyncio.subprocess.Process) -> None:
        """Gracefully terminate a subprocess."""
        if process.returncode is not None:
            return
        try:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        except ProcessLookupError:
            pass
