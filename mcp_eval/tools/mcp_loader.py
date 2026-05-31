"""MCP tool loader with real MCP client integration, timeout handling, and OAuth support."""

import asyncio
import hashlib
import json
import os
import webbrowser
from contextlib import AsyncExitStack
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse, parse_qs, urlencode, urljoin

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.auth.oauth2 import OAuthClientProvider
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientMetadata, OAuthClientInformationFull, OAuthToken

from .base import Tool, ToolResult

ConnectionType = Literal["stdio", "sse", "http", "streamable_http"]

_OAUTH_TOKENS_DIR = Path(".mcp_oauth_tokens")


class FileTokenStorage:
    """File-based OAuth token storage for a single MCP server.

    Saves tokens and client info as JSON files under .mcp_oauth_tokens/<server_name>/.
    """

    def __init__(self, server_name: str):
        self._dir = _OAUTH_TOKENS_DIR / server_name
        self._dir.mkdir(parents=True, exist_ok=True)

    async def get_tokens(self) -> OAuthToken | None:
        path = self._dir / "token.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return OAuthToken(**data)
        except Exception:
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        path = self._dir / "token.json"
        path.write_text(tokens.model_dump_json(indent=2), encoding="utf-8")

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        path = self._dir / "client_info.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return OAuthClientInformationFull(**data)
        except Exception:
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        path = self._dir / "client_info.json"
        path.write_text(client_info.model_dump_json(indent=2), encoding="utf-8")


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that captures the OAuth callback code."""

    auth_code: str | None = None
    state: str | None = None

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        _OAuthCallbackHandler.auth_code = params.get("code", [None])[0]
        _OAuthCallbackHandler.state = params.get("state", [None])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"<html><body><h2>Authorization successful! You can close this page.</h2></body></html>")

    def log_message(self, format, *args):
        pass


async def _open_browser_for_auth(url: str) -> None:
    """Open the OAuth authorization URL in the user's default browser."""
    print(f"Opening browser for OAuth authorization...\n  {url}")
    webbrowser.open(url)


async def _listen_for_oauth_callback(port: int = 8000) -> tuple[str, str | None]:
    """Start a temporary HTTP server on localhost to receive the OAuth callback.

    Returns (auth_code, state).
    """
    _OAuthCallbackHandler.auth_code = None
    _OAuthCallbackHandler.state = None

    server = HTTPServer(("127.0.0.1", port), _OAuthCallbackHandler)
    server.timeout = 300  # 5 minute max wait
    print(f"Waiting for OAuth callback on http://127.0.0.1:{port}/ ...")

    def _handle():
        server.handle_request()

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _handle)

    if _OAuthCallbackHandler.auth_code is None:
        raise RuntimeError("OAuth callback did not receive an authorization code")

    return _OAuthCallbackHandler.auth_code, _OAuthCallbackHandler.state


def _build_oauth_http_client(server_name: str, server_url: str, callback_port: int) -> httpx.AsyncClient:
    """Build an httpx.AsyncClient with OAuth support using stored tokens."""
    storage = FileTokenStorage(server_name)
    client_metadata = OAuthClientMetadata(
        redirect_uris=[f"http://127.0.0.1:{callback_port}/"],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
        client_name="mcp-eval",
        scope="tasks:read tasks:write",
    )

    provider = OAuthClientProvider(
        server_url=server_url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=_open_browser_for_auth,
        callback_handler=lambda: _listen_for_oauth_callback(callback_port),
    )

    return httpx.AsyncClient(auth=provider)


def _server_name_from_url(url: str) -> str:
    """Derive a filesystem-safe directory name from a URL for token storage."""
    parsed = urlparse(url)
    name = parsed.hostname or "unknown"
    return name.replace(".", "_").replace(":", "_")


@dataclass
class MCPTimeoutConfig:
    """MCP timeout configuration."""

    connect_timeout: float = 30.0
    execute_timeout: float = 60.0
    sse_read_timeout: float = 120.0


_default_timeout_config = MCPTimeoutConfig()


def set_mcp_timeout_config(
    connect_timeout: float | None = None,
    execute_timeout: float | None = None,
    sse_read_timeout: float | None = None,
) -> None:
    global _default_timeout_config
    if connect_timeout is not None:
        _default_timeout_config.connect_timeout = connect_timeout
    if execute_timeout is not None:
        _default_timeout_config.execute_timeout = execute_timeout
    if sse_read_timeout is not None:
        _default_timeout_config.sse_read_timeout = sse_read_timeout


def get_mcp_timeout_config() -> MCPTimeoutConfig:
    return _default_timeout_config


class MCPTool(Tool):
    """Wrapper for MCP tools with timeout handling."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        session: ClientSession,
        execute_timeout: float | None = None,
    ):
        self._name = name
        self._description = description
        self._parameters = parameters
        self._session = session
        self._execute_timeout = execute_timeout

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs) -> ToolResult:
        """Execute MCP tool via the session with timeout protection."""
        timeout = self._execute_timeout or _default_timeout_config.execute_timeout

        try:
            async with asyncio.timeout(timeout):
                result = await self._session.call_tool(self._name, arguments=kwargs)

            content_parts = []
            for item in result.content:
                if hasattr(item, "text"):
                    content_parts.append(item.text)
                else:
                    content_parts.append(str(item))

            content_str = "\n".join(content_parts)
            is_error = result.isError if hasattr(result, "isError") else False

            return ToolResult(success=not is_error, content=content_str, error=None if not is_error else "Tool returned error")

        except TimeoutError:
            return ToolResult(
                success=False,
                content="",
                error=f"MCP tool execution timed out after {timeout}s. The remote server may be slow or unresponsive.",
            )
        except Exception as e:
            return ToolResult(success=False, content="", error=f"MCP tool execution failed: {str(e)}")


class MCPServerConnection:
    """Manages connection to a single MCP server (STDIO or URL-based) with timeout handling."""

    def __init__(
        self,
        name: str,
        connection_type: ConnectionType = "stdio",
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        connect_timeout: float | None = None,
        execute_timeout: float | None = None,
        sse_read_timeout: float | None = None,
        oauth_enabled: bool = False,
        oauth_callback_port: int = 8000,
    ):
        self.name = name
        self.connection_type = connection_type
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.url = url
        self.headers = headers or {}
        self.connect_timeout = connect_timeout
        self.execute_timeout = execute_timeout
        self.sse_read_timeout = sse_read_timeout
        self.oauth_enabled = oauth_enabled
        self.oauth_callback_port = oauth_callback_port
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack | None = None
        self.tools: list[MCPTool] = []

    def _get_connect_timeout(self) -> float:
        return self.connect_timeout or _default_timeout_config.connect_timeout

    def _get_sse_read_timeout(self) -> float:
        return self.sse_read_timeout or _default_timeout_config.sse_read_timeout

    def _get_execute_timeout(self) -> float:
        return self.execute_timeout or _default_timeout_config.execute_timeout

    async def connect(self) -> bool:
        """Connect to the MCP server with timeout protection."""
        connect_timeout = self._get_connect_timeout()

        try:
            self.exit_stack = AsyncExitStack()

            async with asyncio.timeout(connect_timeout):
                if self.connection_type == "stdio":
                    read_stream, write_stream = await self._connect_stdio()
                elif self.connection_type == "sse":
                    read_stream, write_stream = await self._connect_sse()
                else:  # http / streamable_http
                    read_stream, write_stream = await self._connect_streamable_http()

                session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
                self.session = session

                await session.initialize()
                tools_list = await session.list_tools()

            execute_timeout = self._get_execute_timeout()
            for tool in tools_list.tools:
                parameters = tool.inputSchema if hasattr(tool, "inputSchema") else {}
                mcp_tool = MCPTool(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=parameters,
                    session=session,
                    execute_timeout=execute_timeout,
                )
                self.tools.append(mcp_tool)

            conn_info = self.url if self.url else self.command
            print(f"Connected to MCP server '{self.name}' ({self.connection_type}: {conn_info}) - loaded {len(self.tools)} tools")
            return True

        except TimeoutError:
            print(f"Connection to MCP server '{self.name}' timed out after {connect_timeout}s")
            if self.exit_stack:
                await self.exit_stack.aclose()
                self.exit_stack = None
            return False

        except Exception as e:
            print(f"Failed to connect to MCP server '{self.name}': {e}")
            if self.exit_stack:
                await self.exit_stack.aclose()
                self.exit_stack = None
            import traceback
            traceback.print_exc()
            return False

    async def _connect_stdio(self):
        server_params = StdioServerParameters(command=self.command, args=self.args, env=self.env if self.env else None)
        return await self.exit_stack.enter_async_context(stdio_client(server_params))

    async def _connect_sse(self):
        connect_timeout = self._get_connect_timeout()
        sse_read_timeout = self._get_sse_read_timeout()

        return await self.exit_stack.enter_async_context(
            sse_client(
                url=self.url,
                headers=self.headers if self.headers else None,
                timeout=connect_timeout,
                sse_read_timeout=sse_read_timeout,
            )
        )

    async def _connect_streamable_http(self):
        """Connect via Streamable HTTP transport.

        When OAuth is enabled, creates an httpx.AsyncClient with OAuthClientProvider
        that transparently handles the Authorization Code + PKCE flow on 401 responses.
        """
        if self.oauth_enabled and self.url:
            http_client = _build_oauth_http_client(self.name, self.url, self.oauth_callback_port)
            read_stream, write_stream, _ = await self.exit_stack.enter_async_context(
                streamable_http_client(url=self.url, http_client=http_client)
            )
        else:
            # Fallback to deprecated API for non-OAuth connections (backward compat)
            connect_timeout = self._get_connect_timeout()
            sse_read_timeout = self._get_sse_read_timeout()
            from mcp.client.streamable_http import streamablehttp_client
            read_stream, write_stream, _ = await self.exit_stack.enter_async_context(
                streamablehttp_client(
                    url=self.url,
                    headers=self.headers if self.headers else None,
                    timeout=connect_timeout,
                    sse_read_timeout=sse_read_timeout,
                )
            )
        return read_stream, write_stream

    async def disconnect(self):
        if self.exit_stack:
            try:
                await asyncio.shield(self.exit_stack.aclose())
            except (Exception, asyncio.CancelledError):
                pass
            finally:
                self.exit_stack = None
                self.session = None


_mcp_connections: list[MCPServerConnection] = []


def _determine_connection_type(server_config: dict) -> ConnectionType:
    explicit_type = server_config.get("type", "").lower()
    if explicit_type in ("stdio", "sse", "http", "streamable_http"):
        return explicit_type
    if server_config.get("url"):
        return "streamable_http"
    return "stdio"


def _resolve_mcp_config_path(config_path: str) -> Path | None:
    config_file = Path(config_path)
    if config_file.exists():
        return config_file
    if config_file.name == "mcp.json":
        example_file = config_file.parent / "mcp-example.json"
        if example_file.exists():
            print(f"mcp.json not found, using template: {example_file}")
            return example_file
    return None


async def load_mcp_tools_async(config_path: str = "mcp.json") -> list[Tool]:
    """Load MCP tools from all servers in config file.

    Supports STDIO, SSE, HTTP, and Streamable HTTP transports.
    For Streamable HTTP servers, OAuth can be enabled via the "oauth" config block.
    """
    global _mcp_connections

    config_file = _resolve_mcp_config_path(config_path)

    if config_file is None:
        print(f"MCP config not found: {config_path}")
        return []

    try:
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)

        mcp_servers = config.get("mcpServers", {})

        if not mcp_servers:
            print("No MCP servers configured")
            return []

        all_tools = []

        for server_name, server_config in mcp_servers.items():
            if server_config.get("disabled", False):
                print(f"Skipping disabled server: {server_name}")
                continue

            conn_type = _determine_connection_type(server_config)
            url = server_config.get("url")
            command = server_config.get("command")

            if conn_type == "stdio" and not command:
                print(f"No command specified for STDIO server: {server_name}")
                continue
            if conn_type in ("sse", "http", "streamable_http") and not url:
                print(f"No url specified for {conn_type.upper()} server: {server_name}")
                continue

            # Parse OAuth config
            oauth_cfg = server_config.get("oauth", {})
            oauth_enabled = oauth_cfg.get("enabled", False) if isinstance(oauth_cfg, dict) else False
            oauth_callback_port = oauth_cfg.get("callback_port", 8000) if isinstance(oauth_cfg, dict) else 8000

            connection = MCPServerConnection(
                name=server_name,
                connection_type=conn_type,
                command=command,
                args=server_config.get("args", []),
                env=server_config.get("env", {}),
                url=url,
                headers=server_config.get("headers", {}),
                connect_timeout=server_config.get("connect_timeout"),
                execute_timeout=server_config.get("execute_timeout"),
                sse_read_timeout=server_config.get("sse_read_timeout"),
                oauth_enabled=oauth_enabled,
                oauth_callback_port=oauth_callback_port,
            )
            success = await connection.connect()

            if success:
                _mcp_connections.append(connection)
                all_tools.extend(connection.tools)

        return all_tools

    except Exception as e:
        print(f"Error loading MCP config: {e}")
        import traceback
        traceback.print_exc()
        return []


async def cleanup_mcp_connections():
    """Clean up all MCP connections."""
    global _mcp_connections
    for connection in _mcp_connections:
        try:
            await connection.disconnect()
        except Exception:
            pass
    _mcp_connections.clear()


def get_connected_server_names() -> list[str]:
    """Return names of all successfully connected MCP servers."""
    return [conn.name for conn in _mcp_connections]


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = os.urandom(32).hex()
    challenge = hashlib.sha256(verifier.encode("ascii")).digest()
    import base64
    challenge_b64 = base64.urlsafe_b64encode(challenge).rstrip(b"=").decode("ascii")
    return verifier, challenge_b64


def _generate_state() -> str:
    return os.urandom(32).hex()


async def login_mcp_server(config_path: str, server_name: str | None = None) -> None:
    """Perform OAuth login for an MCP server, saving tokens for later use.

    This manually implements the OAuth Authorization Code + PKCE flow:
    1. Discover OAuth metadata from /.well-known/oauth-authorization-server
    2. Dynamically register a client (POST /register)
    3. Open browser for user authorization
    4. Listen for callback on localhost
    5. Exchange authorization code for token (POST /token)
    6. Save client_info + token to .mcp_oauth_tokens/<server_name>/
    """
    config_file = _resolve_mcp_config_path(config_path)
    if config_file is None:
        print(f"MCP config not found: {config_path}")
        return

    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)

    mcp_servers = config.get("mcpServers", {})

    # Filter to target server
    if server_name:
        if server_name not in mcp_servers:
            print(f"MCP server '{server_name}' not found. Available: {list(mcp_servers.keys())}")
            return
        target = {server_name: mcp_servers[server_name]}
    else:
        # Auto-select first server with OAuth enabled
        target = {}
        for name, cfg in mcp_servers.items():
            oauth_cfg = cfg.get("oauth", {})
            if isinstance(oauth_cfg, dict) and oauth_cfg.get("enabled"):
                target = {name: cfg}
                break
        if not target:
            print("No OAuth-enabled MCP server found in config.")
            return

    srv_name, srv_config = next(iter(target.items()))
    url = srv_config.get("url")
    if not url:
        print(f"Server '{srv_name}' has no URL configured.")
        return

    oauth_cfg = srv_config.get("oauth", {})
    callback_port = oauth_cfg.get("callback_port", 8000) if isinstance(oauth_cfg, dict) else 8000

    print(f"Logging in to MCP server: {srv_name} ({url})")

    # Step 1: Discover OAuth metadata
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    well_known_url = f"{base_url}/.well-known/oauth-authorization-server"

    print(f"Discovering OAuth metadata from {well_known_url}...")
    async with httpx.AsyncClient() as client:
        resp = await client.get(well_known_url)
        if resp.status_code != 200:
            print(f"Failed to discover OAuth metadata: {resp.status_code}")
            return
        oauth_meta = resp.json()

    authorization_endpoint = oauth_meta.get("authorization_endpoint")
    token_endpoint = oauth_meta.get("token_endpoint")
    registration_endpoint = oauth_meta.get("registration_endpoint")

    if not authorization_endpoint or not token_endpoint:
        print("OAuth metadata missing required endpoints.")
        return

    print(f"  Authorization: {authorization_endpoint}")
    print(f"  Token:         {token_endpoint}")

    # Step 2: Dynamic Client Registration (if available)
    client_id = None
    client_secret = None
    if registration_endpoint:
        print(f"Registering client at {registration_endpoint}...")
        reg_data = {
            "redirect_uris": [f"http://127.0.0.1:{callback_port}/"],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
            "client_name": "mcp-eval",
        }
        async with httpx.AsyncClient() as client:
            reg_resp = await client.post(registration_endpoint, json=reg_data)
            if reg_resp.status_code not in (200, 201):
                print(f"Client registration failed: {reg_resp.status_code} {reg_resp.text}")
                return
            reg_result = reg_resp.json()
            client_id = reg_result.get("client_id")
            client_secret = reg_result.get("client_secret")
            print(f"  Registered client_id: {client_id}")
    else:
        print("No registration endpoint found. Cannot proceed without client_id.")
        return

    # Step 3: Generate PKCE + build authorization URL
    code_verifier, code_challenge = _generate_pkce()
    state = _generate_state()

    scope = reg_result.get("scope", oauth_meta.get("scopes_supported", ["tasks:read tasks:write"]))
    if isinstance(scope, list):
        scope = " ".join(scope)

    auth_params = {
        "client_id": client_id,
        "redirect_uri": f"http://127.0.0.1:{callback_port}/",
        "response_type": "code",
        "scope": scope,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    auth_url = f"{authorization_endpoint}?{urlencode(auth_params)}"

    # Step 4: Open browser + listen for callback
    print(f"\nOpening browser for authorization...")
    webbrowser.open(auth_url)

    _OAuthCallbackHandler.auth_code = None
    _OAuthCallbackHandler.state = None
    server = HTTPServer(("127.0.0.1", callback_port), _OAuthCallbackHandler)
    server.timeout = 300

    print(f"Waiting for callback on http://127.0.0.1:{callback_port}/ ...")

    def _handle():
        server.handle_request()

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _handle)

    if _OAuthCallbackHandler.auth_code is None:
        print("Did not receive authorization code.")
        return

    if _OAuthCallbackHandler.state != state:
        print("State mismatch — possible CSRF attack. Aborting.")
        return

    auth_code = _OAuthCallbackHandler.auth_code
    print(f"Received authorization code.")

    # Step 5: Exchange code for token
    print("Exchanging code for token...")
    token_data = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": f"http://127.0.0.1:{callback_port}/",
        "client_id": client_id,
        "code_verifier": code_verifier,
    }
    if client_secret:
        token_data["client_secret"] = client_secret

    async with httpx.AsyncClient() as client:
        token_resp = await client.post(token_endpoint, data=token_data)
        if token_resp.status_code != 200:
            print(f"Token exchange failed: {token_resp.status_code} {token_resp.text}")
            return
        token_result = token_resp.json()

    # Step 6: Save to FileTokenStorage
    storage = FileTokenStorage(srv_name)

    oauth_token = OAuthToken(
        access_token=token_result["access_token"],
        token_type=token_result.get("token_type", "Bearer"),
        expires_in=token_result.get("expires_in"),
        scope=token_result.get("scope"),
        refresh_token=token_result.get("refresh_token"),
    )
    await storage.set_tokens(oauth_token)

    client_info = OAuthClientInformationFull(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uris=[f"http://127.0.0.1:{callback_port}/"],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
        client_name="mcp-eval",
        scope=scope,
    )
    await storage.set_client_info(client_info)

    print(f"\nLogin successful! Token saved to {storage._dir}/")
    print(f"  access_token:  ...{oauth_token.access_token[-8:]}")
    if oauth_token.refresh_token:
        print(f"  refresh_token: ...{oauth_token.refresh_token[-8:]}")
    if oauth_token.expires_in:
        print(f"  expires_in:    {oauth_token.expires_in}s")
