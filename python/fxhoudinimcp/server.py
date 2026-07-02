"""FastMCP server definition for FXHoudini-MCP."""

from __future__ import annotations

# Built-in
import json
import logging
import os
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager

# Third-party
from mcp.server.fastmcp import FastMCP

# Internal
from fxhoudinimcp import session_model
from fxhoudinimcp.bridge import HoudiniBridge
from fxhoudinimcp.errors import ConnectionError as HoudiniConnectionError
from fxhoudinimcp._loader import load_markdown
from fxhoudinimcp._version import __version__

logger = logging.getLogger(__name__)

# PP12-115d: refuse-to-route-if-stale enforcement (ADR-0006). Borrowed v1
# tunable from fxhoudinimcp_server.startup._query_health's precedent.
_PROBE_TIMEOUT = 0.5


def _probe_pid(host: str, port: int, timeout: float = _PROBE_TIMEOUT) -> int | None:
    """Sync mcp.health probe -- return the live pid on this port, or None.

    Mirrors fxhoudinimcp_server.startup._query_health's MECHANISM (a POST of
    a form-encoded ``json=["mcp.health", [], {}]`` body to
    ``http://{host}:{port}/api``), but lives in the fxhoudinimcp package
    because fxhoudinimcp and fxhoudinimcp_server cannot import each other
    (ADR-0006 Sec 1.2/1.3).

    NEVER RAISES (ADR-0006 invariant 5): any failure along the way --
    connection refused, timeout, a non-dict JSON body, malformed JSON, a
    response whose .read() itself raises, or a missing/non-int 'pid' key --
    returns None rather than propagating.

    Args:
        host:    The Houdini hwebserver host.
        port:    The port to probe.
        timeout: Socket timeout in seconds; defaults to _PROBE_TIMEOUT.

    Returns:
        The int pid reported by mcp.health, or None on any failure.
    """
    url = f"http://{host}:{port}/api"
    body = urllib.parse.urlencode(
        {"json": json.dumps(["mcp.health", [], {}])}
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except Exception:
        return None

    try:
        data = json.loads(payload)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    pid = data.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool):
        return None
    return pid


def _get_bridge(ctx) -> HoudiniBridge:
    """Extract the HoudiniBridge for the active session from the MCP context.

    Backward-compatible: existing tool wrapper tests that inject
    ``{"bridge": <mock>}`` into the lifespan context continue to work
    because the new 115b state dict uses "bridges" (plural) and "active_port",
    not "bridge" (singular).  The isinstance check below distinguishes the
    legacy test shape from the runtime shape.

    PP12-115b: the runtime state dict has the shape::

        {
            "host": str,
            "base_port": int,
            "active_port": int,
            "active_pid": int | None,
            "bridges": dict[int, HoudiniBridge],
        }

    PP12-115d (ADR-0006): when ``active_pid`` is not None (a session was
    selected via ``houdini_select_session``), refuse to route if the live
    pid on ``active_port`` has drifted (Houdini restarted) or the port is
    unreachable (Houdini closed) -- raising ``ConnectionError`` instead of
    silently returning a bridge for a session that is no longer the one the
    caller selected.  When ``active_pid`` is None, this guard performs ZERO
    probing and ZERO enforcement -- routing stays byte-identical to before
    115d (the single most load-bearing invariant of this unit).
    """
    state = ctx.request_context.lifespan_context
    # Legacy / test path: {"bridge": <bridge or mock>}
    if isinstance(state, dict) and "bridge" in state:
        return state["bridge"]
    # Runtime path: full 115b state dict with per-port bridge registry
    port = state["active_port"]
    active_pid = state.get("active_pid")

    if active_pid is not None:
        live_pid = _probe_pid(state["host"], port, _PROBE_TIMEOUT)
        sessions = [{"port": port, "pid": live_pid}] if live_pid is not None else []
        if session_model.active_pid_stale(sessions, port, active_pid):
            reason = "unreachable" if live_pid is None else "drift"
            if live_pid is None:
                message = (
                    f"Session on port {port} is unreachable (selected pid "
                    f"{active_pid}; Houdini closed or not answering). "
                    f"Re-run houdini_select_session to re-bind."
                )
            else:
                message = (
                    f"Session on port {port} is stale (selected pid "
                    f"{active_pid}, live pid {live_pid} -- Houdini restarted). "
                    f"Re-run houdini_select_session to re-bind."
                )
            raise HoudiniConnectionError(
                message,
                details={
                    "port": port,
                    "expected_pid": active_pid,
                    "live_pid": live_pid,
                    "reason": reason,
                },
            )

    bridges = state["bridges"]
    if port not in bridges:
        bridges[port] = HoudiniBridge(host=state["host"], port=port)
    return bridges[port]


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Manage the Houdini bridge connection lifecycle.

    PP12-115b: yields a full state dict that supports multi-session routing.
    The base port bridge is created eagerly at startup; additional session
    bridges are created lazily by _get_bridge on first access.
    """
    host = os.getenv("HOUDINI_HOST", "localhost")
    base_port = int(os.getenv("HOUDINI_PORT", "8100"))

    base_bridge = HoudiniBridge(host=host, port=base_port)

    try:
        info = await base_bridge.health_check()
        logger.info(
            "Connected to Houdini %s", info.get("houdini_version", "unknown")
        )
    except Exception as e:
        logger.warning("Cannot reach Houdini at startup: %s", e)
        logger.warning("Tools will attempt to connect on first use.")

    state = {
        "host": host,
        "base_port": base_port,
        "active_port": base_port,
        "active_pid": None,
        "bridges": {base_port: base_bridge},
    }

    try:
        yield state
    finally:
        for bridge in list(state["bridges"].values()):
            await bridge.close()


mcp = FastMCP(
    name="FXHoudini",
    instructions=load_markdown("server_instructions.md"),
    lifespan=lifespan,
)
mcp._mcp_server.version = __version__
