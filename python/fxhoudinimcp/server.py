"""FastMCP server definition for FXHoudini-MCP."""

from __future__ import annotations

# Built-in
import logging
import os
from contextlib import asynccontextmanager

# Third-party
from mcp.server.fastmcp import FastMCP

# Internal
from fxhoudinimcp.bridge import HoudiniBridge
from fxhoudinimcp._loader import load_markdown
from fxhoudinimcp._version import __version__

logger = logging.getLogger(__name__)


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
            "bridges": dict[int, HoudiniBridge],
        }
    """
    state = ctx.request_context.lifespan_context
    # Legacy / test path: {"bridge": <bridge or mock>}
    if isinstance(state, dict) and "bridge" in state:
        return state["bridge"]
    # Runtime path: full 115b state dict with per-port bridge registry
    port = state["active_port"]
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
