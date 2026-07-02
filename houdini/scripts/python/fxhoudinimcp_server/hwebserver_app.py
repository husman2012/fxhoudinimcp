"""hwebserver endpoint registration for the FXHoudini-MCP plugin.

Registers API functions on Houdini's built-in HTTP server that the
external MCP server communicates with over HTTP.

Calling convention (JSON-encoded RPC):
    POST /api
    Body: json=["mcp.execute", [], {"command": "...", "params": {...}, "request_id": "..."}]

hwebserver auto-serialises the returned dict/list to JSON.
"""

from __future__ import annotations

# Built-in
import os

# Third-party
import hwebserver

# Internal
from fxhoudinimcp_server import dispatcher


@hwebserver.apiFunction(namespace="mcp")
def execute(request, command="", params=None, request_id=""):
    """Single entry point for all MCP tool calls.

    Args:
        request: hwebserver.Request (always first arg).
        command: Dotted command name (e.g. "scene.get_scene_info").
        params: Tool-specific parameters dict.
        request_id: Correlation ID echoed back in the response.
    """
    if params is None:
        params = {}

    result = dispatcher.dispatch(command, params)
    result["request_id"] = request_id
    return result


@hwebserver.apiFunction(namespace="mcp")
def health(request):
    """Health check endpoint. Returns Houdini version, session info, and gate status."""
    import hou

    result = {
        "status": "ok",
        "houdini_version": hou.applicationVersionString(),
        "hip_file": hou.hipFile.name(),
        "pid": os.getpid(),
    }

    # Additive gate fields — present when gate is installed, omitted otherwise.
    # Failure here must not break the health endpoint (fail-open for health only).
    try:
        gate_attr = getattr(hou.session, "_fxhoudinimcp_gate", None)
        if gate_attr is not None:
            result["gate_mode"] = gate_attr.config.mode.value
            result["pending_count"] = len(gate_attr.queue.list())  # FIX-4: public TTL-purging API
    except Exception:  # noqa: BLE001
        pass  # health endpoint is fail-open — gate status is advisory

    return result


@hwebserver.apiFunction(namespace="mcp")
def list_commands(request):
    """List all registered command names for introspection."""
    return {"commands": dispatcher.list_commands()}
