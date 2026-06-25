"""sessions.py — MCP tools for multi-session discovery and routing (PP12-115b).

Three components:

_scan_sessions(host, base_port, max_ports) -> list
    Internal async helper.  Probes base_port .. base_port + max_ports - 1
    via temporary HoudiniBridge instances; returns a session entry per live
    port; dead ports are silently skipped.  max_ports is clamped to [1, 64].

houdini_list_sessions(ctx, max_ports=16) -> dict
    Scans ports base_port .. base_port + max_ports - 1 via _scan_sessions.
    Returns::

        {
            "sessions": [<entry>, ...],   # active=True on current active_port
            "active_port": <int>,
        }

houdini_select_session(ctx, port=None, hip=None) -> dict
    Selects a session by port number OR hip-file substring via
    session_model.resolve_target.  Exactly one of port/hip must be supplied.
    On success, updates lifespan_context["active_port"] and returns::

        {"ok": True, "session": <entry>, "active_port": <int>}

    On failure (no match, ambiguous match, or no args), returns::

        {"ok": False, "error": <str>, "active_port": <int>}
    WITHOUT changing active_port.

Both MCP tools:
  - @mcp.tool(meta={"require_approval": False})  — READ-ONLY, UNGATED
  - Bridge instances created here are for discovery only; they are
    closed in a finally block and are NOT added to state["bridges"].  Only
    _get_bridge()'s lazy-create path adds bridges to the registry.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context

from fxhoudinimcp.server import mcp
from fxhoudinimcp.bridge import HoudiniBridge
from fxhoudinimcp.errors import ConnectionError as HoudiniConnectionError
from fxhoudinimcp import session_model

_DEFAULT_MAX_PORTS = 16


async def _scan_sessions(host, base_port, max_ports):
    """Probe base_port..base_port+max_ports-1; return a session entry per live port."""
    max_ports = max(1, min(max_ports, 64))   # R3: clamp to a sane range
    entries = []
    for p in range(base_port, base_port + max_ports):
        b = HoudiniBridge(host, p)
        try:
            health = await b.health_check()
            entries.append(session_model.build_session_entry(p, health))
        except HoudiniConnectionError:
            pass
        finally:
            await b.close()
    return entries


@mcp.tool(meta={"require_approval": False})
async def houdini_list_sessions(
    ctx: Context,
    max_ports: int = _DEFAULT_MAX_PORTS,
) -> dict[str, Any]:
    """Scan adjacent ports and return all live Houdini sessions.

    Probes ports from base_port up to base_port + max_ports - 1.  Each
    live port is converted into a session entry via session_model; dead
    ports are silently skipped.

    Args:
        ctx:       MCP context (injected by FastMCP).
        max_ports: Number of consecutive ports to scan starting at base_port.
                   Defaults to 16.

    Returns:
        dict with keys:
          - "sessions": list of session-entry dicts, each with an "active" bool
          - "active_port": the currently selected port
    """
    state = ctx.request_context.lifespan_context
    active_port = state["active_port"]

    entries = await _scan_sessions(state["host"], state["base_port"], max_ports)

    return {
        "sessions": session_model.mark_active(entries, active_port),
        "active_port": active_port,
        "active_pid": state.get("active_pid"),
        "active_pid_stale": session_model.active_pid_stale(entries, active_port, state.get("active_pid")),
    }


@mcp.tool(meta={"require_approval": False})
async def houdini_select_session(
    ctx: Context,
    port: int | None = None,
    hip: str | None = None,
) -> dict[str, Any]:
    """Switch the active Houdini session, selecting by port number or hip-file name.

    Scans all live sessions then uses session_model.resolve_target to locate
    a unique match.  On success, updates lifespan_context["active_port"] and
    returns the session entry.  On failure (no match, ambiguous match, no
    selector supplied), returns an error dict WITHOUT changing active_port.

    Args:
        ctx:  MCP context (injected by FastMCP).
        port: The port number of the Houdini session to select.
        hip:  Substring of the hip filename to match (case-insensitive).

    Returns:
        On success::
            {"ok": True, "session": <session-entry-dict>, "active_port": <int>}
        On failure::
            {"ok": False, "error": <str>, "active_port": <int>}
    """
    state = ctx.request_context.lifespan_context
    if port is not None and hip is not None:
        return {"ok": False, "error": "select_session takes exactly one of 'port' or 'hip', not both", "active_port": state["active_port"]}
    selector = port if port is not None else hip
    if selector is None:
        return {"ok": False, "error": "select_session requires 'port' or 'hip'", "active_port": state["active_port"]}
    entries = await _scan_sessions(state["host"], state["base_port"], _DEFAULT_MAX_PORTS)
    target, reason = session_model.resolve_with_reason(entries, selector)
    if target is None:
        if reason == "ambiguous":
            err = f"ambiguous: multiple live sessions match {selector!r} — pass a more specific hip substring or a port"
        else:
            err = f"no live session matches {selector!r}"
        return {"ok": False, "error": err, "active_port": state["active_port"]}
    state["active_port"] = target
    entry = next((e for e in entries if e["port"] == target), None)
    state["active_pid"] = entry.get("pid") if entry else None
    return {"ok": True, "session": entry, "active_port": target, "active_pid": state["active_pid"]}
