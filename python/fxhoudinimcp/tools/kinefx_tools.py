"""MCP tool wrappers for the 3 read-only KineFX/APEX handlers (PP12-110 PR-2).

All three tools are:
- require_approval=False  (read-only — they never prompt/gate)
- Capability.READONLY on the handler side

The wrappers call bridge.call(command, **kwargs) — NOT bridge.execute.
"""

from __future__ import annotations

from typing import Any

from fxhoudinimcp.server import mcp, _get_bridge


@mcp.tool(meta={"require_approval": False})
async def kinefx_probe(ctx: Any, node_path: str = "/obj") -> dict[str, Any]:
    """Probe which KineFX/APEX node types exist at or under node_path.

    Returns a dict with Houdini version and a boolean map of
    the 7 canonical KineFX/APEX node types.

    Args:
        ctx: MCP request context (injected by FastMCP).
        node_path: Scene path to inspect (default ``/obj``).
    """
    bridge = _get_bridge()
    return await bridge.call("kinefx_probe", node_path=node_path)


@mcp.tool(meta={"require_approval": False})
async def query_skeleton(
    ctx: Any,
    node_path: str,
    frame: float | None = None,
) -> dict[str, Any]:
    """Read joint hierarchy and transforms from a cooked skeleton SOP.

    Serialises the skeleton using the §7.3 JSON shape::

        {"count": N, "joints": [{"name": ..., "parent": ..., "rest": {...}}, ...]}

    Args:
        ctx: MCP request context (injected by FastMCP).
        node_path: Path to the skeleton SOP node.
        frame: Optional frame number to sample; uses current frame if None.
    """
    bridge = _get_bridge()
    return await bridge.call("query_skeleton", node_path=node_path, frame=frame)


@mcp.tool(meta={"require_approval": False})
async def inspect_apex(ctx: Any, node_path: str) -> dict[str, Any]:
    """Summarise an APEX graph node — nodes, wires, control count.

    Returns::

        {
            "nodes": [{"name": ..., "node_type": ..., "ports": [...]}, ...],
            "wires": [{"src": ..., "dst": ...}, ...],
            "control_count": <int>
        }

    Args:
        ctx: MCP request context (injected by FastMCP).
        node_path: Path to the APEX graph node.
    """
    bridge = _get_bridge()
    return await bridge.call("inspect_apex", node_path=node_path)
