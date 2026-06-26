"""MCP wrapper: render_lint_settings — Karma render graph pre-render linting.

READ-ONLY, UNGATED (require_approval=False, Capability.READONLY) — FR-10.

The wrapper delegates to the handler registered as "render_lint_settings"
via bridge.execute.  No logic lives here; all lint logic is in the
homedini handoff_linter engine, accessed Houdini-side through the handler.

PP12-114 / pp12-114c — unitId: pp12-114c
"""
from __future__ import annotations

from mcp.server.fastmcp import Context

import fxhoudinimcp.server as _fxserver

# mcp is used by the @mcp.tool() decorator at module import time.
mcp = _fxserver.mcp


@mcp.tool(meta={"require_approval": False})
async def render_lint_settings(
    ctx: Context,
    render_node: str,
    preset: str = "nuke_safe",
) -> dict:
    """Read a Karma render node's USD stage and run handoff_linter rules on it.

    Returns the §4.2 result shape::

        {
            "render_node": str,
            "preset": str,
            "results": [RuleResult.to_dict(), ...],
            "summary": {"ok": int, "warn": int, "error": int},
            "ready_to_render": bool,
        }

    or an FR-2 error shape if the node is missing or invalid::

        {"ok": False, "error": "<reason>"}

    Args:
        ctx: MCP lifespan context — injected by FastMCP; hidden from client schema.
        render_node: Scene path of the Karma render node (e.g. ``"/stage/karma1"``).
        preset: Rule preset name to evaluate. Defaults to ``"nuke_safe"``.
    """
    # Access _get_bridge through the module reference so that
    # `patch("fxhoudinimcp.server._get_bridge", ...)` intercepts it correctly
    # in tests (the local import would cache the original function object).
    bridge = _fxserver._get_bridge(ctx)
    return await bridge.execute(
        "render_lint_settings",
        {"render_node": render_node, "preset": preset},
    )
