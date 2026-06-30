"""MCP wrappers: houdini_usd_inspect_layer, houdini_usd_validate.

Both wrappers are READ-ONLY, UNGATED (require_approval=False,
Capability.READONLY) — FR-10.  Each wrapper delegates to the correspondingly
named handler registered on the Houdini side via bridge.execute.  No logic
lives here; all domain logic is in the USD handlers accessed Houdini-side
through the handlers.

PP12-112 / pp12-112b (houdini_usd_inspect_layer, houdini_usd_validate)

Contract: imports NO hou, NO pxr — this module must be importable off-DCC
for the wrapper pytest suite (CL-015).
"""
from __future__ import annotations

from mcp.server.fastmcp import Context

import fxhoudinimcp.server as _fxserver

# mcp is used by the @mcp.tool() decorator at module import time.
mcp = _fxserver.mcp


@mcp.tool(meta={"require_approval": False})
async def houdini_usd_inspect_layer(
    ctx: Context,
    node_or_layer: str,
) -> dict:
    """Inspect a USD layer from a LOP node or file path and return its summary.

    Returns the §4.2 LayerSummary shape on success::

        {
            "ok": True,
            "default_prim": str | None,
            "root_prims": [str, ...],
            "sublayers": [str, ...],
            "current_format": str,
            "has_mtlx_material": bool,
        }

    or an FR-2/FR-5 error shape on failure::

        {"ok": False, "error": "<reason>"}

    Args:
        ctx: MCP lifespan context — injected by FastMCP; hidden from client schema.
        node_or_layer: Houdini LOP node path (e.g. ``"/stage/lop1"``) or a
            USD file path; supports Houdini variable expansions such as
            ``"$HIP/out.usda"``.
    """
    # Access _get_bridge through the module reference so that
    # `patch("fxhoudinimcp.server._get_bridge", ...)` intercepts it correctly
    # in tests (a local import would cache the original function object).
    bridge = _fxserver._get_bridge(ctx)
    return await bridge.execute(
        "usd_inspect_layer",
        {"node_or_layer": node_or_layer},
    )


@mcp.tool(meta={"require_approval": False})
async def houdini_usd_validate(
    ctx: Context,
    target: str,
    out_path: str | None = None,
    actual_format: str | None = None,
    texture_paths: list | None = None,
    checks: list | None = None,
) -> dict:
    """Run USD discipline checks against a layer summary.

    Returns a B-1 compliant validation shape on success::

        {
            "ok": True,
            "mode": "minimal" | "preflight" | "postwrite",
            "omitted_checks": [str, ...],
            "verdict": str,
            "checks": [{"id": str, "status": str, "msg": str?}, ...],
            "wrote_files": bool,
        }

    or an FR-2/FR-5 error shape on failure::

        {"ok": False, "error": "<reason>"}

    Args:
        ctx: MCP lifespan context — injected by FastMCP; hidden from client schema.
        target: Houdini LOP node path or USD file path; supports Houdini
            variable expansions (e.g. ``"$HIP/out.usda"``).
        out_path: Intended output file path — enables preflight mode when set
            without *actual_format*, or postwrite mode when both are set.
            Default: ``None`` (minimal mode).
        actual_format: Format string from magic-byte detection (postwrite
            mode; requires *out_path*).  Default: ``None``.
        texture_paths: List of texture file path strings for the absolute-path
            portability check (postwrite mode).  Default: ``None``.
        checks: Reserved — must be ``None``; non-None triggers a fail-loud
            rejection. Default: ``None``.
    """
    bridge = _fxserver._get_bridge(ctx)
    return await bridge.execute(
        "usd_validate",
        {
            "target": target,
            "out_path": out_path,
            "actual_format": actual_format,
            "texture_paths": texture_paths,
            "checks": checks,
        },
    )
