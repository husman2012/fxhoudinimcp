"""MCP wrappers: render_lint_settings, render_parse_exr, render_read_pixels,
render_compare.

All wrappers are READ-ONLY, UNGATED (require_approval=False,
Capability.READONLY) — FR-10.  Each wrapper delegates to the correspondingly
named handler registered on the Houdini side via bridge.execute.  No logic
lives here; all domain logic is in the homedini engine / OIIO reader accessed
Houdini-side through the handlers.

PP12-114 / pp12-114c (render_lint_settings, render_parse_exr)
PP12-114 / pp12-114e (render_read_pixels)
PP12-114 / pp12-114f (render_compare)
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


@mcp.tool(meta={"require_approval": False})
async def render_parse_exr(
    ctx: Context,
    exr_path: str,
    subimage: int | None = None,
) -> dict:
    """Parse an EXR file via hoiiotool and return its channel/metadata manifest.

    Returns the §4.2 ExrManifest shape::

        {
            "exr_path": str,
            "is_multipart": bool,
            "subimages": int,
            "compression": str,
            "xres": int,
            "yres": int,
            "channels": [{"name": str, "layer": str | None, "dtype": str}, ...],
            "crypto_layers": [...],
            "metadata": {str: str},
        }

    or an FR-2/FR-5 error shape if the path is invalid or hoiiotool fails::

        {"ok": False, "error": "<reason>"}

    Args:
        ctx: MCP lifespan context — injected by FastMCP; hidden from client schema.
        exr_path: Path to the EXR file; supports Houdini variable expansion.
        subimage: When set, inspect only this subimage index.  ``None`` (default)
            inspects all subimages.
    """
    bridge = _fxserver._get_bridge(ctx)
    return await bridge.execute(
        "render_parse_exr",
        {"exr_path": exr_path, "subimage": subimage},
    )


@mcp.tool(meta={"require_approval": False})
async def render_read_pixels(
    ctx: Context,
    source: str,
    plane: str = "C",
    mode: str = "summary",
    roi: list[int] | None = None,
    max_pixels: int = 4096,
    downsample: int = 1,
    page: int = 0,
    page_size: int = 1024,
) -> dict:
    """Read pixel data from an on-disk EXR file via OIIO.

    Returns the §4.2 ReadbackResult shape::

        {
            "ok": True,
            "xres": int,
            "yres": int,
            "channels": int,
            "dtype": str,
            "mode": str,
            "plane": str,
            "pixels": [...],
        }

    or an FR-2/FR-5 error shape on failure::

        {"ok": False, "error": "<reason>"}

    Args:
        ctx: MCP lifespan context — injected by FastMCP; hidden from client schema.
        source: Path to the EXR file; supports Houdini variable expansion
            (e.g. ``"$HIP/render/beauty.0001.exr"``).  In-scene COP node
            paths are not yet supported (EXR-source v1).
        plane: AOV plane name.  ``"C"`` or ``"beauty"`` select the top-level
            beauty channels (R/G/B with no dot in name).  Default: ``"C"``.
        mode: Readback mode — ``"summary"`` (metadata only, no pixel data),
            ``"sample"`` (spaced sample of pixels), or ``"roi"``
            (``[x0, y0, x1, y1]`` half-open bounding-box slice).
        roi: ``[x0, y0, x1, y1]`` half-open bounding box for ``mode="roi"``.
        max_pixels: Maximum pixel count before auto-downsampling.  Default: 4096.
        downsample: Manual downsample factor (1 = no downsampling).
        page: Page index for paginated reads.
        page_size: Page size in pixels.  Default: 1024.
    """
    bridge = _fxserver._get_bridge(ctx)
    return await bridge.execute(
        "render_read_pixels",
        {
            "source": source,
            "plane": plane,
            "mode": mode,
            "roi": roi,
            "max_pixels": max_pixels,
            "downsample": downsample,
            "page": page,
            "page_size": page_size,
        },
    )


@mcp.tool(meta={"require_approval": False})
async def render_compare(
    ctx: Context,
    a: str,
    b: str,
    planes: list[str] | None = None,
    metric: str = "stats",
) -> dict:
    """Compare two EXR renders A and B plane-by-plane and return comparison metrics.

    Returns the §4.2 CompareReport shape **directly** on success (no ``ok``
    key — callers must NOT gate on ``result.get("ok")`` being True)::

        {
            "aovs_a": [...],
            "aovs_b": [...],
            "common": [...],
            "selected": [...],
            "per_plane": {plane: {metric_key: value, ...}, ...},
        }

    or an FR-2/FR-5 error shape on failure::

        {"ok": False, "error": "<reason>"}

    Args:
        ctx: MCP lifespan context — injected by FastMCP; hidden from client schema.
        a: Path to render A; supports Houdini variable expansion.  In-scene
            COP node paths are not yet supported (EXR-source v1).
        b: Path to render B; same constraints as ``a``.
        planes: List of AOV plane names to compare.  ``None`` (default) compares
            all planes common to both renders.
        metric: Comparison metric — ``"stats"``, ``"mae"``, or ``"psnr"``.
            Validated before file access.  Default: ``"stats"``.
    """
    bridge = _fxserver._get_bridge(ctx)
    return await bridge.execute(
        "render_compare",
        {"a": a, "b": b, "planes": planes, "metric": metric},
    )
