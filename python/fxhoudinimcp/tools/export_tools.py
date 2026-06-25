"""
export_tools.py — MCP wrapper tools for the engine-export member (PP12-111 PR-3/PR-4).

PR-3 (read-only / ungated, FR-10):
  - houdini_export_probe_versions: reports Houdini build, Labs VAT ROP version,
    ROP-Alembic/FBX availability, and optional skew_table verdict when target_ue given.
  - houdini_export_validate_budget: DRY-RUN budget check (reads geometry, writes nothing).

PR-4 (GATED / mutating):
  - houdini_export_vat: bake labs::vertex_animation_textures ROP → mesh + textures
    + ExportManifest sidecar. Registered with require_approval=True (PP12-109 gate).

Convention notes (grounded against kinefx_tools.py):
  - ctx: Context  (NOT ctx: Any — 4-bug convention class, bug #1)
  - _get_bridge(ctx)  (NOT _get_bridge() — bug #2)
  - bridge.execute(...)  (NOT bridge.call(...) — bug #3)
  - handler(**params) dispatching convention satisfied by bridge.execute (bug #4)
  - bridge.call() does NOT exist on HoudiniBridge -- never use it.
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context

from fxhoudinimcp.server import mcp, _get_bridge


@mcp.tool(meta={"require_approval": False})
async def houdini_export_probe_versions(
    ctx: Context,
    target_ue: str | None = None,
) -> dict[str, Any]:
    """Report Houdini build, Labs VAT ROP version, ROP availability, and optional skew verdict.

    Args:
        ctx:       MCP context (injected by FastMCP).
        target_ue: Optional Unreal Engine version string (e.g. "5.4").
                   When supplied, the result includes a 'skew' compatibility block.

    Returns:
        dict with keys: houdini, labs_vat_rop, rop_alembic, rop_fbx,
        and optionally 'skew' when target_ue is provided.
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute("probe_versions", {"target_ue": target_ue})


@mcp.tool(meta={"require_approval": True})
async def houdini_export_vat(
    ctx: Context,
    node: str,
    out_dir: str,
    export_type: str = "soft",
    asset_name: str | None = None,
    frame_range: list | None = None,
    target_ue: str | None = None,
) -> dict[str, Any]:
    """Bake a labs::vertex_animation_textures ROP (mesh + textures + manifest sidecar). GATED — mutating.

    Creates and cooks a VAT ROP, writes textures and mesh to out_dir, and
    emits an ExportManifest sidecar (.export.json) for downstream validation.

    Args:
        ctx:         MCP context (injected by FastMCP).
        node:        Houdini SOP node path (e.g. "/obj/geo1/box1").
        out_dir:     Output directory for textures, mesh, and sidecar.
        export_type: VAT mode — "soft" (default), "rigid", "fluid", or "sprite".
        asset_name:  Asset base name.  Derived from 'node' leaf when None.
        frame_range: [start, end] frame list.  Uses scene playbar range when None.
        target_ue:   Optional UE version string (e.g. "5.4") for skew annotation.

    Returns:
        dict with keys: ok, node, mesh, textures, sidecar, vat_version,
        version_triple.  On failure: ok=False, error=..., wrote_files=False.
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute("export_vat", {
        "node": node,
        "out_dir": out_dir,
        "export_type": export_type,
        "asset_name": asset_name,
        "frame_range": frame_range,
        "target_ue": target_ue,
    })


@mcp.tool(meta={"require_approval": True})
async def houdini_export_alembic_ue(
    ctx: Context,
    node: str,
    out_path: str,
    deforming: bool = True,
    frame_range: list | None = None,
) -> dict[str, Any]:
    """Bake an Alembic (.abc) via rop_alembic for Unreal Engine import. GATED — mutating.

    Creates and cooks a SOP-context rop_alembic ROP, writes the .abc file, and
    emits an ExportManifest sidecar (.export.json) for downstream validation.

    When deforming=True (default), packed_transform=0 (Deform Geometry) is used,
    preserving per-frame vertex positions for skeletal/deforming assets.
    When deforming=False, packed_transform=1 (Transform Geometry) is used,
    writing only the transform matrix per frame (rigid bodies).

    Args:
        ctx:         MCP context (injected by FastMCP).
        node:        Houdini SOP node path (e.g. "/obj/geo1/attribwrangle1").
        out_path:    Output .abc file path (e.g. "/tmp/out.abc").
        deforming:   True -> Deform Geometry (packed_transform=0, default);
                     False -> Transform Geometry (packed_transform=1).
        frame_range: [start, end] or [start, end, inc] frame list.
                     Uses scene playbar range when None.

    Returns:
        dict with keys: ok, node, out_path, sidecar, tool_version, manifest.
        On failure: ok=False, error=..., wrote_files=False.
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute("export_alembic_ue", {
        "node": node,
        "out_path": out_path,
        "deforming": deforming,
        "frame_range": frame_range,
    })


@mcp.tool(meta={"require_approval": True})
async def houdini_export_fbx(
    ctx: Context,
    node: str,
    out_path: str,
    frame_range: list | None = None,
) -> dict[str, Any]:
    """Bake an FBX (.fbx) via rop_fbx for Unreal Engine or other DCC import. GATED — mutating.

    Creates and cooks a SOP-context rop_fbx ROP, writes the .fbx file, and
    emits an ExportManifest sidecar (.export.json) for downstream validation.

    Args:
        ctx:         MCP context (injected by FastMCP).
        node:        Houdini SOP node path (e.g. "/obj/geo1/attribwrangle1").
        out_path:    Output .fbx file path (e.g. "/tmp/out.fbx").
        frame_range: [start, end] or [start, end, inc] frame list.
                     Uses scene playbar range when None.

    Returns:
        dict with keys: ok, node, out_path, sidecar, tool_version, manifest.
        On failure: ok=False, error=..., wrote_files=False.
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute("export_fbx", {
        "node": node,
        "out_path": out_path,
        "frame_range": frame_range,
    })


@mcp.tool(meta={"require_approval": False})
async def houdini_export_validate_budget(
    ctx: Context,
    node: str,
    target: str,
    budget_preset: str | None = None,
) -> dict[str, Any]:
    """DRY-RUN budget check against a SOP node's geometry.

    Reads geometry statistics via read-only introspection. Writes nothing.
    The result always has wrote_files=False.

    Args:
        ctx:           MCP context (injected by FastMCP).
        node:          Houdini SOP node path (e.g. "/obj/geo1/box1").
        target:        Export target; one of "vat", "alembic_ue", "fbx",
                       "niagara", "chaos_gc".
        budget_preset: Optional named budget preset (e.g. "ue_realtime").
                       Defaults to the UE_REALTIME preset when None.

    Returns:
        dict with keys: verdict, checks, wrote_files (always False).
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute(
        "validate_budget",
        {"node": node, "target": target, "budget_preset": budget_preset},
    )
