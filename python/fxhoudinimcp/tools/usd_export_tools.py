"""MCP wrappers: houdini_usd_inspect_layer, houdini_usd_validate,
houdini_usd_export_layer, houdini_usd_export_rop.

houdini_usd_inspect_layer / houdini_usd_validate are READ-ONLY, UNGATED
(require_approval=False, Capability.READONLY) — FR-10.

houdini_usd_export_layer is GATED (require_approval=True, Capability.MUTATING)
— the FIRST mutating tool in this family. It writes a composed USD layer to
disk via Sdf.Layer.Export(). A SINGLE bridge.execute call (matching every
other shipped gated wrapper, e.g. houdini_export_vat) — the pre-flight
validation lives server-side in the gate's preview_fn and the post-write
validation is INLINE in the handler; the wrapper does no result
interpretation and returns bridge.execute's result VERBATIM, including a
pending-approval / preview response shape from the 109 gate (that is NOT a
failure — it must not be reinterpreted or swallowed).

houdini_usd_export_rop is GATED (require_approval=True, Capability.MUTATING)
— the SECOND mutating tool in this family. It drives the /out-context `usd`
ROP to write a chosen LOP node's composed stage to disk (current frame or a
[start, end] range) — NOT a full-time-history flatten like
houdini_usd_export_layer. Same SINGLE bridge.execute, verbatim-passthrough
contract as houdini_usd_export_layer.

Each wrapper delegates to the correspondingly named handler registered on the
Houdini side via bridge.execute.  No logic lives here; all domain logic is in
the USD handlers accessed Houdini-side through the handlers.

PP12-112 / pp12-112b (houdini_usd_inspect_layer, houdini_usd_validate)
PP12-112 / pp12-112c (houdini_usd_export_layer)
PP12-112 / pp12-112d (houdini_usd_export_rop)

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


@mcp.tool(meta={"require_approval": True})
async def houdini_usd_export_layer(
    ctx: Context,
    node: str,
    out_path: str,
    flatten: bool = False,
    default_prim: str | None = None,
) -> dict:
    """Write a composed USD layer to disk via Sdf.Layer.Export(). GATED — mutating.

    First mutating tool of the usd_export family (require_approval=True,
    PP12-109 security gate). Format is chosen by *out_path*'s file EXTENSION
    (.usda -> ascii, .usdc/.usd -> crate, .usdz -> packaged). Injects NO
    /World or /root wrapper. Pre-flight validation is shown in the gate's
    preview before approval; a post-write usd_validate is embedded in the
    result under 'validator_post'.

    A SINGLE bridge.execute call — the wrapper performs no result
    interpretation and returns bridge.execute's result VERBATIM, including a
    pending-approval / preview response shape from the 109 gate (that is a
    normal, valid return value, not an error).

    Args:
        ctx: MCP lifespan context — injected by FastMCP; hidden from client schema.
        node: Houdini LOP node path (e.g. "/stage/lop1") or a USD file path;
            supports Houdini variable expansions such as "$HIP/out.usda".
        out_path: Output file path; supports Houdini variable expansions
            (e.g. "$HIP/out.usdc").
        flatten: When True, export a flattened single-layer composition.
            When False (default), export the stage's root layer as-is.
        default_prim: Optional prim path to set as the layer's defaultPrim
            before export. Default: None (unset).
    """
    # Access _get_bridge through the module reference so that
    # `patch("fxhoudinimcp.server._get_bridge", ...)` intercepts it correctly
    # in tests (a local import would cache the original function object).
    bridge = _fxserver._get_bridge(ctx)
    return await bridge.execute(
        "usd_export_layer",
        {
            "node_path": node,
            "out_path": out_path,
            "flatten": flatten,
            "default_prim": default_prim,
        },
    )


@mcp.tool(meta={"require_approval": True})
async def houdini_usd_export_rop(
    ctx: Context,
    lop_node: str,
    out_path: str,
    frame_range: list | None = None,
) -> dict:
    """Drive the /out `usd` ROP to write lop_node's composed stage to disk. GATED — mutating.

    Second mutating tool of the usd_export family (require_approval=True,
    PP12-109 security gate). Writes AT THE CURRENT FRAME (frame_range=None)
    or ACROSS a [start, end] RANGE (frame_range=[f1, f2]) — NOT a
    full-time-history flatten like houdini_usd_export_layer. Format is
    chosen by *out_path*'s file EXTENSION (.usda -> ascii, .usdc/.usd ->
    crate). Injects NO /World or /root wrapper. Pre-flight validation is
    shown in the gate's preview before approval; a post-write usd_validate
    is embedded in the result under 'validator_post'.

    A SINGLE bridge.execute call — the wrapper performs no result
    interpretation and returns bridge.execute's result VERBATIM, including a
    pending-approval / preview response shape from the 109 gate (that is a
    normal, valid return value, not an error).

    Args:
        ctx: MCP lifespan context — injected by FastMCP; hidden from client schema.
        lop_node: Houdini LOP node path (e.g. "/stage/sphere1") whose
            composed stage the ROP renders. Named `lop_node` (not `node` as
            in houdini_usd_export_layer) because this tool renders a
            specific LOP node's composed stage — the more precise name.
        out_path: Output file path; supports Houdini variable expansions
            (e.g. "$HIP/out.usdc").
        frame_range: None (default) exports the current frame only. A
            [start, end] pair of numbers exports that frame range.
    """
    # Access _get_bridge through the module reference so that
    # `patch("fxhoudinimcp.server._get_bridge", ...)` intercepts it correctly
    # in tests (a local import would cache the original function object).
    bridge = _fxserver._get_bridge(ctx)
    return await bridge.execute(
        "usd_export_rop",
        {
            "lop_node": lop_node,
            "out_path": out_path,
            "frame_range": frame_range,
        },
    )


@mcp.tool(meta={"require_approval": False})
async def houdini_mtlx_inspect(
    ctx: Context,
    mtlx_path_or_doc: str,
) -> dict:
    """Parse a .mtlx via the MaterialX Python API and return a structural summary.

    READ-ONLY, UNGATED (require_approval=False) — mirrors
    houdini_usd_inspect_layer's convention exactly. v1 accepts a FILE PATH
    only (with Houdini $-var expansion) — an inline MaterialX doc-string is
    NOT supported, even though the parameter is named ``mtlx_path_or_doc``
    per spec.md §4.1.

    Returns the MtlxSummary shape on success::

        {
            "ok": True,
            "nodegraphs": [str, ...],
            "surface_nodes": [str, ...],
            "inputs_with_abs_paths": [str, ...],
            "validate": {"ok": bool, "errors": [str, ...]},
        }

    or an FR-2/FR-5 error shape on failure::

        {"ok": False, "error": "<reason>"}

    Args:
        ctx: MCP lifespan context — injected by FastMCP; hidden from client schema.
        mtlx_path_or_doc: A .mtlx file path; supports Houdini variable
            expansions such as ``"$HIP/material.mtlx"``.
    """
    # Access _get_bridge through the module reference so that
    # `patch("fxhoudinimcp.server._get_bridge", ...)` intercepts it correctly
    # in tests (a local import would cache the original function object).
    bridge = _fxserver._get_bridge(ctx)
    return await bridge.execute(
        "mtlx_inspect",
        {"mtlx_path_or_doc": mtlx_path_or_doc},
    )


@mcp.tool(meta={"require_approval": True})
async def houdini_mtlx_edit(
    ctx: Context,
    mtlx_path: str,
    edits: list,
    out_path: str,
) -> dict:
    """Apply node/input value edits to a .mtlx and write via the MaterialX API. GATED.

    Third mutating tool of the usd_export/MaterialX family
    (require_approval=True, PP12-109 security gate). Existing inputs ONLY —
    never creates a new input. Edits are applied via the MaterialX API
    (Input.setValueString) and written with mx.writeToXmlFile — NEVER regex
    on the XML. Pre-flight resolution (existence/ambiguity per edit) is
    shown in the gate's preview before approval; a post-write round-trip
    verify + usd-style validate is embedded in the result.

    A SINGLE bridge.execute call — the wrapper performs no result
    interpretation and returns bridge.execute's result VERBATIM, including a
    pending-approval / preview response shape from the 109 gate (that is a
    normal, valid return value, not an error).

    Args:
        ctx: MCP lifespan context — injected by FastMCP; hidden from client schema.
        mtlx_path: Source .mtlx file path; supports Houdini variable
            expansions (e.g. "$HIP/source.mtlx").
        edits: A non-empty list of {"node": str, "input": str, "value": str}
            dicts. "node" may be unqualified (must resolve unambiguously) or
            qualified as "nodegraph/node". "value" MUST be a string (v1).
        out_path: Output .mtlx file path; supports Houdini variable
            expansions (e.g. "$HIP/edited.mtlx").
    """
    # Access _get_bridge through the module reference so that
    # `patch("fxhoudinimcp.server._get_bridge", ...)` intercepts it correctly
    # in tests (a local import would cache the original function object).
    bridge = _fxserver._get_bridge(ctx)
    return await bridge.execute(
        "mtlx_edit",
        {
            "mtlx_path": mtlx_path,
            "out_path": out_path,
            "edits": edits,
        },
    )
