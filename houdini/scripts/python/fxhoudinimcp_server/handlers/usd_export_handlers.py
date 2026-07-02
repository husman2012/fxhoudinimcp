"""Handlers: usd_inspect_layer, usd_validate, usd_export_layer, usd_export_rop.

usd_inspect_layer — Inspect a USD layer from a LOP node or file path.
usd_validate      — Run USD discipline checks against a layer summary.
usd_export_layer  — Write a composed USD layer to disk (Sdf.Layer.Export()).
                     GATED (Capability.MUTATING) — the first mutating tool of
                     the USD/MaterialX export family. Registered with a
                     preview_fn (_preview_export_layer) + preview_required=True
                     so the PP12-109 security gate can show a pre-flight
                     preview before approving the write (ADR-0005 pattern).
usd_export_rop    — Drive the /out-context `usd` ROP to write a chosen LOP's
                     composed /stage to disk (current frame or a [start,end]
                     range). GATED (Capability.MUTATING) — the SECOND mutating
                     tool of the family. Registered with a preview_fn
                     (_preview_export_rop) + preview_required=True, mirroring
                     usd_export_layer's registration exactly.

usd_inspect_layer / usd_validate are READ-ONLY, UNGATED (Capability.READONLY)
— FR-10.
FR-2: missing/invalid arguments -> {ok: False, error: "..."} (never silent).
FR-5: unexpected exceptions -> {ok: False, error: str(exc)} (never propagate).

PP12-112 / pp12-112b (usd_inspect_layer, usd_validate)
PP12-112 / pp12-112c (usd_export_layer, _preview_export_layer)
PP12-112 / pp12-112d (usd_export_rop, _preview_export_rop)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# sys.path bootstrap — 5 levels up from this file reaches the fork root;
# +/python adds the FastMCP-side fxhoudinimcp package so the loader import
# below resolves when this module is imported from hython.
#
#  __file__: .../fxhoudinimcp/houdini/scripts/python/fxhoudinimcp_server/handlers/usd_export_handlers.py
#   1 up -> .../handlers/
#   2 up -> .../fxhoudinimcp_server/
#   3 up -> .../python/
#   4 up -> .../scripts/
#   5 up -> .../houdini/
#   6 up -> .../fxhoudinimcp/             (fork root)
#  +/python -> .../fxhoudinimcp/python/
# ---------------------------------------------------------------------------
import logging as _logging
import math as _math
import os as _os
import sys as _sys

_PY = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..", "python")
)
if _PY not in _sys.path:
    _sys.path.insert(0, _PY)

import hou  # noqa: E402  (hython / Houdini-side interpreter only)
from fxhoudinimcp_server.dispatcher import Capability, register_handler  # noqa: E402
from fxhoudinimcp.usd_export_model import LayerSummary  # noqa: E402
from fxhoudinimcp.discipline_checks import (  # noqa: E402
    run_discipline_checks,
    format_for_extension,
    format_from_magic_bytes,
)

# USD modules — required; fail loud if unavailable (M-1: no silent fabrication)
try:
    from pxr import Usd, UsdShade  # noqa: E402
    HAS_PXR = True
except ImportError:
    HAS_PXR = False

_log = _logging.getLogger(__name__)

# Prim names injected by Houdini that must not appear in root_prims (M-4)
_HOUDINI_INTERNAL_PRIMS = frozenset({"HoudiniLayerInfo"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_pxr() -> None:
    """Raise if pxr modules are not available (fail-loud per M-1)."""
    if not HAS_PXR:
        raise hou.OperationFailed(
            "USD (pxr) modules are not available in this Houdini session. "
            "Ensure you are running a Houdini build with USD support."
        )


def _get_stage(target: str):
    """Resolve *target* to a (Usd.Stage, source_kind) pair.

    M-5 EXPLICIT BRANCHES — no silent fallthrough:

    1. If ``hou.node(target)`` resolves to a node:
       a. Node has ``stage()`` -> return (node.stage(), 'node')
       b. Node has NO ``stage()`` -> raise 'not a LOP node'
    2. If ``hou.node(target)`` returns None:
       -> treat target as a file path; open via Usd.Stage.Open(expandString(target))

    Args:
        target: A Houdini scene node path (e.g. ``"/stage/lop1"``) or a USD
                file path (optionally containing Houdini variable expansions
                such as ``"$HIP/out.usda"``).

    Returns:
        ``(stage, 'node')`` when resolved from a LOP node, or
        ``(stage, 'file')`` when opened from disk.

    Raises:
        hou.OperationFailed: if *target* resolves to a non-LOP node, or if
            file-path opening fails (node is None but file cannot be opened).
    """
    _require_pxr()

    node = hou.node(target)
    if node is not None:
        # Branch A: node found — must have stage() to be a LOP node
        if not hasattr(node, "stage"):
            raise hou.OperationFailed(
                f"Target '{target}' resolves to a Houdini node but is not a LOP "
                "node (no stage() method). Provide a LOP network node path or "
                "a USD file path instead."
            )
        stage = node.stage()
        if stage is None:
            raise hou.OperationFailed(
                f"LOP node '{target}' has no USD stage (node may not be cooked)."
            )
        return stage, "node"
    else:
        # Branch B: not a scene node — try file path
        expanded = hou.text.expandString(target)
        stage = Usd.Stage.Open(expanded)
        if stage is None:
            raise hou.OperationFailed(
                f"Target {target!r} is not an existing Houdini node and could "
                f"not be opened as a USD file path ({expanded!r})."
            )
        return stage, "file"


def _root_prims(stage) -> list:
    """Return top-level prim paths from *stage*, excluding Houdini-internal prims.

    M-4: filters out 'HoudiniLayerInfo' and any undefined prims at root level.
    """
    pseudo_root = stage.GetPseudoRoot()
    result = []
    for prim in pseudo_root.GetChildren():
        # Skip Houdini-injected internal prims (M-4)
        if prim.GetName() in _HOUDINI_INTERNAL_PRIMS:
            continue
        # Skip undefined/invalid prims
        if not prim.IsValid():
            continue
        result.append(str(prim.GetPath()))
    return result


def _default_prim(stage) -> "str | None":
    """Return the stage's defaultPrim as a prim path, or None when unset.

    Uses the COMPOSED stage default prim (stage.GetDefaultPrim()), not the
    root layer's authored token (stage.GetRootLayer().defaultPrim).  This
    matters when defaultPrim is authored in a sublayer rather than the root
    layer: the root-layer token reads as '' in that case, causing
    rule_default_prim_set to false-fail even though the composed stage has a
    valid default prim.

    Returns the prim path (e.g. ``"/asset"``), matching the spec §7.3 example
    and the shipped lops_handlers._get_stage_info exemplar which uses
    ``str(GetDefaultPrim().GetPath())``.  Returns ``None`` when the composed
    stage has no valid default prim.
    """
    dp = stage.GetDefaultPrim()
    return str(dp.GetPath()) if dp and dp.IsValid() else None


def _current_format(stage) -> str:
    """Detect the USD format of the stage's root layer.

    Returns:
        'in-memory'  — root layer is anonymous (no backing file).
        A format string from format_for_extension() — for file-backed layers
            whose extension is a known USD format.
        'unknown'    — file-backed layer whose extension is not recognized.
    """
    root_layer = stage.GetRootLayer()
    if root_layer.anonymous:
        return "in-memory"
    # File-backed: derive from extension
    identifier = root_layer.identifier
    try:
        return format_for_extension(identifier)
    except ValueError:
        return "unknown"


def _has_mtlx_material(stage) -> bool:
    """Return True if the stage contains at least one MaterialX material or shader.

    Two heuristics are applied (OR-logic — either is sufficient):

    1. **mtlx render-context surface output**: a ``UsdShade.Material`` prim
       whose ``mtlx`` render-context surface output
       (``GetSurfaceOutput("mtlx")``) has a connected source.  This is the
       standard way Houdini's Material Library LOP wires a MaterialX material
       for the mtlx render context.

    2. **ND_ shader id**: a ``UsdShade.Shader`` prim whose
       ``GetShaderId()`` begins with the ``ND_`` prefix.  MaterialX node
       definitions are universally prefixed ``ND_`` (e.g.
       ``ND_standard_surface_surfaceshader``), so any shader with this prefix
       is a MaterialX node regardless of how its surface output is connected.
       This catches shaders wired to the *default* (non-mtlx-named) surface
       output, which heuristic 1 would miss.

    Known remaining gaps (neither heuristic covers these):
    - Prims inside **unloaded payloads** are not traversed — a MaterialX
      material in an unloaded payload is invisible to this check.
    - **External ``.mtlx`` file references** composed via a ``mtlx``-format
      layer or ``AssetAPI`` are not surfaced as UsdShade prims in the
      traversal; only in-memory or already-composed shaders are detected.
    - A **displacement-only MaterialX network** (no surface shader, no ND_
      displacement shader id) will not be detected.

    M-1: raises hou.OperationFailed if pxr is unavailable — never silently
    returns False (that would fabricate a result the caller cannot distinguish
    from a genuine False).
    """
    _require_pxr()  # M-1: fail loud if pxr missing
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            mat = UsdShade.Material(prim)
            out = mat.GetSurfaceOutput("mtlx")
            if out and out.HasConnectedSource():
                return True
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            shader_id = shader.GetShaderId()  # returns plain str; '' when unset
            if shader_id and str(shader_id).startswith("ND_"):
                return True
    return False


# ---------------------------------------------------------------------------
# usd_inspect_layer
# ---------------------------------------------------------------------------

def usd_inspect_layer(*, node_or_layer: str) -> dict:
    """Inspect a USD layer and return its structural summary.

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
        node_or_layer: Houdini LOP node path (e.g. ``"/stage/lop1"``) or a
            USD file path; supports Houdini variable expansions such as
            ``"$HIP/out.usda"``.
    """
    # FR-2: reject empty target before touching hou.*
    if not node_or_layer or not node_or_layer.strip():
        return {"ok": False, "error": "node_or_layer must be a non-empty node path or file path"}

    try:
        stage, _source_kind = _get_stage(node_or_layer)

        default_prim_val = _default_prim(stage)
        root_prims_val = _root_prims(stage)

        root_layer = stage.GetRootLayer()
        sublayers_val = [str(s) for s in root_layer.subLayerPaths]

        current_format_val = _current_format(stage)
        has_mtlx_val = _has_mtlx_material(stage)

        summary = LayerSummary(
            default_prim=default_prim_val,
            root_prims=root_prims_val,
            sublayers=sublayers_val,
            current_format=current_format_val,
            has_mtlx_material=has_mtlx_val,
        )
        return {"ok": True, **summary.to_dict()}

    except Exception as exc:
        _log.warning("usd_inspect_layer failed for node_or_layer=%r: %s", node_or_layer, exc)
        return {"ok": False, "error": str(exc)}


register_handler("usd_inspect_layer", usd_inspect_layer, Capability.READONLY)


# ---------------------------------------------------------------------------
# usd_validate
# ---------------------------------------------------------------------------

def usd_validate(
    *,
    target: str,
    out_path: "str | None" = None,
    actual_format: "str | None" = None,
    texture_paths: "list | None" = None,
    checks: "list | None" = None,
) -> dict:
    """Run USD discipline checks against a layer summary.

    M-3: if *checks* is not None, immediately return an error — custom check
    selection is not yet implemented.

    Returns a B-1 compliant shape on success::

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
        target:         Houdini LOP node path or USD file path.
        out_path:       Intended output file path (enables preflight mode).
        actual_format:  Format string from magic-byte detection (enables
                        postwrite mode; requires out_path).
        texture_paths:  List of texture paths for abs-path check (postwrite).
        checks:         Reserved — must be None; non-None triggers a fail-loud
                        rejection (M-3).
    """
    # M-3: checks parameter not implemented — reject non-None values explicitly
    if checks is not None:
        return {
            "ok": False,
            "error": (
                "The 'checks' parameter is not yet implemented. "
                "Custom check selection (checks != None) is reserved for a future release. "
                "Pass checks=None to run the default discipline checks."
            ),
        }

    # FR-2: reject empty target
    if not target or not target.strip():
        return {"ok": False, "error": "target must be a non-empty node path or file path"}

    try:
        stage, _source_kind = _get_stage(target)

        default_prim_val = _default_prim(stage)
        root_prims_val = _root_prims(stage)

        summary_dict = {
            "default_prim": default_prim_val,
            "root_prims": root_prims_val,
        }

        report = run_discipline_checks(
            summary_dict,
            out_path=out_path,
            actual_format=actual_format,
            texture_paths=texture_paths,
        )

        # Determine mode and omitted_checks (B-1 completeness signal)
        if out_path is not None and actual_format is not None:
            mode = "postwrite"
            omitted_checks: list = []
        elif out_path is not None:
            mode = "preflight"
            omitted_checks = ["format_matches_ext", "abs_texture_paths"]
        else:
            mode = "minimal"
            omitted_checks = ["format_extension_known", "format_matches_ext", "abs_texture_paths"]

        return {
            "ok": True,
            "mode": mode,
            "omitted_checks": omitted_checks,
            **report.to_dict(),
        }

    except Exception as exc:
        _log.warning("usd_validate failed for target=%r: %s", target, exc)
        return {"ok": False, "error": str(exc)}


register_handler("usd_validate", usd_validate, Capability.READONLY)


# ---------------------------------------------------------------------------
# usd_export_layer (GATED — Capability.MUTATING) + _preview_export_layer
# ---------------------------------------------------------------------------
#
# 109-gate preview function (pp12-112c / ADR-0005 pattern, mirrors
# export_handlers._preview_vat exactly). Runs on the main thread via
# _run_preview / hdefereval (CL-016); hou.* access is safe here. It is pure
# read-only — no scene mutation, no file write — so preview_required=True
# causes DENY on raise/timeout only.


def _preview_export_layer(params: dict) -> dict:
    """Return the 109-gate approval payload for usd_export_layer WITHOUT writing.

    READ-ONLY: performs the pre-flight usd_validate (preflight mode — out_path
    set, actual_format None) and reports the resolved output format, but does
    NOT export/write anything. A raise here causes the gate to DENY the call
    (preview_required=True).

    Called POSITIONALLY by the gate middleware as ``preview_fn(params)`` — a
    single ``params: dict`` argument, NOT ``**params`` (the opposite
    convention from the keyword-only handler; matches _preview_vat's shape).

    Args:
        params: dict with keys node_path, out_path, flatten, default_prim
            (the same params dict the handler will later receive as kwargs).

    Returns:
        {
            "out_path": <hou.text.expandString(out_path)>,
            "resolved_format": "usda" | "usdc" | "usdz",
            "pre_validation": <usd_validate(...) preflight-mode result>,
            "flatten": bool,
            "default_prim": str | None,
            "no_world_wrapper": True,
        }

    Raises:
        ValueError: propagated from format_for_extension() when out_path's
            extension is not a recognized USD format — the gate DENIES the
            call so the operator is never asked to approve an unknown-format
            export (rev3 lockedFieldContract).
        hou.OperationFailed: propagated from usd_validate's underlying
            _get_stage() when node_path cannot be resolved.
    """
    node_path = params["node_path"]
    out_path = params["out_path"]

    expanded = hou.text.expandString(out_path)
    resolved_format = format_for_extension(expanded)  # ValueError -> gate DENY

    pre = usd_validate(target=node_path, out_path=out_path)  # preflight mode

    return {
        "out_path": expanded,
        "resolved_format": resolved_format,
        "pre_validation": pre,
        "flatten": params.get("flatten", False),
        "default_prim": params.get("default_prim"),
        "no_world_wrapper": True,
    }


def usd_export_layer(
    *,
    node_path: str,
    out_path: str,
    flatten: bool = False,
    default_prim: "str | None" = None,
) -> dict:
    """Write a composed USD layer to disk via Sdf.Layer.Export(). GATED — mutating.

    Format is chosen by *out_path*'s file EXTENSION (.usda -> ascii,
    .usdc/.usd -> crate, .usdz -> packaged) — reuses format_for_extension().
    Injects NO /World or /root wrapper: the authored stage's root layer (or
    its Flatten()-ed composition) is exported as-is (usd-publish-discipline.md).

    Runs an INLINE post-write usd_validate (postwrite mode: out_path +
    actual_format both set) after the write and embeds the result under
    'validator_post'.

    Returns:
        On success::

            {
                "ok": True,
                "out_path": <expanded out_path>,
                "format": <format_for_extension(expanded)>,
                "actual_format": <format_from_magic_bytes(header)>,
                "validator_post": <usd_validate(...) postwrite-mode result>,
            }

        On failure (FR-2/FR-5)::

            {"ok": False, "error": "<reason>"}

    Args:
        node_path: Houdini LOP node path (e.g. "/stage/lop1") or a USD file
            path — resolved via _get_stage() (node-or-file source). The
            mutation this handler performs is the WRITE to out_path, not to
            node_path — reading a file-backed node_path is not itself a
            mutation risk.
        out_path: Output file path; supports Houdini variable expansions
            (e.g. "$HIP/out.usdc").
        flatten: When True, export stage.Flatten() (a single flattened
            layer). When False (default), export stage.GetRootLayer() as-is.
        default_prim: When set, resolves to a prim on the stage via
            GetPrimAtPath() and calls stage.SetDefaultPrim(prim) BEFORE
            flatten/export so a subsequent Flatten() captures it (Phase-0
            probe confirmed: SetDefaultPrim before Flatten propagates into
            the flattened layer's defaultPrim). An invalid default_prim path
            returns an FR-2-style error WITHOUT calling SetDefaultPrim.
    """
    try:
        # FR-2: reject empty node_path/out_path before touching hou.*.
        # Moved INSIDE the try (green-1, codex-pair-reviewer Major): the
        # dispatcher calls handler(**params) from a JSON-decoded dict with no
        # runtime type enforcement, so a non-string truthy arg (int/dict/list/
        # bool from a malformed direct-dispatch call) would raise
        # AttributeError on .strip() — that must land in the FR-5 envelope
        # below, not leak as an unhandled exception from a MUTATING handler.
        if not node_path or not node_path.strip():
            return {"ok": False, "error": "node_path must be a non-empty node path or file path"}
        if not out_path or not out_path.strip():
            return {"ok": False, "error": "out_path must be a non-empty file path"}

        stage, _source_kind = _get_stage(node_path)

        if default_prim:
            prim = stage.GetPrimAtPath(default_prim)
            if not prim.IsValid():
                return {
                    "ok": False,
                    "error": f"default_prim {default_prim!r} does not resolve to a valid prim",
                }
            stage.SetDefaultPrim(prim)  # BEFORE flatten so Flatten() captures it

        layer = stage.Flatten() if flatten else stage.GetRootLayer()

        expanded = hou.text.expandString(out_path)
        # No /World or /root wrapper injected — the authored root is preserved
        # as-is (usd-publish-discipline.md).
        ok = layer.Export(expanded)

        with open(expanded, "rb") as fh:
            header = fh.read(16)
        actual_format = format_from_magic_bytes(header)  # ValueError -> FR-5 below
        ext_format = format_for_extension(expanded)

        validator_post = usd_validate(
            target=expanded, out_path=out_path, actual_format=actual_format
        )  # postwrite mode

        return {
            "ok": bool(ok),
            "out_path": expanded,
            "format": ext_format,
            "actual_format": actual_format,
            "validator_post": validator_post,
        }

    except Exception as exc:
        _log.warning(
            "usd_export_layer failed for node_path=%r out_path=%r: %s",
            node_path, out_path, exc,
        )
        return {"ok": False, "error": str(exc)}


register_handler(
    "usd_export_layer",
    usd_export_layer,
    Capability.MUTATING,
    preview_fn=_preview_export_layer,
    preview_required=True,
)


# ---------------------------------------------------------------------------
# usd_export_rop (GATED — Capability.MUTATING) + _preview_export_rop
# ---------------------------------------------------------------------------
#
# Drives the /out-context `usd` ROP to write a chosen LOP's composed /stage to
# disk — AT THE CURRENT FRAME (frame_range=None -> trange=0) or ACROSS A
# GIVEN [start, end] RANGE (frame_range=[f1, f2] -> trange=1). NOT a full-
# time-history flatten like usd_export_layer's Sdf.Layer.Export() — this is
# the second, ROP-driven gated write of the family (pp12-112d, plan rev3).
#
# ROP-driving idiom grounded on the shipped export_handlers.rop_alembic
# pattern (createNode -> set trange/f -> parm('execute').pressButton() with
# render() fallback -> node.errors() -> destroy()), substituting the
# LOP-targeting `loppath` path-string parm for soppath/setInput wiring.
#
# Phase-0 hython probe (2026-07-01, live Houdini 21.0.729) confirmed:
#   - the /out `usd` node type exists with parms loppath / lopoutput /
#     trange / f1,f2,f3 (accessed via parmTuple('f'), NOT parm('f') — a
#     ParmTuple, matching rop_alembic's f-tuple pattern) / execute.
#   - NO source-mode/enable toggle gates loppath (no use_sop_path analog) —
#     setting loppath alone is sufficient to drive the ROP off that LOP's
#     composed stage; the closest same-named token, 'flattensoplayers', is
#     an unrelated SOP-import-layer option, not a source-mode gate.
#   - loppath resolves an ABSOLUTE LOP node path from /out and the composed
#     stage's real content (e.g. a cooked sphere LOP's '/sphere1' prim) lands
#     in the written file — no silent no-op.
#   - lopoutput format is resolved by file EXTENSION exactly like
#     format_for_extension(): a .usdc lopoutput writes PXR-USDC crate magic
#     bytes; a .usda lopoutput writes a '#usda 1.0' ASCII header.
#   - node.errors() returns a TUPLE (empty tuple on success), not a list —
#     truthiness-checked below, matching the shipped rop_alembic pattern.
#   - trange=1 + parmTuple('f').set((f1, f2, finc)) on a time-varying LOP
#     genuinely cooks + writes multiple USD time samples (confirmed via an
#     xform LOP driven by the $F Hscript expression) — range mode is real,
#     not a silent current-frame collapse.


def _preview_export_rop(params: dict) -> dict:
    """Return the 109-gate approval payload for usd_export_rop WITHOUT writing.

    READ-ONLY: performs the pre-flight usd_validate (preflight mode — out_path
    set, actual_format None) and reports the resolved output format, but does
    NOT create a ROP or write anything. A raise here causes the gate to DENY
    the call (preview_required=True).

    Called POSITIONALLY by the gate middleware as ``preview_fn(params)`` — a
    single ``params: dict`` argument, NOT ``**params`` (the same convention
    as _preview_export_layer).

    Unlike _preview_export_layer (which reuses _get_stage's node-or-file
    fallback), this preview validates the SAME domain the handler will drive
    a ROP against: lop_node MUST resolve to a real Houdini node with a
    stage() method whose composed stage is non-None (a cooked LOP node). A
    file path or a missing/uncooked node is rejected HERE (raise -> DENY) so
    the operator is never asked to approve an export the handler would then
    reject (plan-6/plan-9 fold — preview and handler agree on domain).

    Args:
        params: dict with keys lop_node, out_path, frame_range (the same
            params dict the handler will later receive as kwargs).

    Returns:
        {
            "out_path": <hou.text.expandString(out_path).replace('\\\\', '/')>,
            "resolved_format": "usda" | "usdc" | "usdz",
            "pre_validation": <usd_validate(...) preflight-mode result>,
            "frame_range": params.get("frame_range"),
            "driven_via": "usd ROP (/out-context)",
            "no_world_wrapper": True,
        }

    Raises:
        hou.OperationFailed: when lop_node does not resolve to a node, is
            not a LOP node (no stage() method), or has no composed stage
            (not cooked).
        ValueError: propagated from format_for_extension() when out_path's
            extension is not a recognized USD format — the gate DENIES the
            call so the operator is never asked to approve an
            unknown-format export.
    """
    lop_node = params["lop_node"]
    out_path = params["out_path"]

    node = hou.node(lop_node)
    if node is None:
        raise hou.OperationFailed(f"LOP node not found: {lop_node}")
    if not hasattr(node, "stage"):
        raise hou.OperationFailed(f"{lop_node} is not a LOP node")
    stage = node.stage()
    if stage is None:
        raise hou.OperationFailed(f"{lop_node} has no composed stage (not cooked)")

    expanded = hou.text.expandString(out_path).replace("\\", "/")
    resolved_format = format_for_extension(expanded)  # ValueError -> gate DENY

    pre = usd_validate(target=lop_node, out_path=out_path)  # preflight mode

    return {
        "out_path": expanded,
        "resolved_format": resolved_format,
        "pre_validation": pre,
        "frame_range": params.get("frame_range"),
        "driven_via": "usd ROP (/out-context)",
        "no_world_wrapper": True,
    }


def usd_export_rop(
    *,
    lop_node: str,
    out_path: str,
    frame_range: "list | None" = None,
) -> dict:
    """Drive the /out `usd` ROP to write lop_node's composed stage to disk. GATED.

    Writes AT THE CURRENT FRAME (frame_range=None -> trange=0) or ACROSS a
    given [start, end] RANGE (frame_range=[f1, f2] -> trange=1) — NOT a
    full-time-history flatten (that is usd_export_layer's job). Format is
    chosen by *out_path*'s file EXTENSION (.usda -> ascii, .usdc/.usd ->
    crate) — reuses format_for_extension(). Injects NO /World or /root
    wrapper: the ROP renders lop_node's composed stage as-is.

    Runs an INLINE post-write usd_validate (postwrite mode) after the write
    and embeds the result under 'validator_post'. A format mismatch between
    the requested (extension-derived) format and the actual (magic-bytes-
    derived) format is a HARD failure (ok=False), not a silently-nested
    validator_post discrepancy (plan-7-cap).

    Returns:
        On success::

            {
                "ok": True,
                "out_path": <expanded out_path>,
                "format": <format_for_extension(expanded)>,
                "actual_format": <format_from_magic_bytes(header)>,
                "validator_post": <usd_validate(...) postwrite-mode result>,
            }

        On failure (FR-2/FR-5)::

            {"ok": False, "error": "<reason>", "out_path": <written_path or None>}

    Args:
        lop_node: Houdini LOP node path (e.g. "/stage/sphere1") whose
            composed stage the ROP will render. Unlike usd_export_layer's
            node_path, this does NOT fall back to opening a file path — the
            ROP renders a specific NODE (plan-6/plan-9: the no-file-fallback
            divergence from usd_export_layer).
        out_path: Output file path; supports Houdini variable expansions
            (e.g. "$HIP/out.usdc").
        frame_range: None for the current-frame-only export (trange=0), or
            a [start, end] pair of finite numbers (start <= end) for a range
            export (trange=1). Any other shape is rejected before any ROP
            is created.
    """
    rop = None
    written_path = None
    try:
        # FR-2: reject empty lop_node/out_path INSIDE the try (mirrors the
        # usd_export_layer green-1 fix — a non-string truthy arg's .strip()
        # AttributeError must land in the FR-5 envelope below, not leak as
        # an unhandled exception from a MUTATING handler). This guard fires
        # BEFORE any hou.node()/ROP lookup, so no ROP can be leaked on this
        # path.
        if not lop_node or not lop_node.strip():
            return {"ok": False, "error": "lop_node must be a non-empty node path"}
        if not out_path or not out_path.strip():
            return {"ok": False, "error": "out_path must be a non-empty file path"}

        # frame_range shape guard — BEFORE any hou.node()/ROP creation.
        if frame_range is not None:
            if (
                not isinstance(frame_range, (list, tuple))
                or len(frame_range) != 2
                or any(
                    isinstance(x, bool) or not isinstance(x, (int, float))
                    for x in frame_range
                )
            ):
                return {
                    "ok": False,
                    "error": "frame_range must be null or a [start, end] pair of numbers",
                }
            # FOLD new-5-cap: also reject non-finite or reversed ranges.
            if any(not _math.isfinite(x) for x in frame_range):
                return {
                    "ok": False,
                    "error": (
                        "frame_range must be a finite [start, end] pair with "
                        "start <= end"
                    ),
                }
            if frame_range[0] > frame_range[1]:
                return {
                    "ok": False,
                    "error": (
                        "frame_range must be a finite [start, end] pair with "
                        "start <= end"
                    ),
                }

        # Node validity — the ROP renders a NODE, not a file (plan-6/plan-9
        # divergence from _get_stage's file-open fallback branch).
        node = hou.node(lop_node)
        if node is None:
            return {"ok": False, "error": f"LOP node not found: {lop_node}"}
        if not hasattr(node, "stage"):
            return {"ok": False, "error": f"{lop_node} is not a LOP node (no stage())"}
        if node.stage() is None:
            return {
                "ok": False,
                "error": f"{lop_node} has no composed stage (not cooked)",
            }

        expanded = hou.text.expandString(out_path).replace("\\", "/")
        ext_format = format_for_extension(expanded)  # ValueError -> FR-5 below

        out_dir = _os.path.dirname(expanded)
        if out_dir:
            _os.makedirs(out_dir, exist_ok=True)

        out_net = hou.node("/out")
        if out_net is None:
            return {"ok": False, "error": "/out context not found"}
        rop = out_net.createNode("usd", "mcp_usd_rop_export")

        # No source-mode/enable toggle gates loppath on the shipped `usd`
        # ROP (Phase-0 probe confirmed) — set loppath + lopoutput directly.
        loppath_parm = rop.parm("loppath")
        if loppath_parm is None:
            raise RuntimeError("parm 'loppath' not found on usd ROP")
        loppath_parm.set(lop_node)

        lopoutput_parm = rop.parm("lopoutput")
        if lopoutput_parm is None:
            raise RuntimeError("parm 'lopoutput' not found on usd ROP")
        lopoutput_parm.set(expanded)

        trange_parm = rop.parm("trange")
        if frame_range:
            if trange_parm is not None:
                trange_parm.set(1)
            f_tuple = rop.parmTuple("f")
            if f_tuple is not None:
                f_tuple.set((frame_range[0], frame_range[1], 1))
        else:
            if trange_parm is not None:
                trange_parm.set(0)

        # No /World or /root wrapper injected — the ROP renders the
        # composed stage of lop_node as-is (usd-publish-discipline.md).

        exec_parm = rop.parm("execute")
        if exec_parm is not None:
            exec_parm.pressButton()
        else:
            rop.render()

        errs = rop.errors()
        if errs:
            raise RuntimeError("\n".join(errs))

        # The write happened at execute (above) — confirm a file actually
        # landed BEFORE destroying the ROP or claiming success (plan-5-cap:
        # a clean cook with errors()==[] but no file is a silent no-op, NOT
        # a success; written_path is set ONLY once the file is confirmed).
        file_exists = _os.path.exists(expanded)
        if file_exists:
            written_path = expanded

        try:
            rop.destroy()
        except Exception:
            pass
        rop = None

        if not file_exists:
            return {
                "ok": False,
                "out_path": None,
                "error": f"ROP cooked without errors but produced no file at {expanded}",
            }

        # POST-WRITE — a failure here means the file WAS written.
        with open(expanded, "rb") as fh:
            header = fh.read(16)
        actual_format = format_from_magic_bytes(header)  # ValueError -> except below

        validator_post = usd_validate(
            target=expanded, out_path=out_path, actual_format=actual_format
        )  # postwrite mode

        if actual_format != ext_format:
            # plan-7-cap: a format-by-extension mismatch is a HARD failure —
            # NOT ok=True with the mismatch merely nested in validator_post.
            return {
                "ok": False,
                "out_path": expanded,
                "error": (
                    f"format mismatch: requested {ext_format} (by extension) "
                    f"but ROP wrote {actual_format}"
                ),
                "format": ext_format,
                "actual_format": actual_format,
                "validator_post": validator_post,
            }

        return {
            "ok": True,
            "out_path": expanded,
            "format": ext_format,
            "actual_format": actual_format,
            "validator_post": validator_post,
        }

    except Exception as exc:
        if rop is not None:
            try:
                rop.destroy()
            except Exception:
                pass
        if written_path:
            msg = f"file written to {written_path} but post-write validation failed: {exc}"
        else:
            msg = str(exc)
        _log.warning(
            "usd_export_rop failed for lop_node=%r out_path=%r: %s",
            lop_node, out_path, exc,
        )
        return {"ok": False, "error": msg, "out_path": written_path}


register_handler(
    "usd_export_rop",
    usd_export_rop,
    Capability.MUTATING,
    preview_fn=_preview_export_rop,
    preview_required=True,
)


# ---------------------------------------------------------------------------
# MaterialX guard (mirrors HAS_PXR/_require_pxr) — PP12-112 / pp12-112e
# ---------------------------------------------------------------------------

try:
    import MaterialX as mx  # noqa: E402
    HAS_MTLX = True
except ImportError:
    HAS_MTLX = False

from fxhoudinimcp.usd_export_model import MtlxSummary  # noqa: E402


def _require_mtlx() -> None:
    """Raise if the MaterialX module is not available (fail-loud per M-1)."""
    if not HAS_MTLX:
        raise hou.OperationFailed(
            "MaterialX module not available in this Houdini session. "
            "Ensure you are running a Houdini build with MaterialX support."
        )


def _mtlx_validate(doc) -> "tuple[bool, str]":
    """Defensively normalize doc.validate() to a (bool, str) pair.

    Houdini's bundled MaterialX Python binds validate() as a (bool, str)
    tuple (Phase-0 probe, 2026-07-01, MaterialX 1.39.3, confirmed
    (True, '') on a valid document) -- but this helper also tolerates a
    bare bool return, in case a different bundled version binds it that
    way (plan-4 defensive fold).
    """
    r = doc.validate()
    return r if isinstance(r, tuple) else (bool(r), "")


def _resolve_node(doc, name: str):
    """Resolve *name* to a (node_or_None, matches_count, resolved_path_or_None).

    Two forms (FOLD new-1 -- strict, no doc.getDescendant, no ambiguity-by-
    heuristic):

    (a) QUALIFIED -- "nodegraph/node" (exactly ONE '/'): looks up the
        nodegraph by name, then the node within it. An invalid qualified
        form (more than one '/') returns (None, 0, None) -- the caller
        treats matches_count==0 as "not found".
    (b) UNQUALIFIED -- collects every node named *name* from doc.getNodes()
        (top-level) PLUS every nodegraph's getNodes() (in-graph). Returns
        (None, 0, None) if no match, (None, N, None) if N>1 matches
        (ambiguous -- the caller fails loud), or (node, 1, <canonical path>)
        if exactly one match.

    GREEN-FIX (green-1): resolved_path is ALWAYS a path CANONICALIZED by
    this function itself -- "<nodegraph.getName()>/<node.getName()>" for an
    in-graph node, or bare "<node.getName()>" for a genuinely top-level
    node -- never node.getNamePath() (whose qualification behavior is
    binding-dependent and was observed to sometimes omit the nodegraph
    prefix even for an in-graph node). This makes resolved_path an
    unambiguous, self-consistent key we fully control, so the post-write
    round-trip re-resolution (new-2, see mtlx_edit) can do a PURE DIRECT
    LOOKUP by this path against the freshly re-parsed document with NO
    search/fallback branch -- PASS 1's uniqueness proof (against `doc`)
    stays valid against `doc2` (the re-parsed doc) because both sides
    agree on the same canonical addressing scheme, not on a value
    (getNamePath()) that PASS 1 never controlled.

    The caller uses matches_count for the 0/1/>1 fail-loud decision
    (plan-2) and resolved_path for the post-write round-trip re-resolution
    (new-2 -- re-resolve by the RECORDED path, never by re-running a
    possibly-ambiguous name search).
    """
    if "/" in name:
        parts = name.split("/")
        if len(parts) != 2:
            return None, 0, None
        graph_name, node_name = parts
        ng = doc.getNodeGraph(graph_name)
        node = ng.getNode(node_name) if ng else None
        if node is None:
            return None, 0, None
        return node, 1, f"{graph_name}/{node.getName()}"

    matches = []  # list of (node, canonical_rpath)
    for n in doc.getNodes():
        if n.getName() == name:
            matches.append((n, n.getName()))
    for ng in doc.getNodeGraphs():
        for n in ng.getNodes():
            if n.getName() == name:
                matches.append((n, f"{ng.getName()}/{n.getName()}"))

    if len(matches) == 0:
        return None, 0, None
    if len(matches) > 1:
        return None, len(matches), None
    node, canonical_rpath = matches[0]
    return node, 1, canonical_rpath


def _validate_edits_shape(edits) -> "tuple[bool, str]":
    """Shared shape guard run by BOTH the preview and the handler (plan-7).

    v1 accepts STRING-valued edits ONLY (plan-6/new-3 -- avoids a numeric
    round-trip false-fail when MaterialX's own serialization of a number
    differs from Python's str()). Returns (True, "") on a well-formed
    edits list, else (False, "<reason>").
    """
    if not isinstance(edits, (list, tuple)) or len(edits) == 0:
        return False, "edits must be a non-empty list of {node,input,value} objects"

    for e in edits:
        if not isinstance(e, dict) or not all(k in e for k in ("node", "input", "value")):
            return False, "each edit needs node/input/value keys"
        if not isinstance(e["value"], str):
            return False, (
                f"edit value for {e.get('node')!r}.{e.get('input')!r} must be a "
                f"string (v1 supports string-valued edits only, per spec.md "
                f"section 4.2 texture-path examples); got "
                f"{type(e['value']).__name__}"
            )

    return True, ""


# ---------------------------------------------------------------------------
# mtlx_inspect (READ-ONLY, UNGATED) — PP12-112 / pp12-112e
# ---------------------------------------------------------------------------

def mtlx_inspect(*, mtlx_path_or_doc: str) -> dict:
    """Parse a .mtlx via the MaterialX Python API and return a structural summary.

    v1 accepts a FILE PATH only (with Houdini $-var expansion) — an inline
    MaterialX doc-string is NOT supported (plan-8; the docstring says so
    explicitly so the parameter name is not misleading, even though it is
    named mtlx_path_or_doc per spec.md section 4.1).

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
        mtlx_path_or_doc: A .mtlx file path (v1: path only); supports
            Houdini variable expansions such as ``"$HIP/material.mtlx"``.
    """
    if not mtlx_path_or_doc or not mtlx_path_or_doc.strip():
        return {"ok": False, "error": "mtlx_path_or_doc must be a non-empty file path"}

    try:
        _require_mtlx()

        expanded = hou.text.expandString(mtlx_path_or_doc)
        doc = mx.createDocument()
        mx.readFromXmlFile(doc, expanded)  # raises on parse/missing-file -> FR-5

        nodegraphs = [ng.getName() for ng in doc.getNodeGraphs()]

        nodes = list(doc.getNodes())
        for ng in doc.getNodeGraphs():
            nodes.extend(ng.getNodes())

        surface_nodes = [n.getName() for n in nodes if n.getType() == "surfaceshader"]

        inputs_with_abs_paths = []
        for n in nodes:
            for inp in n.getInputs():
                if inp.getType() == "filename":
                    val = inp.getValueString()
                    if val and _os.path.isabs(val):
                        inputs_with_abs_paths.append(inp.getName())

        valid, message = _mtlx_validate(doc)

        summary = MtlxSummary(
            nodegraphs=nodegraphs,
            surface_nodes=surface_nodes,
            inputs_with_abs_paths=inputs_with_abs_paths,
            validate_ok=bool(valid),
            validate_errors=([message] if (not valid and message) else []),
        )
        return {"ok": True, **summary.to_dict()}

    except Exception as exc:
        _log.warning("mtlx_inspect failed for mtlx_path_or_doc=%r: %s", mtlx_path_or_doc, exc)
        return {"ok": False, "error": str(exc)}


register_handler("mtlx_inspect", mtlx_inspect, Capability.READONLY)


# ---------------------------------------------------------------------------
# mtlx_edit (GATED — Capability.MUTATING) + _preview_mtlx_edit
# ---------------------------------------------------------------------------
#
# THIRD mutating tool of the usd_export/MaterialX family (pp12-112e, plan
# rev3). Applies node/input value edits via the MaterialX API
# (Input.setValueString) and writes with mx.writeToXmlFile — NEVER regex on
# the XML (usd-publish-discipline.md). Two-pass resolve-then-write: PASS 1
# resolves EVERY edit (no mutation) and rejects the WHOLE call if any edit
# fails to resolve (node not found / ambiguous / input not found / duplicate
# target) BEFORE any write; PASS 2 applies + writes only once every edit is
# known-good. A post-write round-trip re-read confirms each edit's value
# actually landed, re-resolving by the RECORDED node path captured during
# PASS 1 (new-2 — never by re-running a possibly-ambiguous name search
# against the re-read document).


def _preview_mtlx_edit(params: dict) -> dict:
    """Return the 109-gate approval payload for mtlx_edit WITHOUT writing.

    READ-ONLY: resolves each edit's target (existence + ambiguity) against
    the parsed source document and reports a pre-validation, but does NOT
    apply or write anything. Called POSITIONALLY by the gate middleware as
    ``preview_fn(params)`` — a single ``params: dict`` argument (mirrors
    _preview_export_rop's convention).

    FOLD plan-7 (preview/handler domain parity): runs the SAME
    _validate_edits_shape guard the handler runs; a shape failure returns an
    operator-visible rejection payload rather than proceeding as if the
    edits were well-formed.

    FOLD plan-1 (do NOT rely on an unverified raise->DENY for a MaterialX
    parse failure): CATCHES the readFromXmlFile failure itself and returns
    an operator-visible ``source_parseable: False`` payload, rather than
    betting on the gate's raise->DENY mechanism for an arbitrary pybind
    exception.

    _require_mtlx() still raises (propagates) when MaterialX is genuinely
    unavailable -- this reuses the SAME raise->DENY path _preview_export_rop
    already relies on for its own ValueError/hou.OperationFailed cases, so
    it is not a new, unverified claim.

    Args:
        params: dict with keys mtlx_path, out_path, edits (the same params
            dict the handler will later receive as kwargs).

    Returns:
        A shape-rejection payload (edits_shape_ok=False), a parse-failure
        payload (source_parseable=False), or the full approval payload
        (source_parseable=True, edits_shape_ok=True, edits_preview=[...],
        pre_validation={...}).

    Raises:
        hou.OperationFailed: propagated from _require_mtlx() when
            MaterialX is unavailable -- the gate DENIES the call.
    """
    _require_mtlx()

    mtlx_path = params["mtlx_path"]
    out_path = params["out_path"]
    edits = params.get("edits")

    shape_ok, shape_error = _validate_edits_shape(edits)
    if not shape_ok:
        return {
            "edits_shape_ok": False,
            "shape_error": shape_error,
            "mtlx_path": mtlx_path,
            "out_path": out_path,
            "edits": edits,
        }

    expanded_src = hou.text.expandString(mtlx_path)
    expanded_out = hou.text.expandString(out_path)

    doc = mx.createDocument()
    try:
        mx.readFromXmlFile(doc, expanded_src)
    except Exception as exc:
        return {
            "source": expanded_src,
            "out_path": expanded_out,
            "source_parseable": False,
            "parse_error": str(exc),
            "edits": edits,
            "no_regex": True,
        }

    edits_preview = []
    for e in edits:
        node, count, _rpath = _resolve_node(doc, e["node"])
        resolved = count == 1
        ambiguous = count > 1
        inp = node.getInput(e["input"]) if (resolved and node is not None) else None
        edits_preview.append({
            "node": e["node"],
            "input": e["input"],
            "value": e["value"],
            "exists": bool(resolved and inp is not None),
            "ambiguous": ambiguous,
            "current_value": (inp.getValueString() if inp is not None else None),
        })

    valid, message = _mtlx_validate(doc)

    return {
        "out_path": expanded_out,
        "source": expanded_src,
        "source_parseable": True,
        "edits": edits,
        "edits_shape_ok": True,
        "edits_preview": edits_preview,
        "pre_validation": {"ok": bool(valid), "errors": ([message] if (not valid and message) else [])},
        "no_regex": True,
    }


def mtlx_edit(*, mtlx_path: str, out_path: str, edits: "list | None" = None) -> dict:
    """Apply node/input value edits to a .mtlx and write via the MaterialX API. GATED.

    Existing inputs ONLY -- never calls addInput to create a new one; an
    edit targeting a non-existent input is rejected. Two-pass
    resolve-then-write (PASS 1 resolves every edit with NO mutation; PASS 2
    applies+writes only if every edit resolved cleanly) so an invalid edit
    anywhere in the list blocks the write of every edit in the call,
    including otherwise-valid ones. A post-write round-trip re-read
    (re-resolved by the RECORDED node path, not a name search) confirms
    each edit's value actually landed before reporting success.

    Returns:
        On success::

            {
                "ok": True,
                "out_path": <expanded out_path>,
                "edits_applied": <int>,
                "validate": {"ok": bool, "errors": [str, ...]},
            }

        On failure (FR-2/FR-5)::

            {"ok": False, "error": "<reason>"}

    Args:
        mtlx_path: Source .mtlx file path; supports Houdini variable
            expansions (e.g. "$HIP/source.mtlx").
        out_path: Output .mtlx file path; supports Houdini variable
            expansions (e.g. "$HIP/edited.mtlx").
        edits: A non-empty list of {"node": str, "input": str, "value": str}
            dicts. "node" may be unqualified (must resolve unambiguously) or
            qualified as "nodegraph/node". "value" MUST be a string (v1).
    """
    try:
        if not mtlx_path or not mtlx_path.strip():
            return {"ok": False, "error": "mtlx_path must be a non-empty file path"}
        if not out_path or not out_path.strip():
            return {"ok": False, "error": "out_path must be a non-empty file path"}

        shape_ok, shape_error = _validate_edits_shape(edits)
        if not shape_ok:
            return {"ok": False, "error": shape_error}

        _require_mtlx()

        expanded_src = hou.text.expandString(mtlx_path)
        expanded_out = hou.text.expandString(out_path)

        doc = mx.createDocument()
        mx.readFromXmlFile(doc, expanded_src)  # raises on parse/missing-file -> FR-5

        # PASS 1: resolve ALL edits (no mutation).
        resolved_targets = []  # list of (rpath, input_name, value, inp)
        # GREEN-FIX (green-2): dedup key is the CANONICAL (rpath, input_name)
        # pair, not id(inp). Now that _resolve_node always returns a
        # canonicalized rpath ("<nodegraph>/<node>" or bare "<node>") that
        # this function fully controls, two edits resolving to the same
        # physical target always produce the identical (rpath, input_name)
        # pair -- a string-keyed dedup is a more defensible, deterministic
        # contract than binding-object identity (id(inp) depends on
        # whether the MaterialX Python binding returns a fresh wrapper
        # object per getInput() call, which is an implementation detail we
        # do not want to depend on).
        seen_targets = set()  # {(rpath, input_name)}
        for e in edits:
            node, count, rpath = _resolve_node(doc, e["node"])
            if count == 0:
                return {
                    "ok": False,
                    "error": f"node {e['node']!r} not found in {mtlx_path!r}",
                }
            if count > 1:
                return {
                    "ok": False,
                    "error": (
                        f"node {e['node']!r} is ambiguous (found in {count} "
                        f'nodegraphs); qualify it as "nodegraph/node"'
                    ),
                }
            inp = node.getInput(e["input"])
            if inp is None:
                return {
                    "ok": False,
                    "error": f"input {e['input']!r} not found on node {e['node']!r}",
                }

            target_key = (rpath, e["input"])
            if target_key in seen_targets:
                return {
                    "ok": False,
                    "error": (
                        f"duplicate edit target {rpath}.{e['input']} (two edits "
                        f"resolve to the same input)"
                    ),
                }
            seen_targets.add(target_key)
            resolved_targets.append((rpath, e["input"], e["value"], inp))

        # PASS 2: apply + write (only reached if every edit resolved cleanly).
        for _rpath, _input_name, value, inp in resolved_targets:
            inp.setValueString(value)  # NO regex; value is a str, no coercion

        out_dir = _os.path.dirname(expanded_out)
        if out_dir:
            _os.makedirs(out_dir, exist_ok=True)

        mx.writeToXmlFile(doc, expanded_out)

        # Post-write round-trip verify -- re-resolve by the RECORDED rpath
        # (new-2), never by re-running the (possibly ambiguous) name search.
        #
        # GREEN-FIX (green-1): rpath is now a path CANONICALIZED by
        # _resolve_node itself (never node.getNamePath(), whose
        # qualification behavior is binding-dependent) -- so this is a
        # PURE DIRECT LOOKUP with NO search/fallback branch. PASS 1's
        # uniqueness proof (against `doc`, pre-write) stays valid against
        # `doc2` (the freshly re-parsed post-write document) because both
        # sides address nodes via the SAME canonical scheme this function
        # controls end to end: "<nodegraph>/<node>" for an in-graph node,
        # bare "<node>" for a top-level node. A nodegraph-wide scan here
        # would be unsafe -- it could match an unrelated same-named node in
        # a DIFFERENT nodegraph of doc2 (the exact red-3 decoy scenario),
        # silently verifying against the wrong node.
        doc2 = mx.createDocument()
        mx.readFromXmlFile(doc2, expanded_out)
        for rpath, input_name, value, _inp in resolved_targets:
            node2 = None
            if "/" in rpath:
                graph_name, node_name = rpath.split("/")
                ng2 = doc2.getNodeGraph(graph_name)
                node2 = ng2.getNode(node_name) if ng2 else None
            else:
                node2 = doc2.getNode(rpath)
            inp2 = node2.getInput(input_name) if node2 is not None else None
            if inp2 is None or inp2.getValueString() != value:
                got = inp2.getValueString() if inp2 is not None else None
                return {
                    "ok": False,
                    "error": (
                        f"edit to {rpath}.{input_name} did not round-trip "
                        f"(expected {value!r}, got {got!r})"
                    ),
                }

        valid2, message2 = _mtlx_validate(doc2)

        return {
            "ok": True,
            "out_path": expanded_out,
            "edits_applied": len(edits),
            "validate": {"ok": bool(valid2), "errors": ([message2] if (not valid2 and message2) else [])},
        }

    except Exception as exc:
        _log.warning("mtlx_edit failed for mtlx_path=%r out_path=%r: %s", mtlx_path, out_path, exc)
        return {"ok": False, "error": str(exc)}


register_handler(
    "mtlx_edit",
    mtlx_edit,
    Capability.MUTATING,
    preview_fn=_preview_mtlx_edit,
    preview_required=True,
)
