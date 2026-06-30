"""Handlers: usd_inspect_layer, usd_validate.

usd_inspect_layer — Inspect a USD layer from a LOP node or file path.
usd_validate      — Run USD discipline checks against a layer summary.

All handlers are READ-ONLY, UNGATED (Capability.READONLY) — FR-10.
FR-2: missing/invalid arguments -> {ok: False, error: "..."} (never silent).
FR-5: unexpected exceptions -> {ok: False, error: str(exc)} (never propagate).

PP12-112 / pp12-112b (usd_inspect_layer, usd_validate)
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
from fxhoudinimcp.discipline_checks import run_discipline_checks, format_for_extension  # noqa: E402

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
