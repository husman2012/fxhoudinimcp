"""
export_handlers.py — Houdini-side handlers for the engine-export member (PP12-111 PR-3).

Handlers:
  - probe_versions: reports Houdini build, Labs VAT ROP version, ROP-Alembic/FBX
    availability, and optional skew_table verdict when target_ue is given.
  - validate_budget: DRY-RUN budget check over SOP node geometry. Reads only.
    WRITES NOTHING. wrote_files is always False.

Convention notes (grounded against character_handlers.py + PP12-110 post-mortem):
  - sys.path bootstrap goes 5 dirs above handlers/ to reach the fork root, then /python.
  - Handler functions take NAMED kwargs (dispatcher calls handler(**params), NOT handler(params)).
  - register_handler() calls are at the BOTTOM of the file.
  - bridge.call() does NOT exist on HoudiniBridge. Never use it. Use bridge.execute().
  - VAT version: base alias labs::vertex_animation_textures gives nameComponents()[3]==''.
    Enumerate ALL rop node types whose names start with 'labs::vertex_animation_textures'
    and take the one with max version component to get the real versioned string.
  - rop_alembic / rop_fbx resolve in the SOP category (hou.sopNodeTypeCategory()).
  - primitivecount via geo.intrinsicValue('primitivecount').
  - gc_piece via geo.primIntAttribValues('unreal_gc_piece').
  - frame range via hou.playbar.frameRange() -> [start, end].
"""
from __future__ import annotations

import logging

import hou

from fxhoudinimcp_server.dispatcher import Capability, register_handler

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# sys.path bootstrap — get the fork's python/ package root onto sys.path.
# handlers/ is at: <fork>/houdini/scripts/python/fxhoudinimcp_server/handlers/
# Going 5 dirs up:  handlers/ -> fxhoudinimcp_server/ -> python/ -> scripts/
#                -> houdini/ -> <fork root>
# Then appending /python gives: <fork>/python — where fxhoudinimcp lives.
# (Grounded against character_handlers.py which uses the same 5-level ascent.)
# ---------------------------------------------------------------------------
import os as _os
import sys as _sys

_PY = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..", "python")
)
if _PY not in _sys.path:
    _sys.path.insert(0, _PY)

from fxhoudinimcp import budget_rules, skew_table  # noqa: E402


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_labs_vat_version() -> str | None:
    """Return the Labs VAT ROP version string, or None if not installed.

    The base alias 'labs::vertex_animation_textures' reports nameComponents()[3]==''
    (empty version string). Enumerate ALL node types in the ROP category whose
    names start with 'labs::vertex_animation_textures' and take the maximum
    version component. This is the only way to get the real versioned string.
    """
    try:
        rop_cat = hou.ropNodeTypeCategory()
        vat_types = [
            nt for name, nt in rop_cat.nodeTypes().items()
            if name.startswith("labs::vertex_animation_textures")
        ]
        if not vat_types:
            return None
        # Get version components and find the maximum
        versioned = []
        for nt in vat_types:
            comps = nt.nameComponents()
            # nameComponents() -> (scope, namespace, name, version)
            version = comps[3] if len(comps) > 3 else ""
            if version:  # only include entries with a real version string
                versioned.append(version)
        if not versioned:
            return None
        # Return max version string using numeric key to avoid lexicographic mis-sort
        # ("3.0.10" > "3.0.5" and "10.0" > "3.0" under numeric, not lexicographic, order)
        return max(versioned, key=lambda v: tuple(int(x) for x in v.split(".") if x.isdigit()))
    except Exception as exc:
        _log.warning("Failed to enumerate Labs VAT ROP versions: %s", exc)
        return None


def _check_rop_type(category: hou.NodeTypeCategory, type_name: str) -> bool:
    """Return True if *type_name* resolves in *category*, False otherwise."""
    try:
        nt = category.nodeTypes().get(type_name)
        return nt is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def probe_versions(target_ue=None) -> dict:
    """Report Houdini build, Labs VAT ROP version, ROP availability, and optional skew.

    Args:
        target_ue: Optional UE version string (e.g. "5.4"). When supplied, includes
                   a 'skew' key with verdict and notes from the skew_table.

    Returns:
        dict with:
          houdini       : Houdini build version string (e.g. "21.0.729")
          labs_vat_rop  : Labs VAT version string or None if not installed
          rop_alembic   : bool — True if rop_alembic resolves in the SOP category
          rop_fbx       : bool — True if rop_fbx resolves in the SOP category
          skew          : (only when target_ue is given) dict with 'verdict' and 'notes'
    """
    houdini_version = hou.applicationVersionString()
    labs_vat = _get_labs_vat_version()

    sop_cat = hou.sopNodeTypeCategory()
    rop_alembic = _check_rop_type(sop_cat, "rop_alembic")
    rop_fbx = _check_rop_type(sop_cat, "rop_fbx")

    result: dict = {
        "houdini": houdini_version,
        "labs_vat_rop": labs_vat,
        "rop_alembic": rop_alembic,
        "rop_fbx": rop_fbx,
    }

    if target_ue is not None:
        # Include skew block only when caller requests it
        # When labs_vat is None, use a sentinel string that will produce a 'warn'
        vat_for_skew = labs_vat if labs_vat is not None else "0.0"
        verdict, notes = skew_table.skew_verdict(
            houdini=houdini_version,
            labs_vat=vat_for_skew,
            ue=target_ue,
        )
        result["skew"] = {"verdict": verdict, "notes": notes}

    return result


def validate_budget(node: str, target: str, budget_preset=None) -> dict:
    """DRY-RUN budget check over a SOP node's geometry. Reads only. Writes nothing.

    Args:
        node:          Houdini SOP node path (e.g. "/obj/geo1/box1").
        target:        Export target; one of "vat", "alembic_ue", "fbx",
                       "niagara", "chaos_gc".
        budget_preset: Optional named budget preset string or dict.
                       When None, uses the UE_REALTIME defaults.

    Returns:
        dict with verdict, checks, wrote_files=False (always).
    """
    try:
        sop_node = hou.node(node)
        if sop_node is None:
            return {
                "ok": False,
                "error": f"Node not found: {node!r}",
                "wrote_files": False,
            }

        geo = sop_node.geometry()
        if geo is None:
            return {
                "ok": False,
                "error": f"Node has no geometry: {node!r}",
                "wrote_files": False,
            }

        # ------------------------------------------------------------------
        # Gather geo_stats from read-only introspection.
        # run_budget_checks skips absent keys, so only include what we can
        # reliably introspect pre-bake. texture_res / vat_textures are not
        # introspectable from raw SOP geo — omit them.
        # ------------------------------------------------------------------
        geo_stats: dict = {}

        # Primitive count (triangles / polys)
        try:
            geo_stats["tris"] = geo.intrinsicValue("primitivecount")
        except Exception:
            pass  # absent key is skipped by run_budget_checks

        # Frame range
        try:
            fr = hou.playbar.frameRange()
            geo_stats["frame_range"] = [fr[0], fr[1]]
        except Exception:
            pass

        # Chaos GC — unreal_gc_piece prim attribute for sequential check
        if target == "chaos_gc":
            try:
                piece_attrib = geo.findPrimAttrib("unreal_gc_piece")
                if piece_attrib is not None:
                    geo_stats["gc_pieces"] = list(geo.primIntAttribValues("unreal_gc_piece"))
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Run pure-logic budget checks (no hou calls inside budget_rules)
        # ------------------------------------------------------------------
        report = budget_rules.run_budget_checks(
            geo_stats=geo_stats,
            target=target,
            preset=budget_preset,
        )

        return report.to_dict()

    except Exception as exc:
        _log.warning("validate_budget failed for node %r target %r: %s", node, target, exc)
        return {
            "ok": False,
            "error": str(exc),
            "wrote_files": False,
        }


# ---------------------------------------------------------------------------
# Handler registration — MUST be at the BOTTOM of the file
# (grounded against character_handlers.py registration pattern)
# ---------------------------------------------------------------------------

register_handler("probe_versions", probe_versions, Capability.READONLY)
register_handler("validate_budget", validate_budget, Capability.READONLY)
