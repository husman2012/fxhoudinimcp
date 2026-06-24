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
from fxhoudinimcp.export_model import (  # noqa: E402
    ExportManifest,
    VersionTriple,
    vat_mode_from_export_type,
)


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


def export_vat(
    node: str,
    out_dir: str,
    export_type: str = "soft",
    asset_name: str | None = None,
    frame_range: list | None = None,
    target_ue: str | None = None,
) -> dict:
    """Bake a labs::vertex_animation_textures ROP: mesh + textures + manifest sidecar.

    Creates a fresh labs::vertex_animation_textures ROP under /out, wires the
    supplied SOP node via the 'soppath' string parm, sets mode/frame-range,
    triggers the bake, collects output paths, and writes an ExportManifest
    sidecar (.export.json).

    FR-2 (fail-loud): param validation BEFORE the outer try; the entire
    mutating body is wrapped in a single try/except Exception so no code path
    raises past the handler boundary.

    Args:
        node:        Houdini SOP node path (e.g. "/obj/geo1/box1").
        out_dir:     Output directory for textures, mesh, and sidecar.
        export_type: VAT mode — "soft" (default), "rigid", "fluid", or "sprite".
        asset_name:  Asset base name for output files.  Derived from 'node' leaf
                     when None.
        frame_range: [start, end] frame list.  Uses scene playbar range when None.
        target_ue:   Optional UE version string (e.g. "5.4") — included in the
                     version_triple for skew-table annotation.

    Returns::

        {
            "ok": True,
            "node": "<rop node path>",
            "mesh": "<mesh file path or ''>"
            "textures": ["<pos path>", "<rot path>", ...],
            "sidecar": "<path to .export.json>",
            "vat_version": "<labs_vat version or None>",
            "version_triple": { "houdini": ..., "labs_vat": ..., ... }
        }

    or on error::

        {"ok": False, "error": "<message>", "wrote_files": False}
    """
    import json as _json
    import os as _os

    # ── param validation (FR-2 early-return — before outer try) ──────────────
    if not node:
        return {"ok": False, "error": "export_vat requires 'node'", "wrote_files": False}
    if not out_dir:
        return {"ok": False, "error": "export_vat requires 'out_dir'", "wrote_files": False}

    # Validate export_type early — vat_mode_from_export_type raises ValueError on bad input.
    try:
        vat_mode = vat_mode_from_export_type(export_type)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "wrote_files": False}

    # ── FR-2 outer envelope — wraps ALL Houdini mutation + file I/O ──────────
    try:
        # rop is bound FIRST so the outer except's `if rop is not None` is always
        # safe even if a statement before createNode raises (FR-2: no UnboundLocalError).
        rop = None
        # Verify the source SOP node exists in the current scene.
        sop_node = hou.node(node)
        if sop_node is None:
            return {"ok": False, "error": f"Node not found: {node!r}", "wrote_files": False}

        # Derive asset_name from the node leaf name when not supplied.
        _asset_name = asset_name if asset_name else sop_node.name()

        # Resolve output directory (expand Houdini variables).
        _out_dir = hou.text.expandString(out_dir)
        _os.makedirs(_out_dir, exist_ok=True)

        # Resolve frame range.
        if frame_range and len(frame_range) >= 2:
            f1, f2 = int(frame_range[0]), int(frame_range[1])
        else:
            fr = hou.playbar.frameRange()
            f1, f2 = int(fr[0]), int(fr[1])

        # Get Labs VAT version for the manifest.
        labs_vat = _get_labs_vat_version()

        # Create the VAT ROP under /out (Driver context — grounded from plan riskNotes).
        out_net = hou.node("/out")
        if out_net is None:
            return {"ok": False, "error": "Scene /out network not found — is a Houdini session active?", "wrote_files": False}

        # Create the VAT ROP under /out (Driver context — grounded from plan riskNotes).
        rop = out_net.createNode("labs::vertex_animation_textures", "mcp_vat_export")

        # FIX 3: helper that destroys the ROP and returns a failure dict.
        def _fail_destroy(msg):
            try:
                rop.destroy()
            except Exception:
                pass
            return {"ok": False, "error": msg, "wrote_files": False}

        # ── Set ROP parameters (grounded parm names from plan riskNotes) ─────

        # soppath — string parm for SOP input path (no graph inputs for ROP nodes).
        soppath_parm = rop.parm("soppath")
        if soppath_parm is None:
            return _fail_destroy("parm 'soppath' not found on labs::vertex_animation_textures — SDK version mismatch?")
        soppath_parm.set(node)

        # mode — int parm: 0=soft, 1=rigid, 2=fluid, 3=sprite.
        mode_parm = rop.parm("mode")
        if mode_parm is None:
            return _fail_destroy("parm 'mode' not found on labs::vertex_animation_textures — SDK version mismatch?")
        mode_parm.set(vat_mode)

        # FIX 4 (plan AC): set target engine (non-fatal — silently skip if absent).
        engine_parm = rop.parm("engine")
        if engine_parm is not None:
            engine_parm.set("unreal")

        # FIX 4 (plan AC): exportpath is functionally required — fail-loud.
        exportpath_parm = rop.parm("exportpath")
        if exportpath_parm is None:
            return _fail_destroy("parm 'exportpath' not found on labs::vertex_animation_textures — SDK version mismatch?")
        exportpath_parm.set(_out_dir)

        # assetname — asset base name for file naming (optional; ROP has sensible default).
        assetname_parm = rop.parm("assetname")
        if assetname_parm is not None:
            assetname_parm.set(_asset_name)

        # f — parmTuple for frame range (start, end) (optional; ROP has sensible default).
        f_tuple = rop.parmTuple("f")
        if f_tuple is not None:
            f_tuple.set((f1, f2))

        # ── Trigger the bake ──────────────────────────────────────────────────
        # Press the execute button first; fall back to rop.render() on exception.
        try:
            exec_parm = rop.parm("execute")
            if exec_parm is not None:
                exec_parm.pressButton()
            else:
                rop.render()
        except Exception as cook_exc:
            errs = rop.errors()
            err_msg = "\n".join(errs) if errs else str(cook_exc)
            return _fail_destroy(err_msg)

        errs = rop.errors()
        if errs:
            return _fail_destroy("\n".join(errs))

        # ── Collect output file paths from the ROP parms ─────────────────────
        mesh = ""
        mesh_enable = rop.parm("enable_geo")
        mesh_path_parm = rop.parm("path_geo")
        if mesh_enable is not None and mesh_path_parm is not None and mesh_enable.eval():
            mesh = hou.text.expandString(mesh_path_parm.eval())

        textures = []
        for tex_name, enable_key, path_key in [
            ("pos",    "enable_pos",    "path_pos"),
            ("rot",    "enable_rot",    "path_rot"),
            ("col",    "enable_col",    "path_col"),
            ("lookup", "enable_lookup", "path_lookup"),
        ]:
            ep = rop.parm(enable_key)
            pp = rop.parm(path_key)
            if ep is not None and pp is not None and ep.eval():
                textures.append(hou.text.expandString(pp.eval()))

        # ── FIX 1: Build version triple via skew_table (not hardcoded) ────────
        _labs_for_skew = labs_vat if labs_vat is not None else "0.0"
        _skew_verdict, _skew_notes = skew_table.skew_verdict(
            houdini=hou.applicationVersionString(),
            labs_vat=_labs_for_skew,
            ue=target_ue,
        )
        version_triple = VersionTriple(
            houdini=hou.applicationVersionString(),
            labs_vat=labs_vat,
            ue=target_ue,
            verdict=_skew_verdict,
            notes=_skew_notes,
        )

        # ── Write ExportManifest sidecar (FR-8) ───────────────────────────────
        out_paths = ([mesh] if mesh else []) + list(textures)
        manifest = ExportManifest(
            tool="houdini_export_vat",
            args={
                "node": node,
                "out_dir": out_dir,
                "export_type": export_type,
                "asset_name": asset_name,
                "frame_range": [f1, f2],
                "target_ue": target_ue,
            },
            out_paths=out_paths,
            version_triple=version_triple,
            validator={},  # FIX 2: honest deferral — PR-7 wires the real validator
        )
        sidecar_path = _os.path.join(_out_dir, f"{_asset_name}.export.json")
        with open(sidecar_path, "w", encoding="utf-8") as _f:
            _json.dump(manifest.to_dict(), _f, indent=2)

        return {
            "ok": True,
            "node": rop.path(),
            "mesh": mesh,
            "textures": textures,
            "sidecar": sidecar_path,
            "vat_version": labs_vat,
            "version_triple": version_triple.to_dict(),
        }

    except Exception as exc:
        # FR-2: catch everything from createNode / parm.set / cook / file I/O.
        # FIX 3 (CVX-003): destroy the ROP if it was created before the exception.
        if rop is not None:
            try:
                rop.destroy()
            except Exception:
                pass
        return {"ok": False, "error": str(exc), "wrote_files": False}


# ---------------------------------------------------------------------------
# Handler registration — MUST be at the BOTTOM of the file
# (grounded against character_handlers.py registration pattern)
# ---------------------------------------------------------------------------

register_handler("probe_versions", probe_versions, Capability.READONLY)
register_handler("validate_budget", validate_budget, Capability.READONLY)
register_handler("export_vat", export_vat, Capability.MUTATING)
