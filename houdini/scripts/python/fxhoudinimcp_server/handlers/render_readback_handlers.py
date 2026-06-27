"""Handlers: render_lint_settings, render_parse_exr, render_read_pixels.

render_lint_settings — Karma render graph pre-render linting.
render_parse_exr     — Parse an on-disk EXR via hoiiotool (ExrManifest).
render_read_pixels   — Read pixel data from an on-disk EXR via OIIO.

All handlers are READ-ONLY, UNGATED (Capability.READONLY) — FR-10.
FR-2: missing/invalid arguments → {ok: False, error: "..."} (never silent).
FR-5: unexpected exceptions → {ok: False, error: str(exc)} (never propagate).

PP12-114 / pp12-114c (render_lint_settings, render_parse_exr)
PP12-114 / pp12-114e (render_read_pixels)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# sys.path bootstrap — 5 levels up from this file reaches the fork root;
# +/python adds the FastMCP-side fxhoudinimcp package so the loader import
# below resolves when this module is imported from hython.
#
#  __file__: .../fxhoudinimcp/houdini/scripts/python/fxhoudinimcp_server/handlers/render_readback_handlers.py
#   1 up → .../handlers/
#   2 up → .../fxhoudinimcp_server/
#   3 up → .../python/
#   4 up → .../scripts/
#   5 up → .../houdini/
#   6 up → .../fxhoudinimcp/             (fork root)
#  +/python → .../fxhoudinimcp/python/
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
from fxhoudinimcp import handoff_linter_loader  # noqa: E402

_log = _logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def render_lint_settings(render_node: str, preset: str = "nuke_safe") -> dict:
    """Read a Karma render node's USD stage and run handoff_linter rules on it.

    Args:
        render_node: Scene path of the Karma (usdrender_rop or similar) node,
            e.g. ``"/stage/karma1"``.  FR-2: must not be None or empty; the
            node must resolve via ``hou.node()``.
        preset: Name of the handoff_linter rule preset to evaluate against.
            Defaults to ``"nuke_safe"``.

    Returns:
        §4.2 shape on success::

            {
                "render_node": str,
                "preset": str,
                "results": [RuleResult.to_dict(), ...],
                "summary": {"ok": int, "warn": int, "error": int},
                "ready_to_render": bool,
            }

        FR-2 error shape on missing/invalid node or engine failure::

            {"ok": False, "error": "<human-readable message>"}
    """
    # FR-2: reject obviously invalid render_node values before touching hou.*
    if not render_node or not render_node.strip():
        return {"ok": False, "error": "render_node must be a non-empty scene path"}

    try:
        # Ensure homedini is importable; fail-loud if the engine is missing.
        if not handoff_linter_loader.ensure_on_path():
            return {
                "ok": False,
                "error": (
                    "handoff_linter engine not found on sys.path. "
                    "Set $HOMEDINI_PYTHON or ensure $UT is configured."
                ),
            }

        # FR-2: hou.node() returns None for any invalid path — surface explicitly.
        node = hou.node(render_node)
        if node is None:
            return {"ok": False, "error": f"Node not found: {render_node!r}"}

        # Import the engine modules via the loader (never vendor them here).
        from homedini.rendering.handoff_linter import stage_reader  # noqa: PLC0415
        from homedini.rendering.handoff_linter import rules as _rules  # noqa: PLC0415
        from homedini.rendering.handoff_linter import presets as _presets  # noqa: PLC0415

        # Read the USD stage report from the render node.
        report = stage_reader.read(node)

        # Load the rule preset by name.
        preset_obj = _presets.load(preset)

        # Evaluate all rules against the stage report.
        results = _rules.evaluate(report, preset_obj)

        # Summarize: count ok / warn / error severity buckets.
        summary = _rules.summarize(results)

        return {
            "render_node": render_node,
            "preset": preset,
            "results": [r.to_dict() for r in results],
            "summary": {"ok": summary["ok"], "warn": summary["warn"], "error": summary["error"]},
            "ready_to_render": summary["ready_to_render"],
        }

    except Exception as exc:  # noqa: BLE001 — all failures surface as {ok: False}
        _log.warning("render_lint_settings failed for %r: %s", render_node, exc, exc_info=True)
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Registration (READONLY — FR-10, ungated)
# ---------------------------------------------------------------------------

register_handler("render_lint_settings", render_lint_settings, Capability.READONLY)


# ---------------------------------------------------------------------------
# Handler: render_parse_exr
# ---------------------------------------------------------------------------

def render_parse_exr(exr_path: str, subimage: "int | None" = None) -> dict:
    """Run hoiiotool on exr_path and return an ExrManifest dict (§4.2 shape).

    Args:
        exr_path: Path to the EXR file; supports Houdini variable expansion
            (e.g. ``"$HIP/render/beauty.0001.exr"``).  FR-2: must not be
            empty or whitespace-only.
        subimage: When set, passes ``--subimage N`` to hoiiotool so only the
            requested subimage block is inspected.  ``None`` inspects all.

    Returns:
        §4.2 shape on success::

            {
                "exr_path": str,          # original (unexpanded) input path
                "is_multipart": bool,
                "subimages": int,
                "compression": str,
                "xres": int,
                "yres": int,
                "channels": [{"name": str, "layer": str | None, "dtype": str}, ...],
                "crypto_layers": [...],
                "metadata": {str: str},
            }

        FR-2/FR-5 error shape on failure::

            {"ok": False, "error": "<human-readable message>"}
    """
    # FR-2: reject obviously invalid exr_path values before touching hou.*.
    if not exr_path or not exr_path.strip():
        return {"ok": False, "error": "exr_path must be a non-empty path"}

    try:
        # Ensure homedini is importable; fail-loud if the engine is missing.
        if not handoff_linter_loader.ensure_on_path():
            return {
                "ok": False,
                "error": (
                    "handoff_linter engine not found on sys.path. "
                    "Set $HOMEDINI_PYTHON or ensure $UT is configured."
                ),
            }

        # Expand Houdini variables ($HIP, $HFS, $JOB, etc.) in the path.
        expanded = hou.text.expandString(exr_path)

        # Import the manifest parser via the loader (never vendor it here).
        from homedini.rendering.handoff_linter.exr_inspector import (  # noqa: PLC0415
            parse_exr_manifest,
        )

        # Parse the EXR into an ExrManifest.
        manifest = parse_exr_manifest(expanded, subimage=subimage)

        # Serialise; echo the ORIGINAL (unexpanded) input path as exr_path
        # so callers can round-trip the value they passed in.
        d = manifest.to_dict()
        d["exr_path"] = exr_path
        return d

    except Exception as exc:  # noqa: BLE001 — all failures surface as {ok: False}
        _log.warning("render_parse_exr failed for %r: %s", exr_path, exc, exc_info=True)
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Registration (READONLY — FR-10, ungated)
# ---------------------------------------------------------------------------

register_handler("render_parse_exr", render_parse_exr, Capability.READONLY)


# ---------------------------------------------------------------------------
# Handler: render_read_pixels
# ---------------------------------------------------------------------------

def render_read_pixels(
    source: str,
    plane: str = "C",
    mode: str = "summary",
    roi: list[int] | None = None,
    max_pixels: int = 4096,
    downsample: int = 1,
    page: int = 0,
    page_size: int = 1024,
) -> dict:
    """Read pixel data from an on-disk EXR file and return a readback dict.

    EXR-source v1: ``source`` must be a file-system path (supports Houdini
    variable expansion such as ``$HIP``).  In-scene COP node paths are not
    yet supported and return ``{ok: False, error: …}``.

    Args:
        source:    Path to the EXR file; supports Houdini variable expansion
                   (e.g. ``"$HIP/render/beauty.0001.exr"``).  FR-2: must not
                   be empty or whitespace-only.
        plane:     AOV plane name.  ``"C"`` / ``"beauty"`` select the top-level
                   beauty channels (R/G/B, no dot in name).  Default: ``"C"``.
        mode:      Readback mode — ``"summary"`` (metadata only, no pixel data),
                   ``"sample"`` (spaced sample of pixels), or ``"roi"``
                   (bounding-box slice).
        roi:       ``[x0, y0, x1, y1]`` half-open bounding box for
                   ``mode="roi"``.
        max_pixels: Maximum pixel count before auto-downsampling.
                   Default: 4096.
        downsample: Manual downsample factor (1 = no downsampling).
        page:      Page index for paginated reads.
        page_size: Page size in pixels.  Default: 1024.

    Returns:
        §4.2 ReadbackResult dict on success::

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

        FR-2/FR-5 error shape on failure::

            {"ok": False, "error": "<human-readable message>"}
    """
    # FR-2: reject obviously invalid source values.
    if not source or not source.strip():
        return {"ok": False, "error": "source must be a non-empty path"}

    # FR-2: reject unknown mode values before touching the file system.
    _VALID_MODES = {"summary", "roi", "sample"}
    if mode not in _VALID_MODES:
        return {
            "ok": False,
            "error": f"mode must be one of {sorted(_VALID_MODES)!r}, got {mode!r}",
        }

    try:
        from fxhoudinimcp import render_readback_reader  # noqa: PLC0415
        from fxhoudinimcp.render_readback_model import build_readback  # noqa: PLC0415

        # Expand Houdini variables ($HIP, $HFS, $JOB, etc.) in the path.
        expanded = hou.text.expandString(source)

        # EXR-source v1: detect in-scene COP/scene-node paths.
        # After expansion, real on-disk paths resolve to an existing file.
        # A hou.node() match (scene path) is not supported in this version.
        if not _os.path.isfile(expanded):
            if hou.node(expanded) is not None:
                return {
                    "ok": False,
                    "error": (
                        f"in-scene plane source not yet supported "
                        f"(EXR-source v1): {source!r}"
                    ),
                }
            return {
                "ok": False,
                "error": f"EXR file not found: {expanded!r}",
            }

        # Read the EXR channels (no subimage arg — reader defaults to None).
        channels, xres, yres, dtype = render_readback_reader.read_exr_plane(
            expanded, plane
        )

        # Delegate all readback logic to the pure-logic model function.
        return build_readback(
            channels=channels,
            plane=plane,
            xres=xres,
            yres=yres,
            dtype=dtype,
            mode=mode,
            roi=roi,
            max_pixels=max_pixels,
            downsample=downsample,
            page=page,
            page_size=page_size,
        )

    except Exception as exc:  # noqa: BLE001 — FR-5: all failures surface as {ok: False}
        _log.warning("render_read_pixels failed for %r: %s", source, exc, exc_info=True)
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Registration (READONLY — FR-10, ungated)
# ---------------------------------------------------------------------------

register_handler("render_read_pixels", render_read_pixels, Capability.READONLY)
