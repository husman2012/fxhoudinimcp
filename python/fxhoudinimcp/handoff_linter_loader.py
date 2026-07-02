"""Thin loader: adds Homedini's scripts/python to sys.path so the fork can import
the handoff_linter engine without vendoring any source.

SYNC NOTE — single source of truth:
    Homedini ``scripts/python/homedini/rendering/handoff_linter/``
    This module IMPORTS it; it never copies or vendors engine source.
    Mirrors the pattern established by the PP12-109 security gate.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Venv-safe: import hou ONLY inside a try/except.
# The loader is imported in the fork venv where hou is absent.
# ---------------------------------------------------------------------------
try:
    import hou as _hou  # type: ignore[import-not-found]
except ImportError:
    _hou = None  # type: ignore[assignment]

# The 5 handoff_linter sub-modules to expose via load().
_ENGINE_MODULES = (
    "homedini.rendering.handoff_linter.handoff_model",
    "homedini.rendering.handoff_linter.presets",
    "homedini.rendering.handoff_linter.rules",
    "homedini.rendering.handoff_linter.exr_inspector",
    "homedini.rendering.handoff_linter.stage_reader",
)


def ensure_on_path() -> bool:
    """Add Homedini's scripts/python directory to sys.path if not already importable.

    Idempotent — safe to call multiple times; returns immediately if homedini is
    already importable.  NEVER raises; returns False if the directory cannot be
    located.

    Resolution order:
    1. Already importable — return True immediately.
    2. $HOMEDINI_PYTHON env var (explicit override).
    3. Houdini-side: ``hou.text.expandString('$UT/scripts/python')`` if hou is
       available in this interpreter.
    4. Sibling-repo layout: ``<fork-root>/../HoudiniUtilTools/scripts/python``
       (``__file__`` is at ``fxhoudinimcp/python/fxhoudinimcp/handoff_linter_loader.py``
       → 4 parent levels up to reach the ``development/`` directory).

    Returns:
        True if ``homedini`` is importable after this call; False otherwise.
    """
    # 1. Already importable — fast-exit (idempotent).
    if _homedini_importable():
        return True

    # Collect candidate paths; try them in order.
    candidates: list[Path] = []

    # 2. Explicit env var override.
    env_val = os.environ.get("HOMEDINI_PYTHON", "").strip()
    if env_val:
        candidates.append(Path(env_val))

    # 3. Houdini-side: $UT/scripts/python via hou.text.expandString.
    if _hou is not None:
        try:
            expanded = _hou.text.expandString("$UT/scripts/python")
            if expanded and "$UT" not in expanded:
                # expandString leaves the literal "$UT" when the variable is unset.
                candidates.append(Path(expanded))
        except Exception:  # noqa: BLE001 — never raise from ensure_on_path
            pass

    # 4. Sibling-repo layout derived from __file__.
    # __file__: .../fxhoudinimcp/python/fxhoudinimcp/handoff_linter_loader.py
    #  .parent → .../fxhoudinimcp/python/fxhoudinimcp/
    #  .parent → .../fxhoudinimcp/python/
    #  .parent → .../fxhoudinimcp/               (fork root)
    #  .parent → .../development/
    #  / "HoudiniUtilTools" / "scripts" / "python"
    try:
        sibling = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "HoudiniUtilTools"
            / "scripts"
            / "python"
        )
        candidates.append(sibling)
    except Exception:  # noqa: BLE001
        pass

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        if _homedini_importable():
            return True

    return False


def load() -> types.SimpleNamespace:
    """Ensure Homedini is on sys.path, then import and return the five handoff_linter modules.

    Returns:
        A ``types.SimpleNamespace`` with attributes:
        ``handoff_model``, ``presets``, ``rules``, ``exr_inspector``, ``stage_reader``.

    Raises:
        ImportError: if Homedini's scripts/python directory cannot be located or
            the engine modules fail to import.  The error message names every
            path tried so the caller can diagnose.
    """
    if not ensure_on_path():
        tried: list[str] = []
        env_val = os.environ.get("HOMEDINI_PYTHON", "").strip()
        if env_val:
            tried.append(f"$HOMEDINI_PYTHON={env_val!r}")
        try:
            sibling = (
                Path(__file__).resolve().parent.parent.parent.parent
                / "HoudiniUtilTools"
                / "scripts"
                / "python"
            )
            tried.append(f"sibling layout: {sibling}")
        except Exception:  # noqa: BLE001
            tried.append("sibling layout: (could not compute)")
        raise ImportError(
            "handoff_linter_loader: Homedini scripts/python is unreachable.  "
            "Paths tried: " + "; ".join(tried) + ".  "
            "Set $HOMEDINI_PYTHON to resolve."
        )

    ns = types.SimpleNamespace()
    for full_name in _ENGINE_MODULES:
        short = full_name.rsplit(".", 1)[-1]
        try:
            module = importlib.import_module(full_name)
        except ImportError as exc:
            raise ImportError(
                f"handoff_linter_loader.load(): failed to import {full_name!r}: {exc}"
            ) from exc
        setattr(ns, short, module)

    return ns


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _homedini_importable() -> bool:
    """Return True if ``homedini`` is currently importable (does not modify sys.path)."""
    try:
        importlib.import_module("homedini")
        return True
    except ImportError:
        return False
