"""
Pure-logic export budget rules.  No hou / Qt / pxr imports.

Pytest-able off-DCC (CL-015 — pure-logic boundary).

Public API
----------
check_gc_sequential(piece_ids)              -> BudgetCheck
check_frame_range(frame_range, max_frames)  -> BudgetCheck
run_budget_checks(geo_stats, target, preset=None) -> BudgetReport

Preset constants
----------------
UE_REALTIME   — conservative default limits for Unreal real-time targets
PRESETS       — registry of named presets  {name: dict}

Numeric band (shared rule for all numeric checks)
--------------------------------------------------
  value <= limit              -> "pass"
  limit < value <= limit * 2  -> "warn"
  value > limit * 2           -> "fail"

The multiplier (2.0) is the named constant _WARN_FAIL_MULTIPLIER.
"""
from __future__ import annotations

from fxhoudinimcp.export_model import BudgetCheck, BudgetReport, verdict_from_checks

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

#: Conservative real-time budget limits for Unreal Engine targets.
UE_REALTIME: dict = {
    "tris": 500_000,
    "texture_res": 4096,
    "max_frames": 600,
    "vat_textures": 4,
    "bones": 256,
}

#: Registry of named presets.  Keys are lowercase; values are limit dicts.
PRESETS: dict = {
    "ue_realtime": UE_REALTIME,
}

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

#: Multiplier applied to the limit to define the warn→fail boundary.
#: value <= limit            → pass
#: limit < value <= limit*M  → warn
#: value > limit*M           → fail
_WARN_FAIL_MULTIPLIER: float = 2.0

#: Allowlist of known export target identifiers.
#: Any target string NOT in this set raises ValueError in run_budget_checks.
#: alembic_ue / fbx / niagara are valid targets that produce only common checks;
#: chaos_gc and vat have additional target-conditional logic.
_VALID_TARGETS: frozenset = frozenset({"vat", "alembic_ue", "fbx", "niagara", "chaos_gc"})

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_preset(preset) -> dict:
    """Resolve *preset* to a limit dict.

    Args:
        preset: None → UE_REALTIME; str → looked up in PRESETS (ValueError if
                unknown); dict → used directly.

    Returns:
        A dict mapping budget-key → numeric limit.

    Raises:
        ValueError: when *preset* is a str not found in PRESETS.
    """
    if preset is None:
        return UE_REALTIME
    if isinstance(preset, dict):
        return preset
    # str lookup
    key = str(preset).lower()
    if key not in PRESETS:
        raise ValueError(
            f"Unknown preset {preset!r}; known presets: {sorted(PRESETS.keys())}"
        )
    return PRESETS[key]


def _check_numeric(check_id: str, value, limit) -> BudgetCheck:
    """Apply the numeric band rule and return a BudgetCheck.

    Band:
      value <= limit            → "pass"
      limit < value <= limit*2  → "warn"
      value > limit*2           → "fail"

    Args:
        check_id : identifier string for the check (e.g. "tris")
        value    : observed numeric value
        limit    : the upper limit for the "pass" zone

    Returns:
        BudgetCheck with id, status, value, and limit populated.
    """
    if value <= limit:
        status = "pass"
    elif value <= limit * _WARN_FAIL_MULTIPLIER:
        status = "warn"
    else:
        status = "fail"
    return BudgetCheck(id=check_id, status=status, value=value, limit=limit)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_gc_sequential(piece_ids: list) -> BudgetCheck:
    """Validate that Geometry Chaos piece IDs form a contiguous 0-based sequence.

    The piece IDs (typically the ``unreal_gc_piece`` attribute values) must form
    the complete integer range 0 .. max(piece_ids) with no gaps.  Duplicates are
    tolerated (the set of unique IDs is evaluated).

    Args:
        piece_ids: list of int piece ID values from the geometry.  May be
                   unsorted and may contain duplicates.

    Returns:
        BudgetCheck with:
          - id     = "gc_piece_sequential"
          - status = "pass" if contiguous from 0; "fail" otherwise
          - detail = human-readable summary (set on both pass and fail)
          - msg    = additional gap detail on fail (set alongside detail)
    """
    unique_ids = set(piece_ids)

    # Guard: empty list cannot form a valid sequence
    if not unique_ids:
        return BudgetCheck(
            id="gc_piece_sequential",
            status="fail",
            detail="no unreal_gc_piece values found; cannot validate sequence",
            msg="gap at 0",
        )

    # Guard: negative piece IDs are invalid — unreal_gc_piece must be 0-based
    if min(unique_ids) < 0:
        neg = min(unique_ids)
        return BudgetCheck(
            id="gc_piece_sequential",
            status="fail",
            detail=f"negative piece ID {neg} found",
            msg=str(neg),
        )

    max_id = max(unique_ids)
    required = set(range(max_id + 1))
    missing = required - unique_ids

    if missing:
        first_missing = min(missing)
        return BudgetCheck(
            id="gc_piece_sequential",
            status="fail",
            detail=f"sequence is non-contiguous; gap at {first_missing}",
            msg=f"gap at {first_missing}",
        )

    return BudgetCheck(
        id="gc_piece_sequential",
        status="pass",
        detail=f"0..{max_id} contiguous",
    )


def check_frame_range(frame_range: list, max_frames: int) -> BudgetCheck:
    """Check that a frame range does not exceed max_frames.

    Frame count is computed as ``end - start + 1`` (inclusive range).

    Args:
        frame_range : [start, end] list (both inclusive integers or floats).
        max_frames  : maximum allowed frame count.

    Returns:
        BudgetCheck with:
          - id    = "frame_range"
          - value = the input [start, end] list (preserved exactly)
          - limit = max_frames
          - status = "pass" if frame_count <= max_frames; "fail" otherwise
    """
    start, end = frame_range[0], frame_range[1]
    frame_count = end - start + 1
    status = "pass" if frame_count <= max_frames else "fail"
    return BudgetCheck(
        id="frame_range",
        status=status,
        value=list(frame_range),
        limit=max_frames,
    )


def run_budget_checks(geo_stats: dict, target: str, preset=None) -> BudgetReport:
    """Run all applicable budget checks for *target* using *preset* limits.

    Common numeric checks are run when the corresponding key is present in
    *geo_stats*:
      - "tris"        → numeric band check
      - "texture_res" → numeric band check
      - "frame_range" → check_frame_range
      - "bones"       → numeric band check

    Target-conditional checks:
      - "chaos_gc" → includes gc_piece_sequential when "gc_pieces" is in geo_stats;
                     NEVER includes vat_textures
      - "vat"      → ALWAYS includes vat_textures (observed value from
                      geo_stats["vat_textures"] when present, else 0);
                     NEVER includes gc_piece_sequential

    Args:
        geo_stats : dict of observed geometry statistics from the scene.
        target    : export target identifier; must be one of "chaos_gc", "vat",
                    "alembic_ue", "fbx", or "niagara".
        preset    : None → UE_REALTIME; str → named preset; dict → used directly.

    Returns:
        BudgetReport with verdict, checks list, and wrote_files=False.

    Raises:
        ValueError: if *preset* is an unknown string name.
        ValueError: if *target* is not in the known-target allowlist.
    """
    if target not in _VALID_TARGETS:
        raise ValueError(
            f"Unknown target {target!r}. Must be one of {sorted(_VALID_TARGETS)}"
        )

    limits = _resolve_preset(preset)
    checks: list[BudgetCheck] = []

    # ------------------------------------------------------------------
    # Common numeric checks — fire only when key is present in geo_stats
    # ------------------------------------------------------------------

    if "tris" in geo_stats:
        checks.append(_check_numeric("tris", geo_stats["tris"], limits["tris"]))

    if "texture_res" in geo_stats:
        checks.append(
            _check_numeric(
                "texture_res", geo_stats["texture_res"], limits["texture_res"]
            )
        )

    if "frame_range" in geo_stats:
        checks.append(
            check_frame_range(geo_stats["frame_range"], limits["max_frames"])
        )

    if "bones" in geo_stats:
        checks.append(_check_numeric("bones", geo_stats["bones"], limits["bones"]))

    # ------------------------------------------------------------------
    # Target-conditional checks
    # ------------------------------------------------------------------

    if target == "chaos_gc":
        # GC sequential check when piece data is provided
        if "gc_pieces" in geo_stats:
            checks.append(check_gc_sequential(geo_stats["gc_pieces"]))
        # vat_textures is NOT applicable to the chaos_gc target

    elif target == "vat":
        # vat_textures is ALWAYS included for the vat target
        # Observed value comes from geo_stats if present, else 0
        vat_value = geo_stats.get("vat_textures", 0)
        checks.append(
            _check_numeric("vat_textures", vat_value, limits["vat_textures"])
        )
        # gc_piece_sequential is NOT applicable to the vat target

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    final_verdict = verdict_from_checks(checks)
    return BudgetReport(verdict=final_verdict, checks=checks, wrote_files=False)
