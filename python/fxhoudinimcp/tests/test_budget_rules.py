"""
Tests for budget_rules.py — pure-logic export-budget validator layer.

No hou / Qt / pxr imports anywhere in this file.  Runs under plain pytest
headless (off-DCC, no Houdini install required).

Covers the public contract of:
  - check_gc_sequential(piece_ids: list[int]) -> BudgetCheck
  - check_frame_range(frame_range: list, max_frames: int) -> BudgetCheck
  - run_budget_checks(geo_stats: dict, target: str, preset=None) -> BudgetReport

Budget-check statuses: "pass", "warn", "fail"
Numeric band: value <= limit → "pass"; limit < value <= limit*2 → "warn"; value > limit*2 → "fail"
UE_REALTIME preset: {tris:500000, texture_res:4096, max_frames:600, vat_textures:4, bones:256}

Target-conditional check set:
  - "chaos_gc"  → includes gc_piece_sequential; skips vat_textures
  - "vat"       → includes vat_textures; skips gc_piece_sequential

Validated behaviors (per pp12-111b locked plan):
  1.  check_gc_sequential([0,1,2,3])   → status='pass', detail contains '0..3' and 'contiguous'
  2.  check_gc_sequential([0,1,3])     → status='fail', detail or msg names 'gap at 2'
  3.  check_gc_sequential([1,2,3])     → status='fail', 'gap at 0' (missing origin)
  4.  check_gc_sequential([])          → status='fail' (empty list)
  5.  Numeric band: value<=limit → 'pass'; 540000 vs 500000 → 'warn'; value>2*limit → 'fail'
  6.  check_frame_range([1,120], max_frames) → id='frame_range', value==[1,120],
      pass when 120 frames <= max
  7.  run_budget_checks: target-conditional check set; preset=None == ue_realtime;
      unknown preset name → ValueError; plain dict passes through; wrote_files==False

TDD phase: RED — this file is authored BEFORE budget_rules.py exists.
All tests fail with ImportError / ModuleNotFoundError on first run.
"""

from __future__ import annotations

import sys
import os

# ---------------------------------------------------------------------------
# Path bootstrap — allow running as a standalone script as well as via pytest.
# Mirrors the pattern in test_skew_table.py.
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_PKG_ROOT))

import pytest

# ---------------------------------------------------------------------------
# Modules under test — do NOT exist yet; all tests below will be RED.
# ---------------------------------------------------------------------------
from fxhoudinimcp.budget_rules import (
    check_gc_sequential,
    check_frame_range,
    run_budget_checks,
)
from fxhoudinimcp.export_model import BudgetCheck, BudgetReport


# ===========================================================================
# Section 1 — check_gc_sequential: return type and field contract
#
# The function must return a BudgetCheck with:
#   id       = 'gc_piece_sequential'
#   status   ∈ {"pass", "fail"}      (no "warn" for this check)
#   detail   = human-readable detail string (populated on both pass and fail)
# ===========================================================================

class TestCheckGcSequentialContract:
    """check_gc_sequential return type and field shape."""

    def test_returns_budget_check_instance(self):
        """check_gc_sequential must return a BudgetCheck instance."""
        result = check_gc_sequential([0, 1, 2])
        assert isinstance(result, BudgetCheck), (
            f"check_gc_sequential must return BudgetCheck, got {type(result)!r}"
        )

    def test_id_is_gc_piece_sequential(self):
        """BudgetCheck.id must be 'gc_piece_sequential'."""
        result = check_gc_sequential([0, 1, 2])
        assert result.id == "gc_piece_sequential", (
            f"BudgetCheck.id must be 'gc_piece_sequential', got {result.id!r}"
        )

    def test_status_in_valid_vocabulary(self):
        """status must be 'pass' or 'fail' (never 'warn')."""
        for pieces in ([0, 1, 2], [0, 2], []):
            result = check_gc_sequential(pieces)
            assert result.status in ("pass", "fail"), (
                f"check_gc_sequential({pieces!r}) returned status={result.status!r}; "
                "expected 'pass' or 'fail'"
            )

    def test_detail_is_string_or_none_not_absent(self):
        """detail field must be a str (or explicitly None, not missing)."""
        result = check_gc_sequential([0, 1, 2])
        # The field exists on the dataclass; accessing it must not raise
        _ = result.detail  # just confirm the field is accessible


# ===========================================================================
# Section 2 — check_gc_sequential: PASS path
#
# A contiguous sequence starting at 0 must yield status='pass'.
# The detail string must contain both the range endpoint and the word
# 'contiguous' so the caller can present the evidence.
# ===========================================================================

class TestCheckGcSequentialPass:
    """check_gc_sequential pass-path assertions."""

    def test_contiguous_0_to_3_pass(self):
        """[0,1,2,3] is contiguous from 0 → status='pass'."""
        result = check_gc_sequential([0, 1, 2, 3])
        assert result.status == "pass", (
            f"[0,1,2,3] must pass; got status={result.status!r}, "
            f"detail={result.detail!r}"
        )

    def test_pass_detail_contains_range(self):
        """Pass detail must contain '0..3' to confirm the verified range."""
        result = check_gc_sequential([0, 1, 2, 3])
        assert result.detail is not None, (
            "detail must be set on pass (not None)"
        )
        assert "0..3" in result.detail, (
            f"detail for [0,1,2,3] must contain '0..3'; got {result.detail!r}"
        )

    def test_pass_detail_contains_contiguous(self):
        """Pass detail must contain the word 'contiguous'."""
        result = check_gc_sequential([0, 1, 2, 3])
        assert result.detail is not None
        assert "contiguous" in result.detail, (
            f"detail for [0,1,2,3] must contain 'contiguous'; got {result.detail!r}"
        )

    def test_single_piece_at_0_passes(self):
        """[0] is a valid single-piece sequence → status='pass'."""
        result = check_gc_sequential([0])
        assert result.status == "pass", (
            f"[0] (single piece at origin) must pass; got {result.status!r}"
        )

    def test_unsorted_input_still_passes(self):
        """Input need not be sorted; [3,1,0,2] is still contiguous."""
        result = check_gc_sequential([3, 1, 0, 2])
        assert result.status == "pass", (
            f"[3,1,0,2] (unsorted contiguous) must pass; got {result.status!r}"
        )

    def test_duplicate_ids_still_passes(self):
        """Duplicates are tolerated; {0,1,2,3} as a set is contiguous."""
        result = check_gc_sequential([0, 1, 1, 2, 2, 3])
        assert result.status == "pass", (
            f"[0,1,1,2,2,3] (duplicates, contiguous after dedup) must pass; "
            f"got {result.status!r}"
        )


# ===========================================================================
# Section 3 — check_gc_sequential: FAIL path
#
# Gap in middle, missing origin (0), or empty list → status='fail'.
# The detail or msg must identify the first missing piece index.
# ===========================================================================

class TestCheckGcSequentialFail:
    """check_gc_sequential fail-path assertions."""

    def test_gap_in_middle_fails(self):
        """[0,1,3] has a gap at 2 → status='fail'."""
        result = check_gc_sequential([0, 1, 3])
        assert result.status == "fail", (
            f"[0,1,3] must fail due to gap at 2; got {result.status!r}"
        )

    def test_gap_in_middle_names_missing_index(self):
        """The gap message must identify 'gap at 2' for [0,1,3]."""
        result = check_gc_sequential([0, 1, 3])
        combined = " ".join(filter(None, [result.detail, result.msg]))
        assert "gap at 2" in combined, (
            f"[0,1,3] gap message must say 'gap at 2'; "
            f"detail={result.detail!r}, msg={result.msg!r}"
        )

    def test_missing_origin_fails(self):
        """[1,2,3] is missing piece 0 → status='fail'."""
        result = check_gc_sequential([1, 2, 3])
        assert result.status == "fail", (
            f"[1,2,3] (missing 0) must fail; got {result.status!r}"
        )

    def test_missing_origin_names_gap_at_0(self):
        """The gap message for [1,2,3] must say 'gap at 0'."""
        result = check_gc_sequential([1, 2, 3])
        combined = " ".join(filter(None, [result.detail, result.msg]))
        assert "gap at 0" in combined, (
            f"[1,2,3] gap message must say 'gap at 0'; "
            f"detail={result.detail!r}, msg={result.msg!r}"
        )

    def test_empty_list_fails(self):
        """[] (empty piece list) → status='fail'."""
        result = check_gc_sequential([])
        assert result.status == "fail", (
            f"[] (empty) must fail; got {result.status!r}"
        )

    def test_gap_reports_minimum_missing(self):
        """For [0, 2, 3], the minimum missing is 1 → 'gap at 1'."""
        result = check_gc_sequential([0, 2, 3])
        combined = " ".join(filter(None, [result.detail, result.msg]))
        assert "gap at 1" in combined, (
            f"[0,2,3] gap message must name 'gap at 1' (minimum missing); "
            f"detail={result.detail!r}, msg={result.msg!r}"
        )


# ===========================================================================
# Section 4 — Numeric band (shared helper logic)
#
# The numeric-band contract is exercised indirectly through run_budget_checks
# (tris check) but is also verifiable via any numeric check exposed by the
# module.  We test here via run_budget_checks with a controlled geo_stats dict.
#
# Band thresholds (UE_REALTIME tris limit = 500_000):
#   value <= 500_000           → pass
#   500_000 < value <= 1_000_000  → warn
#   value > 1_000_000          → fail
# ===========================================================================

class TestNumericBand:
    """Numeric-band logic via run_budget_checks tris check."""

    # ------------------------------------------------------------------
    # We run run_budget_checks with chaos_gc target (no vat requirement)
    # and a geo_stats dict that contains only 'tris' so the other numeric
    # checks (texture_res, bones) are skipped for clarity.
    # ------------------------------------------------------------------

    def _tris_check(self, tris_value: int) -> BudgetCheck:
        """Run run_budget_checks with only 'tris' in geo_stats, return the tris check."""
        geo_stats = {"tris": tris_value}
        report = run_budget_checks(geo_stats, target="chaos_gc", preset="ue_realtime")
        checks_by_id = {c.id: c for c in report.checks}
        assert "tris" in checks_by_id, (
            f"Expected a 'tris' check in report.checks; "
            f"got ids: {list(checks_by_id.keys())}"
        )
        return checks_by_id["tris"]

    def test_at_limit_passes(self):
        """500_000 tris == limit → status='pass'."""
        check = self._tris_check(500_000)
        assert check.status == "pass", (
            f"500_000 tris (at limit) must pass; got {check.status!r}"
        )

    def test_below_limit_passes(self):
        """250_000 tris < limit → status='pass'."""
        check = self._tris_check(250_000)
        assert check.status == "pass", (
            f"250_000 tris (below limit) must pass; got {check.status!r}"
        )

    def test_warn_zone_540k(self):
        """540_000 tris is in the warn zone (limit < value <= 2*limit) → status='warn'."""
        check = self._tris_check(540_000)
        assert check.status == "warn", (
            f"540_000 tris (warn zone) must be 'warn'; got {check.status!r}"
        )

    def test_warn_zone_at_double_limit(self):
        """1_000_000 tris == 2*limit boundary → still 'warn' (value <= 2*limit)."""
        check = self._tris_check(1_000_000)
        assert check.status == "warn", (
            f"1_000_000 tris (exactly 2*limit) must be 'warn'; got {check.status!r}"
        )

    def test_fail_zone_above_double(self):
        """1_000_001 tris > 2*limit → status='fail'."""
        check = self._tris_check(1_000_001)
        assert check.status == "fail", (
            f"1_000_001 tris (above 2*limit) must fail; got {check.status!r}"
        )

    def test_far_above_limit_fails(self):
        """3_000_000 tris >> 2*limit → status='fail'."""
        check = self._tris_check(3_000_000)
        assert check.status == "fail", (
            f"3_000_000 tris must fail; got {check.status!r}"
        )

    def test_limit_recorded_on_check(self):
        """The tris BudgetCheck must record the limit value (500_000 for ue_realtime)."""
        check = self._tris_check(250_000)
        assert check.limit == 500_000, (
            f"tris check.limit must be 500_000 (ue_realtime preset); "
            f"got {check.limit!r}"
        )

    def test_value_recorded_on_check(self):
        """The tris BudgetCheck must record the observed value."""
        check = self._tris_check(300_000)
        assert check.value == 300_000, (
            f"tris check.value must equal the observed value 300_000; "
            f"got {check.value!r}"
        )


# ===========================================================================
# Section 5 — check_frame_range: return type and field contract
# ===========================================================================

class TestCheckFrameRangeContract:
    """check_frame_range return type and field contract."""

    def test_returns_budget_check(self):
        """check_frame_range must return a BudgetCheck."""
        result = check_frame_range([1, 120], 600)
        assert isinstance(result, BudgetCheck), (
            f"check_frame_range must return BudgetCheck, got {type(result)!r}"
        )

    def test_id_is_frame_range(self):
        """BudgetCheck.id must be 'frame_range'."""
        result = check_frame_range([1, 120], 600)
        assert result.id == "frame_range", (
            f"check.id must be 'frame_range'; got {result.id!r}"
        )

    def test_value_equals_input_range(self):
        """BudgetCheck.value must equal the input [start, end] list."""
        result = check_frame_range([1, 120], 600)
        assert result.value == [1, 120], (
            f"check.value must be [1, 120]; got {result.value!r}"
        )

    def test_value_round_trips_both_endpoints(self):
        """check_frame_range stores [start, end] — both endpoints visible."""
        result = check_frame_range([10, 250], 600)
        assert result.value == [10, 250], (
            f"check.value must be [10, 250]; got {result.value!r}"
        )


# ===========================================================================
# Section 6 — check_frame_range: PASS and FAIL paths
#
# Frame count = end - start + 1 (e.g. [1, 120] → 120 frames).
# pass when frame_count <= max_frames.
# fail when frame_count > max_frames.
# ===========================================================================

class TestCheckFrameRangePassFail:
    """check_frame_range pass and fail paths."""

    def test_pass_when_within_limit(self):
        """[1, 120] with max_frames=600 → 120 frames ≤ 600 → pass."""
        result = check_frame_range([1, 120], 600)
        assert result.status == "pass", (
            f"[1,120] / max=600 must pass; got {result.status!r}"
        )

    def test_pass_at_exact_limit(self):
        """[1, 600] with max_frames=600 → exactly 600 frames → pass."""
        result = check_frame_range([1, 600], 600)
        assert result.status == "pass", (
            f"[1,600] at exact limit must pass; got {result.status!r}"
        )

    def test_fail_when_over_limit(self):
        """[1, 601] with max_frames=600 → 601 frames > 600 → fail."""
        result = check_frame_range([1, 601], 600)
        assert result.status == "fail", (
            f"[1,601] must fail (over limit); got {result.status!r}"
        )

    def test_pass_small_range(self):
        """[0, 30] with max_frames=600 → 31 frames → pass."""
        result = check_frame_range([0, 30], 600)
        assert result.status == "pass", (
            f"[0,30] must pass; got {result.status!r}"
        )

    def test_fail_large_range(self):
        """[1, 1200] with max_frames=600 → 1200 frames > 600 → fail."""
        result = check_frame_range([1, 1200], 600)
        assert result.status == "fail", (
            f"[1,1200] must fail; got {result.status!r}"
        )


# ===========================================================================
# Section 7 — run_budget_checks: BudgetReport shape
#
# run_budget_checks(geo_stats, target, preset=None) -> BudgetReport
#   - verdict   : aggregated "pass"/"warn"/"fail"
#   - checks    : list of BudgetCheck
#   - wrote_files: MUST be False (budget checks never write files)
# ===========================================================================

class TestRunBudgetChecksShape:
    """run_budget_checks BudgetReport shape."""

    def test_returns_budget_report(self):
        """run_budget_checks must return a BudgetReport."""
        report = run_budget_checks({}, target="chaos_gc", preset="ue_realtime")
        assert isinstance(report, BudgetReport), (
            f"run_budget_checks must return BudgetReport, got {type(report)!r}"
        )

    def test_wrote_files_is_false(self):
        """wrote_files must always be False — budget checks never write files."""
        report = run_budget_checks({}, target="chaos_gc", preset="ue_realtime")
        assert report.wrote_files is False, (
            f"wrote_files must be False; got {report.wrote_files!r}"
        )

    def test_verdict_is_valid_string(self):
        """verdict must be one of 'pass', 'warn', 'fail'."""
        report = run_budget_checks({}, target="chaos_gc", preset="ue_realtime")
        assert report.verdict in ("pass", "warn", "fail"), (
            f"verdict must be 'pass'/'warn'/'fail'; got {report.verdict!r}"
        )

    def test_checks_is_list(self):
        """checks must be a list."""
        report = run_budget_checks({}, target="chaos_gc", preset="ue_realtime")
        assert isinstance(report.checks, list), (
            f"report.checks must be a list; got {type(report.checks)!r}"
        )

    def test_verdict_aggregates_checks(self):
        """verdict must equal verdict_from_checks(report.checks)."""
        from fxhoudinimcp.export_model import verdict_from_checks
        geo_stats = {"tris": 300_000}
        report = run_budget_checks(geo_stats, target="chaos_gc", preset="ue_realtime")
        expected = verdict_from_checks(report.checks)
        assert report.verdict == expected, (
            f"report.verdict={report.verdict!r} must equal "
            f"verdict_from_checks(checks)={expected!r}"
        )


# ===========================================================================
# Section 8 — run_budget_checks: preset resolution
#
# preset=None resolves to the ue_realtime defaults.
# preset="ue_realtime" is also explicit and equivalent.
# An unknown preset name raises ValueError.
# A plain dict passes through as the preset directly.
# ===========================================================================

class TestRunBudgetChecksPreset:
    """run_budget_checks preset resolution."""

    def test_none_preset_uses_ue_realtime(self):
        """preset=None uses UE_REALTIME limits (tris limit = 500_000)."""
        # 250_000 tris < 500_000 → pass under ue_realtime
        geo_stats = {"tris": 250_000}
        report_none = run_budget_checks(geo_stats, target="chaos_gc", preset=None)
        report_ue   = run_budget_checks(geo_stats, target="chaos_gc", preset="ue_realtime")
        checks_none = {c.id: c for c in report_none.checks}
        checks_ue   = {c.id: c for c in report_ue.checks}
        assert "tris" in checks_none and "tris" in checks_ue, (
            "Both preset=None and preset='ue_realtime' must produce a tris check"
        )
        assert checks_none["tris"].limit == checks_ue["tris"].limit, (
            f"preset=None and preset='ue_realtime' must give the same tris limit; "
            f"got {checks_none['tris'].limit!r} vs {checks_ue['tris'].limit!r}"
        )

    def test_ue_realtime_tris_limit(self):
        """UE_REALTIME preset tris limit is 500_000."""
        geo_stats = {"tris": 1}
        report = run_budget_checks(geo_stats, target="chaos_gc", preset="ue_realtime")
        checks = {c.id: c for c in report.checks}
        assert checks["tris"].limit == 500_000, (
            f"ue_realtime tris limit must be 500_000; got {checks['tris'].limit!r}"
        )

    def test_ue_realtime_texture_res_limit(self):
        """UE_REALTIME preset texture_res limit is 4096."""
        geo_stats = {"tris": 1, "texture_res": 1024}
        report = run_budget_checks(geo_stats, target="chaos_gc", preset="ue_realtime")
        checks = {c.id: c for c in report.checks}
        assert "texture_res" in checks, (
            "Expected a 'texture_res' check when geo_stats contains 'texture_res'"
        )
        assert checks["texture_res"].limit == 4096, (
            f"ue_realtime texture_res limit must be 4096; "
            f"got {checks['texture_res'].limit!r}"
        )

    def test_ue_realtime_max_frames_limit(self):
        """UE_REALTIME preset max_frames is 600."""
        geo_stats = {"frame_range": [1, 100]}
        report = run_budget_checks(geo_stats, target="chaos_gc", preset="ue_realtime")
        checks = {c.id: c for c in report.checks}
        assert "frame_range" in checks, (
            "Expected a 'frame_range' check when geo_stats contains 'frame_range'"
        )
        # The frame_range check uses max_frames=600 internally; the check's limit
        # may expose it or embed it in msg — the pass/fail behavior validates it.
        result = checks["frame_range"]
        # 100 frames <= 600 must pass
        assert result.status == "pass", (
            f"[1,100] with ue_realtime (max_frames=600) must pass; "
            f"got {result.status!r}"
        )

    def test_ue_realtime_bones_limit(self):
        """UE_REALTIME preset bones limit is 256."""
        geo_stats = {"tris": 1, "bones": 100}
        report = run_budget_checks(geo_stats, target="chaos_gc", preset="ue_realtime")
        checks = {c.id: c for c in report.checks}
        assert "bones" in checks, (
            "Expected a 'bones' check when geo_stats contains 'bones'"
        )
        assert checks["bones"].limit == 256, (
            f"ue_realtime bones limit must be 256; got {checks['bones'].limit!r}"
        )

    def test_unknown_preset_name_raises_value_error(self):
        """An unknown preset name string must raise ValueError."""
        with pytest.raises(ValueError):
            run_budget_checks({}, target="chaos_gc", preset="no_such_preset_xyz")

    def test_dict_preset_passes_through(self):
        """A plain dict preset is used directly without raising."""
        custom = {"tris": 100, "texture_res": 512, "max_frames": 30,
                  "vat_textures": 2, "bones": 64}
        # Should not raise
        report = run_budget_checks({"tris": 50}, target="chaos_gc", preset=custom)
        assert isinstance(report, BudgetReport)

    def test_dict_preset_tris_limit_used(self):
        """A dict preset's 'tris' key is used as the tris limit."""
        custom = {"tris": 100, "texture_res": 512, "max_frames": 30,
                  "vat_textures": 2, "bones": 64}
        geo_stats = {"tris": 50}
        report = run_budget_checks(geo_stats, target="chaos_gc", preset=custom)
        checks = {c.id: c for c in report.checks}
        assert "tris" in checks, "Expected tris check with dict preset"
        assert checks["tris"].limit == 100, (
            f"Dict preset tris limit must be 100; got {checks['tris'].limit!r}"
        )

    def test_dict_preset_fail_when_over_limit(self):
        """Dict preset: value > limit → fail using custom limit."""
        custom = {"tris": 100, "texture_res": 512, "max_frames": 30,
                  "vat_textures": 2, "bones": 64}
        geo_stats = {"tris": 300}  # 300 > 2*100 → fail
        report = run_budget_checks(geo_stats, target="chaos_gc", preset=custom)
        checks = {c.id: c for c in report.checks}
        assert checks["tris"].status == "fail", (
            f"tris=300 with limit=100 (2*100=200) must fail; "
            f"got {checks['tris'].status!r}"
        )


# ===========================================================================
# Section 9 — run_budget_checks: target-conditional check set
#
# "chaos_gc" target:
#   - includes gc_piece_sequential check when 'gc_pieces' in geo_stats
#   - does NOT include vat_textures check
#
# "vat" target:
#   - includes vat_textures check when geo_stats has no specific override
#   - does NOT include gc_piece_sequential check
#
# Both targets include tris/texture_res/frame_range/bones checks
# when the corresponding geo_stats keys are present.
# ===========================================================================

class TestRunBudgetChecksTargetConditional:
    """Target-conditional check set."""

    def test_chaos_gc_includes_gc_piece_sequential(self):
        """chaos_gc target includes gc_piece_sequential check when gc_pieces present."""
        geo_stats = {"gc_pieces": [0, 1, 2, 3]}
        report = run_budget_checks(geo_stats, target="chaos_gc", preset="ue_realtime")
        check_ids = {c.id for c in report.checks}
        assert "gc_piece_sequential" in check_ids, (
            f"chaos_gc target must include gc_piece_sequential; "
            f"got check ids: {check_ids}"
        )

    def test_chaos_gc_does_not_include_vat_textures(self):
        """chaos_gc target must NOT include the vat_textures check."""
        geo_stats = {"tris": 100}
        report = run_budget_checks(geo_stats, target="chaos_gc", preset="ue_realtime")
        check_ids = {c.id for c in report.checks}
        assert "vat_textures" not in check_ids, (
            f"chaos_gc target must not include vat_textures; "
            f"got check ids: {check_ids}"
        )

    def test_vat_includes_vat_textures(self):
        """vat target includes vat_textures check."""
        geo_stats = {"tris": 100}
        report = run_budget_checks(geo_stats, target="vat", preset="ue_realtime")
        check_ids = {c.id for c in report.checks}
        assert "vat_textures" in check_ids, (
            f"vat target must include vat_textures; got check ids: {check_ids}"
        )

    def test_vat_does_not_include_gc_piece_sequential(self):
        """vat target must NOT include gc_piece_sequential check."""
        geo_stats = {"tris": 100}
        report = run_budget_checks(geo_stats, target="vat", preset="ue_realtime")
        check_ids = {c.id for c in report.checks}
        assert "gc_piece_sequential" not in check_ids, (
            f"vat target must not include gc_piece_sequential; "
            f"got check ids: {check_ids}"
        )

    def test_tris_check_present_for_both_targets_when_in_geo_stats(self):
        """tris check appears for both chaos_gc and vat when 'tris' in geo_stats."""
        geo_stats = {"tris": 100}
        for target in ("chaos_gc", "vat"):
            report = run_budget_checks(geo_stats, target=target, preset="ue_realtime")
            check_ids = {c.id for c in report.checks}
            assert "tris" in check_ids, (
                f"tris check must appear for target={target!r}; "
                f"got ids: {check_ids}"
            )

    def test_texture_res_check_only_when_key_present(self):
        """texture_res check appears only when 'texture_res' key is in geo_stats."""
        # Without the key
        report_no = run_budget_checks({}, target="chaos_gc", preset="ue_realtime")
        ids_no = {c.id for c in report_no.checks}
        assert "texture_res" not in ids_no, (
            "texture_res check must not appear when key absent from geo_stats"
        )
        # With the key
        report_yes = run_budget_checks({"texture_res": 2048}, target="chaos_gc",
                                       preset="ue_realtime")
        ids_yes = {c.id for c in report_yes.checks}
        assert "texture_res" in ids_yes, (
            "texture_res check must appear when 'texture_res' is in geo_stats"
        )

    def test_bones_check_only_when_key_present(self):
        """bones check appears only when 'bones' key is in geo_stats."""
        report_no = run_budget_checks({}, target="chaos_gc", preset="ue_realtime")
        ids_no = {c.id for c in report_no.checks}
        assert "bones" not in ids_no, (
            "bones check must not appear when key absent from geo_stats"
        )
        report_yes = run_budget_checks({"bones": 128}, target="chaos_gc",
                                       preset="ue_realtime")
        ids_yes = {c.id for c in report_yes.checks}
        assert "bones" in ids_yes, (
            "bones check must appear when 'bones' is in geo_stats"
        )

    def test_frame_range_check_only_when_key_present(self):
        """frame_range check appears only when 'frame_range' key is in geo_stats."""
        report_no = run_budget_checks({}, target="chaos_gc", preset="ue_realtime")
        ids_no = {c.id for c in report_no.checks}
        assert "frame_range" not in ids_no, (
            "frame_range check must not appear when key absent from geo_stats"
        )
        report_yes = run_budget_checks({"frame_range": [1, 100]}, target="chaos_gc",
                                       preset="ue_realtime")
        ids_yes = {c.id for c in report_yes.checks}
        assert "frame_range" in ids_yes, (
            "frame_range check must appear when 'frame_range' is in geo_stats"
        )


# ===========================================================================
# Section 10 — run_budget_checks: vat_textures numeric band
#
# UE_REALTIME vat_textures limit = 4.
# The numeric band applies: value<=4 → pass; 4<value<=8 → warn; value>8 → fail.
# ===========================================================================

class TestVatTexturesBand:
    """vat_textures numeric band via vat target."""

    def _vat_check(self, value: int) -> BudgetCheck:
        geo_stats = {"tris": 1}  # include tris to ensure other checks are minimal
        report = run_budget_checks(geo_stats, target="vat", preset="ue_realtime")
        checks = {c.id: c for c in report.checks}
        # Override: we need vat_textures check driven by preset, not geo_stats key.
        # The vat_textures check is always included for the "vat" target.
        # We inject a specific value via geo_stats["vat_textures"] if needed,
        # OR the check runs against the preset default.
        # Since we want to test different values, re-run with a custom dict preset.
        custom = {"tris": 999_999, "texture_res": 9999, "max_frames": 9999,
                  "vat_textures": 4, "bones": 9999}
        # We need to pass a geo_stats that overrides the vat_textures threshold.
        # Actually: vat_textures in geo_stats is the OBSERVED value, limit comes
        # from the preset. Use geo_stats={"vat_textures": value} so the check fires.
        report2 = run_budget_checks(
            {"vat_textures": value},
            target="vat",
            preset="ue_realtime",
        )
        checks2 = {c.id: c for c in report2.checks}
        assert "vat_textures" in checks2, (
            f"Expected vat_textures check; got ids: {list(checks2.keys())}"
        )
        return checks2["vat_textures"]

    def test_at_limit_passes(self):
        """vat_textures=4 == limit → pass."""
        check = self._vat_check(4)
        assert check.status == "pass", (
            f"vat_textures=4 (at limit) must pass; got {check.status!r}"
        )

    def test_warn_zone(self):
        """vat_textures=6 (limit < 6 <= 2*4=8) → warn."""
        check = self._vat_check(6)
        assert check.status == "warn", (
            f"vat_textures=6 (warn zone) must be 'warn'; got {check.status!r}"
        )

    def test_fail_zone(self):
        """vat_textures=9 (> 2*4=8) → fail."""
        check = self._vat_check(9)
        assert check.status == "fail", (
            f"vat_textures=9 (fail zone) must be 'fail'; got {check.status!r}"
        )

    def test_limit_is_4_for_ue_realtime(self):
        """vat_textures limit for ue_realtime preset must be 4."""
        check = self._vat_check(1)
        assert check.limit == 4, (
            f"vat_textures limit must be 4; got {check.limit!r}"
        )


# ===========================================================================
# Section 11 — hou-free import verification (CL-015)
#
# budget_rules.py must import with zero hou/Qt/pxr at module top-level.
# Mirrors the purity section pattern from test_skew_table.py.
# ===========================================================================

class TestHouFreeImport:
    """Confirm budget_rules.py carries no hou/Qt/pxr dependency (CL-015)."""

    def test_module_importable_without_hou(self):
        """budget_rules must load under plain Python with no hou installed."""
        import fxhoudinimcp.budget_rules as br
        assert br is not None

    def test_hou_not_in_budget_rules_source(self):
        """budget_rules.py must not reference 'hou' as a top-level import."""
        import fxhoudinimcp.budget_rules as br
        import inspect
        import re
        source = inspect.getsource(br)
        source_no_comments = re.sub(r"#.*", "", source)
        assert "import hou" not in source_no_comments, (
            "budget_rules.py must not import hou (CL-015 — pure-logic boundary)"
        )

    def test_pyside6_not_in_budget_rules_source(self):
        """budget_rules.py must not import PySide6."""
        import fxhoudinimcp.budget_rules as br
        import inspect
        import re
        source = inspect.getsource(br)
        source_no_comments = re.sub(r"#.*", "", source)
        assert "PySide6" not in source_no_comments, (
            "budget_rules.py must not import PySide6 (CL-015)"
        )

    def test_pxr_not_in_budget_rules_source(self):
        """budget_rules.py must not import pxr."""
        import fxhoudinimcp.budget_rules as br
        import inspect
        import re
        source = inspect.getsource(br)
        source_no_comments = re.sub(r"#.*", "", source)
        assert "import pxr" not in source_no_comments and \
               "from pxr" not in source_no_comments, (
            "budget_rules.py must not import pxr (CL-015)"
        )


# ===========================================================================
# Section 12 — Round-2 hardening (pp12-111b)
#
# Pins two accepted Codex findings from the tier-2 review round-1:
#
#   CX-001  run_budget_checks L251: if/elif, no else → unknown/typo target
#           silently returns a passing BudgetReport instead of raising ValueError.
#           Fix: an allowlist of known targets {vat, alembic_ue, fbx, niagara,
#           chaos_gc}; targets outside the set raise ValueError.
#
#   CX-002  check_gc_sequential L136-148: negative piece IDs pass silently.
#           Example: [-1, 0] → unique={-1,0}, max=0, required={0}, missing={}
#           → "pass".  Fix: if min(unique_ids) < 0 → fail with the negative
#           value named in the message.
#
# These tests must be RED against the current impl and GREEN after hou-dev's
# round-2 fix lands.  The regression-guard tests within each CX section must
# be GREEN against both the current impl and the fixed impl.
# ===========================================================================

class TestRound2Hardening:
    """Pins CX-001 (unknown target raises ValueError) and CX-002 (negative
    gc piece IDs → fail) from pp12-111b round-1 adjudication.

    Tests labelled "RED against current impl" fail NOW and are fixed by hou-dev.
    Tests labelled "regression guard" pass NOW and must continue to pass after
    the fix.
    """

    # -----------------------------------------------------------------------
    # CX-001: unknown / typo target → ValueError
    # -----------------------------------------------------------------------

    def test_cx001_typo_target_raises_value_error(self):
        """[RED] run_budget_checks with a typo target must raise ValueError.

        "chaos_GC" is a casing typo — not in the known-target allowlist.
        Current impl: silently returns a passing BudgetReport.
        Fixed impl: raises ValueError naming the unknown target.
        """
        with pytest.raises(ValueError):
            run_budget_checks({}, target="chaos_GC", preset="ue_realtime")

    def test_cx001_misspelled_target_raises_value_error(self):
        """[RED] run_budget_checks with 'vat_character' (invalid) must raise ValueError.

        'vat_character' is not in the known-target allowlist.
        """
        with pytest.raises(ValueError):
            run_budget_checks({}, target="vat_character", preset="ue_realtime")

    def test_cx001_bogus_target_raises_value_error(self):
        """[RED] run_budget_checks with a completely bogus target must raise ValueError.

        'bogus' has no resemblance to any valid target.
        """
        with pytest.raises(ValueError):
            run_budget_checks({}, target="bogus", preset="ue_realtime")

    def test_cx001_empty_string_target_raises_value_error(self):
        """[RED] run_budget_checks with '' (empty string) target must raise ValueError."""
        with pytest.raises(ValueError):
            run_budget_checks({}, target="", preset="ue_realtime")

    # -----------------------------------------------------------------------
    # CX-001: regression guards — valid non-special targets must NOT raise
    #
    # alembic_ue, fbx, niagara are valid targets that have no gc/vat checks.
    # An over-broad fix that rejects them would be wrong.
    # These pass against the current impl and must continue to pass after the fix.
    # -----------------------------------------------------------------------

    def test_cx001_alembic_ue_target_returns_budget_report(self):
        """[regression guard] 'alembic_ue' is a valid target — must NOT raise."""
        report = run_budget_checks({"tris": 100}, target="alembic_ue",
                                   preset="ue_realtime")
        assert isinstance(report, BudgetReport), (
            f"'alembic_ue' target must return BudgetReport, got {type(report)!r}"
        )

    def test_cx001_alembic_ue_no_gc_check(self):
        """[regression guard] 'alembic_ue' target must NOT include gc_piece_sequential."""
        report = run_budget_checks({"tris": 100, "gc_pieces": [0, 1, 2]},
                                   target="alembic_ue", preset="ue_realtime")
        check_ids = {c.id for c in report.checks}
        assert "gc_piece_sequential" not in check_ids, (
            f"'alembic_ue' must not include gc_piece_sequential; got {check_ids}"
        )

    def test_cx001_alembic_ue_no_vat_check(self):
        """[regression guard] 'alembic_ue' target must NOT include vat_textures."""
        report = run_budget_checks({"tris": 100}, target="alembic_ue",
                                   preset="ue_realtime")
        check_ids = {c.id for c in report.checks}
        assert "vat_textures" not in check_ids, (
            f"'alembic_ue' must not include vat_textures; got {check_ids}"
        )

    def test_cx001_fbx_target_returns_budget_report(self):
        """[regression guard] 'fbx' is a valid target — must NOT raise."""
        report = run_budget_checks({"tris": 100}, target="fbx",
                                   preset="ue_realtime")
        assert isinstance(report, BudgetReport), (
            f"'fbx' target must return BudgetReport, got {type(report)!r}"
        )

    def test_cx001_fbx_no_gc_check(self):
        """[regression guard] 'fbx' target must NOT include gc_piece_sequential."""
        report = run_budget_checks({"gc_pieces": [0, 1, 2]},
                                   target="fbx", preset="ue_realtime")
        check_ids = {c.id for c in report.checks}
        assert "gc_piece_sequential" not in check_ids, (
            f"'fbx' must not include gc_piece_sequential; got {check_ids}"
        )

    def test_cx001_fbx_no_vat_check(self):
        """[regression guard] 'fbx' target must NOT include vat_textures."""
        report = run_budget_checks({}, target="fbx", preset="ue_realtime")
        check_ids = {c.id for c in report.checks}
        assert "vat_textures" not in check_ids, (
            f"'fbx' must not include vat_textures; got {check_ids}"
        )

    def test_cx001_niagara_target_returns_budget_report(self):
        """[regression guard] 'niagara' is a valid target — must NOT raise."""
        report = run_budget_checks({"tris": 100}, target="niagara",
                                   preset="ue_realtime")
        assert isinstance(report, BudgetReport), (
            f"'niagara' target must return BudgetReport, got {type(report)!r}"
        )

    def test_cx001_niagara_no_gc_check(self):
        """[regression guard] 'niagara' target must NOT include gc_piece_sequential."""
        report = run_budget_checks({"gc_pieces": [0, 1]},
                                   target="niagara", preset="ue_realtime")
        check_ids = {c.id for c in report.checks}
        assert "gc_piece_sequential" not in check_ids, (
            f"'niagara' must not include gc_piece_sequential; got {check_ids}"
        )

    def test_cx001_niagara_no_vat_check(self):
        """[regression guard] 'niagara' target must NOT include vat_textures."""
        report = run_budget_checks({}, target="niagara", preset="ue_realtime")
        check_ids = {c.id for c in report.checks}
        assert "vat_textures" not in check_ids, (
            f"'niagara' must not include vat_textures; got {check_ids}"
        )

    # -----------------------------------------------------------------------
    # CX-002: negative gc piece IDs → fail
    # -----------------------------------------------------------------------

    def test_cx002_negative_id_in_two_element_list_fails(self):
        """[RED] check_gc_sequential([-1, 0]) must return status='fail'.

        Current impl: unique={-1,0}, max(unique)=0, required={0}, missing={} → 'pass'.
        Fixed impl: min(unique) < 0 → fail with the negative value in the message.
        """
        result = check_gc_sequential([-1, 0])
        assert result.status == "fail", (
            f"check_gc_sequential([-1, 0]) must fail (negative ID -1); "
            f"got status={result.status!r}"
        )

    def test_cx002_negative_id_message_names_negative_value(self):
        """[RED] The fail message for [-1, 0] must reference the negative value (-1)."""
        result = check_gc_sequential([-1, 0])
        combined = " ".join(filter(None, [result.detail, result.msg]))
        assert "-1" in combined, (
            f"Fail message for [-1, 0] must name '-1'; "
            f"detail={result.detail!r}, msg={result.msg!r}"
        )

    def test_cx002_negative_id_in_three_element_list_fails(self):
        """[RED] check_gc_sequential([-2, 0, 1]) must return status='fail'.

        Current impl: unique={-2,0,1}, max=1, required={0,1}, missing={} → 'pass'.
        """
        result = check_gc_sequential([-2, 0, 1])
        assert result.status == "fail", (
            f"check_gc_sequential([-2, 0, 1]) must fail (negative ID -2); "
            f"got status={result.status!r}"
        )

    def test_cx002_negative_id_three_element_message_names_negative_value(self):
        """[RED] The fail message for [-2, 0, 1] must reference the negative value (-2)."""
        result = check_gc_sequential([-2, 0, 1])
        combined = " ".join(filter(None, [result.detail, result.msg]))
        assert "-2" in combined, (
            f"Fail message for [-2, 0, 1] must name '-2'; "
            f"detail={result.detail!r}, msg={result.msg!r}"
        )

    def test_cx002_all_positive_contiguous_still_passes(self):
        """[regression guard] check_gc_sequential([0, 1, 2]) must still pass.

        The negative-ID guard must not affect valid contiguous sequences.
        """
        result = check_gc_sequential([0, 1, 2])
        assert result.status == "pass", (
            f"[0, 1, 2] must still pass after the negative-ID fix; "
            f"got status={result.status!r}"
        )

    def test_cx002_regression_guard_contiguous_four(self):
        """[regression guard] check_gc_sequential([0, 1, 2, 3]) still passes."""
        result = check_gc_sequential([0, 1, 2, 3])
        assert result.status == "pass", (
            f"[0, 1, 2, 3] must still pass; got status={result.status!r}"
        )
