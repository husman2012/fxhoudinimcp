"""
Tests for build_compare — PP12-114 PR-6a (RED phase).

TDD phase: RED — this file is authored BEFORE hou-dev implements build_compare.

Three sources of redness:
  1. build_compare does NOT exist yet in render_readback_model.py
     → ImportError / AttributeError on import.
  2. RR-1 fix has NOT been applied to compute_plane_delta:
     non-finite samples excluded from abs-error NUMERATOR but denominator still
     uses the FULL channel length n (not the finite-sample count).
     Symptom: compute_plane_delta([[0.0, nan]], [[1.0, 1.0]], "C").mae == 0.5
     Expected after fix: mae == 1.0 (one finite sample, not two).
  3. RR-2 fix has NOT been applied to compute_plane_delta:
     peak_value <= 0 does NOT raise ValueError yet.

Contract (§4.2 locked shape for build_compare):
  build_compare(
      aovs_a: list[str],
      aovs_b: list[str],
      channels_a: dict[str, list[list[float]]],
      channels_b: dict[str, list[list[float]]],
      planes: list[str] | None = None,
      metric: str = "stats",
      peak_value: float = 1.0,
  ) -> dict

  Return shape == CompareReport.to_dict():
    {
      "aovs_only_in_a": list[str],
      "aovs_only_in_b": list[str],
      "aovs_common":    list[str],
      "per_plane": [
        {
          "plane":         str,
          "mean_delta":    list[float],
          "max_abs_delta": list[float],
          "mae":           float,
          "psnr":          float | None,   # None when non-finite
          "moved":         bool,
        },
        ...
      ],
      "verdict": str,
    }

  build_compare REUSES aov_presence_diff, compute_plane_delta, build_verdict,
  CompareReport.to_dict() — it does NOT re-implement them.

Cross-references:
  - Plan pp12-114f-model lockedFieldContract (BINDING)
  - tdd-with-agents.md §4: hou-test writes red; hou-dev turns green
  - CL-015: pure-logic module, no hou/Qt/pxr
  - Mirror-test guard: assert PUBLIC results / values, NEVER internal call order
"""

from __future__ import annotations

import json
import math
import sys
import os

# ---------------------------------------------------------------------------
# Path bootstrap — allow running standalone and via pytest.
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_PKG_ROOT))

import pytest

# ---------------------------------------------------------------------------
# Imports from the module under test.
#
# compute_plane_delta is imported at module level (it exists — RR-1/RR-2 tests
# can be collected and produce their own RED failures).
#
# build_compare does NOT exist yet; a graceful stub is installed so that
# collection succeeds.  Tests that call the stub fail with a clear ImportError
# that describes the missing symbol — this is the expected RED failure.
# ---------------------------------------------------------------------------
from fxhoudinimcp.render_readback_model import compute_plane_delta

try:
    from fxhoudinimcp.render_readback_model import build_compare  # type: ignore[attr-defined]
except ImportError:
    def build_compare(*args, **kwargs):  # type: ignore[misc]
        """Stub: build_compare does not exist yet (RED phase).

        Raises ImportError with a descriptive message so every test that calls
        this stub fails with a clear RED failure.
        hou-dev must implement build_compare in render_readback_model.py.
        """
        raise ImportError(
            "build_compare is not yet implemented in render_readback_model.py — "
            "this is the expected RED failure.  hou-dev must add it."
        )


# ===========================================================================
# Section A — RR-1: non-finite denominator bug in compute_plane_delta
#
# The RR-1 fix: non-finite (NaN/Inf) samples must be excluded from BOTH the
# numerator AND the denominator of every average in compute_plane_delta.
#
# Key worked example from the locked plan contract:
#   a = [[0.0, nan]]
#   b = [[1.0, 1.0]]
#   plane = "C"
#
# Before RR-1 fix (current code):
#   numerator  = |0.0 - 1.0| = 1.0  (nan pixel skipped — numerator correct)
#   denominator = n = 2              (full length — BUG: nan pixel counted)
#   mae = 1.0 / 2 = 0.5
#
# After RR-1 fix (what hou-dev must implement):
#   numerator   = |0.0 - 1.0| = 1.0  (finite sample)
#   denominator = 1                   (only the ONE finite sample)
#   mae = 1.0 / 1 = 1.0
#
# These tests will be RED until hou-dev applies the RR-1 fix.
# ===========================================================================


class TestRR1NonFiniteDenominator:
    """RR-1: MAE denominator must use finite-sample count, not full buffer length."""

    def test_rr1_nan_in_a_mae_uses_finite_denominator(self):
        """Worked example: a=[[0.0, nan]], b=[[1.0, 1.0]] → mae == 1.0 (NOT 0.5).

        Current code (pre-fix) returns 0.5 because the denominator is the
        full channel length 2 instead of the 1 finite-sample count.
        After RR-1 fix the denominator must count only finite samples.
        """
        a = [[0.0, float("nan")]]
        b = [[1.0, 1.0]]
        delta = compute_plane_delta(a, b, "C")
        assert abs(delta.mae - 1.0) < 1e-9, (
            f"RR-1: nan in a → mae must be 1.0 (one finite sample, |0-1|=1.0 / 1), "
            f"got {delta.mae}. "
            "Current code gives 0.5 because denominator uses full length 2 not finite count 1."
        )

    def test_rr1_nan_in_a_moved_true(self):
        """With one finite differing sample (nan excluded), moved must be True."""
        a = [[0.0, float("nan")]]
        b = [[1.0, 1.0]]
        delta = compute_plane_delta(a, b, "C")
        assert delta.moved is True, (
            "RR-1: one finite differing sample → moved must be True"
        )

    def test_rr1_nan_in_b_mae_uses_finite_denominator(self):
        """Symmetry: nan in b — denominator must count only the finite b samples."""
        # a=[[0.0, 0.0]], b=[[1.0, nan]] → one finite pair (0.0, 1.0), mae=1.0
        a = [[0.0, 0.0]]
        b = [[1.0, float("nan")]]
        delta = compute_plane_delta(a, b, "C")
        assert abs(delta.mae - 1.0) < 1e-9, (
            f"RR-1: nan in b → mae must be 1.0, got {delta.mae}"
        )

    def test_rr1_inf_in_a_mae_uses_finite_denominator(self):
        """Inf (not NaN) in a must also be excluded from the denominator."""
        # a=[[0.0, inf]], b=[[0.5, 0.5]] → one finite pair (0.0, 0.5), mae=0.5
        a = [[0.0, float("inf")]]
        b = [[0.5, 0.5]]
        delta = compute_plane_delta(a, b, "C")
        assert abs(delta.mae - 0.5) < 1e-9, (
            f"RR-1: inf in a → mae must be 0.5 (finite sample only), got {delta.mae}"
        )

    def test_rr1_multiple_nans_only_finite_samples_averaged(self):
        """3 nan + 1 finite sample: denominator must be 1 (not 4)."""
        # Only last pixel (0.0 vs 1.0) is finite → mae = 1.0 / 1 = 1.0
        a = [[float("nan"), float("nan"), float("nan"), 0.0]]
        b = [[float("nan"), float("nan"), float("nan"), 1.0]]
        delta = compute_plane_delta(a, b, "C")
        assert abs(delta.mae - 1.0) < 1e-9, (
            f"RR-1: 3 nan + 1 finite diff sample → mae must be 1.0, got {delta.mae}"
        )

    def test_rr1_all_nan_mae_is_zero(self):
        """All NaN pixels → no finite samples → mae must be 0.0 (empty sum / 0 = 0)."""
        a = [[float("nan"), float("nan")]]
        b = [[float("nan"), float("nan")]]
        delta = compute_plane_delta(a, b, "C")
        # No finite sample to compute a difference from → mae == 0.0
        assert delta.mae == 0.0, (
            f"RR-1: all nan → mae must be 0.0 (no finite samples), got {delta.mae}"
        )

    def test_rr1_multichannel_nan_denominator_per_channel_correct(self):
        """Multi-channel: each channel accumulates independently into total_count.

        ch0: [[0.0, nan]] vs [[1.0, 1.0]] → 1 finite pair, abs_err=1.0
        ch1: [[0.5, 0.5]] vs [[0.5, 0.5]] → 2 finite pairs, abs_err=0.0
        total_abs_err = 1.0, total_finite_count = 3
        mae = 1.0 / 3 ≈ 0.3333...
        """
        a = [[0.0, float("nan")], [0.5, 0.5]]
        b = [[1.0, 1.0],          [0.5, 0.5]]
        delta = compute_plane_delta(a, b, "C")
        expected_mae = 1.0 / 3.0
        assert abs(delta.mae - expected_mae) < 1e-9, (
            f"RR-1 multichannel: mae must be {expected_mae:.6f} "
            f"(1 finite diff + 2 identical finite = 3 finite total), got {delta.mae}"
        )


# ===========================================================================
# Section A2 — RR-1 psnr-overflow regression: FINITE-but-huge values
# ===========================================================================


class TestRR1PsnrOverflowRegression:
    """RR-1 psnr-overflow: finite-but-huge values still produce psnr=0.0 sentinel."""

    def test_rr1_psnr_overflow_finite_huge_value(self):
        """Rev2: compute_plane_delta([[0.0]], [[1e200]], 'C') → psnr sentinel 0.0 (NOT None).

        1e200 is math.isfinite() == True → NOT excluded by RR-1's non-finite filter.
        abs_err = 1e200; mse = (1e200)^2 = inf (float overflow of a FINITE value).
        mse==inf from finite-but-huge → psnr 0.0 sentinel (finite, json-safe).
        psnr==None would incorrectly signal 'no finite samples'; the correct sentinel
        for overflow from finite values is 0.0.
        """
        a = [[0.0]]
        b = [[1e200]]
        delta = compute_plane_delta(a, b, "C")
        assert delta.moved is True, (
            f"1e200 overflow → moved must be True (finite pixel differs), got {delta.moved!r}"
        )
        d = delta.to_dict()
        assert d["psnr"] == 0.0, (
            f"mse=inf from finite-but-huge value → to_dict psnr must be 0.0 (overflow sentinel), "
            f"got {d['psnr']!r}. "
            "None would indicate no-finite-samples; 0.0 indicates overflow from finite samples."
        )
        import json as _json
        _json.dumps(d)  # must be json-serializable


# ===========================================================================
# Section B — RR-2: peak_value <= 0 must raise ValueError
#
# The RR-2 fix: compute_plane_delta must validate peak_value > 0 and raise
# ValueError('compute_plane_delta: peak_value must be > 0, got <v>') otherwise.
#
# Current code (pre-fix) silently uses peak_value=0 which causes a
# ZeroDivisionError or math domain error when computing PSNR, or silently
# produces wrong results.
#
# After RR-2 fix (what hou-dev must implement):
#   peak_value <= 0 → raises ValueError immediately, before any computation.
#
# These tests will be RED until hou-dev applies the RR-2 fix.
# ===========================================================================


class TestRR2PeakValueValidation:
    """RR-2: peak_value <= 0 must raise ValueError with a descriptive message."""

    def test_rr2_peak_value_zero_raises_valueerror(self):
        """peak_value=0 must raise ValueError (not ZeroDivisionError or silent wrong result)."""
        a = [[0.0, 0.5]]
        b = [[0.5, 1.0]]
        with pytest.raises(ValueError, match="peak_value"):
            compute_plane_delta(a, b, "C", peak_value=0)

    def test_rr2_peak_value_negative_raises_valueerror(self):
        """peak_value=-1.0 must raise ValueError."""
        a = [[0.0, 0.5]]
        b = [[0.5, 1.0]]
        with pytest.raises(ValueError, match="peak_value"):
            compute_plane_delta(a, b, "C", peak_value=-1.0)

    def test_rr2_peak_value_negative_small_raises_valueerror(self):
        """Any negative peak_value, including very small negatives, must raise."""
        a = [[0.0]]
        b = [[1.0]]
        with pytest.raises(ValueError):
            compute_plane_delta(a, b, "C", peak_value=-0.0001)

    def test_rr2_error_message_mentions_peak_value(self):
        """The ValueError message must mention 'peak_value'."""
        a = [[0.0, 0.5]]
        b = [[0.5, 1.0]]
        with pytest.raises(ValueError) as exc_info:
            compute_plane_delta(a, b, "C", peak_value=0)
        assert "peak_value" in str(exc_info.value), (
            f"ValueError message must mention 'peak_value', got: {exc_info.value!r}"
        )

    def test_rr2_error_message_includes_bad_value(self):
        """The ValueError message must include the offending value."""
        a = [[0.0]]
        b = [[1.0]]
        with pytest.raises(ValueError) as exc_info:
            compute_plane_delta(a, b, "C", peak_value=-5.0)
        msg = str(exc_info.value)
        # The message should contain the bad value
        assert "-5" in msg or "got" in msg, (
            f"ValueError message should include the bad value, got: {msg!r}"
        )

    def test_rr2_positive_peak_value_does_not_raise(self):
        """Positive peak_value must NOT raise (regression guard)."""
        a = [[0.0, 0.5]]
        b = [[0.5, 1.0]]
        # Should complete without exception
        delta = compute_plane_delta(a, b, "C", peak_value=1.0)
        assert isinstance(delta.mae, float)

    def test_rr2_large_positive_peak_value_does_not_raise(self):
        """Large positive peak_value (e.g. 1000.0 for HDR/depth AOVs) must NOT raise."""
        a = [[0.0, 50.0]]
        b = [[50.0, 100.0]]
        delta = compute_plane_delta(a, b, "C", peak_value=1000.0)
        assert isinstance(delta.mae, float)


# ===========================================================================
# Section C — build_compare: return shape (§4.2)
#
# build_compare does NOT exist yet → these tests fail at import above.
# When hou-dev implements it, these tests verify the locked §4.2 return shape.
# ===========================================================================


class TestBuildCompareReturnShape:
    """build_compare return shape == CompareReport.to_dict() § 4.2."""

    def _minimal_channels(self, value: float = 0.5, size: int = 4):
        """One-channel buffer, `size` pixels, constant value."""
        return [[value] * size]

    def test_returns_dict(self):
        """build_compare must return a dict."""
        aovs_a = ["C"]
        aovs_b = ["C"]
        channels_a = {"C": self._minimal_channels(0.5)}
        channels_b = {"C": self._minimal_channels(0.5)}
        result = build_compare(aovs_a, aovs_b, channels_a, channels_b)
        assert isinstance(result, dict), (
            f"build_compare must return dict, got {type(result)!r}"
        )

    def test_return_shape_has_aovs_only_in_a(self):
        """§4.2: result must contain 'aovs_only_in_a' key."""
        result = build_compare(["C"], ["C"], {"C": self._minimal_channels()}, {"C": self._minimal_channels()})
        assert "aovs_only_in_a" in result, "Missing key 'aovs_only_in_a'"

    def test_return_shape_has_aovs_only_in_b(self):
        """§4.2: result must contain 'aovs_only_in_b' key."""
        result = build_compare(["C"], ["C"], {"C": self._minimal_channels()}, {"C": self._minimal_channels()})
        assert "aovs_only_in_b" in result, "Missing key 'aovs_only_in_b'"

    def test_return_shape_has_aovs_common(self):
        """§4.2: result must contain 'aovs_common' key."""
        result = build_compare(["C"], ["C"], {"C": self._minimal_channels()}, {"C": self._minimal_channels()})
        assert "aovs_common" in result, "Missing key 'aovs_common'"

    def test_return_shape_has_per_plane(self):
        """§4.2: result must contain 'per_plane' key as a list."""
        result = build_compare(["C"], ["C"], {"C": self._minimal_channels()}, {"C": self._minimal_channels()})
        assert "per_plane" in result, "Missing key 'per_plane'"
        assert isinstance(result["per_plane"], list), (
            f"'per_plane' must be a list, got {type(result['per_plane'])!r}"
        )

    def test_return_shape_has_verdict(self):
        """§4.2: result must contain 'verdict' key as a str."""
        result = build_compare(["C"], ["C"], {"C": self._minimal_channels()}, {"C": self._minimal_channels()})
        assert "verdict" in result, "Missing key 'verdict'"
        assert isinstance(result["verdict"], str), (
            f"'verdict' must be str, got {type(result['verdict'])!r}"
        )

    def test_per_plane_items_are_dicts(self):
        """§4.2: each item in per_plane must be a dict (PlaneDelta.to_dict())."""
        result = build_compare(["C"], ["C"], {"C": self._minimal_channels()}, {"C": self._minimal_channels()})
        for item in result["per_plane"]:
            assert isinstance(item, dict), (
                f"per_plane items must be dicts, got {type(item)!r}"
            )

    def test_per_plane_item_has_plane_key(self):
        """§4.2: each per_plane dict must have 'plane', 'mae', 'moved', 'psnr' keys."""
        result = build_compare(["C"], ["C"], {"C": self._minimal_channels()}, {"C": self._minimal_channels()})
        assert len(result["per_plane"]) > 0, "Expected at least one per_plane entry"
        item = result["per_plane"][0]
        for key in ("plane", "mean_delta", "max_abs_delta", "mae", "psnr", "moved"):
            assert key in item, f"per_plane[0] missing key '{key}'"

    def test_is_json_serializable(self):
        """§4.2: build_compare result must be json.dumps-able (psnr=inf → None)."""
        result = build_compare(["C"], ["C"], {"C": self._minimal_channels()}, {"C": self._minimal_channels()})
        json.dumps(result)  # must not raise


# ===========================================================================
# Section D — build_compare: AOV presence diff propagates correctly
# ===========================================================================


class TestBuildCompareAovPresence:
    """build_compare must propagate aov_presence_diff into the §4.2 envelope."""

    def _ch(self, value: float = 0.5, size: int = 4):
        return [[value] * size]

    def test_aov_only_in_a_recorded(self):
        """An AOV present in aovs_a but absent in aovs_b appears in aovs_only_in_a."""
        aovs_a = ["C", "N"]
        aovs_b = ["C"]
        channels_a = {"C": self._ch(), "N": self._ch()}
        channels_b = {"C": self._ch()}
        result = build_compare(aovs_a, aovs_b, channels_a, channels_b)
        assert "N" in result["aovs_only_in_a"], (
            f"'N' must be in aovs_only_in_a, got: {result['aovs_only_in_a']}"
        )
        assert "N" not in result["aovs_only_in_b"]
        assert "N" not in result["aovs_common"]

    def test_aov_only_in_b_recorded(self):
        """An AOV present in aovs_b but absent in aovs_a appears in aovs_only_in_b."""
        aovs_a = ["C"]
        aovs_b = ["C", "Pz"]
        channels_a = {"C": self._ch()}
        channels_b = {"C": self._ch(), "Pz": self._ch()}
        result = build_compare(aovs_a, aovs_b, channels_a, channels_b)
        assert "Pz" in result["aovs_only_in_b"], (
            f"'Pz' must be in aovs_only_in_b, got: {result['aovs_only_in_b']}"
        )

    def test_aovs_common_correct(self):
        """AOVs present in both appear in aovs_common."""
        aovs_a = ["C", "N", "Pz"]
        aovs_b = ["C", "N"]
        channels_a = {"C": self._ch(), "N": self._ch(), "Pz": self._ch()}
        channels_b = {"C": self._ch(), "N": self._ch()}
        result = build_compare(aovs_a, aovs_b, channels_a, channels_b)
        assert set(result["aovs_common"]) == {"C", "N"}, (
            f"aovs_common must be {{'C', 'N'}}, got {result['aovs_common']}"
        )

    def test_selected_plane_absent_from_channels_a_raises_valueerror(self):
        """Rev2: a SELECTED plane absent from channels_a MUST raise ValueError (not skip).

        'N' is common (in both aovs) and present in channels_b — but missing from
        channels_a.  Because 'N' IS selected (default planes=None → all common),
        build_compare must raise ValueError naming the missing plane.
        """
        aovs_a = ["C", "N"]
        aovs_b = ["C", "N"]
        channels_a = {"C": self._ch()}              # "N" missing from channels_a
        channels_b = {"C": self._ch(), "N": self._ch()}
        with pytest.raises(ValueError):
            build_compare(aovs_a, aovs_b, channels_a, channels_b)

    def test_selected_plane_absent_from_channels_b_raises_valueerror(self):
        """Rev2: a SELECTED plane absent from channels_b MUST raise ValueError (not skip).

        'N' is common (in both aovs) and present in channels_a — but missing from
        channels_b.  Because 'N' IS selected, build_compare must raise ValueError.
        """
        aovs_a = ["C", "N"]
        aovs_b = ["C", "N"]
        channels_a = {"C": self._ch(), "N": self._ch()}
        channels_b = {"C": self._ch()}              # "N" missing from channels_b
        with pytest.raises(ValueError):
            build_compare(aovs_a, aovs_b, channels_a, channels_b)

    def test_missing_channels_single_selected_plane_raises(self):
        """Rev2 pinned: planes=['C'], channels_b missing 'C' → ValueError.

        build_compare(['C'], ['C'], {'C': buf}, {}, planes=['C']) → raises ValueError.
        """
        with pytest.raises(ValueError):
            build_compare(["C"], ["C"], {"C": self._ch()}, {}, planes=["C"])

    def test_common_but_not_selected_plane_missing_channels_no_raise(self):
        """Rev2 NEGATIVE: a plane in common but NOT in planes= may have missing channels — no raise.

        planes=['C']: only 'C' is selected.  'N' is in both aov lists (common) but
        NOT selected — it is NOT required to be in channels_a or channels_b.
        build_compare must NOT raise even though 'N' is absent from channels_a.
        """
        aovs_a = ["C", "N"]
        aovs_b = ["C", "N"]
        channels_a = {"C": self._ch()}              # 'N' absent — but 'N' not selected
        channels_b = {"C": self._ch(), "N": self._ch()}
        # planes=['C'] → only 'C' is selected; 'N' is common but not requested
        result = build_compare(aovs_a, aovs_b, channels_a, channels_b, planes=["C"])
        plane_names = {item["plane"] for item in result["per_plane"]}
        assert plane_names == {"C"}, (
            f"planes=['C'] with 'N' not selected must not raise and must produce "
            f"only 'C' in per_plane, got {plane_names}"
        )

    def test_all_common_aovs_in_per_plane_by_default(self):
        """When planes=None, all common AOVs are included in per_plane."""
        aovs_a = ["C", "N"]
        aovs_b = ["C", "N"]
        channels_a = {"C": self._ch(), "N": self._ch()}
        channels_b = {"C": self._ch(), "N": self._ch()}
        result = build_compare(aovs_a, aovs_b, channels_a, channels_b)
        plane_names = {item["plane"] for item in result["per_plane"]}
        assert plane_names == {"C", "N"}, (
            f"planes=None must include all common AOVs, got {plane_names}"
        )


# ===========================================================================
# Section E — build_compare: planes selection
# ===========================================================================


class TestBuildComparePlanesSelection:
    """build_compare planes= parameter selects which planes to compare."""

    def _ch(self, value: float = 0.5, size: int = 4):
        return [[value] * size]

    def test_planes_none_includes_all_common(self):
        """planes=None (default) → all common AOVs in per_plane."""
        aovs = ["C", "N", "Pz"]
        channels = {"C": self._ch(), "N": self._ch(), "Pz": self._ch()}
        result = build_compare(aovs, aovs, channels, channels, planes=None)
        plane_names = {item["plane"] for item in result["per_plane"]}
        assert plane_names == {"C", "N", "Pz"}

    def test_planes_list_filters_to_requested(self):
        """planes=['C'] → only 'C' in per_plane, not 'N' or 'Pz'."""
        aovs = ["C", "N", "Pz"]
        channels = {"C": self._ch(), "N": self._ch(), "Pz": self._ch()}
        result = build_compare(aovs, aovs, channels, channels, planes=["C"])
        plane_names = {item["plane"] for item in result["per_plane"]}
        assert plane_names == {"C"}, (
            f"planes=['C'] must restrict per_plane to {{'C'}}, got {plane_names}"
        )

    def test_planes_list_preserves_requested_order(self):
        """planes order is preserved in the per_plane output."""
        aovs = ["C", "N", "Pz"]
        channels = {"C": self._ch(), "N": self._ch(), "Pz": self._ch()}
        result = build_compare(aovs, aovs, channels, channels, planes=["Pz", "C"])
        plane_names = [item["plane"] for item in result["per_plane"]]
        # The order in the result must match the requested planes= order
        assert plane_names == ["Pz", "C"], (
            f"per_plane order must match planes= parameter, got {plane_names}"
        )

    def test_planes_list_not_in_common_skipped_no_raise(self):
        """A plane in planes= that is not in common is silently skipped (no raise)."""
        aovs_a = ["C"]
        aovs_b = ["C"]
        channels_a = {"C": self._ch()}
        channels_b = {"C": self._ch()}
        # Request "N" which is NOT in either aov list
        result = build_compare(aovs_a, aovs_b, channels_a, channels_b, planes=["C", "N"])
        plane_names = {item["plane"] for item in result["per_plane"]}
        # "N" not in common → skipped; "C" still present
        assert "N" not in plane_names, (
            "Plane not in common must be silently skipped, not raise"
        )
        assert "C" in plane_names


# ===========================================================================
# Section F — build_compare: metric validation
# ===========================================================================


class TestBuildCompareMetricValidation:
    """build_compare metric= parameter validation."""

    def _ch(self, value: float = 0.5, size: int = 4):
        return [[value] * size]

    def test_metric_stats_accepted(self):
        """metric='stats' (default) is accepted without error."""
        result = build_compare(["C"], ["C"], {"C": self._ch()}, {"C": self._ch()}, metric="stats")
        assert "per_plane" in result

    def test_metric_mae_accepted(self):
        """metric='mae' is accepted without error."""
        result = build_compare(["C"], ["C"], {"C": self._ch()}, {"C": self._ch()}, metric="mae")
        assert "per_plane" in result

    def test_metric_psnr_accepted(self):
        """metric='psnr' is accepted without error."""
        result = build_compare(["C"], ["C"], {"C": self._ch()}, {"C": self._ch()}, metric="psnr")
        assert "per_plane" in result

    def test_unknown_metric_raises_valueerror(self):
        """An unknown metric value must raise ValueError."""
        with pytest.raises(ValueError):
            build_compare(["C"], ["C"], {"C": self._ch()}, {"C": self._ch()}, metric="unknown_metric")

    def test_empty_string_metric_raises_valueerror(self):
        """An empty string metric must raise ValueError."""
        with pytest.raises(ValueError):
            build_compare(["C"], ["C"], {"C": self._ch()}, {"C": self._ch()}, metric="")

    def test_metric_case_sensitive_uppercase_raises(self):
        """Metric matching is case-sensitive; 'STATS' (uppercase) must raise ValueError."""
        with pytest.raises(ValueError):
            build_compare(["C"], ["C"], {"C": self._ch()}, {"C": self._ch()}, metric="STATS")

    def test_v1_full_per_plane_returned_for_all_metrics(self):
        """v1 contract: full per_plane dict is returned regardless of metric value."""
        aovs = ["C"]
        channels = {"C": self._ch()}
        for metric in ("stats", "mae", "psnr"):
            result = build_compare(aovs, aovs, channels, channels, metric=metric)
            assert len(result["per_plane"]) == 1, (
                f"metric={metric!r}: per_plane must have 1 entry for 1 common AOV"
            )
            item = result["per_plane"][0]
            # v1 always returns the full set of per_plane fields
            for key in ("plane", "mae", "psnr", "moved", "mean_delta", "max_abs_delta"):
                assert key in item, (
                    f"metric={metric!r}: per_plane[0] missing key '{key}'"
                )


# ===========================================================================
# Section G — build_compare: mae / moved / psnr values
# ===========================================================================


class TestBuildCompareDeltaValues:
    """build_compare per_plane values must match compute_plane_delta results."""

    def _ch(self, value: float = 0.5, size: int = 4):
        return [[value] * size]

    def test_identical_renders_mae_zero(self):
        """Identical renders → per_plane[0].mae == 0.0."""
        aovs = ["C"]
        channels = {"C": self._ch(0.5)}
        result = build_compare(aovs, aovs, channels, channels)
        item = result["per_plane"][0]
        assert item["mae"] == 0.0, (
            f"Identical renders → mae must be 0.0, got {item['mae']}"
        )

    def test_identical_renders_moved_false(self):
        """Identical renders → per_plane[0].moved == False."""
        aovs = ["C"]
        channels = {"C": self._ch(0.5)}
        result = build_compare(aovs, aovs, channels, channels)
        item = result["per_plane"][0]
        assert item["moved"] is False, (
            f"Identical renders → moved must be False, got {item['moved']}"
        )

    def test_identical_renders_psnr_is_none(self):
        """Identical renders → per_plane[0].psnr is None (json-safe sentinel for inf)."""
        aovs = ["C"]
        channels = {"C": self._ch(0.5)}
        result = build_compare(aovs, aovs, channels, channels)
        item = result["per_plane"][0]
        assert item["psnr"] is None, (
            f"Identical renders → psnr must be None (JSON-safe inf), got {item['psnr']!r}"
        )

    def test_changed_renders_mae_nonzero(self):
        """Changed renders → per_plane[0].mae > 0.0."""
        aovs = ["C"]
        channels_a = {"C": [[0.0, 0.0, 0.0, 0.0]]}
        channels_b = {"C": [[1.0, 1.0, 1.0, 1.0]]}
        result = build_compare(aovs, aovs, channels_a, channels_b)
        item = result["per_plane"][0]
        assert item["mae"] > 0.0, (
            f"Changed renders → mae must be > 0.0, got {item['mae']}"
        )

    def test_changed_renders_moved_true(self):
        """Changed renders → per_plane[0].moved == True."""
        aovs = ["C"]
        channels_a = {"C": [[0.0, 0.0, 0.0, 0.0]]}
        channels_b = {"C": [[1.0, 1.0, 1.0, 1.0]]}
        result = build_compare(aovs, aovs, channels_a, channels_b)
        item = result["per_plane"][0]
        assert item["moved"] is True, (
            f"Changed renders → moved must be True, got {item['moved']}"
        )

    def test_known_mae_value_propagated(self):
        """Known MAE: all-zero vs all-one single channel, 4 pixels → mae == 1.0."""
        aovs = ["C"]
        channels_a = {"C": [[0.0, 0.0, 0.0, 0.0]]}
        channels_b = {"C": [[1.0, 1.0, 1.0, 1.0]]}
        result = build_compare(aovs, aovs, channels_a, channels_b)
        item = result["per_plane"][0]
        assert abs(item["mae"] - 1.0) < 1e-9, (
            f"all-zero vs all-one → mae must be 1.0, got {item['mae']}"
        )

    def test_result_is_json_serializable(self):
        """The full result must be json.dumps-able (psnr=inf → None already handled)."""
        aovs = ["C"]
        channels_a = {"C": [[0.0, 0.0, 0.0, 0.0]]}
        channels_b = {"C": [[1.0, 1.0, 1.0, 1.0]]}
        result = build_compare(aovs, aovs, channels_a, channels_b)
        json.dumps(result)  # must not raise

    def test_all_nonfinite_buffer_sentinel(self):
        """Rev2: plane with entirely non-finite buffers → moved=True, mae=0.0, psnr=None.

        When NO finite samples exist, compute_plane_delta returns the 'no finite samples'
        sentinel: moved=True (cannot confirm unchanged), mae=0.0 (empty average),
        psnr=None (json-safe, not inf).
        This is DISTINCT from identical/unchanged renders (moved=False, psnr=None).
        """
        aovs = ["C"]
        channels_a = {"C": [[float("nan"), float("inf")]]}
        channels_b = {"C": [[float("nan"), float("inf")]]}
        result = build_compare(aovs, aovs, channels_a, channels_b)
        item = result["per_plane"][0]
        assert item["moved"] is True, (
            f"all-non-finite → moved must be True (no finite samples to confirm unchanged), "
            f"got moved={item['moved']!r}"
        )
        assert item["mae"] == 0.0, (
            f"all-non-finite → mae must be 0.0 (empty sum / 0 finite samples), got {item['mae']}"
        )
        assert item["psnr"] is None, (
            f"all-non-finite → psnr must be None (json-safe, no finite samples), got {item['psnr']!r}"
        )
        import json as _json
        _json.dumps(result)  # must not raise


# ===========================================================================
# Section H — build_compare: verdict is populated
# ===========================================================================


class TestBuildCompareVerdict:
    """build_compare must populate the 'verdict' field via build_verdict."""

    def _ch(self, value: float = 0.5, size: int = 4):
        return [[value] * size]

    def test_verdict_is_nonempty_string(self):
        """'verdict' must be a non-empty string."""
        result = build_compare(["C"], ["C"], {"C": self._ch()}, {"C": self._ch()})
        assert isinstance(result["verdict"], str)
        assert len(result["verdict"]) > 0

    def test_verdict_reflects_unchanged_planes(self):
        """Identical renders → verdict mentions 'no change' or 'unchanged'."""
        result = build_compare(["C"], ["C"], {"C": self._ch()}, {"C": self._ch()})
        verdict = result["verdict"]
        assert "no change" in verdict.lower() or "unchanged" in verdict.lower(), (
            f"Identical renders → verdict must mention no-change, got {verdict!r}"
        )

    def test_verdict_mentions_lost_aov(self):
        """When B is missing an AOV that A has, verdict must mention it."""
        aovs_a = ["C", "N"]
        aovs_b = ["C"]
        channels_a = {"C": self._ch(), "N": self._ch()}
        channels_b = {"C": self._ch()}
        result = build_compare(aovs_a, aovs_b, channels_a, channels_b)
        assert "N" in result["verdict"], (
            f"Lost AOV 'N' must appear in verdict, got {result['verdict']!r}"
        )

    def test_verdict_mentions_changed_plane(self):
        """When a common plane changes, verdict must mention the plane name."""
        aovs = ["C"]
        channels_a = {"C": [[0.0, 0.0, 0.0, 0.0]]}
        channels_b = {"C": [[1.0, 1.0, 1.0, 1.0]]}
        result = build_compare(aovs, aovs, channels_a, channels_b)
        assert "C" in result["verdict"], (
            f"Changed plane 'C' must appear in verdict, got {result['verdict']!r}"
        )
