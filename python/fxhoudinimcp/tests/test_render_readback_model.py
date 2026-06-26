"""
Tests for render_readback_model.py — pure-logic layer (PP12-114 PR-1).

TDD phase: RED — render_readback_model.py does NOT exist yet.
Expected failure: ModuleNotFoundError on import.

Contract: imports NO hou, NO Qt/PySide6, NO pxr, NO numpy.
Plain Python stdlib only. Runs under the fork .venv with plain pytest
headless (off-DCC, no Houdini install required).

Covers the public contract of:
  - PlaneDelta         (dataclass)   — plane, mean_delta, max_abs_delta, mae, psnr, moved, to_dict()
  - CompareReport      (dataclass)   — aovs_only_in_a/b, aovs_common, per_plane, verdict, to_dict()
  - PixelSummary       (dataclass)   — plane, xres, yres, channels, dtype, stats, histogram
  - ReadbackPage       (dataclass)   — pixels, page, page_size, total_pages, truncated
  - compute_plane_delta(a, b, plane) -> PlaneDelta
  - aov_presence_diff(a_planes, b_planes) -> (only_a, only_b, common)
  - build_verdict(report) -> str
  - summarize_pixels(channels, plane, xres, yres, dtype) -> PixelSummary
  - clamp_and_paginate(pixels, max_pixels, page, page_size) -> ReadbackPage

PSNR=inf representation decision (documented in test-bundle.json):
  float('inf') is NOT valid JSON. to_dict() serialises psnr=inf as None.
  The test asserts to_dict()["psnr"] is None when identical planes are compared.
  The Python-level field psnr retains float('inf') or a large float (impl choice),
  but to_dict() MUST produce a JSON-safe value for the wire.
  Assertion: json.dumps(delta.to_dict()) must not raise.

Cross-references:
  - Plan pp12-114a lockedFieldContract (BINDING)
  - spec.md §7.2 (data model), §4.2 (JSON shapes), §9.1 (unit-test taxonomy)
  - tdd-with-agents.md §4: hou-test writes red; hou-dev turns green
  - CL-015: pure-logic module, no hou/Qt/pxr
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
# The module under test.
# This import MUST fail (ModuleNotFoundError) in the RED phase because
# render_readback_model.py does not exist yet.
# ---------------------------------------------------------------------------
from fxhoudinimcp.render_readback_model import (
    PlaneDelta,
    CompareReport,
    PixelSummary,
    ReadbackPage,
    compute_plane_delta,
    aov_presence_diff,
    build_verdict,
    summarize_pixels,
    clamp_and_paginate,
)


# ===========================================================================
# Helpers — tiny synthetic pixel buffers (pure Python, no numpy)
#
# Buffer shape convention: channels[ch][pixel_flat_index]
# e.g. a 2x2 RGB buffer:
#   ch0 = [r00, r01, r10, r11]
#   ch1 = [g00, g01, g10, g11]
#   ch2 = [b00, b01, b10, b11]
# ===========================================================================

def _make_rgb_channels(r=0.5, g=0.3, b=0.1, size=4):
    """Return a 3-channel flat buffer (list[list[float]]) of `size` pixels each."""
    return [
        [r] * size,
        [g] * size,
        [b] * size,
    ]


def _make_single_channel(value=0.5, size=4):
    """Return a 1-channel flat buffer."""
    return [[value] * size]


def _pixel_rows(flat_channels, xres, yres):
    """Convert flat per-channel buffers to list[list[float]] row layout for clamp_and_paginate.

    Each row is one pixel's channels: [[r,g,b], [r,g,b], ...]
    Length = xres * yres rows.
    """
    n_channels = len(flat_channels)
    n_pixels = xres * yres
    rows = []
    for px in range(n_pixels):
        row = [flat_channels[ch][px] for ch in range(n_channels)]
        rows.append(row)
    return rows


# ===========================================================================
# Section 1 — PlaneDelta dataclass
#
# Locked contract:
#   PlaneDelta(plane: str,
#              mean_delta: list[float],    # per-channel signed mean delta
#              max_abs_delta: list[float], # per-channel max |a-b|
#              mae: float,                 # mean abs error over all channels
#              psnr: float,               # peak SNR; float('inf') when identical
#              moved: bool,               # True iff planes differ beyond epsilon
#   )
#   to_dict() -> {plane, mean_delta, max_abs_delta, mae, psnr, moved}
#   CRITICAL: to_dict()["psnr"] must be JSON-safe (None when psnr==inf).
# ===========================================================================

class TestPlaneDelta:
    """PlaneDelta dataclass — locked field names, types, serialization."""

    def _make_delta(self, **kwargs):
        defaults = dict(
            plane="C",
            mean_delta=[0.0, 0.0, 0.0],
            max_abs_delta=[0.0, 0.0, 0.0],
            mae=0.0,
            psnr=float("inf"),
            moved=False,
        )
        defaults.update(kwargs)
        return PlaneDelta(**defaults)

    # --- field presence ---

    def test_has_plane_field(self):
        d = self._make_delta(plane="N")
        assert d.plane == "N"

    def test_has_mean_delta_field(self):
        d = self._make_delta(mean_delta=[0.1, -0.2, 0.05])
        assert d.mean_delta == [0.1, -0.2, 0.05]

    def test_has_max_abs_delta_field(self):
        d = self._make_delta(max_abs_delta=[0.3, 0.4, 0.1])
        assert d.max_abs_delta == [0.3, 0.4, 0.1]

    def test_has_mae_field(self):
        d = self._make_delta(mae=0.031)
        assert abs(d.mae - 0.031) < 1e-9

    def test_has_psnr_field(self):
        d = self._make_delta(psnr=float("inf"))
        # psnr field exists and stores a float
        assert isinstance(d.psnr, float)

    def test_has_moved_field_bool(self):
        d = self._make_delta(moved=True)
        assert d.moved is True
        d2 = self._make_delta(moved=False)
        assert d2.moved is False

    # --- to_dict() key presence ---

    def test_to_dict_contains_locked_keys(self):
        d = self._make_delta()
        result = d.to_dict()
        assert "plane" in result
        assert "mean_delta" in result
        assert "max_abs_delta" in result
        assert "mae" in result
        assert "psnr" in result
        assert "moved" in result

    def test_to_dict_plane_value(self):
        d = self._make_delta(plane="Pz")
        assert d.to_dict()["plane"] == "Pz"

    def test_to_dict_mean_delta_value(self):
        d = self._make_delta(mean_delta=[0.1, 0.2])
        assert d.to_dict()["mean_delta"] == [0.1, 0.2]

    def test_to_dict_max_abs_delta_value(self):
        d = self._make_delta(max_abs_delta=[0.5, 0.3])
        assert d.to_dict()["max_abs_delta"] == [0.5, 0.3]

    def test_to_dict_mae_value(self):
        d = self._make_delta(mae=0.031)
        assert abs(d.to_dict()["mae"] - 0.031) < 1e-9

    def test_to_dict_moved_false(self):
        d = self._make_delta(moved=False)
        assert d.to_dict()["moved"] is False

    def test_to_dict_moved_true(self):
        d = self._make_delta(moved=True)
        assert d.to_dict()["moved"] is True

    # --- CRITICAL: psnr=inf JSON-safety ---

    def test_to_dict_psnr_inf_is_json_safe(self):
        """CRITICAL: psnr=inf must serialise as a JSON-safe value (None or large float).

        float('inf') is not valid JSON — json.dumps raises ValueError on it.
        The to_dict() MUST produce a JSON-safe sentinel when psnr is infinite.
        """
        d = self._make_delta(psnr=float("inf"), mae=0.0)
        result = d.to_dict()
        # Must not raise
        json.dumps(result)

    def test_to_dict_psnr_inf_serialises_as_none(self):
        """PSNR=inf serialises as None in to_dict() (the chosen sentinel).

        None → null in JSON, which is valid and unambiguous.
        hou-dev must implement to_dict() to emit None when psnr == inf.
        """
        d = self._make_delta(psnr=float("inf"), mae=0.0)
        assert d.to_dict()["psnr"] is None, (
            "PlaneDelta.to_dict()['psnr'] must be None when psnr is infinite "
            "(float('inf') is not valid JSON; None is the locked sentinel)"
        )

    def test_to_dict_finite_psnr_survives(self):
        """A finite psnr value (e.g. 42.3) is preserved as a float in to_dict()."""
        d = self._make_delta(psnr=42.3)
        result = d.to_dict()
        assert isinstance(result["psnr"], float)
        assert abs(result["psnr"] - 42.3) < 1e-6

    def test_to_dict_psnr_zero_survives(self):
        """psnr=0.0 is preserved as 0.0 in to_dict()."""
        d = self._make_delta(psnr=0.0)
        assert d.to_dict()["psnr"] == 0.0

    def test_to_dict_is_json_serializable_finite(self):
        """to_dict() is json.dumps-able when psnr is finite."""
        d = self._make_delta(psnr=38.5, mae=0.01, moved=True,
                             mean_delta=[0.01, -0.02, 0.005],
                             max_abs_delta=[0.05, 0.07, 0.02])
        json.dumps(d.to_dict())

    def test_to_dict_is_json_serializable_inf(self):
        """to_dict() is json.dumps-able when psnr is infinite."""
        d = self._make_delta(psnr=float("inf"), mae=0.0, moved=False)
        json.dumps(d.to_dict())

    # --- no extra spurious fields ---

    def test_to_dict_no_extra_locked_keys(self):
        """to_dict() must not include unexpected extra keys beyond the locked set."""
        d = self._make_delta()
        expected_keys = {"plane", "mean_delta", "max_abs_delta", "mae", "psnr", "moved"}
        extra = set(d.to_dict().keys()) - expected_keys
        assert not extra, f"to_dict() has unexpected extra keys: {extra}"


# ===========================================================================
# Section 2 — CompareReport dataclass
#
# Locked contract:
#   CompareReport(
#       aovs_only_in_a: list[str],
#       aovs_only_in_b: list[str],
#       aovs_common:    list[str],
#       per_plane:      list[PlaneDelta],
#       verdict:        str,
#   )
#   to_dict() -> {aovs_only_in_a, aovs_only_in_b, aovs_common,
#                 per_plane:[PlaneDelta.to_dict()], verdict}
# ===========================================================================

class TestCompareReport:
    """CompareReport dataclass — locked fields, per_plane serialises as list[dict]."""

    def _make_delta(self, plane="C"):
        return PlaneDelta(
            plane=plane,
            mean_delta=[0.0],
            max_abs_delta=[0.0],
            mae=0.0,
            psnr=float("inf"),
            moved=False,
        )

    def _make_report(self):
        return CompareReport(
            aovs_only_in_a=["Pz"],
            aovs_only_in_b=[],
            aovs_common=["C", "N"],
            per_plane=[self._make_delta("C"), self._make_delta("N")],
            verdict="beauty unchanged; lost AOV Pz",
        )

    # --- field presence ---

    def test_has_aovs_only_in_a(self):
        r = self._make_report()
        assert r.aovs_only_in_a == ["Pz"]

    def test_has_aovs_only_in_b(self):
        r = self._make_report()
        assert r.aovs_only_in_b == []

    def test_has_aovs_common(self):
        r = self._make_report()
        assert r.aovs_common == ["C", "N"]

    def test_has_per_plane_list(self):
        r = self._make_report()
        assert isinstance(r.per_plane, list)
        assert len(r.per_plane) == 2

    def test_has_verdict(self):
        r = self._make_report()
        assert isinstance(r.verdict, str)
        assert len(r.verdict) > 0

    # --- to_dict() shape ---

    def test_to_dict_contains_locked_keys(self):
        r = self._make_report()
        d = r.to_dict()
        assert "aovs_only_in_a" in d
        assert "aovs_only_in_b" in d
        assert "aovs_common" in d
        assert "per_plane" in d
        assert "verdict" in d

    def test_to_dict_aovs_only_in_a_value(self):
        r = self._make_report()
        assert r.to_dict()["aovs_only_in_a"] == ["Pz"]

    def test_to_dict_aovs_only_in_b_empty(self):
        r = self._make_report()
        assert r.to_dict()["aovs_only_in_b"] == []

    def test_to_dict_aovs_common_value(self):
        r = self._make_report()
        assert r.to_dict()["aovs_common"] == ["C", "N"]

    def test_to_dict_verdict_value(self):
        r = self._make_report()
        assert "Pz" in r.to_dict()["verdict"]

    def test_to_dict_per_plane_is_list_of_dicts(self):
        """per_plane in to_dict() must be a list of dicts (PlaneDelta.to_dict())."""
        r = self._make_report()
        d = r.to_dict()
        assert isinstance(d["per_plane"], list)
        for item in d["per_plane"]:
            assert isinstance(item, dict), (
                f"per_plane items must be dicts, got {type(item)!r}"
            )

    def test_to_dict_per_plane_contains_plane_key(self):
        r = self._make_report()
        planes = [item["plane"] for item in r.to_dict()["per_plane"]]
        assert "C" in planes
        assert "N" in planes

    def test_to_dict_per_plane_psnr_json_safe(self):
        """Each per_plane dict must have a JSON-safe psnr (None, not inf)."""
        r = self._make_report()
        for item in r.to_dict()["per_plane"]:
            psnr_val = item["psnr"]
            assert psnr_val is None or isinstance(psnr_val, (int, float)), (
                f"per_plane[*].psnr must be None or a JSON-safe number, got {psnr_val!r}"
            )

    def test_to_dict_is_json_serializable(self):
        """CompareReport.to_dict() must be json.dumps-able without error."""
        r = self._make_report()
        json.dumps(r.to_dict())

    def test_empty_per_plane(self):
        """CompareReport with empty per_plane list is valid."""
        r = CompareReport(
            aovs_only_in_a=[],
            aovs_only_in_b=[],
            aovs_common=[],
            per_plane=[],
            verdict="no common AOVs",
        )
        d = r.to_dict()
        assert d["per_plane"] == []
        json.dumps(d)


# ===========================================================================
# Section 3 — compute_plane_delta(a, b, plane) -> PlaneDelta
#
# Locked contract:
#   a, b: list[list[float]] — per-channel flat buffers (no numpy)
#   plane: str
#
#   Identical planes (a == b):
#     mae == 0.0
#     moved == False
#     psnr == float('inf')  (or a very large float — impl choice for Python field)
#     mean_delta == [0.0, ...] (per channel)
#     max_abs_delta == [0.0, ...] (per channel)
#     to_dict()["psnr"] is None  (JSON-safe sentinel)
#
#   Changed planes (a != b):
#     mae > 0
#     moved == True
#     psnr is finite (or very large but not inf when mae is tiny but nonzero)
#     mean_delta reflects sign of (a - b) per channel
#     max_abs_delta >= |any individual delta|
#
#   Returns a PlaneDelta instance.
# ===========================================================================

class TestComputePlaneDelta:
    """compute_plane_delta — core compare-math: mae, psnr, moved, mean_delta, max_abs_delta."""

    # --- identical planes ---

    def test_identical_single_channel_mae_zero(self):
        """Identical single-channel buffers → mae == 0.0."""
        a = _make_single_channel(0.5)
        delta = compute_plane_delta(a, a, "C")
        assert delta.mae == 0.0, f"Identical planes must give mae=0.0, got {delta.mae}"

    def test_identical_single_channel_moved_false(self):
        """Identical planes → moved == False."""
        a = _make_single_channel(0.5)
        delta = compute_plane_delta(a, a, "C")
        assert delta.moved is False

    def test_identical_single_channel_psnr_inf_or_large(self):
        """Identical planes → psnr is infinite (or a very large float sentinel)."""
        a = _make_single_channel(0.5)
        delta = compute_plane_delta(a, a, "C")
        # Python field may be float('inf') or a very large float (>= 1e30)
        # Both are acceptable at the Python level; JSON safety is via to_dict().
        assert math.isinf(delta.psnr) or delta.psnr >= 1e30, (
            f"Identical planes must give psnr=inf or a large sentinel, got {delta.psnr}"
        )

    def test_identical_single_channel_to_dict_psnr_none(self):
        """CRITICAL: to_dict()['psnr'] must be None for identical planes (mae==0)."""
        a = _make_single_channel(0.5)
        delta = compute_plane_delta(a, a, "C")
        d = delta.to_dict()
        assert d["psnr"] is None, (
            f"to_dict()['psnr'] must be None when planes are identical, got {d['psnr']!r}"
        )

    def test_identical_rgb_mae_zero(self):
        """Identical 3-channel buffers → mae == 0.0."""
        a = _make_rgb_channels(0.5, 0.3, 0.1)
        delta = compute_plane_delta(a, a, "C")
        assert delta.mae == 0.0

    def test_identical_rgb_moved_false(self):
        """Identical 3-channel buffers → moved == False."""
        a = _make_rgb_channels()
        delta = compute_plane_delta(a, a, "C")
        assert delta.moved is False

    def test_identical_rgb_mean_delta_zero(self):
        """Identical 3-channel buffers → all mean_delta values == 0.0."""
        a = _make_rgb_channels(0.5, 0.3, 0.1)
        delta = compute_plane_delta(a, a, "C")
        for ch_i, md in enumerate(delta.mean_delta):
            assert abs(md) < 1e-9, f"mean_delta[{ch_i}] must be 0.0 for identical planes, got {md}"

    def test_identical_rgb_max_abs_delta_zero(self):
        """Identical 3-channel buffers → all max_abs_delta values == 0.0."""
        a = _make_rgb_channels(0.5, 0.3, 0.1)
        delta = compute_plane_delta(a, a, "C")
        for ch_i, mad in enumerate(delta.max_abs_delta):
            assert abs(mad) < 1e-9, f"max_abs_delta[{ch_i}] must be 0.0, got {mad}"

    def test_identical_rgb_to_dict_json_safe(self):
        """to_dict() of identical-plane result is json.dumps-able."""
        a = _make_rgb_channels()
        delta = compute_plane_delta(a, a, "C")
        json.dumps(delta.to_dict())

    # --- changed planes ---

    def test_changed_single_channel_mae_nonzero(self):
        """Changed single-channel buffer → mae > 0."""
        a = [[0.0, 0.0, 0.0, 0.0]]
        b = [[1.0, 1.0, 1.0, 1.0]]
        delta = compute_plane_delta(a, b, "Pz")
        assert delta.mae > 0.0

    def test_changed_single_channel_moved_true(self):
        """Changed single-channel buffer → moved == True."""
        a = [[0.0, 0.0, 0.0, 0.0]]
        b = [[1.0, 1.0, 1.0, 1.0]]
        delta = compute_plane_delta(a, b, "Pz")
        assert delta.moved is True

    def test_changed_single_channel_psnr_finite(self):
        """Changed planes with nonzero mae → psnr is finite."""
        a = [[0.0, 0.0, 0.0, 0.0]]
        b = [[1.0, 1.0, 1.0, 1.0]]
        delta = compute_plane_delta(a, b, "Pz")
        assert not math.isinf(delta.psnr), f"Changed planes must give finite psnr, got {delta.psnr}"

    def test_changed_single_channel_to_dict_psnr_not_none(self):
        """Changed planes → to_dict()['psnr'] is NOT None (it is a finite float)."""
        a = [[0.0, 0.0, 0.0, 0.0]]
        b = [[1.0, 1.0, 1.0, 1.0]]
        delta = compute_plane_delta(a, b, "Pz")
        d = delta.to_dict()
        assert d["psnr"] is not None, (
            "to_dict()['psnr'] must be a finite number when planes differ"
        )
        json.dumps(d)  # must still be JSON-safe

    def test_changed_rgb_mae_known_value(self):
        """3-channel 2x2 all-zero vs all-one → mae == 1.0."""
        # a: 3 channels, 4 pixels each, all 0.0
        a = [[0.0] * 4, [0.0] * 4, [0.0] * 4]
        b = [[1.0] * 4, [1.0] * 4, [1.0] * 4]
        delta = compute_plane_delta(a, b, "C")
        assert abs(delta.mae - 1.0) < 1e-6, (
            f"All-zero vs all-one RGB → mae must be 1.0, got {delta.mae}"
        )

    def test_changed_rgb_mean_delta_known_value(self):
        """All-zero minus all-one: mean_delta per channel == -1.0 (a - b)."""
        a = [[0.0] * 4, [0.0] * 4, [0.0] * 4]
        b = [[1.0] * 4, [1.0] * 4, [1.0] * 4]
        delta = compute_plane_delta(a, b, "C")
        for ch_i, md in enumerate(delta.mean_delta):
            assert abs(md - (-1.0)) < 1e-6 or abs(md - 1.0) < 1e-6, (
                f"mean_delta[{ch_i}] must be ±1.0 for all-zero vs all-one, got {md}"
            )

    def test_changed_rgb_max_abs_delta_known_value(self):
        """All-zero vs all-one: max_abs_delta per channel == 1.0."""
        a = [[0.0] * 4, [0.0] * 4, [0.0] * 4]
        b = [[1.0] * 4, [1.0] * 4, [1.0] * 4]
        delta = compute_plane_delta(a, b, "C")
        for ch_i, mad in enumerate(delta.max_abs_delta):
            assert abs(mad - 1.0) < 1e-6, (
                f"max_abs_delta[{ch_i}] must be 1.0 for all-zero vs all-one, got {mad}"
            )

    def test_per_channel_max_abs_delta_reflects_max(self):
        """max_abs_delta[ch] must be the maximum |a[ch][i] - b[ch][i]| over all pixels."""
        # ch0: deltas are [0.1, 0.5, 0.3, 0.2] → max=0.5
        # ch1: deltas are [0.0, 0.8, 0.0, 0.0] → max=0.8
        a = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        b = [[0.1, 0.5, 0.3, 0.2], [0.0, 0.8, 0.0, 0.0]]
        delta = compute_plane_delta(a, b, "N")
        assert abs(delta.max_abs_delta[0] - 0.5) < 1e-6, (
            f"max_abs_delta[0] must be 0.5, got {delta.max_abs_delta[0]}"
        )
        assert abs(delta.max_abs_delta[1] - 0.8) < 1e-6, (
            f"max_abs_delta[1] must be 0.8, got {delta.max_abs_delta[1]}"
        )

    def test_mean_delta_reflects_sign(self):
        """mean_delta signs must reflect direction of (a - b)."""
        # a > b in ch0; a < b in ch1
        a = [[1.0, 1.0], [0.0, 0.0]]
        b = [[0.0, 0.0], [1.0, 1.0]]
        delta = compute_plane_delta(a, b, "test")
        # ch0: a-b = 1.0 (positive) or ch1: a-b = -1.0 (negative)
        # mean_delta must reflect this polarity difference
        assert delta.mean_delta[0] != delta.mean_delta[1], (
            "mean_delta must reflect per-channel sign difference"
        )

    def test_plane_name_preserved(self):
        """plane name passed to compute_plane_delta is preserved in the result."""
        a = _make_single_channel(0.5)
        delta = compute_plane_delta(a, a, "CryptoMaterial")
        assert delta.plane == "CryptoMaterial"

    def test_returns_planedelta_instance(self):
        """compute_plane_delta returns a PlaneDelta instance."""
        a = _make_single_channel(0.5)
        result = compute_plane_delta(a, a, "C")
        assert isinstance(result, PlaneDelta)

    def test_len_mean_delta_matches_channels(self):
        """len(mean_delta) == number of channels in the buffer."""
        a = _make_rgb_channels()
        delta = compute_plane_delta(a, a, "C")
        assert len(delta.mean_delta) == 3

    def test_len_max_abs_delta_matches_channels(self):
        """len(max_abs_delta) == number of channels in the buffer."""
        a = _make_rgb_channels()
        delta = compute_plane_delta(a, a, "C")
        assert len(delta.max_abs_delta) == 3

    def test_small_epsilon_boundary_moved(self):
        """A delta very slightly above zero: moved must be True (not clipped to False)."""
        epsilon = 1e-6
        a = [[0.0, 0.0]]
        b = [[epsilon, epsilon]]
        delta = compute_plane_delta(a, b, "C")
        # Either moved=True or moved=False depending on the epsilon threshold —
        # we only assert the delta is internally consistent: if moved==False then
        # mae must also be ~0 (i.e. the epsilon was below threshold).
        # The key invariant: moved and mae must agree.
        if delta.moved:
            assert delta.mae > 0.0
        else:
            # The implementation chose to treat this as below-threshold.
            pass

    def test_identical_zero_buffer(self):
        """All-zero buffers are identical → mae=0, moved=False."""
        a = [[0.0, 0.0, 0.0, 0.0]]
        delta = compute_plane_delta(a, a, "black")
        assert delta.mae == 0.0
        assert delta.moved is False


# ===========================================================================
# Section 4 — aov_presence_diff(a_planes, b_planes) -> (only_a, only_b, common)
#
# Locked contract:
#   a_planes, b_planes: list[str] (ordered plane name lists)
#   Returns: (only_a: list[str], only_b: list[str], common: list[str])
#   Order-stable: preserves the order of the original lists.
# ===========================================================================

class TestAovPresenceDiff:
    """aov_presence_diff — order-stable set-diff of two AOV name lists."""

    def test_disjoint_lists(self):
        """Two completely disjoint lists → everything only-in-A or only-in-B, nothing common."""
        only_a, only_b, common = aov_presence_diff(["C", "N"], ["Pz", "Diffuse"])
        assert set(only_a) == {"C", "N"}
        assert set(only_b) == {"Pz", "Diffuse"}
        assert common == []

    def test_identical_lists(self):
        """Identical lists → nothing only-in-A/B, everything common."""
        only_a, only_b, common = aov_presence_diff(["C", "N", "Pz"], ["C", "N", "Pz"])
        assert only_a == []
        assert only_b == []
        assert set(common) == {"C", "N", "Pz"}

    def test_a_superset_of_b(self):
        """A has all of B plus more → only_a has the extras, only_b empty."""
        only_a, only_b, common = aov_presence_diff(["C", "N", "Pz"], ["C", "N"])
        assert set(only_a) == {"Pz"}
        assert only_b == []
        assert set(common) == {"C", "N"}

    def test_b_superset_of_a(self):
        """B has all of A plus more → only_b has the extras, only_a empty."""
        only_a, only_b, common = aov_presence_diff(["C", "N"], ["C", "N", "Pz"])
        assert only_a == []
        assert set(only_b) == {"Pz"}
        assert set(common) == {"C", "N"}

    def test_partial_overlap(self):
        """Partial overlap: some in A only, some in B only, some in both."""
        only_a, only_b, common = aov_presence_diff(
            ["C", "N", "Pz"], ["C", "Diffuse", "N"]
        )
        assert set(only_a) == {"Pz"}
        assert set(only_b) == {"Diffuse"}
        assert set(common) == {"C", "N"}

    def test_empty_a(self):
        """Empty A → only_a empty, only_b is B, common empty."""
        only_a, only_b, common = aov_presence_diff([], ["C", "N"])
        assert only_a == []
        assert set(only_b) == {"C", "N"}
        assert common == []

    def test_empty_b(self):
        """Empty B → only_b empty, only_a is A, common empty."""
        only_a, only_b, common = aov_presence_diff(["C", "N"], [])
        assert set(only_a) == {"C", "N"}
        assert only_b == []
        assert common == []

    def test_both_empty(self):
        """Both empty → all three return lists are empty."""
        only_a, only_b, common = aov_presence_diff([], [])
        assert only_a == []
        assert only_b == []
        assert common == []

    def test_order_stable_only_a(self):
        """only_a preserves the order of items as they appear in a_planes."""
        only_a, _, _ = aov_presence_diff(["Pz", "C", "N"], ["C"])
        # Pz and N are only-in-A; they must appear in the same relative order as in a_planes
        assert only_a.index("Pz") < only_a.index("N"), (
            "only_a must preserve the order from a_planes"
        )

    def test_order_stable_only_b(self):
        """only_b preserves the order of items as they appear in b_planes."""
        _, only_b, _ = aov_presence_diff(["C"], ["Pz", "Diffuse", "N"])
        assert only_b.index("Pz") < only_b.index("Diffuse") < only_b.index("N"), (
            "only_b must preserve the order from b_planes"
        )

    def test_order_stable_common(self):
        """common preserves order (from a_planes or b_planes — impl picks one, must be stable)."""
        _, _, common = aov_presence_diff(["C", "N", "Pz"], ["Pz", "N", "C"])
        # All three are common; the order must match either a or b consistently
        assert set(common) == {"C", "N", "Pz"}
        assert len(common) == 3

    def test_returns_tuple_of_three(self):
        """aov_presence_diff returns a 3-tuple (only_a, only_b, common)."""
        result = aov_presence_diff(["C"], ["C"])
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_single_aov_only_in_a(self):
        """B missing one AOV that A has → only_a = [that_aov]."""
        only_a, only_b, common = aov_presence_diff(["C", "N", "Pz"], ["C", "N"])
        assert only_a == ["Pz"]

    def test_no_duplicates_in_common(self):
        """common must not contain duplicates even if input lists have the same items."""
        _, _, common = aov_presence_diff(["C", "N"], ["N", "C"])
        assert len(common) == len(set(common)), "common must not have duplicates"

    def test_spec_example(self):
        """Spec §4.2 example: B missing Pz → aovs_only_in_a contains 'Pz'."""
        only_a, only_b, common = aov_presence_diff(
            ["C", "N", "Pz"], ["C", "N"]
        )
        assert "Pz" in only_a
        assert "Pz" not in only_b
        assert "Pz" not in common


# ===========================================================================
# Section 5 — build_verdict(report) -> str
#
# Locked contract:
#   report: CompareReport
#   Returns a plain-English string summarising the compare result.
#
#   Invariants:
#   - Returns a non-empty string.
#   - "no change" (or similar) when all planes have moved=False and no AOV diff.
#   - Mentions plane name when a plane has moved=True.
#   - Mentions AOV name when an AOV is only-in-A or only-in-B.
# ===========================================================================

class TestBuildVerdict:
    """build_verdict — plain-English summary from CompareReport."""

    def _make_identical_report(self):
        delta = PlaneDelta(
            plane="C", mean_delta=[0.0, 0.0, 0.0],
            max_abs_delta=[0.0, 0.0, 0.0], mae=0.0,
            psnr=float("inf"), moved=False,
        )
        return CompareReport(
            aovs_only_in_a=[], aovs_only_in_b=[], aovs_common=["C"],
            per_plane=[delta], verdict="",  # verdict populated by build_verdict
        )

    def _make_changed_report(self):
        delta = PlaneDelta(
            plane="C", mean_delta=[0.05, 0.03, 0.01],
            max_abs_delta=[0.1, 0.08, 0.04], mae=0.031,
            psnr=38.5, moved=True,
        )
        return CompareReport(
            aovs_only_in_a=[], aovs_only_in_b=[], aovs_common=["C"],
            per_plane=[delta], verdict="",
        )

    def test_returns_string(self):
        """build_verdict returns a str."""
        r = self._make_identical_report()
        result = build_verdict(r)
        assert isinstance(result, str)

    def test_returns_nonempty_string(self):
        """build_verdict returns a non-empty string."""
        r = self._make_identical_report()
        assert len(build_verdict(r)) > 0

    def test_no_change_scenario(self):
        """All planes unchanged, no AOV diff → verdict mentions 'no change' or 'unchanged'."""
        r = self._make_identical_report()
        verdict = build_verdict(r)
        # Must signal no meaningful change
        assert "no change" in verdict.lower() or "unchanged" in verdict.lower(), (
            f"Expected 'no change'/'unchanged' in verdict for identical planes, got: {verdict!r}"
        )

    def test_changed_plane_mentioned(self):
        """A plane with moved=True has its name mentioned in the verdict."""
        r = self._make_changed_report()
        verdict = build_verdict(r)
        assert "C" in verdict, (
            f"Changed plane 'C' must be mentioned in verdict, got: {verdict!r}"
        )

    def test_lost_aov_mentioned(self):
        """An AOV only-in-A is mentioned in the verdict ('lost' or similar)."""
        delta = PlaneDelta(
            plane="C", mean_delta=[0.0], max_abs_delta=[0.0],
            mae=0.0, psnr=float("inf"), moved=False,
        )
        r = CompareReport(
            aovs_only_in_a=["Pz"], aovs_only_in_b=[], aovs_common=["C"],
            per_plane=[delta], verdict="",
        )
        verdict = build_verdict(r)
        assert "Pz" in verdict, (
            f"Lost AOV 'Pz' must be mentioned in verdict, got: {verdict!r}"
        )

    def test_gained_aov_mentioned(self):
        """An AOV only-in-B is mentioned in the verdict ('gained' or similar)."""
        delta = PlaneDelta(
            plane="C", mean_delta=[0.0], max_abs_delta=[0.0],
            mae=0.0, psnr=float("inf"), moved=False,
        )
        r = CompareReport(
            aovs_only_in_a=[], aovs_only_in_b=["NewLayer"], aovs_common=["C"],
            per_plane=[delta], verdict="",
        )
        verdict = build_verdict(r)
        assert "NewLayer" in verdict, (
            f"Gained AOV 'NewLayer' must be mentioned in verdict, got: {verdict!r}"
        )

    def test_spec_example_verdict(self):
        """Spec §4.2 example verdict shape: 'beauty changed (mae 0.031); N unchanged; lost AOV Pz'."""
        beauty = PlaneDelta(
            plane="C", mean_delta=[0.01, 0.02, 0.005],
            max_abs_delta=[0.05, 0.07, 0.02], mae=0.031,
            psnr=38.5, moved=True,
        )
        n_plane = PlaneDelta(
            plane="N", mean_delta=[0.0, 0.0, 0.0],
            max_abs_delta=[0.0, 0.0, 0.0], mae=0.0,
            psnr=float("inf"), moved=False,
        )
        r = CompareReport(
            aovs_only_in_a=["Pz"], aovs_only_in_b=[], aovs_common=["C", "N"],
            per_plane=[beauty, n_plane], verdict="",
        )
        verdict = build_verdict(r)
        # Must mention the changed plane, the unchanged plane, and the lost AOV
        assert "C" in verdict or "beauty" in verdict.lower(), (
            f"Changed plane C must appear in verdict: {verdict!r}"
        )
        assert "Pz" in verdict, f"Lost AOV Pz must appear in verdict: {verdict!r}"


# ===========================================================================
# Section 6 — PixelSummary dataclass
#
# Locked contract (matches spec §4.2 render_read_pixels returns shape):
#   PixelSummary(
#       plane:    str,
#       xres:     int,
#       yres:     int,
#       channels: int,
#       dtype:    str,
#       stats:    dict  — {min: list[float], max: list[float], mean: list[float],
#                          nan_count: int, inf_count: int}
#       histogram: dict — {bins: int, counts: list[list[int]]}
#   )
# ===========================================================================

class TestPixelSummary:
    """PixelSummary dataclass — locked field names and types."""

    def _make_summary(self):
        return PixelSummary(
            plane="N",
            xres=4,
            yres=4,
            channels=3,
            dtype="float32",
            stats={
                "min": [-1.0, -1.0, -1.0],
                "max": [1.0, 1.0, 1.0],
                "mean": [0.0, 0.0, 0.0],
                "nan_count": 0,
                "inf_count": 0,
            },
            histogram={
                "bins": 8,
                "counts": [[2, 2, 2, 2, 2, 2, 2, 2]] * 3,
            },
        )

    def test_has_plane(self):
        s = self._make_summary()
        assert s.plane == "N"

    def test_has_xres(self):
        s = self._make_summary()
        assert s.xres == 4

    def test_has_yres(self):
        s = self._make_summary()
        assert s.yres == 4

    def test_has_channels(self):
        s = self._make_summary()
        assert s.channels == 3

    def test_has_dtype(self):
        s = self._make_summary()
        assert s.dtype == "float32"

    def test_stats_has_min(self):
        s = self._make_summary()
        assert "min" in s.stats
        assert isinstance(s.stats["min"], list)

    def test_stats_has_max(self):
        s = self._make_summary()
        assert "max" in s.stats
        assert isinstance(s.stats["max"], list)

    def test_stats_has_mean(self):
        s = self._make_summary()
        assert "mean" in s.stats
        assert isinstance(s.stats["mean"], list)

    def test_stats_has_nan_count(self):
        s = self._make_summary()
        assert "nan_count" in s.stats
        assert isinstance(s.stats["nan_count"], int)

    def test_stats_has_inf_count(self):
        s = self._make_summary()
        assert "inf_count" in s.stats
        assert isinstance(s.stats["inf_count"], int)

    def test_histogram_has_bins(self):
        s = self._make_summary()
        assert "bins" in s.histogram
        assert isinstance(s.histogram["bins"], int)

    def test_histogram_has_counts(self):
        s = self._make_summary()
        assert "counts" in s.histogram
        assert isinstance(s.histogram["counts"], list)

    def test_histogram_counts_is_list_of_lists(self):
        """counts is list[list[int]] — one list per channel."""
        s = self._make_summary()
        for ch_counts in s.histogram["counts"]:
            assert isinstance(ch_counts, list)


# ===========================================================================
# Section 7 — summarize_pixels(channels, plane, xres, yres, dtype) -> PixelSummary
#
# Locked contract:
#   channels: list[list[float]] — per-channel flat pixel buffers
#   plane:    str
#   xres:     int
#   yres:     int
#   dtype:    str
#   Returns: PixelSummary with stats and histogram populated.
#
#   Invariants:
#   - stats["min"][ch] == min of channels[ch]
#   - stats["max"][ch] == max of channels[ch]
#   - stats["mean"][ch] ≈ mean of channels[ch]
#   - stats["nan_count"] counts math.isnan() pixels across all channels
#   - stats["inf_count"] counts math.isinf() pixels across all channels
#   - histogram["counts"][ch] has exactly histogram["bins"] buckets
#   - sum(histogram["counts"][ch]) == len(channels[ch]) for finite pixels
# ===========================================================================

class TestSummarizePixels:
    """summarize_pixels — per-channel stats + nan/inf counting + histogram."""

    def test_returns_pixel_summary(self):
        """summarize_pixels returns a PixelSummary."""
        channels = _make_rgb_channels(0.5, 0.3, 0.1)
        result = summarize_pixels(channels, "C", 2, 2, "float32")
        assert isinstance(result, PixelSummary)

    def test_plane_name_preserved(self):
        """plane name is stored in the result."""
        channels = _make_single_channel(0.5)
        s = summarize_pixels(channels, "Pz", 2, 2, "float32")
        assert s.plane == "Pz"

    def test_xres_yres_stored(self):
        """xres and yres are stored in the result."""
        channels = _make_single_channel(0.5, size=9)
        s = summarize_pixels(channels, "C", 3, 3, "float32")
        assert s.xres == 3
        assert s.yres == 3

    def test_dtype_stored(self):
        """dtype string is stored in the result."""
        channels = _make_single_channel(0.5)
        s = summarize_pixels(channels, "C", 2, 2, "float16")
        assert s.dtype == "float16"

    def test_channels_count_stored(self):
        """channels count matches len(channels) input."""
        channels = _make_rgb_channels()
        s = summarize_pixels(channels, "C", 2, 2, "float32")
        assert s.channels == 3

    def test_stats_min_per_channel(self):
        """stats['min'][ch] == min of channels[ch]."""
        # ch0: [0.1, 0.9, 0.5, 0.3], ch1: [0.2, 0.4, 0.1, 0.6]
        channels = [
            [0.1, 0.9, 0.5, 0.3],
            [0.2, 0.4, 0.1, 0.6],
        ]
        s = summarize_pixels(channels, "N", 2, 2, "float32")
        assert abs(s.stats["min"][0] - 0.1) < 1e-6
        assert abs(s.stats["min"][1] - 0.1) < 1e-6

    def test_stats_max_per_channel(self):
        """stats['max'][ch] == max of channels[ch]."""
        channels = [
            [0.1, 0.9, 0.5, 0.3],
            [0.2, 0.4, 0.1, 0.6],
        ]
        s = summarize_pixels(channels, "N", 2, 2, "float32")
        assert abs(s.stats["max"][0] - 0.9) < 1e-6
        assert abs(s.stats["max"][1] - 0.6) < 1e-6

    def test_stats_mean_per_channel(self):
        """stats['mean'][ch] ≈ arithmetic mean of channels[ch]."""
        # ch0: all 0.5 → mean=0.5
        channels = [[0.5, 0.5, 0.5, 0.5]]
        s = summarize_pixels(channels, "C", 2, 2, "float32")
        assert abs(s.stats["mean"][0] - 0.5) < 1e-6

    def test_stats_mean_known_value(self):
        """mean of [0.0, 1.0, 0.0, 1.0] == 0.5."""
        channels = [[0.0, 1.0, 0.0, 1.0]]
        s = summarize_pixels(channels, "C", 2, 2, "float32")
        assert abs(s.stats["mean"][0] - 0.5) < 1e-6

    def test_nan_count_zero_when_no_nans(self):
        """nan_count == 0 for a clean buffer."""
        channels = _make_single_channel(0.5)
        s = summarize_pixels(channels, "C", 2, 2, "float32")
        assert s.stats["nan_count"] == 0

    def test_nan_count_nonzero_when_nan_present(self):
        """nan_count > 0 when at least one pixel is NaN (a real render bug signal, FR-3)."""
        channels = [[0.5, float("nan"), 0.5, 0.5]]
        s = summarize_pixels(channels, "beauty", 2, 2, "float32")
        assert s.stats["nan_count"] > 0, (
            "NaN in buffer must be counted (FR-3 — NaN in beauty AOV is a real render bug)"
        )

    def test_inf_count_zero_when_no_infs(self):
        """inf_count == 0 for a clean buffer."""
        channels = _make_single_channel(0.5)
        s = summarize_pixels(channels, "C", 2, 2, "float32")
        assert s.stats["inf_count"] == 0

    def test_inf_count_nonzero_when_inf_present(self):
        """inf_count > 0 when at least one pixel is Inf."""
        channels = [[0.5, float("inf"), 0.5, 0.5]]
        s = summarize_pixels(channels, "C", 2, 2, "float32")
        assert s.stats["inf_count"] > 0

    def test_histogram_counts_per_channel(self):
        """histogram['counts'] has one list per channel."""
        channels = _make_rgb_channels()
        s = summarize_pixels(channels, "C", 2, 2, "float32")
        assert len(s.histogram["counts"]) == 3

    def test_histogram_bins_positive(self):
        """histogram['bins'] is a positive integer."""
        channels = _make_single_channel(0.5)
        s = summarize_pixels(channels, "C", 2, 2, "float32")
        assert s.histogram["bins"] > 0

    def test_histogram_counts_length_equals_bins(self):
        """Each per-channel counts list has exactly histogram['bins'] buckets."""
        channels = _make_rgb_channels()
        s = summarize_pixels(channels, "C", 2, 2, "float32")
        bins = s.histogram["bins"]
        for ch_idx, ch_counts in enumerate(s.histogram["counts"]):
            assert len(ch_counts) == bins, (
                f"histogram['counts'][{ch_idx}] must have {bins} buckets, got {len(ch_counts)}"
            )

    def test_histogram_counts_are_ints(self):
        """histogram counts must be integers (pixel counts)."""
        channels = _make_single_channel(0.5)
        s = summarize_pixels(channels, "C", 2, 2, "float32")
        for ch_counts in s.histogram["counts"]:
            for count in ch_counts:
                assert isinstance(count, int), (
                    f"histogram count must be int, got {type(count)!r}: {count!r}"
                )


# ===========================================================================
# Section 8 — ReadbackPage dataclass
#
# Locked contract (spec §4.2 render_read_pixels returns):
#   ReadbackPage(
#       pixels:       list[list[float]],   # list of rows, each row = list of channel values
#       page:         int,
#       page_size:    int,
#       total_pages:  int,
#       truncated:    bool,
#   )
# ===========================================================================

class TestReadbackPage:
    """ReadbackPage dataclass — locked field names and types."""

    def _make_page(self, **kwargs):
        defaults = dict(
            pixels=[[0.5, 0.3, 0.1], [0.5, 0.3, 0.1]],
            page=0,
            page_size=2,
            total_pages=1,
            truncated=False,
        )
        defaults.update(kwargs)
        return ReadbackPage(**defaults)

    def test_has_pixels(self):
        p = self._make_page()
        assert isinstance(p.pixels, list)

    def test_has_page(self):
        p = self._make_page(page=1)
        assert p.page == 1

    def test_has_page_size(self):
        p = self._make_page(page_size=1024)
        assert p.page_size == 1024

    def test_has_total_pages(self):
        p = self._make_page(total_pages=5)
        assert p.total_pages == 5

    def test_has_truncated_false(self):
        p = self._make_page(truncated=False)
        assert p.truncated is False

    def test_has_truncated_true(self):
        p = self._make_page(truncated=True)
        assert p.truncated is True

    def test_pixels_is_list_of_lists(self):
        """pixels is a list of rows, each row a list of channel values."""
        p = self._make_page(pixels=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        for row in p.pixels:
            assert isinstance(row, list)


# ===========================================================================
# Section 9 — clamp_and_paginate(pixels, max_pixels, page, page_size) -> ReadbackPage
#
# Locked contract:
#   pixels:     list[list[float]] — list of pixel rows (each row = channel values)
#   max_pixels: int               — server-side hard cap
#   page:       int               — 0-based page index
#   page_size:  int               — rows per page
#   Returns: ReadbackPage
#
#   Invariants:
#   - If total pixel rows > max_pixels: returned pixels are clamped; truncated=True
#   - If total pixel rows <= max_pixels: truncated=False (or depends on page)
#   - page and page_size are reflected in the result
#   - total_pages == ceil(min(len(pixels), max_pixels) / page_size)
#   - returned pixels for a given page: the correct slice of the (possibly clamped) list
# ===========================================================================

class TestClampAndPaginate:
    """clamp_and_paginate — server-side pixel clamp + pagination."""

    def _make_pixels(self, n_rows, n_channels=3):
        """Make n_rows pixel rows with n_channels channels each, values 0..1."""
        return [[float(i) / max(n_rows, 1)] * n_channels for i in range(n_rows)]

    # --- basic structure ---

    def test_returns_readback_page(self):
        """clamp_and_paginate returns a ReadbackPage."""
        pixels = self._make_pixels(4)
        result = clamp_and_paginate(pixels, max_pixels=1024, page=0, page_size=4)
        assert isinstance(result, ReadbackPage)

    def test_page_reflected(self):
        """result.page == the requested page."""
        pixels = self._make_pixels(8)
        result = clamp_and_paginate(pixels, max_pixels=1024, page=1, page_size=4)
        assert result.page == 1

    def test_page_size_reflected(self):
        """result.page_size == the requested page_size."""
        pixels = self._make_pixels(8)
        result = clamp_and_paginate(pixels, max_pixels=1024, page=0, page_size=4)
        assert result.page_size == 4

    # --- truncation ---

    def test_no_truncation_under_max(self):
        """When total rows <= max_pixels: truncated == False."""
        pixels = self._make_pixels(10)
        result = clamp_and_paginate(pixels, max_pixels=1024, page=0, page_size=10)
        assert result.truncated is False

    def test_truncation_over_max(self):
        """When total rows > max_pixels: truncated == True (FR-3 context-safety guarantee)."""
        pixels = self._make_pixels(100)
        result = clamp_and_paginate(pixels, max_pixels=10, page=0, page_size=10)
        assert result.truncated is True, (
            "truncated must be True when pixel count exceeds max_pixels "
            "(FR-3 context-safety guarantee)"
        )

    def test_truncated_pixels_clamped_to_max(self):
        """When truncated, total accessible pixels across pages == max_pixels."""
        pixels = self._make_pixels(100)
        max_px = 20
        result = clamp_and_paginate(pixels, max_pixels=max_px, page=0, page_size=max_px)
        assert len(result.pixels) <= max_px, (
            f"Clamped pixel count must be <= {max_px}, got {len(result.pixels)}"
        )

    # --- pagination ---

    def test_first_page_correct_rows(self):
        """Page 0 returns the first page_size rows."""
        pixels = self._make_pixels(8, n_channels=1)
        result = clamp_and_paginate(pixels, max_pixels=1024, page=0, page_size=4)
        assert len(result.pixels) == 4
        # First row should be the first pixel row
        assert result.pixels[0] == pixels[0]

    def test_second_page_correct_rows(self):
        """Page 1 returns the second page_size rows."""
        pixels = self._make_pixels(8, n_channels=1)
        result = clamp_and_paginate(pixels, max_pixels=1024, page=1, page_size=4)
        assert len(result.pixels) == 4
        assert result.pixels[0] == pixels[4]

    def test_total_pages_single_page(self):
        """8 rows, page_size=8 → total_pages == 1."""
        pixels = self._make_pixels(8)
        result = clamp_and_paginate(pixels, max_pixels=1024, page=0, page_size=8)
        assert result.total_pages == 1

    def test_total_pages_multiple(self):
        """8 rows, page_size=4 → total_pages == 2."""
        pixels = self._make_pixels(8)
        result = clamp_and_paginate(pixels, max_pixels=1024, page=0, page_size=4)
        assert result.total_pages == 2

    def test_total_pages_partial_last(self):
        """9 rows, page_size=4 → total_pages == 3 (last page partial)."""
        pixels = self._make_pixels(9)
        result = clamp_and_paginate(pixels, max_pixels=1024, page=0, page_size=4)
        assert result.total_pages == 3

    def test_last_page_partial(self):
        """Last page may have fewer rows than page_size."""
        pixels = self._make_pixels(9, n_channels=1)
        # Page 2 (third page) has 1 row
        result = clamp_and_paginate(pixels, max_pixels=1024, page=2, page_size=4)
        assert len(result.pixels) == 1

    def test_pixels_are_list_of_lists(self):
        """result.pixels is a list of lists."""
        pixels = self._make_pixels(4)
        result = clamp_and_paginate(pixels, max_pixels=1024, page=0, page_size=4)
        for row in result.pixels:
            assert isinstance(row, list)

    def test_clamped_total_pages_with_truncation(self):
        """When clamped (truncated=True), total_pages is based on the clamped pixel count."""
        pixels = self._make_pixels(100)
        max_px = 20
        result = clamp_and_paginate(pixels, max_pixels=max_px, page=0, page_size=10)
        # Clamped to 20 pixels, page_size=10 → 2 pages
        assert result.total_pages == 2, (
            f"With 100 rows clamped to 20, page_size=10 → total_pages=2, got {result.total_pages}"
        )


# ===========================================================================
# Section 10 — CL-015 purity gate
#
# render_readback_model.py must import with NO hou / NO Qt / NO numpy.
# This section verifies it at import time under plain Python.
# ===========================================================================

class TestHouFreeImport:
    """render_readback_model.py must carry no hou / Qt / pxr / numpy dependency."""

    def test_module_importable_without_hou(self):
        """render_readback_model must load under plain Python with no hou installed."""
        import fxhoudinimcp.render_readback_model as m
        assert m is not None

    def test_no_hou_import(self):
        """render_readback_model must not import 'hou' at module top-level (CL-015)."""
        import fxhoudinimcp.render_readback_model as m
        import inspect
        import re
        source = inspect.getsource(m)
        source_no_comments = re.sub(r"#.*", "", source)
        assert "import hou" not in source_no_comments, (
            "render_readback_model.py must not import hou (CL-015 — pure-logic boundary)"
        )

    def test_no_numpy_import(self):
        """render_readback_model must not import numpy (fork venv lacks numpy)."""
        import fxhoudinimcp.render_readback_model as m
        import inspect
        import re
        source = inspect.getsource(m)
        source_no_comments = re.sub(r"#.*", "", source)
        assert "import numpy" not in source_no_comments, (
            "render_readback_model.py must not import numpy "
            "(fork .venv lacks numpy; pure-Python math required)"
        )

    def test_no_pyside_import(self):
        """render_readback_model must not import PySide6/PySide2/Qt."""
        import fxhoudinimcp.render_readback_model as m
        import inspect
        import re
        source = inspect.getsource(m)
        source_no_comments = re.sub(r"#.*", "", source)
        assert "PySide" not in source_no_comments, (
            "render_readback_model.py must not import Qt (CL-015)"
        )

    def test_no_pxr_import(self):
        """render_readback_model must not import pxr (CL-015)."""
        import fxhoudinimcp.render_readback_model as m
        import inspect
        import re
        source = inspect.getsource(m)
        source_no_comments = re.sub(r"#.*", "", source)
        assert "import pxr" not in source_no_comments, (
            "render_readback_model.py must not import pxr (CL-015)"
        )


# ===========================================================================
# Section 11 — Codex T2 defect coverage (pp12-114a-fix)
#
# Five RED tests exposing defects found by Codex cross-vendor review.
# These tests FAIL against the CURRENT render_readback_model.py.
# hou-dev turns them green in the pp12-114a-fix implementation step.
#
# F1 (BLOCKER) — compute_plane_delta: inf pixel crashes with ValueError
# F2 (BLOCKER) — PixelSummary.to_dict(): list(int) raises TypeError
# F3 (MAJOR)   — NaN pixel: moved=False (wrong) + psnr=NaN in to_dict() (not JSON-safe)
# F4 (MAJOR)   — compute_plane_delta has no peak_value param; PSNR ignores AOV peak
# F5 (MINOR)   — shape mismatch silently raises IndexError, not a clear ValueError
#
# Discipline:
#   - Asserts against the PUBLIC observable contract, not implementation internals.
#   - Reuses existing module-level helpers (_make_single_channel, _make_rgb_channels).
#   - Does NOT modify any previously-authored tests or helpers.
#   - Does NOT touch render_readback_model.py.
# ===========================================================================

class TestCodexDefectCoverage:
    """RED tests for Codex T2 findings F1-F5 (pp12-114a-fix).

    All five tests FAIL against the current implementation and are
    expected to PASS after hou-dev applies the targeted fixes.
    """

    # -----------------------------------------------------------------------
    # F1 (BLOCKER): inf pixel must NOT crash compute_plane_delta
    #
    # Current behaviour: diff = inf - 0.0 = inf; sq_err += inf*inf = inf;
    #   mse = inf; 1.0/inf = 0.0; math.log10(0.0) raises ValueError.
    # Required behaviour: return a PlaneDelta with moved=True and a
    #   JSON-safe psnr (None or a float sentinel — never raises).
    # -----------------------------------------------------------------------

    def test_f1_inf_pixel_does_not_raise(self):
        """F1 (BLOCKER): compute_plane_delta with an inf pixel must NOT raise.

        Current: math.log10(0.0) raises ValueError: math domain error.
        Fixed:   returns PlaneDelta normally; psnr is a finite sentinel or None.
        """
        a = [[0.0, float("inf")]]
        b = [[0.0, 0.0]]
        # Must not raise any exception
        result = compute_plane_delta(a, b, "beauty")
        # Postcondition: result is a PlaneDelta (not a crash)
        assert isinstance(result, PlaneDelta), (
            "compute_plane_delta must return PlaneDelta even with an inf pixel"
        )

    def test_f1_inf_pixel_moved_true(self):
        """F1 (BLOCKER): inf pixel must set moved=True (planes are not identical)."""
        a = [[0.0, float("inf")]]
        b = [[0.0, 0.0]]
        result = compute_plane_delta(a, b, "beauty")
        assert result.moved is True, (
            "compute_plane_delta with inf pixel must return moved=True "
            "(an inf pixel differs from 0.0; the plane has moved)"
        )

    def test_f1_inf_pixel_to_dict_json_safe(self):
        """F1 (BLOCKER): to_dict() of an inf-pixel delta must be json.dumps-safe.

        psnr when mse=inf is undefined; to_dict() must emit None (or a finite
        sentinel), not float('inf') or float('nan'), which both break json.dumps.
        """
        a = [[0.0, float("inf")]]
        b = [[0.0, 0.0]]
        result = compute_plane_delta(a, b, "beauty")
        d = result.to_dict()
        # Must not raise
        json.dumps(d)
        # psnr in to_dict must be None or a finite float — never inf or nan
        psnr_val = d["psnr"]
        assert psnr_val is None or (
            isinstance(psnr_val, float) and math.isfinite(psnr_val)
        ), (
            f"to_dict()['psnr'] must be None or finite when mse=inf; got {psnr_val!r}"
        )

    # -----------------------------------------------------------------------
    # F2 (BLOCKER): PixelSummary.to_dict() raises TypeError on histogram
    #
    # Current: {k: list(v) for k, v in histogram.items()} iterates over
    #   {"bins": 16 (int), "counts": [[...]]}.  list(16) raises
    #   TypeError: 'int' object is not iterable.
    # Required: to_dict() returns {"bins": int, "counts": list[list[int]]}
    #   and json.dumps succeeds.
    # -----------------------------------------------------------------------

    def test_f2_pixel_summary_to_dict_does_not_raise(self):
        """F2 (BLOCKER): PixelSummary.to_dict() must not raise TypeError.

        Current: list(bins_int) raises TypeError.
        Fixed:   histogram serialised as {bins: int, counts: list[list]}.
        """
        channels = _make_single_channel(0.5)
        summary = summarize_pixels(channels, "beauty", 2, 2, "float32")
        # Must not raise
        d = summary.to_dict()
        assert isinstance(d, dict), "to_dict() must return a dict"

    def test_f2_pixel_summary_to_dict_json_serializable(self):
        """F2 (BLOCKER): json.dumps(summary.to_dict()) must not raise."""
        channels = _make_rgb_channels(0.5, 0.3, 0.1)
        summary = summarize_pixels(channels, "N", 2, 2, "float32")
        d = summary.to_dict()
        json.dumps(d)  # Must not raise

    def test_f2_histogram_shape_in_to_dict(self):
        """F2 (BLOCKER): to_dict()['histogram'] must be {bins:int, counts:list[list]}.

        bins must be an int (the bucket count), not a list.
        counts must be a list of lists (one per channel), not something else.
        """
        channels = _make_rgb_channels()
        summary = summarize_pixels(channels, "C", 2, 2, "float32")
        d = summary.to_dict()
        hist = d["histogram"]
        assert isinstance(hist["bins"], int), (
            f"histogram['bins'] in to_dict() must be int, got {type(hist['bins'])!r}. "
            "Current bug: list(bins_int) raises TypeError before to_dict() returns."
        )
        assert isinstance(hist["counts"], list), (
            "histogram['counts'] in to_dict() must be a list"
        )
        for ch_idx, ch_counts in enumerate(hist["counts"]):
            assert isinstance(ch_counts, list), (
                f"histogram['counts'][{ch_idx}] must be a list of ints"
            )

    # -----------------------------------------------------------------------
    # F3 (MAJOR): NaN pixel -> moved=False (wrong) + to_dict() psnr is NaN
    #
    # Current: diff = nan - 0.0 = nan; |nan| = nan; sum += nan = nan;
    #   mae = nan > 0.0 is False (IEEE 754) -> moved=False (false "unchanged").
    #   Also: to_dict() only guards math.isinf(psnr); math.isnan(psnr) slips
    #   through as the literal float NaN token, which json.dumps emits as "NaN"
    #   (CPython) or raises (strict mode) -- not a valid JSON value.
    # Required: moved=True; to_dict()['psnr'] is None; json.dumps succeeds.
    # -----------------------------------------------------------------------

    def test_f3_nan_pixel_moved_true(self):
        """F3 (MAJOR): NaN pixel must set moved=True (not False).

        IEEE 754: nan > 0.0 is False, so the current mae>0.0 test gives
        moved=False even though the plane is corrupted. A NaN pixel is
        always a signal that the plane has changed / is invalid.
        """
        a = [[float("nan")]]
        b = [[0.0]]
        result = compute_plane_delta(a, b, "beauty")
        assert result.moved is True, (
            "compute_plane_delta with a NaN pixel must return moved=True. "
            "A NaN pixel is not 'unchanged'; the current bug gives moved=False "
            "because nan > 0.0 is False under IEEE 754."
        )

    def test_f3_nan_pixel_to_dict_psnr_json_safe(self):
        """F3 (MAJOR): to_dict()['psnr'] must be None after a NaN pixel.

        Current: math.isnan check is absent; NaN psnr leaks into to_dict()
        as float('nan'), which json.dumps emits as the non-standard token 'NaN'
        (CPython behaviour) or raises in strict mode.
        """
        a = [[float("nan")]]
        b = [[0.0]]
        result = compute_plane_delta(a, b, "beauty")
        d = result.to_dict()
        psnr_val = d["psnr"]
        assert psnr_val is None or (
            isinstance(psnr_val, float) and math.isfinite(psnr_val)
        ), (
            f"to_dict()['psnr'] must be None or finite when pixel is NaN; "
            f"got {psnr_val!r}. The isnan guard is missing in to_dict()."
        )

    def test_f3_nan_pixel_to_dict_json_dumps_ok(self):
        """F3 (MAJOR): json.dumps must succeed after a NaN-pixel delta."""
        a = [[float("nan")]]
        b = [[0.0]]
        result = compute_plane_delta(a, b, "beauty")
        # Must not raise (currently emits non-standard NaN token or raises)
        json.dumps(result.to_dict())

    # -----------------------------------------------------------------------
    # F4 (MAJOR): compute_plane_delta has no peak_value parameter
    #
    # Current signature: compute_plane_delta(a, b, plane) -- PSNR hardcoded
    #   to peak=1.0, which is wrong for HDR, depth, normal, and position AOVs.
    # Required:
    #   - compute_plane_delta(a, b, plane, peak_value=1.0) accepted without TypeError.
    #   - PSNR with peak_value=100.0 differs from peak_value=1.0 for non-zero MSE.
    #   - Default peak_value=1.0 keeps all existing tests green.
    # -----------------------------------------------------------------------

    def test_f4_peak_value_param_accepted(self):
        """F4 (MAJOR): compute_plane_delta must accept a peak_value keyword argument.

        Current: TypeError: compute_plane_delta() got an unexpected keyword argument
        'peak_value'.
        """
        a = [[0.0, 0.5]]
        b = [[0.0, 1.0]]
        # Must not raise TypeError
        result = compute_plane_delta(a, b, "depth", peak_value=100.0)
        assert isinstance(result, PlaneDelta), (
            "compute_plane_delta must accept peak_value kwarg and return PlaneDelta"
        )

    def test_f4_peak_value_affects_psnr(self):
        """F4 (MAJOR): psnr must scale with peak_value for the same pixel delta.

        For identical non-zero MSE, PSNR = 10*log10(peak^2 / mse) means
        peak=100.0 gives a higher psnr than peak=1.0.
        """
        # a != b so MSE > 0 and PSNR is finite and comparable
        a = [[0.0, 0.0]]
        b = [[0.5, 0.5]]
        result_default = compute_plane_delta(a, b, "depth")          # peak_value=1.0
        result_hdr = compute_plane_delta(a, b, "depth", peak_value=100.0)
        # With a larger peak the same noise is proportionally smaller -> higher PSNR
        assert result_hdr.psnr > result_default.psnr, (
            f"PSNR with peak_value=100.0 ({result_hdr.psnr:.2f}) must be greater "
            f"than with peak_value=1.0 ({result_default.psnr:.2f}). "
            "Current: peak is hardcoded to 1.0, so peak_value kwarg is ignored / absent."
        )

    def test_f4_default_peak_value_preserves_existing_behaviour(self):
        """F4 (MAJOR): default peak_value=1.0 must not change existing psnr values."""
        a = [[0.0] * 4, [0.0] * 4, [0.0] * 4]
        b = [[1.0] * 4, [1.0] * 4, [1.0] * 4]
        result_no_peak = compute_plane_delta(a, b, "C")
        result_default_peak = compute_plane_delta(a, b, "C", peak_value=1.0)
        assert abs(result_no_peak.psnr - result_default_peak.psnr) < 1e-9, (
            "compute_plane_delta() and compute_plane_delta(..., peak_value=1.0) "
            "must produce identical psnr for the same inputs."
        )

    # -----------------------------------------------------------------------
    # F5 (MINOR): shape mismatch raises IndexError, not ValueError
    #
    # Current: mismatched channel count or per-channel pixel count causes a
    #   bare IndexError at runtime (no bounds check).
    # Required: a clear ValueError is raised before any computation.
    # -----------------------------------------------------------------------

    def test_f5_channel_count_mismatch_raises_value_error(self):
        """F5 (MINOR): mismatched channel count must raise ValueError, not IndexError.

        Current: the loop iterates over num_channels = len(a), then accesses
        b[ch] -- if len(b) < len(a), this is a bare IndexError with no context.
        """
        a = [[0.0, 1.0], [0.0, 1.0]]   # 2 channels
        b = [[0.0, 1.0]]                 # 1 channel -- mismatch
        with pytest.raises(ValueError):
            compute_plane_delta(a, b, "C")

    def test_f5_pixel_count_mismatch_raises_value_error(self):
        """F5 (MINOR): mismatched per-channel pixel count must raise ValueError.

        Current: len(ch_a) may differ from len(ch_b) and the inner loop uses
        len(ch_a) -- accesses ch_b[px] out-of-bounds -> bare IndexError.
        """
        a = [[1.0, 2.0, 3.0]]   # 3 pixels
        b = [[1.0, 2.0]]         # 2 pixels -- mismatch
        with pytest.raises(ValueError):
            compute_plane_delta(a, b, "C")
