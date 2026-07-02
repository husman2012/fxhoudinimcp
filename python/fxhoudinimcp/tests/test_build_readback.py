"""
Tests for build_readback() — pure-logic response builder (PP12-114 PR-4, RED phase).

TDD phase: RED — build_readback() does NOT exist yet.
Expected failure: ImportError on 'from fxhoudinimcp.render_readback_model import build_readback'

Contract: imports NO hou, NO Qt/PySide6, NO pxr, NO numpy.
Plain Python stdlib only.  Off-DCC, pytest-able headless.

Covers the public contract of:
  build_readback(channels, plane, xres, yres, dtype,
                 mode, roi, max_pixels, downsample,
                 page, page_size, num_bins) -> dict

Locked §4.2 return shape:
  {
    'plane':       str,
    'xres':        int,   # ALWAYS source dims, NOT roi dims
    'yres':        int,
    'channels':    int,
    'dtype':       str,
    'mode':        str,
    'stats':       {'min':[..], 'max':[..], 'mean':[..],
                    'nan_count':int, 'inf_count':int},
    'histogram':   {'bins':int, 'counts':[[..],..] },
    'pixels':      [[..], ..],   # [] in 'summary' mode
    'page':        int,
    'page_size':   int,
    'total_pages': int,
    'truncated':   bool,
  }

Functional ACs (5):
  AC-1  summary mode  — pixels=[], total_pages=1, truncated=False
  AC-2  roi mode      — pixel rows from roi crop + clamp_and_paginate
  AC-3  sample mode   — full-frame strided, clamp_and_paginate; truncated=True when rows>max_pixels
  AC-4  empty path    — channels==[] → zero/empty response; never raise
  AC-5  xres/yres invariant — always source dims in the returned dict, never roi dims

Cross-references:
  - Plan pp12-114e-model lockedFieldContract (BINDING)
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
# Import build_readback from the module.
# In the RED phase this raises ImportError because the function does not
# exist yet in render_readback_model.py.
# ---------------------------------------------------------------------------
from fxhoudinimcp.render_readback_model import build_readback


# ===========================================================================
# Helpers
# ===========================================================================

def _flat_channels(value: float, n_channels: int, n_pixels: int):
    """Return per-channel flat buffers with constant value.

    Layout: channels[ch][pixel_flat_index]
    """
    return [[value] * n_pixels for _ in range(n_channels)]


def _rgb_2x2(r=0.5, g=0.3, b=0.1):
    """3-channel 2x2 (4 pixel) RGB buffer."""
    return [
        [r, r, r, r],
        [g, g, g, g],
        [b, b, b, b],
    ]


def _call_summary(channels=None, xres=2, yres=2):
    """Minimal valid call in 'summary' mode."""
    if channels is None:
        channels = _rgb_2x2()
    return build_readback(
        channels=channels,
        plane="C",
        xres=xres,
        yres=yres,
        dtype="float32",
        mode="summary",
    )


def _call_roi(channels=None, xres=4, yres=4, roi=None):
    """Minimal valid call in 'roi' mode."""
    if channels is None:
        channels = _flat_channels(0.5, 3, xres * yres)
    if roi is None:
        roi = [0, 0, 2, 2]  # x0,y0,x1,y1 (2x2 crop)
    return build_readback(
        channels=channels,
        plane="C",
        xres=xres,
        yres=yres,
        dtype="float32",
        mode="roi",
        roi=roi,
    )


def _call_sample(channels=None, xres=4, yres=4, downsample=1, max_pixels=4096):
    """Minimal valid call in 'sample' mode."""
    if channels is None:
        channels = _flat_channels(0.5, 3, xres * yres)
    return build_readback(
        channels=channels,
        plane="C",
        xres=xres,
        yres=yres,
        dtype="float32",
        mode="sample",
        downsample=downsample,
        max_pixels=max_pixels,
    )


# Required §4.2 top-level keys
_REQUIRED_KEYS = {
    "plane", "xres", "yres", "channels", "dtype", "mode",
    "stats", "histogram", "pixels", "page", "page_size",
    "total_pages", "truncated",
}
_REQUIRED_STATS_KEYS = {"min", "max", "mean", "nan_count", "inf_count"}
_REQUIRED_HIST_KEYS = {"bins", "counts"}


# ===========================================================================
# Section 1 — Top-level return shape (all modes)
# ===========================================================================

class TestReturnShape:
    """build_readback returns a dict with the locked §4.2 keys in all modes."""

    def test_returns_dict_summary(self):
        result = _call_summary()
        assert isinstance(result, dict)

    def test_returns_dict_roi(self):
        result = _call_roi()
        assert isinstance(result, dict)

    def test_returns_dict_sample(self):
        result = _call_sample()
        assert isinstance(result, dict)

    def test_all_top_level_keys_present_summary(self):
        result = _call_summary()
        missing = _REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing top-level keys in summary mode: {missing}"

    def test_all_top_level_keys_present_roi(self):
        result = _call_roi()
        missing = _REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing top-level keys in roi mode: {missing}"

    def test_all_top_level_keys_present_sample(self):
        result = _call_sample()
        missing = _REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing top-level keys in sample mode: {missing}"

    def test_stats_sub_keys_present(self):
        result = _call_summary()
        stats = result["stats"]
        missing = _REQUIRED_STATS_KEYS - set(stats.keys())
        assert not missing, f"Missing stats keys: {missing}"

    def test_histogram_sub_keys_present(self):
        result = _call_summary()
        hist = result["histogram"]
        missing = _REQUIRED_HIST_KEYS - set(hist.keys())
        assert not missing, f"Missing histogram keys: {missing}"

    def test_result_is_json_serializable_summary(self):
        """The entire dict must be json.dumps-able (no float inf/nan leaking out)."""
        result = _call_summary()
        json.dumps(result)  # must not raise

    def test_result_is_json_serializable_roi(self):
        result = _call_roi()
        json.dumps(result)

    def test_result_is_json_serializable_sample(self):
        result = _call_sample()
        json.dumps(result)


# ===========================================================================
# AC-1 — summary mode: pixels=[], total_pages=1, truncated=False
# ===========================================================================

class TestSummaryMode:
    """AC-1: 'summary' mode must return empty pixels, total_pages=1, truncated=False."""

    def test_mode_field_reflects_summary(self):
        result = _call_summary()
        assert result["mode"] == "summary"

    def test_pixels_empty_in_summary(self):
        result = _call_summary()
        assert result["pixels"] == [], (
            "summary mode must return pixels=[] (no pixel data returned, only stats)"
        )

    def test_total_pages_one_in_summary(self):
        result = _call_summary()
        assert result["total_pages"] == 1, (
            "summary mode must return total_pages=1"
        )

    def test_truncated_false_in_summary(self):
        result = _call_summary()
        assert result["truncated"] is False, (
            "summary mode must return truncated=False"
        )

    def test_stats_populated_in_summary(self):
        """Stats must be computed even in summary mode (that is the whole point)."""
        channels = [[0.2, 0.4, 0.6, 0.8]]  # 1-channel 4-pixel buffer
        result = build_readback(
            channels=channels, plane="Pz", xres=2, yres=2,
            dtype="float32", mode="summary",
        )
        # min should be close to 0.2, max to 0.8
        assert len(result["stats"]["min"]) == 1
        assert result["stats"]["min"][0] <= 0.2 + 1e-6
        assert result["stats"]["max"][0] >= 0.8 - 1e-6

    def test_histogram_populated_in_summary(self):
        """Histogram must be computed in summary mode."""
        result = _call_summary()
        hist = result["histogram"]
        assert isinstance(hist["bins"], int) and hist["bins"] > 0
        assert isinstance(hist["counts"], list) and len(hist["counts"]) > 0

    def test_page_zero_in_summary(self):
        """page must be 0 in summary mode (no pagination needed)."""
        result = _call_summary()
        assert result["page"] == 0


# ===========================================================================
# AC-2 — roi mode: pixel rows from roi crop + clamp_and_paginate
# ===========================================================================

class TestRoiMode:
    """AC-2: 'roi' mode applies a rect crop then returns paginated pixel rows."""

    def test_mode_field_reflects_roi(self):
        result = _call_roi()
        assert result["mode"] == "roi"

    def test_pixels_nonempty_for_valid_roi(self):
        """A valid roi crop with pixels inside must yield a non-empty pixels list."""
        # 4x4 source, roi=[0,0,2,2] → 4 pixels in crop
        channels = _flat_channels(0.5, 3, 16)
        result = build_readback(
            channels=channels, plane="C", xres=4, yres=4,
            dtype="float32", mode="roi", roi=[0, 0, 2, 2],
        )
        assert len(result["pixels"]) > 0, (
            "roi mode with a valid crop must return non-empty pixels"
        )

    def test_pixels_are_list_of_lists_in_roi(self):
        """Each pixel row is a list of floats (one per channel)."""
        result = _call_roi()
        for row in result["pixels"]:
            assert isinstance(row, list), f"Each pixel row must be a list, got {type(row)!r}"

    def test_pixel_row_length_matches_channels_in_roi(self):
        """Each pixel row has len == number of channels."""
        channels = _rgb_2x2()  # 3 channels, 4 pixels
        result = build_readback(
            channels=channels, plane="C", xres=2, yres=2,
            dtype="float32", mode="roi", roi=[0, 0, 2, 2],
        )
        for row in result["pixels"]:
            assert len(row) == 3, (
                f"Pixel row length must match channel count (3), got {len(row)}"
            )

    def test_roi_crop_reduces_pixel_count_vs_full_frame(self):
        """A strict sub-roi must return fewer pixels than a full-frame roi."""
        channels = _flat_channels(0.5, 1, 16)  # 4x4 single-channel
        full = build_readback(
            channels=channels, plane="C", xres=4, yres=4,
            dtype="float32", mode="roi", roi=[0, 0, 4, 4],
        )
        half = build_readback(
            channels=channels, plane="C", xres=4, yres=4,
            dtype="float32", mode="roi", roi=[0, 0, 2, 2],
        )
        # The full-frame roi has 16 pixels; the half roi has 4 pixels.
        # After pagination (page_size default 1024), all fit on one page.
        # We only compare the number of pixels returned in the first page.
        assert len(half["pixels"]) < len(full["pixels"]), (
            "A strict sub-roi must return fewer pixels than the full-frame roi"
        )

    def test_roi_pagination_page_field(self):
        """result['page'] reflects the requested page index."""
        channels = _flat_channels(0.5, 1, 16)
        result = build_readback(
            channels=channels, plane="C", xres=4, yres=4,
            dtype="float32", mode="roi", roi=[0, 0, 4, 4],
            page=0, page_size=4,
        )
        assert result["page"] == 0

    def test_roi_pagination_page_size_reflected(self):
        """result['page_size'] reflects the requested page_size."""
        channels = _flat_channels(0.5, 1, 16)
        result = build_readback(
            channels=channels, plane="C", xres=4, yres=4,
            dtype="float32", mode="roi", roi=[0, 0, 4, 4],
            page=0, page_size=4,
        )
        assert result["page_size"] == 4


# ===========================================================================
# AC-3 — sample mode: full-frame strided, truncated when rows>max_pixels
# ===========================================================================

class TestSampleMode:
    """AC-3: 'sample' mode covers the full frame with a stride (downsample)."""

    def test_mode_field_reflects_sample(self):
        result = _call_sample()
        assert result["mode"] == "sample"

    def test_pixels_nonempty_in_sample(self):
        """sample mode on a non-empty buffer must yield pixels."""
        result = _call_sample(xres=2, yres=2)
        # 4 pixels total; default max_pixels=4096 → all fit
        assert len(result["pixels"]) > 0

    def test_truncated_true_when_pixels_exceed_max(self):
        """truncated=True when total pixels (after stride) exceed max_pixels."""
        # 4x4 = 16 pixels, max_pixels=4 → truncated
        channels = _flat_channels(0.5, 3, 16)
        result = build_readback(
            channels=channels, plane="C", xres=4, yres=4,
            dtype="float32", mode="sample", max_pixels=4,
        )
        assert result["truncated"] is True, (
            "truncated must be True when pixel count exceeds max_pixels in sample mode"
        )

    def test_truncated_false_when_pixels_fit(self):
        """truncated=False when pixel count <= max_pixels."""
        channels = _flat_channels(0.5, 1, 4)  # 2x2 = 4 pixels
        result = build_readback(
            channels=channels, plane="C", xres=2, yres=2,
            dtype="float32", mode="sample", max_pixels=4096,
        )
        assert result["truncated"] is False

    def test_downsample_reduces_pixel_count(self):
        """downsample=2 must yield fewer pixel rows than downsample=1."""
        channels = _flat_channels(0.5, 1, 16)  # 4x4
        full = build_readback(
            channels=channels, plane="C", xres=4, yres=4,
            dtype="float32", mode="sample", downsample=1, max_pixels=4096,
        )
        strided = build_readback(
            channels=channels, plane="C", xres=4, yres=4,
            dtype="float32", mode="sample", downsample=2, max_pixels=4096,
        )
        assert len(strided["pixels"]) < len(full["pixels"]), (
            "downsample=2 must yield fewer pixel rows than downsample=1"
        )

    def test_sample_pixel_row_length_matches_channels(self):
        """Each pixel row in sample mode has len == number of channels."""
        channels = _rgb_2x2()  # 3-channel, 4 pixels
        result = build_readback(
            channels=channels, plane="C", xres=2, yres=2,
            dtype="float32", mode="sample",
        )
        for row in result["pixels"]:
            assert len(row) == 3, (
                f"Pixel row must have 3 channel values, got {len(row)}"
            )


# ===========================================================================
# AC-4 — empty path: channels==[] → zero/empty response, never raise
# ===========================================================================

class TestEmptyPath:
    """AC-4: channels==[] must return a valid zero/empty dict and never raise."""

    def test_empty_channels_summary_does_not_raise(self):
        """build_readback with channels=[] in summary mode must not raise."""
        build_readback(channels=[], plane="C", xres=2, yres=2,
                       dtype="float32", mode="summary")

    def test_empty_channels_roi_does_not_raise(self):
        """build_readback with channels=[] in roi mode must not raise."""
        build_readback(channels=[], plane="C", xres=2, yres=2,
                       dtype="float32", mode="roi", roi=[0, 0, 2, 2])

    def test_empty_channels_sample_does_not_raise(self):
        """build_readback with channels=[] in sample mode must not raise."""
        build_readback(channels=[], plane="C", xres=2, yres=2,
                       dtype="float32", mode="sample")

    def test_empty_channels_returns_dict(self):
        """build_readback with channels=[] returns a dict."""
        result = build_readback(channels=[], plane="C", xres=2, yres=2,
                                dtype="float32", mode="summary")
        assert isinstance(result, dict)

    def test_empty_channels_all_keys_present(self):
        """All §4.2 keys must be present even when channels=[]."""
        result = build_readback(channels=[], plane="C", xres=2, yres=2,
                                dtype="float32", mode="summary")
        missing = _REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys with empty channels: {missing}"

    def test_empty_channels_channels_field_is_zero(self):
        """channels field in result must be 0 when input channels=[]."""
        result = build_readback(channels=[], plane="C", xres=2, yres=2,
                                dtype="float32", mode="summary")
        assert result["channels"] == 0

    def test_empty_channels_pixels_is_empty_list(self):
        """pixels must be [] when channels=[]."""
        result = build_readback(channels=[], plane="C", xres=2, yres=2,
                                dtype="float32", mode="summary")
        assert result["pixels"] == []

    def test_empty_channels_is_json_serializable(self):
        """The empty-channel result must still be json.dumps-able."""
        result = build_readback(channels=[], plane="C", xres=2, yres=2,
                                dtype="float32", mode="summary")
        json.dumps(result)


# ===========================================================================
# AC-5 — xres/yres invariant: always source dims, never roi dims
# ===========================================================================

class TestXresYresInvariant:
    """AC-5: xres/yres in result always equal the SOURCE image dims, not roi dims."""

    def test_xres_yres_are_source_dims_in_summary(self):
        """In summary mode, xres/yres must equal the passed-in source dims."""
        result = build_readback(
            channels=_flat_channels(0.5, 3, 9),
            plane="C", xres=3, yres=3,
            dtype="float32", mode="summary",
        )
        assert result["xres"] == 3, f"xres must be source dim 3, got {result['xres']}"
        assert result["yres"] == 3, f"yres must be source dim 3, got {result['yres']}"

    def test_xres_yres_are_source_dims_not_roi_dims(self):
        """In roi mode, xres/yres must still be the SOURCE dims even though the roi is smaller."""
        # Source: 4x4 (xres=4, yres=4); ROI: [0,0,2,2] → 2x2 crop
        channels = _flat_channels(0.5, 3, 16)
        result = build_readback(
            channels=channels, plane="C",
            xres=4, yres=4,
            dtype="float32", mode="roi",
            roi=[0, 0, 2, 2],
        )
        assert result["xres"] == 4, (
            f"xres must be SOURCE dim (4), not roi width (2), got {result['xres']}"
        )
        assert result["yres"] == 4, (
            f"yres must be SOURCE dim (4), not roi height (2), got {result['yres']}"
        )

    def test_xres_yres_are_source_dims_in_sample(self):
        """In sample mode with downsample, xres/yres must still equal the source dims."""
        channels = _flat_channels(0.5, 1, 16)  # 4x4
        result = build_readback(
            channels=channels, plane="C",
            xres=4, yres=4,
            dtype="float32", mode="sample",
            downsample=2,
        )
        assert result["xres"] == 4, (
            f"xres must be source dim (4), not strided count, got {result['xres']}"
        )
        assert result["yres"] == 4, (
            f"yres must be source dim (4), not strided count, got {result['yres']}"
        )


# ===========================================================================
# Section 6 — Metadata fields (plane, dtype, channels count)
# ===========================================================================

class TestMetadataFields:
    """plane, dtype, channels count, mode preserved verbatim in the result."""

    def test_plane_name_preserved(self):
        result = build_readback(
            channels=_rgb_2x2(), plane="Pz", xres=2, yres=2,
            dtype="float32", mode="summary",
        )
        assert result["plane"] == "Pz"

    def test_dtype_preserved(self):
        result = build_readback(
            channels=_rgb_2x2(), plane="C", xres=2, yres=2,
            dtype="float16", mode="summary",
        )
        assert result["dtype"] == "float16"

    def test_channels_count_matches_input(self):
        """channels field equals len(channels) passed in."""
        channels = _rgb_2x2()  # 3 channels
        result = build_readback(
            channels=channels, plane="C", xres=2, yres=2,
            dtype="float32", mode="summary",
        )
        assert result["channels"] == 3

    def test_mode_preserved_summary(self):
        result = build_readback(
            channels=_rgb_2x2(), plane="C", xres=2, yres=2,
            dtype="float32", mode="summary",
        )
        assert result["mode"] == "summary"

    def test_mode_preserved_roi(self):
        result = _call_roi()
        assert result["mode"] == "roi"

    def test_mode_preserved_sample(self):
        result = _call_sample()
        assert result["mode"] == "sample"


class TestUnknownModeHardening:
    """M-01: build_readback must raise ValueError for unknown mode.

    RED until hou-dev hardens the mode dispatch in render_readback_model.py.
    Currently the 'else:' branch silently runs sample for any unrecognised mode,
    which masks caller errors.  After hardening, an unrecognised mode must raise
    ValueError immediately.
    """

    def test_unknown_mode_raises_value_error(self):
        """M-01 hardening: unknown mode must raise ValueError, not silently run sample."""
        with pytest.raises(ValueError):
            build_readback(
                channels=[[0.0, 1.0]],
                plane="C",
                xres=2,
                yres=1,
                dtype="float32",
                mode="bogus",
            )
