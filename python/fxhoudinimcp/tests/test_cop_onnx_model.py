"""
Tests for cop_onnx_model.py — pure-logic layer (PP12-113 PR-1).

TDD phase: RED — cop_onnx_model.py does NOT exist yet.
Expected failure: ModuleNotFoundError on import.

Contract: imports NO hou, NO Qt/PySide6, NO pxr, NO numpy, NO MaterialX.
Plain Python stdlib only. Runs under the fork .venv with plain pytest
headless (off-DCC, no Houdini install required).

Covers the public contract of:
  - TensorSpec     (dataclass) — name, shape, dtype, layout_guess='unknown', to_dict()
  - OnnxContract   (dataclass) — model_path, inputs, outputs, opset, producer,
                                  loadable, error, to_dict()
  - PixelSummary   (dataclass) — plane, xres, yres, channels, dtype, min, max, mean,
                                  nan_count, inf_count, histogram_bins, histogram_counts,
                                  to_dict() nesting into {stats:{...}, histogram:{...}}
  - ReadbackPage   (dataclass) — plane, mode, pixels, page, page_size, total_pages,
                                  truncated; __post_init__ raises ValueError on a bad mode
  - guess_layout(shape) -> str
  - clamp_readback(xres, yres, channels, max_pixels) -> int
  - paginate(total_items, page, page_size) -> dict
  - a pixel nan/inf-safe stats helper (count_nan_inf / summarize-style)

Cross-references:
  - Plan pp12-113a lockedFieldContract (BINDING)
  - spec.md §7.2 (data model), §9.1 (unit tests)
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
# cop_onnx_model.py does not exist yet.
# ---------------------------------------------------------------------------
from fxhoudinimcp.cop_onnx_model import (
    TensorSpec,
    OnnxContract,
    PixelSummary,
    ReadbackPage,
    guess_layout,
    clamp_readback,
    paginate,
    count_nan_inf,
)


# ===========================================================================
# Section 1 — TensorSpec dataclass
#
# Locked contract:
#   TensorSpec(name: str, shape: list, dtype: str, layout_guess: str = "unknown")
#   shape holds ints OR the literal string "dynamic" for symbolic dims.
#   to_dict() -> {name, shape, dtype, layout_guess}
# ===========================================================================

class TestTensorSpec:
    """TensorSpec dataclass — field names, dynamic-dim preservation, to_dict()."""

    def test_minimal_construction(self):
        t = TensorSpec(name="input", shape=[1, 3, 512, 512], dtype="float32")
        assert t.name == "input"
        assert t.shape == [1, 3, 512, 512]
        assert t.dtype == "float32"

    def test_layout_guess_defaults_to_unknown(self):
        t = TensorSpec(name="input", shape=[1, 256], dtype="float32")
        assert t.layout_guess == "unknown"

    def test_layout_guess_can_be_set(self):
        t = TensorSpec(name="input", shape=[1, 3, 512, 512], dtype="float32", layout_guess="NCHW")
        assert t.layout_guess == "NCHW"

    def test_dynamic_dim_preserved_in_shape(self):
        """shape may contain the literal string 'dynamic' for symbolic dims."""
        t = TensorSpec(name="input", shape=[1, 3, "dynamic", "dynamic"], dtype="float32")
        assert t.shape == [1, 3, "dynamic", "dynamic"]

    def test_to_dict_keys(self):
        t = TensorSpec(name="output", shape=[1, 1000], dtype="float32", layout_guess="unknown")
        d = t.to_dict()
        assert set(d.keys()) == {"name", "shape", "dtype", "layout_guess"}, (
            f"Unexpected keys in TensorSpec.to_dict(): {set(d.keys())!r}"
        )

    def test_to_dict_values(self):
        t = TensorSpec(name="input", shape=[1, 3, 512, 512], dtype="float32", layout_guess="NCHW")
        d = t.to_dict()
        assert d["name"] == "input"
        assert d["shape"] == [1, 3, 512, 512]
        assert d["dtype"] == "float32"
        assert d["layout_guess"] == "NCHW"

    def test_to_dict_preserves_dynamic_dims(self):
        """to_dict() must round-trip 'dynamic' dims end-to-end, not coerce them to None/0."""
        t = TensorSpec(name="input", shape=[1, 3, "dynamic", "dynamic"], dtype="float32")
        d = t.to_dict()
        assert d["shape"] == [1, 3, "dynamic", "dynamic"], (
            f"TensorSpec.to_dict() must preserve 'dynamic' dims verbatim, got {d['shape']!r}"
        )

    def test_to_dict_is_json_serialisable(self):
        t = TensorSpec(name="input", shape=[1, 3, "dynamic", "dynamic"], dtype="float32", layout_guess="NCHW")
        json.dumps(t.to_dict())


# ===========================================================================
# Section 2 — OnnxContract dataclass
#
# Locked contract:
#   OnnxContract(model_path: str, inputs: list[TensorSpec], outputs: list[TensorSpec],
#                opset: int|None, producer: str|None, loadable: bool, error: str|None)
#   to_dict() -> {model_path, inputs:[t.to_dict()...], outputs:[t.to_dict()...],
#                 opset, producer, loadable, error}
#   MUST round-trip dynamic dims via to_dict() end-to-end.
# ===========================================================================

class TestOnnxContract:
    """OnnxContract dataclass — nested TensorSpec serialisation, dynamic-dim round-trip."""

    def _make_contract(self, **kwargs):
        defaults = dict(
            model_path="/models/identity.onnx",
            inputs=[TensorSpec(name="input", shape=[1, 3, 512, 512], dtype="float32", layout_guess="NCHW")],
            outputs=[TensorSpec(name="output", shape=[1, 1000], dtype="float32")],
            opset=17,
            producer="pytorch",
            loadable=True,
            error=None,
        )
        defaults.update(kwargs)
        return OnnxContract(**defaults)

    def test_minimal_construction(self):
        c = self._make_contract()
        assert c.model_path == "/models/identity.onnx"
        assert c.loadable is True
        assert c.error is None

    def test_opset_can_be_none(self):
        c = self._make_contract(opset=None)
        assert c.opset is None

    def test_producer_can_be_none(self):
        c = self._make_contract(producer=None)
        assert c.producer is None

    def test_loadable_false_with_error(self):
        c = self._make_contract(loadable=False, error="invalid protobuf", inputs=[], outputs=[])
        assert c.loadable is False
        assert c.error == "invalid protobuf"

    def test_to_dict_keys(self):
        c = self._make_contract()
        d = c.to_dict()
        assert set(d.keys()) == {
            "model_path", "inputs", "outputs", "opset", "producer", "loadable", "error",
        }, f"Unexpected keys in OnnxContract.to_dict(): {set(d.keys())!r}"

    def test_to_dict_inputs_is_list_of_dicts(self):
        c = self._make_contract()
        d = c.to_dict()
        assert isinstance(d["inputs"], list)
        assert len(d["inputs"]) == 1
        assert isinstance(d["inputs"][0], dict)
        assert d["inputs"][0]["name"] == "input"

    def test_to_dict_outputs_is_list_of_dicts(self):
        c = self._make_contract()
        d = c.to_dict()
        assert isinstance(d["outputs"], list)
        assert d["outputs"][0]["name"] == "output"

    def test_to_dict_dynamic_dims_round_trip(self):
        """CRITICAL: building from a shape containing 'dynamic' must survive to_dict() end-to-end."""
        c = self._make_contract(
            inputs=[TensorSpec(name="input", shape=[1, 3, "dynamic", "dynamic"], dtype="float32", layout_guess="NCHW")],
        )
        d = c.to_dict()
        assert d["inputs"][0]["shape"] == [1, 3, "dynamic", "dynamic"], (
            "OnnxContract.to_dict() must preserve 'dynamic' dims through the nested "
            f"TensorSpec, got {d['inputs'][0]['shape']!r}"
        )

    def test_to_dict_is_json_serialisable(self):
        c = self._make_contract()
        json.dumps(c.to_dict())

    def test_to_dict_error_none_json_safe(self):
        c = self._make_contract()
        d = c.to_dict()
        assert d["error"] is None
        json.dumps(d)


# ===========================================================================
# Section 3 — PixelSummary dataclass
#
# Locked contract:
#   PixelSummary(plane, xres, yres, channels, dtype,
#                min, max, mean,                 # per-channel lists
#                nan_count, inf_count,
#                histogram_bins, histogram_counts)  # per-channel list of per-bin lists
#   to_dict() -> {plane,xres,yres,channels,dtype,
#                 stats:{min,max,mean,nan_count,inf_count},
#                 histogram:{bins,counts}}
# ===========================================================================

class TestPixelSummary:
    """PixelSummary dataclass — locked fields, nested to_dict() shape (spec §4.2)."""

    def _make_summary(self, **kwargs):
        defaults = dict(
            plane="beauty",
            xres=4,
            yres=4,
            channels=3,
            dtype="float32",
            min=[-1.0, -1.0, -1.0],
            max=[1.0, 1.0, 1.0],
            mean=[0.0, 0.0, 0.0],
            nan_count=0,
            inf_count=0,
            histogram_bins=8,
            histogram_counts=[[2] * 8, [2] * 8, [2] * 8],
        )
        defaults.update(kwargs)
        return PixelSummary(**defaults)

    def test_minimal_construction(self):
        s = self._make_summary()
        assert s.plane == "beauty"
        assert s.xres == 4
        assert s.yres == 4
        assert s.channels == 3
        assert s.dtype == "float32"

    def test_per_channel_lists_length(self):
        s = self._make_summary()
        assert len(s.min) == s.channels
        assert len(s.max) == s.channels
        assert len(s.mean) == s.channels

    def test_to_dict_top_level_keys(self):
        s = self._make_summary()
        d = s.to_dict()
        assert set(d.keys()) == {"plane", "xres", "yres", "channels", "dtype", "stats", "histogram"}, (
            f"Unexpected top-level keys in PixelSummary.to_dict(): {set(d.keys())!r}"
        )

    def test_to_dict_stats_nested_shape(self):
        s = self._make_summary()
        d = s.to_dict()
        assert set(d["stats"].keys()) == {"min", "max", "mean", "nan_count", "inf_count"}, (
            f"Unexpected keys in to_dict()['stats']: {set(d['stats'].keys())!r}"
        )

    def test_to_dict_histogram_nested_shape(self):
        s = self._make_summary()
        d = s.to_dict()
        assert set(d["histogram"].keys()) == {"bins", "counts"}, (
            f"Unexpected keys in to_dict()['histogram']: {set(d['histogram'].keys())!r}"
        )

    def test_to_dict_stats_values(self):
        s = self._make_summary(nan_count=2, inf_count=1)
        d = s.to_dict()
        assert d["stats"]["min"] == [-1.0, -1.0, -1.0]
        assert d["stats"]["max"] == [1.0, 1.0, 1.0]
        assert d["stats"]["mean"] == [0.0, 0.0, 0.0]
        assert d["stats"]["nan_count"] == 2
        assert d["stats"]["inf_count"] == 1

    def test_to_dict_histogram_values(self):
        s = self._make_summary(histogram_bins=8)
        d = s.to_dict()
        assert d["histogram"]["bins"] == 8
        assert len(d["histogram"]["counts"]) == 3

    def test_to_dict_is_json_serialisable(self):
        s = self._make_summary()
        json.dumps(s.to_dict())

    def test_min_max_mean_can_contain_none_for_all_nan_channel(self):
        """An all-nan channel reports min/max/mean as None (sentinel, per lockedFieldContract)."""
        s = self._make_summary(min=[None, -1.0, -1.0], max=[None, 1.0, 1.0], mean=[None, 0.0, 0.0], nan_count=4)
        d = s.to_dict()
        assert d["stats"]["min"][0] is None
        assert d["stats"]["max"][0] is None
        assert d["stats"]["mean"][0] is None
        json.dumps(d)


# ===========================================================================
# Section 4 — ReadbackPage dataclass
#
# Locked contract:
#   ReadbackPage(plane: str, mode: str, pixels: list, page: int, page_size: int,
#                total_pages: int, truncated: bool)
#   mode in {"summary","roi","sample"} — __post_init__ raises ValueError otherwise.
#   to_dict() -> {plane,mode,pixels,page,page_size,total_pages,truncated}
# ===========================================================================

class TestReadbackPage:
    """ReadbackPage dataclass — mode validation and to_dict() contract."""

    def _make_page(self, **kwargs):
        defaults = dict(
            plane="beauty",
            mode="roi",
            pixels=[[0.5, 0.3, 0.1], [0.4, 0.2, 0.1]],
            page=0,
            page_size=2,
            total_pages=1,
            truncated=False,
        )
        defaults.update(kwargs)
        return ReadbackPage(**defaults)

    def test_summary_mode_is_valid(self):
        p = self._make_page(mode="summary", pixels=[])
        assert p.mode == "summary"

    def test_roi_mode_is_valid(self):
        p = self._make_page(mode="roi")
        assert p.mode == "roi"

    def test_sample_mode_is_valid(self):
        p = self._make_page(mode="sample")
        assert p.mode == "sample"

    def test_invalid_mode_raises(self):
        """M-4-style pattern: an unrecognised mode must raise ValueError at construction."""
        with pytest.raises(ValueError):
            self._make_page(mode="bogus")

    def test_invalid_mode_full_frame_raises(self):
        with pytest.raises(ValueError):
            self._make_page(mode="full")

    def test_invalid_mode_empty_string_raises(self):
        with pytest.raises(ValueError):
            self._make_page(mode="")

    def test_to_dict_keys(self):
        p = self._make_page()
        d = p.to_dict()
        assert set(d.keys()) == {
            "plane", "mode", "pixels", "page", "page_size", "total_pages", "truncated",
        }, f"Unexpected keys in ReadbackPage.to_dict(): {set(d.keys())!r}"

    def test_to_dict_values(self):
        p = self._make_page(page=1, page_size=4, total_pages=3, truncated=True)
        d = p.to_dict()
        assert d["page"] == 1
        assert d["page_size"] == 4
        assert d["total_pages"] == 3
        assert d["truncated"] is True

    def test_to_dict_is_json_serialisable(self):
        p = self._make_page()
        json.dumps(p.to_dict())


# ===========================================================================
# Section 5 — guess_layout(shape) -> str
#
# Locked contract (spec §7.2 + §9.1):
#   [1,3,512,512]           -> "NCHW"
#   [1,512,512,3]           -> "NHWC"
#   [1,256]                 -> "unknown"
#   [1,3,'dynamic','dynamic'] -> "NCHW" (dynamic image dims count as large)
#   rank != 4, or a channel-count that doesn't sit at index 1 or -1 -> "unknown"
#   never raises on odd shapes.
# ===========================================================================

class TestGuessLayout:
    """guess_layout — pinned NCHW/NHWC/unknown cases per spec §9.1."""

    def test_nchw_pinned_case(self):
        assert guess_layout([1, 3, 512, 512]) == "NCHW"

    def test_nhwc_pinned_case(self):
        assert guess_layout([1, 512, 512, 3]) == "NHWC"

    def test_rank2_unknown_pinned_case(self):
        assert guess_layout([1, 256]) == "unknown"

    def test_nchw_with_dynamic_image_dims(self):
        """Dynamic image dims count as 'large' — still resolves to NCHW."""
        assert guess_layout([1, 3, "dynamic", "dynamic"]) == "NCHW"

    def test_nhwc_with_dynamic_image_dims(self):
        assert guess_layout([1, "dynamic", "dynamic", 3]) == "NHWC"

    def test_nchw_single_channel(self):
        """Channel count of 1 at index 1 (grayscale, NCHW)."""
        assert guess_layout([1, 1, 256, 256]) == "NCHW"

    def test_nchw_four_channel_rgba(self):
        """Channel count of 4 (RGBA) at index 1 -> NCHW."""
        assert guess_layout([1, 4, 256, 256]) == "NCHW"

    def test_nhwc_single_channel(self):
        assert guess_layout([1, 256, 256, 1]) == "NHWC"

    def test_rank3_is_unknown(self):
        """Rank != 4 -> unknown."""
        assert guess_layout([3, 512, 512]) == "unknown"

    def test_rank1_is_unknown(self):
        assert guess_layout([1000]) == "unknown"

    def test_rank5_is_unknown(self):
        assert guess_layout([1, 3, 4, 512, 512]) == "unknown"

    def test_ambiguous_channel_position_is_unknown(self):
        """A shape where neither index 1 nor index -1 is a plausible channel count -> unknown."""
        assert guess_layout([1, 512, 512, 512]) == "unknown"

    def test_does_not_raise_on_odd_shapes(self):
        """guess_layout must never raise — always degrade to 'unknown'."""
        # Empty shape
        assert guess_layout([]) == "unknown"
        # All-dynamic shape
        assert guess_layout(["dynamic", "dynamic", "dynamic", "dynamic"]) == "unknown" or \
            guess_layout(["dynamic", "dynamic", "dynamic", "dynamic"]) in ("NCHW", "NHWC", "unknown")
        # Negative-ish / zero dims (shouldn't crash)
        assert guess_layout([1, 3, 0, 0]) in ("NCHW", "NHWC", "unknown")

    def test_returns_string(self):
        result = guess_layout([1, 3, 512, 512])
        assert isinstance(result, str)


# ===========================================================================
# Section 6 — clamp_readback(xres, yres, channels, max_pixels) -> int
#
# Locked contract:
#   Returns a positive integer stride s such that
#     ceil(xres/s) * ceil(yres/s) <= max_pixels
#   stride == 1 when the full plane already fits.
#   Monotonic: bigger plane or smaller max_pixels -> stride >= previous stride.
#   max_pixels <= 0 or non-positive dims -> ValueError.
# ===========================================================================

class TestClampReadback:
    """clamp_readback — server-side stride computation for context-safe readback."""

    def test_full_plane_fits_stride_one(self):
        """A small plane under max_pixels needs no downsampling -> stride 1."""
        assert clamp_readback(xres=32, yres=32, channels=3, max_pixels=4096) == 1

    def test_large_plane_stride_keeps_within_budget(self):
        """Pinned spec case: a 4096x4096 plane with max_pixels=4096 -> strided count <= 4096."""
        stride = clamp_readback(xres=4096, yres=4096, channels=3, max_pixels=4096)
        assert stride >= 1
        strided_count = math.ceil(4096 / stride) * math.ceil(4096 / stride)
        assert strided_count <= 4096, (
            f"clamp_readback stride={stride} yields {strided_count} pixels, "
            f"exceeding max_pixels=4096"
        )

    def test_stride_is_positive_integer(self):
        stride = clamp_readback(xres=1024, yres=1024, channels=3, max_pixels=1000)
        assert isinstance(stride, int)
        assert stride >= 1

    def test_monotonic_bigger_plane_needs_ge_stride(self):
        """A bigger plane (same max_pixels) requires a stride >= the smaller plane's stride."""
        small_stride = clamp_readback(xres=512, yres=512, channels=3, max_pixels=4096)
        big_stride = clamp_readback(xres=4096, yres=4096, channels=3, max_pixels=4096)
        assert big_stride >= small_stride

    def test_monotonic_smaller_max_pixels_needs_ge_stride(self):
        """A smaller max_pixels budget (same plane) requires a stride >= the looser budget's."""
        loose_stride = clamp_readback(xres=2048, yres=2048, channels=3, max_pixels=100000)
        tight_stride = clamp_readback(xres=2048, yres=2048, channels=3, max_pixels=1000)
        assert tight_stride >= loose_stride

    def test_max_pixels_zero_raises(self):
        with pytest.raises(ValueError):
            clamp_readback(xres=100, yres=100, channels=3, max_pixels=0)

    def test_max_pixels_negative_raises(self):
        with pytest.raises(ValueError):
            clamp_readback(xres=100, yres=100, channels=3, max_pixels=-10)

    def test_xres_zero_raises(self):
        with pytest.raises(ValueError):
            clamp_readback(xres=0, yres=100, channels=3, max_pixels=4096)

    def test_yres_negative_raises(self):
        with pytest.raises(ValueError):
            clamp_readback(xres=100, yres=-1, channels=3, max_pixels=4096)

    def test_strided_pixel_count_never_exceeds_budget_various_sizes(self):
        """General invariant across several plane sizes: strided count always <= max_pixels."""
        for xres, yres, max_px in [(100, 100, 50), (2000, 1000, 500), (8192, 4320, 4096)]:
            stride = clamp_readback(xres=xres, yres=yres, channels=3, max_pixels=max_px)
            strided_count = math.ceil(xres / stride) * math.ceil(yres / stride)
            assert strided_count <= max_px, (
                f"xres={xres} yres={yres} max_pixels={max_px}: stride={stride} "
                f"yields {strided_count} > {max_px}"
            )


# ===========================================================================
# Section 7 — paginate(total_items, page, page_size) -> dict
#
# Locked contract:
#   Returns {page, page_size, total_pages, start, end, truncated}
#   total_pages = ceil(total_items/page_size), >=1, and 1 when total_items==0.
#   start = page*page_size; end = min(start+page_size, total_items).
#   truncated = (end < total_items) OR (page < total_pages-1).
#   page out of range (page<0 or page>=total_pages when total_items>0) -> ValueError.
# ===========================================================================

class TestPaginate:
    """paginate — page-boundary math per spec §9.1 pinned case (2500 @ 1024)."""

    def test_pinned_total_pages(self):
        """2500 items @ page_size 1024 -> total_pages == 3."""
        result = paginate(total_items=2500, page=0, page_size=1024)
        assert result["total_pages"] == 3, (
            f"2500 items @ page_size 1024 must give total_pages=3, got {result['total_pages']}"
        )

    def test_pinned_page_2_start_end(self):
        """page 2 -> start 2048, end 2500 (last partial page)."""
        result = paginate(total_items=2500, page=2, page_size=1024)
        assert result["start"] == 2048
        assert result["end"] == 2500

    def test_page_0_start_end(self):
        result = paginate(total_items=2500, page=0, page_size=1024)
        assert result["start"] == 0
        assert result["end"] == 1024

    def test_page_1_start_end(self):
        result = paginate(total_items=2500, page=1, page_size=1024)
        assert result["start"] == 1024
        assert result["end"] == 2048

    def test_page_0_truncated_true_when_more_pages(self):
        """page 0 -> truncated True because page < total_pages-1."""
        result = paginate(total_items=2500, page=0, page_size=1024)
        assert result["truncated"] is True

    def test_last_page_truncated_false(self):
        """The final page (page 2 of 3) -> truncated False."""
        result = paginate(total_items=2500, page=2, page_size=1024)
        assert result["truncated"] is False

    def test_total_items_zero_gives_total_pages_one(self):
        """total_items==0 -> total_pages==1 (never 0)."""
        result = paginate(total_items=0, page=0, page_size=1024)
        assert result["total_pages"] == 1

    def test_total_items_zero_start_end(self):
        result = paginate(total_items=0, page=0, page_size=1024)
        assert result["start"] == 0
        assert result["end"] == 0

    def test_single_full_page(self):
        """total_items == page_size exactly -> total_pages == 1."""
        result = paginate(total_items=1024, page=0, page_size=1024)
        assert result["total_pages"] == 1
        assert result["end"] == 1024
        assert result["truncated"] is False

    def test_page_and_page_size_reflected(self):
        result = paginate(total_items=2500, page=1, page_size=1024)
        assert result["page"] == 1
        assert result["page_size"] == 1024

    def test_page_out_of_range_negative_raises(self):
        with pytest.raises(ValueError):
            paginate(total_items=2500, page=-1, page_size=1024)

    def test_page_out_of_range_too_large_raises(self):
        """3 total pages (0,1,2) -> page=3 is out of range."""
        with pytest.raises(ValueError):
            paginate(total_items=2500, page=3, page_size=1024)

    def test_page_zero_valid_when_zero_items(self):
        """page=0 must be valid even when total_items==0 (single empty page)."""
        result = paginate(total_items=0, page=0, page_size=1024)
        assert result["page"] == 0

    def test_returns_dict_with_expected_keys(self):
        result = paginate(total_items=100, page=0, page_size=10)
        assert set(result.keys()) == {"page", "page_size", "total_pages", "start", "end", "truncated"}, (
            f"Unexpected keys from paginate(): {set(result.keys())!r}"
        )

    def test_is_json_serialisable(self):
        result = paginate(total_items=2500, page=2, page_size=1024)
        json.dumps(result)


# ===========================================================================
# Section 8 — pixel nan/inf-safe stats helper (count_nan_inf)
#
# Locked contract:
#   Given a flat list of floats (or a list of per-channel lists), counts NaN and
#   +/-Inf occurrences WITHOUT dropping them; computes min/max/mean IGNORING
#   nan/inf. An all-nan channel -> min/max/mean None + nan_count == len.
# ===========================================================================

class TestCountNanInf:
    """count_nan_inf — nan/inf-honest stats helper (FR-6: never silently dropped)."""

    def test_clean_list_no_nan_no_inf(self):
        result = count_nan_inf([0.1, 0.2, 0.3, 0.4])
        assert result["nan_count"] == 0
        assert result["inf_count"] == 0

    def test_nan_counted_not_dropped(self):
        """A NaN in the input must be counted, not silently dropped from the stats."""
        values = [0.5, float("nan"), 0.5, 0.5]
        result = count_nan_inf(values)
        assert result["nan_count"] == 1, (
            f"NaN must be counted (FR-6 — never silently dropped), got nan_count={result['nan_count']}"
        )

    def test_positive_inf_counted(self):
        values = [0.5, float("inf"), 0.5]
        result = count_nan_inf(values)
        assert result["inf_count"] == 1

    def test_negative_inf_counted(self):
        values = [0.5, float("-inf"), 0.5]
        result = count_nan_inf(values)
        assert result["inf_count"] == 1

    def test_both_pos_and_neg_inf_counted(self):
        values = [float("inf"), float("-inf"), 0.5]
        result = count_nan_inf(values)
        assert result["inf_count"] == 2

    def test_nan_and_inf_together(self):
        """A stub list with both NaN and +/-Inf reports the correct counts for each."""
        values = [0.1, float("nan"), float("inf"), float("-inf"), 0.2]
        result = count_nan_inf(values)
        assert result["nan_count"] == 1
        assert result["inf_count"] == 2

    def test_min_max_mean_ignore_nan_and_inf(self):
        """min/max/mean must be computed over the FINITE values only."""
        values = [0.0, 1.0, float("nan"), float("inf"), 0.5]
        result = count_nan_inf(values)
        assert result["min"] == 0.0
        assert result["max"] == 1.0
        assert abs(result["mean"] - 0.5) < 1e-6

    def test_all_nan_channel_min_max_mean_none(self):
        """An all-NaN input -> min/max/mean are None (sentinel), nan_count == len."""
        values = [float("nan"), float("nan"), float("nan")]
        result = count_nan_inf(values)
        assert result["min"] is None
        assert result["max"] is None
        assert result["mean"] is None
        assert result["nan_count"] == 3

    def test_all_inf_channel_min_max_mean_none(self):
        """An all-Inf input (no finite values) -> min/max/mean are None."""
        values = [float("inf"), float("-inf"), float("inf")]
        result = count_nan_inf(values)
        assert result["min"] is None
        assert result["max"] is None
        assert result["mean"] is None
        assert result["inf_count"] == 3

    def test_empty_list_min_max_mean_none(self):
        result = count_nan_inf([])
        assert result["min"] is None
        assert result["max"] is None
        assert result["mean"] is None
        assert result["nan_count"] == 0
        assert result["inf_count"] == 0

    def test_known_mean_value(self):
        values = [0.0, 1.0, 0.0, 1.0]
        result = count_nan_inf(values)
        assert abs(result["mean"] - 0.5) < 1e-6

    def test_result_is_json_serialisable_with_finite_values(self):
        values = [0.1, 0.2, 0.3]
        result = count_nan_inf(values)
        json.dumps(result)

    def test_result_is_json_serialisable_with_all_nan(self):
        """None sentinels (not NaN/Inf floats) keep the result JSON-safe."""
        values = [float("nan"), float("nan")]
        result = count_nan_inf(values)
        json.dumps(result)

    def test_stats_helper_works_on_plain_python_lists(self):
        """count_nan_inf must operate on plain Python lists — no numpy array required as input."""
        values = list([0.1, 0.2, float("nan"), 0.3])  # a genuine plain list, not an array-like
        result = count_nan_inf(values)
        assert result["nan_count"] == 1
        # The real numpy-free purity check (no `import numpy` inside cop_onnx_model.py)
        # is asserted in TestModulePurity below via source inspection.


# ===========================================================================
# Section 9 — Module purity (CL-015): no hou/Qt/pxr/numpy/MaterialX imports
# ===========================================================================

class TestModulePurity:
    """cop_onnx_model.py must import NO hou, Qt/PySide6, pxr, numpy, or MaterialX (CL-015)."""

    def test_source_has_no_forbidden_imports(self):
        import fxhoudinimcp.cop_onnx_model as mod
        src_path = mod.__file__
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        forbidden_tokens = ["import hou", "import PySide6", "import PySide2", "import pxr", "import numpy", "import MaterialX"]
        for token in forbidden_tokens:
            assert token not in source, (
                f"cop_onnx_model.py must not contain {token!r} (CL-015 purity boundary)"
            )


# ===========================================================================
# Section 10 — contract_from_setup_shapes(model_path, raw_inputs, raw_outputs,
#                                          opset=None, producer=None) -> OnnxContract
#              (PP12-113 PR-2 — RED, does not exist yet)
#
# Locked contract (plan pp12-113b lockedFieldContract):
#   NEW pure function APPENDED to cop_onnx_model.py. PR-1 symbols (above)
#   are BYTE-UNCHANGED — this section only ADDS coverage for the new symbol.
#
#   Signature: contract_from_setup_shapes(model_path: str, raw_inputs: list,
#                                          raw_outputs: list, opset=None,
#                                          producer=None) -> OnnxContract
#
#   raw_inputs / raw_outputs are plain dicts the HANDLER assembles from node
#   parms: [{'name': str, 'dtype': str|None, 'shape': list[int|'dynamic']}, ...]
#
#   Behavior:
#     - Builds a TensorSpec per raw entry.
#     - INPUT TensorSpecs get layout_guess = guess_layout(shape).
#     - OUTPUT TensorSpecs get layout_guess = 'unknown' (fixed, not guessed).
#     - dtype passed through verbatim from the raw dict's 'dtype' key.
#     - Dynamic dims (the literal string 'dynamic') preserved verbatim.
#     - Sets loadable=True, error=None on the returned OnnxContract.
#     - opset / producer forwarded verbatim (default None).
#     - to_dict() on the result round-trips per the existing OnnxContract
#       contract (Section 2 above) — inputs/outputs as lists of dicts.
#
#   This is a SYNTHETIC-input pure test: independent of any live Houdini
#   parm read. The handler (hou-dev, hython-smoke) owns normalizing the
#   ACTUAL parm sentinel (-1/0/blank) to the 'dynamic' literal BEFORE
#   calling this helper — that normalization is out of scope here.
# ===========================================================================

class TestContractFromSetupShapes:
    """contract_from_setup_shapes — pure raw-dict -> OnnxContract mapping (RED)."""

    def _raw_input_nchw_dynamic(self):
        return {
            "name": "input",
            "dtype": "float32",
            "shape": [1, 3, "dynamic", "dynamic"],
        }

    def _raw_output_basic(self):
        return {
            "name": "output",
            "dtype": "float32",
            "shape": [1, 3, "dynamic", "dynamic"],
        }

    def test_module_exposes_contract_from_setup_shapes(self):
        """RED GATE: contract_from_setup_shapes must be importable from cop_onnx_model."""
        from fxhoudinimcp.cop_onnx_model import contract_from_setup_shapes  # noqa: F401
        assert callable(contract_from_setup_shapes)

    def test_returns_onnx_contract_instance(self):
        from fxhoudinimcp.cop_onnx_model import contract_from_setup_shapes

        result = contract_from_setup_shapes(
            model_path="/models/identity.onnx",
            raw_inputs=[self._raw_input_nchw_dynamic()],
            raw_outputs=[self._raw_output_basic()],
        )
        assert isinstance(result, OnnxContract), (
            f"contract_from_setup_shapes must return an OnnxContract, got {type(result)!r}"
        )

    def test_model_path_forwarded(self):
        from fxhoudinimcp.cop_onnx_model import contract_from_setup_shapes

        result = contract_from_setup_shapes(
            model_path="/models/identity.onnx",
            raw_inputs=[self._raw_input_nchw_dynamic()],
            raw_outputs=[self._raw_output_basic()],
        )
        assert result.model_path == "/models/identity.onnx"

    def test_input_layout_guess_matches_guess_layout(self):
        """Input TensorSpec.layout_guess must equal guess_layout(shape) — pinned NCHW case."""
        from fxhoudinimcp.cop_onnx_model import contract_from_setup_shapes

        raw_input = self._raw_input_nchw_dynamic()
        expected_layout = guess_layout(raw_input["shape"])
        assert expected_layout == "NCHW", (
            "sanity: [1,3,'dynamic','dynamic'] must guess NCHW per Section 5 pinned case"
        )

        result = contract_from_setup_shapes(
            model_path="/models/identity.onnx",
            raw_inputs=[raw_input],
            raw_outputs=[self._raw_output_basic()],
        )
        assert len(result.inputs) == 1
        assert result.inputs[0].layout_guess == expected_layout, (
            f"input TensorSpec.layout_guess must equal guess_layout(shape) == {expected_layout!r}, "
            f"got {result.inputs[0].layout_guess!r}"
        )

    def test_output_layout_guess_is_always_unknown(self):
        """Output TensorSpecs get layout_guess='unknown' regardless of shape (fixed, not guessed)."""
        from fxhoudinimcp.cop_onnx_model import contract_from_setup_shapes

        # Use a shape that WOULD guess NCHW on an input, to prove outputs are NOT guessed.
        raw_output = {
            "name": "output",
            "dtype": "float32",
            "shape": [1, 3, 512, 512],
        }
        result = contract_from_setup_shapes(
            model_path="/models/identity.onnx",
            raw_inputs=[self._raw_input_nchw_dynamic()],
            raw_outputs=[raw_output],
        )
        assert len(result.outputs) == 1
        assert result.outputs[0].layout_guess == "unknown", (
            f"output TensorSpec.layout_guess must always be 'unknown', "
            f"got {result.outputs[0].layout_guess!r}"
        )

    def test_dynamic_dim_preserved_verbatim(self):
        """The literal 'dynamic' string in a raw shape must survive into the TensorSpec."""
        from fxhoudinimcp.cop_onnx_model import contract_from_setup_shapes

        result = contract_from_setup_shapes(
            model_path="/models/identity.onnx",
            raw_inputs=[self._raw_input_nchw_dynamic()],
            raw_outputs=[self._raw_output_basic()],
        )
        assert result.inputs[0].shape == [1, 3, "dynamic", "dynamic"], (
            f"dynamic dims must be preserved verbatim, got {result.inputs[0].shape!r}"
        )
        assert result.outputs[0].shape == [1, 3, "dynamic", "dynamic"], (
            f"dynamic dims must be preserved verbatim in outputs too, got {result.outputs[0].shape!r}"
        )

    def test_dtype_passthrough_from_raw_dict(self):
        """dtype must pass through verbatim from the raw dict's 'dtype' key."""
        from fxhoudinimcp.cop_onnx_model import contract_from_setup_shapes

        raw_input = dict(self._raw_input_nchw_dynamic())
        raw_input["dtype"] = "float16"
        raw_output = dict(self._raw_output_basic())
        raw_output["dtype"] = "int64"

        result = contract_from_setup_shapes(
            model_path="/models/identity.onnx",
            raw_inputs=[raw_input],
            raw_outputs=[raw_output],
        )
        assert result.inputs[0].dtype == "float16", (
            f"input dtype must pass through verbatim, got {result.inputs[0].dtype!r}"
        )
        assert result.outputs[0].dtype == "int64", (
            f"output dtype must pass through verbatim, got {result.outputs[0].dtype!r}"
        )

    def test_name_passthrough_from_raw_dict(self):
        from fxhoudinimcp.cop_onnx_model import contract_from_setup_shapes

        result = contract_from_setup_shapes(
            model_path="/models/identity.onnx",
            raw_inputs=[self._raw_input_nchw_dynamic()],
            raw_outputs=[self._raw_output_basic()],
        )
        assert result.inputs[0].name == "input"
        assert result.outputs[0].name == "output"

    def test_loadable_true_error_none(self):
        """A successful build must set loadable=True, error=None."""
        from fxhoudinimcp.cop_onnx_model import contract_from_setup_shapes

        result = contract_from_setup_shapes(
            model_path="/models/identity.onnx",
            raw_inputs=[self._raw_input_nchw_dynamic()],
            raw_outputs=[self._raw_output_basic()],
        )
        assert result.loadable is True
        assert result.error is None

    def test_opset_and_producer_forwarded(self):
        """opset/producer are forwarded verbatim; default to None when omitted."""
        from fxhoudinimcp.cop_onnx_model import contract_from_setup_shapes

        result_defaults = contract_from_setup_shapes(
            model_path="/models/identity.onnx",
            raw_inputs=[self._raw_input_nchw_dynamic()],
            raw_outputs=[self._raw_output_basic()],
        )
        assert result_defaults.opset is None
        assert result_defaults.producer is None

        result_set = contract_from_setup_shapes(
            model_path="/models/identity.onnx",
            raw_inputs=[self._raw_input_nchw_dynamic()],
            raw_outputs=[self._raw_output_basic()],
            opset=17,
            producer="pytorch",
        )
        assert result_set.opset == 17
        assert result_set.producer == "pytorch"

    def test_multiple_inputs_and_outputs_all_mapped(self):
        """Multiple raw entries all become TensorSpecs, order preserved."""
        from fxhoudinimcp.cop_onnx_model import contract_from_setup_shapes

        raw_inputs = [
            {"name": "input_a", "dtype": "float32", "shape": [1, 3, "dynamic", "dynamic"]},
            {"name": "input_b", "dtype": "float32", "shape": [1, 256]},
        ]
        raw_outputs = [
            {"name": "output_a", "dtype": "float32", "shape": [1, 1000]},
            {"name": "output_b", "dtype": "int64", "shape": [1]},
        ]
        result = contract_from_setup_shapes(
            model_path="/models/multi.onnx",
            raw_inputs=raw_inputs,
            raw_outputs=raw_outputs,
        )
        assert [t.name for t in result.inputs] == ["input_a", "input_b"]
        assert [t.name for t in result.outputs] == ["output_a", "output_b"]
        # input_b is rank-2 -> guess_layout degrades to 'unknown'
        assert result.inputs[1].layout_guess == "unknown"

    def test_empty_inputs_and_outputs_produce_empty_lists(self):
        """A model with no inputs/outputs (edge case) still returns valid empty lists."""
        from fxhoudinimcp.cop_onnx_model import contract_from_setup_shapes

        result = contract_from_setup_shapes(
            model_path="/models/empty.onnx",
            raw_inputs=[],
            raw_outputs=[],
        )
        assert result.inputs == []
        assert result.outputs == []

    def test_to_dict_round_trips(self):
        """The resulting OnnxContract.to_dict() must use the existing OnnxContract contract."""
        from fxhoudinimcp.cop_onnx_model import contract_from_setup_shapes

        result = contract_from_setup_shapes(
            model_path="/models/identity.onnx",
            raw_inputs=[self._raw_input_nchw_dynamic()],
            raw_outputs=[self._raw_output_basic()],
            opset=17,
            producer="pytorch",
        )
        d = result.to_dict()
        assert set(d.keys()) == {
            "model_path", "inputs", "outputs", "opset", "producer", "loadable", "error",
        }
        assert d["model_path"] == "/models/identity.onnx"
        assert d["opset"] == 17
        assert d["producer"] == "pytorch"
        assert d["loadable"] is True
        assert d["error"] is None
        assert d["inputs"][0]["shape"] == [1, 3, "dynamic", "dynamic"]
        assert d["inputs"][0]["layout_guess"] == "NCHW"
        assert d["outputs"][0]["layout_guess"] == "unknown"
        json.dumps(d)

    def test_is_json_serialisable_end_to_end(self):
        from fxhoudinimcp.cop_onnx_model import contract_from_setup_shapes

        result = contract_from_setup_shapes(
            model_path="/models/identity.onnx",
            raw_inputs=[self._raw_input_nchw_dynamic()],
            raw_outputs=[self._raw_output_basic()],
        )
        json.dumps(result.to_dict())


# ===========================================================================
# Section 11 — choose_provider(requested: str, available: list) -> tuple
#              (PP12-113 PR-3 — RED, does not exist yet)
#
# Locked contract (plan pp12-113c lockedFieldContract, FOLD m2-provider-edge):
#   NEW pure function APPENDED to cop_onnx_model.py. PR-1/PR-2 symbols
#   (above) are BYTE-UNCHANGED — this section only ADDS coverage for the
#   new symbol.
#
#   Signature: choose_provider(requested: str, available: list) -> tuple
#              -> (will_bind: str, warning: str | None)
#
#   Behavior (pure — never touches hou/Qt/pxr; the HANDLER is responsible
#   for reading node.parm('provider').menuItems() at RUNTIME and passing
#   it in as `available`):
#     - requested (case-INSENSITIVE) IS in available
#         -> (requested.lower(), None)                    -- no warning
#     - requested NOT in available AND 'automatic' IS in available
#         -> ('automatic', <non-empty warning str>)        -- safe fallback
#     - requested NOT in available AND 'automatic' NOT in available
#         -> (available[0], <non-empty warning str>)       -- first-available fallback
#     - available == [] (no Execution Provider options at all)
#         -> raises ValueError                              -- the ONLY raise
#            case; the handler maps this to
#            {ok: False, error: 'onnx node exposes no Execution Provider options'}
#     - NEVER raises merely because `requested` is unavailable (FR-4) —
#       raising is reserved EXCLUSIVELY for the available==[] case.
#
#   This is a SYNTHETIC-input pure test: independent of any live Houdini
#   parm read. The handler (hou-dev, hython-smoke) owns calling
#   node.parm('provider').menuItems() to build `available` at runtime —
#   that read is out of scope here.
# ===========================================================================

class TestChooseProvider:
    """choose_provider — pure requested/available -> (will_bind, warning) mapping (RED)."""

    # A realistic platform-filtered available list (Windows, per the
    # Phase-0 live probe): lowercase, no coreml.
    _WINDOWS_AVAILABLE = ["automatic", "cpu", "cuda", "directml"]

    def test_module_exposes_choose_provider(self):
        """RED GATE: choose_provider must be importable from cop_onnx_model."""
        from fxhoudinimcp.cop_onnx_model import choose_provider  # noqa: F401
        assert callable(choose_provider)

    def test_returns_a_two_tuple(self):
        from fxhoudinimcp.cop_onnx_model import choose_provider

        result = choose_provider("cuda", self._WINDOWS_AVAILABLE)
        assert isinstance(result, tuple), f"choose_provider must return a tuple, got {type(result)!r}"
        assert len(result) == 2, f"choose_provider must return a 2-tuple, got len={len(result)!r}"

    def test_requested_available_no_warning(self):
        """requested is in available (exact case) -> (requested.lower(), None)."""
        from fxhoudinimcp.cop_onnx_model import choose_provider

        will_bind, warning = choose_provider("cuda", self._WINDOWS_AVAILABLE)
        assert will_bind == "cuda"
        assert warning is None

    def test_requested_available_case_insensitive(self):
        """requested matching is case-INSENSITIVE — 'CUDA' matches 'cuda' in available."""
        from fxhoudinimcp.cop_onnx_model import choose_provider

        will_bind, warning = choose_provider("CUDA", self._WINDOWS_AVAILABLE)
        assert will_bind == "cuda", (
            f"a case-insensitive match must bind the LOWERCASE available token, got {will_bind!r}"
        )
        assert warning is None

    def test_requested_available_mixed_case_variants(self):
        """Mixed-case requested strings all resolve to the same lowercase match."""
        from fxhoudinimcp.cop_onnx_model import choose_provider

        for variant in ("DirectML", "directml", "DIRECTML", "DirectMl"):
            will_bind, warning = choose_provider(variant, self._WINDOWS_AVAILABLE)
            assert will_bind == "directml", f"variant {variant!r} must resolve to 'directml', got {will_bind!r}"
            assert warning is None, f"variant {variant!r} must not produce a warning, got {warning!r}"

    def test_requested_unavailable_falls_back_to_automatic_when_present(self):
        """requested not in available, but 'automatic' IS -> ('automatic', <warning>)."""
        from fxhoudinimcp.cop_onnx_model import choose_provider

        will_bind, warning = choose_provider("tensorrt", self._WINDOWS_AVAILABLE)
        assert will_bind == "automatic"
        assert isinstance(warning, str) and warning, (
            f"a non-empty warning string is required on fallback, got {warning!r}"
        )

    def test_requested_unavailable_falls_back_to_first_available_when_no_automatic(self):
        """requested not in available AND 'automatic' NOT in available
        -> (available[0], <warning>)."""
        from fxhoudinimcp.cop_onnx_model import choose_provider

        available_no_automatic = ["cpu", "cuda", "directml"]
        will_bind, warning = choose_provider("tensorrt", available_no_automatic)
        assert will_bind == "cpu", (
            f"must fall back to available[0] ('cpu') when 'automatic' is not present, got {will_bind!r}"
        )
        assert isinstance(warning, str) and warning, (
            f"a non-empty warning string is required on fallback, got {warning!r}"
        )

    def test_requested_unavailable_never_raises(self):
        """FR-4: choose_provider must NEVER raise merely because `requested`
        is unavailable — raising is reserved exclusively for available==[]."""
        from fxhoudinimcp.cop_onnx_model import choose_provider

        # Should not raise for any of these unavailable-but-non-empty cases.
        choose_provider("tensorrt", self._WINDOWS_AVAILABLE)
        choose_provider("coreml", ["cpu", "cuda"])
        choose_provider("nonexistent_provider_xyz", ["automatic"])

    def test_empty_available_raises_value_error(self):
        """available == [] (no Execution Provider options at all) -> raises
        ValueError. This is the ONLY raise case."""
        from fxhoudinimcp.cop_onnx_model import choose_provider

        with pytest.raises(ValueError):
            choose_provider("cuda", [])

    def test_automatic_requested_directly(self):
        """Requesting 'automatic' directly (already in available) resolves
        cleanly with no warning — not treated as a fallback case."""
        from fxhoudinimcp.cop_onnx_model import choose_provider

        will_bind, warning = choose_provider("automatic", self._WINDOWS_AVAILABLE)
        assert will_bind == "automatic"
        assert warning is None

    def test_single_option_available_list(self):
        """A single-entry available list (edge case, non-empty) still works
        without raising."""
        from fxhoudinimcp.cop_onnx_model import choose_provider

        will_bind, warning = choose_provider("cpu", ["cpu"])
        assert will_bind == "cpu"
        assert warning is None

        will_bind, warning = choose_provider("cuda", ["cpu"])
        assert will_bind == "cpu"
        assert isinstance(warning, str) and warning
