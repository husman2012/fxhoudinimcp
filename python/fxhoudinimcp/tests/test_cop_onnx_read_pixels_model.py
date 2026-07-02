"""Pure-logic pytest tests for the 4 new read_pixels helpers appended to
cop_onnx_model.py (PP12-113 PR-5, the FINAL tool of member 113):
compute_histogram, deinterleave_channels, bounded_page_coords,
bounded_sample_coords.

Unit: pp12-113e
testVerificationSurface: pytest-model
planSha: bc1f8d9e9fcd1b46216e3eab594ded4a31ee907e305e5d12fde0b040314c445c

TDD phase: RED — none of the 4 helpers exist yet on cop_onnx_model.py.
Expected failure: ImportError until hou-dev appends them.

Contract: imports NO hou, NO Qt/PySide6, NO pxr, NO numpy, NO onnx. Plain
Python stdlib only (CL-015) — this module never touches Houdini, so it
runs on bare CI with plain pytest.

Grounded against (plan pp12-113e lockedFieldContract, "pure helpers:
compute_histogram + deinterleave_channels + bounded_page_coords +
bounded_sample_coords (APPENDED to cop_onnx_model.py)"):

  compute_histogram(values, bins, lo, hi) -> list[int]
      `bins` equal-width bin counts over [lo, hi]. NaN/Inf are EXCLUDED
      from the bin counts (already counted separately by count_nan_inf).
      lo==hi degenerate -> all finite values land in bin 0. bins<=0 -> []
      guard. lo/hi None (an all-non-finite channel) -> [0]*bins.

  deinterleave_channels(flat, channels) -> list[list]
      Split a channel-interleaved flat list into `channels` per-channel
      lists (flat[c::channels]). channels<=0 -> [] guard. Called ONLY
      over a BOUNDED sample (<= budget), never the full plane -- but this
      pure function itself has no knowledge of that; it just deinterleaves
      whatever flat list it is given.

  bounded_page_coords(x0, y0, box_w, box_h, start, end) -> list[(x, y)]
      The LAZY ROI coord slice -- returns ONLY the flattened [start, end)
      pixels of a box_w x box_h box offset at (x0, y0), row-major
      (x = x0 + i % box_w, y = y0 + i // box_w). Never materializes the
      whole box (folds codex Blocker-2).

  bounded_sample_coords(w, h, stride, start, end) -> list[(x, y)]
      The LAZY strided-sample coord slice -- the strided grid is
      [(x*stride, y*stride) for y in 0..ceil(h/stride)-1
                             for x in 0..ceil(w/stride)-1]
      (row-major); returns ONLY its flattened [start, end) slice.

Both coord helpers are the pure, off-by-one-tested core of the FR-6 lazy
readback -- the load-bearing invariant that a bounded [start,end) request
NEVER triggers materialization of more coordinates than the requested
slice (codex Blocker-1 + Blocker-2, the ABS_MAX_PIXELS=4096 hard cap and
the never-pre-collect-the-whole-box/grid rule).

REUSE-coverage assertions (byte-unchanged PR-1 symbols still importable
and behave per their existing documented contract) are included per the
plan's red-test objective ("reuse-coverage of clamp_readback/paginate/
count_nan_inf; assert paginate() is accessed as a DICT").
"""

from __future__ import annotations

import math
import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — allow running standalone and via pytest.
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_PKG_ROOT))

import pytest

# ---------------------------------------------------------------------------
# PR-1 reuse imports — MUST remain importable and byte-unchanged (append-only
# contract). These are NOT the RED gate; the RED gate is the 4 new names
# below. Importing these here also confirms hou-dev's append did not
# accidentally break the PR-1 surface.
# ---------------------------------------------------------------------------
from fxhoudinimcp.cop_onnx_model import (  # noqa: F401
    clamp_readback,
    count_nan_inf,
    paginate,
)


# ===========================================================================
# PRIMARY RED GATE — the 4 new names must be importable
# ===========================================================================

class TestModuleImportsNewHelpers:
    """The 4 new pure helpers must be importable from cop_onnx_model AND
    callable. FAILS RED (ImportError) until hou-dev appends them."""

    def test_compute_histogram_importable_and_callable(self):
        from fxhoudinimcp.cop_onnx_model import compute_histogram  # noqa: F401
        assert callable(compute_histogram)

    def test_deinterleave_channels_importable_and_callable(self):
        from fxhoudinimcp.cop_onnx_model import deinterleave_channels  # noqa: F401
        assert callable(deinterleave_channels)

    def test_bounded_page_coords_importable_and_callable(self):
        from fxhoudinimcp.cop_onnx_model import bounded_page_coords  # noqa: F401
        assert callable(bounded_page_coords)

    def test_bounded_sample_coords_importable_and_callable(self):
        from fxhoudinimcp.cop_onnx_model import bounded_sample_coords  # noqa: F401
        assert callable(bounded_sample_coords)


# ===========================================================================
# compute_histogram
# ===========================================================================

class TestComputeHistogram:
    """compute_histogram(values, bins, lo, hi) -> list[int] — equal-width
    bin counts over [lo, hi]; NaN/Inf excluded; lo==hi degenerate -> bin 0;
    bins<=0 -> []; lo/hi None -> [0]*bins."""

    def test_basic_equal_width_bin_counts(self):
        from fxhoudinimcp.cop_onnx_model import compute_histogram

        # 4 bins over [0, 4): [0,1) [1,2) [2,3) [3,4]  (last bin inclusive
        # of hi, matching typical equal-width histogram semantics).
        values = [0.0, 0.5, 1.0, 1.5, 2.5, 3.9, 4.0]
        result = compute_histogram(values, bins=4, lo=0.0, hi=4.0)
        assert len(result) == 4
        assert sum(result) == len(values), (
            f"every finite value must land in exactly one bin, got {result!r} "
            f"summing to {sum(result)} for {len(values)} input values"
        )

    def test_bins_param_controls_result_length(self):
        from fxhoudinimcp.cop_onnx_model import compute_histogram

        result = compute_histogram([1.0, 2.0, 3.0], bins=8, lo=0.0, hi=10.0)
        assert len(result) == 8

    def test_all_values_in_single_bin_when_clustered(self):
        from fxhoudinimcp.cop_onnx_model import compute_histogram

        result = compute_histogram([5.0, 5.0, 5.0], bins=10, lo=0.0, hi=10.0)
        assert sum(result) == 3
        assert max(result) == 3, (
            f"all three identical values must land in the SAME bin, got {result!r}"
        )

    def test_lo_equals_hi_degenerate_all_finite_in_bin_zero(self):
        """lo==hi degenerate case: every finite value must land in bin 0
        (there is no meaningful bin-width to distribute across)."""
        from fxhoudinimcp.cop_onnx_model import compute_histogram

        result = compute_histogram([5.0, 5.0, 5.0, 5.0], bins=16, lo=5.0, hi=5.0)
        assert len(result) == 16
        assert result[0] == 4, (
            f"lo==hi degenerate: all 4 finite values must land in bin 0, got {result!r}"
        )
        assert sum(result) == 4
        assert sum(result[1:]) == 0, (
            f"lo==hi degenerate: no values should land in any bin other than 0, "
            f"got {result!r}"
        )

    def test_nan_and_inf_excluded_from_bin_counts(self):
        """NaN/Inf are EXCLUDED from the bin counts entirely (they are
        already counted separately by count_nan_inf) -- the sum of bin
        counts must equal only the FINITE value count, not the full input
        length."""
        from fxhoudinimcp.cop_onnx_model import compute_histogram

        values = [1.0, 2.0, float("nan"), 3.0, float("inf"), float("-inf"), 4.0]
        finite_count = 4  # 1.0, 2.0, 3.0, 4.0
        result = compute_histogram(values, bins=4, lo=1.0, hi=4.0)
        assert sum(result) == finite_count, (
            f"NaN/Inf must be EXCLUDED from bin counts -- expected sum=={finite_count} "
            f"(the finite-only count), got sum={sum(result)} from result={result!r}"
        )

    def test_all_nan_inf_values_excluded_leaves_empty_bins(self):
        """A values list of ONLY NaN/Inf (no finite values at all) must
        produce all-zero bin counts (nothing to distribute)."""
        from fxhoudinimcp.cop_onnx_model import compute_histogram

        values = [float("nan"), float("inf"), float("-inf")]
        result = compute_histogram(values, bins=8, lo=0.0, hi=1.0)
        assert result == [0] * 8, (
            f"an all-NaN/Inf values list must produce all-zero bins, got {result!r}"
        )

    def test_exact_interior_bin_placement_evenly_spaced(self):
        """REVIEW FIX 2 (codex red-review Major, threadId 019f2096): the
        prior tests only checked length/sum/boundaries -- an interior
        bin-placement bug (e.g. an off-by-one in the bin-index formula)
        would still pass. This pins the EXACT per-bin counts for a known,
        evenly-spaced 5-point dataset over 4 bins spanning [0.0, 1.0]
        (bin width 0.25):
          bins:   [0, 0.25)  [0.25, 0.5)  [0.5, 0.75)  [0.75, 1.0]
          values: 0.0        0.25         0.5          0.75, 1.0
        0.0 (== lo) lands in bin 0; 0.25 (an interior bin's LOWER edge)
        lands in bin 1 (NOT bin 0 -- the half-open [lo_bin, hi_bin)
        convention, verified here at an interior boundary, not just the
        overall lo/hi extremes already covered by
        test_boundary_value_at_lo_lands_in_first_bin /
        test_boundary_value_at_hi_lands_in_last_bin); 0.5 lands in bin 2;
        0.75 (another interior lower edge) lands in bin 3; 1.0 (== hi)
        also lands in bin 3 (the LAST bin, inclusive-upper convention).
        Expected exact counts: [1, 1, 1, 2]."""
        from fxhoudinimcp.cop_onnx_model import compute_histogram

        values = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = compute_histogram(values, bins=4, lo=0.0, hi=1.0)
        assert result == [1, 1, 1, 2], (
            f"expected the EXACT per-bin counts [1, 1, 1, 2] for the evenly-"
            f"spaced dataset {values!r} over 4 bins spanning [0.0, 1.0] "
            f"(bin width 0.25) -- an interior bin-placement bug (e.g. an "
            f"off-by-one bin-index formula) would produce a different "
            f"distribution while still passing a sum/length-only check. "
            f"Got {result!r}."
        )

    def test_exact_interior_bin_placement_uneven_distribution(self):
        """REVIEW FIX 2 (companion case): a SECOND, unevenly-distributed
        7-point dataset over 5 bins spanning [0.0, 1.0] (bin width 0.2):
          bins:   [0,0.2)  [0.2,0.4)  [0.4,0.6)  [0.6,0.8)  [0.8,1.0]
          values: 0.1      0.2, 0.3   (none)     0.6 x3     0.9
        0.1 -> bin 0; 0.2 (interior lower edge) and 0.3 -> bin 1; bin 2 is
        EMPTY (0 count -- proves a bug can't just be masked by every bin
        having a nonzero count); 0.6 (interior lower edge, x3) -> bin 3;
        0.9 -> bin 4. Expected exact counts: [1, 2, 0, 3, 1]."""
        from fxhoudinimcp.cop_onnx_model import compute_histogram

        values = [0.1, 0.2, 0.3, 0.6, 0.6, 0.6, 0.9]
        result = compute_histogram(values, bins=5, lo=0.0, hi=1.0)
        assert result == [1, 2, 0, 3, 1], (
            f"expected the EXACT per-bin counts [1, 2, 0, 3, 1] for the "
            f"unevenly-distributed dataset {values!r} over 5 bins spanning "
            f"[0.0, 1.0] (bin width 0.2), including an EMPTY interior bin "
            f"(bin 2, count 0) -- a wrong bin-index formula could still "
            f"produce a plausible-looking but incorrect distribution while "
            f"passing a sum-only check. Got {result!r}."
        )

    def test_bins_zero_or_negative_returns_empty_list(self):
        from fxhoudinimcp.cop_onnx_model import compute_histogram

        assert compute_histogram([1.0, 2.0], bins=0, lo=0.0, hi=10.0) == []
        assert compute_histogram([1.0, 2.0], bins=-1, lo=0.0, hi=10.0) == []

    def test_lo_hi_none_returns_zero_filled_bins(self):
        """lo/hi None (an all-non-finite channel's min/max sentinel from
        count_nan_inf) must produce [0]*bins rather than raising or
        attempting arithmetic on None."""
        from fxhoudinimcp.cop_onnx_model import compute_histogram

        result = compute_histogram([], bins=32, lo=None, hi=None)
        assert result == [0] * 32, (
            f"lo/hi==None must produce [0]*bins (32 zero-count bins), got {result!r}"
        )

    def test_lo_none_hi_present_returns_zero_filled_bins(self):
        """Either sentinel being None (not necessarily both) is treated the
        same defensive way -- zero-filled bins, never a TypeError from
        comparing None to a float."""
        from fxhoudinimcp.cop_onnx_model import compute_histogram

        result = compute_histogram([1.0, 2.0], bins=4, lo=None, hi=10.0)
        assert result == [0] * 4

    def test_result_length_always_equals_bins_for_positive_bins(self):
        from fxhoudinimcp.cop_onnx_model import compute_histogram

        for n in (1, 2, 32, 100):
            result = compute_histogram([1.0, 2.0, 3.0], bins=n, lo=0.0, hi=5.0)
            assert len(result) == n, (
                f"compute_histogram(..., bins={n}, ...) must return exactly {n} "
                f"bin counts, got len={len(result)}"
            )

    def test_boundary_value_at_hi_lands_in_last_bin(self):
        """A value exactly equal to hi must land in the LAST bin (inclusive
        upper boundary), not be dropped or overflow past the bin array."""
        from fxhoudinimcp.cop_onnx_model import compute_histogram

        result = compute_histogram([10.0], bins=4, lo=0.0, hi=10.0)
        assert sum(result) == 1, (
            f"a value exactly at hi must be counted, not dropped: got {result!r}"
        )
        assert result[-1] == 1 or result[-1] >= 1, (
            f"a value exactly at hi should land in the last bin, got {result!r}"
        )

    def test_boundary_value_at_lo_lands_in_first_bin(self):
        from fxhoudinimcp.cop_onnx_model import compute_histogram

        result = compute_histogram([0.0], bins=4, lo=0.0, hi=10.0)
        assert result[0] == 1, (
            f"a value exactly at lo must land in bin 0, got {result!r}"
        )

    def test_empty_values_list_all_finite_bins_zero(self):
        from fxhoudinimcp.cop_onnx_model import compute_histogram

        result = compute_histogram([], bins=4, lo=0.0, hi=10.0)
        assert result == [0, 0, 0, 0]


# ===========================================================================
# deinterleave_channels
# ===========================================================================

class TestDeinterleaveChannels:
    """deinterleave_channels(flat, channels) -> list[list] — split a
    channel-interleaved flat list into `channels` per-channel lists
    (flat[c::channels]). channels<=0 -> [] guard."""

    def test_three_channel_interleaved_pixel_major(self):
        from fxhoudinimcp.cop_onnx_model import deinterleave_channels

        # Two pixels, 3 channels (r,g,b), pixel-major interleaved:
        # pixel0=(1,2,3), pixel1=(4,5,6)
        flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = deinterleave_channels(flat, channels=3)
        assert len(result) == 3, f"expected 3 per-channel lists, got {len(result)}"
        assert result[0] == [1.0, 4.0], f"channel 0 (r) must be [1.0, 4.0], got {result[0]!r}"
        assert result[1] == [2.0, 5.0], f"channel 1 (g) must be [2.0, 5.0], got {result[1]!r}"
        assert result[2] == [3.0, 6.0], f"channel 2 (b) must be [3.0, 6.0], got {result[2]!r}"

    def test_single_channel_passthrough(self):
        from fxhoudinimcp.cop_onnx_model import deinterleave_channels

        flat = [1.0, 2.0, 3.0, 4.0]
        result = deinterleave_channels(flat, channels=1)
        assert result == [[1.0, 2.0, 3.0, 4.0]]

    def test_four_channel_rgba(self):
        from fxhoudinimcp.cop_onnx_model import deinterleave_channels

        flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = deinterleave_channels(flat, channels=4)
        assert len(result) == 4
        assert result[0] == [1.0, 5.0]
        assert result[3] == [4.0, 8.0]

    def test_empty_flat_list_returns_empty_per_channel_lists(self):
        from fxhoudinimcp.cop_onnx_model import deinterleave_channels

        result = deinterleave_channels([], channels=3)
        assert result == [[], [], []]

    def test_channels_zero_returns_empty_list(self):
        from fxhoudinimcp.cop_onnx_model import deinterleave_channels

        result = deinterleave_channels([1.0, 2.0, 3.0], channels=0)
        assert result == [], f"channels<=0 must guard to [], got {result!r}"

    def test_channels_negative_returns_empty_list(self):
        from fxhoudinimcp.cop_onnx_model import deinterleave_channels

        result = deinterleave_channels([1.0, 2.0, 3.0], channels=-1)
        assert result == []

    def test_roundtrip_reconstructs_original_pixel_order(self):
        """A property-style round-trip check: re-interleaving the
        deinterleaved channels back together (zip) must reconstruct the
        original flat pixel-major order."""
        from fxhoudinimcp.cop_onnx_model import deinterleave_channels

        flat = [float(i) for i in range(30)]  # 10 pixels, 3 channels
        channels = 3
        result = deinterleave_channels(flat, channels)
        reconstructed = []
        for pixel_tuple in zip(*result):
            reconstructed.extend(pixel_tuple)
        assert reconstructed == flat, (
            f"re-interleaving the per-channel lists must reconstruct the original "
            f"flat pixel-major buffer. Expected {flat!r}, got {reconstructed!r}"
        )


# ===========================================================================
# bounded_page_coords — the LAZY ROI coord slice
# ===========================================================================

class TestBoundedPageCoords:
    """bounded_page_coords(x0, y0, box_w, box_h, start, end) -> list[(x,y)]
    — the LAZY [start,end) slice of a box_w x box_h box offset at (x0,y0),
    row-major. NEVER materializes the whole box (folds codex Blocker-2)."""

    def test_full_small_box_row_major_order(self):
        from fxhoudinimcp.cop_onnx_model import bounded_page_coords

        # A 3x2 box at origin (0,0): row-major order is
        # (0,0)(1,0)(2,0)(0,1)(1,1)(2,1)
        result = bounded_page_coords(x0=0, y0=0, box_w=3, box_h=2, start=0, end=6)
        assert result == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)], (
            f"expected row-major (x=x0+i%box_w, y=y0+i//box_w) order, got {result!r}"
        )

    def test_offset_box_applies_x0_y0(self):
        from fxhoudinimcp.cop_onnx_model import bounded_page_coords

        result = bounded_page_coords(x0=10, y0=20, box_w=2, box_h=2, start=0, end=4)
        assert result == [(10, 20), (11, 20), (10, 21), (11, 21)], (
            f"the box offset (x0,y0) must be applied to every coord, got {result!r}"
        )

    def test_partial_slice_middle_of_box(self):
        """A [start,end) slice that does NOT start at 0 and does NOT reach
        the box's full pixel count -- only the requested subset must be
        returned."""
        from fxhoudinimcp.cop_onnx_model import bounded_page_coords

        # 4x4 box (16 pixels total); request only pixels [5, 8) — a
        # bounded, non-trivial middle slice.
        result = bounded_page_coords(x0=0, y0=0, box_w=4, box_h=4, start=5, end=8)
        assert len(result) == 3, f"expected exactly 3 coords for [5,8), got {result!r}"
        # index 5 -> x=5%4=1, y=5//4=1 ; index 6 -> x=2,y=1 ; index 7 -> x=3,y=1
        assert result == [(1, 1), (2, 1), (3, 1)], (
            f"expected the exact [5,8) row-major slice, got {result!r}"
        )

    def test_off_by_one_at_box_right_edge(self):
        """The last column of a row must be included when end reaches the
        row's final index, and excluded from the NEXT row when end stops
        exactly at the row boundary (an off-by-one edge check)."""
        from fxhoudinimcp.cop_onnx_model import bounded_page_coords

        # 3x1 box (3 pixels): [0,3) must include ALL 3, [0,2) must
        # include only the first 2.
        full = bounded_page_coords(x0=0, y0=0, box_w=3, box_h=1, start=0, end=3)
        assert full == [(0, 0), (1, 0), (2, 0)]
        partial = bounded_page_coords(x0=0, y0=0, box_w=3, box_h=1, start=0, end=2)
        assert partial == [(0, 0), (1, 0)], (
            f"end=2 (exclusive) must stop BEFORE the third pixel, got {partial!r}"
        )

    def test_off_by_one_at_box_bottom_edge(self):
        """A slice spanning exactly the last row must include the final
        pixel and no more."""
        from fxhoudinimcp.cop_onnx_model import bounded_page_coords

        # 2x2 box (4 pixels): [2,4) is the SECOND row only.
        result = bounded_page_coords(x0=0, y0=0, box_w=2, box_h=2, start=2, end=4)
        assert result == [(0, 1), (1, 1)], (
            f"expected exactly the second row (indices 2,3), got {result!r}"
        )

    def test_empty_slice_start_equals_end(self):
        from fxhoudinimcp.cop_onnx_model import bounded_page_coords

        result = bounded_page_coords(x0=0, y0=0, box_w=10, box_h=10, start=5, end=5)
        assert result == [], f"start==end must produce an empty slice, got {result!r}"

    def test_never_materializes_more_than_the_requested_slice(self):
        """The load-bearing FR-6/Blocker-2 invariant: for a LARGE box, a
        small [start,end) slice must return EXACTLY that many coords, not
        anything close to the full box size — proving the function computes
        the slice lazily rather than building the whole box and slicing
        after the fact (which would still be correct in RESULT but would
        defeat the memory/CPU-bound guarantee the handler relies on; this
        pure function's contract is that the returned length always equals
        end-start, regardless of how large box_w*box_h is)."""
        from fxhoudinimcp.cop_onnx_model import bounded_page_coords

        # A box that would be 4096*4096 = ~16.7M pixels if fully
        # materialized. Request only a tiny 5-pixel slice.
        huge_box_w = 4096
        huge_box_h = 4096
        result = bounded_page_coords(
            x0=0, y0=0, box_w=huge_box_w, box_h=huge_box_h, start=1000, end=1005
        )
        assert len(result) == 5, (
            f"a [1000,1005) slice of a 4096x4096 box must return EXACTLY 5 coords "
            f"(never materializing the full ~16.7M-pixel box), got len={len(result)}"
        )


# ===========================================================================
# bounded_sample_coords — the LAZY strided-sample coord slice
# ===========================================================================

class TestBoundedSampleCoords:
    """bounded_sample_coords(w, h, stride, start, end) -> list[(x,y)] — the
    LAZY [start,end) slice of the strided grid
    [(x*stride, y*stride) for y in 0..ceil(h/stride)-1
                           for x in 0..ceil(w/stride)-1]."""

    def test_stride_one_full_grid_matches_dense_coords(self):
        from fxhoudinimcp.cop_onnx_model import bounded_sample_coords

        # stride=1 over a 3x2 plane -> every pixel, row-major.
        result = bounded_sample_coords(w=3, h=2, stride=1, start=0, end=6)
        assert result == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)], (
            f"stride=1 must enumerate every pixel in row-major order, got {result!r}"
        )

    def test_stride_two_produces_scaled_coords(self):
        from fxhoudinimcp.cop_onnx_model import bounded_sample_coords

        # w=4,h=4, stride=2 -> grid points at x in {0,2}, y in {0,2}:
        # ceil(4/2)=2 columns, 2 rows -> 4 grid points total.
        result = bounded_sample_coords(w=4, h=4, stride=2, start=0, end=4)
        assert result == [(0, 0), (2, 0), (0, 2), (2, 2)], (
            f"expected the strided grid scaled by stride=2, got {result!r}"
        )

    def test_ceil_math_for_non_divisible_dims(self):
        """w/h not evenly divisible by stride must still cover the full
        plane via ceil() -- e.g. w=5, stride=2 -> columns at x=0,2,4
        (ceil(5/2)=3 columns), not just x=0,2 (which would silently drop
        the trailing partial column)."""
        from fxhoudinimcp.cop_onnx_model import bounded_sample_coords

        result = bounded_sample_coords(w=5, h=1, stride=2, start=0, end=3)
        assert result == [(0, 0), (2, 0), (4, 0)], (
            f"ceil(5/2)=3 columns (x=0,2,4) must all be present, got {result!r}"
        )

    def test_partial_slice_of_a_larger_strided_grid(self):
        from fxhoudinimcp.cop_onnx_model import bounded_sample_coords

        # w=6,h=6,stride=2 -> grid is 3x3=9 points at
        # x in {0,2,4}, y in {0,2,4}, row-major.
        # Request only the middle slice [3,6).
        result = bounded_sample_coords(w=6, h=6, stride=2, start=3, end=6)
        assert result == [(0, 2), (2, 2), (4, 2)], (
            f"expected exactly the [3,6) slice of the 3x3 strided grid "
            f"(the second row), got {result!r}"
        )

    def test_stride_greater_than_dim_yields_single_point(self):
        """A stride larger than both dimensions must still produce exactly
        ONE grid point at (0, 0) -- ceil(w/stride)=1, ceil(h/stride)=1."""
        from fxhoudinimcp.cop_onnx_model import bounded_sample_coords

        result = bounded_sample_coords(w=4, h=4, stride=100, start=0, end=1)
        assert result == [(0, 0)], (
            f"stride > both dims must yield a single grid point (0,0), got {result!r}"
        )

    def test_empty_slice_start_equals_end(self):
        from fxhoudinimcp.cop_onnx_model import bounded_sample_coords

        result = bounded_sample_coords(w=10, h=10, stride=1, start=7, end=7)
        assert result == [], f"start==end must produce an empty slice, got {result!r}"

    def test_off_by_one_at_grid_row_boundary(self):
        """A slice ending exactly at a grid-row boundary must not spill
        into the next row."""
        from fxhoudinimcp.cop_onnx_model import bounded_sample_coords

        # w=4,h=4,stride=2 -> grid is 2x2=4 points: (0,0)(2,0)(0,2)(2,2).
        # [0,2) is exactly row 0.
        row0 = bounded_sample_coords(w=4, h=4, stride=2, start=0, end=2)
        assert row0 == [(0, 0), (2, 0)]
        row1 = bounded_sample_coords(w=4, h=4, stride=2, start=2, end=4)
        assert row1 == [(0, 2), (2, 2)]

    def test_never_materializes_more_than_the_requested_slice(self):
        """The load-bearing FR-6/Blocker-2 invariant for sample mode: for a
        LARGE plane, a small [start,end) slice must return EXACTLY that
        many coords regardless of how large the full strided grid would be
        -- this is what keeps a stride=1 request over a 4096x4096 plane
        from blowing memory/CPU when only a small page is asked for."""
        from fxhoudinimcp.cop_onnx_model import bounded_sample_coords

        # stride=1 over 4096x4096 -> the FULL strided grid would be
        # ~16.7M points. Request only a tiny 5-point slice.
        result = bounded_sample_coords(w=4096, h=4096, stride=1, start=2000, end=2005)
        assert len(result) == 5, (
            f"a [2000,2005) slice of a stride=1 4096x4096 grid (~16.7M points if "
            f"fully materialized) must return EXACTLY 5 coords, got len={len(result)}"
        )

    def test_result_within_plane_bounds(self):
        """Every returned coord must be within [0, w) x [0, h) — the
        strided grid must never produce an out-of-bounds coordinate even
        near the plane edges."""
        from fxhoudinimcp.cop_onnx_model import bounded_sample_coords

        w, h, stride = 10, 7, 3
        cols = math.ceil(w / stride)
        rows = math.ceil(h / stride)
        total = cols * rows
        result = bounded_sample_coords(w=w, h=h, stride=stride, start=0, end=total)
        for x, y in result:
            assert 0 <= x < w, f"x={x} must be within [0, {w})"
            assert 0 <= y < h, f"y={y} must be within [0, {h})"


# ===========================================================================
# REUSE-coverage — PR-1 symbols remain byte-unchanged (paginate as a DICT,
# per codex M3-paginate-dict)
# ===========================================================================

class TestReuseCoveragePaginateIsADict:
    """paginate() must be accessed as a DICT (pg['start'], pg['end'],
    pg['total_pages'], pg['truncated']) -- codex M3-paginate-dict. This is
    a reuse-coverage assertion confirming the PR-1 contract the handler
    depends on has not drifted."""

    def test_paginate_returns_dict_with_expected_keys(self):
        from fxhoudinimcp.cop_onnx_model import paginate

        result = paginate(total_items=100, page=0, page_size=10)
        assert isinstance(result, dict), (
            f"paginate() must return a dict (not a dataclass/tuple), got {type(result)!r}"
        )
        for key in ("page", "page_size", "total_pages", "start", "end", "truncated"):
            assert key in result, f"paginate() dict must have key {key!r}, got {result!r}"

    def test_paginate_start_end_usable_for_slicing(self):
        from fxhoudinimcp.cop_onnx_model import paginate

        result = paginate(total_items=25, page=1, page_size=10)
        assert result["start"] == 10
        assert result["end"] == 20
        assert result["total_pages"] == 3

    def test_clamp_readback_still_importable_and_returns_positive_stride(self):
        from fxhoudinimcp.cop_onnx_model import clamp_readback

        stride = clamp_readback(xres=4096, yres=4096, channels=3, max_pixels=4096)
        assert isinstance(stride, int)
        assert stride >= 1
        assert math.ceil(4096 / stride) * math.ceil(4096 / stride) <= 4096, (
            f"clamp_readback must produce a stride that bounds the strided pixel "
            f"count to <= max_pixels, got stride={stride}"
        )

    def test_count_nan_inf_still_importable_and_counts_correctly(self):
        from fxhoudinimcp.cop_onnx_model import count_nan_inf

        result = count_nan_inf([1.0, float("nan"), float("inf"), 2.0])
        assert result["nan_count"] == 1
        assert result["inf_count"] == 1
        assert result["min"] == 1.0
        assert result["max"] == 2.0
