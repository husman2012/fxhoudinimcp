"""Handler-level (mocked-hou) pytest tests for cop_onnx_read_pixels
(PP12-113 PR-5, READONLY -- the FINAL tool of member 113) -- codex
B1/B2/M3/M4/M5 mandatory red coverage.

Unit: pp12-113e
testVerificationSurface: pytest-model
planSha: bc1f8d9e9fcd1b46216e3eab594ded4a31ee907e305e5d12fde0b040314c445c

These tests import the HANDLER module (cop_onnx_handlers.py) with a mocked
`hou` module installed into sys.modules BEFORE import, per
test-fixture-conventions.md §2.3 (`monkeypatch.setitem(sys.modules, 'hou',
fake)`). This lets the handler's Python control flow be exercised off-DCC
without a real Houdini session, using MagicMock fakes for the
hou.Node / hou.ImageLayer surface.

This is a genuinely mocked-hou rung (test-fixture-conventions.md §2 — the
handler cannot be fully split into *_model.py because it calls
hou.node()/node.layer(idx)/layer.bufferIndex(x,y) as its core job), NOT a
substitute for the real hython-smoke
(cop_onnx_read_pixels_hython_smoke.py), which remains the authority on
real cooked-node pixel-readback behavior against a real Houdini session.

PLAN-REVIEW FOLD (rev2) coverage this file pins (codex Blocker-1/Blocker-2
/M3/M4/M5, per the plan pp12-113e lockedFieldContract):
  - B1-abs-cap: ABS_MAX_PIXELS=4096 is an ABSOLUTE server ceiling applied
    to ALL modes regardless of caller params (max_pixels=10_000_000 /
    page_size=10_000_000 must still clamp to <=4096, truncated:True).
  - B2-lazy: roi/sample coords are computed LAZILY (only the [start,end)
    page slice) -- bufferIndex is called ONLY for the page's coords, NEVER
    the whole box/grid. Proven via CALL-COUNT assertions (a legitimate
    call-assertion per test-fixture-conventions.md §2.3: "call-assertions
    only when the call IS the contract" -- the bounded-call-count IS the
    FR-6 contract here).
  - M3-paginate-dict: paginate() is accessed as a DICT (already asserted
    in test_cop_onnx_read_pixels_model.py; this file's roi/sample doubles
    implicitly exercise the handler's dict-style access via successful
    pagination behavior).
  - M4-summary-memory: summary NEVER reads the full plane via
    allBufferElements() -- it strides via clamp_readback(...,budget) +
    bufferIndex over a budget-bounded sample. Proven via bufferIndex
    call-count bounded by budget, NOT xres*yres.
  - M5-histogram-contract: SUMMARY_HISTOGRAM_BINS=32 locked; per-channel
    lo/hi = finite min/max from count_nan_inf; an injected NaN+Inf pixel
    is COUNTED (never dropped).
  - READ-ONLY: layer(idx)==None (a stale/uncooked node) -> {ok:True,
    cooked:False, message} -- and node.cook is NEVER called (this handler
    never cooks; read-only per the grounded design).

Assertions target the RETURNED DICT (the public contract) -- never which
hou.* methods were called or in what order (tdd-with-agents.md §2 mirror-
test ban), EXCEPT where the call-COUNT itself IS the FR-6 bounded-work
contract (bufferIndex call-count for the lazy/bounded invariant) --
per test-fixture-conventions.md §2.3's explicit carve-out.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# mock_hou fixture — installs a fake `hou` module into sys.modules BEFORE
# the handler module is imported (test-fixture-conventions.md §2.3).
# Mirrors test_cop_onnx_run_inference_handler.py (pp12-113d) exactly,
# including its sys.path bootstrap for the fork's non-standard python/ +
# houdini/scripts/python/ roots.
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_hou(monkeypatch):
    """Install a mock `hou` module so cop_onnx_handlers.py loads off-DCC.

    Returns the mock so tests can configure hou.node.return_value, etc.
    hou.copNodeTypeCategory resolves to a sentinel MagicMock so the
    handler's category-equality check (`node.type().category() ==
    hou.copNodeTypeCategory()`) can be pinned per-test.
    """
    fake = MagicMock(name="hou")
    monkeypatch.setitem(sys.modules, "hou", fake)

    # Ensure the fork's non-standard package roots are importable (mirrors
    # test_cop_onnx_run_inference_handler.py's bootstrap, pp12-113d):
    # pytest's rootdir discovery does not put houdini/scripts/python
    # (fxhoudinimcp_server's home) on sys.path by default.
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    _pkg_python = os.path.join(_repo_root, "python")
    _houdini_handlers = os.path.join(_repo_root, "houdini", "scripts", "python")
    for _p in (_pkg_python, _houdini_handlers):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    return fake


def _import_handler_module():
    """Import cop_onnx_handlers fresh (relies on mock_hou already being
    installed in sys.modules AND sys.path already patched by the caller's
    mock_hou fixture)."""
    import importlib
    if "fxhoudinimcp_server.handlers.cop_onnx_handlers" in sys.modules:
        importlib.reload(sys.modules["fxhoudinimcp_server.handlers.cop_onnx_handlers"])
        return sys.modules["fxhoudinimcp_server.handlers.cop_onnx_handlers"]
    import fxhoudinimcp_server.handlers.cop_onnx_handlers as _handlers
    return _handlers


def _make_fake_onnx_node(cop_category_sentinel, output_names=("output1",)):
    """A fake hou.Node whose type().name()=='onnx' and
    type().category()==cop_category_sentinel (the Cop-category match)."""
    node = MagicMock(name="fake_onnx_node")
    node.type.return_value.name.return_value = "onnx"
    node.type.return_value.category.return_value = cop_category_sentinel
    node.path.return_value = "/obj/copnet1/agent_onnx"
    node.outputNames.return_value = output_names
    return node


class _FakeImageLayer:
    """A fake hou.ImageLayer whose bufferIndex(x, y) is DETERMINISTIC per
    coordinate (not a MagicMock auto-return) so per-pixel values are
    predictable and CALL-COUNT-trackable.

    channels: number of channels per pixel.
    width/height: the plane's bufferResolution().
    value_fn: (x, y, channel_index) -> float. Defaults to a simple
        deterministic pattern. Override to inject NaN/Inf at specific
        coords for the nan/inf-counting tests.
    """

    def __init__(self, width, height, channels, value_fn=None):
        self._width = width
        self._height = height
        self._channels = channels
        self._value_fn = value_fn or (lambda x, y, c: float((x + y + c) % 10))
        self.buffer_index_call_count = 0
        self.buffer_index_calls = []

    def bufferResolution(self):
        return (self._width, self._height)

    def channelCount(self):
        return self._channels

    def storageType(self):
        return "imageLayerStorageType.Float32"

    def bufferIndex(self, x, y):
        self.buffer_index_call_count += 1
        self.buffer_index_calls.append((x, y))
        return tuple(self._value_fn(x, y, c) for c in range(self._channels))


# ---------------------------------------------------------------------------
# PRIMARY RED GATE — the handler function must exist
# ---------------------------------------------------------------------------

class TestHandlerImport:
    def test_read_pixels_handler_importable(self, mock_hou):
        """cop_onnx_read_pixels must be importable from cop_onnx_handlers.
        FAILS RED (AttributeError) until hou-dev implements it."""
        handlers = _import_handler_module()
        assert hasattr(handlers, "cop_onnx_read_pixels"), (
            "cop_onnx_handlers.py must expose cop_onnx_read_pixels."
        )
        assert callable(handlers.cop_onnx_read_pixels)

    def test_existing_pr2_pr3_pr4_handlers_unaffected(self, mock_hou):
        """Append-only contract: PR-2/PR-3/PR-4's handlers must remain
        importable — the new handler must not clobber its siblings."""
        handlers = _import_handler_module()
        for name in (
            "cop_onnx_list_models",
            "cop_onnx_inspect_model",
            "cop_onnx_setup_node",
            "cop_onnx_set_provider",
            "cop_onnx_run_inference",
        ):
            assert hasattr(handlers, name), (
                f"cop_onnx_handlers.py must still expose {name} (append-only)."
            )


# ---------------------------------------------------------------------------
# READ-ONLY: a stale/uncooked node (layer(idx) is None) -> 'not cooked',
# and node.cook is NEVER called.
# ---------------------------------------------------------------------------

class TestStaleNodeReadOnly:
    """layer(idx) returns None on a stale/uncooked node (GROUNDED: no
    auto-cook) -> the handler REPORTS 'not cooked' rather than cooking.
    This is the READ-ONLY guarantee's central test: the handler must
    NEVER call node.cook."""

    def test_layer_none_returns_not_cooked_message(self, mock_hou):
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)
        node.layer.return_value = None

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_read_pixels(node_path="/obj/copnet1/agent_onnx")

        assert result["ok"] is True, (
            f"a stale/uncooked node is a valid, reportable outcome, not an error -- "
            f"must be ok:True, got {result!r}"
        )
        assert result["cooked"] is False, (
            f"a stale node (layer(idx) is None) must report cooked:False, got {result!r}"
        )
        assert "message" in result and "not cooked" in result["message"].lower(), (
            f"expected an informative 'not cooked' message, got {result!r}"
        )

    def test_stale_node_never_triggers_cook(self, mock_hou):
        """The READ-ONLY guarantee: cop_onnx_read_pixels must NEVER call
        node.cook(), even on a stale node. This is a legitimate call-
        assertion per test-fixture-conventions.md §2.3 -- "never cooking"
        IS the read-only contract this handler exists to provide."""
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)
        node.layer.return_value = None

        mock_hou.node.return_value = node

        handlers.cop_onnx_read_pixels(node_path="/obj/copnet1/agent_onnx")

        assert node.cook.call_count == 0, (
            f"cop_onnx_read_pixels must NEVER call node.cook() -- it is genuinely "
            f"read-only (a stale node is REPORTED not cooked, not mutated). "
            f"Got node.cook.call_count={node.cook.call_count}."
        )


# ---------------------------------------------------------------------------
# wrong-category / bad-target validation
# ---------------------------------------------------------------------------

class TestTargetValidation:
    """Target validation requires BOTH type().name()=='onnx' AND
    type().category()==hou.copNodeTypeCategory() -- a bare name-only check
    is insufficient (mirrors the PR-4 run_inference codex Blocker-1
    pattern, applied here to read_pixels)."""

    def test_wrong_category_is_ok_false(self, mock_hou):
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        wrong_sentinel = MagicMock(name="Sop_category_sentinel")
        assert wrong_sentinel is not cop_sentinel

        node = MagicMock(name="fake_sop_onnx_node")
        node.type.return_value.name.return_value = "onnx"
        node.type.return_value.category.return_value = wrong_sentinel  # NOT the Cop category
        node.path.return_value = "/obj/geo1/onnx1"

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_read_pixels(node_path="/obj/geo1/onnx1")

        assert result["ok"] is False, (
            f"a node named 'onnx' but NOT of the Cop category must be REJECTED "
            f"(ok:False) -- a name-only check is insufficient. Got {result!r}."
        )

    def test_unresolved_node_path_is_ok_false(self, mock_hou):
        handlers = _import_handler_module()

        mock_hou.node.return_value = None

        result = handlers.cop_onnx_read_pixels(node_path="/obj/does_not_exist")

        assert result["ok"] is False, (
            f"an unresolved node_path must be ok:False (bad TARGET), got {result!r}"
        )

    def test_wrong_type_name_is_ok_false(self, mock_hou):
        """A resolved node whose type().name() is NOT 'onnx' at all
        (e.g. a copnet) must also be rejected."""
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = MagicMock(name="fake_copnet_node")
        node.type.return_value.name.return_value = "copnet"
        node.type.return_value.category.return_value = cop_sentinel
        node.path.return_value = "/obj/copnet1"

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_read_pixels(node_path="/obj/copnet1")

        assert result["ok"] is False, (
            f"a node whose type().name() != 'onnx' must be ok:False, got {result!r}"
        )

    def test_empty_node_path_is_ok_false(self, mock_hou):
        """FR-2: an empty node_path must be rejected before any hou.node()
        call."""
        handlers = _import_handler_module()

        result = handlers.cop_onnx_read_pixels(node_path="")

        assert result["ok"] is False, (
            f"an empty node_path must be ok:False (FR-2), got {result!r}"
        )


# ---------------------------------------------------------------------------
# bad plane name
# ---------------------------------------------------------------------------

class TestBadPlaneName:
    """A plane name not in node.outputNames() must be rejected (ok:False)
    with the available names surfaced."""

    def test_plane_not_in_output_names_is_ok_false(self, mock_hou):
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel, output_names=("output1",))
        layer = _FakeImageLayer(width=4, height=4, channels=3)
        node.layer.return_value = layer

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_read_pixels(
            node_path="/obj/copnet1/agent_onnx", plane="nonexistent_plane"
        )

        assert result["ok"] is False, (
            f"a plane name not in outputNames() must be ok:False, got {result!r}"
        )
        assert "nonexistent_plane" in str(result.get("error", "")), (
            f"the error should mention the requested plane name, got {result!r}"
        )


# ---------------------------------------------------------------------------
# summary mode — per-channel stats, 32-bin histogram, nan/inf counted
# ---------------------------------------------------------------------------

class TestSummaryMode:
    """summary over a fake ImageLayer -> per-channel stats + a 32-bin
    histogram + nan/inf counted (M5-histogram-contract). Injecting a NaN
    and an Inf pixel must result in them being COUNTED, never dropped."""

    def test_summary_returns_per_channel_stats_and_32_bin_histogram(self, mock_hou):
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)
        layer = _FakeImageLayer(width=8, height=8, channels=3)
        node.layer.return_value = layer

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_read_pixels(
            node_path="/obj/copnet1/agent_onnx", mode="summary"
        )

        assert result["ok"] is True, result
        assert result["cooked"] is True
        assert result["mode"] == "summary"
        assert result["channels"] == 3
        stats = result["stats"]
        assert len(stats["min"]) == 3
        assert len(stats["max"]) == 3
        assert len(stats["mean"]) == 3
        histogram = result["histogram"]
        assert histogram["bins"] == 32, (
            f"SUMMARY_HISTOGRAM_BINS must be LOCKED at 32 (spec §4.2), "
            f"got histogram bins={histogram['bins']!r}"
        )
        assert len(histogram["counts"]) == 3, (
            f"histogram counts must have one list PER CHANNEL, got "
            f"{len(histogram['counts'])} lists for 3 channels"
        )
        for per_channel_counts in histogram["counts"]:
            assert len(per_channel_counts) == 32

    def test_summary_nan_inf_pixels_are_counted_not_dropped_exact_values(self, mock_hou):
        """REVIEW FIX 1 (codex red-review Major, threadId 019f2096): a
        shape-only assertion (nan_count/inf_count >= 1, list lengths only)
        could pass a real impl bug (e.g. off-by-one counting, wrong-channel
        aggregation, a mis-scaled mean). This double uses a TINY
        DETERMINISTIC 2x2, single-channel fake plane -- small enough that
        clamp_readback(2, 2, 1, 4096) == stride 1, so summary reads ALL 4
        pixels (no sampling ambiguity) -- with EXACTLY ONE NaN pixel and
        EXACTLY ONE +Inf pixel, and two KNOWN finite values (2.0, 4.0), so
        every stat is independently EXACT-value-computable and pinned:
          pixels: (0,0)=NaN, (1,0)=+Inf, (0,1)=2.0, (1,1)=4.0
          finite values: [2.0, 4.0] -> min=2.0, max=4.0, mean=3.0
          nan_count == 1 (EXACTLY, not >=1)
          inf_count == 1 (EXACTLY, not >=1)
          histogram: 32 equal-width bins over [2.0, 4.0] (bin width 0.0625)
            -> 2.0 lands in bin 0 (lo boundary), 4.0 lands in bin 31 (hi
            boundary, the LAST bin) -- pins the same bin-edge convention
            asserted directly against compute_histogram in
            test_cop_onnx_read_pixels_model.py's TestComputeHistogram
            boundary tests, exercised HERE through the full handler path.
        """
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)

        def _value_fn(x, y, c):
            if (x, y) == (0, 0):
                return float("nan")
            if (x, y) == (1, 0):
                return float("inf")
            if (x, y) == (0, 1):
                return 2.0
            if (x, y) == (1, 1):
                return 4.0
            raise AssertionError(f"unexpected coord ({x}, {y}) on a 2x2 plane")

        layer = _FakeImageLayer(width=2, height=2, channels=1, value_fn=_value_fn)
        node.layer.return_value = layer

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_read_pixels(
            node_path="/obj/copnet1/agent_onnx", mode="summary", max_pixels=4096
        )

        assert result["ok"] is True, result
        assert result.get("stride", 1) == 1, (
            f"a 2x2 plane with max_pixels=4096 must NOT be strided (stride "
            f"must be 1, all 4 pixels read) -- got stride={result.get('stride')!r}, "
            f"which would make the exact-value assertions below ambiguous."
        )
        assert layer.buffer_index_call_count == 4, (
            f"a 2x2 plane at stride=1 must call bufferIndex EXACTLY 4 times "
            f"(all pixels), got {layer.buffer_index_call_count}."
        )

        stats = result["stats"]
        assert stats["nan_count"] == 1, (
            f"EXACTLY one NaN pixel was injected -- nan_count must be EXACTLY "
            f"1 (not merely >=1), got nan_count={stats['nan_count']!r}"
        )
        assert stats["inf_count"] == 1, (
            f"EXACTLY one +Inf pixel was injected -- inf_count must be "
            f"EXACTLY 1 (not merely >=1), got inf_count={stats['inf_count']!r}"
        )
        assert stats["min"] == [2.0], (
            f"the finite values are [2.0, 4.0] -- min must be EXACTLY 2.0, "
            f"got min={stats['min']!r}"
        )
        assert stats["max"] == [4.0], (
            f"the finite values are [2.0, 4.0] -- max must be EXACTLY 4.0, "
            f"got max={stats['max']!r}"
        )
        assert stats["mean"] == [3.0], (
            f"the finite values are [2.0, 4.0] -- mean must be EXACTLY 3.0, "
            f"got mean={stats['mean']!r}"
        )

        histogram = result["histogram"]
        assert histogram["bins"] == 32
        counts = histogram["counts"][0]
        assert len(counts) == 32
        assert sum(counts) == 2, (
            f"exactly 2 finite values must be distributed across the "
            f"histogram bins (NaN/Inf excluded), got sum={sum(counts)} "
            f"from counts={counts!r}"
        )
        assert counts[0] == 1, (
            f"the value 2.0 (== lo) must land in bin 0 (the first bin, lo "
            f"boundary), got counts={counts!r}"
        )
        assert counts[31] == 1, (
            f"the value 4.0 (== hi) must land in bin 31 (the LAST bin, hi "
            f"boundary -- the inclusive-upper-bound convention), got "
            f"counts={counts!r}"
        )
        assert sum(counts[1:31]) == 0, (
            f"no value should land in any of the 30 interior bins for this "
            f"2-point dataset at the extremes, got counts={counts!r}"
        )

    def test_summary_stride_and_sampled_flag_present(self, mock_hou):
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)
        layer = _FakeImageLayer(width=100, height=100, channels=3)
        node.layer.return_value = layer

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_read_pixels(
            node_path="/obj/copnet1/agent_onnx", mode="summary", max_pixels=64
        )

        assert result["ok"] is True, result
        assert "sampled" in result
        assert "stride" in result
        assert result["stride"] >= 1


# ---------------------------------------------------------------------------
# roi mode — bounded/lazy: bufferIndex called ONLY for the page's coords
# ---------------------------------------------------------------------------

class TestRoiModeBoundedLazy:
    """roi mode must be LAZY and BOUNDED -- bufferIndex is called ONLY for
    the requested page's coordinates, NOT the whole box (codex Blocker-2).
    Proven via call-count, a legitimate assertion here per
    test-fixture-conventions.md §2.3."""

    def test_roi_bounded_page_calls_buffer_index_only_for_page_slice(self, mock_hou):
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)
        layer = _FakeImageLayer(width=64, height=64, channels=3)
        node.layer.return_value = layer

        mock_hou.node.return_value = node

        # A 16x16 box = 256 pixels total; page_size=10 -> page 0 should
        # read EXACTLY 10 pixels via bufferIndex, not the full 256.
        result = handlers.cop_onnx_read_pixels(
            node_path="/obj/copnet1/agent_onnx",
            mode="roi",
            roi=[0, 0, 16, 16],
            page=0,
            page_size=10,
        )

        assert result["ok"] is True, result
        assert layer.buffer_index_call_count == 10, (
            f"roi page 0 (page_size=10) over a 16x16=256-pixel box must call "
            f"bufferIndex EXACTLY 10 times (the page's own coords), NOT the "
            f"whole box (256). Got {layer.buffer_index_call_count} calls."
        )
        assert len(result["pixels"]) == 10

    def test_roi_inverted_box_is_ok_false(self, mock_hou):
        """REVIEW FIX 3 (codex red-review Major, threadId 019f2096): the
        LOCKED contract distinguishes an INVERTED/EMPTY box (x1<=x0 or
        y1<=y0 -- reject, ok:False) from an out-of-bounds-but-VALID box
        (clamp, ok:True) -- see
        test_roi_out_of_bounds_but_valid_box_is_clamped_ok_true below for
        the clamp half of this pair. x1 < x0 AND y1 < y0 (strictly
        inverted, not merely equal) is pinned here."""
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)
        layer = _FakeImageLayer(width=64, height=64, channels=3)
        node.layer.return_value = layer

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_read_pixels(
            node_path="/obj/copnet1/agent_onnx",
            mode="roi",
            roi=[16, 16, 4, 4],  # x1 < x0, y1 < y0 -- inverted
        )

        assert result["ok"] is False, (
            f"an inverted roi box (x1<x0, y1<y0) must be rejected, got {result!r}"
        )

    def test_roi_empty_box_x1_equals_x0_is_ok_false(self, mock_hou):
        """REVIEW FIX 3: an EMPTY box (x1<=x0 or y1<=y0, zero width/height)
        must ALSO be rejected -- the LOCKED contract's reject condition is
        'x1<=x0 or y1<=y0', not merely 'x1<x0 or y1<y0' (a zero-area box
        [10,10,10,10] is empty, not merely non-inverted-but-zero-size)."""
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)
        layer = _FakeImageLayer(width=64, height=64, channels=3)
        node.layer.return_value = layer

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_read_pixels(
            node_path="/obj/copnet1/agent_onnx",
            mode="roi",
            roi=[10, 10, 10, 10],  # x1==x0, y1==y0 -- zero-area, empty
        )

        assert result["ok"] is False, (
            f"an empty/zero-area roi box (x1==x0, y1==y0) must be rejected, "
            f"got {result!r}"
        )

    def test_roi_single_axis_empty_box_is_ok_false(self, mock_hou):
        """REVIEW RE-REVIEW FIX (round 2, codex threadId 019f20a4): the
        prior empty-box test only covered [10,10,10,10] -- BOTH axes empty
        simultaneously. A wrong impl using an `and` condition
        (x1<=x0 AND y1<=y0) instead of the LOCKED `or` condition
        (x1<=x0 OR y1<=y0) would still incorrectly PASS that test (both
        axes ARE <=0-width there, so the buggy `and` would still evaluate
        True and reject it). This test pins the SINGLE-axis-empty cases,
        which an `and`-based impl would WRONGLY ACCEPT (since only one of
        the two `and`-conjuncts would be True):
          - roi=[10,10,10,20]: x1==x0 (x-axis empty), y1>y0 (y-axis valid)
            -> must be ok:False.
          - roi=[10,10,20,10]: y1==y0 (y-axis empty), x1>x0 (x-axis valid)
            -> must be ok:False.
        Both must reject -- proving the reject condition is evaluated as
        `x1<=x0 OR y1<=y0`, not `x1<=x0 AND y1<=y0`."""
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        layer_x = _FakeImageLayer(width=64, height=64, channels=3)
        node_x = _make_fake_onnx_node(cop_sentinel)
        node_x.layer.return_value = layer_x
        mock_hou.node.return_value = node_x

        result_x_empty = handlers.cop_onnx_read_pixels(
            node_path="/obj/copnet1/agent_onnx",
            mode="roi",
            roi=[10, 10, 10, 20],  # x1==x0 (x-axis empty), y1>y0 (y-axis valid)
        )
        assert result_x_empty["ok"] is False, (
            f"a roi box empty on ONLY the x-axis (x1==x0, y1>y0) must be "
            f"rejected -- an `and`-based reject condition would wrongly "
            f"ACCEPT this (only the x-conjunct is True). Got {result_x_empty!r}."
        )

        layer_y = _FakeImageLayer(width=64, height=64, channels=3)
        node_y = _make_fake_onnx_node(cop_sentinel)
        node_y.layer.return_value = layer_y
        mock_hou.node.return_value = node_y

        result_y_empty = handlers.cop_onnx_read_pixels(
            node_path="/obj/copnet1/agent_onnx",
            mode="roi",
            roi=[10, 10, 20, 10],  # y1==y0 (y-axis empty), x1>x0 (x-axis valid)
        )
        assert result_y_empty["ok"] is False, (
            f"a roi box empty on ONLY the y-axis (y1==y0, x1>x0) must be "
            f"rejected -- an `and`-based reject condition would wrongly "
            f"ACCEPT this (only the y-conjunct is True). Got {result_y_empty!r}."
        )

    def test_roi_out_of_bounds_but_valid_box_is_clamped_ok_true(self, mock_hou):
        """REVIEW FIX 3 (codex red-review Major, threadId 019f2096): the
        LOCKED contract (plan pp12-113e lockedFieldContract, handler roi
        mode: 'clamp to [0,w]x[0,h]') is UNAMBIGUOUS -- an out-of-bounds-
        but-otherwise-VALID box (x0<x1, y0<y1, but extending past the
        plane) must be CLAMPED, i.e. ok:True with the effective box bounded
        to the plane -- NOT rejected. This pins the exact behavior (was
        previously an if-either-branch-passes assertion, which could not
        distinguish 'the impl clamps' from 'the impl rejects' -- both
        looked green)."""
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)
        layer = _FakeImageLayer(width=64, height=64, channels=3)
        node.layer.return_value = layer

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_read_pixels(
            node_path="/obj/copnet1/agent_onnx",
            mode="roi",
            roi=[0, 0, 10_000, 10_000],  # far beyond the 64x64 plane, but valid (x0<x1,y0<y1)
            page=0,
            page_size=1024,
        )

        assert result["ok"] is True, (
            f"an out-of-bounds-but-VALID roi box (x0<x1, y0<y1, extending "
            f"past the plane) must be CLAMPED (ok:True), NOT rejected. "
            f"Got {result!r}."
        )
        assert result["cooked"] is True

        # The clamped box's total pixel count must equal exactly the FULL
        # 64x64 plane (the effective box, once clamped to [0,64]x[0,64],
        # is the entire plane) -- proven via total_pages * page_size math
        # and, definitively, every bufferIndex call landing within bounds.
        assert layer.buffer_index_calls, (
            "a clamped roi must still call bufferIndex for its (bounded) page"
        )
        for x, y in layer.buffer_index_calls:
            assert 0 <= x < 64 and 0 <= y < 64, (
                f"an out-of-bounds roi must be CLAMPED to the plane -- "
                f"bufferIndex must never be called with an out-of-plane "
                f"coordinate, got call ({x}, {y}) on a 64x64 plane"
            )

        # The clamped effective box is the full 64x64=4096-pixel plane;
        # with page_size=1024 that must yield exactly 4 total pages, and
        # page 0 must return exactly 1024 pixels (page_size, not the
        # smaller ABS_MAX_PIXELS budget mixed in ambiguously -- page_size
        # itself is well under ABS_MAX_PIXELS=4096 here).
        assert result["total_pages"] == 4, (
            f"a clamped box of 64x64=4096 pixels at page_size=1024 must "
            f"report EXACTLY 4 total_pages, got {result.get('total_pages')!r} "
            f"-- this pins that the box was clamped to EXACTLY the 64x64 "
            f"plane (not e.g. left un-clamped at 10000x10000, which would "
            f"produce a vastly larger total_pages)."
        )
        assert len(result["pixels"]) == 1024, (
            f"page 0 of a clamped 4096-pixel box at page_size=1024 must "
            f"return EXACTLY 1024 pixels, got {len(result['pixels'])}"
        )


# ---------------------------------------------------------------------------
# sample mode — bounded/lazy strided sample
# ---------------------------------------------------------------------------

class TestSampleModeBoundedLazy:
    """sample mode must be LAZY and BOUNDED via bounded_sample_coords --
    bufferIndex is called ONLY for the requested page's strided
    coordinates, NEVER the whole strided grid."""

    def test_sample_bounded_page_calls_buffer_index_only_for_page_slice(self, mock_hou):
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)
        layer = _FakeImageLayer(width=64, height=64, channels=3)
        node.layer.return_value = layer

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_read_pixels(
            node_path="/obj/copnet1/agent_onnx",
            mode="sample",
            downsample=2,
            page=0,
            page_size=10,
        )

        assert result["ok"] is True, result
        assert layer.buffer_index_call_count == 10, (
            f"sample page 0 (page_size=10) must call bufferIndex EXACTLY 10 "
            f"times (the page's own strided coords), NOT the entire strided "
            f"grid. Got {layer.buffer_index_call_count} calls."
        )
        assert len(result["pixels"]) == 10

    def test_sample_stride_field_present_and_matches_downsample(self, mock_hou):
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)
        layer = _FakeImageLayer(width=64, height=64, channels=3)
        node.layer.return_value = layer

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_read_pixels(
            node_path="/obj/copnet1/agent_onnx",
            mode="sample",
            downsample=4,
            page=0,
            page_size=1024,
        )

        assert result["ok"] is True, result
        assert result["stride"] == 4, (
            f"an explicit downsample=4 must be honored as the stride, "
            f"got stride={result.get('stride')!r}"
        )


# ---------------------------------------------------------------------------
# M4-summary-memory: summary must NOT read the full plane
# ---------------------------------------------------------------------------

class TestSummaryNeverReadsFullPlane:
    """summary must strive via clamp_readback(...,budget)+bufferIndex over
    a budget-bounded sample -- NEVER read a full plane of xres*yres pixels
    (codex M4). Proven via bufferIndex call-count bounded well below
    xres*yres for a plane that would otherwise be huge."""

    def test_summary_over_large_plane_calls_buffer_index_far_fewer_than_full_plane(
        self, mock_hou
    ):
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)
        # A 512x512x3 plane = 262,144 pixels if read in full. Budget the
        # request down via max_pixels so the strided sample must be small.
        layer = _FakeImageLayer(width=512, height=512, channels=3)
        node.layer.return_value = layer

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_read_pixels(
            node_path="/obj/copnet1/agent_onnx", mode="summary", max_pixels=1024
        )

        assert result["ok"] is True, result
        full_plane_pixel_count = 512 * 512
        assert layer.buffer_index_call_count <= 1024, (
            f"summary with max_pixels=1024 over a 512x512 plane must call "
            f"bufferIndex AT MOST 1024 times (budget-bounded strided sample), "
            f"NEVER the full {full_plane_pixel_count} pixels. Got "
            f"{layer.buffer_index_call_count} calls."
        )


# ---------------------------------------------------------------------------
# FR-6 STRESS TESTS (codex Blockers, MANDATORY per the plan's red-test
# objective): huge caller params over a LARGE fake plane must still clamp
# to <= ABS_MAX_PIXELS=4096 and NEVER call bufferIndex millions of times.
# ---------------------------------------------------------------------------

class TestFR6StressAbsoluteClampAllModes:
    """The load-bearing FR-6 invariant, stress-tested across all 3 modes:
    a caller passing HUGE max_pixels/page_size over a LARGE fake plane
    must still be HARD-CLAMPED to <= ABS_MAX_PIXELS=4096, with
    truncated:True, and bufferIndex called <= 4096 times (NEVER millions)
    -- proving both the pixel-count clamp AND the lazy coordinate
    computation (never materializing the full box/grid before slicing)."""

    _LARGE_DIM = 4096  # a 4096x4096 plane -> ~16.7M pixels if fully read

    def test_roi_huge_params_over_large_plane_clamps_to_abs_max_pixels(self, mock_hou):
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)
        layer = _FakeImageLayer(width=self._LARGE_DIM, height=self._LARGE_DIM, channels=3)
        node.layer.return_value = layer

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_read_pixels(
            node_path="/obj/copnet1/agent_onnx",
            mode="roi",
            roi=[0, 0, self._LARGE_DIM, self._LARGE_DIM],  # the FULL huge plane as the box
            max_pixels=10_000_000,
            page_size=10_000_000,
            page=0,
        )

        assert result["ok"] is True, result
        pixel_count = len(result["pixels"])
        assert pixel_count <= 4096, (
            f"roi with max_pixels=10_000_000 and page_size=10_000_000 over a "
            f"{self._LARGE_DIM}x{self._LARGE_DIM} box MUST be HARD-CLAMPED to "
            f"<= 4096 (ABS_MAX_PIXELS), regardless of the caller's huge "
            f"params. Got {pixel_count} pixels."
        )
        assert result.get("truncated") is True, (
            f"a request this large must report truncated:True, got {result!r}"
        )
        assert layer.buffer_index_call_count <= 4096, (
            f"bufferIndex must be called AT MOST 4096 times (NOT millions) -- "
            f"this proves the coords were computed LAZILY (only the bounded "
            f"page slice), not by first materializing the full "
            f"{self._LARGE_DIM * self._LARGE_DIM}-pixel box. Got "
            f"{layer.buffer_index_call_count} calls."
        )

    def test_sample_huge_params_over_large_plane_clamps_to_abs_max_pixels(self, mock_hou):
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)
        layer = _FakeImageLayer(width=self._LARGE_DIM, height=self._LARGE_DIM, channels=3)
        node.layer.return_value = layer

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_read_pixels(
            node_path="/obj/copnet1/agent_onnx",
            mode="sample",
            max_pixels=10_000_000,
            page_size=10_000_000,
            page=0,
            downsample=1,  # request the densest possible stride
        )

        assert result["ok"] is True, result
        pixel_count = len(result["pixels"])
        assert pixel_count <= 4096, (
            f"sample with max_pixels=10_000_000, page_size=10_000_000, "
            f"downsample=1 over a {self._LARGE_DIM}x{self._LARGE_DIM} plane "
            f"MUST be HARD-CLAMPED to <= 4096 (ABS_MAX_PIXELS). Got "
            f"{pixel_count} pixels."
        )
        assert result.get("truncated") is True, (
            f"a request this large must report truncated:True, got {result!r}"
        )
        assert layer.buffer_index_call_count <= 4096, (
            f"bufferIndex must be called AT MOST 4096 times (NOT millions), "
            f"proving lazy bounded coordinate computation. Got "
            f"{layer.buffer_index_call_count} calls."
        )

    def test_summary_over_large_plane_with_huge_max_pixels_stays_bounded(self, mock_hou):
        """summary is inherently bounded by its own budget-derived stride,
        but must ALSO respect the ABS_MAX_PIXELS ceiling when the caller
        passes a huge max_pixels -- sampled:True, stride>1, and
        bufferIndex call-count bounded by the budget (NOT the ~16.7M full
        plane)."""
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)
        layer = _FakeImageLayer(width=self._LARGE_DIM, height=self._LARGE_DIM, channels=3)
        node.layer.return_value = layer

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_read_pixels(
            node_path="/obj/copnet1/agent_onnx",
            mode="summary",
            max_pixels=10_000_000,  # far beyond ABS_MAX_PIXELS
        )

        assert result["ok"] is True, result
        assert result.get("sampled") is True, (
            f"summary over a {self._LARGE_DIM}x{self._LARGE_DIM} plane must be "
            f"sampled (strided), got {result!r}"
        )
        assert result.get("stride", 1) > 1, (
            f"summary over a huge plane must use stride > 1 to stay budget-"
            f"bounded, got stride={result.get('stride')!r}"
        )
        assert layer.buffer_index_call_count <= 4096, (
            f"summary's bufferIndex call-count must be bounded to "
            f"<= 4096 (ABS_MAX_PIXELS), NEVER anywhere close to the full "
            f"{self._LARGE_DIM * self._LARGE_DIM} (~16.7M) pixel plane. Got "
            f"{layer.buffer_index_call_count} calls."
        )
        # Summary response size must also stay tiny -- serialize and check
        # a generous upper bound (well under any full-frame dump size).
        import json
        serialized_len = len(json.dumps(result))
        assert serialized_len < 20_000, (
            f"a summary result must stay small (~1-2KB per the spec) even "
            f"for a huge plane -- got a serialized size of {serialized_len} "
            f"bytes, suggesting a full/near-full pixel dump leaked into the "
            f"response."
        )
