"""Hython-smoke tests for the cop_onnx_read_pixels READONLY dispatcher path
(PP12-113 PR-5, the FINAL tool of member 113).

Unit: pp12-113e
testVerificationSurface: hython-smoke
planSha: bc1f8d9e9fcd1b46216e3eab594ded4a31ee907e305e5d12fde0b040314c445c

Run under hython (Houdini's headless Python interpreter):
    hython tests/cop_onnx_read_pixels_hython_smoke.py

Two-mode guard:
  - If fxhoudinimcp_server.handlers.cop_onnx_handlers does not yet expose
    cop_onnx_read_pixels (hou-dev has not implemented it yet), the import
    raises ImportError with the message "expected RED (hou-dev has not
    implemented cop_onnx_read_pixels yet)". This is the RED gate for the
    hython surface.
  - Once hou-dev implements the handler, the tests run and PASS GREEN.

Dispatcher-based (NOT direct handler calls) — mirrors
cop_onnx_run_inference_hython_smoke.py (pp12-113d):
  - Importing cop_onnx_handlers triggers register_handler(
    'cop_onnx_read_pixels', ...) as a side-effect (in addition to the
    PR-2/PR-3/PR-4 registrations, which must remain present -- append-only).
  - Tests then call dispatch('cop_onnx_read_pixels', {...}) -- the REAL
    dispatcher path that exercises the full handler(**params) calling
    convention.

READONLY contract asserted here (per plan pp12-113e lockedFieldContract,
GROUNDED via the orchestrator's pre-red hython probe -- see
_artifacts/houdini-orchestrator/pp12-113e/read-pixels-api-memo.md):
  - capability_of('cop_onnx_read_pixels') == Capability.READONLY (NOT
    MUTATING -- this is the FIRST tool of the cop_onnx family to be
    genuinely read-only pixel access; distinct from PR-2's
    inspect/list_models which are also READONLY, and from PR-3/PR-4's
    GATED setup_node/set_provider/run_inference).
  - preview_of('cop_onnx_read_pixels') has NO preview_fn (READONLY tools
    have no 109-gate preview hook).
  - STALE: read_pixels against a FRESH, UNCOOKED cop/onnx node ->
    {ok:True, cooked:False, message contains 'not cooked'} -- and the
    node remains uncooked afterward (no accidental cook).
  - HAPPY (summary): read_pixels against the GROUNDED cooked graph
    (multi_input.onnx + 2 constants + per-input resample 64x64, cooked via
    node.cook(force=True) in THIS smoke's own setup -- read_pixels itself
    never cooks) -> cooked:True, per-channel stats + a 32-bin histogram,
    a small (<2KB) serialized response.
  - HAPPY (roi): a [0,0,16,16] roi over the cooked 64x64x3 plane ->
    paginates; pixels are channel tuples.
  - HAPPY (sample): a strided sample -> clamped pixel count.
  - The cop/onnx node validation (name=='onnx' AND
    type().category()==hou.copNodeTypeCategory()) rejects a bare-named
    non-Cop node.
  - NEVER presses 'reload' (houdini-001) and NEVER cooks -- read_pixels is
    genuinely read-only; the cooking (via node.cook directly, NOT via the
    gated run_inference tool) is this smoke's OWN setup step for the
    HAPPY-path fixture, not read_pixels's job.

Fixtures (SAME committed .onnx binary as PR-2/PR-3/PR-4,
tests/fixtures/):
  multi_input.onnx      — 2 inputs "input_a"/"input_b" [1,3,64,64] float32
                           -> 1 output "output" (Add op). STATIC shape --
                           this is the GROUNDED cookable fixture (per the
                           run-inference-api-memo).
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — same pattern as cop_onnx_run_inference_hython_smoke.py (pp12-113d)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_PKG_PYTHON = os.path.join(_REPO_ROOT, "python")
_HOUDINI_HANDLERS = os.path.join(_REPO_ROOT, "houdini", "scripts", "python")
_FIXTURES_DIR = os.path.join(_HERE, "fixtures")

for _p in (_REPO_ROOT, _PKG_PYTHON, _HOUDINI_HANDLERS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_MULTI_FIXTURE = os.path.join(_FIXTURES_DIR, "multi_input.onnx").replace("\\", "/")

# ---------------------------------------------------------------------------
# pytest / hython compat shim
# ---------------------------------------------------------------------------
try:
    import pytest
    _PYTEST_AVAILABLE = True
except ImportError:
    _PYTEST_AVAILABLE = False

    class _FakePytest:
        """Minimal pytest shim so decorator syntax works under plain hython."""
        class mark:
            class parametrize:
                def __init__(self, *a, **kw): pass
                def __call__(self, f): return f
            asyncio = lambda f: f  # noqa: E731

        @staticmethod
        def skip(reason: str = ""):
            raise SystemExit(f"SKIP: {reason}")

    pytest = _FakePytest()

# ---------------------------------------------------------------------------
# Pass/fail counters (used when running directly under hython)
# ---------------------------------------------------------------------------
_PASS_COUNT = 0
_FAIL_COUNT = 0
_ERRORS: list = []


def _record_pass(name: str) -> None:
    global _PASS_COUNT
    _PASS_COUNT += 1
    print(f"  PASS  {name}")


def _record_fail(name: str, reason: str) -> None:
    global _FAIL_COUNT
    _FAIL_COUNT += 1
    _ERRORS.append((name, reason))
    print(f"  FAIL  {name}: {reason}")


# ---------------------------------------------------------------------------
# Dual-mode import guard: RED = ImportError; GREEN = returns (dispatch,
# capability_of, preview_of)
# ---------------------------------------------------------------------------
def _get_dispatcher_surface():
    """Import cop_onnx_handlers (triggers handler registration) and return
    (dispatch, capability_of, preview_of).

    If cop_onnx_read_pixels is not yet registered (hou-dev has not
    implemented it), raises ImportError with a message indicating this is
    the expected RED state.
    """
    try:
        # Importing the handler module triggers register_handler side-effects.
        import fxhoudinimcp_server.handlers.cop_onnx_handlers as _handlers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"expected RED (hou-dev has not implemented cop_onnx_handlers yet): {exc}"
        ) from exc

    try:
        from fxhoudinimcp_server.dispatcher import capability_of, dispatch, preview_of
    except ImportError as exc:
        raise ImportError(
            f"fxhoudinimcp_server.dispatcher not found — is fxhoudinimcp_server on sys.path? {exc}"
        ) from exc

    # RED GATE: cop_onnx_read_pixels must be REGISTERED (capability_of
    # returns non-None) for this to be considered GREEN. PR-2/PR-3/PR-4's
    # commands existing is not sufficient.
    if capability_of("cop_onnx_read_pixels") is None:
        raise ImportError(
            "expected RED (hou-dev has not implemented cop_onnx_read_pixels yet): "
            "cop_onnx_handlers imported but cop_onnx_read_pixels is not registered"
        )

    # TEST-HARNESS FIX (hou-dev, mechanical, zero assertion/contract change):
    # wrap the returned plain module-level functions in staticmethod() so
    # cls._dispatch = dispatch (etc.) does NOT get bound as an instance
    # method via the descriptor protocol when accessed as self._dispatch(...)
    # -- an unbound self would otherwise be injected as an extra positional
    # arg. Mirrors the proven, working pattern in the sibling PR-2/PR-3/PR-4
    # files.
    return staticmethod(dispatch), staticmethod(capability_of), staticmethod(preview_of)


# ---------------------------------------------------------------------------
# Tiny result helpers (mirroring cop_onnx_run_inference_hython_smoke.py)
# ---------------------------------------------------------------------------

def _unwrap(result: dict) -> dict:
    """Unwrap the dispatcher envelope.

    The dispatcher wraps every handler result in:
        {'status': 'success'|'error', 'data': <handler_payload>, 'timing_ms': ...}
    The handler's real {ok, error, ...} dict is under result["data"].
    """
    if isinstance(result, dict) and "data" in result:
        return result["data"]
    return result


def _is_ok(result: dict) -> bool:
    payload = _unwrap(result)
    return isinstance(payload, dict) and payload.get("ok") is True


def _get_error(result: dict) -> str:
    payload = _unwrap(result)
    return str(payload.get("error", ""))


def _make_copnet_parent(name: str):
    """Create a real copnet under /obj. Caller is responsible for
    destroying it after the test."""
    import hou
    obj = hou.node("/obj")
    return obj.createNode("copnet", name)


def _build_cooked_onnx_node(net, node_name: str = "agent_onnx"):
    """Build the GROUNDED cookable graph AND cook it directly (via
    node.cook(force=True) -- NOT via the gated run_inference tool, since
    this smoke's own setup step is not read_pixels's job and does not need
    to go through the 109 gate):
    copnet -> constant x2 -> onnx.setInput(i, constant_i);
    onnx.parm('modelfile').set(multi_input.onnx);
    onnx.parm('setupshapes').pressButton();
    for i in 1..2: resample_enable{i}=1, resample_size{i}1/2=64;
    node.cook(force=True).

    Returns the onnx node, ALREADY cooked and ready for read_pixels to
    read. Mirrors _build_cookable_onnx_node from
    cop_onnx_run_inference_hython_smoke.py (pp12-113d) plus an explicit
    cook step (that file's version left cooking to the run_inference
    handler under test; here read_pixels is under test, so this smoke
    cooks the fixture itself).
    """
    const_a = net.createNode("constant", "const_a")
    const_b = net.createNode("constant", "const_b")

    onnx_node = net.createNode("onnx", node_name)
    onnx_node.setInput(0, const_a)
    onnx_node.setInput(1, const_b)

    onnx_node.parm("modelfile").set(_MULTI_FIXTURE)
    # NEVER press 'reload' -- houdini-001 segfault. setupshapes alone
    # (re)reads the file at modelfile and is safe/sufficient.
    onnx_node.parm("setupshapes").pressButton()

    for i in (1, 2):
        enable_parm = onnx_node.parm(f"resample_enable{i}")
        if enable_parm is not None:
            enable_parm.set(1)
        size1_parm = onnx_node.parm(f"resample_size{i}1")
        size2_parm = onnx_node.parm(f"resample_size{i}2")
        if size1_parm is not None:
            size1_parm.set(64)
        if size2_parm is not None:
            size2_parm.set(64)

    onnx_node.cook(force=True)

    return onnx_node


def _build_uncooked_onnx_node(net, node_name: str = "agent_onnx_stale"):
    """A fresh, UNCOOKED cop/onnx node (the STALE-path fixture): modelfile
    set + setupshapes pressed (so it has a real plane/output contract) but
    NEVER cooked. read_pixels against this must report cooked:False,
    'not cooked', and NOT trigger a cook itself."""
    const_a = net.createNode("constant", "const_a_stale")
    const_b = net.createNode("constant", "const_b_stale")

    onnx_node = net.createNode("onnx", node_name)
    onnx_node.setInput(0, const_a)
    onnx_node.setInput(1, const_b)

    onnx_node.parm("modelfile").set(_MULTI_FIXTURE)
    onnx_node.parm("setupshapes").pressButton()

    for i in (1, 2):
        enable_parm = onnx_node.parm(f"resample_enable{i}")
        if enable_parm is not None:
            enable_parm.set(1)
        size1_parm = onnx_node.parm(f"resample_size{i}1")
        size2_parm = onnx_node.parm(f"resample_size{i}2")
        if size1_parm is not None:
            size1_parm.set(64)
        if size2_parm is not None:
            size2_parm.set(64)

    # Deliberately NEVER cooked.
    return onnx_node


# ===========================================================================
# Test class: dispatcher import and handler registration
# ===========================================================================

class TestCopOnnxReadPixelsDispatcherImport:
    """RED GATE: cop_onnx_read_pixels must be registered on the
    dispatcher.

    On RED (before hou-dev implements):
        ImportError: "expected RED (hou-dev has not implemented
        cop_onnx_read_pixels yet)"

    On GREEN (after hou-dev implements):
        The module imports cleanly and the new command is registered.
    """

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def test_dispatch_callable(self):
        assert callable(self._dispatch)

    def test_pr2_pr3_pr4_commands_still_registered(self):
        """Append-only contract: PR-2's cop_onnx_list_models /
        cop_onnx_inspect_model, PR-3's cop_onnx_setup_node /
        cop_onnx_set_provider, and PR-4's cop_onnx_run_inference must
        still be registered -- hou-dev must NOT have clobbered them."""
        for cmd in (
            "cop_onnx_list_models",
            "cop_onnx_inspect_model",
            "cop_onnx_setup_node",
            "cop_onnx_set_provider",
            "cop_onnx_run_inference",
        ):
            assert self._capability_of(cmd) is not None, (
                f"{cmd} (PR-2/PR-3/PR-4) must remain registered (append-only contract)"
            )


# ===========================================================================
# Test class: capability + preview contract (READONLY — Capability.READONLY)
# ===========================================================================

class TestCopOnnxReadPixelsCapabilityContract:
    """cop_onnx_read_pixels must be registered as Capability.READONLY,
    NOT MUTATING -- this is a genuinely read-only tool (no cook, no
    mutation) per the grounded design. Unlike PR-3/PR-4's GATED tools, it
    has NO preview_fn."""

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def test_read_pixels_is_readonly(self):
        from fxhoudinimcp_server.dispatcher import Capability
        cap = self._capability_of("cop_onnx_read_pixels")
        assert cap == Capability.READONLY, (
            f"cop_onnx_read_pixels must be Capability.READONLY (ungated), got {cap!r}"
        )

    def test_read_pixels_has_no_preview_fn(self):
        """READONLY tools (like PR-2's inspect/list) have NO preview_fn --
        the 109 gate never queues a READONLY call."""
        preview = self._preview_of("cop_onnx_read_pixels")
        assert preview.get("preview_fn") is None, (
            f"cop_onnx_read_pixels must NOT register a preview_fn (READONLY -- "
            f"no 109-gate queue). Got preview={preview!r}."
        )


# ===========================================================================
# Test class: STALE path — a fresh, uncooked node
# ===========================================================================

class TestReadPixelsStalePath:
    """read_pixels against a FRESH, UNCOOKED cop/onnx node -> {ok:True,
    cooked:False, message contains 'not cooked'} -- and read_pixels
    NEVER triggers a cook itself (genuinely read-only)."""

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def test_fixture_exists(self):
        assert os.path.isfile(_MULTI_FIXTURE), (
            f"fixture missing: {_MULTI_FIXTURE!r} — this fixture is committed "
            "as a binary fixture shared with PR-2/PR-3/PR-4."
        )

    def test_stale_node_returns_not_cooked(self):
        net = _make_copnet_parent("_mcp_read_pixels_smoke_stale")
        try:
            onnx_node = _build_uncooked_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_read_pixels",
                {"node_path": onnx_node.path()},
            )
            payload = _unwrap(result)
            assert _is_ok(result), (
                f"a stale/uncooked node is a valid outcome (not an error), "
                f"must be ok:True, got payload={payload!r}"
            )
            assert payload["cooked"] is False, (
                f"a fresh, uncooked node must report cooked:False, got payload={payload!r}"
            )
            assert "not cooked" in str(payload.get("message", "")).lower(), (
                f"expected an informative 'not cooked' message, got payload={payload!r}"
            )
        finally:
            net.destroy()

    def test_stale_node_read_pixels_does_not_cook_it(self):
        """read_pixels must NEVER trigger a cook -- verified by checking
        that the node remains needsToCook()==True (or equivalently, that
        no output plane is populated) after the call."""
        net = _make_copnet_parent("_mcp_read_pixels_smoke_stale_nocook")
        try:
            onnx_node = _build_uncooked_onnx_node(net)
            needs_cook_before = onnx_node.needsToCook()

            self._dispatch(
                "cop_onnx_read_pixels",
                {"node_path": onnx_node.path()},
            )

            needs_cook_after = onnx_node.needsToCook()
            assert needs_cook_after == needs_cook_before, (
                f"read_pixels must NEVER trigger a cook -- needsToCook() must be "
                f"unchanged by the call. Before={needs_cook_before!r}, "
                f"After={needs_cook_after!r}."
            )
        finally:
            net.destroy()


# ===========================================================================
# Test class: HAPPY path — summary mode over a real cooked node
# ===========================================================================

class TestReadPixelsHappyPathSummary:
    """read_pixels against the GROUNDED cooked graph -> summary mode:
    cooked:True, per-channel stats + a 32-bin histogram, a small
    (<2KB) serialized response."""

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def test_summary_returns_cooked_true_with_stats_and_histogram(self):
        net = _make_copnet_parent("_mcp_read_pixels_smoke_summary")
        try:
            onnx_node = _build_cooked_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_read_pixels",
                {"node_path": onnx_node.path(), "mode": "summary"},
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            assert payload["cooked"] is True, (
                f"a real cooked node must report cooked:True, got payload={payload!r}"
            )
            assert payload["mode"] == "summary"
            assert payload["channels"] == 3, (
                f"the grounded fixture's output plane has 3 channels, got {payload!r}"
            )
            histogram = payload["histogram"]
            assert histogram["bins"] == 32, (
                f"SUMMARY_HISTOGRAM_BINS must be LOCKED at 32, got {histogram!r}"
            )
            stats = payload["stats"]
            assert len(stats["min"]) == 3
            assert len(stats["max"]) == 3
            assert len(stats["mean"]) == 3
        finally:
            net.destroy()

    def test_summary_response_is_small(self):
        """A summary response must be small (~1-2KB) regardless of the
        underlying plane resolution -- proving no full-frame dump leaked
        into the response."""
        import json

        net = _make_copnet_parent("_mcp_read_pixels_smoke_summary_size")
        try:
            onnx_node = _build_cooked_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_read_pixels",
                {"node_path": onnx_node.path(), "mode": "summary"},
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            serialized_len = len(json.dumps(payload))
            assert serialized_len < 5000, (
                f"a summary response over a 64x64x3 plane should be well under "
                f"5KB (spec targets ~1-2KB); got {serialized_len} bytes -- "
                f"suggests a full-frame pixel dump leaked into the response."
            )
        finally:
            net.destroy()

    def test_default_mode_is_summary(self):
        """mode omitted entirely must default to 'summary' (per the
        lockedFieldContract's wrapper default)."""
        net = _make_copnet_parent("_mcp_read_pixels_smoke_defaultmode")
        try:
            onnx_node = _build_cooked_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_read_pixels",
                {"node_path": onnx_node.path()},
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            assert payload.get("mode") == "summary", (
                f"omitting mode must default to 'summary', got payload={payload!r}"
            )
        finally:
            net.destroy()


# ===========================================================================
# Test class: HAPPY path — roi mode over a real cooked node
# ===========================================================================

class TestReadPixelsHappyPathRoi:
    """A [0,0,16,16] roi over the cooked 64x64x3 plane -> paginates;
    pixels are channel tuples/lists."""

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def test_roi_16x16_returns_bounded_pixels(self):
        net = _make_copnet_parent("_mcp_read_pixels_smoke_roi")
        try:
            onnx_node = _build_cooked_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_read_pixels",
                {
                    "node_path": onnx_node.path(),
                    "mode": "roi",
                    "roi": [0, 0, 16, 16],
                    "page": 0,
                    "page_size": 1024,
                },
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            assert payload["cooked"] is True
            pixels = payload["pixels"]
            assert len(pixels) == 16 * 16, (
                f"a 16x16 roi with page_size >= 256 must return all 256 pixels "
                f"in one page, got {len(pixels)}"
            )
            first_pixel = pixels[0]
            assert len(first_pixel) == 3, (
                f"each pixel row must have 3 channel values (RGB), got {first_pixel!r}"
            )
        finally:
            net.destroy()

    def test_roi_paginates_when_page_size_smaller_than_box(self):
        net = _make_copnet_parent("_mcp_read_pixels_smoke_roi_paginate")
        try:
            onnx_node = _build_cooked_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_read_pixels",
                {
                    "node_path": onnx_node.path(),
                    "mode": "roi",
                    "roi": [0, 0, 16, 16],  # 256 pixels
                    "page": 0,
                    "page_size": 50,
                },
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            assert len(payload["pixels"]) == 50, (
                f"page_size=50 over a 256-pixel box must return exactly 50 "
                f"pixels for page 0, got {len(payload['pixels'])}"
            )
            assert payload["total_pages"] > 1, (
                f"a 256-pixel box with page_size=50 must span multiple pages, "
                f"got total_pages={payload.get('total_pages')!r}"
            )
            assert payload["truncated"] is True
        finally:
            net.destroy()


# ===========================================================================
# Test class: HAPPY path — sample mode over a real cooked node
# ===========================================================================

class TestReadPixelsHappyPathSample:
    """A strided sample over the cooked plane -> clamped pixel count."""

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def test_sample_with_explicit_downsample(self):
        net = _make_copnet_parent("_mcp_read_pixels_smoke_sample")
        try:
            onnx_node = _build_cooked_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_read_pixels",
                {
                    "node_path": onnx_node.path(),
                    "mode": "sample",
                    "downsample": 4,
                    "page": 0,
                    "page_size": 1024,
                },
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            assert payload["cooked"] is True
            assert payload["stride"] == 4, (
                f"explicit downsample=4 must set stride=4, got {payload!r}"
            )
            # 64x64 at stride 4 -> ceil(64/4)=16 columns, 16 rows -> 256 points.
            assert len(payload["pixels"]) == 256, (
                f"64x64 plane at stride=4 must yield 16x16=256 sample points, "
                f"got {len(payload['pixels'])}"
            )
        finally:
            net.destroy()

    def test_sample_max_pixels_clamps_result(self):
        net = _make_copnet_parent("_mcp_read_pixels_smoke_sample_clamp")
        try:
            onnx_node = _build_cooked_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_read_pixels",
                {
                    "node_path": onnx_node.path(),
                    "mode": "sample",
                    "max_pixels": 16,
                    "page": 0,
                    "page_size": 1024,
                },
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            assert len(payload["pixels"]) <= 16, (
                f"max_pixels=16 must clamp the sample result to at most 16 "
                f"points, got {len(payload['pixels'])}"
            )
        finally:
            net.destroy()


# ===========================================================================
# Test class: target validation — bad node_path / non-onnx target
# ===========================================================================

class TestReadPixelsTargetValidation:
    """A bad node_path (unresolved) or a non-onnx / non-Cop-category node
    is ok:False."""

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def test_nonexistent_node_path_is_ok_false(self):
        result = self._dispatch(
            "cop_onnx_read_pixels",
            {"node_path": "/obj/definitely_does_not_exist_probe"},
        )
        payload = _unwrap(result)
        assert payload.get("ok") is False, (
            f"an unresolved node_path must be ok:False (a bad TARGET), got {payload!r}"
        )

    def test_non_onnx_node_is_ok_false(self):
        """A real node that is NOT type 'onnx' (e.g. the copnet itself)
        must be ok:False."""
        net = _make_copnet_parent("_mcp_read_pixels_smoke_nononnx")
        try:
            result = self._dispatch(
                "cop_onnx_read_pixels",
                {"node_path": net.path()},
            )
            payload = _unwrap(result)
            assert payload.get("ok") is False, (
                f"a non-onnx node target must be ok:False, got {payload!r}"
            )
        finally:
            net.destroy()

    def test_wrong_category_onnx_node_is_ok_false(self):
        """A node named 'onnx' but resolved under a non-Cop context (a
        Sop/onnx name collision, per the PR-3/PR-4 setup_node/
        run_inference docstring finding) must be rejected."""
        import hou

        geo_net = hou.node("/obj").createNode("geo", "_mcp_read_pixels_smoke_sop_onnx")
        try:
            sop_onnx = geo_net.createNode("onnx")
            assert sop_onnx.type().name() == "onnx", (
                f"test precondition: the SOP-context node must be named 'onnx', "
                f"got {sop_onnx.type().name()!r}"
            )
            assert sop_onnx.type().category() != hou.copNodeTypeCategory(), (
                "test precondition: the SOP-context 'onnx' node's category must "
                f"NOT be the Cop category, got category={sop_onnx.type().category()!r}"
            )

            result = self._dispatch(
                "cop_onnx_read_pixels",
                {"node_path": sop_onnx.path()},
            )
            payload = _unwrap(result)
            assert payload.get("ok") is False, (
                f"a node named 'onnx' but NOT of the Cop category must be "
                f"REJECTED (ok:False), got {payload!r}"
            )
        finally:
            geo_net.destroy()

    def test_bad_plane_name_is_ok_false(self):
        net = _make_copnet_parent("_mcp_read_pixels_smoke_badplane")
        try:
            onnx_node = _build_cooked_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_read_pixels",
                {"node_path": onnx_node.path(), "plane": "definitely_not_a_real_plane"},
            )
            payload = _unwrap(result)
            assert payload.get("ok") is False, (
                f"a bad plane name must be ok:False, got {payload!r}"
            )
        finally:
            net.destroy()


# ===========================================================================
# Test class: no reload pressed, no crash
# ===========================================================================

class TestReadPixelsNoReloadNoCrash:
    """houdini-001 (catalog): read_pixels must NEVER press 'reload' --
    verified indirectly: a successful call with no crash/segfault is
    evidence the reload button was never pressed."""

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def test_no_reload_pressed_no_crash(self):
        net = _make_copnet_parent("_mcp_read_pixels_smoke_noreload")
        try:
            onnx_node = _build_cooked_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_read_pixels",
                {"node_path": onnx_node.path(), "mode": "summary"},
            )
            assert _is_ok(result), _unwrap(result)
        finally:
            net.destroy()


# ===========================================================================
# Test registry and runner (for direct hython execution)
# ===========================================================================

_TEST_CLASSES = [
    TestCopOnnxReadPixelsDispatcherImport,
    TestCopOnnxReadPixelsCapabilityContract,
    TestReadPixelsStalePath,
    TestReadPixelsHappyPathSummary,
    TestReadPixelsHappyPathRoi,
    TestReadPixelsHappyPathSample,
    TestReadPixelsTargetValidation,
    TestReadPixelsNoReloadNoCrash,
]


def _run_class(cls):
    setup_failed = False
    try:
        if hasattr(cls, "setup_class"):
            cls.setup_class()
    except Exception as exc:
        _record_fail(f"{cls.__name__}.setup_class", repr(exc))
        setup_failed = True

    obj = cls()
    if setup_failed:
        return

    for name in dir(obj):
        if not name.startswith("test_"):
            continue
        method = getattr(obj, name)
        if not callable(method):
            continue
        full_name = f"{cls.__name__}.{name}"
        try:
            method()
            _record_pass(full_name)
        except SystemExit as exc:
            print(f"  SKIP  {full_name}: {exc}")
        except AssertionError as exc:
            _record_fail(full_name, str(exc))
        except Exception as exc:
            _record_fail(full_name, repr(exc))

    if hasattr(cls, "teardown_class"):
        try:
            cls.teardown_class()
        except Exception:
            pass


def main() -> int:
    print("=" * 72)
    print("cop_onnx_read_pixels_hython_smoke.py — pp12-113e")
    print("=" * 72)

    # Pre-check: is this the RED phase or GREEN phase?
    try:
        _get_dispatcher_surface()
        print("[mode] GREEN — cop_onnx_read_pixels registered; running all tests.")
    except ImportError as exc:
        print(f"[mode] RED — {exc}")
        print("Expected RED failure. Wrapper pytest red gate has already been confirmed.")
        print("Hython-smoke RED gate confirmed: cop_onnx_read_pixels not yet implemented.")
        return 1  # non-zero = failure (RED gate confirmed)

    for cls in _TEST_CLASSES:
        print(f"\n--- {cls.__name__} ---")
        _run_class(cls)

    print("\n" + "=" * 72)
    print(f"Results: {_PASS_COUNT} passed, {_FAIL_COUNT} failed")

    if _ERRORS:
        print("\nFailures:")
        for name, reason in _ERRORS:
            print(f"  FAIL  {name}")
            print(f"        {reason}")

    return 0 if _FAIL_COUNT == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
