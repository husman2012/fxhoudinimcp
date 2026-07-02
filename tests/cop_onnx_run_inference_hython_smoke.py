"""Hython-smoke tests for the cop_onnx_run_inference GATED dispatcher path
(PP12-113 PR-4).

Unit: pp12-113d
testVerificationSurface: hython-smoke
planSha: 92ce0bfd3ac81683321af721d8bff6bd50e7c67010fdfca5e82f53f97845adc9

Run under hython (Houdini's headless Python interpreter):
    hython tests/cop_onnx_run_inference_hython_smoke.py

Two-mode guard:
  - If fxhoudinimcp_server.handlers.cop_onnx_handlers does not yet expose
    cop_onnx_run_inference (hou-dev has not implemented it yet), the
    import raises ImportError with the message "expected RED (hou-dev has
    not implemented cop_onnx_run_inference yet)". This is the RED gate for
    the hython surface.
  - Once hou-dev implements the handler, the tests run and PASS GREEN.

Dispatcher-based (NOT direct handler calls) — mirrors
cop_onnx_setup_hython_smoke.py (pp12-113c) and cop_onnx_hython_smoke.py
(pp12-113b):
  - Importing cop_onnx_handlers triggers register_handler(
    'cop_onnx_run_inference', ...) as a side-effect (in addition to the
    PR-2/PR-3 registrations, which must remain present -- append-only).
  - Tests then call dispatch('cop_onnx_run_inference', {...}) -- the REAL
    dispatcher path that exercises the full handler(**params) calling
    convention.

GATED contract asserted here (per plan pp12-113d lockedFieldContract,
GROUNDED via the orchestrator's pre-red hython probe -- see
_artifacts/houdini-orchestrator/pp12-113d/run-inference-api-memo.md):
  - capability_of('cop_onnx_run_inference') == Capability.MUTATING
  - preview_of('cop_onnx_run_inference') carries a real preview_fn with
    preview_required=True
  - _preview_run_inference(params) RAISES (-> gate DENY) on an
    unresolved node_path AND on a resolved-but-non-cop/onnx node -- an
    INVALID target is denied at the gate, not merely flagged. The preview
    is READ-ONLY (never cooks).
  - HAPPY: run_inference against a cookable multi_input.onnx graph (2
    constants -> onnx, per-input resample wired to 64x64) cooks cleanly:
    cooked:true, errors:[], a non-empty output_planes manifest
    ({name:'output1', xres:64, yres:64, channels:3, dtype:'float32'}),
    bound_provider=='automatic'.
  - SAD: run_inference against an input-less / unconfigured onnx node
    surfaces cooked:false + a surfaced error (['src is missing']) -- ok:True
    (a failed cook is REPORTED, not raised).
  - The cop/onnx node validation (name=='onnx' AND
    type().category()==hou.copNodeTypeCategory()) rejects a bare-named
    non-Cop node (folds codex Blocker-1).
  - NEVER presses 'reload' or 'setupshapes' -- run_inference only cooks an
    already-configured node (houdini-001 + scope; setupshapes is PR-3's
    job).

Fixtures (SAME committed .onnx binaries as PR-2/PR-3, tests/fixtures/):
  multi_input.onnx      — 2 inputs "input_a"/"input_b" [1,3,64,64] float32
                           -> 1 output "output" (Add op). STATIC shape --
                           this is the GROUNDED cookable fixture (per the
                           memo; identity_dynamic.onnx has 2 dynamic axes
                           and is NOT cookable -- "Only can deduce a single
                           dynamic axis, 2 axes are dynamic").
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — same pattern as cop_onnx_setup_hython_smoke.py (pp12-113c)
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

    If cop_onnx_run_inference is not yet registered (hou-dev has not
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

    # RED GATE: cop_onnx_run_inference must be REGISTERED (capability_of
    # returns non-None) for this to be considered GREEN. PR-2/PR-3's
    # commands existing is not sufficient.
    if capability_of("cop_onnx_run_inference") is None:
        raise ImportError(
            "expected RED (hou-dev has not implemented cop_onnx_run_inference yet): "
            "cop_onnx_handlers imported but cop_onnx_run_inference is not registered"
        )

    # TEST-HARNESS FIX (hou-dev, mechanical, zero assertion/contract change):
    # wrap the returned plain module-level functions in staticmethod() so
    # cls._dispatch = dispatch (etc.) does NOT get bound as an instance
    # method via the descriptor protocol when accessed as self._dispatch(...)
    # -- an unbound self would otherwise be injected as an extra positional
    # arg (TypeError: dispatch() takes 2 positional arguments but 3 were
    # given). Mirrors the proven, working pattern in the sibling PR-2/PR-3
    # files (cop_onnx_hython_smoke.py / cop_onnx_setup_hython_smoke.py).
    return staticmethod(dispatch), staticmethod(capability_of), staticmethod(preview_of)


# ---------------------------------------------------------------------------
# Tiny result helpers (mirroring cop_onnx_setup_hython_smoke.py)
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


def _build_cookable_onnx_node(net, node_name: str = "agent_onnx"):
    """Build the GROUNDED cookable graph (per the run-inference-api-memo):
    copnet -> constant x2 -> onnx.setInput(i, constant_i);
    onnx.parm('modelfile').set(multi_input.onnx);
    onnx.parm('setupshapes').pressButton();
    for i in 1..2: resample_enable{i}=1, resample_size{i}1/2=64.

    Returns the onnx node, ALREADY configured and ready to cook cleanly.
    setupshapes is pressed here (this is the cookable-fixture setup, NOT
    run_inference's job -- run_inference only cooks an already-configured
    node per the LOCKED scope).
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

    return onnx_node


def _build_uncookable_onnx_node(net, node_name: str = "agent_onnx_bad"):
    """An input-less / unconfigured onnx node -- SAD path fixture. No
    inputs wired, no modelfile set, no setupshapes pressed. Cooking this
    must surface cooked:false + a surfaced error (['src is missing'] per
    the grounded probe), never a silent success."""
    return net.createNode("onnx", node_name)


# ===========================================================================
# Test class: dispatcher import and handler registration
# ===========================================================================

class TestCopOnnxRunInferenceDispatcherImport:
    """RED GATE: cop_onnx_run_inference must be registered on the
    dispatcher.

    On RED (before hou-dev implements):
        ImportError: "expected RED (hou-dev has not implemented
        cop_onnx_run_inference yet)"

    On GREEN (after hou-dev implements):
        The module imports cleanly and the new command is registered.
    """

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def test_dispatch_callable(self):
        assert callable(self._dispatch)

    def test_pr2_pr3_commands_still_registered(self):
        """Append-only contract: PR-2's cop_onnx_list_models /
        cop_onnx_inspect_model and PR-3's cop_onnx_setup_node /
        cop_onnx_set_provider must still be registered -- hou-dev must
        NOT have clobbered them."""
        for cmd in (
            "cop_onnx_list_models",
            "cop_onnx_inspect_model",
            "cop_onnx_setup_node",
            "cop_onnx_set_provider",
        ):
            assert self._capability_of(cmd) is not None, (
                f"{cmd} (PR-2/PR-3) must remain registered (append-only contract)"
            )


# ===========================================================================
# Test class: capability + preview contract (GATED — Capability.MUTATING)
# ===========================================================================

class TestCopOnnxRunInferenceCapabilityContract:
    """cop_onnx_run_inference must be registered as Capability.MUTATING
    WITH a preview_fn and preview_required=True — the SHIPPED PR-3 GATED
    pattern."""

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def test_run_inference_is_mutating(self):
        from fxhoudinimcp_server.dispatcher import Capability
        cap = self._capability_of("cop_onnx_run_inference")
        assert cap == Capability.MUTATING, (
            f"cop_onnx_run_inference must be Capability.MUTATING (GATED), got {cap!r}"
        )

    def test_run_inference_has_preview_required(self):
        preview = self._preview_of("cop_onnx_run_inference")
        assert preview.get("preview_fn") is not None, (
            "cop_onnx_run_inference must register a preview_fn (109 gate GATED pattern)"
        )
        assert preview.get("preview_required") is True, (
            "cop_onnx_run_inference must have preview_required=True"
        )


# ===========================================================================
# Test class: preview RAISES on an INVALID target; preview NEVER cooks
# ===========================================================================

class TestCopOnnxRunInferencePreviewDeniesInvalidTarget:
    """_preview_run_inference MUST RAISE (-> gate DENY) when node_path does
    NOT resolve OR is NOT a Copernicus cop/onnx node (folds codex
    Blocker-1: a bare 'onnx' name can resolve to an unrelated Sop/onnx
    under a SOP context -- name-only is insufficient). The preview is
    READ-ONLY and must NEVER cook the target."""

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def test_preview_raises_on_nonexistent_node(self):
        preview = self._preview_of("cop_onnx_run_inference")
        preview_fn = preview["preview_fn"]
        params = {"node_path": "/obj/definitely_does_not_exist_probe", "frame": None}
        raised = False
        try:
            preview_fn(params)
        except Exception:
            raised = True
        assert raised, (
            "_preview_run_inference must RAISE (-> gate DENY) when node_path "
            "does not resolve to a real node"
        )

    def test_preview_raises_on_non_onnx_node(self):
        """A real but non-'onnx'-type node_path (e.g. a copnet itself)
        must also raise."""
        net = _make_copnet_parent("_mcp_run_inference_smoke_wrongtype")
        try:
            preview = self._preview_of("cop_onnx_run_inference")
            preview_fn = preview["preview_fn"]
            params = {"node_path": net.path(), "frame": None}
            raised = False
            try:
                preview_fn(params)
            except Exception:
                raised = True
            assert raised, (
                "_preview_run_inference must RAISE when node_path resolves but "
                "is NOT a Copernicus cop/onnx node"
            )
        finally:
            net.destroy()

    def test_preview_raises_on_wrong_category_onnx_node(self):
        """FOLD codex Blocker-1 (the wrong-category preview gap): a node
        whose type().name()=='onnx' but whose type().category() is NOT the
        Cop category must ALSO raise (-> gate DENY). A name-only preview
        guard would incorrectly PASS this node through.

        Per the PR-3 setup_node docstring (cop_onnx_handlers.py):
        createNode('onnx') under a NON-cop parent (e.g. a geo/SOP network)
        can SILENTLY SUCCEED -- Houdini resolves the bare name 'onnx' to
        an UNRELATED Sop/onnx node type under a SOP context. This builds
        exactly that: a real geo/SOP network with a child literally named
        'onnx' (type().name()=='onnx', type().category()=='Sop', NOT
        'Cop') and asserts the preview denies it."""
        import hou

        geo_net = hou.node("/obj").createNode("geo", "_mcp_run_inference_smoke_sop_onnx")
        try:
            sop_onnx = geo_net.createNode("onnx")
            # Sanity precondition: this really is the name-collision case
            # the fold targets -- name matches 'onnx' but category is NOT
            # the Cop category (it resolved to the unrelated Sop/onnx type
            # under this SOP context).
            assert sop_onnx.type().name() == "onnx", (
                f"test precondition: the SOP-context node must be named 'onnx' "
                f"(the name-collision case), got {sop_onnx.type().name()!r}"
            )
            assert sop_onnx.type().category() != hou.copNodeTypeCategory(), (
                "test precondition: the SOP-context 'onnx' node's category must "
                "NOT be the Cop category (this is the wrong-category collision "
                f"the fold targets), got category={sop_onnx.type().category()!r}"
            )

            preview = self._preview_of("cop_onnx_run_inference")
            preview_fn = preview["preview_fn"]
            params = {"node_path": sop_onnx.path(), "frame": None}
            raised = False
            try:
                preview_fn(params)
            except Exception:
                raised = True
            assert raised, (
                "_preview_run_inference must RAISE (-> gate DENY) when node_path "
                "resolves to a node named 'onnx' but whose category is NOT the "
                "Cop category (a Sop/onnx name collision) -- a name-only guard "
                "would incorrectly pass this through (codex Blocker-1)."
            )
        finally:
            geo_net.destroy()

    def test_preview_does_not_cook(self):
        """Preview is READ-ONLY -- calling it must NOT cook the target.

        FOLD (the no-cook-assertion-not-observable gap): using a CLEAN
        cookable node here is not observable -- if the preview wrongly
        cooked but the cook happened to leave errors()==[], the
        before/after comparison would still pass (the accidental cook is
        invisible). Instead this uses an UNCONFIGURED / input-less onnx
        node, where an ACTUAL cook WOULD populate errors() with
        'src is missing' (per the grounded probe / _build_uncookable_onnx_node).
        Asserting errors() is STILL EMPTY after calling the preview proves
        no cook happened -- any accidental cook is now detectable."""
        net = _make_copnet_parent("_mcp_run_inference_smoke_preview_ro")
        try:
            onnx_node = _build_uncookable_onnx_node(net)
            errors_before = list(onnx_node.errors())
            assert errors_before == [], (
                "test precondition: an uncooked, unconfigured onnx node must "
                f"start with errors()==[], got {errors_before!r}"
            )

            preview = self._preview_of("cop_onnx_run_inference")
            preview_fn = preview["preview_fn"]
            params = {"node_path": onnx_node.path(), "frame": None}
            preview_fn(params)

            errors_after = list(onnx_node.errors())
            assert errors_after == [], (
                f"_preview_run_inference must NOT cook the node -- if it had "
                f"accidentally cooked this input-less/unconfigured node, "
                f"errors() would now contain 'src is missing'. Got "
                f"errors_after={errors_after!r} (expected still empty)."
            )
        finally:
            net.destroy()


# ===========================================================================
# Test class: HAPPY path — a real clean cook
# ===========================================================================

class TestRunInferenceHappyPath:
    """cop_onnx_run_inference against the GROUNDED cookable graph
    (multi_input.onnx + 2 constants + per-input resample 64x64) -- a
    clean cook: cooked:true, errors:[], a non-empty output_planes manifest,
    bound_provider=='automatic'."""

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def test_fixture_exists(self):
        assert os.path.isfile(_MULTI_FIXTURE), (
            f"fixture missing: {_MULTI_FIXTURE!r} — this fixture is committed "
            "as a binary fixture shared with PR-2/PR-3."
        )

    def test_clean_cook_returns_ok_true_cooked_true(self):
        net = _make_copnet_parent("_mcp_run_inference_smoke_happy")
        try:
            onnx_node = _build_cookable_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_run_inference",
                {"node_path": onnx_node.path(), "frame": None},
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            assert payload["cooked"] is True, (
                f"a clean cook on the grounded cookable graph must report cooked:true, "
                f"got payload={payload!r}"
            )
            assert payload["errors"] == [], (
                f"a clean cook must report errors:[], got {payload['errors']!r}"
            )
        finally:
            net.destroy()

    def test_clean_cook_output_plane_manifest(self):
        """output_planes must contain the grounded manifest:
        {name:'output1', xres:64, yres:64, channels:3, dtype:'float32'}."""
        net = _make_copnet_parent("_mcp_run_inference_smoke_manifest")
        try:
            onnx_node = _build_cookable_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_run_inference",
                {"node_path": onnx_node.path(), "frame": None},
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            output_planes = payload["output_planes"]
            assert output_planes, (
                "a clean cook must report a non-empty output_planes manifest"
            )
            plane = output_planes[0]
            assert plane["name"] == "output1", plane
            assert plane["xres"] == 64, plane
            assert plane["yres"] == 64, plane
            assert plane["channels"] == 3, plane
            assert plane["dtype"] == "float32", (
                f"dtype must be normalized to the plain token 'float32' "
                f"(via normalize_plane_dtype), got {plane['dtype']!r}"
            )
        finally:
            net.destroy()

    def test_clean_cook_bound_provider(self):
        """bound_provider must be read via evalAsString() -- the token
        'automatic', NOT the menu index (.eval() would return 0)."""
        net = _make_copnet_parent("_mcp_run_inference_smoke_provider")
        try:
            onnx_node = _build_cookable_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_run_inference",
                {"node_path": onnx_node.path(), "frame": None},
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            assert payload["bound_provider"] == "automatic", (
                f"bound_provider must be the evalAsString() TOKEN 'automatic' "
                f"(not the .eval() index 0), got {payload['bound_provider']!r}"
            )
        finally:
            net.destroy()

    def test_clean_cook_cook_ms_is_a_positive_number(self):
        net = _make_copnet_parent("_mcp_run_inference_smoke_cookms")
        try:
            onnx_node = _build_cookable_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_run_inference",
                {"node_path": onnx_node.path(), "frame": None},
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            assert isinstance(payload["cook_ms"], (int, float)), payload
            assert payload["cook_ms"] >= 0, payload
        finally:
            net.destroy()

    def test_frame_defaults_to_current_frame(self):
        """frame=None must cook at the current hou.frame() (default 1.0)
        -- proven indirectly by a successful clean cook with frame omitted
        from the call (None passed explicitly, matching the wrapper's
        default)."""
        net = _make_copnet_parent("_mcp_run_inference_smoke_frame")
        try:
            onnx_node = _build_cookable_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_run_inference",
                {"node_path": onnx_node.path(), "frame": None},
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            assert payload["cooked"] is True
        finally:
            net.destroy()

    def test_no_reload_pressed_no_crash(self):
        """houdini-001 (catalog): the handler must NEVER press 'reload' --
        verified indirectly: a successful call with no crash/segfault is
        evidence the reload button was never pressed. run_inference also
        must never press 'setupshapes' (that is the cookable-fixture
        setup's job, done above in _build_cookable_onnx_node, NOT
        run_inference's)."""
        net = _make_copnet_parent("_mcp_run_inference_smoke_noreload")
        try:
            onnx_node = _build_cookable_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_run_inference",
                {"node_path": onnx_node.path(), "frame": None},
            )
            assert _is_ok(result), _unwrap(result)
        finally:
            net.destroy()


# ===========================================================================
# Test class: SAD path — an input-less / unconfigured cook
# ===========================================================================

class TestRunInferenceSadPath:
    """cop_onnx_run_inference against an input-less / unconfigured onnx
    node -- cooked:false + a surfaced error, ok:True (a failed cook is
    REPORTED, not raised)."""

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def test_failed_cook_returns_ok_true_cooked_false(self):
        net = _make_copnet_parent("_mcp_run_inference_smoke_sad")
        try:
            onnx_node = _build_uncookable_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_run_inference",
                {"node_path": onnx_node.path(), "frame": None},
            )
            payload = _unwrap(result)
            assert _is_ok(result), (
                f"a failed cook must be REPORTED as ok:True (never raised as a "
                f"handler exception), got payload={payload!r}"
            )
            assert payload["cooked"] is False, (
                f"an input-less/unconfigured node cook must report cooked:false, "
                f"got payload={payload!r}"
            )
        finally:
            net.destroy()

    def test_failed_cook_surfaces_a_real_error(self):
        net = _make_copnet_parent("_mcp_run_inference_smoke_saderror")
        try:
            onnx_node = _build_uncookable_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_run_inference",
                {"node_path": onnx_node.path(), "frame": None},
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            errors = payload["errors"]
            assert errors, (
                "a failed cook must surface a non-empty errors list "
                "(FR-5 no-silent-success)"
            )
            assert any("src is missing" in e for e in errors), (
                f"expected the grounded 'src is missing' error to be surfaced "
                f"verbatim, got errors={errors!r}"
            )
        finally:
            net.destroy()

    def test_failed_cook_output_planes_empty_or_degrades_gracefully(self):
        """On a failed cook, output_planes is typically []. This test only
        asserts the key is present and does not blow up the call."""
        net = _make_copnet_parent("_mcp_run_inference_smoke_sadplanes")
        try:
            onnx_node = _build_uncookable_onnx_node(net)
            result = self._dispatch(
                "cop_onnx_run_inference",
                {"node_path": onnx_node.path(), "frame": None},
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            assert "output_planes" in payload
            assert isinstance(payload["output_planes"], list)
        finally:
            net.destroy()


# ===========================================================================
# Test class: target validation — bad node_path / non-onnx target
# ===========================================================================

class TestRunInferenceTargetValidation:
    """A bad node_path (unresolved) or a non-onnx / non-Cop-category node
    is ok:False (bad TARGET, distinct from a failed COOK which is
    ok:True+cooked:False)."""

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def test_nonexistent_node_path_is_ok_false(self):
        result = self._dispatch(
            "cop_onnx_run_inference",
            {"node_path": "/obj/definitely_does_not_exist_probe", "frame": None},
        )
        payload = _unwrap(result)
        assert payload.get("ok") is False, (
            f"an unresolved node_path must be ok:False (a bad TARGET), got {payload!r}"
        )

    def test_non_onnx_node_is_ok_false(self):
        """A real node that is NOT type 'onnx' (e.g. the copnet itself)
        must be ok:False — folds codex Blocker-1 (name+category check)."""
        net = _make_copnet_parent("_mcp_run_inference_smoke_nononnx")
        try:
            result = self._dispatch(
                "cop_onnx_run_inference",
                {"node_path": net.path(), "frame": None},
            )
            payload = _unwrap(result)
            assert payload.get("ok") is False, (
                f"a non-onnx node target must be ok:False, got {payload!r}"
            )
        finally:
            net.destroy()


# ===========================================================================
# Test registry and runner (for direct hython execution)
# ===========================================================================

_TEST_CLASSES = [
    TestCopOnnxRunInferenceDispatcherImport,
    TestCopOnnxRunInferenceCapabilityContract,
    TestCopOnnxRunInferencePreviewDeniesInvalidTarget,
    TestRunInferenceHappyPath,
    TestRunInferenceSadPath,
    TestRunInferenceTargetValidation,
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
    print("cop_onnx_run_inference_hython_smoke.py — pp12-113d")
    print("=" * 72)

    # Pre-check: is this the RED phase or GREEN phase?
    try:
        _get_dispatcher_surface()
        print("[mode] GREEN — cop_onnx_run_inference registered; running all tests.")
    except ImportError as exc:
        print(f"[mode] RED — {exc}")
        print("Expected RED failure. Wrapper pytest red gate has already been confirmed.")
        print("Hython-smoke RED gate confirmed: cop_onnx_run_inference not yet implemented.")
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
