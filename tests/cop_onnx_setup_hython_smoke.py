"""Hython-smoke tests for the cop_onnx_setup_node / cop_onnx_set_provider
GATED dispatcher path (PP12-113 PR-3).

Unit: pp12-113c
testVerificationSurface: hython-smoke
planSha: 93096c3c0443cab7842583f6ba15abdbd56e3da289b87f6f11d02c2e9da0fa8b

Run under hython (Houdini's headless Python interpreter):
    hython tests/cop_onnx_setup_hython_smoke.py

Two-mode guard:
  - If fxhoudinimcp_server.handlers.cop_onnx_handlers does not yet expose
    cop_onnx_setup_node / cop_onnx_set_provider (hou-dev has not
    implemented them yet), the import raises ImportError with the
    message "expected RED (hou-dev has not implemented
    cop_onnx_setup_node/cop_onnx_set_provider yet)". This is the RED
    gate for the hython surface.
  - Once hou-dev implements the handlers, the tests run and PASS GREEN.

Dispatcher-based (NOT direct handler calls) — mirrors
cop_onnx_hython_smoke.py (pp12-113b) and
usd_export_inspect_hython_smoke.py (pp12-112b/d):
  - Importing cop_onnx_handlers triggers register_handler(
    'cop_onnx_setup_node', ...) and register_handler(
    'cop_onnx_set_provider', ...) as side-effects (in addition to the
    PR-2 'cop_onnx_list_models' / 'cop_onnx_inspect_model' registrations,
    which must remain present -- append-only).
  - Tests then call dispatch('cop_onnx_setup_node', {...}) /
    dispatch('cop_onnx_set_provider', {...}) -- the REAL dispatcher path
    that exercises the full handler(**params) calling convention.

GATED contract asserted here (per plan pp12-113c lockedFieldContract,
PLAN-REVIEW FOLD rev2):
  - capability_of('cop_onnx_setup_node') == Capability.MUTATING
  - capability_of('cop_onnx_set_provider') == Capability.MUTATING
  - preview_of(...) for both carries a real preview_fn with
    preview_required=True
  - _preview_setup_node (params) RAISES (-> gate DENY) on a bogus/
    nonexistent parent_path -- an INVALID target is denied at the gate,
    not merely flagged for operator-reject (FOLD M1-preview-DENY).
  - setup_node against a REAL fixture .onnx creates a PERSISTENT
    cop/onnx node under the given parent (still exists after the call
    returns -- NOT a scratch node, contrast PR-2's inspect_model).
  - The returned tensor mapping matches the GROUNDED probe result: 1:1
    node_inputs==model_inputs / node_outputs==model_outputs after
    setupshapes; input_tensors[i].name == model_input_name{i};
    cop_input_index == i (1-based); output_tensors[i].cop_plane ==
    node_output_name{i} (the 'n_'-prefixed COP plane token, e.g.
    'n_output').
  - set_provider against the real node sets the provider and reports
    available_providers (from node.parm('provider').menuItems() at
    RUNTIME) + will_bind.

Fixtures (SAME committed .onnx binaries as PR-2, tests/fixtures/):
  identity_dynamic.onnx — 1 input "input" [1,3,'H','W'] float32 -> 1 output
                           "output" same shape (Identity op). H/W are
                           symbolic (dynamic) dims.
  multi_input.onnx      — 2 inputs "input_a"/"input_b" [1,3,64,64] float32
                           -> 1 output "output" (Add op). Confirms
                           multi-input instance-count semantics carry
                           through setup_node's tensor-mapping read.
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — same pattern as cop_onnx_hython_smoke.py (pp12-113b)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_PKG_PYTHON = os.path.join(_REPO_ROOT, "python")
_HOUDINI_HANDLERS = os.path.join(_REPO_ROOT, "houdini", "scripts", "python")
_FIXTURES_DIR = os.path.join(_HERE, "fixtures")

for _p in (_REPO_ROOT, _PKG_PYTHON, _HOUDINI_HANDLERS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_IDENTITY_FIXTURE = os.path.join(_FIXTURES_DIR, "identity_dynamic.onnx").replace("\\", "/")
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

    If cop_onnx_setup_node / cop_onnx_set_provider are not yet registered
    (hou-dev has not implemented them), raises ImportError with a message
    indicating this is the expected RED state.
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

    # RED GATE: the two new commands must be REGISTERED (capability_of
    # returns non-None) for this to be considered GREEN. PR-2's two
    # commands existing is not sufficient.
    if capability_of("cop_onnx_setup_node") is None or capability_of("cop_onnx_set_provider") is None:
        raise ImportError(
            "expected RED (hou-dev has not implemented "
            "cop_onnx_setup_node/cop_onnx_set_provider yet): "
            "cop_onnx_handlers imported but the two new commands are not registered"
        )

    # TEST-HARNESS FIX (hou-dev, mechanical, zero assertion/contract change):
    # wrap the returned plain module-level functions in staticmethod() so
    # cls._dispatch = dispatch (etc.) does NOT get bound as an instance
    # method via the descriptor protocol when accessed as self._dispatch(...)
    # -- an unbound self would otherwise be injected as an extra positional
    # arg (TypeError: dispatch() takes 2 positional arguments but 3 were
    # given). Mirrors the proven, working pattern in the sibling PR-2 file
    # cop_onnx_hython_smoke.py (`cls._dispatch = staticmethod(_get_dispatcher())`).
    return staticmethod(dispatch), staticmethod(capability_of), staticmethod(preview_of)


# ---------------------------------------------------------------------------
# Tiny result helpers (mirroring cop_onnx_hython_smoke.py)
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


def _is_failure(result: dict) -> bool:
    payload = _unwrap(result)
    return isinstance(payload, dict) and payload.get("ok") is False


def _get_error(result: dict) -> str:
    payload = _unwrap(result)
    return str(payload.get("error", ""))


def _make_copnet_parent(name: str) -> str:
    """Create a real copnet under /obj to use as setup_node's parent_path.
    Caller is responsible for destroying it after the test."""
    import hou
    obj = hou.node("/obj")
    net = obj.createNode("copnet", name)
    return net.path()


# ===========================================================================
# Test class: dispatcher import and handler registration
# ===========================================================================

class TestCopOnnxSetupDispatcherImport:
    """RED GATE: cop_onnx_setup_node / cop_onnx_set_provider must be
    registered on the dispatcher.

    On RED (before hou-dev implements):
        ImportError: "expected RED (hou-dev has not implemented
        cop_onnx_setup_node/cop_onnx_set_provider yet)"

    On GREEN (after hou-dev implements):
        The module imports cleanly and both new commands are registered.
    """

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def test_dispatch_callable(self):
        assert callable(self._dispatch)

    def test_pr2_commands_still_registered(self):
        """Append-only contract: PR-2's cop_onnx_list_models /
        cop_onnx_inspect_model must still be registered — hou-dev must
        NOT have clobbered them."""
        assert self._capability_of("cop_onnx_list_models") is not None, (
            "cop_onnx_list_models (PR-2) must remain registered (append-only contract)"
        )
        assert self._capability_of("cop_onnx_inspect_model") is not None, (
            "cop_onnx_inspect_model (PR-2) must remain registered (append-only contract)"
        )


# ===========================================================================
# Test class: capability + preview contract (GATED — Capability.MUTATING)
# ===========================================================================

class TestCopOnnxSetupCapabilityContract:
    """Both cop_onnx_setup_node and cop_onnx_set_provider must be
    registered as Capability.MUTATING WITH a preview_fn and
    preview_required=True — the SHIPPED 112 GATED pattern."""

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def test_setup_node_is_mutating(self):
        from fxhoudinimcp_server.dispatcher import Capability
        cap = self._capability_of("cop_onnx_setup_node")
        assert cap == Capability.MUTATING, (
            f"cop_onnx_setup_node must be Capability.MUTATING (GATED), got {cap!r}"
        )

    def test_set_provider_is_mutating(self):
        from fxhoudinimcp_server.dispatcher import Capability
        cap = self._capability_of("cop_onnx_set_provider")
        assert cap == Capability.MUTATING, (
            f"cop_onnx_set_provider must be Capability.MUTATING (GATED), got {cap!r}"
        )

    def test_setup_node_has_preview_required(self):
        preview = self._preview_of("cop_onnx_setup_node")
        assert preview.get("preview_fn") is not None, (
            "cop_onnx_setup_node must register a preview_fn (109 gate GATED pattern)"
        )
        assert preview.get("preview_required") is True, (
            "cop_onnx_setup_node must have preview_required=True"
        )

    def test_set_provider_has_preview_required(self):
        preview = self._preview_of("cop_onnx_set_provider")
        assert preview.get("preview_fn") is not None, (
            "cop_onnx_set_provider must register a preview_fn (109 gate GATED pattern)"
        )
        assert preview.get("preview_required") is True, (
            "cop_onnx_set_provider must have preview_required=True"
        )


# ===========================================================================
# Test class: preview RAISES on an INVALID target (FOLD M1-preview-DENY)
# ===========================================================================

class TestCopOnnxSetupPreviewDeniesInvalidTarget:
    """FOLD M1-preview-DENY: _preview_setup_node MUST RAISE (-> gate DENY)
    when parent_path does NOT resolve OR is NOT a COP network;
    _preview_set_provider MUST RAISE when node_path does NOT resolve OR
    is NOT type 'onnx'. A raise here is the correct DENY behavior -- NOT
    a returned flag for the operator to reject."""

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def test_preview_setup_node_raises_on_nonexistent_parent(self):
        preview = self._preview_of("cop_onnx_setup_node")
        preview_fn = preview["preview_fn"]
        params = {
            "parent_path": "/obj/definitely_does_not_exist_probe",
            "model_path": _IDENTITY_FIXTURE,
            "node_name": "agent_onnx",
            "setup_shapes": True,
            "flip_input": None,
            "flip_output": None,
        }
        raised = False
        try:
            preview_fn(params)
        except Exception:
            raised = True
        assert raised, (
            "_preview_setup_node must RAISE (-> gate DENY) when parent_path "
            "does not resolve to a real node (FOLD M1-preview-DENY)"
        )

    def test_preview_setup_node_raises_on_non_cop_parent(self):
        """A real but NON-COP parent (e.g. /obj itself, an OBJ context
        node) must also raise -- 'exists' is not sufficient, it must be a
        COP network."""
        preview = self._preview_of("cop_onnx_setup_node")
        preview_fn = preview["preview_fn"]
        params = {
            "parent_path": "/obj",
            "model_path": _IDENTITY_FIXTURE,
            "node_name": "agent_onnx",
            "setup_shapes": True,
            "flip_input": None,
            "flip_output": None,
        }
        raised = False
        try:
            preview_fn(params)
        except Exception:
            raised = True
        assert raised, (
            "_preview_setup_node must RAISE when parent_path resolves but is "
            "NOT a COP network (e.g. /obj) (FOLD M1-preview-DENY)"
        )

    def test_preview_set_provider_raises_on_nonexistent_node(self):
        preview = self._preview_of("cop_onnx_set_provider")
        preview_fn = preview["preview_fn"]
        params = {"node_path": "/obj/definitely_does_not_exist_probe", "provider": "cuda"}
        raised = False
        try:
            preview_fn(params)
        except Exception:
            raised = True
        assert raised, (
            "_preview_set_provider must RAISE (-> gate DENY) when node_path "
            "does not resolve to a real node (FOLD M1-preview-DENY)"
        )

    def test_preview_set_provider_raises_on_non_onnx_node(self):
        """A real but non-'onnx'-type node_path must also raise."""
        import hou
        preview = self._preview_of("cop_onnx_set_provider")
        preview_fn = preview["preview_fn"]
        net = hou.node("/obj").createNode("copnet", "_mcp_onnx_setup_smoke_wrongtype")
        try:
            params = {"node_path": net.path(), "provider": "cuda"}
            raised = False
            try:
                preview_fn(params)
            except Exception:
                raised = True
            assert raised, (
                "_preview_set_provider must RAISE when node_path resolves but "
                "is NOT type 'onnx' (FOLD M1-preview-DENY)"
            )
        finally:
            net.destroy()

    def test_preview_setup_node_does_not_mutate_scene(self):
        """Preview is READ-ONLY -- calling it (even on a valid target) must
        not create a node."""
        import hou
        parent_path = _make_copnet_parent("_mcp_onnx_setup_smoke_preview_ro")
        try:
            before = len(hou.node(parent_path).children())
            preview = self._preview_of("cop_onnx_setup_node")
            preview_fn = preview["preview_fn"]
            params = {
                "parent_path": parent_path,
                "model_path": _IDENTITY_FIXTURE,
                "node_name": "agent_onnx",
                "setup_shapes": True,
                "flip_input": None,
                "flip_output": None,
            }
            preview_fn(params)
            after = len(hou.node(parent_path).children())
            assert after == before, (
                f"_preview_setup_node must NOT mutate the scene: before={before} after={after}"
            )
        finally:
            hou.node(parent_path).destroy()


# ===========================================================================
# Test class: cop_onnx_setup_node against a real single-input fixture
# ===========================================================================

class TestSetupNodeSingleInput:
    """cop_onnx_setup_node against identity_dynamic.onnx — a real
    single-input, dynamic-dim model. Confirms the node PERSISTS (is not
    destroyed) and the tensor mapping matches the grounded probe result."""

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def test_fixture_exists(self):
        assert os.path.isfile(_IDENTITY_FIXTURE), (
            f"fixture missing: {_IDENTITY_FIXTURE!r} — Phase-0 make_fixture.py "
            "must be run once to generate it (committed as a binary fixture)."
        )

    def test_returns_ok_true_and_node_persists(self):
        """The created node must STILL EXIST after the call returns — this
        is a PERSISTENT node (contrast PR-2's inspect_model scratch-node,
        which is always destroyed)."""
        import hou
        parent_path = _make_copnet_parent("_mcp_onnx_setup_smoke_persist")
        try:
            result = self._dispatch(
                "cop_onnx_setup_node",
                {
                    "parent_path": parent_path,
                    "model_path": _IDENTITY_FIXTURE,
                    "node_name": "agent_onnx",
                    "setup_shapes": True,
                    "flip_input": None,
                    "flip_output": None,
                },
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            node_path = payload["node_path"]
            node = hou.node(node_path)
            assert node is not None, (
                f"the created node at {node_path!r} must PERSIST (still exist) "
                "after cop_onnx_setup_node returns — it is NOT a scratch node"
            )
            assert node.type().name() == "onnx"
        finally:
            hou.node(parent_path).destroy()

    def test_modelfile_set_and_setupshapes_applied(self):
        import hou
        parent_path = _make_copnet_parent("_mcp_onnx_setup_smoke_modelfile")
        try:
            result = self._dispatch(
                "cop_onnx_setup_node",
                {
                    "parent_path": parent_path,
                    "model_path": _IDENTITY_FIXTURE,
                    "node_name": "agent_onnx",
                    "setup_shapes": True,
                    "flip_input": None,
                    "flip_output": None,
                },
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            node = hou.node(payload["node_path"])
            assert node.parm("modelfile").eval() == _IDENTITY_FIXTURE
            # setupshapes having been applied is proven by model_inputs/
            # model_outputs being populated (>0) -- a fresh unconfigured
            # node's model_inputs defaults to 1 with an EMPTY name, but a
            # configured node has a REAL tensor name.
            assert node.parm("model_input_name1").eval() == "input", (
                "setupshapes must have populated the real tensor name "
                "(proves setupshapes was pressed, not just modelfile set)"
            )
        finally:
            hou.node(parent_path).destroy()

    def test_tensor_mapping_matches_grounded_probe(self):
        """LOCKED (plan-review FOLD, grounded via the orchestrator's live
        hython probe): 1:1 node_inputs==model_inputs / node_outputs==
        model_outputs after setupshapes. input_tensors[i].name ==
        model_input_name{i}; cop_input_index == i (1-based, derived from
        the LIVE model_inputs count -- not a literal). output_tensors[i].
        cop_plane == the LIVE node_output_name{i} parm read directly off
        the node setup_node created (the 'n_'-prefixed COP plane token)
        -- NOT a hardcoded/fallback string. A handler that fabricates
        cop_plane instead of reading the real parm must FAIL this test."""
        import hou
        parent_path = _make_copnet_parent("_mcp_onnx_setup_smoke_mapping")
        try:
            result = self._dispatch(
                "cop_onnx_setup_node",
                {
                    "parent_path": parent_path,
                    "model_path": _IDENTITY_FIXTURE,
                    "node_name": "agent_onnx",
                    "setup_shapes": True,
                    "flip_input": None,
                    "flip_output": None,
                },
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload

            input_tensors = payload["input_tensors"]
            output_tensors = payload["output_tensors"]
            assert len(input_tensors) == 1, input_tensors
            assert len(output_tensors) == 1, output_tensors

            node = hou.node(payload["node_path"])
            assert node is not None, (
                f"node_path {payload['node_path']!r} must resolve to a real node "
                "so we can read the LIVE parms it actually configured"
            )

            # cop_input_index must be derived from the LIVE model_inputs
            # count (1-based), not a literal -- read the actual parm.
            live_model_inputs = node.parm("model_inputs").eval()
            assert live_model_inputs == 1, (
                f"expected model_inputs==1 on the single-input fixture, got {live_model_inputs!r}"
            )
            assert input_tensors[0]["name"] == "input", input_tensors[0]
            assert input_tensors[0]["cop_input_index"] == 1, (
                f"cop_input_index must be 1-based derived from the live "
                f"model_inputs count ({live_model_inputs}), got "
                f"{input_tensors[0]['cop_input_index']!r}"
            )

            # cop_plane MUST equal the LIVE node_output_name1 parm the node
            # actually holds after setupshapes -- read it directly off the
            # node, do NOT compare against a hardcoded literal. This is
            # what catches a handler that fabricates/falls back the value
            # instead of reading the real COP-plane parm.
            live_node_output_name1 = node.parm("node_output_name1").eval()
            assert live_node_output_name1, (
                "node_output_name1 must be populated on a configured onnx "
                "node (proves setupshapes wired the COP-plane parm)"
            )
            assert output_tensors[0]["name"] == "output", output_tensors[0]
            assert output_tensors[0]["cop_plane"] == live_node_output_name1, (
                f"cop_plane must equal the LIVE node_output_name1 parm "
                f"({live_node_output_name1!r}) read directly off the node "
                f"-- got {output_tensors[0]['cop_plane']!r}. A hardcoded or "
                f"fallback cop_plane must fail this assertion."
            )
        finally:
            hou.node(parent_path).destroy()

    def test_multi_input_tensor_mapping(self):
        """Confirms multi-input instance-count semantics carry through
        setup_node's tensor-mapping read (2 inputs, 1:1, in order,
        1-based cop_input_index derived from the LIVE model_inputs
        count -- not a literal)."""
        import hou
        parent_path = _make_copnet_parent("_mcp_onnx_setup_smoke_multi")
        try:
            result = self._dispatch(
                "cop_onnx_setup_node",
                {
                    "parent_path": parent_path,
                    "model_path": _MULTI_FIXTURE,
                    "node_name": "agent_onnx_multi",
                    "setup_shapes": True,
                    "flip_input": None,
                    "flip_output": None,
                },
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload

            node = hou.node(payload["node_path"])
            assert node is not None
            live_model_inputs = node.parm("model_inputs").eval()
            assert live_model_inputs == 2, (
                f"expected model_inputs==2 on the multi-input fixture, got {live_model_inputs!r}"
            )
            expected_indices = list(range(1, live_model_inputs + 1))

            input_tensors = payload["input_tensors"]
            assert len(input_tensors) == 2, input_tensors
            assert [t["name"] for t in input_tensors] == ["input_a", "input_b"]
            assert [t["cop_input_index"] for t in input_tensors] == expected_indices, (
                f"cop_input_index must be 1-based, in order, and derived from "
                f"the live model_inputs count ({live_model_inputs}), got "
                f"{[t['cop_input_index'] for t in input_tensors]!r}"
            )
        finally:
            hou.node(parent_path).destroy()

    def test_flip_input_applied(self):
        """flip_input=True must set every input-instance flip parm
        input_flip{i} to 1."""
        parent_path = _make_copnet_parent("_mcp_onnx_setup_smoke_flip")
        try:
            result = self._dispatch(
                "cop_onnx_setup_node",
                {
                    "parent_path": parent_path,
                    "model_path": _IDENTITY_FIXTURE,
                    "node_name": "agent_onnx",
                    "setup_shapes": True,
                    "flip_input": True,
                    "flip_output": None,
                },
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            import hou
            node = hou.node(payload["node_path"])
            flip_parm = node.parm("input_flip1")
            assert flip_parm is not None, "input_flip1 parm must exist on a configured onnx node"
            assert flip_parm.eval() == 1, (
                f"flip_input=True must set input_flip1 to 1, got {flip_parm.eval()!r}"
            )
        finally:
            import hou
            hou.node(parent_path).destroy()

    def test_no_reload_pressed_no_crash(self):
        """houdini-001 (catalog): the handler must NEVER press 'reload'
        before/after 'setupshapes' -- verified indirectly: a successful
        call with no crash/segfault is evidence the reload button was
        never pressed on a freshly modelfile-set node."""
        parent_path = _make_copnet_parent("_mcp_onnx_setup_smoke_noreload")
        try:
            result = self._dispatch(
                "cop_onnx_setup_node",
                {
                    "parent_path": parent_path,
                    "model_path": _IDENTITY_FIXTURE,
                    "node_name": "agent_onnx",
                    "setup_shapes": True,
                    "flip_input": None,
                    "flip_output": None,
                },
            )
            assert _is_ok(result), _unwrap(result)
        finally:
            import hou
            hou.node(parent_path).destroy()


# ===========================================================================
# Test class: cop_onnx_set_provider
# ===========================================================================

class TestSetProvider:
    """cop_onnx_set_provider against a real configured onnx node."""

    @classmethod
    def setup_class(cls):
        cls._dispatch, cls._capability_of, cls._preview_of = _get_dispatcher_surface()

    def _make_configured_node(self, name: str):
        import hou
        parent_path = _make_copnet_parent(name)
        setup_result = self._dispatch(
            "cop_onnx_setup_node",
            {
                "parent_path": parent_path,
                "model_path": _IDENTITY_FIXTURE,
                "node_name": "agent_onnx",
                "setup_shapes": True,
                "flip_input": None,
                "flip_output": None,
            },
        )
        payload = _unwrap(setup_result)
        assert _is_ok(setup_result), payload
        return parent_path, payload["node_path"]

    def test_sets_available_provider(self):
        """Setting a provider that IS on the runtime menuItems() list must
        succeed and report will_bind == the requested provider."""
        parent_path, node_path = self._make_configured_node("_mcp_onnx_setup_smoke_provider1")
        try:
            import hou
            node = hou.node(node_path)
            available = list(node.parm("provider").menuItems())
            # Pick any non-'automatic' available provider if present, else
            # 'automatic' itself.
            candidates = [p for p in available if p != "automatic"] or available
            target = candidates[0]

            result = self._dispatch(
                "cop_onnx_set_provider", {"node_path": node_path, "provider": target}
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            assert payload["will_bind"] == target, (
                f"an available provider must bind exactly, got will_bind="
                f"{payload['will_bind']!r} for requested={target!r}"
            )
            assert payload["available_providers"] == available, (
                "available_providers must be the REAL runtime menuItems(), "
                "not a hardcoded list"
            )
        finally:
            import hou
            hou.node(parent_path).destroy()

    def test_unavailable_provider_falls_back_never_errors(self):
        """FR-4 + PLAN-REVIEW FOLD m2-provider-edge: an unavailable
        provider request ('tensorrt', which is NEVER on any platform's
        menuItems() list -- it is not a real Houdini onnx provider token
        on Windows/macOS/Linux) must fall back to EXACTLY 'automatic'
        with a warning -- NEVER error. Per the locked contract, when the
        requested provider is unavailable AND 'automatic' IS in
        available_providers (which it always is -- 'automatic' is the
        universal safe default on every platform), will_bind MUST equal
        'automatic' exactly. A handler that instead falls back to
        whatever provider happens to be first/available (e.g. 'cpu' or
        'cuda') must FAIL this test -- accepting 'any available
        provider' is the hollow assertion this test closes."""
        parent_path, node_path = self._make_configured_node("_mcp_onnx_setup_smoke_provider2")
        try:
            import hou
            node = hou.node(node_path)
            available = list(node.parm("provider").menuItems())
            assert "tensorrt" not in available, (
                "test precondition: 'tensorrt' must not be a real available "
                f"provider on this platform, got available={available!r}"
            )
            assert "automatic" in available, (
                "test precondition: 'automatic' must be present in "
                f"available_providers for the locked-contract fallback to "
                f"apply, got available={available!r}"
            )

            result = self._dispatch(
                "cop_onnx_set_provider", {"node_path": node_path, "provider": "tensorrt"}
            )
            payload = _unwrap(result)
            assert _is_ok(result), (
                f"an unavailable provider must NEVER error (FR-4), got {payload!r}"
            )
            assert payload["will_bind"] == "automatic", (
                f"LOCKED CONTRACT (FOLD m2-provider-edge): when the requested "
                f"provider is unavailable AND 'automatic' is in "
                f"available_providers, will_bind MUST equal 'automatic' "
                f"exactly -- got will_bind={payload['will_bind']!r}. Falling "
                f"back to any other available provider (cpu/cuda/directml) "
                f"is NOT the locked contract and must fail this test."
            )
            assert payload["warnings"], (
                "an unavailable-provider fallback must surface a non-empty warnings list"
            )
        finally:
            hou.node(parent_path).destroy()

    def test_available_providers_platform_filtered_no_coreml_on_windows(self):
        """The runtime menuItems() list must be platform-filtered -- on
        Windows, 'coreml' (a macOS-only provider) must NOT appear."""
        import platform
        parent_path, node_path = self._make_configured_node("_mcp_onnx_setup_smoke_provider3")
        try:
            result = self._dispatch(
                "cop_onnx_set_provider", {"node_path": node_path, "provider": "automatic"}
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            if platform.system() == "Windows":
                assert "coreml" not in payload["available_providers"], (
                    f"'coreml' must not appear in available_providers on Windows, "
                    f"got {payload['available_providers']!r}"
                )
        finally:
            import hou
            hou.node(parent_path).destroy()


# ===========================================================================
# Test registry and runner (for direct hython execution)
# ===========================================================================

_TEST_CLASSES = [
    TestCopOnnxSetupDispatcherImport,
    TestCopOnnxSetupCapabilityContract,
    TestCopOnnxSetupPreviewDeniesInvalidTarget,
    TestSetupNodeSingleInput,
    TestSetProvider,
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
    print("cop_onnx_setup_hython_smoke.py — pp12-113c")
    print("=" * 72)

    # Pre-check: is this the RED phase or GREEN phase?
    try:
        _get_dispatcher_surface()
        print("[mode] GREEN — cop_onnx_setup_node/cop_onnx_set_provider registered; running all tests.")
    except ImportError as exc:
        print(f"[mode] RED — {exc}")
        print("Expected RED failure. Wrapper pytest red gate has already been confirmed.")
        print("Hython-smoke RED gate confirmed: cop_onnx_setup_node/cop_onnx_set_provider not yet implemented.")
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
