"""Hython-smoke tests for the cop_onnx_list_models / cop_onnx_inspect_model
dispatcher path (PP12-113 PR-2).

Unit: pp12-113b
testVerificationSurface: hython-smoke
planSha: 22e035c1811c41e521c71c799a1c4e8bfad96e8e20799da3e153ae0009b42e7f

Run under hython (Houdini's headless Python interpreter):
    hython tests/cop_onnx_hython_smoke.py

Two-mode guard:
  - If fxhoudinimcp_server.handlers.cop_onnx_handlers is absent (hou-dev has
    not implemented yet), the import raises ImportError with the message
    "expected RED (hou-dev has not implemented cop_onnx_handlers yet)".
    This is the RED gate for the hython surface.
  - Once hou-dev implements the handlers, the tests run and PASS GREEN.

Dispatcher-based (NOT direct handler calls):
  - Importing cop_onnx_handlers triggers register_handler('cop_onnx_list_models', ...)
    and register_handler('cop_onnx_inspect_model', ...) as side-effects.
  - Tests then call dispatch('cop_onnx_inspect_model', {...}) — the real
    dispatcher path that exercises the full handler(**params) calling
    convention, mirroring usd_export_inspect_hython_smoke.py (pp12-112b).

Fixtures (built by tests/fixtures/*, see make_fixture.py in the Phase-0
scratchpad — committed as .onnx binaries under tests/fixtures/):
  identity_dynamic.onnx — 1 input "input" [1,3,'H','W'] float32 -> 1 output
                           "output" same shape (Identity op). H/W are
                           symbolic (dynamic) dims.
  multi_input.onnx      — 2 inputs "input_a"/"input_b" [1,3,64,64] float32
                           -> 1 output "output" (Add op). Confirms
                           multi-input instance-count semantics.

Confirmed real return values on these fixtures (Phase-0 hython probe,
2026-07-01, live Houdini 21.0.729):
  cop_onnx_inspect_model(identity_dynamic.onnx):
    ok=True, inputs==[{"name": "input", "shape": [1, 3, "dynamic", "dynamic"],
    "dtype": "float32", "layout_guess": "NCHW"}], outputs same shape,
    layout_guess "unknown"
  cop_onnx_inspect_model(multi_input.onnx):
    ok=True, len(inputs) == 2, names == ["input_a", "input_b"]
  Scratch node/net cleanup: children(/obj) count identical before and
    after every call (guaranteed finally-destroy).

REV2 FOLD (codex tier-2 BLOCK 2026-07-01 -- hython-smoke additions):
  - node_path (path A) is READ-ONLY: after an inspect via node_path, the
    caller's node's modelfile + model_inputs are UNCHANGED (no mutation).
  - /obj child-count before == after a scratch-node (path B) inspect --
    no orphan NODE OR NET -- on BOTH the success path AND a forced-
    failure path (a bad model_path that fails inside setupshapes/parm
    read but still hits the finally).
  - The scratch net /obj/_mcp_onnx_inspect must not exist AT ALL after
    any path-B call (it is created+destroyed together every call, never
    reused/persisted -- REV2 FOLD B2).

Acceptance tests covered:
  FR-2/FR-5: result shape on success / failure (empty model_path/node_path)
  Scratch-node cleanup: no orphan node/net left in /obj after inspect,
    success or forced-failure
  node_path read-only: no mutation of the caller's existing node
  cop_onnx_list_models: globs .onnx files under a given root, never
    raises on a missing root
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — same pattern as usd_export_inspect_hython_smoke.py
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
# Dual-mode import guard: RED = ImportError; GREEN = returns dispatch
# ---------------------------------------------------------------------------
def _get_dispatcher():
    """Import cop_onnx_handlers (triggers handler registration) and return dispatch.

    If fxhoudinimcp_server.handlers.cop_onnx_handlers is absent, raises
    ImportError with a message indicating this is the expected RED state.
    Once hou-dev implements the handlers, this returns the real dispatcher.
    """
    try:
        # Importing the handler module triggers register_handler side-effects.
        import fxhoudinimcp_server.handlers.cop_onnx_handlers as _handlers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"expected RED (hou-dev has not implemented cop_onnx_handlers yet): {exc}"
        ) from exc

    try:
        from fxhoudinimcp_server.dispatcher import dispatch
    except ImportError as exc:
        raise ImportError(
            f"fxhoudinimcp_server.dispatcher not found — is fxhoudinimcp_server on sys.path? {exc}"
        ) from exc

    return dispatch


# ---------------------------------------------------------------------------
# Tiny result helpers (mirroring usd_export_inspect_hython_smoke.py)
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


def _obj_child_count() -> int:
    import hou
    return len(hou.node("/obj").children())


def _scratch_net_exists() -> bool:
    """True if the scratch copnet /obj/_mcp_onnx_inspect currently exists.

    REV2 FOLD B2: the handler no longer reuses/persists a scratch copnet
    across calls -- it creates AND destroys both the scratch node and the
    scratch net together on every scratch-node-mechanism call (including
    a copnet-created-but-onnx-createNode-then-failed path). So after any
    such call this net must NOT exist at all -- not "exist but be empty".
    """
    import hou
    return hou.node("/obj/_mcp_onnx_inspect") is not None


def _get_node_path_for_fixture(model_path: str) -> str:
    """Build a real onnx node with modelfile set + setupshapes already
    pressed against the given fixture, OUTSIDE the handler under test, so
    node_path-branch (path A, read-only) tests have a node whose
    model_inputs/model_outputs are already populated. Returns the node's
    path. Caller is responsible for destroying the node (and its parent
    net, if it created one) after the test.
    """
    import hou
    obj = hou.node("/obj")
    net = obj.createNode("copnet", "_mcp_onnx_smoke_fixture_node")
    node = net.createNode("onnx")
    node.parm("modelfile").set(model_path)
    node.parm("setupshapes").pressButton()
    return node.path()


# ---------------------------------------------------------------------------
# Test class: dispatcher import and handler registration
# ---------------------------------------------------------------------------

class TestCopOnnxDispatcherImport:
    """RED GATE: fxhoudinimcp_server.handlers.cop_onnx_handlers must be importable.

    On RED (before hou-dev implements):
        ImportError: "expected RED (hou-dev has not implemented cop_onnx_handlers yet)"

    On GREEN (after hou-dev implements):
        The module imports cleanly and registers both handlers as side-effects.
    """

    @classmethod
    def setup_class(cls):
        cls._dispatch = staticmethod(_get_dispatcher())  # raises ImportError on RED

    def test_cop_onnx_handlers_importable(self):
        assert callable(self._dispatch), (
            "dispatch must be a callable after handlers are imported."
        )

    def test_cop_onnx_list_models_registered(self):
        try:
            self._dispatch("cop_onnx_list_models", {"roots": ["/tmp/does_not_exist_probe"]})
        except KeyError as exc:
            if "cop_onnx_list_models" in str(exc) or "unknown command" in str(exc).lower():
                raise AssertionError(
                    "'cop_onnx_list_models' not found in dispatcher registry."
                ) from exc
        except Exception:
            pass  # Handler ran — registration confirmed.

    def test_cop_onnx_inspect_model_registered(self):
        try:
            self._dispatch("cop_onnx_inspect_model", {"model_path": "", "node_path": None})
        except KeyError as exc:
            if "cop_onnx_inspect_model" in str(exc) or "unknown command" in str(exc).lower():
                raise AssertionError(
                    "'cop_onnx_inspect_model' not found in dispatcher registry."
                ) from exc
        except Exception:
            pass  # Handler ran — registration confirmed.


# ---------------------------------------------------------------------------
# Test class: cop_onnx_inspect_model against a real single-input fixture
# ---------------------------------------------------------------------------

class TestInspectModelSingleInput:
    """cop_onnx_inspect_model against identity_dynamic.onnx — a real
    single-input, dynamic-dim model."""

    @classmethod
    def setup_class(cls):
        cls._dispatch = staticmethod(_get_dispatcher())
        cls._before_count = _obj_child_count()

    def test_fixture_exists(self):
        assert os.path.isfile(_IDENTITY_FIXTURE), (
            f"fixture missing: {_IDENTITY_FIXTURE!r} — Phase-0 make_fixture.py "
            "must be run once to generate it (committed as a binary fixture)."
        )

    def test_returns_ok_true(self):
        result = self._dispatch(
            "cop_onnx_inspect_model", {"model_path": _IDENTITY_FIXTURE, "node_path": None}
        )
        assert _is_ok(result), f"expected ok=True, got {_unwrap(result)!r}"

    def test_model_path_forwarded(self):
        result = self._dispatch(
            "cop_onnx_inspect_model", {"model_path": _IDENTITY_FIXTURE, "node_path": None}
        )
        payload = _unwrap(result)
        assert payload["model_path"] == _IDENTITY_FIXTURE

    def test_one_input_one_output(self):
        result = self._dispatch(
            "cop_onnx_inspect_model", {"model_path": _IDENTITY_FIXTURE, "node_path": None}
        )
        payload = _unwrap(result)
        assert len(payload["inputs"]) == 1, payload["inputs"]
        assert len(payload["outputs"]) == 1, payload["outputs"]

    def test_input_name_and_shape(self):
        result = self._dispatch(
            "cop_onnx_inspect_model", {"model_path": _IDENTITY_FIXTURE, "node_path": None}
        )
        payload = _unwrap(result)
        input_spec = payload["inputs"][0]
        assert input_spec["name"] == "input", input_spec
        assert input_spec["shape"] == [1, 3, "dynamic", "dynamic"], (
            f"dynamic dims must normalize to the literal 'dynamic' sentinel "
            f"(observed cop/onnx raw value -1), got {input_spec['shape']!r}"
        )

    def test_input_layout_guess_nchw(self):
        result = self._dispatch(
            "cop_onnx_inspect_model", {"model_path": _IDENTITY_FIXTURE, "node_path": None}
        )
        payload = _unwrap(result)
        assert payload["inputs"][0]["layout_guess"] == "NCHW", payload["inputs"][0]

    def test_output_layout_guess_unknown(self):
        result = self._dispatch(
            "cop_onnx_inspect_model", {"model_path": _IDENTITY_FIXTURE, "node_path": None}
        )
        payload = _unwrap(result)
        assert payload["outputs"][0]["layout_guess"] == "unknown", payload["outputs"][0]

    def test_node_path_reads_correct_contract(self):
        """REV2 FOLD B1: node_path (path A) against a REAL, already-
        configured node returns the same contract shape as the
        scratch-node mechanism reading the same fixture.
        """
        import hou
        node_path = _get_node_path_for_fixture(_IDENTITY_FIXTURE)
        net_path = hou.node(node_path).parent().path()
        try:
            result = self._dispatch(
                "cop_onnx_inspect_model", {"model_path": None, "node_path": node_path}
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            assert len(payload["inputs"]) == 1, payload["inputs"]
            assert payload["inputs"][0]["name"] == "input", payload["inputs"][0]
        finally:
            hou.node(net_path).destroy()

    def test_node_path_does_not_mutate_caller_node(self):
        """REV2 FOLD B1 -- the load-bearing no-mutation proof: after a
        node_path inspect, the caller's node's modelfile + model_inputs
        parms are UNCHANGED from before the call (no set(), no
        pressButton() on an existing node_path node).
        """
        import hou
        node_path = _get_node_path_for_fixture(_IDENTITY_FIXTURE)
        node = hou.node(node_path)
        net_path = node.parent().path()
        try:
            modelfile_before = node.parm("modelfile").eval()
            n_inputs_before = node.parm("model_inputs").eval()
            n_outputs_before = node.parm("model_outputs").eval()

            result = self._dispatch(
                "cop_onnx_inspect_model", {"model_path": None, "node_path": node_path}
            )
            assert _is_ok(result), _unwrap(result)

            assert node.parm("modelfile").eval() == modelfile_before, (
                "node_path branch mutated the caller node's modelfile parm — "
                "must be strictly READ-ONLY (REV2 FOLD B1)"
            )
            assert node.parm("model_inputs").eval() == n_inputs_before, (
                "node_path branch changed model_inputs count on the caller's "
                "node — must not re-run setupshapes on an existing node_path node"
            )
            assert node.parm("model_outputs").eval() == n_outputs_before, (
                "node_path branch changed model_outputs count on the caller's node"
            )
        finally:
            hou.node(net_path).destroy()

    def test_node_path_wins_over_model_path_when_both_given(self):
        """REV2 FOLD B1: when both model_path and node_path are given,
        node_path wins and model_path is ignored -- the multi-input
        fixture path must NOT be read even though it is passed.
        """
        import hou
        node_path = _get_node_path_for_fixture(_IDENTITY_FIXTURE)
        net_path = hou.node(node_path).parent().path()
        try:
            result = self._dispatch(
                "cop_onnx_inspect_model",
                {"model_path": _MULTI_FIXTURE, "node_path": node_path},
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            # The node_path node was set up against the SINGLE-input
            # identity fixture -- if node_path correctly won, we get 1
            # input (identity's), not 2 (multi_input's).
            assert len(payload["inputs"]) == 1, (
                f"node_path must win over model_path when both are given; "
                f"got {len(payload['inputs'])} inputs (looks like model_path "
                f"({_MULTI_FIXTURE}) was used instead): {payload}"
            )
        finally:
            hou.node(net_path).destroy()

    def test_node_path_reports_node_own_model_path_not_bogus_arg(self):
        """REV3 FOLD B1-metadata: when a DIFFERENT/bogus model_path is
        passed alongside node_path, the returned contract's model_path
        field must report the NODE's OWN configured modelfile (the
        identity fixture this node was set up against), NEVER the bogus
        caller-supplied model_path (the multi_input fixture). This closes
        the codex round-2 residual: node_path mode ignores model_path for
        READING (REV2 FOLD B1) but was still ECHOING it into the returned
        contract's model_path field -- now fixed to echo the node's own
        modelfile parm value instead.
        """
        import hou
        node_path = _get_node_path_for_fixture(_IDENTITY_FIXTURE)
        node = hou.node(node_path)
        net_path = node.parent().path()
        try:
            node_own_modelfile = node.parm("modelfile").eval()
            assert node_own_modelfile == _IDENTITY_FIXTURE, (
                "test fixture setup assumption broken: node's own modelfile "
                f"parm ({node_own_modelfile!r}) does not match the identity "
                f"fixture path ({_IDENTITY_FIXTURE!r}) it was set up against"
            )

            bogus_model_path = _MULTI_FIXTURE
            assert bogus_model_path != node_own_modelfile, (
                "test fixture setup assumption broken: the bogus model_path "
                "must differ from the node's own modelfile for this "
                "assertion to be meaningful"
            )

            result = self._dispatch(
                "cop_onnx_inspect_model",
                {"model_path": bogus_model_path, "node_path": node_path},
            )
            payload = _unwrap(result)
            assert _is_ok(result), payload
            assert payload["model_path"] == node_own_modelfile, (
                f"node_path mode must report the NODE's OWN modelfile "
                f"({node_own_modelfile!r}) as the contract's model_path, "
                f"NOT the bogus caller-supplied model_path "
                f"({bogus_model_path!r}); got {payload['model_path']!r} "
                f"(REV3 FOLD B1-metadata)"
            )
        finally:
            hou.node(net_path).destroy()

    def test_no_orphan_scratch_node(self):
        """REV2 FOLD B2: neither the scratch onnx NODE nor its parent
        scratch NET must survive past a single inspect call. The handler
        creates and destroys both together on every scratch-node-
        mechanism (path B) call -- no persistent-net-reuse. This is the
        guaranteed-cleanup proof for the READONLY guarantee.
        """
        before = _obj_child_count()
        self._dispatch(
            "cop_onnx_inspect_model", {"model_path": _IDENTITY_FIXTURE, "node_path": None}
        )
        after = _obj_child_count()
        assert after == before, (
            f"/obj child count changed after a scratch-node inspect: "
            f"before={before} after={after} (expected identical — "
            f"guaranteed finally-destroy of BOTH node and net)"
        )
        assert not _scratch_net_exists(), (
            "the scratch net /obj/_mcp_onnx_inspect must not exist after a "
            "path-B call (REV2 FOLD B2 — no persistent-net-reuse)"
        )


# ---------------------------------------------------------------------------
# Test class: cop_onnx_inspect_model multi-input instance-count semantics
# ---------------------------------------------------------------------------

class TestInspectModelMultiInput:
    """cop_onnx_inspect_model against multi_input.onnx — confirms
    multi-input instance-count semantics (2 inputs, in order)."""

    @classmethod
    def setup_class(cls):
        cls._dispatch = staticmethod(_get_dispatcher())

    def test_fixture_exists(self):
        assert os.path.isfile(_MULTI_FIXTURE), f"fixture missing: {_MULTI_FIXTURE!r}"

    def test_two_inputs_in_order(self):
        result = self._dispatch(
            "cop_onnx_inspect_model", {"model_path": _MULTI_FIXTURE, "node_path": None}
        )
        payload = _unwrap(result)
        assert _is_ok(result), payload
        assert len(payload["inputs"]) == 2, payload["inputs"]
        assert [t["name"] for t in payload["inputs"]] == ["input_a", "input_b"]

    def test_cleanup_after_multi_input(self):
        before = _obj_child_count()
        self._dispatch(
            "cop_onnx_inspect_model", {"model_path": _MULTI_FIXTURE, "node_path": None}
        )
        after = _obj_child_count()
        assert after == before, (
            f"/obj child count changed after a multi-input scratch inspect: "
            f"before={before} after={after}"
        )
        assert not _scratch_net_exists(), (
            "orphan scratch net left after a multi-input inspect (REV2 FOLD B2)"
        )


# ---------------------------------------------------------------------------
# Test class: FR-2 empty model_path guard + cleanup-on-exception
# ---------------------------------------------------------------------------

class TestInspectModelFailureShape:
    """FR-2: neither model_path nor node_path -> {ok: False, error}. Also
    confirms a nonexistent/forced-failure model_path does not leak a
    scratch node or net (REV2 FOLD: the guaranteed finally-destroy must
    hold on the FORCED-FAILURE path, not just the success path)."""

    @classmethod
    def setup_class(cls):
        cls._dispatch = staticmethod(_get_dispatcher())

    def test_empty_model_path_and_node_path_rejected(self):
        result = self._dispatch(
            "cop_onnx_inspect_model", {"model_path": "", "node_path": None}
        )
        assert _is_failure(result), _unwrap(result)
        assert _get_error(result), "error message must be non-empty"

    def test_nonexistent_model_path_does_not_leak(self):
        """Forced-failure path: a model_path that does not exist on disk.
        Whether Houdini treats this as ok=False or raises inside
        setupshapes/parm-read (caught by FR-5), the guaranteed finally
        must still destroy BOTH the scratch node and the scratch net —
        no orphan either way, and /obj child count must be unchanged.
        """
        before = _obj_child_count()
        result = self._dispatch(
            "cop_onnx_inspect_model",
            {"model_path": "/tmp/does_not_exist_probe.onnx", "node_path": None},
        )
        after = _obj_child_count()
        assert after == before, (
            f"/obj child count changed after a FORCED-FAILURE scratch inspect: "
            f"before={before} after={after}. result={_unwrap(result)!r}"
        )
        assert not _scratch_net_exists(), (
            f"orphan scratch net leaked on a FORCED-FAILURE model_path "
            f"(REV2 FOLD B2 finally must destroy on every path). "
            f"result={_unwrap(result)!r}"
        )

    def test_node_path_with_unconfigured_node_rejected_not_mutated(self):
        """REV2 FOLD B1 (live-probe corrected): node_path given but the
        node has no model configured (empty modelfile -- NOT a
        model_inputs/model_outputs count check: a fresh onnx node's
        multiparms default to 1 each with empty-named placeholder
        entries, confirmed live) must return {ok: False, error} WITHOUT
        mutating the node (no modelfile set, no setupshapes pressed) --
        never silently populate it.
        """
        import hou
        net = hou.node("/obj").createNode("copnet", "_mcp_onnx_smoke_unconfigured")
        node = net.createNode("onnx")
        try:
            modelfile_before = node.parm("modelfile").eval()
            n_inputs_before = node.parm("model_inputs").eval()
            assert modelfile_before == "", (
                f"test precondition: a fresh onnx node's modelfile must be "
                f"empty, got {modelfile_before!r}"
            )
            result = self._dispatch(
                "cop_onnx_inspect_model", {"model_path": None, "node_path": node.path()}
            )
            assert _is_failure(result), _unwrap(result)
            assert _get_error(result), "error message must be non-empty"
            assert node.parm("modelfile").eval() == modelfile_before, (
                "node_path branch must not mutate modelfile on an unconfigured node"
            )
            assert node.parm("model_inputs").eval() == n_inputs_before, (
                "node_path branch must not press setupshapes / change "
                "model_inputs on a READ-ONLY inspect"
            )
        finally:
            net.destroy()


# ---------------------------------------------------------------------------
# Test class: cop_onnx_list_models
# ---------------------------------------------------------------------------

class TestListModels:
    """cop_onnx_list_models globs .onnx files under a given root."""

    @classmethod
    def setup_class(cls):
        cls._dispatch = staticmethod(_get_dispatcher())

    def test_lists_fixtures_dir(self):
        result = self._dispatch("cop_onnx_list_models", {"roots": [_FIXTURES_DIR]})
        payload = _unwrap(result)
        assert _is_ok(result), payload
        paths = [m["path"] for m in payload["models"]]
        assert any("identity_dynamic.onnx" in p for p in paths), paths
        assert any("multi_input.onnx" in p for p in paths), paths

    def test_missing_root_noted_not_raised(self):
        result = self._dispatch(
            "cop_onnx_list_models", {"roots": ["/definitely/does/not/exist/probe"]}
        )
        payload = _unwrap(result)
        assert _is_ok(result), payload
        assert "/definitely/does/not/exist/probe" in payload["missing_roots"]
        assert payload["models"] == []


# ---------------------------------------------------------------------------
# Test registry and runner (for direct hython execution)
# ---------------------------------------------------------------------------

_TEST_CLASSES = [
    TestCopOnnxDispatcherImport,
    TestInspectModelSingleInput,
    TestInspectModelMultiInput,
    TestInspectModelFailureShape,
    TestListModels,
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
    print("cop_onnx_hython_smoke.py — pp12-113b")
    print("=" * 72)

    # Pre-check: is this the RED phase or GREEN phase?
    try:
        _get_dispatcher()
        print("[mode] GREEN — cop_onnx_handlers importable; running all tests.")
    except ImportError as exc:
        print(f"[mode] RED — {exc}")
        print("Expected RED failure. Wrapper pytest red gate has already been confirmed.")
        print("Hython-smoke RED gate confirmed: cop_onnx_handlers not yet implemented.")
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
