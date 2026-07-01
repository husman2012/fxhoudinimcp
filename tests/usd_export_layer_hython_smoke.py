"""Hython-smoke tests for the usd_export_layer dispatcher path (PP12-112 PR-3, pp12-112c).

Unit: pp12-112c
testVerificationSurface: hython-smoke
planSha: d29cbeba4743dff2f7006c135bd8c7ba591b6fcd7e77496b3d875ada4dcae254

Run under hython (Houdini's headless Python interpreter):
    hython tests/usd_export_layer_hython_smoke.py

Two-mode guard (mirrors usd_export_inspect_hython_smoke.py, pp12-112b):
  - If fxhoudinimcp_server.handlers.usd_export_handlers does not yet register
    'usd_export_layer' (hou-dev has not implemented it), dispatch() returns
    the dispatcher's own {"status": "error", "error": {"code": "UNKNOWN_COMMAND", ...}}
    shape -- this IS the expected RED signal for this file (the dispatcher
    module itself already exists from pp12-112b; only the NEW command is red).
  - Once hou-dev implements the gated handler + preview_fn + registration,
    the tests run and PASS GREEN.

Dispatcher-based (NOT direct handler calls) -- exercises the REAL
handler(**params) calling convention:
  - Importing usd_export_handlers triggers register_handler('usd_export_layer', ...)
    as a side effect (alongside the pp12-112b usd_inspect_layer/usd_validate
    registrations already shipped).
  - Tests call dispatch('usd_export_layer', {'node_path': ..., 'out_path': ...})
    -- the real dispatcher path, matching every other PP12 hython smoke.

Fixture (mirrors pp12-112b non-vacuous recipe -- orchestrator-confirmed):
  import hou
  sph = hou.node("/stage").createNode("sphere")   # a real Solaris LOP
  sph.cook(force=True)
  # export sph.path() (e.g. '/stage/sphere1'), NOT '/stage'

Acceptance tests covered (plan rev3 decomposition[hou-test].acceptanceTests):
  - dispatch('usd_export_layer', {'node_path': <fixture>, 'out_path': '$HIP/x.usdc'})
    writes crate-magic bytes (PXR-USDC), re-opens in pxr, root has NO /World or
    /root, format=='usdc', actual_format=='usdc', a validator_post postwrite
    verdict is present.
  - The same to '$HIP/x.usda' -> ascii header (#usda), format=='usda'.
  - capability_of('usd_export_layer') == Capability.MUTATING AND it has a
    registered preview_fn with preview_required=True (mirrors export_vat).
  - The pp12-112b read-only tools (usd_inspect_layer, usd_validate) remain
    Capability.READONLY (no regression from adding the gated sibling).
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap -- same pattern as usd_export_inspect_hython_smoke.py
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_PKG_PYTHON = os.path.join(_REPO_ROOT, "python")
_HOUDINI_HANDLERS = os.path.join(_REPO_ROOT, "houdini", "scripts", "python")

for _p in (_REPO_ROOT, _PKG_PYTHON, _HOUDINI_HANDLERS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
_ERRORS: list[tuple[str, str]] = []


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
# Dispatcher import (usd_export_handlers module already exists from pp12-112b;
# 'usd_export_layer' registration is the NEW red-gate surface for this file)
# ---------------------------------------------------------------------------
def _get_dispatcher_and_registry():
    """Import usd_export_handlers (triggers handler registration) and return
    (dispatch, capability_of, preview_of).

    Unlike usd_export_inspect_hython_smoke.py (pp12-112b), the module itself
    is expected to ALREADY exist (pp12-112b shipped it) -- the red signal for
    THIS file is the 'usd_export_layer' COMMAND being unregistered, which
    dispatch() surfaces as {"status": "error", "error": {"code": "UNKNOWN_COMMAND"}}
    rather than an ImportError. Tests below assert on that dispatch-level
    signal directly (see TestUsdExportLayerRedGate).
    """
    import fxhoudinimcp_server.handlers.usd_export_handlers as _handlers  # noqa: F401
    from fxhoudinimcp_server.dispatcher import dispatch, capability_of, preview_of
    return dispatch, capability_of, preview_of


# ---------------------------------------------------------------------------
# Real cooked LOP fixture (mirrors pp12-112b non-vacuous recipe)
# ---------------------------------------------------------------------------

def _build_sphere_lop() -> str:
    """Create a real cooked sphere LOP in /stage and return its node path.

    Recipe (orchestrator-confirmed against live hython, pp12-112b):
        sph = hou.node("/stage").createNode("sphere")
        sph.cook(force=True)
        return sph.path()  # e.g. '/stage/sphere1'
    """
    import hou

    stage_net = hou.node("/stage")
    if stage_net is None:
        raise RuntimeError(
            "hou.node('/stage') returned None -- /stage LOP network is missing. "
            "This should not occur in a headless hython 21 session. "
            "Cannot build non-vacuous fixture."
        )
    sph = stage_net.createNode("sphere")
    sph.cook(force=True)
    return sph.path()


def _destroy_node(path: str) -> None:
    try:
        import hou
        node = hou.node(path)
        if node is not None:
            node.destroy()
    except Exception:
        pass


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


def _is_unknown_command(result: dict) -> bool:
    """True when dispatch() reports the command itself is unregistered (RED gate)."""
    return (
        isinstance(result, dict)
        and result.get("status") == "error"
        and isinstance(result.get("error"), dict)
        and result["error"].get("code") == "UNKNOWN_COMMAND"
    )


# ---------------------------------------------------------------------------
# Test class: ULC-D1 -- RED gate: 'usd_export_layer' must be a registered command
# ---------------------------------------------------------------------------

class TestUsdExportLayerRedGate:
    """ULC-D1 (RED GATE): 'usd_export_layer' must be registered in the dispatcher.

    Before hou-dev implements the handler, dispatch('usd_export_layer', {...})
    returns {"status": "error", "error": {"code": "UNKNOWN_COMMAND", ...}} --
    this is the expected RED failure for this file. Once hou-dev registers
    the handler via register_handler('usd_export_layer', ...), this command
    resolves and the remaining test classes exercise real behavior.
    """

    @classmethod
    def setup_class(cls):
        _dispatch, _capability_of, _preview_of = _get_dispatcher_and_registry()
        cls._dispatch = staticmethod(_dispatch)
        cls._capability_of = staticmethod(_capability_of)
        cls._preview_of = staticmethod(_preview_of)

    def test_usd_export_layer_is_registered(self):
        """dispatch('usd_export_layer', ...) must NOT return UNKNOWN_COMMAND.

        FAILS RED (as intended) until hou-dev calls
        register_handler('usd_export_layer', usd_export_layer, Capability.MUTATING,
                          preview_fn=_preview_export_layer, preview_required=True).
        """
        result = self._dispatch("usd_export_layer", {
            "node_path": "/dummy/probe",
            "out_path": "$HIP/probe.usdc",
            "flatten": False,
            "default_prim": None,
        })
        assert not _is_unknown_command(result), (
            "'usd_export_layer' not found in the dispatcher registry. hou-dev must call "
            "register_handler('usd_export_layer', usd_export_layer, Capability.MUTATING, "
            "preview_fn=_preview_export_layer, preview_required=True) in "
            "usd_export_handlers.py. "
            f"Got dispatch result: {result!r}."
        )


# ---------------------------------------------------------------------------
# Test class: ULC-D2 -- .usdc write: crate magic, no /World|/root, format match
# ---------------------------------------------------------------------------

class TestUsdcExportWrite:
    """ULC-D2: exporting a real cooked sphere LOP to a .usdc path must produce
    a genuine USD crate-binary file with no injected /World or /root wrapper,
    and the handler's returned format/actual_format/validator_post must all
    agree it is a crate-format postwrite-validated file.
    """

    @classmethod
    def setup_class(cls):
        _dispatch, _capability_of, _preview_of = _get_dispatcher_and_registry()
        cls._dispatch = staticmethod(_dispatch)
        cls._capability_of = staticmethod(_capability_of)
        cls._preview_of = staticmethod(_preview_of)
        cls._sphere_path = _build_sphere_lop()

        import hou
        cls._out_path_expr = "$HIP/pp12_112c_test.usdc"
        cls._out_path_expanded = hou.text.expandString(cls._out_path_expr)

        raw = cls._dispatch("usd_export_layer", {
            "node_path": cls._sphere_path,
            "out_path": cls._out_path_expr,
            "flatten": False,
            "default_prim": None,
        })
        cls._payload = _unwrap(raw)

    def test_export_reports_ok_true(self):
        """The export must succeed (ok=True) on a real cooked sphere LOP."""
        assert self._payload.get("ok") is True, (
            f"usd_export_layer on a real cooked sphere LOP -> .usdc must return ok=True. "
            f"Got ok={self._payload.get('ok')!r}, error={self._payload.get('error')!r}."
        )

    def test_written_file_has_crate_magic_bytes(self):
        """The written file's first bytes must be the USD crate magic (PXR-USDC)."""
        assert os.path.exists(self._out_path_expanded), (
            f"Expected a written file at {self._out_path_expanded!r} but it does not exist."
        )
        with open(self._out_path_expanded, "rb") as fh:
            header = fh.read(16)
        assert header[:8] == b"PXR-USDC", (
            f"Written .usdc file must begin with the USD crate magic bytes b'PXR-USDC'. "
            f"Got header={header!r}."
        )

    def test_out_path_in_payload_matches_expanded_path(self):
        """red-3 (Minor): the success payload's 'out_path' is part of the
        documented handler return shape -- assert it matches the $HIP-expanded
        path actually written to disk."""
        assert self._payload.get("out_path") == self._out_path_expanded, (
            f"usd_export_layer success payload must include 'out_path' matching "
            f"the $HIP-expanded write target. Expected {self._out_path_expanded!r}, "
            f"got {self._payload.get('out_path')!r}."
        )

    def test_file_reopens_in_pxr_with_no_world_or_root(self):
        """The written file must re-open as a valid USD stage with NO /World or /root
        root prim (usd-publish-discipline.md -- no UE-style wrapper prims).
        """
        from pxr import Usd

        stage = Usd.Stage.Open(self._out_path_expanded)
        assert stage is not None, (
            f"Written file at {self._out_path_expanded!r} must re-open as a valid USD stage."
        )
        root_prim_paths = [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()]
        assert "/World" not in root_prim_paths, (
            f"Written layer must NOT contain a '/World' wrapper prim. "
            f"Got root prims={root_prim_paths!r}."
        )
        assert "/root" not in root_prim_paths, (
            f"Written layer must NOT contain a '/root' wrapper prim. "
            f"Got root prims={root_prim_paths!r}."
        )

    def test_format_and_actual_format_are_usdc(self):
        """Both the extension-derived format and the magic-bytes-derived
        actual_format must report 'usdc' for a .usdc export."""
        assert self._payload.get("format") == "usdc", (
            f"format (extension-derived) must be 'usdc'. Got {self._payload.get('format')!r}."
        )
        assert self._payload.get("actual_format") == "usdc", (
            f"actual_format (magic-bytes-derived, via format_from_magic_bytes reuse) "
            f"must be 'usdc'. Got {self._payload.get('actual_format')!r}."
        )

    def test_validator_post_postwrite_verdict_present(self):
        """The handler must run an INLINE post-write usd_validate and embed its
        result under 'validator_post' with mode=='postwrite'."""
        validator_post = self._payload.get("validator_post")
        assert isinstance(validator_post, dict), (
            f"validator_post must be a dict (the inline post-write usd_validate result). "
            f"Got {type(validator_post)!r}: {validator_post!r}."
        )
        assert validator_post.get("mode") == "postwrite", (
            f"validator_post.mode must be 'postwrite' (out_path + actual_format both set). "
            f"Got mode={validator_post.get('mode')!r}."
        )
        assert "verdict" in validator_post, (
            f"validator_post must carry a 'verdict' key. Got keys={list(validator_post.keys())!r}."
        )

    @classmethod
    def teardown_class(cls):
        _destroy_node(cls._sphere_path)
        try:
            if os.path.exists(cls._out_path_expanded):
                os.remove(cls._out_path_expanded)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test class: ULC-D3 -- .usda write: ascii header, format=='usda'
# ---------------------------------------------------------------------------

class TestUsdaExportWrite:
    """ULC-D3: exporting to a .usda path must produce an ASCII-format USD file
    (magic prefix b'#usda') and the handler must report format=='usda'.
    """

    @classmethod
    def setup_class(cls):
        _dispatch, _capability_of, _preview_of = _get_dispatcher_and_registry()
        cls._dispatch = staticmethod(_dispatch)
        cls._capability_of = staticmethod(_capability_of)
        cls._preview_of = staticmethod(_preview_of)
        cls._sphere_path = _build_sphere_lop()

        import hou
        cls._out_path_expr = "$HIP/pp12_112c_test.usda"
        cls._out_path_expanded = hou.text.expandString(cls._out_path_expr)

        raw = cls._dispatch("usd_export_layer", {
            "node_path": cls._sphere_path,
            "out_path": cls._out_path_expr,
            "flatten": False,
            "default_prim": None,
        })
        cls._payload = _unwrap(raw)

    def test_export_reports_ok_true(self):
        """The export must succeed (ok=True) on a real cooked sphere LOP -> .usda."""
        assert self._payload.get("ok") is True, (
            f"usd_export_layer on a real cooked sphere LOP -> .usda must return ok=True. "
            f"Got ok={self._payload.get('ok')!r}, error={self._payload.get('error')!r}."
        )

    def test_written_file_has_ascii_header(self):
        """The written file's first bytes must be the USD ASCII magic (#usda)."""
        assert os.path.exists(self._out_path_expanded), (
            f"Expected a written file at {self._out_path_expanded!r} but it does not exist."
        )
        with open(self._out_path_expanded, "rb") as fh:
            header = fh.read(8)
        assert header[:5] == b"#usda", (
            f"Written .usda file must begin with the USD ASCII magic bytes b'#usda'. "
            f"Got header={header!r}."
        )

    def test_format_is_usda(self):
        """format must report 'usda' for a .usda export."""
        assert self._payload.get("format") == "usda", (
            f"format (extension-derived) must be 'usda'. Got {self._payload.get('format')!r}."
        )
        assert self._payload.get("actual_format") == "usda", (
            f"actual_format (magic-bytes-derived) must be 'usda'. "
            f"Got {self._payload.get('actual_format')!r}."
        )

    def test_out_path_in_payload_matches_expanded_path(self):
        """red-3 (Minor): the success payload's 'out_path' is part of the
        documented handler return shape -- assert it matches the $HIP-expanded
        path actually written to disk."""
        assert self._payload.get("out_path") == self._out_path_expanded, (
            f"usd_export_layer success payload must include 'out_path' matching "
            f"the $HIP-expanded write target. Expected {self._out_path_expanded!r}, "
            f"got {self._payload.get('out_path')!r}."
        )

    @classmethod
    def teardown_class(cls):
        _destroy_node(cls._sphere_path)
        try:
            if os.path.exists(cls._out_path_expanded):
                os.remove(cls._out_path_expanded)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test class: ULC-D4 -- Capability.MUTATING + registered preview_fn (109 gate)
# ---------------------------------------------------------------------------

class TestUsdExportLayerGateRegistration:
    """ULC-D4: 'usd_export_layer' must be registered as Capability.MUTATING
    WITH a preview_fn (preview_required=True) -- mirroring every shipped
    gated export handler (export_vat, export_alembic_ue, ...).

    An un-previewed mutating handler was FOLD plan-2 (BLOCKER) in the rev2/3
    adjudication: every mutating write handler in this codebase must be
    preview-gated so the 109 gate can show the operator a pre-flight preview
    before approving the write.
    """

    @classmethod
    def setup_class(cls):
        _dispatch, _capability_of, _preview_of = _get_dispatcher_and_registry()
        cls._dispatch = staticmethod(_dispatch)
        cls._capability_of = staticmethod(_capability_of)
        cls._preview_of = staticmethod(_preview_of)

    def test_capability_is_mutating(self):
        """capability_of('usd_export_layer') must be Capability.MUTATING."""
        from fxhoudinimcp_server.dispatcher import Capability

        cap = self._capability_of("usd_export_layer")
        assert cap == Capability.MUTATING, (
            f"'usd_export_layer' must be registered with Capability.MUTATING "
            f"(it writes a USD layer to disk). Got cap={cap!r}."
        )

    def test_preview_fn_is_registered_and_required(self):
        """preview_of('usd_export_layer') must have a non-None preview_fn with
        preview_required=True (mirrors export_vat's registration)."""
        preview_record = self._preview_of("usd_export_layer")
        assert preview_record.get("preview_fn") is not None, (
            "'usd_export_layer' must register a preview_fn (_preview_export_layer) "
            "so the 109 gate can show a pre-flight preview before approval. "
            f"Got preview_record={preview_record!r}."
        )
        assert preview_record.get("preview_required") is True, (
            f"'usd_export_layer' must set preview_required=True (a preview_fn failure "
            f"must DENY the call, not silently queue without a preview). "
            f"Got preview_required={preview_record.get('preview_required')!r}."
        )

    def test_preview_fn_returns_approval_payload_dict(self):
        """Calling the registered preview_fn(params) directly must return a dict
        (the 109-gate approval payload), not raise, for a valid params dict.

        This exercises the preview_fn(params: dict) -> dict positional-argument
        convention directly -- the rev3 lockedFieldContract's critical FOLD
        (plan2-1): the HANDLER is keyword-only (dispatcher calls handler(**params))
        but the PREVIEW_FN takes ONE POSITIONAL params: dict argument (the gate
        calls preview_fn(params), matching _preview_vat's shape exactly).
        """
        preview_record = self._preview_of("usd_export_layer")
        preview_fn = preview_record.get("preview_fn")
        assert preview_fn is not None, "preview_fn must be registered (see prior test)."

        sphere_path = _build_sphere_lop()
        try:
            import hou
            out_path_expr = "$HIP/pp12_112c_preview_test.usdc"
            expanded = hou.text.expandString(out_path_expr)

            # preview_fn is called POSITIONALLY with a single params dict -- NOT
            # with **kwargs. This is the opposite convention from the handler.
            params = {
                "node_path": sphere_path,
                "out_path": out_path_expr,
                "flatten": False,
                "default_prim": None,
            }
            result = preview_fn(params)

            assert isinstance(result, dict), (
                f"preview_fn(params) must return a dict (the 109-gate approval payload). "
                f"Got {type(result)!r}: {result!r}."
            )

            # red-1 (Major): the rev3 contract's preview payload has SIX keys.
            # Assert ALL SIX with their expected values -- not just the two
            # this test previously checked (resolved_format, pre_validation).
            assert result.get("out_path") == expanded, (
                f"preview_fn result 'out_path' must be the $HIP-expanded path "
                f"(hou.text.expandString(out_path)). Expected {expanded!r}, "
                f"got {result.get('out_path')!r}."
            )
            assert result.get("resolved_format") == "usdc", (
                f"resolved_format for a .usdc out_path must be 'usdc'. "
                f"Got {result.get('resolved_format')!r}."
            )
            assert "pre_validation" in result, (
                f"preview_fn result must include 'pre_validation' (the pre-flight "
                f"usd_validate reused in preflight mode). Got keys={list(result.keys())!r}."
            )
            assert isinstance(result["pre_validation"], dict), (
                f"pre_validation must be a dict. Got {type(result['pre_validation'])!r}."
            )
            assert result.get("flatten") == params["flatten"], (
                f"preview_fn result 'flatten' must echo the request's flatten value "
                f"({params['flatten']!r}). Got {result.get('flatten')!r}."
            )
            assert result.get("default_prim") == params["default_prim"], (
                f"preview_fn result 'default_prim' must echo the request's default_prim "
                f"value ({params['default_prim']!r}). Got {result.get('default_prim')!r}."
            )
            assert result.get("no_world_wrapper") is True, (
                f"preview_fn result 'no_world_wrapper' must be True (the preview attests "
                f"no /World or /root wrapper will be injected). "
                f"Got {result.get('no_world_wrapper')!r}."
            )

            # red-2 (Major): the preview_fn must be genuinely READ-ONLY -- it
            # must write NOTHING at out_path. Assert this BEFORE the finally
            # cleanup so a preview that accidentally writes the file is caught
            # as a real failure, not silently cleaned up first.
            assert not os.path.exists(expanded), (
                f"preview_fn must be READ-ONLY and must not write any file at "
                f"out_path. Found an unexpected file at {expanded!r} after calling "
                f"preview_fn -- the preview_fn must not perform the export."
            )
        finally:
            _destroy_node(sphere_path)
            try:
                import hou as _hou
                expanded = _hou.text.expandString(out_path_expr)
                if os.path.exists(expanded):
                    os.remove(expanded)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Test class: ULC-D5 -- no regression: pp12-112b read-only tools stay READONLY
# ---------------------------------------------------------------------------

class TestReadOnlyToolsUnchanged:
    """ULC-D5: usd_inspect_layer and usd_validate (pp12-112b) must remain
    Capability.READONLY after the gated sibling (usd_export_layer) is added.
    A regression here would mean the new registration accidentally clobbered
    or altered the existing pp12-112b registrations.
    """

    @classmethod
    def setup_class(cls):
        _dispatch, _capability_of, _preview_of = _get_dispatcher_and_registry()
        cls._dispatch = staticmethod(_dispatch)
        cls._capability_of = staticmethod(_capability_of)
        cls._preview_of = staticmethod(_preview_of)

    def test_usd_inspect_layer_still_readonly(self):
        from fxhoudinimcp_server.dispatcher import Capability

        cap = self._capability_of("usd_inspect_layer")
        assert cap == Capability.READONLY, (
            f"'usd_inspect_layer' (pp12-112b) must remain Capability.READONLY. Got cap={cap!r}."
        )

    def test_usd_validate_still_readonly(self):
        from fxhoudinimcp_server.dispatcher import Capability

        cap = self._capability_of("usd_validate")
        assert cap == Capability.READONLY, (
            f"'usd_validate' (pp12-112b) must remain Capability.READONLY. Got cap={cap!r}."
        )


# ---------------------------------------------------------------------------
# Test registry and runner (for direct hython execution)
# ---------------------------------------------------------------------------

_TEST_CLASSES = [
    TestUsdExportLayerRedGate,
    TestUsdcExportWrite,
    TestUsdaExportWrite,
    TestUsdExportLayerGateRegistration,
    TestReadOnlyToolsUnchanged,
]


def _run_class(cls):
    """Instantiate the class and run all test_* methods; record pass/fail."""
    obj = cls()
    setup_failed = False

    if hasattr(cls, "setup_class"):
        try:
            cls.setup_class()
        except Exception as exc:
            _record_fail(f"{cls.__name__}.setup_class", repr(exc))
            setup_failed = True

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
    print("usd_export_layer_hython_smoke.py -- pp12-112c")
    print("=" * 72)

    try:
        dispatch, capability_of, preview_of = _get_dispatcher_and_registry()
    except ImportError as exc:
        print(f"[mode] RED -- usd_export_handlers module itself is not importable: {exc}")
        print("Expected only if pp12-112b's module is somehow absent. "
              "For pp12-112c's own red gate, the module import succeeds but "
              "'usd_export_layer' is UNKNOWN_COMMAND -- see TestUsdExportLayerRedGate.")
        return 1

    probe = dispatch("usd_export_layer", {
        "node_path": "/dummy/probe", "out_path": "$HIP/probe.usdc",
        "flatten": False, "default_prim": None,
    })
    if _is_unknown_command(probe):
        print("[mode] RED -- 'usd_export_layer' not yet registered (expected before hou-dev).")
        print("Running TestUsdExportLayerRedGate only (the other classes require the real handler).")
        _run_class(TestUsdExportLayerRedGate)
        print("\n" + "=" * 72)
        print(f"Results: {_PASS_COUNT} passed, {_FAIL_COUNT} failed")
        return 1  # RED gate confirmed -- exit non-zero

    print("[mode] GREEN -- 'usd_export_layer' registered; running all tests.")
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
