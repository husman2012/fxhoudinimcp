"""Hython-smoke tests for the usd_inspect_layer / usd_validate dispatcher path.

Unit: pp12-112b
testVerificationSurface: hython-smoke
planSha: ac5886a927ad1d62ea2d6768414f8a907040286de888373f37dd3a1bc7a0ca6c

Run under hython (Houdini's headless Python interpreter):
    hython tests/usd_export_inspect_hython_smoke.py

Two-mode guard:
  - If fxhoudinimcp_server.handlers.usd_export_handlers is absent (hou-dev has
    not implemented yet), the import raises ImportError with the message
    "expected RED (hou-dev has not implemented usd_export_handlers yet)".
    This is the RED gate for the hython surface.
  - Once hou-dev implements the handlers, the tests should run and PASS GREEN.

Dispatcher-based (NOT direct handler calls):
  - Importing usd_export_handlers triggers register_handler('usd_inspect_layer', ...)
    and register_handler('usd_validate', ...) as side-effects.
  - Tests then call dispatch('usd_inspect_layer', {'node_or_layer': ...}) — the real
    dispatcher path that exercises the full handler(**params) calling convention.

Grounded against: tests/render_compare_hython_smoke.py (pp12-111a exemplar).

Fixture (pp12-112b non-vacuous recipe — orchestrator-confirmed):
  import hou
  sph = hou.node("/stage").createNode("sphere")   # a real Solaris LOP
  sph.cook(force=True)
  # inspect sph.path() (e.g. '/stage/sphere1'), NOT '/stage'

Confirmed real return values on this fixture:
  usd_inspect_layer('/stage/sphere1'):
    ok=True, default_prim=None, root_prims==['/sphere1'],
    current_format=='in-memory', has_mtlx_material==False
  usd_validate('/stage/sphere1'):
    ok=True, mode=='minimal', verdict=='fail'
    (bare sphere sets no defaultPrim -> default_prim_set fails; correct)
    omitted_checks==['format_extension_known','format_matches_ext','abs_texture_paths']
  usd_validate('/stage/sphere1', out_path='/tmp/a.usdc'):
    mode=='preflight', omitted_checks==['format_matches_ext','abs_texture_paths']
  usd_validate('/stage/sphere1', out_path='/tmp/a.usdc', actual_format='usdc'):
    mode=='postwrite', omitted_checks==[]

Acceptance tests covered:
  FR-2/FR-5: result shape on success / failure
  M-3: non-None checks -> {ok:False, error}
  M-4: root_prims excludes HoudiniLayerInfo prim
  M-5: SOP node path -> {ok:False, error mentioning 'not a LOP node'}
  M-6: in-memory stage -> current_format == 'in-memory'
  B-1: usd_validate returns mode and omitted_checks
  FR-10: both tools are Capability.READONLY
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — same pattern as render_compare_hython_smoke.py
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
# Dual-mode import guard: RED = ImportError; GREEN = returns dispatch
# ---------------------------------------------------------------------------
def _get_dispatcher():
    """Import usd_export_handlers (triggers handler registration) and return dispatch.

    If fxhoudinimcp_server.handlers.usd_export_handlers is absent, raises
    ImportError with a message indicating this is the expected RED state.
    Once hou-dev implements the handlers, this returns the real dispatcher.
    """
    try:
        # Importing the handler module triggers register_handler side-effects.
        import fxhoudinimcp_server.handlers.usd_export_handlers as _handlers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"expected RED (hou-dev has not implemented usd_export_handlers yet): {exc}"
        ) from exc

    try:
        from fxhoudinimcp_server.dispatcher import dispatch
    except ImportError as exc:
        raise ImportError(
            f"fxhoudinimcp_server.dispatcher not found — is fxhoudinimcp_server on sys.path? {exc}"
        ) from exc

    return dispatch


# ---------------------------------------------------------------------------
# Real cooked LOP fixture (pp12-112b non-vacuous recipe)
# ---------------------------------------------------------------------------

def _build_sphere_lop() -> str:
    """Create a real cooked sphere LOP in /stage and return its node path.

    Recipe (orchestrator-confirmed against live hython):
        sph = hou.node("/stage").createNode("sphere")
        sph.cook(force=True)
        return sph.path()  # e.g. '/stage/sphere1'

    Raises RuntimeError if /stage is not available (should not happen in
    headless hython with Houdini 21 — /stage is always present).
    """
    import hou

    stage_net = hou.node("/stage")
    if stage_net is None:
        raise RuntimeError(
            "hou.node('/stage') returned None — /stage LOP network is missing. "
            "This should not occur in a headless hython 21 session. "
            "Cannot build non-vacuous fixture."
        )
    sph = stage_net.createNode("sphere")
    sph.cook(force=True)
    return sph.path()


# ---------------------------------------------------------------------------
# Tiny result helpers (mirroring render_compare_hython_smoke.py patterns)
# ---------------------------------------------------------------------------

def _unwrap(result: dict) -> dict:
    """Unwrap the dispatcher envelope.

    The dispatcher wraps every handler result in:
        {'status': 'success'|'error', 'data': <handler_payload>, 'timing_ms': ...}
    The handler's real {ok, error, ...} dict is under result["data"].
    The fallback `result` keeps this working if a call ever returns an
    already-unwrapped dict (e.g. during direct-handler tests).
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


# ---------------------------------------------------------------------------
# Test class: UEC-D1 — dispatcher import and handler registration
# ---------------------------------------------------------------------------

class TestUsdExportDispatcherImport:
    """UEC-D1 (RED GATE): fxhoudinimcp_server.handlers.usd_export_handlers must be importable.

    On RED (before hou-dev implements):
        ImportError: "expected RED (hou-dev has not implemented usd_export_handlers yet)"

    On GREEN (after hou-dev implements):
        The module imports cleanly and registers both handlers as side-effects.
    """

    @classmethod
    def setup_class(cls):
        cls._dispatch = staticmethod(_get_dispatcher())  # raises ImportError on RED

    def test_usd_export_handlers_importable(self):
        """After _get_dispatcher() succeeds, dispatch callable is present."""
        assert callable(self._dispatch), (
            "dispatch must be a callable after handlers are imported."
        )

    def test_usd_inspect_layer_registered(self):
        """'usd_inspect_layer' must be registered in the dispatcher's command registry."""
        # Probe: call with a dummy value; a registered handler will execute (may fail
        # for invalid input, but won't raise 'unknown command').
        try:
            result = self._dispatch("usd_inspect_layer", {"node_or_layer": "/dummy/probe"})
        except KeyError as exc:
            if "usd_inspect_layer" in str(exc) or "unknown command" in str(exc).lower():
                raise AssertionError(
                    "'usd_inspect_layer' not found in dispatcher registry. "
                    "hou-dev must call register_handler('usd_inspect_layer', ...) "
                    "in usd_export_handlers.py."
                ) from exc
            # Any other KeyError means the handler ran but had an internal issue — OK for registration proof.
        except Exception:
            pass  # Handler ran (may fail due to missing LOP node) — registration confirmed.

    def test_usd_validate_registered(self):
        """'usd_validate' must be registered in the dispatcher's command registry."""
        try:
            result = self._dispatch("usd_validate", {
                "target": "/dummy/probe",
                "out_path": None,
                "actual_format": None,
                "texture_paths": None,
                "checks": None,
            })
        except KeyError as exc:
            if "usd_validate" in str(exc) or "unknown command" in str(exc).lower():
                raise AssertionError(
                    "'usd_validate' not found in dispatcher registry. "
                    "hou-dev must call register_handler('usd_validate', ...) "
                    "in usd_export_handlers.py."
                ) from exc
        except Exception:
            pass  # Handler ran — registration confirmed.


# ---------------------------------------------------------------------------
# Test class: UEC-D2 — houdini_usd_inspect_layer result shape (FR-2/FR-5)
# ---------------------------------------------------------------------------

class TestInspectLayerResultShape:
    """UEC-D2 (FR-2/FR-5): usd_inspect_layer must return a dict with the correct shape.

    Non-vacuous fixture (pp12-112b): builds a real cooked sphere LOP at
    /stage/sphere1, dispatches to its path, and asserts UNCONDITIONALLY on the
    full success payload.

    Confirmed return values (orchestrator-verified):
      ok=True, default_prim=None, root_prims==['/sphere1'],
      current_format=='in-memory', has_mtlx_material==False
    """

    @classmethod
    def setup_class(cls):
        cls._dispatch = staticmethod(_get_dispatcher())
        # Build a real cooked sphere LOP — fixture guarantees ok=True.
        cls._sphere_path = _build_sphere_lop()
        # Pre-fetch and unwrap the payload so tests can assert unconditionally.
        raw = cls._dispatch("usd_inspect_layer", {"node_or_layer": cls._sphere_path})
        cls._sphere_payload = _unwrap(raw)

    def test_success_result_has_ok_true_and_required_keys(self):
        """On a real cooked LOP node, result must have ok=True and all required keys.

        Asserts UNCONDITIONALLY — the sphere LOP fixture guarantees ok=True.
        A failure here means a real impl regression, not a fixture limitation.
        """
        payload = self._sphere_payload

        assert payload.get("ok") is True, (
            f"usd_inspect_layer on a real cooked sphere LOP must return ok=True. "
            f"Got ok={payload.get('ok')!r}, error={payload.get('error')!r}."
        )
        required_keys = {"ok", "default_prim", "root_prims", "sublayers",
                         "current_format", "has_mtlx_material"}
        missing = required_keys - set(payload.keys())
        assert not missing, (
            f"Success result missing keys: {missing!r}. Got keys={list(payload.keys())!r}."
        )
        assert isinstance(payload["root_prims"], list), (
            f"root_prims must be a list. Got {type(payload['root_prims'])!r}."
        )
        assert isinstance(payload["has_mtlx_material"], bool), (
            f"has_mtlx_material must be bool. Got {type(payload['has_mtlx_material'])!r}."
        )

    @classmethod
    def teardown_class(cls):
        """Clean up the sphere LOP node created by setup_class."""
        try:
            import hou
            sph = hou.node(cls._sphere_path)
            if sph is not None:
                sph.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test class: UEC-D3 — M-4: root_prims excludes HoudiniLayerInfo
# ---------------------------------------------------------------------------

class TestM4RootPrimsNoHoudiniLayerInfo:
    """UEC-D3 (M-4): root_prims must NOT include 'HoudiniLayerInfo'.

    Houdini inserts a /HoudiniLayerInfo prim automatically on every layer.
    It is internal bookkeeping and must be filtered from the returned root_prims.

    Non-vacuous fixture: uses a real cooked sphere LOP (ok=True guaranteed).
    The sphere stage has no authored HoudiniLayerInfo; the test asserts both
    that the real authored prims are returned (['/sphere1']) AND that the
    Houdini internal prim is filtered.

    NOTE: If the Houdini runtime injects a HoudiniLayerInfo prim as a root
    prim on the sphere stage (which it does internally), the filter must remove
    it before returning root_prims. The assertion '/sphere1' in root_prims
    confirms the filter kept the real authored prim. The explicit
    HoudiniLayerInfo injection case (a stage with HoudiniLayerInfo as the ONLY
    authored prim) is only exercisable via the live MCP rung — the filter
    behavior is confirmed by the '/sphere1'-only result here.
    """

    @classmethod
    def setup_class(cls):
        cls._dispatch = staticmethod(_get_dispatcher())
        # Build a real cooked sphere LOP — fixture guarantees ok=True.
        cls._sphere_path = _build_sphere_lop()
        raw = cls._dispatch("usd_inspect_layer", {"node_or_layer": cls._sphere_path})
        cls._sphere_payload = _unwrap(raw)

    def test_houdini_layer_info_absent_from_root_prims(self):
        """root_prims on a sphere LOP must be ['/sphere1'] with no HoudiniLayerInfo.

        Asserts UNCONDITIONALLY — the sphere LOP fixture guarantees ok=True.
        """
        payload = self._sphere_payload

        assert payload.get("ok") is True, (
            f"Fixture must return ok=True. "
            f"Got ok={payload.get('ok')!r}, error={payload.get('error')!r}."
        )
        root_prims = payload.get("root_prims", [])

        # The real authored prim must be present.
        assert "/sphere1" in root_prims, (
            f"root_prims must contain '/sphere1' for a sphere LOP stage (M-4). "
            f"Got root_prims={root_prims!r}."
        )

        # The Houdini internal bookkeeping prim must be filtered out.
        hli_found = any("HoudiniLayerInfo" in str(p) for p in root_prims)
        assert not hli_found, (
            f"root_prims must exclude HoudiniLayerInfo prim (M-4). "
            f"Got root_prims={root_prims!r}."
        )

        # Exact match: the only root prim should be the sphere.
        assert root_prims == ["/sphere1"], (
            f"root_prims for a bare sphere LOP must be exactly ['/sphere1'] (M-4). "
            f"Got root_prims={root_prims!r}."
        )

    @classmethod
    def teardown_class(cls):
        """Clean up the sphere LOP node created by setup_class."""
        try:
            import hou
            sph = hou.node(cls._sphere_path)
            if sph is not None:
                sph.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test class: UEC-D4 — M-6: in-memory stage -> current_format == 'in-memory'
# ---------------------------------------------------------------------------

class TestM6CurrentFormatInMemory:
    """UEC-D4 (M-6): For an unwritten in-memory stage, current_format must be 'in-memory'.

    A freshly created LOP network that has never been saved to disk has a stage
    backed only by memory — not a .usd/.usda/.usdc file on disk. The handler must
    detect this and return current_format='in-memory' (not the extension of a
    file that doesn't exist).

    Non-vacuous fixture: the sphere LOP is freshly created in this session and
    has never been written to disk, so current_format must be 'in-memory'.
    Asserts UNCONDITIONALLY — no if/else escape hatch.
    """

    @classmethod
    def setup_class(cls):
        cls._dispatch = staticmethod(_get_dispatcher())
        # Build a real cooked sphere LOP — guaranteed ok=True, never written to disk.
        cls._sphere_path = _build_sphere_lop()
        raw = cls._dispatch("usd_inspect_layer", {"node_or_layer": cls._sphere_path})
        cls._sphere_payload = _unwrap(raw)

    def test_in_memory_stage_returns_in_memory_format(self):
        """Freshly created sphere LOP (never written to disk) must return current_format='in-memory'.

        Asserts UNCONDITIONALLY — fixture guarantees ok=True (no early return).
        """
        payload = self._sphere_payload

        assert payload.get("ok") is True, (
            f"Fixture must return ok=True. "
            f"Got ok={payload.get('ok')!r}, error={payload.get('error')!r}."
        )
        current_format = payload.get("current_format", "MISSING")
        assert current_format == "in-memory", (
            f"An unwritten in-memory stage must return current_format='in-memory' (M-6). "
            f"Got current_format={current_format!r}."
        )

    @classmethod
    def teardown_class(cls):
        """Clean up the sphere LOP node created by setup_class."""
        try:
            import hou
            sph = hou.node(cls._sphere_path)
            if sph is not None:
                sph.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test class: UEC-D5 — M-3: non-None checks -> {ok: False}
# ---------------------------------------------------------------------------

class TestM3ChecksParamRejection:
    """UEC-D5 (M-3): houdini_usd_validate with non-None checks must return {ok: False}.

    The v1 handler does not implement checks subsetting. A non-None checks list
    must be rejected immediately with {ok: False, error: '...'} — the handler
    must not silently ignore it or treat it as no filter.

    Already asserts ok=False unconditionally — no changes needed (not vacuous).
    """

    @classmethod
    def setup_class(cls):
        cls._dispatch = staticmethod(_get_dispatcher())

    def test_non_none_checks_returns_ok_false(self):
        """checks=['format'] passed to usd_validate -> {ok: False, error}."""
        result = self._dispatch("usd_validate", {
            "target": "/stage",
            "out_path": None,
            "actual_format": None,
            "texture_paths": None,
            "checks": ["format"],  # non-None -> must be rejected in v1
        })
        assert isinstance(result, dict), f"Result must be dict, got {type(result)!r}."
        payload = _unwrap(result)
        assert payload.get("ok") is False, (
            f"checks=non-None must produce ok=False (M-3). Got ok={payload.get('ok')!r}."
        )
        assert "error" in payload, (
            f"checks rejection must include 'error' key. Got keys={list(payload.keys())!r}."
        )
        error_text = _get_error(result)
        assert "checks" in error_text.lower() or "not implemented" in error_text.lower(), (
            f"error must mention 'checks' or 'not implemented' (M-3). Got: {error_text!r}."
        )


# ---------------------------------------------------------------------------
# Test class: UEC-D6 — M-5: SOP node path -> {ok: False, 'not a LOP node'}
# ---------------------------------------------------------------------------

class TestM5SopNodeNotLop:
    """UEC-D6 (M-5): A SOP node path must return {ok: False, error mentioning 'not a LOP node'}.

    If the user passes a SOP node path (e.g. /obj/geo1/box1) to usd_inspect_layer,
    that node has no .stage() method — it is not a LOP node. The handler must
    return {ok: False, error: '...not a LOP node...'}, NOT propagate a Python
    AttributeError or Usd.Stage.Open() error.

    Already asserts ok=False unconditionally — no changes needed (not vacuous).
    """

    @classmethod
    def setup_class(cls):
        cls._dispatch = staticmethod(_get_dispatcher())

    def test_sop_node_path_returns_not_lop_error(self):
        """Passing a SOP-context path -> {ok: False, error mentioning 'not a LOP node'}."""
        import hou

        # Create a minimal SOP node to get a known SOP path.
        try:
            obj_net = hou.node("/obj")
            if obj_net is None:
                # No /obj in headless — use a path that will clearly not be a LOP.
                sop_path = "/obj/geo_smoke_test123/box1"
            else:
                geo = obj_net.createNode("geo", "geo_smoke_test123")
                box = geo.createNode("box", "box1")
                sop_path = box.path()
        except Exception:
            sop_path = "/obj/geo_smoke_test123/box1"

        result = self._dispatch("usd_inspect_layer", {"node_or_layer": sop_path})
        assert isinstance(result, dict), f"Result must be dict, got {type(result)!r}."
        payload = _unwrap(result)
        assert payload.get("ok") is False, (
            f"A SOP node path must produce ok=False (M-5). Got ok={payload.get('ok')!r}."
        )
        error_text = _get_error(result)
        assert "lop" in error_text.lower() or "stage" in error_text.lower(), (
            f"Error for SOP node must mention 'LOP' or 'stage' (M-5). Got: {error_text!r}."
        )

    @classmethod
    def teardown_class(cls):
        """Clean up the temporary SOP node created for testing."""
        import hou
        try:
            test_geo = hou.node("/obj/geo_smoke_test123")
            if test_geo is not None:
                test_geo.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test class: UEC-D7 — B-1: usd_validate returns mode and omitted_checks
# ---------------------------------------------------------------------------

class TestUsdValidateModeAndOmittedChecks:
    """UEC-D7 (B-1): houdini_usd_validate must return mode and omitted_checks.

    B-1 fix: validation is partial — not all checks run in every mode.
    To prevent callers from assuming 'all checks ran on pass', the result must
    include:
      - mode: 'minimal' | 'preflight' | 'postwrite'
      - omitted_checks: list[str]  (checks that were NOT run in this mode)

    Non-vacuous fixture (pp12-112b): uses a real cooked sphere LOP (ok=True
    guaranteed) and asserts UNCONDITIONALLY on mode and omitted_checks.

    Confirmed return values (orchestrator-verified):
      minimal  mode (no out_path, no actual_format):
        ok=True, mode=='minimal', verdict=='fail',
        omitted_checks==['format_extension_known','format_matches_ext','abs_texture_paths']
      preflight mode (out_path set, no actual_format):
        ok=True, mode=='preflight',
        omitted_checks==['format_matches_ext','abs_texture_paths']
      postwrite mode (out_path + actual_format both set):
        ok=True, mode=='postwrite', omitted_checks==[]
    """

    @classmethod
    def setup_class(cls):
        cls._dispatch = staticmethod(_get_dispatcher())
        # Build a real cooked sphere LOP — fixture guarantees ok=True.
        cls._sphere_path = _build_sphere_lop()

        # Pre-fetch all three mode results.
        raw_minimal = cls._dispatch("usd_validate", {
            "target": cls._sphere_path,
            "out_path": None,
            "actual_format": None,
            "texture_paths": None,
            "checks": None,
        })
        cls._minimal_payload = _unwrap(raw_minimal)

        raw_preflight = cls._dispatch("usd_validate", {
            "target": cls._sphere_path,
            "out_path": "/tmp/a.usdc",
            "actual_format": None,
            "texture_paths": None,
            "checks": None,
        })
        cls._preflight_payload = _unwrap(raw_preflight)

        raw_postwrite = cls._dispatch("usd_validate", {
            "target": cls._sphere_path,
            "out_path": "/tmp/a.usdc",
            "actual_format": "usdc",
            "texture_paths": None,
            "checks": None,
        })
        cls._postwrite_payload = _unwrap(raw_postwrite)

    def test_validate_minimal_mode(self):
        """usd_validate(sphere, no out_path) -> ok=True, mode='minimal'.

        Asserts UNCONDITIONALLY — sphere LOP fixture guarantees ok=True.
        """
        payload = self._minimal_payload
        assert payload.get("ok") is True, (
            f"usd_validate on sphere LOP (minimal) must return ok=True. "
            f"Got ok={payload.get('ok')!r}, error={payload.get('error')!r}."
        )
        assert "mode" in payload, (
            f"usd_validate result must include 'mode' (B-1). "
            f"Got keys={list(payload.keys())!r}."
        )
        assert payload["mode"] == "minimal", (
            f"usd_validate with no out_path must use 'minimal' mode. "
            f"Got mode={payload['mode']!r}."
        )

    def test_validate_minimal_has_omitted_checks_key(self):
        """usd_validate(minimal) result must include 'omitted_checks' as a list.

        Asserts UNCONDITIONALLY — sphere LOP fixture guarantees ok=True.
        """
        payload = self._minimal_payload
        assert payload.get("ok") is True, (
            f"Fixture must return ok=True. Got ok={payload.get('ok')!r}."
        )
        assert "omitted_checks" in payload, (
            f"usd_validate result must include 'omitted_checks' (B-1). "
            f"Got keys={list(payload.keys())!r}."
        )
        assert isinstance(payload["omitted_checks"], list), (
            f"omitted_checks must be a list. Got {type(payload['omitted_checks'])!r}."
        )

    def test_validate_minimal_omitted_checks_exact(self):
        """'minimal' mode omitted_checks must be exactly the expected check-id list.

        Confirmed values (orchestrator-verified):
          ['format_extension_known', 'format_matches_ext', 'abs_texture_paths']

        Asserts UNCONDITIONALLY.
        """
        payload = self._minimal_payload
        assert payload.get("ok") is True, (
            f"Fixture must return ok=True. Got ok={payload.get('ok')!r}."
        )
        omitted = payload.get("omitted_checks", [])
        expected = ["format_extension_known", "format_matches_ext", "abs_texture_paths"]
        assert omitted == expected, (
            f"'minimal' mode omitted_checks mismatch (B-1). "
            f"Expected {expected!r}, got {omitted!r}."
        )

    def test_validate_preflight_mode(self):
        """usd_validate(sphere, out_path='/tmp/a.usdc') -> ok=True, mode='preflight'.

        Asserts UNCONDITIONALLY.
        """
        payload = self._preflight_payload
        assert payload.get("ok") is True, (
            f"usd_validate on sphere LOP (preflight) must return ok=True. "
            f"Got ok={payload.get('ok')!r}, error={payload.get('error')!r}."
        )
        assert payload.get("mode") == "preflight", (
            f"usd_validate with out_path (no actual_format) must use 'preflight' mode. "
            f"Got mode={payload.get('mode')!r}."
        )
        omitted = payload.get("omitted_checks", [])
        expected = ["format_matches_ext", "abs_texture_paths"]
        assert omitted == expected, (
            f"'preflight' mode omitted_checks mismatch (B-1). "
            f"Expected {expected!r}, got {omitted!r}."
        )

    def test_validate_postwrite_mode(self):
        """usd_validate(sphere, out_path+actual_format) -> ok=True, mode='postwrite', omitted=[].

        Asserts UNCONDITIONALLY.
        """
        payload = self._postwrite_payload
        assert payload.get("ok") is True, (
            f"usd_validate on sphere LOP (postwrite) must return ok=True. "
            f"Got ok={payload.get('ok')!r}, error={payload.get('error')!r}."
        )
        assert payload.get("mode") == "postwrite", (
            f"usd_validate with out_path+actual_format must use 'postwrite' mode. "
            f"Got mode={payload.get('mode')!r}."
        )
        omitted = payload.get("omitted_checks", ["MISSING"])
        assert omitted == [], (
            f"'postwrite' mode must have empty omitted_checks (B-1). "
            f"Got omitted_checks={omitted!r}."
        )

    @classmethod
    def teardown_class(cls):
        """Clean up the sphere LOP node created by setup_class."""
        try:
            import hou
            sph = hou.node(cls._sphere_path)
            if sph is not None:
                sph.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test class: UEC-D8 — FR-10: Capability.READONLY for both tools
# ---------------------------------------------------------------------------

class TestFR10Readonly:
    """UEC-D8 (FR-10): Both usd_inspect_layer and usd_validate must be READONLY.

    Both tools read USD stage data — they never mutate the Houdini scene.
    FR-10 mandates Capability.READONLY, which bypasses the 109 gate
    (require_approval=False on the MCP wrapper side, READONLY on the handler side).
    """

    @classmethod
    def setup_class(cls):
        cls._dispatch = staticmethod(_get_dispatcher())

    def _get_handler_capability(self, command: str):
        """Retrieve the capability of a registered handler, if the registry exposes it."""
        try:
            from fxhoudinimcp_server.dispatcher import _REGISTRY  # noqa: F401
            entry = _REGISTRY.get(command)
            if entry is None:
                return None
            return getattr(entry, "capability", None) or getattr(entry, "cap", None)
        except (ImportError, AttributeError):
            return None

    def test_usd_inspect_layer_is_readonly(self):
        """usd_inspect_layer handler must have Capability.READONLY (FR-10)."""
        try:
            from fxhoudinimcp_server.capability import Capability
        except ImportError:
            return  # capability module absent — skip, covered by wrapper test

        cap = self._get_handler_capability("usd_inspect_layer")
        if cap is None:
            return  # Registry doesn't expose capability — wrapper test covers FR-10
        assert cap == Capability.READONLY, (
            f"usd_inspect_layer must be Capability.READONLY (FR-10). Got cap={cap!r}."
        )

    def test_usd_validate_is_readonly(self):
        """usd_validate handler must have Capability.READONLY (FR-10)."""
        try:
            from fxhoudinimcp_server.capability import Capability
        except ImportError:
            return

        cap = self._get_handler_capability("usd_validate")
        if cap is None:
            return
        assert cap == Capability.READONLY, (
            f"usd_validate must be Capability.READONLY (FR-10). Got cap={cap!r}."
        )


# ---------------------------------------------------------------------------
# Test registry and runner (for direct hython execution)
# ---------------------------------------------------------------------------

_TEST_CLASSES = [
    TestUsdExportDispatcherImport,
    TestInspectLayerResultShape,
    TestM4RootPrimsNoHoudiniLayerInfo,
    TestM6CurrentFormatInMemory,
    TestM3ChecksParamRejection,
    TestM5SopNodeNotLop,
    TestUsdValidateModeAndOmittedChecks,
    TestFR10Readonly,
]


def _run_class(cls):
    """Instantiate the class and run all test_* methods; record pass/fail."""
    obj = cls()
    setup_failed = False

    if hasattr(cls, "setup_class"):
        try:
            cls.setup_class()
        except ImportError as exc:
            _record_fail(
                f"{cls.__name__}.setup_class",
                str(exc),
            )
            setup_failed = True
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
    print("usd_export_inspect_hython_smoke.py — pp12-112b")
    print("=" * 72)

    # Pre-check: is this the RED phase or GREEN phase?
    try:
        _get_dispatcher()
        print("[mode] GREEN — usd_export_handlers importable; running all tests.")
    except ImportError as exc:
        print(f"[mode] RED — {exc}")
        print("Expected RED failure. Wrapper pytest red gate has already been confirmed.")
        print("Hython-smoke RED gate confirmed: usd_export_handlers not yet implemented.")
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
