"""Hython-smoke tests for the mtlx_inspect / mtlx_edit dispatcher path
(PP12-112 PR-5, pp12-112e).

Unit: pp12-112e
testVerificationSurface: hython-smoke
planSha: f1babc2821e387aade3fa83df71973c9773ece3e94e974b22ddbc697db823ae3

Run under hython (Houdini's headless Python interpreter, which ships the
MaterialX Python module):
    hython tests/mtlx_hython_smoke.py

Two-mode guard (mirrors usd_export_rop_hython_smoke.py, pp12-112d):
  - If fxhoudinimcp_server.handlers.usd_export_handlers does not yet register
    'mtlx_inspect'/'mtlx_edit' (hou-dev has not implemented them), dispatch()
    returns the dispatcher's own {"status": "error", "error": {"code":
    "UNKNOWN_COMMAND", ...}} shape -- this IS the expected RED signal for
    this file (the dispatcher module itself already exists from
    pp12-112b/c/d; only the NEW commands are red).
  - Once hou-dev implements the handlers + preview_fn + registration, the
    tests run and PASS GREEN.

Dispatcher-based (NOT direct handler calls) -- exercises the REAL
handler(**params) calling convention:
  - Importing usd_export_handlers triggers register_handler('mtlx_inspect', ...)
    and register_handler('mtlx_edit', ...) as side effects (alongside the
    pp12-112b/c/d registrations already shipped).
  - Tests call dispatch('mtlx_inspect', {...}) / dispatch('mtlx_edit', {...})
    -- the real dispatcher path, matching every other PP12 hython smoke.

STATICMETHOD HARNESS FIX (112c/d lesson, mandatory here too): dispatcher
functions assigned as class attrs in setup_class MUST be wrapped with
staticmethod(), otherwise `self` is injected as an extra positional arg when
called as `self._dispatch(...)` from an instance method, breaking the call
signature. See setup_class in every test class below.

Fixture: a small HAND-AUTHORED, VALID MaterialX 1.39-shaped .mtlx document
written to a real temp file at test time (NOT exported from a Houdini
Material Library LOP -- authoring one directly keeps the fixture
deterministic and independent of any live-scene setup). It contains:
  - a nodegraph "NG_main" holding a "standard_surface1" node whose
    getType() == 'surfaceshader' (the surface-node filter target) and a
    "basecolor_file" input of type "filename" with an ABSOLUTE path value
    (the inputs_with_abs_paths filter target);
  - a SECOND nodegraph "NG_alt" holding a node ALSO literally named
    "standard_surface1" (deliberately duplicated) so an unqualified lookup
    for "standard_surface1" is genuinely ambiguous across the two graphs --
    exercising the plan-2/new-1 ambiguity-fail-loud contract end-to-end
    against a REAL MaterialX document (not just the mock-hou rung).

Acceptance tests covered (plan rev3 decomposition[hou-test].acceptanceTests):
  - inspect a real .mtlx -> nodegraphs/surface_nodes/inputs_with_abs_paths/
    validate populated correctly.
  - edit an EXISTING input's value on the qualified NG_main/standard_surface1
    path -> re-read confirms the change landed + validate ok + NO
    text-substitution (setValueString via the real MaterialX API).
  - capability_of('mtlx_edit') == Capability.MUTATING with a registered
    preview_fn + preview_required=True; capability_of('mtlx_inspect') ==
    Capability.READONLY.
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap -- same pattern as usd_export_rop_hython_smoke.py
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
# Dispatcher import (usd_export_handlers module already exists from
# pp12-112b/c/d; 'mtlx_inspect'/'mtlx_edit' registration is the NEW red-gate
# surface for this file)
# ---------------------------------------------------------------------------
def _get_dispatcher_and_registry():
    """Import usd_export_handlers (triggers handler registration) and return
    (dispatch, capability_of, preview_of).

    The red signal for THIS file is the 'mtlx_inspect'/'mtlx_edit' COMMANDs
    being unregistered, which dispatch() surfaces as {"status": "error",
    "error": {"code": "UNKNOWN_COMMAND"}} rather than an ImportError. Tests
    below assert on that dispatch-level signal directly (see
    TestMtlxRedGate).
    """
    import fxhoudinimcp_server.handlers.usd_export_handlers as _handlers  # noqa: F401
    from fxhoudinimcp_server.dispatcher import dispatch, capability_of, preview_of
    return dispatch, capability_of, preview_of


# ---------------------------------------------------------------------------
# Real .mtlx fixture -- hand-authored, valid MaterialX 1.39-shaped XML,
# written to a real temp file. Two nodegraphs, each holding a node literally
# named "standard_surface1" (the ambiguity fixture), plus a filename input
# with an absolute path on the NG_main copy.
# ---------------------------------------------------------------------------

_ABS_TEXTURE_PATH = (
    "C:/textures/basecolor.tex" if os.name == "nt" else "/textures/basecolor.tex"
)

_FIXTURE_MTLX_XML = f"""<?xml version="1.0"?>
<materialx version="1.39">
  <nodegraph name="NG_main">
    <standard_surface name="standard_surface1" type="surfaceshader">
      <input name="base_color" type="color3" value="1.0, 1.0, 1.0" />
      <input name="basecolor_file" type="filename" value="{_ABS_TEXTURE_PATH}" />
    </standard_surface>
  </nodegraph>
  <nodegraph name="NG_alt">
    <standard_surface name="standard_surface1" type="surfaceshader">
      <input name="base_color" type="color3" value="0.2, 0.2, 0.2" />
    </standard_surface>
  </nodegraph>
</materialx>
"""


def _write_fixture_mtlx(path: str) -> str:
    """Write the hand-authored fixture .mtlx to *path* (a real file on disk).

    Returns the path written (identity passthrough for chaining).
    """
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_FIXTURE_MTLX_XML)
    return path


def _hip_fixture_path(name: str) -> str:
    """Resolve a $HIP-relative fixture path via hou.text.expandString when
    hou is available (hython), else a bare temp-dir path (best-effort for
    non-hython smoke-shim runs)."""
    try:
        import hou
        return hou.text.expandString(f"$HIP/{name}")
    except ImportError:
        import tempfile
        return os.path.join(tempfile.gettempdir(), name)


def _remove_if_exists(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tiny result helpers (mirroring usd_export_rop_hython_smoke.py)
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
# Test class: MHS-1 -- RED gate: 'mtlx_inspect'/'mtlx_edit' must be
#             registered commands
# ---------------------------------------------------------------------------

class TestMtlxRedGate:
    """MHS-1 (RED GATE): 'mtlx_inspect' and 'mtlx_edit' must be registered in
    the dispatcher.

    Before hou-dev implements the handlers, dispatch('mtlx_inspect'/
    'mtlx_edit', {...}) returns {"status": "error", "error": {"code":
    "UNKNOWN_COMMAND", ...}} -- this is the expected RED failure for this
    file. Once hou-dev registers the handlers via register_handler(...),
    these commands resolve and the remaining test classes exercise real
    behavior.
    """

    @classmethod
    def setup_class(cls):
        _dispatch, _capability_of, _preview_of = _get_dispatcher_and_registry()
        cls._dispatch = staticmethod(_dispatch)
        cls._capability_of = staticmethod(_capability_of)
        cls._preview_of = staticmethod(_preview_of)

    def test_mtlx_inspect_is_registered(self):
        """dispatch('mtlx_inspect', ...) must NOT return UNKNOWN_COMMAND.

        FAILS RED (as intended) until hou-dev calls
        register_handler('mtlx_inspect', mtlx_inspect, Capability.READONLY).
        """
        result = self._dispatch("mtlx_inspect", {"mtlx_path_or_doc": "$HIP/probe.mtlx"})
        assert not _is_unknown_command(result), (
            "'mtlx_inspect' not found in the dispatcher registry. hou-dev must call "
            "register_handler('mtlx_inspect', mtlx_inspect, Capability.READONLY) in "
            "usd_export_handlers.py. "
            f"Got dispatch result: {result!r}."
        )

    def test_mtlx_edit_is_registered(self):
        """dispatch('mtlx_edit', ...) must NOT return UNKNOWN_COMMAND.

        FAILS RED (as intended) until hou-dev calls
        register_handler('mtlx_edit', mtlx_edit, Capability.MUTATING,
                          preview_fn=_preview_mtlx_edit, preview_required=True).
        """
        result = self._dispatch("mtlx_edit", {
            "mtlx_path": "$HIP/probe.mtlx",
            "out_path": "$HIP/probe_out.mtlx",
            "edits": [{"node": "n1", "input": "i1", "value": "v1"}],
        })
        assert not _is_unknown_command(result), (
            "'mtlx_edit' not found in the dispatcher registry. hou-dev must call "
            "register_handler('mtlx_edit', mtlx_edit, Capability.MUTATING, "
            "preview_fn=_preview_mtlx_edit, preview_required=True) in "
            "usd_export_handlers.py. "
            f"Got dispatch result: {result!r}."
        )


# ---------------------------------------------------------------------------
# Test class: MHS-2 -- mtlx_inspect on the real fixture: nodegraphs,
#             surface_nodes, inputs_with_abs_paths, validate populated
# ---------------------------------------------------------------------------

class TestMtlxInspectRealFixture:
    """MHS-2: inspecting the hand-authored fixture .mtlx must correctly
    populate nodegraphs, surface_nodes, inputs_with_abs_paths, and the
    validate block, via the REAL MaterialX Python API (mx.readFromXmlFile +
    doc.getNodeGraphs()/getNodes() + doc.validate()) -- not a stub."""

    @classmethod
    def setup_class(cls):
        _dispatch, _capability_of, _preview_of = _get_dispatcher_and_registry()
        cls._dispatch = staticmethod(_dispatch)
        cls._capability_of = staticmethod(_capability_of)
        cls._preview_of = staticmethod(_preview_of)

        cls._fixture_expr = "$HIP/pp12_112e_inspect_fixture.mtlx"
        cls._fixture_expanded = _hip_fixture_path("pp12_112e_inspect_fixture.mtlx")
        _write_fixture_mtlx(cls._fixture_expanded)

        raw = cls._dispatch("mtlx_inspect", {"mtlx_path_or_doc": cls._fixture_expr})
        cls._payload = _unwrap(raw)

    def test_inspect_reports_ok_true(self):
        assert self._payload.get("ok") is True, (
            f"mtlx_inspect on a valid fixture .mtlx must return ok=True. "
            f"Got ok={self._payload.get('ok')!r}, error={self._payload.get('error')!r}."
        )

    def test_nodegraphs_include_both_fixture_graphs(self):
        nodegraphs = self._payload.get("nodegraphs")
        assert isinstance(nodegraphs, list), (
            f"nodegraphs must be a list. Got {type(nodegraphs)!r}: {nodegraphs!r}."
        )
        assert set(nodegraphs) >= {"NG_main", "NG_alt"}, (
            f"nodegraphs must include both fixture graphs 'NG_main' and 'NG_alt'. "
            f"Got nodegraphs={nodegraphs!r}."
        )

    def test_surface_nodes_include_fixture_surface_node(self):
        """The surface-node filter (getType()=='surfaceshader') must find the
        standard_surface1 node(s) -- both duplicated instances across the
        two nodegraphs."""
        surface_nodes = self._payload.get("surface_nodes")
        assert isinstance(surface_nodes, list), (
            f"surface_nodes must be a list. Got {type(surface_nodes)!r}: {surface_nodes!r}."
        )
        assert "standard_surface1" in surface_nodes, (
            f"surface_nodes must include 'standard_surface1' (getType()== "
            f"'surfaceshader' on the fixture's standard_surface elements). "
            f"Got surface_nodes={surface_nodes!r}."
        )

    def test_inputs_with_abs_paths_includes_fixture_filename_input(self):
        """The filename-input absolute-path filter must find
        'basecolor_file' (type='filename', value is an absolute path)."""
        inputs_with_abs_paths = self._payload.get("inputs_with_abs_paths")
        assert isinstance(inputs_with_abs_paths, list), (
            f"inputs_with_abs_paths must be a list. Got "
            f"{type(inputs_with_abs_paths)!r}: {inputs_with_abs_paths!r}."
        )
        assert "basecolor_file" in inputs_with_abs_paths, (
            f"inputs_with_abs_paths must include 'basecolor_file' (a "
            f"filename-typed input whose value {_ABS_TEXTURE_PATH!r} is an "
            f"absolute path). Got inputs_with_abs_paths={inputs_with_abs_paths!r}."
        )

    def test_validate_block_present_and_ok(self):
        validate = self._payload.get("validate")
        assert isinstance(validate, dict), (
            f"validate must be a dict (nested under the 'validate' key per "
            f"MtlxSummary.to_dict()). Got {type(validate)!r}: {validate!r}."
        )
        assert "ok" in validate and "errors" in validate, (
            f"validate must carry 'ok' and 'errors' keys. Got keys="
            f"{list(validate.keys())!r}."
        )
        assert validate.get("ok") is True, (
            f"A well-formed fixture document must validate ok=True. "
            f"Got validate={validate!r}."
        )

    @classmethod
    def teardown_class(cls):
        _remove_if_exists(cls._fixture_expanded)


# ---------------------------------------------------------------------------
# Test class: MHS-3 -- mtlx_edit on the real fixture: qualified-path edit,
#             re-read confirms the change, validate ok, no text-substitution
# ---------------------------------------------------------------------------

class TestMtlxEditRealFixture:
    """MHS-3: editing an EXISTING input's value via the qualified
    'NG_main/standard_surface1' path must write a new .mtlx file whose
    re-read (via a fresh mtlx_inspect-equivalent re-parse) confirms the
    requested value landed EXACTLY -- via mx.Input.setValueString (the real
    MaterialX API), never a text/regex substitution on the raw XML."""

    @classmethod
    def setup_class(cls):
        _dispatch, _capability_of, _preview_of = _get_dispatcher_and_registry()
        cls._dispatch = staticmethod(_dispatch)
        cls._capability_of = staticmethod(_capability_of)
        cls._preview_of = staticmethod(_preview_of)

        cls._source_expr = "$HIP/pp12_112e_edit_source.mtlx"
        cls._source_expanded = _hip_fixture_path("pp12_112e_edit_source.mtlx")
        _write_fixture_mtlx(cls._source_expanded)

        cls._out_expr = "$HIP/pp12_112e_edit_out.mtlx"
        cls._out_expanded = _hip_fixture_path("pp12_112e_edit_out.mtlx")

        cls._requested_value = "0.55, 0.55, 0.55"

        raw = cls._dispatch("mtlx_edit", {
            "mtlx_path": cls._source_expr,
            "out_path": cls._out_expr,
            "edits": [{
                "node": "NG_main/standard_surface1",
                "input": "base_color",
                "value": cls._requested_value,
            }],
        })
        cls._payload = _unwrap(raw)

    def test_edit_reports_ok_true(self):
        assert self._payload.get("ok") is True, (
            f"Editing an existing input via the qualified path on a valid "
            f"fixture must return ok=True. Got ok={self._payload.get('ok')!r}, "
            f"error={self._payload.get('error')!r}."
        )

    def test_out_file_written(self):
        assert os.path.exists(self._out_expanded), (
            f"Expected a written .mtlx file at {self._out_expanded!r} but it "
            f"does not exist."
        )

    def test_edits_applied_count(self):
        assert self._payload.get("edits_applied") == 1, (
            f"edits_applied must report 1 (a single edit was requested). "
            f"Got {self._payload.get('edits_applied')!r}."
        )

    def test_written_value_matches_requested_exactly_no_regex(self):
        """Re-read the WRITTEN file via a fresh mtlx_inspect dispatch call
        and confirm the edited input's value round-trips EXACTLY. Also
        confirms the file's raw text does NOT contain the ORIGINAL value
        (proving a real mutation happened, not a no-op) while genuinely
        containing valid re-parseable MaterialX XML (proving the write went
        through the real MaterialX API, not a corrupting text substitution).
        """
        # Re-parse via the real handler path (mtlx_inspect on the written
        # file) rather than opening the XML text with a regex -- this
        # exercises the SAME MaterialX API surface end-to-end.
        raw = self._dispatch("mtlx_inspect", {"mtlx_path_or_doc": self._out_expr})
        payload = _unwrap(raw)
        assert payload.get("ok") is True, (
            f"The written .mtlx file must itself be a valid, re-parseable "
            f"MaterialX document. Got payload={payload!r}."
        )
        assert payload.get("validate", {}).get("ok") is True, (
            f"The written .mtlx file must validate ok=True after the edit. "
            f"Got validate={payload.get('validate')!r}."
        )

        # Direct API re-read of the specific edited value (not just overall
        # document validity) via a second MaterialX parse.
        import MaterialX as mx  # noqa: local import -- hython-only surface

        doc = mx.createDocument()
        mx.readFromXmlFile(doc, self._out_expanded)
        ng = doc.getNodeGraph("NG_main")
        assert ng is not None, (
            "NG_main nodegraph must survive the edit+write round-trip."
        )
        node = ng.getNode("standard_surface1")
        assert node is not None, (
            "standard_surface1 node in NG_main must survive the edit+write round-trip."
        )
        edited_input = node.getInput("base_color")
        assert edited_input is not None, (
            "base_color input must survive the edit+write round-trip."
        )
        assert edited_input.getValueString() == self._requested_value, (
            f"The edited input's re-read value must EXACTLY match the "
            f"requested value (string==string, no regex/text-substitution "
            f"drift). Expected {self._requested_value!r}, got "
            f"{edited_input.getValueString()!r}."
        )

        # NG_alt's UNRELATED duplicate node must be untouched (proves the
        # qualified-path resolution edited the correct instance, not both).
        ng_alt = doc.getNodeGraph("NG_alt")
        assert ng_alt is not None, "NG_alt nodegraph must survive the edit+write round-trip."
        alt_node = ng_alt.getNode("standard_surface1")
        assert alt_node is not None, "NG_alt's standard_surface1 must survive the edit+write round-trip."
        alt_input = alt_node.getInput("base_color")
        assert alt_input is not None, "NG_alt's base_color input must survive the edit+write round-trip."
        assert alt_input.getValueString() != self._requested_value, (
            f"NG_alt's standard_surface1.base_color must remain UNTOUCHED by "
            f"an edit qualified to NG_main/standard_surface1 -- the qualified "
            f"path must resolve to exactly one node. Got "
            f"{alt_input.getValueString()!r} (must differ from the requested "
            f"value {self._requested_value!r})."
        )

    @classmethod
    def teardown_class(cls):
        _remove_if_exists(cls._source_expanded)
        _remove_if_exists(cls._out_expanded)


# ---------------------------------------------------------------------------
# Test class: MHS-4 -- ambiguity fail-loud against the REAL fixture's
#             duplicate node name (unqualified 'standard_surface1')
# ---------------------------------------------------------------------------

class TestMtlxEditAmbiguityRealFixture:
    """MHS-4: an UNQUALIFIED edit targeting 'standard_surface1' (present in
    BOTH NG_main and NG_alt in the fixture) must be rejected as ambiguous --
    against the REAL MaterialX-parsed document, not a mock. Proves the
    ambiguity-fail-loud contract holds end-to-end through the dispatcher."""

    @classmethod
    def setup_class(cls):
        _dispatch, _capability_of, _preview_of = _get_dispatcher_and_registry()
        cls._dispatch = staticmethod(_dispatch)
        cls._capability_of = staticmethod(_capability_of)
        cls._preview_of = staticmethod(_preview_of)

        cls._source_expr = "$HIP/pp12_112e_ambig_source.mtlx"
        cls._source_expanded = _hip_fixture_path("pp12_112e_ambig_source.mtlx")
        _write_fixture_mtlx(cls._source_expanded)

        cls._out_expr = "$HIP/pp12_112e_ambig_out.mtlx"
        cls._out_expanded = _hip_fixture_path("pp12_112e_ambig_out.mtlx")

        raw = cls._dispatch("mtlx_edit", {
            "mtlx_path": cls._source_expr,
            "out_path": cls._out_expr,
            "edits": [{
                "node": "standard_surface1",  # UNQUALIFIED -- ambiguous
                "input": "base_color",
                "value": "0.9, 0.9, 0.9",
            }],
        })
        cls._payload = _unwrap(raw)

    def test_unqualified_ambiguous_edit_rejected(self):
        assert self._payload.get("ok") is False, (
            f"An unqualified edit target present in TWO nodegraphs of the "
            f"REAL fixture document must be rejected as ambiguous. "
            f"Got payload={self._payload!r}."
        )
        error_text = str(self._payload.get("error", "")).lower()
        assert "ambiguous" in error_text, (
            f"The rejection error must name the ambiguity explicitly. "
            f"Got error={self._payload.get('error')!r}."
        )

    def test_no_out_file_written_on_ambiguity_reject(self):
        assert not os.path.exists(self._out_expanded), (
            f"An ambiguity-rejected edit must NOT write any output file at "
            f"{self._out_expanded!r}."
        )

    @classmethod
    def teardown_class(cls):
        _remove_if_exists(cls._source_expanded)
        _remove_if_exists(cls._out_expanded)


# ---------------------------------------------------------------------------
# Test class: MHS-5 -- Capability registration: mtlx_inspect READONLY;
#             mtlx_edit MUTATING with a registered preview_fn (109 gate)
# ---------------------------------------------------------------------------

class TestMtlxCapabilityRegistration:
    """MHS-5: 'mtlx_inspect' must be Capability.READONLY (UNGATED);
    'mtlx_edit' must be Capability.MUTATING WITH a preview_fn
    (preview_required=True) -- mirroring usd_export_layer/usd_export_rop's
    registration exactly.
    """

    @classmethod
    def setup_class(cls):
        _dispatch, _capability_of, _preview_of = _get_dispatcher_and_registry()
        cls._dispatch = staticmethod(_dispatch)
        cls._capability_of = staticmethod(_capability_of)
        cls._preview_of = staticmethod(_preview_of)

    def test_mtlx_inspect_capability_is_readonly(self):
        from fxhoudinimcp_server.dispatcher import Capability

        cap = self._capability_of("mtlx_inspect")
        assert cap == Capability.READONLY, (
            f"'mtlx_inspect' must be registered with Capability.READONLY "
            f"(pure introspection, no scene/file mutation). Got cap={cap!r}."
        )

    def test_mtlx_edit_capability_is_mutating(self):
        from fxhoudinimcp_server.dispatcher import Capability

        cap = self._capability_of("mtlx_edit")
        assert cap == Capability.MUTATING, (
            f"'mtlx_edit' must be registered with Capability.MUTATING "
            f"(it writes a MaterialX document to disk). Got cap={cap!r}."
        )

    def test_mtlx_edit_preview_fn_is_registered_and_required(self):
        """preview_of('mtlx_edit') must have a non-None preview_fn with
        preview_required=True (mirrors usd_export_rop's registration)."""
        preview_record = self._preview_of("mtlx_edit")
        assert preview_record.get("preview_fn") is not None, (
            "'mtlx_edit' must register a preview_fn (_preview_mtlx_edit) so "
            "the 109 gate can show a pre-flight preview before approval. "
            f"Got preview_record={preview_record!r}."
        )
        assert preview_record.get("preview_required") is True, (
            f"'mtlx_edit' must set preview_required=True (a preview_fn "
            f"failure must DENY the call, not silently queue without a "
            f"preview). Got preview_required="
            f"{preview_record.get('preview_required')!r}."
        )

    def test_mtlx_inspect_has_no_preview_registration(self):
        """mtlx_inspect is READONLY/UNGATED -- it should carry no preview_fn
        (the 109 gate does not apply to ungated readonly commands)."""
        preview_record = self._preview_of("mtlx_inspect")
        assert preview_record.get("preview_fn") is None, (
            f"'mtlx_inspect' (READONLY) should not register a preview_fn. "
            f"Got preview_record={preview_record!r}."
        )


# ---------------------------------------------------------------------------
# Test class: MHS-6 -- no regression: existing usd_export family tools
#             (usd_inspect_layer, usd_validate, usd_export_layer,
#             usd_export_rop) stay unchanged after the mtlx sibling pair
# ---------------------------------------------------------------------------

class TestExistingToolsUnchanged:
    """MHS-6: usd_inspect_layer/usd_validate (READONLY), usd_export_layer
    and usd_export_rop (MUTATING, both with a preview_fn) must remain
    unchanged after the mtlx_inspect/mtlx_edit sibling pair is added. A
    regression here would mean the new registrations accidentally clobbered
    or altered an existing registration.
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

    def test_usd_export_layer_still_mutating_with_preview(self):
        from fxhoudinimcp_server.dispatcher import Capability

        cap = self._capability_of("usd_export_layer")
        assert cap == Capability.MUTATING, (
            f"'usd_export_layer' (pp12-112c) must remain Capability.MUTATING. Got cap={cap!r}."
        )
        preview_record = self._preview_of("usd_export_layer")
        assert preview_record.get("preview_fn") is not None, (
            "'usd_export_layer' (pp12-112c) must retain its registered preview_fn "
            "after the mtlx sibling pair is added."
        )

    def test_usd_export_rop_still_mutating_with_preview(self):
        from fxhoudinimcp_server.dispatcher import Capability

        cap = self._capability_of("usd_export_rop")
        assert cap == Capability.MUTATING, (
            f"'usd_export_rop' (pp12-112d) must remain Capability.MUTATING. Got cap={cap!r}."
        )
        preview_record = self._preview_of("usd_export_rop")
        assert preview_record.get("preview_fn") is not None, (
            "'usd_export_rop' (pp12-112d) must retain its registered preview_fn "
            "after the mtlx sibling pair is added."
        )


# ---------------------------------------------------------------------------
# Test registry and runner (for direct hython execution)
# ---------------------------------------------------------------------------

_TEST_CLASSES = [
    TestMtlxRedGate,
    TestMtlxInspectRealFixture,
    TestMtlxEditRealFixture,
    TestMtlxEditAmbiguityRealFixture,
    TestMtlxCapabilityRegistration,
    TestExistingToolsUnchanged,
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
    print("mtlx_hython_smoke.py -- pp12-112e")
    print("=" * 72)

    try:
        dispatch, capability_of, preview_of = _get_dispatcher_and_registry()
    except ImportError as exc:
        print(f"[mode] RED -- usd_export_handlers module itself is not importable: {exc}")
        print("Expected only if pp12-112b/c/d's module is somehow absent. "
              "For pp12-112e's own red gate, the module import succeeds but "
              "'mtlx_inspect'/'mtlx_edit' are UNKNOWN_COMMAND -- see TestMtlxRedGate.")
        return 1

    probe = dispatch("mtlx_inspect", {"mtlx_path_or_doc": "$HIP/probe.mtlx"})
    if _is_unknown_command(probe):
        print("[mode] RED -- 'mtlx_inspect'/'mtlx_edit' not yet registered (expected before hou-dev).")
        print("Running TestMtlxRedGate only (the other classes require the real handlers).")
        _run_class(TestMtlxRedGate)
        print("\n" + "=" * 72)
        print(f"Results: {_PASS_COUNT} passed, {_FAIL_COUNT} failed")
        return 1  # RED gate confirmed -- exit non-zero

    print("[mode] GREEN -- 'mtlx_inspect'/'mtlx_edit' registered; running all tests.")
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
