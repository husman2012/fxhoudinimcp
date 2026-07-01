"""Handler-level mock-hou/mock-MaterialX pytest for mtlx_inspect / mtlx_edit's
edge branches (PP12-112 PR-5, pp12-112e) -- the REQUIRED rev3 fold red
assertions that a real hython-driven .mtlx fixture cannot cheaply force
(ambiguity resolution, duplicate-target rejection, round-trip verification,
preview parse-catch, string-only value guard, HAS_MTLX-absent fail-loud, and
the shared edit-shape guard).

Unit: pp12-112e
testVerificationSurface: pytest-model (mock-hou/mock-MaterialX rung, per
    test-fixture-conventions.md section 2.3 -- these edge branches cannot be
    reliably forced through a real MaterialX document parse/write cycle
    without extremely fragile fixture construction (e.g. forcing "the
    round-trip re-read observes a DIFFERENT value than what was written"),
    so they are pinned here rather than in the hython-smoke file)
planSha: f1babc2821e387aade3fa83df71973c9773ece3e94e974b22ddbc697db823ae3

Why a mock-hou/mock-MaterialX file, not hython-smoke, for these branches
---------------------------------------------------------------------------
The plan rev3 lockedFieldContract identifies FOUR round-2-fold Major findings
(new-1 unsafe qualified-path resolution; new-2 round-trip re-resolves by
ambiguous name; new-3 numeric round-trip false-fail; new-4 duplicate edit
targets) plus the round-1 folds (plan-1 preview-catches-parse-fail; plan-2
_resolve_node fail-loud-on-ambiguity; plan-6 scalar/string value guard;
plan-7 shared shape guard) that require PRECISE, deterministic control over:
  - a document containing TWO nodes with the SAME unqualified name in
    different nodegraphs (to force the ambiguous-name branch);
  - two edits that resolve to the SAME (node, input) target (to force the
    duplicate-target reject);
  - a value that is genuinely NOT a string (int/float/bool/None/list/dict)
    to force the plan-6/new-3 string-only guard;
  - MaterialX being genuinely unavailable (HAS_MTLX=False) to force the
    fail-loud R-4 path;
  - a source that raises on parse to force the preview's OWN try/except
    (plan-1), independent of whatever the gate's raise->DENY mechanism does.

Per test-fixture-conventions.md section 2.3, a mocked `MaterialX` module
(configured MagicMock, NOT bare) is installed into sys.modules alongside a
mocked `hou` and `pxr` (usd_export_handlers.py already imports pxr/hou at
module scope; the MaterialX import must be import-guarded the same way pxr
is -- HAS_PXR/HAS_MTLX). The module-under-test is imported INSIDE each test,
AFTER the mocks are installed (test-fixture-conventions.md section 2.3's
"import inside the test" rule).

This file also carries the HAS_MTLX-absent fail-loud test (MaterialX import
genuinely fails -- sys.modules entry removed / import raises) and the
shared edit-shape guard tests (non-list edits, empty edits, missing
node/input/value keys).
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# hou / pxr / MaterialX mock installation
# ---------------------------------------------------------------------------

class _FakeOperationFailed(Exception):
    """A REAL Exception subclass standing in for hou.OperationFailed.

    test-fixture-conventions.md section 2.3: a bare MagicMock() for
    hou.OperationFailed would not behave as a raisable/catchable exception
    class -- it must be a real Exception subclass so `raise hou.OperationFailed(...)`
    and `except hou.OperationFailed:` work exactly as they do against the
    real hou module.
    """


def _install_hou_pxr_mocks(monkeypatch):
    """Install configured mocks for `hou` and `pxr` into sys.modules.

    Mirrors test_usd_export_rop_handler_edge_cases.py's mock_hou_and_pxr
    fixture exactly (same repo-root sys.path bootstrap, same
    hou.text.expandString identity passthrough default).
    """
    import os

    hou_mock = MagicMock(name="hou")
    hou_mock.OperationFailed = _FakeOperationFailed
    hou_mock.text = MagicMock(name="hou.text")
    hou_mock.text.expandString = MagicMock(side_effect=lambda s: s)

    pxr_mock = MagicMock(name="pxr")
    pxr_mock.Usd = MagicMock(name="pxr.Usd")
    pxr_mock.UsdShade = MagicMock(name="pxr.UsdShade")

    monkeypatch.setitem(sys.modules, "hou", hou_mock)
    monkeypatch.setitem(sys.modules, "pxr", pxr_mock)

    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    _pkg_python = os.path.join(_repo_root, "python")
    _houdini_handlers = os.path.join(_repo_root, "houdini", "scripts", "python")
    for _p in (_pkg_python, _houdini_handlers):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    return hou_mock, pxr_mock


class _FakeMxInput:
    """A configurable fake MaterialX Input element."""

    def __init__(self, name, value="1.0", value_type="float"):
        self._name = name
        self._value = value
        self._type = value_type

    def getName(self):
        return self._name

    def getValueString(self):
        return self._value

    def getType(self):
        return self._type

    def setValueString(self, value):
        self._value = value


class _FakeMxNode:
    """A configurable fake MaterialX Node element."""

    def __init__(self, name, category="standard_surface", node_type="surfaceshader",
                 inputs=None, name_path=None):
        self._name = name
        self._category = category
        self._type = node_type
        self._inputs = inputs or {}
        self._name_path = name_path or name

    def getName(self):
        return self._name

    def getCategory(self):
        return self._category

    def getType(self):
        return self._type

    def getInputs(self):
        return list(self._inputs.values())

    def getInput(self, name):
        return self._inputs.get(name)

    def getNamePath(self):
        return self._name_path


class _FakeMxNodeGraph:
    """A configurable fake MaterialX NodeGraph element."""

    def __init__(self, name, nodes=None):
        self._name = name
        self._nodes = nodes or {}

    def getName(self):
        return self._name

    def getNodes(self):
        return list(self._nodes.values())

    def getNode(self, name):
        return self._nodes.get(name)


class _FakeMxDocument:
    """A configurable fake MaterialX Document."""

    def __init__(self, nodegraphs=None, top_nodes=None, validate_result=(True, "")):
        self._nodegraphs = nodegraphs or {}
        self._top_nodes = top_nodes or {}
        self._validate_result = validate_result

    def getNodeGraphs(self):
        return list(self._nodegraphs.values())

    def getNodeGraph(self, name):
        return self._nodegraphs.get(name)

    def getNodes(self):
        return list(self._top_nodes.values())

    def getNode(self, name):
        return self._top_nodes.get(name)

    def validate(self):
        return self._validate_result


def _install_materialx_mock(monkeypatch, doc_factory=None, read_side_effect=None):
    """Install a configured mock MaterialX module into sys.modules.

    Args:
        doc_factory: a zero-arg callable returning a fresh _FakeMxDocument
            each time createDocument() is called (so repeated calls -- e.g.
            the handler's post-write round-trip re-read -- get independently
            mutable state when needed). Defaults to an empty document.
        read_side_effect: optional side_effect for readFromXmlFile (e.g. to
            raise on a specific path to simulate a parse failure).

    Returns the configured mock module.
    """
    # red-10-round2 fix (Minor): bound the mock's attribute surface to the
    # plan's grounded MaterialX API list via spec=[...] (the list form of
    # `spec`, which MagicMock supports even without a real class to
    # introspect -- it does not require `MaterialX` to be genuinely
    # importable in this environment). With spec set, ANY attribute access
    # not in this list raises AttributeError instead of being silently
    # auto-vivified as a magically-successful callable -- so a typo'd or
    # non-existent MaterialX API call in the implementation (e.g. a
    # misspelled `mx.creatDocument`) now fails loudly in this mock, rather
    # than being silently fabricated. This does not require introspecting
    # the real pybind11 C-extension class (which TestMaterialxMockHas
    # DocumentedMinimalSurface's docstring correctly notes is not available
    # here) -- spec=[<name list>] is a documented MagicMock feature that
    # works from a plain list of attribute names.
    _MTLX_DOCUMENTED_SURFACE = ["createDocument", "readFromXmlFile", "writeToXmlFile", "getVersionString"]
    mx_mock = MagicMock(name="MaterialX", spec=_MTLX_DOCUMENTED_SURFACE)
    mx_mock.getVersionString = MagicMock(return_value="1.39.0")

    if doc_factory is None:
        doc_factory = lambda: _FakeMxDocument()  # noqa: E731

    mx_mock.createDocument = MagicMock(side_effect=lambda: doc_factory())

    if read_side_effect is not None:
        mx_mock.readFromXmlFile = MagicMock(side_effect=read_side_effect)
    else:
        mx_mock.readFromXmlFile = MagicMock(return_value=None)

    mx_mock.writeToXmlFile = MagicMock(return_value=None)

    monkeypatch.setitem(sys.modules, "MaterialX", mx_mock)
    return mx_mock


def _fresh_import_usd_export_handlers():
    """Import (or re-import) usd_export_handlers fresh, after mocks are set."""
    mod_name = "fxhoudinimcp_server.handlers.usd_export_handlers"
    if mod_name in sys.modules:
        return importlib.reload(sys.modules[mod_name])
    return importlib.import_module(mod_name)


@pytest.fixture()
def mock_env(monkeypatch):
    """Install hou + pxr + a DEFAULT (empty-doc) MaterialX mock.

    Individual tests further configure MaterialX.createDocument's returned
    document via monkeypatching sys.modules['MaterialX'].createDocument, or
    by calling _install_materialx_mock again with a custom doc_factory.
    """
    hou_mock, pxr_mock = _install_hou_pxr_mocks(monkeypatch)
    mx_mock = _install_materialx_mock(monkeypatch)

    class _Namespace:
        hou = hou_mock
        pxr = pxr_mock
        mx = mx_mock

    return _Namespace()


# ---------------------------------------------------------------------------
# Sanity: module imports with MaterialX mocked -> HAS_MTLX True
# ---------------------------------------------------------------------------

class TestModuleImportsWithMockedMaterialX:
    """Sanity: usd_export_handlers must import cleanly with MaterialX mocked,
    and HAS_MTLX must become True (exercising the real code path, not the
    _require_mtlx() early-raise short-circuit)."""

    def test_module_imports_and_has_mtlx_true(self, mock_env):
        mod = _fresh_import_usd_export_handlers()
        assert hasattr(mod, "HAS_MTLX"), (
            "usd_export_handlers must define a HAS_MTLX guard flag (mirroring "
            "HAS_PXR), populated by a guarded `import MaterialX as mx` at "
            "module scope. FAILS RED until hou-dev implements it."
        )
        assert mod.HAS_MTLX is True, (
            "With MaterialX mocked into sys.modules, usd_export_handlers.HAS_MTLX "
            "must be True (exercising the real code path)."
        )


# ---------------------------------------------------------------------------
# R-4: HAS_MTLX-absent -> fail-loud, never a fabricated result
# ---------------------------------------------------------------------------

class TestHasMtlxAbsentFailLoud:
    """R-4 (REQUIRED): when MaterialX is unavailable, mtlx_inspect and
    mtlx_edit must both return {ok: False, error: ...} (fail-loud, M-1) --
    NEVER a fabricated result. The preview_fn must let the raise propagate
    (gate DENY).
    """

    def _install_mtlx_absent(self, monkeypatch):
        """Actively force `import MaterialX as mx` to raise ImportError,
        regardless of whether MaterialX happens to be pip-installed in the
        current venv.

        red-1 (REQUIRED fix): the prior version of this helper only removed
        any existing sys.modules['MaterialX'] mock and relied on MaterialX
        genuinely NOT being installed in the pytest venv -- that is an
        environment assumption, not an active guarantee. If MaterialX were
        ever pip-installed into the dev/test venv (e.g. a future dependency
        pull, a shared venv, a CI image change), `import MaterialX as mx`
        would SUCCEED and this whole HAS_MTLX-absent fail-loud test class
        would silently stop testing the fail-loud path -- while still
        reporting green, because the handler would just use the real module.

        Fix: install an explicit `sys.modules['MaterialX'] = None` sentinel.
        Per Python import-system semantics (PEP 328 / importlib docs), a
        `None` entry in sys.modules for a module name makes ANY subsequent
        `import MaterialX` (or `import MaterialX as mx`) raise ImportError
        immediately, regardless of whether a real MaterialX package is
        installed and importable on sys.path. This actively forces the
        ImportError branch rather than hoping the environment cooperates.
        """
        monkeypatch.setitem(sys.modules, "MaterialX", None)

    def test_mtlx_inspect_returns_ok_false_when_mtlx_absent(self, monkeypatch):
        self._install_hou_pxr(monkeypatch)
        self._install_mtlx_absent(monkeypatch)
        mod = _fresh_import_usd_export_handlers()

        if not hasattr(mod, "HAS_MTLX"):
            pytest.fail("usd_export_handlers.HAS_MTLX does not exist yet -- RED (expected).")
        assert mod.HAS_MTLX is False, (
            "Without MaterialX installed/mocked, HAS_MTLX must be False."
        )
        if not hasattr(mod, "mtlx_inspect"):
            pytest.fail("usd_export_handlers.mtlx_inspect does not exist yet -- RED (expected).")

        result = mod.mtlx_inspect(mtlx_path_or_doc="$HIP/material.mtlx")
        assert result.get("ok") is False, (
            f"mtlx_inspect must return ok=False when MaterialX is unavailable "
            f"(fail-loud, never fabricate a result). Got result={result!r}."
        )
        assert isinstance(result.get("error"), str) and result["error"], (
            f"mtlx_inspect must report a non-empty error string when MaterialX "
            f"is unavailable. Got error={result.get('error')!r}."
        )

    def test_mtlx_edit_returns_ok_false_when_mtlx_absent(self, monkeypatch):
        self._install_hou_pxr(monkeypatch)
        self._install_mtlx_absent(monkeypatch)
        mod = _fresh_import_usd_export_handlers()

        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path="$HIP/edited.mtlx",
            edits=[{"node": "n1", "input": "base_color", "value": "0.5"}],
        )
        assert result.get("ok") is False, (
            f"mtlx_edit must return ok=False when MaterialX is unavailable "
            f"(fail-loud, never fabricate a result). Got result={result!r}."
        )
        assert isinstance(result.get("error"), str) and result["error"], (
            f"mtlx_edit must report a non-empty error string when MaterialX "
            f"is unavailable. Got error={result.get('error')!r}."
        )

    def test_preview_mtlx_edit_raises_when_mtlx_absent(self, monkeypatch):
        """The preview_fn must let the HAS_MTLX-absent raise propagate so the
        gate DENIES the call -- never silently proceed.

        red-9-round2 fix (Minor, REQUIRED): narrowed from an unnarrowed
        `pytest.raises(Exception)` (which any exception from any cause in
        the preview body would satisfy -- including an unrelated bug) to the
        SPECIFIC exception type and message this unit's own established
        convention uses. usd_export_handlers.py's existing `_require_pxr()`
        (the sibling M-1 fail-loud guard this unit's docstring says
        `_require_mtlx()` mirrors) raises `hou.OperationFailed(...)` with a
        message naming the unavailable module -- see
        houdini/scripts/python/fxhoudinimcp_server/handlers/
        usd_export_handlers.py's `_require_pxr`. This test asserts the SAME
        exception class (via the mocked `_FakeOperationFailed`, installed as
        `hou.OperationFailed` by `_install_hou_pxr_mocks` -- a REAL Exception
        subclass, not a bare MagicMock, so it is genuinely raisable/
        catchable) AND that the message names "MaterialX" -- so a generic,
        unrelated exception from some other bug in the preview body does NOT
        silently satisfy this test.
        """
        self._install_hou_pxr(monkeypatch)
        self._install_mtlx_absent(monkeypatch)
        mod = _fresh_import_usd_export_handlers()

        if not hasattr(mod, "_preview_mtlx_edit"):
            pytest.fail("usd_export_handlers._preview_mtlx_edit does not exist yet -- RED (expected).")

        hou_mock = sys.modules["hou"]
        with pytest.raises(hou_mock.OperationFailed) as exc_info:
            mod._preview_mtlx_edit({
                "mtlx_path": "$HIP/source.mtlx",
                "out_path": "$HIP/edited.mtlx",
                "edits": [{"node": "n1", "input": "base_color", "value": "0.5"}],
            })
        assert "materialx" in str(exc_info.value).lower(), (
            f"The HAS_MTLX-absent raise must name 'MaterialX' in its message "
            f"(mirroring _require_pxr's convention of naming the unavailable "
            f"module), not a generic/unrelated error. "
            f"Got exception message={str(exc_info.value)!r}."
        )

    def _install_hou_pxr(self, monkeypatch):
        _install_hou_pxr_mocks(monkeypatch)


# ---------------------------------------------------------------------------
# plan-6/new-3: string-only edit values (v1 = STRINGS ONLY)
# ---------------------------------------------------------------------------

class TestStringOnlyEditValueGuard:
    """plan-6/new-3 (REQUIRED): an edit whose 'value' is NOT a string (int,
    float, bool, None, list, dict) must be rejected {ok: False, error: ...}
    -- v1 accepts string-valued edits ONLY. A string value must succeed
    (assuming the node/input resolve).

    Rationale (new-3): allowing int/float made the post-write round-trip
    verify (getValueString() == str(value)) FALSE-FAIL when MaterialX's
    serialization differs from Python str() (e.g. 1.0 -> '1'). String-only
    removes the ambiguity.
    """

    @pytest.mark.parametrize("bad_value", [1, 1.0, True, False, None, [1, 2], {"a": 1}])
    def test_non_string_value_rejected(self, mock_env, bad_value):
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path="$HIP/edited.mtlx",
            edits=[{"node": "standard_surface1", "input": "base_color", "value": bad_value}],
        )
        assert result.get("ok") is False, (
            f"An edit value of {bad_value!r} (type {type(bad_value).__name__}) must be "
            f"rejected -- v1 supports STRING-valued edits only. Got result={result!r}."
        )
        error_text = str(result.get("error", "")).lower()
        assert "string" in error_text, (
            f"The rejection error must mention the string-only requirement. "
            f"Got error={result.get('error')!r}."
        )

    def test_string_value_accepted_shape_wise(self, monkeypatch):
        """A string-valued edit must pass the shape/type guard (it may still
        fail later for other reasons, e.g. node-not-found in this minimal
        doc, but NOT for being non-string)."""
        _install_hou_pxr_mocks(monkeypatch)
        # Empty doc -> node not found is the expected downstream failure,
        # proving the string-type guard itself did not reject it.
        _install_materialx_mock(monkeypatch, doc_factory=lambda: _FakeMxDocument())
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path="$HIP/edited.mtlx",
            edits=[{"node": "standard_surface1", "input": "base_color", "value": "0.5"}],
        )
        # Must NOT be rejected for a string-type reason.
        error_text = str(result.get("error", "")).lower()
        assert "must be a string" not in error_text, (
            f"A string-valued edit must not be rejected by the string-only type "
            f"guard. Got error={result.get('error')!r}."
        )


# ---------------------------------------------------------------------------
# plan-2/new-1: ambiguity fail-loud (unqualified name resolves in TWO graphs)
# ---------------------------------------------------------------------------

class TestAmbiguousNodeResolutionFailsLoud:
    """plan-2/new-1 (REQUIRED): an unqualified node name that resolves to a
    node present in TWO different nodegraphs must return {ok: False, error}
    naming the ambiguity -- NOT silently edit the wrong node. A qualified
    'nodegraph/node' path must resolve unambiguously.
    """

    def _build_ambiguous_doc(self):
        """Two nodegraphs (NG_a, NG_b), each containing a node literally
        named 'shared_name' -- an unqualified lookup for 'shared_name' must
        find BOTH and be rejected as ambiguous."""
        node_a = _FakeMxNode("shared_name", inputs={
            "base_color": _FakeMxInput("base_color", "1.0", "color3"),
        })
        node_b = _FakeMxNode("shared_name", inputs={
            "base_color": _FakeMxInput("base_color", "0.2", "color3"),
        })
        ng_a = _FakeMxNodeGraph("NG_a", nodes={"shared_name": node_a})
        ng_b = _FakeMxNodeGraph("NG_b", nodes={"shared_name": node_b})
        return _FakeMxDocument(nodegraphs={"NG_a": ng_a, "NG_b": ng_b})

    def test_unqualified_ambiguous_name_rejected_no_write(self, monkeypatch):
        _install_hou_pxr_mocks(monkeypatch)
        _install_materialx_mock(monkeypatch, doc_factory=self._build_ambiguous_doc)
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        mx_mock = sys.modules["MaterialX"]
        result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path="$HIP/edited.mtlx",
            edits=[{"node": "shared_name", "input": "base_color", "value": "0.9"}],
        )
        assert result.get("ok") is False, (
            f"An unqualified node name present in TWO nodegraphs must be "
            f"rejected as ambiguous, NOT silently edit one of them. "
            f"Got result={result!r}."
        )
        error_text = str(result.get("error", "")).lower()
        assert "ambiguous" in error_text, (
            f"The rejection error must name the ambiguity explicitly. "
            f"Got error={result.get('error')!r}."
        )
        # No write must have occurred on this rejected path.
        mx_mock.writeToXmlFile.assert_not_called()

    def test_qualified_nodegraph_node_path_resolves_unambiguously(self, monkeypatch):
        """A qualified 'NG_a/shared_name' path must resolve to exactly the
        node in NG_a, not be treated as ambiguous -- AND the edit must
        actually SUCCEED end-to-end (round-trip verified), not merely avoid
        the word "ambiguous" in an error string.

        red-2 (REQUIRED fix): the prior version of this test only asserted
        "ambiguous" was absent from the error text. That is satisfied by ANY
        non-ambiguity failure -- e.g. a not-found bug, a mis-resolution bug
        that silently edits the WRONG node then fails for an unrelated
        reason (round-trip mismatch), or any other bug that happens not to
        say "ambiguous". None of those would be caught by the old assertion.
        Fix: assert ok=True and that the write actually landed on the
        correct (NG_a) node's input -- proving genuine successful qualified
        resolution, not just an absence of one particular error keyword.
        """
        _install_hou_pxr_mocks(monkeypatch)

        # Build a document whose SECOND createDocument() call (the handler's
        # post-write round-trip re-read) reflects the edit having correctly
        # landed on NG_a's node (value '0.9'), so a genuine end-to-end
        # success requires the whole resolve->apply->write->reread chain to
        # operate on the CORRECT (qualified) node.
        call_count = {"n": 0}

        def _doc_factory():
            call_count["n"] += 1
            if call_count["n"] == 1:
                return self._build_ambiguous_doc()
            # Post-write re-read: NG_a's node now shows the edited value;
            # NG_b's node is untouched (still '0.2') -- proves the edit
            # targeted NG_a specifically, not NG_b, not both, not neither.
            reread_node_a = _FakeMxNode("shared_name", inputs={
                "base_color": _FakeMxInput("base_color", "0.9", "color3"),
            })
            reread_node_b = _FakeMxNode("shared_name", inputs={
                "base_color": _FakeMxInput("base_color", "0.2", "color3"),
            })
            reread_ng_a = _FakeMxNodeGraph("NG_a", nodes={"shared_name": reread_node_a})
            reread_ng_b = _FakeMxNodeGraph("NG_b", nodes={"shared_name": reread_node_b})
            return _FakeMxDocument(nodegraphs={"NG_a": reread_ng_a, "NG_b": reread_ng_b})

        _install_materialx_mock(monkeypatch, doc_factory=_doc_factory)
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path="$HIP/edited.mtlx",
            edits=[{"node": "NG_a/shared_name", "input": "base_color", "value": "0.9"}],
        )
        error_text = str(result.get("error", "")).lower()
        assert "ambiguous" not in error_text, (
            f"A qualified 'nodegraph/node' path must resolve unambiguously -- "
            f"it must NOT be rejected as ambiguous. Got result={result!r}."
        )
        assert result.get("ok") is True, (
            f"A qualified 'NG_a/shared_name' edit against a document where "
            f"the unqualified name is ambiguous must SUCCEED end-to-end "
            f"(resolve to exactly the NG_a node, apply, write, and round-trip "
            f"verify) -- not merely avoid the word 'ambiguous'. "
            f"Got result={result!r}."
        )
        assert result.get("edits_applied") == 1, (
            f"The qualified-path edit must report exactly 1 edit applied. "
            f"Got {result.get('edits_applied')!r}, full result={result!r}."
        )


# ---------------------------------------------------------------------------
# new-4: duplicate edit targets rejected BEFORE any write
# ---------------------------------------------------------------------------

class TestDuplicateEditTargetsRejected:
    """new-4 (REQUIRED): two edits resolving to the SAME (node, input) target
    must be rejected {ok: False, error} BEFORE any write -- else the second
    silently overwrites the first and the round-trip verify would falsely
    fail against the first edit's expected value.
    """

    def _build_single_node_doc(self):
        node = _FakeMxNode("standard_surface1", inputs={
            "base_color": _FakeMxInput("base_color", "1.0", "color3"),
        })
        ng = _FakeMxNodeGraph("NG_main", nodes={"standard_surface1": node})
        return _FakeMxDocument(nodegraphs={"NG_main": ng})

    def test_duplicate_target_same_unqualified_name_rejected(self, monkeypatch):
        _install_hou_pxr_mocks(monkeypatch)
        _install_materialx_mock(monkeypatch, doc_factory=self._build_single_node_doc)
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        mx_mock = sys.modules["MaterialX"]
        result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path="$HIP/edited.mtlx",
            edits=[
                {"node": "standard_surface1", "input": "base_color", "value": "0.5"},
                {"node": "standard_surface1", "input": "base_color", "value": "0.7"},
            ],
        )
        assert result.get("ok") is False, (
            f"Two edits resolving to the SAME (node, input) target must be "
            f"rejected before any write. Got result={result!r}."
        )
        error_text = str(result.get("error", "")).lower()
        assert "duplicate" in error_text, (
            f"The rejection error must name the duplicate-target condition. "
            f"Got error={result.get('error')!r}."
        )
        mx_mock.writeToXmlFile.assert_not_called()

    def test_duplicate_target_via_qualified_and_unqualified_alias_rejected(self, monkeypatch):
        """A duplicate detected via the RESOLVED path (rpath) even when one
        edit uses the qualified form and the other the unqualified form for
        the SAME node -- both resolve to the same (rpath, input)."""
        _install_hou_pxr_mocks(monkeypatch)
        _install_materialx_mock(monkeypatch, doc_factory=self._build_single_node_doc)
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        mx_mock = sys.modules["MaterialX"]
        result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path="$HIP/edited.mtlx",
            edits=[
                {"node": "standard_surface1", "input": "base_color", "value": "0.5"},
                {"node": "NG_main/standard_surface1", "input": "base_color", "value": "0.7"},
            ],
        )
        assert result.get("ok") is False, (
            f"Two edits resolving to the SAME target via different name forms "
            f"(unqualified vs qualified) must still be rejected as a duplicate. "
            f"Got result={result!r}."
        )
        mx_mock.writeToXmlFile.assert_not_called()


# ---------------------------------------------------------------------------
# plan-3/new-2: round-trip verify -- confirms the value actually landed
# ---------------------------------------------------------------------------

class TestPostWriteRoundTripVerify:
    """plan-3/new-2 (REQUIRED): a successful edit must re-read the WRITTEN
    file and confirm the edited input's value equals the requested string.
    A wrong/no-op edit (the round-trip re-read observes a DIFFERENT or
    missing value) must return {ok: False} naming the round-trip mismatch --
    re-resolved by the RECORDED rpath, not by re-running the (possibly
    ambiguous) name search.
    """

    def _build_single_node_doc_pre_write(self):
        node = _FakeMxNode(
            "standard_surface1",
            inputs={"base_color": _FakeMxInput("base_color", "1.0", "color3")},
            name_path="standard_surface1",
        )
        ng = _FakeMxNodeGraph("NG_main", nodes={"standard_surface1": node})
        return _FakeMxDocument(nodegraphs={"NG_main": ng})

    def test_round_trip_mismatch_reported_as_failure(self, monkeypatch):
        """The post-write re-read document reports a value that does NOT
        match what was requested -- the handler must catch this and fail,
        not report ok=True."""
        _install_hou_pxr_mocks(monkeypatch)

        # First createDocument() call -> the pre-write doc the handler reads
        # + edits in-memory. Second createDocument() call -> the post-write
        # re-read doc, deliberately built with a MISMATCHED value to force
        # the round-trip-failure branch.
        call_count = {"n": 0}

        def _doc_factory():
            call_count["n"] += 1
            if call_count["n"] == 1:
                return self._build_single_node_doc_pre_write()
            # Second (post-write re-read) doc: value is WRONG (still '1.0'
            # instead of the requested '0.5') -- simulates a value that did
            # not actually land in the written file.
            mismatched_node = _FakeMxNode(
                "standard_surface1",
                inputs={"base_color": _FakeMxInput("base_color", "1.0", "color3")},
                name_path="standard_surface1",
            )
            mismatched_ng = _FakeMxNodeGraph("NG_main", nodes={"standard_surface1": mismatched_node})
            return _FakeMxDocument(nodegraphs={"NG_main": mismatched_ng})

        _install_materialx_mock(monkeypatch, doc_factory=_doc_factory)
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path="$HIP/edited.mtlx",
            edits=[{"node": "standard_surface1", "input": "base_color", "value": "0.5"}],
        )
        assert result.get("ok") is False, (
            f"A post-write round-trip mismatch (re-read value != requested value) "
            f"must be reported as a failure, NOT ok=True. Got result={result!r}."
        )
        error_text = str(result.get("error", "")).lower()
        assert "round-trip" in error_text or "round trip" in error_text, (
            f"The failure error must name the round-trip mismatch explicitly. "
            f"Got error={result.get('error')!r}."
        )

    def test_round_trip_success_reported_as_ok_true(self, monkeypatch):
        """When the post-write re-read DOES observe the requested value, the
        handler must report ok=True with edits_applied and a validate block."""
        _install_hou_pxr_mocks(monkeypatch)

        call_count = {"n": 0}

        def _doc_factory():
            call_count["n"] += 1
            if call_count["n"] == 1:
                return self._build_single_node_doc_pre_write()
            # Second (post-write re-read) doc: value CORRECTLY reflects the
            # requested edit -- '0.5'.
            matched_node = _FakeMxNode(
                "standard_surface1",
                inputs={"base_color": _FakeMxInput("base_color", "0.5", "color3")},
                name_path="standard_surface1",
            )
            matched_ng = _FakeMxNodeGraph("NG_main", nodes={"standard_surface1": matched_node})
            return _FakeMxDocument(nodegraphs={"NG_main": matched_ng})

        _install_materialx_mock(monkeypatch, doc_factory=_doc_factory)
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path="$HIP/edited.mtlx",
            edits=[{"node": "standard_surface1", "input": "base_color", "value": "0.5"}],
        )
        assert result.get("ok") is True, (
            f"A post-write round-trip that confirms the requested value landed "
            f"must return ok=True. Got result={result!r}, error={result.get('error')!r}."
        )
        assert result.get("edits_applied") == 1, (
            f"edits_applied must report the count of successfully applied edits. "
            f"Got {result.get('edits_applied')!r}."
        )
        assert "validate" in result, (
            f"A successful edit result must include a 'validate' block. "
            f"Got keys={list(result.keys())!r}."
        )

    def test_round_trip_reresolves_by_recorded_path_not_by_name_search(self, monkeypatch):
        """new-2 (REQUIRED -- the KEY red-3 fold): the post-write round-trip
        re-resolution MUST use the RECORDED node path (rpath) captured
        during the resolve pass, NOT re-run the (possibly ambiguous or
        differently-resolving) name search against the re-read document.

        Scenario: the edit targets an UNQUALIFIED name that is unambiguous
        in the PRE-write document (resolves to exactly one node, in
        NG_edit). The edit is applied and written. But in the POST-write
        re-read document, a node with the SAME unqualified name ALSO exists
        in a second nodegraph (NG_decoy) -- as if some other process/content
        in the file also uses that name, or (more simply here) simulating
        that a NAME-based re-search on the re-read doc would now find TWO
        matches (ambiguous) where a PATH-based re-resolution finds exactly
        the recorded one.

        - A WRONG implementation that re-resolves by re-running the
          unqualified name search against the post-write doc would find the
          name ambiguous (two matches) and either (a) raise/fail the
          round-trip for the wrong reason ("ambiguous" instead of a genuine
          mismatch), or (b) pick an arbitrary one of the two and potentially
          report a false pass/fail depending on which one it happens to
          pick. Either way it does NOT deterministically re-resolve via the
          value at the RECORDED path.
        - A CORRECT implementation re-resolves by the recorded rpath
          ("NG_edit/standard_surface1", from node.getNamePath() captured at
          resolve time) directly via doc2.getNodeGraph(g).getNode(n) --
          entirely bypassing the ambiguous name search -- and finds exactly
          the correct node with the correct edited value, so the round-trip
          succeeds with ok=True.

        This test asserts the CORRECT (recorded-path) behavior: ok=True,
        despite the post-write document now containing a same-named node in
        a second nodegraph that would make a name-based re-search ambiguous.

        red-3-round2 fix (REQUIRED -- codex-pair-reviewer round-2 cap
        finding, threadId 019f1e70): the post-write doc's nodegraphs dict is
        built with NG_decoy inserted BEFORE NG_edit (see
        _build_post_write_doc below) -- this ordering is what makes the test
        actually discriminate a naive first-match-wins name-search from a
        correct recorded-path implementation. With the decoy first, a
        first-hit name search reads the decoy's UNRELATED value '9.9'
        (mismatching the requested '0.77') and fails; only a recorded-path
        re-resolution (which ignores iteration order entirely) reads the
        correct '0.77' and passes.
        """
        _install_hou_pxr_mocks(monkeypatch)

        def _build_pre_write_doc():
            # Pre-write: 'standard_surface1' exists ONLY in NG_edit -- the
            # unqualified name is unambiguous at resolve time.
            node = _FakeMxNode(
                "standard_surface1",
                inputs={"base_color": _FakeMxInput("base_color", "1.0", "color3")},
                name_path="NG_edit/standard_surface1",
            )
            ng_edit = _FakeMxNodeGraph("NG_edit", nodes={"standard_surface1": node})
            return _FakeMxDocument(nodegraphs={"NG_edit": ng_edit})

        def _build_post_write_doc():
            # Post-write re-read: a SECOND node also named
            # 'standard_surface1' now exists in NG_decoy -- a name-based
            # re-search for the unqualified name 'standard_surface1' against
            # THIS document would find TWO matches (ambiguous). The
            # RECORDED-PATH node (NG_edit/standard_surface1) correctly shows
            # the edited value '0.77'; the decoy node
            # (NG_decoy/standard_surface1) shows an unrelated, unedited
            # value '9.9' -- proving a correct implementation reads from the
            # recorded path, not from an ambiguous re-search that could pick
            # either node.
            #
            # red-3-round2 fix (REQUIRED -- codex-pair-reviewer round-2 cap
            # finding, threadId 019f1e70): the nodegraphs dict below is
            # DELIBERATELY ordered {"NG_decoy": ..., "NG_edit": ...} -- decoy
            # FIRST, edit SECOND. This ordering is LOAD-BEARING and must NOT
            # be "tidied" back to NG_edit-first.
            #
            # Why: Python dicts preserve insertion order (3.7+), and
            # _FakeMxDocument.getNodeGraphs()/getNode() iterate/lookup over
            # that same insertion order. With NG_edit inserted FIRST (the
            # prior, buggy ordering), a WRONG implementation that re-resolves
            # the round-trip via a naive FIRST-MATCH-WINS name search (scan
            # nodegraphs in order, return the first node whose unqualified
            # name matches, with NO ambiguity check) would hit NG_edit's
            # node FIRST, read its correctly-edited value '0.77', and pass
            # every assertion below -- WITHOUT ever implementing recorded-
            # rpath re-resolution. That made the test unable to discriminate
            # a naive first-hit-wins name-search from a correct recorded-path
            # implementation.
            #
            # With NG_decoy inserted FIRST instead: a first-match-wins name
            # search over getNodeGraphs() now encounters NG_decoy's node
            # FIRST and reads its value '9.9' -- which does NOT equal the
            # requested '0.77', so the round-trip verify FAILS (ok=False) for
            # a naive/buggy first-hit implementation. Only a CORRECT
            # implementation -- one that re-resolves via the RECORDED rpath
            # ("NG_edit/standard_surface1", captured at resolve time) directly
            # via doc2.getNodeGraph("NG_edit").getNode("standard_surface1"),
            # bypassing name search and iteration order entirely -- reads the
            # correct '0.77' value and passes (ok=True), regardless of which
            # nodegraph happens to be inserted/iterated first.
            edited_node = _FakeMxNode(
                "standard_surface1",
                inputs={"base_color": _FakeMxInput("base_color", "0.77", "color3")},
                name_path="NG_edit/standard_surface1",
            )
            decoy_node = _FakeMxNode(
                "standard_surface1",
                inputs={"base_color": _FakeMxInput("base_color", "9.9", "color3")},
                name_path="NG_decoy/standard_surface1",
            )
            ng_edit = _FakeMxNodeGraph("NG_edit", nodes={"standard_surface1": edited_node})
            ng_decoy = _FakeMxNodeGraph("NG_decoy", nodes={"standard_surface1": decoy_node})
            # DO NOT reorder: NG_decoy MUST be the first key so any
            # first-match-wins iteration (getNodeGraphs() -> list(dict.values()))
            # encounters the decoy node before the correctly-edited node.
            return _FakeMxDocument(nodegraphs={"NG_decoy": ng_decoy, "NG_edit": ng_edit})

        call_count = {"n": 0}

        def _doc_factory():
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _build_pre_write_doc()
            return _build_post_write_doc()

        _install_materialx_mock(monkeypatch, doc_factory=_doc_factory)
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path="$HIP/edited.mtlx",
            edits=[{"node": "standard_surface1", "input": "base_color", "value": "0.77"}],
        )
        assert result.get("ok") is True, (
            f"The post-write round-trip re-resolution must use the RECORDED "
            f"node path (captured during the resolve pass) to re-find the "
            f"edited node in the re-read document -- NOT re-run the "
            f"unqualified name search (which would now be ambiguous, since "
            f"the post-write document contains a second same-named node in "
            f"a different nodegraph). A wrong (name-search) implementation "
            f"fails this scenario (either an 'ambiguous' error or a false "
            f"mismatch against whichever node the ambiguous search happens "
            f"to pick); a correct (recorded-path) implementation succeeds "
            f"deterministically. Got result={result!r}."
        )
        assert result.get("edits_applied") == 1, (
            f"Expected edits_applied=1. Got {result.get('edits_applied')!r}, "
            f"full result={result!r}."
        )
        error_text = str(result.get("error", "")).lower()
        assert "ambiguous" not in error_text, (
            f"The round-trip re-resolution must not fail with an ambiguity "
            f"error -- that would indicate a name-based re-search rather "
            f"than a recorded-path re-resolution. Got result={result!r}."
        )


# ---------------------------------------------------------------------------
# plan-1: preview parse-catch -- unparseable source -> operator-visible
#         source_parseable: False, NOT an uncaught raise
# ---------------------------------------------------------------------------

class TestPreviewParseCatch:
    """plan-1 (REQUIRED): _preview_mtlx_edit on an unparseable source must
    return a payload with source_parseable: False (operator-visible), NOT
    an uncaught raise -- the preview CATCHES the readFromXmlFile parse
    failure itself, rather than betting on the gate's raise->DENY mechanism
    for an arbitrary pybind exception.
    """

    def test_unparseable_source_returns_source_parseable_false(self, monkeypatch):
        _install_hou_pxr_mocks(monkeypatch)

        def _raise_on_read(doc, path):
            raise RuntimeError(f"MaterialX parse error: malformed XML at {path}")

        _install_materialx_mock(
            monkeypatch,
            doc_factory=lambda: _FakeMxDocument(),
            read_side_effect=_raise_on_read,
        )
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "_preview_mtlx_edit"):
            pytest.fail("usd_export_handlers._preview_mtlx_edit does not exist yet -- RED (expected).")

        # Must NOT raise -- the preview catches the parse failure itself.
        result = mod._preview_mtlx_edit({
            "mtlx_path": "$HIP/broken.mtlx",
            "out_path": "$HIP/edited.mtlx",
            "edits": [{"node": "n1", "input": "base_color", "value": "0.5"}],
        })
        assert isinstance(result, dict), (
            f"_preview_mtlx_edit must return a dict (operator-visible payload) "
            f"even when the source is unparseable -- not raise. Got {type(result)!r}."
        )
        assert result.get("source_parseable") is False, (
            f"An unparseable source must yield source_parseable=False in the "
            f"preview payload. Got result={result!r}."
        )
        assert "parse_error" in result, (
            f"The preview payload for an unparseable source must include a "
            f"'parse_error' key. Got keys={list(result.keys())!r}."
        )

    def test_parseable_source_returns_source_parseable_true(self, monkeypatch):
        """A normal parseable source must NOT be flagged source_parseable=False."""
        _install_hou_pxr_mocks(monkeypatch)

        node = _FakeMxNode("standard_surface1", inputs={
            "base_color": _FakeMxInput("base_color", "1.0", "color3"),
        })
        ng = _FakeMxNodeGraph("NG_main", nodes={"standard_surface1": node})
        _install_materialx_mock(
            monkeypatch,
            doc_factory=lambda: _FakeMxDocument(nodegraphs={"NG_main": ng}),
        )
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "_preview_mtlx_edit"):
            pytest.fail("usd_export_handlers._preview_mtlx_edit does not exist yet -- RED (expected).")

        result = mod._preview_mtlx_edit({
            "mtlx_path": "$HIP/source.mtlx",
            "out_path": "$HIP/edited.mtlx",
            "edits": [{"node": "standard_surface1", "input": "base_color", "value": "0.5"}],
        })
        # red-5 (REQUIRED fix): assert source_parseable is EXPLICITLY True,
        # not merely "not False" -- the prior assertion
        # `result.get("source_parseable", True) is not False` is satisfied
        # by an implementation that OMITS the source_parseable key entirely
        # (the .get(..., True) default silently supplies True), which means
        # this test would pass even if the field were never populated for
        # the success path. A real implementation must positively populate
        # source_parseable=True on a successful parse -- the field's
        # presence and value are part of the documented preview payload
        # contract (lockedFieldContract: "source_parseable: True").
        assert result.get("source_parseable") is True, (
            f"A parseable source must yield source_parseable=True EXPLICITLY "
            f"in the preview payload (not merely absent-and-defaulted). "
            f"Got result={result!r}, source_parseable={result.get('source_parseable')!r}."
        )


# ---------------------------------------------------------------------------
# plan-7: shared edit-shape guard -- BOTH preview and handler run it
# ---------------------------------------------------------------------------

class TestEditShapeGuard:
    """plan-7 (REQUIRED): the SAME edit-shape guard runs in both the preview
    and the handler. Each case must return {ok: False, error: ...} (handler)
    or an operator-visible shape-rejection payload (preview) WITHOUT
    creating/mutating anything.
    """

    @pytest.mark.parametrize("bad_edits", [
        None,
        [],
        "not-a-list",
        {"node": "n1", "input": "i1", "value": "v1"},  # a dict, not a list of dicts
    ])
    def test_handler_rejects_non_list_or_empty_edits(self, mock_env, bad_edits):
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx", out_path="$HIP/edited.mtlx", edits=bad_edits
        )
        assert result.get("ok") is False, (
            f"edits={bad_edits!r} must be rejected (a non-empty list of "
            f"{{node,input,value}} objects is required). Got result={result!r}."
        )
        assert isinstance(result.get("error"), str) and result["error"], (
            f"The rejection must carry a non-empty error string. "
            f"Got error={result.get('error')!r}."
        )

    @pytest.mark.parametrize("bad_edit_item", [
        {},
        {"node": "n1"},
        {"node": "n1", "input": "i1"},
        {"input": "i1", "value": "v1"},
        {"node": "n1", "value": "v1"},
        "not-a-dict",
        123,
    ])
    def test_handler_rejects_edit_missing_required_keys(self, mock_env, bad_edit_item):
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path="$HIP/edited.mtlx",
            edits=[bad_edit_item],
        )
        assert result.get("ok") is False, (
            f"An edit item {bad_edit_item!r} missing node/input/value keys "
            f"must be rejected. Got result={result!r}."
        )

    @pytest.mark.parametrize("bad_edits", [
        None,
        [],
        "not-a-list",
        {"node": "n1", "input": "i1", "value": "v1"},  # a dict, not a list of dicts
        [{"node": "n1"}],                               # missing input/value
        [{"node": "n1", "input": "i1"}],                 # missing value
        [{"input": "i1", "value": "v1"}],                 # missing node
        [{"node": "n1", "value": "v1"}],                   # missing input
        [123],                                             # non-dict item
        [1, {"node": "n1", "input": "i1", "value": "v1"}],  # one bad item mixed in
    ], ids=[
        "edits-None", "edits-empty-list", "edits-not-a-list", "edits-bare-dict",
        "item-missing-input-value", "item-missing-value", "item-missing-node",
        "item-missing-input", "item-non-dict", "mixed-good-and-bad-item",
    ])
    def test_preview_rejects_same_shape_failures_as_handler(self, mock_env, bad_edits):
        """FOLD plan-7 domain parity (red-4 REQUIRED fix): the preview must
        run the SAME _validate_edits_shape guard the handler runs, and
        surface a rejection payload rather than proceeding as if the edits
        were well-formed -- across the FULL set of malformed shapes the
        handler-side guard covers (non-list, empty list, missing
        node/input/value key, non-string/non-dict value), not just the
        single "missing input/value" case the prior version of this test
        exercised. Each malformed case must independently trigger the
        preview's shape-rejection payload.
        """
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "_preview_mtlx_edit"):
            pytest.fail("usd_export_handlers._preview_mtlx_edit does not exist yet -- RED (expected).")

        result = mod._preview_mtlx_edit({
            "mtlx_path": "$HIP/source.mtlx",
            "out_path": "$HIP/edited.mtlx",
            "edits": bad_edits,
        })
        assert isinstance(result, dict), (
            f"_preview_mtlx_edit must return a dict even on a shape failure "
            f"(operator-visible rejection), not raise an unrelated error. "
            f"edits={bad_edits!r}. Got {type(result)!r}."
        )
        assert result.get("edits_shape_ok") is False, (
            f"A shape-invalid edits list ({bad_edits!r}) must produce "
            f"edits_shape_ok=False in the preview payload. Got result={result!r}."
        )
        assert "shape_error" in result, (
            f"The preview payload for a shape failure (edits={bad_edits!r}) "
            f"must include a 'shape_error' key. Got keys={list(result.keys())!r}."
        )


# ---------------------------------------------------------------------------
# No-regex: the edit path uses the MaterialX API (setValueString), never
#           text substitution on the raw XML/doc content.
# ---------------------------------------------------------------------------

class TestNoRegexEditPath:
    """The edit path must use setValueString on the resolved MaterialX Input
    element -- never a regex/text-substitution over the document's XML. This
    test proves the fake node's setValueString is what actually changes the
    observable value (the round-trip verify above already proves this
    indirectly; this test asserts the mechanism directly via a spy)."""

    def test_edit_calls_set_value_string_on_the_resolved_input(self, monkeypatch):
        _install_hou_pxr_mocks(monkeypatch)

        target_input = _FakeMxInput("base_color", "1.0", "color3")
        node = _FakeMxNode(
            "standard_surface1", inputs={"base_color": target_input}, name_path="standard_surface1"
        )
        ng = _FakeMxNodeGraph("NG_main", nodes={"standard_surface1": node})

        call_count = {"n": 0}

        def _doc_factory():
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _FakeMxDocument(nodegraphs={"NG_main": ng})
            # Post-write re-read doc must reflect the mutation performed on
            # the SAME target_input object (setValueString mutates it
            # in-place), proving the API call -- not a text substitution --
            # is what changed the value.
            reread_node = _FakeMxNode(
                "standard_surface1",
                inputs={"base_color": _FakeMxInput("base_color", target_input.getValueString(), "color3")},
                name_path="standard_surface1",
            )
            reread_ng = _FakeMxNodeGraph("NG_main", nodes={"standard_surface1": reread_node})
            return _FakeMxDocument(nodegraphs={"NG_main": reread_ng})

        _install_materialx_mock(monkeypatch, doc_factory=_doc_factory)
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path="$HIP/edited.mtlx",
            edits=[{"node": "standard_surface1", "input": "base_color", "value": "0.42"}],
        )
        assert target_input.getValueString() == "0.42", (
            f"The edit must call setValueString on the resolved MaterialX Input "
            f"element (the API, not a text/regex substitution). Expected the "
            f"target input's value to be mutated to '0.42', got "
            f"{target_input.getValueString()!r}."
        )
        assert result.get("ok") is True, (
            f"The edit-via-API path must succeed end-to-end (shape ok, resolve "
            f"ok, write ok, round-trip ok). Got result={result!r}."
        )


# ---------------------------------------------------------------------------
# red-6 (KEY FOLD): two-pass resolve-then-write invariant -- an invalid edit
#                   anywhere in the list must prevent ANY write, including of
#                   the other (valid) edits in the same call.
# ---------------------------------------------------------------------------

class TestTwoPassResolveThenWriteInvariant:
    """red-6 (REQUIRED -- the KEY two-pass fold): the handler MUST resolve
    ALL edits first (PASS 1, no mutation) and only apply+write (PASS 2) if
    EVERY edit resolved successfully. An edits list containing one valid
    edit and one invalid edit (e.g. targeting a non-existent node/input)
    must return {ok: False} for the WHOLE call, AND the valid edit must NOT
    have been written -- proving resolve-ALL-then-write, not
    write-as-you-go (where the first edit could land before the second one
    is discovered to be invalid).

    This directly targets a write-as-you-go implementation, which would:
      (a) successfully mutate the first (valid) edit's Input object via
          setValueString before discovering the second edit is invalid, and/or
      (b) call writeToXmlFile at least once despite the overall call failing.
    A correct two-pass implementation does neither.
    """

    def _build_doc_with_one_node(self):
        """A document with exactly one node/input -- edit #1 targets this
        real node+input (valid); edit #2 targets a node that does not exist
        anywhere in the document (invalid -- 'node not found')."""
        target_input = _FakeMxInput("base_color", "1.0", "color3")
        node = _FakeMxNode(
            "standard_surface1",
            inputs={"base_color": target_input},
            name_path="NG_main/standard_surface1",
        )
        ng = _FakeMxNodeGraph("NG_main", nodes={"standard_surface1": node})
        doc = _FakeMxDocument(nodegraphs={"NG_main": ng})
        return doc, target_input

    def test_invalid_edit_in_list_blocks_write_of_valid_edit_too(self, monkeypatch):
        _install_hou_pxr_mocks(monkeypatch)

        doc, target_input = self._build_doc_with_one_node()
        # A single, stable document instance -- if PASS 2 ever mutated
        # target_input via setValueString, that mutation would be directly
        # observable on this SAME object (no round-trip needed to catch a
        # write-as-you-go implementation applying edit #1 in-memory before
        # discovering edit #2 is invalid).
        _install_materialx_mock(monkeypatch, doc_factory=lambda: doc)
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        mx_mock = sys.modules["MaterialX"]
        result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path="$HIP/edited.mtlx",
            edits=[
                # Edit #1: VALID (targets the real node/input).
                {"node": "standard_surface1", "input": "base_color", "value": "0.5"},
                # Edit #2: INVALID (targets a node that does not exist).
                {"node": "does_not_exist", "input": "base_color", "value": "0.9"},
            ],
        )
        assert result.get("ok") is False, (
            f"An edits list containing ANY invalid edit (here: edit #2 "
            f"targets a non-existent node) must fail the WHOLE call -- "
            f"even though edit #1 targets a real, resolvable node/input. "
            f"Got result={result!r}."
        )
        # PASS 1 (resolve-all, no mutation) must have run BEFORE any write:
        # the valid edit's target input must NOT have been mutated, proving
        # the implementation did not apply edit #1 before discovering edit
        # #2 was invalid (write-as-you-go would mutate this in-place).
        assert target_input.getValueString() == "1.0", (
            f"The valid edit (edit #1, targeting a real node/input) must "
            f"NOT have been applied -- the handler must resolve ALL edits "
            f"BEFORE applying/writing ANY of them (two-pass: resolve-then-"
            f"write). A write-as-you-go implementation would have already "
            f"mutated this input to '0.5' before discovering edit #2 was "
            f"invalid. Got getValueString()={target_input.getValueString()!r}."
        )
        # No write must have occurred at all -- neither edit landed on disk.
        mx_mock.writeToXmlFile.assert_not_called()

    def test_invalid_edit_error_names_the_failing_edit(self, monkeypatch):
        """The failure must name the SPECIFIC invalid edit (node not found),
        not just fail generically -- confirming PASS 1 actually inspected
        edit #2's target and didn't merely fail on an unrelated exception."""
        _install_hou_pxr_mocks(monkeypatch)

        doc, target_input = self._build_doc_with_one_node()
        _install_materialx_mock(monkeypatch, doc_factory=lambda: doc)
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path="$HIP/edited.mtlx",
            edits=[
                {"node": "standard_surface1", "input": "base_color", "value": "0.5"},
                {"node": "does_not_exist", "input": "base_color", "value": "0.9"},
            ],
        )
        assert result.get("ok") is False
        error_text = str(result.get("error", "")).lower()
        assert "does_not_exist" in error_text or "not found" in error_text, (
            f"The failure error must name the specific invalid edit (a node "
            f"'does_not_exist' that could not be resolved) -- proving PASS 1 "
            f"genuinely inspected and rejected edit #2's target, rather than "
            f"failing for an unrelated/generic reason. Got "
            f"error={result.get('error')!r}."
        )


# ---------------------------------------------------------------------------
# red-7: existing-inputs-only -- an edit targeting a non-existent input on a
#        real node must be rejected, and addInput must NEVER be called.
# ---------------------------------------------------------------------------

class TestExistingInputsOnlyNoAddInput:
    """red-7 (REQUIRED): mtlx_edit only edits EXISTING inputs on a node --
    it must never call addInput to create a new one. An edit targeting an
    input name that does not exist on an otherwise-real, resolvable node
    must be rejected {ok: False} naming input-not-found, and addInput must
    NEVER have been called on the node (proving the implementation did not
    fall back to creating the input rather than failing loud).
    """

    def _build_doc_with_node_missing_target_input(self):
        """A document with a real node that has ONE input ('base_color')
        but NOT the input the edit will target ('roughness')."""
        node = _FakeMxNode(
            "standard_surface1",
            inputs={"base_color": _FakeMxInput("base_color", "1.0", "color3")},
            name_path="NG_main/standard_surface1",
        )
        # Attach a spyable addInput -- MagicMock so we can assert
        # non-invocation. _FakeMxNode does not define addInput by default;
        # attach one dynamically here so the assertion below is meaningful
        # regardless of whether the real implementation would call it on a
        # real MaterialX Node (which DOES have addInput) or not.
        node.addInput = MagicMock(name="addInput")
        ng = _FakeMxNodeGraph("NG_main", nodes={"standard_surface1": node})
        return _FakeMxDocument(nodegraphs={"NG_main": ng}), node

    def test_nonexistent_input_on_real_node_rejected(self, monkeypatch):
        _install_hou_pxr_mocks(monkeypatch)
        doc, node = self._build_doc_with_node_missing_target_input()
        _install_materialx_mock(monkeypatch, doc_factory=lambda: doc)
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        mx_mock = sys.modules["MaterialX"]
        result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path="$HIP/edited.mtlx",
            edits=[{"node": "standard_surface1", "input": "roughness", "value": "0.3"}],
        )
        assert result.get("ok") is False, (
            f"An edit targeting an input ('roughness') that does not exist "
            f"on an otherwise-real, resolvable node ('standard_surface1') "
            f"must be rejected -- mtlx_edit only edits EXISTING inputs, it "
            f"never creates new ones. Got result={result!r}."
        )
        error_text = str(result.get("error", "")).lower()
        assert "input" in error_text and (
            "not found" in error_text or "not exist" in error_text
        ), (
            f"The rejection error must name the input-not-found condition. "
            f"Got error={result.get('error')!r}."
        )
        node.addInput.assert_not_called()
        mx_mock.writeToXmlFile.assert_not_called()


# ---------------------------------------------------------------------------
# red-8: FR-2 empty-path guards -- empty/whitespace mtlx_path, out_path
#        (edit) and mtlx_path_or_doc (inspect) each rejected {ok: False}.
# ---------------------------------------------------------------------------

class TestFr2EmptyPathGuards:
    """red-8 (REQUIRED): FR-2 (empty-input rejection) must be enforced for
    EVERY path-shaped parameter across BOTH handlers -- empty string AND
    whitespace-only string, for mtlx_path and out_path on mtlx_edit, and for
    mtlx_path_or_doc on mtlx_inspect. Each must independently return
    {ok: False, error: ...} rather than proceeding (and, for mtlx_edit,
    without calling MaterialX at all -- confirmed by MaterialX.createDocument
    never being invoked for these malformed-empty-path cases).
    """

    @pytest.mark.parametrize("bad_path", ["", "   ", "\t", "\n"])
    def test_mtlx_edit_rejects_empty_mtlx_path(self, mock_env, bad_path):
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        result = mod.mtlx_edit(
            mtlx_path=bad_path,
            out_path="$HIP/edited.mtlx",
            edits=[{"node": "n1", "input": "i1", "value": "v1"}],
        )
        assert result.get("ok") is False, (
            f"mtlx_path={bad_path!r} (empty/whitespace) must be rejected by "
            f"the FR-2 empty-input guard. Got result={result!r}."
        )
        assert isinstance(result.get("error"), str) and result["error"], (
            f"The rejection must carry a non-empty error string for "
            f"mtlx_path={bad_path!r}. Got error={result.get('error')!r}."
        )

    @pytest.mark.parametrize("bad_path", ["", "   ", "\t", "\n"])
    def test_mtlx_edit_rejects_empty_out_path(self, mock_env, bad_path):
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path=bad_path,
            edits=[{"node": "n1", "input": "i1", "value": "v1"}],
        )
        assert result.get("ok") is False, (
            f"out_path={bad_path!r} (empty/whitespace) must be rejected by "
            f"the FR-2 empty-input guard. Got result={result!r}."
        )
        assert isinstance(result.get("error"), str) and result["error"], (
            f"The rejection must carry a non-empty error string for "
            f"out_path={bad_path!r}. Got error={result.get('error')!r}."
        )

    @pytest.mark.parametrize("bad_path", ["", "   ", "\t", "\n"])
    def test_mtlx_inspect_rejects_empty_mtlx_path_or_doc(self, mock_env, bad_path):
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_inspect"):
            pytest.fail("usd_export_handlers.mtlx_inspect does not exist yet -- RED (expected).")

        result = mod.mtlx_inspect(mtlx_path_or_doc=bad_path)
        assert result.get("ok") is False, (
            f"mtlx_path_or_doc={bad_path!r} (empty/whitespace) must be "
            f"rejected by the FR-2 empty-input guard. Got result={result!r}."
        )
        assert isinstance(result.get("error"), str) and result["error"], (
            f"The rejection must carry a non-empty error string for "
            f"mtlx_path_or_doc={bad_path!r}. Got error={result.get('error')!r}."
        )

    def test_empty_mtlx_path_never_reaches_materialx_parse(self, mock_env, monkeypatch):
        """FR-2 must reject BEFORE any MaterialX API call -- confirming the
        guard runs early (not merely happens to fail downstream when
        readFromXmlFile is handed an empty path)."""
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        mx_mock = sys.modules["MaterialX"]
        mx_mock.createDocument.reset_mock()
        mx_mock.readFromXmlFile.reset_mock()

        result = mod.mtlx_edit(
            mtlx_path="",
            out_path="$HIP/edited.mtlx",
            edits=[{"node": "n1", "input": "i1", "value": "v1"}],
        )
        assert result.get("ok") is False
        mx_mock.readFromXmlFile.assert_not_called()


# ---------------------------------------------------------------------------
# red-9 (Minor): tighten the over-broad exception-catch assertion in the
#                round-trip-mismatch test to the specific expected substring.
# ---------------------------------------------------------------------------

class TestExceptionAssertionSpecificity:
    """red-9 (Minor): a regression guard proving the mtlx_edit failure-path
    error strings are SPECIFIC to their failure condition, not a single
    generic "an error occurred" catch-all -- confirming distinct failure
    modes (node-not-found vs input-not-found vs round-trip-mismatch vs
    ambiguous vs duplicate vs string-only) each produce their own
    identifiable substring, so a test asserting on one condition cannot be
    silently satisfied by an unrelated exception's generic text.
    """

    def _build_single_node_doc(self):
        node = _FakeMxNode(
            "standard_surface1",
            inputs={"base_color": _FakeMxInput("base_color", "1.0", "color3")},
            name_path="NG_main/standard_surface1",
        )
        ng = _FakeMxNodeGraph("NG_main", nodes={"standard_surface1": node})
        return _FakeMxDocument(nodegraphs={"NG_main": ng})

    def test_node_not_found_error_is_distinct_from_input_not_found_error(self, monkeypatch):
        _install_hou_pxr_mocks(monkeypatch)
        _install_materialx_mock(monkeypatch, doc_factory=self._build_single_node_doc)
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        node_not_found_result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path="$HIP/edited.mtlx",
            edits=[{"node": "does_not_exist", "input": "base_color", "value": "0.5"}],
        )
        input_not_found_result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path="$HIP/edited.mtlx",
            edits=[{"node": "standard_surface1", "input": "does_not_exist_input", "value": "0.5"}],
        )
        assert node_not_found_result.get("ok") is False
        assert input_not_found_result.get("ok") is False

        node_err = str(node_not_found_result.get("error", "")).lower()
        input_err = str(input_not_found_result.get("error", "")).lower()

        assert "node" in node_err and "not found" in node_err, (
            f"A node-not-found failure must mention 'node' and 'not found' "
            f"specifically. Got error={node_not_found_result.get('error')!r}."
        )
        assert "input" in input_err and (
            "not found" in input_err or "not exist" in input_err
        ), (
            f"An input-not-found failure must mention 'input' and the "
            f"not-found/not-exist condition specifically. "
            f"Got error={input_not_found_result.get('error')!r}."
        )
        assert node_err != input_err, (
            "A node-not-found failure and an input-not-found failure must "
            "produce DISTINCT error text -- a single generic catch-all "
            "message for both failure modes would make it impossible to "
            "tell which condition actually failed. "
            f"node_err={node_err!r}, input_err={input_err!r}."
        )


# ---------------------------------------------------------------------------
# red-10 (Minor): the fake MaterialX mock module carries a documented
#                 minimal surface (not a bare, unspecced MagicMock) so it
#                 cannot silently fabricate attributes the real module lacks.
# ---------------------------------------------------------------------------

class TestMaterialxMockHasDocumentedMinimalSurface:
    """red-10 (Minor): _install_materialx_mock's returned mock must expose
    (at minimum) exactly the documented MaterialX API surface this unit's
    contract depends on -- createDocument, readFromXmlFile, writeToXmlFile,
    getVersionString -- proving the test suite's fake module models a real,
    bounded API rather than an unbounded MagicMock that would silently
    fabricate any attribute access (including a typo'd or non-existent
    MaterialX API call in the implementation) as a benign, callable mock.

    red-10-round2 fix (Minor, REQUIRED): the mock is now bound via
    `MagicMock(spec=[<documented attribute name list>])`. This does NOT
    require introspecting the real pybind11 C-extension class (which
    genuinely isn't available for `spec=` to walk in this project's
    environment) -- `spec` also accepts a plain list of attribute names,
    which is sufficient to make the mock raise `AttributeError` on any
    attribute access NOT in that list, exactly like a real bounded API
    would reject a typo'd/non-existent call. This closes the gap the prior
    version of this test class honestly flagged in its own docstring: a
    typo'd or non-existent MaterialX attribute access in the implementation
    (e.g. a misspelled `mx.creatDocument`) is no longer silently
    auto-vivified as a magically-successful mock call -- it now raises
    AttributeError, just as it would against the real (bounded) MaterialX
    module.
    """

    def test_documented_materialx_surface_present(self, mock_env):
        mx = mock_env.mx
        # The plan's grounded MaterialX API surface (lockedFieldContract):
        # createDocument, readFromXmlFile, writeToXmlFile, getVersionString.
        for attr in ("createDocument", "readFromXmlFile", "writeToXmlFile", "getVersionString"):
            assert hasattr(mx, attr), (
                f"The mocked MaterialX module must expose '{attr}' -- part "
                f"of the plan's grounded MaterialX API surface this unit "
                f"depends on. Got dir(mx) subset missing {attr!r}."
            )

    def test_get_version_string_returns_a_realistic_string(self, mock_env):
        version = mock_env.mx.getVersionString()
        assert isinstance(version, str) and version, (
            f"getVersionString() must return a non-empty string (the mock "
            f"models the real MaterialX API's documented return type). "
            f"Got {version!r}."
        )

    def test_undocumented_attribute_access_raises_attribute_error(self, mock_env):
        """red-10-round2 (REQUIRED -- the key new assertion): an attribute
        NOT on the documented MaterialX API surface must raise
        AttributeError when accessed on the mock -- proving spec= is
        actually bounding the mock's surface, not merely a curated-but-
        unenforced list of attrs that happen to be pre-set. A bare/unspecced
        MagicMock would auto-vivify `mx.someTypoMethod` as a new, silently
        callable MagicMock instead of raising -- this test proves that
        does NOT happen here.
        """
        mx = mock_env.mx
        with pytest.raises(AttributeError):
            mx.someTypoedNonExistentMaterialXMethod


# ---------------------------------------------------------------------------
# red-11 (Minor): the mixed-alias duplicate-target test names "duplicate"
#                 explicitly in its error assertion.
# ---------------------------------------------------------------------------

class TestMixedAliasDuplicateErrorNamesduplicate:
    """red-11 (Minor): the existing
    test_duplicate_target_via_qualified_and_unqualified_alias_rejected (in
    TestDuplicateEditTargetsRejected above) asserted only ok=False and
    writeToXmlFile-not-called for the mixed-alias case -- it did NOT assert
    the error text names "duplicate" (unlike its sibling
    test_duplicate_target_same_unqualified_name_rejected, which does). This
    class adds that missing assertion as an explicit, standalone regression
    guard so the mixed-alias path cannot silently regress to a generic
    "ambiguous" or "not found" error instead of correctly identifying it as
    a duplicate-target condition.
    """

    def _build_single_node_doc(self):
        node = _FakeMxNode("standard_surface1", inputs={
            "base_color": _FakeMxInput("base_color", "1.0", "color3"),
        })
        ng = _FakeMxNodeGraph("NG_main", nodes={"standard_surface1": node})
        return _FakeMxDocument(nodegraphs={"NG_main": ng})

    def test_mixed_alias_duplicate_error_names_duplicate_explicitly(self, monkeypatch):
        _install_hou_pxr_mocks(monkeypatch)
        _install_materialx_mock(monkeypatch, doc_factory=self._build_single_node_doc)
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "mtlx_edit"):
            pytest.fail("usd_export_handlers.mtlx_edit does not exist yet -- RED (expected).")

        result = mod.mtlx_edit(
            mtlx_path="$HIP/source.mtlx",
            out_path="$HIP/edited.mtlx",
            edits=[
                {"node": "standard_surface1", "input": "base_color", "value": "0.5"},
                {"node": "NG_main/standard_surface1", "input": "base_color", "value": "0.7"},
            ],
        )
        assert result.get("ok") is False
        error_text = str(result.get("error", "")).lower()
        assert "duplicate" in error_text, (
            f"A duplicate target detected via mixed qualified/unqualified "
            f"aliases (both resolving to the SAME (rpath, input)) must "
            f"explicitly name the duplicate-target condition -- not a "
            f"generic 'ambiguous' or 'not found' error. "
            f"Got error={result.get('error')!r}."
        )
