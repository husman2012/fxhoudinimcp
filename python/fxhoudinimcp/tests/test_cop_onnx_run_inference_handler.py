"""Handler-level (mocked-hou) pytest tests for cop_onnx_run_inference
(PP12-113 PR-4, GATED) — codex Major-4 mandatory red coverage.

Unit: pp12-113d
testVerificationSurface: pytest-model
planSha: 92ce0bfd3ac81683321af721d8bff6bd50e7c67010fdfca5e82f53f97845adc9

These tests import the HANDLER module (cop_onnx_handlers.py) with a mocked
`hou` module installed into sys.modules BEFORE import, per
test-fixture-conventions.md §2.3 (`monkeypatch.setitem(sys.modules, 'hou',
fake)`). This lets the handler's Python control flow be exercised off-DCC
without a real Houdini session — proving the FR-5 no-silent-success cook
algorithm order (capture cook_exc; read errors; fold str(cook_exc) when
errors is empty; THEN cooked=cooked_from_errors(errors)) using MagicMock
fakes for the hou.Node / hou.OperationFailed surface.

This is a genuinely mocked-hou rung (test-fixture-conventions.md §2 — the
handler cannot be fully split into *_model.py because it calls
hou.node()/node.cook()/node.errors() as its core job), NOT a substitute
for the real hython-smoke (cop_onnx_run_inference_hython_smoke.py), which
remains the authority on real cook behavior against a real Houdini
session.

PLAN-REVIEW FOLD (rev2) coverage this file pins (codex Major-3/Major-4):
  - M3-cook-order-footgun: the LOCKED handler algorithm order --
    capture cook_exc; read errors AFTER cook (whether or not it raised);
    if cook_exc is not None and not errors: errors=[str(cook_exc)]; THEN
    cooked=cooked_from_errors(errors). A raised cook is NEVER reported
    cooked:true.
  - M4-red-coverage (MANDATORY): a handler-level double where node.cook()
    RAISES hou.OperationFailed AND node.errors()==[] -- asserts ok:True,
    cooked:False, errors:[str(exc)] (the raised-but-empty-errors branch).
  - A second double: cook() RAISES AND node.errors() is non-empty --
    cooked:False + those (real) errors, NOT the exception string.
  - A third double: a clean cook (no raise, errors []) with a fake node
    exposing outputNames()/layer(i) (bufferResolution/channelCount/
    storageType) -- cooked:True + a non-empty output_planes manifest.
  - B1-target-validation: a fake node whose type().category() is NOT the
    Cop category -> ok:False (folds codex Blocker-1 -- name+category
    check, not name-only).

Assertions target the RETURNED DICT (the public contract) -- never which
hou.* methods were called or in what order (tdd-with-agents.md §2 mirror-
test ban; test-fixture-conventions.md §2.3 discipline). The ONE exception
is the target-validation test, which necessarily inspects what fake attrs
the handler reads off node.type() -- but even there the assertion is on
the RETURNED ok:False, not on a call-count/call-order assertion.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# mock_hou fixture — installs a fake `hou` module into sys.modules BEFORE
# the handler module is imported (test-fixture-conventions.md §2.3).
# ---------------------------------------------------------------------------

class _FakeOperationFailed(Exception):
    """Stand-in for hou.OperationFailed — a plain Exception subclass is
    sufficient since the handler only does `except hou.OperationFailed as e`
    and `raise hou.OperationFailed(...)`."""


@pytest.fixture()
def mock_hou(monkeypatch):
    """Install a mock `hou` module so cop_onnx_handlers.py loads off-DCC.

    Returns the mock so tests can configure hou.node.return_value, etc.
    hou.OperationFailed is set to a REAL Exception subclass (not a
    MagicMock) so `except hou.OperationFailed` / `raise
    hou.OperationFailed(...)` behave like real exception handling.
    hou.copNodeTypeCategory / hou.cop2NodeTypeCategory both resolve to a
    sentinel MagicMock so the handler's category-equality check
    (`node.type().category() == hou.copNodeTypeCategory()`) can be pinned
    per-test by setting the fake node's type().category() return value to
    either the sentinel (Cop match) or a distinct sentinel (mismatch).
    """
    fake = MagicMock(name="hou")
    fake.OperationFailed = _FakeOperationFailed
    monkeypatch.setitem(sys.modules, "hou", fake)

    # Ensure the fork's non-standard package roots are importable (mirrors
    # the sys.path bootstrap the shipped exemplar
    # test_usd_export_rop_handler_edge_cases.py performs): pytest's rootdir
    # discovery does not put houdini/scripts/python (fxhoudinimcp_server's
    # home) on sys.path by default.
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


def _make_fake_onnx_node(cop_category_sentinel):
    """A fake hou.Node whose type().name()=='onnx' and
    type().category()==cop_category_sentinel (the Cop-category match)."""
    node = MagicMock(name="fake_onnx_node")
    node.type.return_value.name.return_value = "onnx"
    node.type.return_value.category.return_value = cop_category_sentinel
    node.path.return_value = "/obj/copnet1/agent_onnx"
    return node


# ---------------------------------------------------------------------------
# PRIMARY RED GATE — the handler function must exist
# ---------------------------------------------------------------------------

class TestHandlerImport:
    def test_run_inference_handler_importable(self, mock_hou):
        """cop_onnx_run_inference must be importable from cop_onnx_handlers.
        FAILS RED (ImportError/AttributeError) until hou-dev implements it."""
        handlers = _import_handler_module()
        assert hasattr(handlers, "cop_onnx_run_inference"), (
            "cop_onnx_handlers.py must expose cop_onnx_run_inference."
        )
        assert callable(handlers.cop_onnx_run_inference)

    def test_preview_run_inference_handler_importable(self, mock_hou):
        """_preview_run_inference must be importable from cop_onnx_handlers.
        FAILS RED until hou-dev implements it."""
        handlers = _import_handler_module()
        assert hasattr(handlers, "_preview_run_inference"), (
            "cop_onnx_handlers.py must expose _preview_run_inference."
        )
        assert callable(handlers._preview_run_inference)


# ---------------------------------------------------------------------------
# M4-red-coverage (MANDATORY, codex Major-4): raised cook + empty errors()
# ---------------------------------------------------------------------------

class TestRaisedCookEmptyErrors:
    """cook() RAISES hou.OperationFailed AND node.errors() returns [] ->
    the handler MUST fold str(cook_exc) into errors -- a raised cook is
    NEVER reported cooked:true. Asserts ok:True, cooked:False,
    errors:[str(exc)]."""

    def test_raised_cook_empty_errors_folds_exception_string(self, mock_hou):
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)

        exc = mock_hou.OperationFailed("model input mismatch")
        node.cook.side_effect = exc
        node.errors.return_value = []
        node.warnings.return_value = []
        node.outputNames.return_value = ()

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_run_inference(node_path="/obj/copnet1/agent_onnx", frame=None)

        assert result["ok"] is True, (
            f"a failed cook is a bad COOK, not a bad TARGET -- must be ok:True, got {result!r}"
        )
        assert result["cooked"] is False, (
            "a raised cook must NEVER be reported cooked:true"
        )
        assert result["errors"] == [str(exc)], (
            f"when the cook RAISES and node.errors() is empty, the handler must fold "
            f"str(cook_exc) into errors -- got errors={result.get('errors')!r}"
        )


# ---------------------------------------------------------------------------
# Second double: cook() RAISES AND node.errors() is non-empty
# ---------------------------------------------------------------------------

class TestRaisedCookNonEmptyErrors:
    """cook() RAISES AND node.errors() returns real errors -> cooked:False
    + THOSE errors (not the exception string) -- errors() is authoritative
    when non-empty."""

    def test_raised_cook_nonempty_errors_uses_real_errors(self, mock_hou):
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)

        exc = mock_hou.OperationFailed("cook failed")
        node.cook.side_effect = exc
        node.errors.return_value = ["src is missing"]
        node.warnings.return_value = []
        node.outputNames.return_value = ()

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_run_inference(node_path="/obj/copnet1/agent_onnx", frame=None)

        assert result["ok"] is True
        assert result["cooked"] is False
        assert result["errors"] == ["src is missing"], (
            f"when node.errors() is non-empty, it is authoritative -- must NOT be "
            f"replaced/mixed with the exception string. Got {result['errors']!r}."
        )


# ---------------------------------------------------------------------------
# Third double: a clean cook (no raise, errors []) with a real manifest
# ---------------------------------------------------------------------------

class TestCleanCook:
    """A clean cook (no raise, errors []) with a fake node exposing
    outputNames()/layer(i) (bufferResolution/channelCount/storageType) ->
    cooked:True + a non-empty output_planes manifest."""

    def test_clean_cook_returns_cooked_true_and_manifest(self, mock_hou):
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)

        node.cook.return_value = None  # no raise -- clean cook
        node.errors.return_value = []
        node.warnings.return_value = []
        node.outputNames.return_value = ("output1",)

        layer = MagicMock(name="fake_image_layer")
        layer.bufferResolution.return_value = (64, 64)
        layer.channelCount.return_value = 3
        layer.storageType.return_value = "imageLayerStorageType.Float32"
        node.layer.return_value = layer

        provider_parm = MagicMock(name="provider_parm")
        provider_parm.evalAsString.return_value = "automatic"
        node.parm.return_value = provider_parm

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_run_inference(node_path="/obj/copnet1/agent_onnx", frame=None)

        assert result["ok"] is True
        assert result["cooked"] is True, (
            f"a clean cook (no raise, errors []) must be cooked:True, got {result!r}"
        )
        assert result["errors"] == []
        output_planes = result["output_planes"]
        assert output_planes, "a clean cook must report a non-empty output_planes manifest"
        plane = output_planes[0]
        assert plane["name"] == "output1"
        assert plane["xres"] == 64
        assert plane["yres"] == 64
        assert plane["channels"] == 3
        assert plane["dtype"] == "float32", (
            f"dtype must be normalized via normalize_plane_dtype, got {plane['dtype']!r}"
        )
        assert result["bound_provider"] == "automatic", (
            f"bound_provider must be read via evalAsString() (the token), "
            f"got {result['bound_provider']!r}"
        )


# ---------------------------------------------------------------------------
# GREEN-REVIEW LOCKING TEST 1 (codex threadId 019f2059, Major): frame must
# be honored -- node.cook() must be called with a frame_range derived from
# the REQUESTED frame, not the current scene frame.
# ---------------------------------------------------------------------------

class TestFrameIsHonored:
    """Cooking at the REQUESTED frame IS the contract (the `frame` param
    exists precisely so a caller can cook a specific frame without first
    moving the playhead) -- a call-assertion is legitimate here per
    test-fixture-conventions.md §2.3 ("call-assertions only when the call
    IS the contract"). GREEN-REVIEW found the shipped impl computes
    frame_to_cook but never passes it to node.cook() -- a non-None frame
    silently cooks whatever the CURRENT scene frame happens to be instead
    of the requested one."""

    def test_explicit_frame_is_passed_to_cook_as_frame_range(self, mock_hou):
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)

        node.cook.return_value = None  # no raise -- clean cook
        node.errors.return_value = []
        node.warnings.return_value = []
        node.outputNames.return_value = ()

        provider_parm = MagicMock(name="provider_parm")
        provider_parm.evalAsString.return_value = "automatic"
        node.parm.return_value = provider_parm

        mock_hou.node.return_value = node

        handlers.cop_onnx_run_inference(node_path="/obj/copnet1/agent_onnx", frame=5)

        assert node.cook.call_count == 1, (
            f"cop_onnx_run_inference must cook exactly once, "
            f"got {node.cook.call_count} calls: {node.cook.call_args_list!r}"
        )
        _args, kwargs = node.cook.call_args
        assert kwargs.get("force") is True, (
            f"node.cook must still be called with force=True, got kwargs={kwargs!r}"
        )
        assert kwargs.get("frame_range") == (5, 5), (
            f"a REQUESTED frame=5 must be honored -- node.cook must be called with "
            f"frame_range=(5, 5) so the cook runs at the requested frame, not "
            f"whatever the current scene frame happens to be. Got kwargs={kwargs!r} "
            f"(call_args={node.cook.call_args!r})."
        )

    def test_frame_none_cooks_at_the_current_scene_frame(self, mock_hou):
        """frame=None must resolve to hou.frame() (the current scene frame)
        and that SAME value must be what node.cook() is told to cook at --
        proven by mocking hou.frame() to a distinctive, non-default value
        and asserting the cook call's frame_range is consistent with it
        (not silently cooking some OTHER frame)."""
        handlers = _import_handler_module()

        mock_hou.frame.return_value = 42.0

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)

        node.cook.return_value = None
        node.errors.return_value = []
        node.warnings.return_value = []
        node.outputNames.return_value = ()

        provider_parm = MagicMock(name="provider_parm")
        provider_parm.evalAsString.return_value = "automatic"
        node.parm.return_value = provider_parm

        mock_hou.node.return_value = node

        handlers.cop_onnx_run_inference(node_path="/obj/copnet1/agent_onnx", frame=None)

        assert node.cook.call_count == 1
        _args, kwargs = node.cook.call_args
        assert kwargs.get("force") is True
        assert kwargs.get("frame_range") == (42.0, 42.0), (
            f"frame=None must cook at hou.frame() (mocked here to 42.0) -- "
            f"node.cook must be called with frame_range=(42.0, 42.0), consistent "
            f"with the resolved current-frame value, not an unrelated/default "
            f"frame. Got kwargs={kwargs!r} (call_args={node.cook.call_args!r})."
        )


# ---------------------------------------------------------------------------
# GREEN-REVIEW LOCKING TEST 2 (codex threadId 019f2059, Major): a failed
# cook must deterministically report output_planes == [] -- NEVER read
# outputNames()/layer(i) when cooked is False, regardless of what stale
# data those methods would return post-failure.
# ---------------------------------------------------------------------------

class TestFailedCookOutputPlanesEmpty:
    """failed cook -> output_planes == [] is a DETERMINISTIC contract, not
    an incidental "happens to be empty because outputNames() returns
    nothing on a failed node" behavior. GREEN-REVIEW found the shipped
    impl reads node.outputNames()/node.layer(i) UNCONDITIONALLY (no `if
    cooked:` guard) -- on a real Houdini node this is usually harmless
    (a failed cook usually yields no readable outputs), but it is not
    GUARANTEED, and the manifest-read loop's per-plane try/except means a
    stale or leftover outputNames() entry post-failure would silently
    populate output_planes on a cook the caller was told failed. This
    fixes that non-determinism by making a fake node whose outputNames()
    WOULD return a real entry even though the cook failed, and pinning
    that the handler must still report output_planes == []."""

    def test_raised_cook_with_stale_output_names_still_reports_empty_planes(self, mock_hou):
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)

        exc = mock_hou.OperationFailed("cook failed")
        node.cook.side_effect = exc
        node.errors.return_value = ["src is missing"]
        node.warnings.return_value = []

        # outputNames() WOULD return a real, non-empty entry -- e.g. stale
        # data left over from a prior successful cook of this same node
        # object, or a node type that reports its declared output slots
        # even when unconfigured. The handler must NOT read this when the
        # cook failed.
        node.outputNames.return_value = ("output1",)
        layer = MagicMock(name="fake_image_layer_poststale")
        layer.bufferResolution.return_value = (64, 64)
        layer.channelCount.return_value = 3
        layer.storageType.return_value = "imageLayerStorageType.Float32"
        node.layer.return_value = layer

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_run_inference(node_path="/obj/copnet1/agent_onnx", frame=None)

        assert result["ok"] is True
        assert result["cooked"] is False, (
            f"a raised cook must be cooked:False, got {result!r}"
        )
        assert result["output_planes"] == [], (
            f"a FAILED cook must DETERMINISTICALLY report output_planes == [] -- "
            f"the handler must guard the output-plane manifest read behind "
            f"`if cooked:` and never read outputNames()/layer(i) when cooked is "
            f"False, regardless of what those methods would otherwise return. "
            f"Got output_planes={result['output_planes']!r}."
        )

    def test_errors_present_no_raise_still_reports_empty_planes(self, mock_hou):
        """Same contract via the errors()-only failure path (no raise, but
        node.errors() is non-empty -- also cooked:False)."""
        handlers = _import_handler_module()

        cop_sentinel = mock_hou.copNodeTypeCategory.return_value
        node = _make_fake_onnx_node(cop_sentinel)

        node.cook.return_value = None  # no raise
        node.errors.return_value = ["shape mismatch"]
        node.warnings.return_value = []

        node.outputNames.return_value = ("output1",)
        layer = MagicMock(name="fake_image_layer_poststale2")
        layer.bufferResolution.return_value = (64, 64)
        layer.channelCount.return_value = 3
        layer.storageType.return_value = "imageLayerStorageType.Float32"
        node.layer.return_value = layer

        mock_hou.node.return_value = node

        result = handlers.cop_onnx_run_inference(node_path="/obj/copnet1/agent_onnx", frame=None)

        assert result["ok"] is True
        assert result["cooked"] is False
        assert result["output_planes"] == [], (
            f"a cook reporting non-empty errors() (cooked:False) must "
            f"DETERMINISTICALLY report output_planes == [], regardless of what "
            f"outputNames()/layer(i) would otherwise return. "
            f"Got output_planes={result['output_planes']!r}."
        )


# ---------------------------------------------------------------------------
# B1-target-validation: a non-Cop-category node -> ok:False
# ---------------------------------------------------------------------------

class TestTargetValidation:
    """Target validation requires BOTH name=='onnx' AND
    type().category()==hou.copNodeTypeCategory() (folds codex Blocker-1 --
    a bare name-only check would let an unrelated Sop/onnx node through)."""

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

        result = handlers.cop_onnx_run_inference(node_path="/obj/geo1/onnx1", frame=None)

        assert result["ok"] is False, (
            f"a node named 'onnx' but NOT of the Cop category must be REJECTED "
            f"(ok:False) -- a name-only check is insufficient (codex Blocker-1). "
            f"Got {result!r}."
        )

    def test_unresolved_node_path_is_ok_false(self, mock_hou):
        handlers = _import_handler_module()

        mock_hou.node.return_value = None

        result = handlers.cop_onnx_run_inference(node_path="/obj/does_not_exist", frame=None)

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

        result = handlers.cop_onnx_run_inference(node_path="/obj/copnet1", frame=None)

        assert result["ok"] is False, (
            f"a node whose type().name() != 'onnx' must be ok:False, got {result!r}"
        )
