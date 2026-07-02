"""pytest wrapper tests for houdini_cop_onnx_run_inference MCP tool
(PP12-113 PR-4, GATED).

Unit: pp12-113d
testVerificationSurface: pytest-model
planSha: 92ce0bfd3ac81683321af721d8bff6bd50e7c67010fdfca5e82f53f97845adc9

These tests are written BEFORE the implementation (red phase). They will
fail with ImportError until hou-dev implements the new wrapper on
fxhoudinimcp/tools/cop_onnx_tools.py (which already ships
houdini_cop_onnx_list_models / houdini_cop_onnx_inspect_model from PR-2 and
houdini_cop_onnx_setup_node / houdini_cop_onnx_set_provider from PR-3) plus
the corresponding handler on
fxhoudinimcp_server/handlers/cop_onnx_handlers.py.

Grounded against (layer-for-layer template, per plan pp12-113d
reuseSurvey + lockedFieldContract):
  - python/fxhoudinimcp/tests/test_cop_onnx_setup_tools.py (pp12-113c) —
    THE exemplar for a GATED wrapper test: MagicMock(spec=HoudiniBridge);
    import-inside-test-after-patch; require_approval=True assertion via
    the mcp tool_map; exactly-one bridge.execute call; exact params-dict
    key set; VERBATIM passthrough of BOTH a success result AND a
    pending-approval/preview shape (never reinterpreted, never raises,
    never gains/loses an 'ok' key); ctx-not-in-schema guard;
    MagicMock(spec=HoudiniBridge) .call-raises-AttributeError PP12-110
    regression guard.
  - houdini/scripts/python/fxhoudinimcp_server/handlers/cop_onnx_handlers.py
    cop_onnx_setup_node / cop_onnx_set_provider + their _preview_* fns —
    the GATED handler-side shape this file's dispatcher-level contract
    mirrors (not exercised directly here; hou-dev's hython-smoke covers
    the handler).

PP12-110 lessons encoded here (mcp-subprocess-delegation.md /
mcp-fork-build-lessons memory):
  - MagicMock(spec=HoudiniBridge) — a non-existent attr (e.g. .call) raises
    AttributeError; a bare MagicMock() would silently accept it and mask
    the bug.
  - Import the subject module INSIDE each test, AFTER _get_bridge is
    patched (module-level import would bind the tool before the patch is
    active).
  - Assert PUBLIC behavior (bridge.execute cmd + params; verbatim result
    passthrough, INCLUDING a pending-approval shape) — NOT internal call
    order or unrelated attributes.
  - @pytest.mark.asyncio on every async test (the wrapper is an async
    @mcp.tool coroutine).
  - The wrapper makes EXACTLY ONE bridge.execute call — no retries, no
    secondary calls, no result-shape interpretation.
  - Must be require_approval=True (GATED — a cook burns GPU/CPU + can
    touch disk; PP12-109 security gate). A pending-approval bridge
    response is NOT a failure and must survive the wrapper untouched.

Locked field contract (plan pp12-113d lockedFieldContract, "wrapper:
houdini_cop_onnx_run_inference"):

    houdini_cop_onnx_run_inference(
        ctx, node_path, frame=None,
    ) -> dict
        -> a SINGLE bridge.execute('cop_onnx_run_inference', {
               'node_path': node_path, 'frame': frame,
           })

Command string is BARE ('cop_onnx_run_inference') — matching the cop_onnx
family convention (not a 'cops.*' prefix).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import REAL HoudiniBridge for spec= (PP12-110 lesson: bare MagicMock()
# silently accepts .call and any other non-existent attribute, masking
# convention bugs).
from fxhoudinimcp.bridge import HoudiniBridge


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def run_inference_bridge_mock():
    """Spec-bound bridge mock returning a successful gated run_inference result
    (a clean cook — cooked:true + a non-empty output-plane manifest)."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "ok": True,
        "cooked": True,
        "node_path": "/obj/copnet1/agent_onnx",
        "bound_provider": "automatic",
        "cook_ms": 12.5,
        "output_planes": [
            {"name": "output1", "xres": 64, "yres": 64, "channels": 3, "dtype": "float32"},
        ],
        "errors": [],
        "warnings": [],
    })
    return mock


@pytest.fixture()
def run_inference_failed_cook_bridge_mock():
    """Spec-bound bridge mock returning a FR-5 no-silent-success failed-cook
    result -- ok:True, cooked:False, errors surfaced verbatim. A failed cook
    is REPORTED, not raised, and is a normal valid return value."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "ok": True,
        "cooked": False,
        "node_path": "/obj/copnet1/agent_onnx",
        "bound_provider": "automatic",
        "cook_ms": 3.1,
        "output_planes": [],
        "errors": ["src is missing"],
        "warnings": [],
    })
    return mock


@pytest.fixture()
def run_inference_pending_approval_bridge_mock():
    """Spec-bound bridge mock returning a 109-gate pending-approval shape.

    A pending/preview response is NOT a failure — the wrapper must pass it
    through verbatim, not reinterpret it as an error (mirrors the
    pp12-113c / pp12-112d URC-4 regression guard).
    """
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "status": "pending_approval",
        "pending_id": "pend-run-inference-xyz789",
        "preview": {
            "action": "run ONNX inference (cook)",
            "node_path": "/obj/copnet1/agent_onnx",
            "frame": None,
            "bound_provider": "automatic",
            "model_configured": True,
            "node_exists": True,
            "node_is_onnx": True,
        },
    })
    return mock


# ---------------------------------------------------------------------------
# module import + callable (PRIMARY RED GATE)
# ---------------------------------------------------------------------------

class TestCopOnnxRunInferenceToolsModuleImport:
    """The new wrapper must be a callable on the EXISTING cop_onnx_tools
    module (which already ships houdini_cop_onnx_list_models /
    houdini_cop_onnx_inspect_model from PR-2 and
    houdini_cop_onnx_setup_node / houdini_cop_onnx_set_provider from PR-3).
    Until hou-dev adds it, this import raises ImportError — the RED signal
    for this file.
    """

    def test_module_importable(self):
        """cop_onnx_tools must remain importable (pp12-113b/c baseline)."""
        import fxhoudinimcp.tools.cop_onnx_tools  # noqa: F401

    def test_existing_pr2_pr3_wrappers_unaffected(self):
        """houdini_cop_onnx_list_models / houdini_cop_onnx_inspect_model
        (PR-2) and houdini_cop_onnx_setup_node / houdini_cop_onnx_set_provider
        (PR-3) must still be importable — the new wrapper must not clobber
        its existing siblings (append-only contract)."""
        from fxhoudinimcp.tools.cop_onnx_tools import (  # noqa: F401
            houdini_cop_onnx_inspect_model,
            houdini_cop_onnx_list_models,
            houdini_cop_onnx_set_provider,
            houdini_cop_onnx_setup_node,
        )
        assert callable(houdini_cop_onnx_list_models)
        assert callable(houdini_cop_onnx_inspect_model)
        assert callable(houdini_cop_onnx_setup_node)
        assert callable(houdini_cop_onnx_set_provider)

    def test_run_inference_callable_on_module(self):
        """houdini_cop_onnx_run_inference must be a callable (the @mcp.tool
        coroutine). FAILS RED until hou-dev adds the wrapper."""
        from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_run_inference  # noqa: F401
        assert callable(houdini_cop_onnx_run_inference), (
            "houdini_cop_onnx_run_inference must be a callable (the @mcp.tool coroutine)."
        )


# ---------------------------------------------------------------------------
# GATED — meta require_approval=True
# ---------------------------------------------------------------------------

def _get_tool_map():
    from fxhoudinimcp.server import mcp
    tool_map: dict = {}
    if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
        tool_map = mcp._tool_manager._tools
    elif hasattr(mcp, "_tools"):
        tool_map = mcp._tools
    return tool_map


class TestCopOnnxRunInferenceRequireApproval:
    """houdini_cop_onnx_run_inference must have meta={'require_approval': True}.

    run_inference is GATED precisely because a cook burns GPU/CPU + can
    touch disk — it is a state-mutating action per spec §4.1. Every
    shipped gated wrapper (houdini_cop_onnx_setup_node,
    houdini_cop_onnx_set_provider, houdini_usd_export_rop, ...) sets
    require_approval=True; this tool must match that convention exactly.
    """

    def test_mcp_tool_meta_require_approval_true(self):
        import fxhoudinimcp.tools.cop_onnx_tools  # noqa: F401
        tool_map = _get_tool_map()
        assert "houdini_cop_onnx_run_inference" in tool_map, (
            "houdini_cop_onnx_run_inference not registered on the mcp server; "
            "hou-dev must define it (decorated with @mcp.tool) in cop_onnx_tools.py."
        )
        tool_obj = tool_map["houdini_cop_onnx_run_inference"]
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", False) if isinstance(meta, dict) else False
        assert require_approval is True, (
            f"houdini_cop_onnx_run_inference meta must have require_approval=True (GATED -- "
            f"a mutating tool; PP12-109 security gate). Got meta={meta!r}."
        )


# ---------------------------------------------------------------------------
# bridge.execute call contract — houdini_cop_onnx_run_inference
# ---------------------------------------------------------------------------

class TestCopOnnxRunInferenceBridgeContract:
    """houdini_cop_onnx_run_inference must delegate to bridge.execute
    EXACTLY ONCE, with command 'cop_onnx_run_inference' and the exact
    2-key params dict from the lockedFieldContract."""

    @pytest.mark.asyncio
    async def test_exactly_one_bridge_execute_call(self, run_inference_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=run_inference_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_run_inference

            ctx_mock = MagicMock()
            await houdini_cop_onnx_run_inference(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
            )

        assert run_inference_bridge_mock.execute.call_count == 1, (
            f"houdini_cop_onnx_run_inference must make exactly ONE bridge.execute call, "
            f"got {run_inference_bridge_mock.execute.call_count}."
        )

    @pytest.mark.asyncio
    async def test_command_string_is_cop_onnx_run_inference(self, run_inference_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=run_inference_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_run_inference

            ctx_mock = MagicMock()
            await houdini_cop_onnx_run_inference(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
            )

        call_args = run_inference_bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "cop_onnx_run_inference", (
            f"Expected bridge.execute('cop_onnx_run_inference', ...) but got command={command!r}. "
            "Command string must match register_handler's first arg exactly "
            "(the 4-bug convention: command == register name == params keys == handler kwargs)."
        )

    @pytest.mark.asyncio
    async def test_params_dict_has_exact_two_keys(self, run_inference_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=run_inference_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_run_inference

            ctx_mock = MagicMock()
            await houdini_cop_onnx_run_inference(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
                frame=42,
            )

        call_args = run_inference_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        expected_keys = {"node_path", "frame"}
        assert set(params.keys()) == expected_keys, (
            f"params dict must be exactly {expected_keys!r}. Got keys={set(params.keys())!r}."
        )
        assert params["node_path"] == "/obj/copnet1/agent_onnx"
        assert params["frame"] == 42

    @pytest.mark.asyncio
    async def test_frame_defaults_to_none_when_omitted(self, run_inference_bridge_mock):
        """frame is documented to default to None (-> hou.frame() at cook
        time, handler-side) -- it must be forwarded verbatim (not omitted)
        when the caller doesn't pass it."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=run_inference_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_run_inference

            ctx_mock = MagicMock()
            await houdini_cop_onnx_run_inference(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
            )

        call_args = run_inference_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert "frame" in params, (
            f"frame must be present as a key (not omitted) even when the caller "
            f"doesn't pass it. Got keys={set(params.keys())!r}."
        )
        assert params.get("frame") is None, (
            f"frame must default to None. Got {params.get('frame')!r}."
        )


# ---------------------------------------------------------------------------
# verbatim passthrough — success, FAILED-COOK, AND pending-approval shapes
# ---------------------------------------------------------------------------

class TestCopOnnxRunInferenceResultPassthrough:
    """The wrapper must return bridge.execute's result VERBATIM. This
    includes:
      - a clean-cook success result (cooked:true + a non-empty output-plane
        manifest);
      - a FAILED-cook result (ok:True, cooked:False, errors surfaced) --
        FR-5 no-silent-success: a failed cook is REPORTED via a normal
        return value, NOT raised, and the wrapper must not reinterpret it
        as a tool-call failure;
      - the 109-gate pending-approval / preview response shape -- the
        wrapper is a thin single bridge.execute call with NO result
        interpretation. A wrapper that checks `if not result.get("ok"):
        raise` would incorrectly treat a pending-approval response (which
        has no "ok" key at all) as a failure.
    """

    @pytest.mark.asyncio
    async def test_success_result_passed_through_verbatim(self, run_inference_bridge_mock):
        expected_result = run_inference_bridge_mock.execute.return_value
        with patch("fxhoudinimcp.server._get_bridge", return_value=run_inference_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_run_inference

            ctx_mock = MagicMock()
            result = await houdini_cop_onnx_run_inference(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
            )

        assert result == expected_result, (
            "Wrapper must pass bridge.execute's result through unchanged. "
            f"Expected {expected_result!r}, got {result!r}."
        )
        assert result["cooked"] is True
        assert result["output_planes"], "a clean cook must report a non-empty output-plane manifest"
        assert result["errors"] == []

    @pytest.mark.asyncio
    async def test_failed_cook_result_passed_through_not_treated_as_failure(
        self, run_inference_failed_cook_bridge_mock
    ):
        """FR-5 no-silent-success: a failed cook (ok:True, cooked:False,
        errors:[...]) is a NORMAL valid return value -- the wrapper must
        NOT raise, must NOT reinterpret it, and must NOT drop the errors
        list."""
        expected_result = run_inference_failed_cook_bridge_mock.execute.return_value
        with patch(
            "fxhoudinimcp.server._get_bridge",
            return_value=run_inference_failed_cook_bridge_mock,
        ):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_run_inference

            ctx_mock = MagicMock()
            # Must NOT raise -- a failed-cook response (ok:True, cooked:False)
            # is a normal, valid return value from bridge.execute.
            result = await houdini_cop_onnx_run_inference(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
            )

        assert result == expected_result, (
            "A failed-cook bridge response must be returned VERBATIM -- "
            "the wrapper must not reinterpret, unwrap, or swallow it. "
            f"Expected {expected_result!r}, got {result!r}."
        )
        assert result["ok"] is True, (
            "A failed cook is ok:True + cooked:False (reported, not raised, not ok:False)."
        )
        assert result["cooked"] is False
        assert result["errors"] == ["src is missing"], (
            f"the cook error(s) must be surfaced verbatim, got {result['errors']!r}."
        )

    @pytest.mark.asyncio
    async def test_pending_approval_result_passed_through_not_treated_as_failure(
        self, run_inference_pending_approval_bridge_mock
    ):
        expected_result = run_inference_pending_approval_bridge_mock.execute.return_value
        with patch(
            "fxhoudinimcp.server._get_bridge",
            return_value=run_inference_pending_approval_bridge_mock,
        ):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_run_inference

            ctx_mock = MagicMock()
            # Must NOT raise -- a pending-approval response is a normal,
            # valid return value from bridge.execute.
            result = await houdini_cop_onnx_run_inference(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
            )

        assert result == expected_result, (
            "A pending-approval bridge response must be returned VERBATIM -- "
            "the wrapper must not reinterpret, unwrap, or swallow it. "
            f"Expected {expected_result!r}, got {result!r}."
        )
        assert result.get("status") == "pending_approval"
        assert "ok" not in result, (
            f"Wrapper must not inject an 'ok' key into a pending-approval response. "
            f"Got keys={list(result.keys())!r}."
        )


# ---------------------------------------------------------------------------
# spec-bound bridge mock — PP12-110 regression guard
# ---------------------------------------------------------------------------

class TestBridgeSpecBoundGuard:
    """MagicMock(spec=HoudiniBridge) — .call raises AttributeError (PP12-110 guard).

    In PP12-110, a wrapper called bridge.call(...) instead of
    bridge.execute(...). A bare MagicMock() silently accepted .call and
    returned a mock — masking the bug. A spec-bound mock raises
    AttributeError because .call is not a real method on HoudiniBridge.
    """

    def test_bridge_call_attribute_does_not_exist(self):
        mock = MagicMock(spec=HoudiniBridge)
        with pytest.raises(AttributeError, match="call"):
            _ = mock.call  # .call must NOT exist on HoudiniBridge

    def test_bridge_execute_attribute_exists(self):
        mock = MagicMock(spec=HoudiniBridge)
        _ = mock.execute  # must NOT raise — execute IS a real method


# ---------------------------------------------------------------------------
# ctx not in tool input schema (ctx is injected, not a tool parameter)
# ---------------------------------------------------------------------------

class TestCtxSchemaGuard:
    """ctx must NOT appear in the new tool's input schema properties.

    FastMCP injects ctx via the Context type annotation — it is NOT a
    parameter the MCP client sends.
    """

    def _get_tool_schema(self, tool_name: str) -> dict:
        tool_map = _get_tool_map()
        tool_obj = tool_map.get(tool_name)
        if tool_obj is None:
            return {}
        return getattr(tool_obj, "parameters", getattr(tool_obj, "inputSchema", {})) or {}

    def test_run_inference_ctx_not_in_schema(self):
        import fxhoudinimcp.tools.cop_onnx_tools  # noqa: F401
        schema = self._get_tool_schema("houdini_cop_onnx_run_inference")
        properties = schema.get("properties", {})
        assert "ctx" not in properties, (
            f"'ctx' must NOT appear in houdini_cop_onnx_run_inference's input schema properties. "
            f"Got properties={list(properties.keys())!r}."
        )
