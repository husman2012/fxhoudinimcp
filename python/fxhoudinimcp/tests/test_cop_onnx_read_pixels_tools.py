"""pytest wrapper tests for houdini_cop_onnx_read_pixels MCP tool
(PP12-113 PR-5, the FINAL tool of member 113, READONLY / UNGATED).

Unit: pp12-113e
testVerificationSurface: pytest-model
planSha: bc1f8d9e9fcd1b46216e3eab594ded4a31ee907e305e5d12fde0b040314c445c

These tests are written BEFORE the implementation (red phase). They will
fail with ImportError until hou-dev implements the new wrapper on
fxhoudinimcp/tools/cop_onnx_tools.py (which already ships
houdini_cop_onnx_list_models / houdini_cop_onnx_inspect_model [PR-2,
READONLY] and houdini_cop_onnx_setup_node / houdini_cop_onnx_set_provider
/ houdini_cop_onnx_run_inference [PR-3/PR-4, GATED]) plus the
corresponding handler on
fxhoudinimcp_server/handlers/cop_onnx_handlers.py.

Grounded against (layer-for-layer template, per plan pp12-113e
reuseSurvey):
  - python/fxhoudinimcp/tests/test_cop_onnx_tools.py (pp12-113b) — THE
    exemplar for a READONLY wrapper test: MagicMock(spec=HoudiniBridge);
    import-inside-test-after-patch; require_approval=False assertion via
    the mcp tool_map; exactly-one bridge.execute call; exact params-dict
    key set; VERBATIM passthrough; ctx-not-in-schema guard;
    MagicMock(spec=HoudiniBridge) .call-raises-AttributeError PP12-110
    regression guard. read_pixels mirrors this READONLY shape exactly
    (NOT the GATED pending-approval pattern from test_cop_onnx_setup_tools.py
    / test_cop_onnx_run_inference_tools.py — read_pixels has NO preview_fn,
    NO pending_approval shape to pass through).
  - houdini/scripts/python/fxhoudinimcp_server/handlers/cop_onnx_handlers.py
    cop_onnx_inspect_model + register_handler(..., Capability.READONLY) —
    the READONLY handler-side shape this file's dispatcher-level contract
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
    passthrough) — NOT internal call order or unrelated attributes.
  - @pytest.mark.asyncio on every async test (the wrapper is an async
    @mcp.tool coroutine).
  - The wrapper makes EXACTLY ONE bridge.execute call — no retries, no
    secondary calls, no result-shape interpretation.
  - Must be require_approval=False (READONLY — genuinely read-only per
    the grounded probe: node.layer(idx) returns None on a stale node, no
    auto-cook. read_pixels REPORTS 'not cooked', never mutates/cooks).

Locked field contract (plan pp12-113e lockedFieldContract, "wrapper:
houdini_cop_onnx_read_pixels"):

    houdini_cop_onnx_read_pixels(
        ctx, node_path, plane=None, mode='summary', roi=None,
        max_pixels=4096, downsample=None, page=0, page_size=1024,
    ) -> dict
        -> a SINGLE bridge.execute('cop_onnx_read_pixels', {
               'node_path': node_path, 'plane': plane, 'mode': mode,
               'roi': roi, 'max_pixels': max_pixels,
               'downsample': downsample, 'page': page,
               'page_size': page_size,
           })

Command string is BARE ('cop_onnx_read_pixels') — matching the cop_onnx
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
def read_pixels_summary_bridge_mock():
    """Spec-bound bridge mock returning a minimal summary-mode result."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "ok": True,
        "cooked": True,
        "mode": "summary",
        "plane": "output1",
        "xres": 64,
        "yres": 64,
        "channels": 3,
        "dtype": "float32",
        "stats": {
            "min": [0.0, 0.0, 0.0],
            "max": [1.0, 1.0, 1.0],
            "mean": [0.5, 0.5, 0.5],
            "nan_count": 0,
            "inf_count": 0,
        },
        "histogram": {"bins": 32, "counts": [[0] * 32, [0] * 32, [0] * 32]},
        "sampled": True,
        "stride": 2,
    })
    return mock


@pytest.fixture()
def read_pixels_not_cooked_bridge_mock():
    """Spec-bound bridge mock returning the read-only 'not cooked' result
    (a stale/uncooked node -- NOT an error, a valid reportable outcome)."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "ok": True,
        "cooked": False,
        "node_path": "/obj/copnet1/agent_onnx",
        "plane": None,
        "message": "node not cooked — run cop_onnx_run_inference first",
    })
    return mock


@pytest.fixture()
def read_pixels_error_bridge_mock():
    """Spec-bound bridge mock returning an FR-5 error shape (e.g. a bad
    plane name or wrong-category node)."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "ok": False,
        "error": "plane 'nope' not found; available: ['output1']",
    })
    return mock


# ---------------------------------------------------------------------------
# module import + callable (PRIMARY RED GATE)
# ---------------------------------------------------------------------------

class TestCopOnnxReadPixelsToolsModuleImport:
    """The new wrapper must be a callable on the EXISTING cop_onnx_tools
    module (which already ships PR-2/PR-3/PR-4 wrappers). Until hou-dev
    adds it, this import raises ImportError — the RED signal for this
    file.
    """

    def test_module_importable(self):
        """cop_onnx_tools must remain importable (append-only baseline)."""
        import fxhoudinimcp.tools.cop_onnx_tools  # noqa: F401

    def test_existing_pr2_pr3_pr4_wrappers_unaffected(self):
        """All 5 existing wrappers (PR-2 READONLY x2, PR-3/PR-4 GATED x3)
        must still be importable — the new wrapper must not clobber its
        existing siblings (append-only contract)."""
        from fxhoudinimcp.tools.cop_onnx_tools import (  # noqa: F401
            houdini_cop_onnx_inspect_model,
            houdini_cop_onnx_list_models,
            houdini_cop_onnx_run_inference,
            houdini_cop_onnx_set_provider,
            houdini_cop_onnx_setup_node,
        )
        assert callable(houdini_cop_onnx_list_models)
        assert callable(houdini_cop_onnx_inspect_model)
        assert callable(houdini_cop_onnx_setup_node)
        assert callable(houdini_cop_onnx_set_provider)
        assert callable(houdini_cop_onnx_run_inference)

    def test_read_pixels_callable_on_module(self):
        """houdini_cop_onnx_read_pixels must be a callable (the @mcp.tool
        coroutine). FAILS RED until hou-dev adds the wrapper."""
        from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_read_pixels  # noqa: F401
        assert callable(houdini_cop_onnx_read_pixels), (
            "houdini_cop_onnx_read_pixels must be a callable (the @mcp.tool coroutine)."
        )


# ---------------------------------------------------------------------------
# READONLY — meta require_approval=False on the new wrapper
# ---------------------------------------------------------------------------

def _get_tool_map():
    from fxhoudinimcp.server import mcp
    tool_map: dict = {}
    if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
        tool_map = mcp._tool_manager._tools
    elif hasattr(mcp, "_tools"):
        tool_map = mcp._tools
    return tool_map


class TestCopOnnxReadPixelsRequireApproval:
    """houdini_cop_onnx_read_pixels must have meta={'require_approval': False}.

    This is a READONLY tool (like PR-2's inspect/list) — read_pixels
    reads node.layer(idx); a stale/uncooked node returns None (no
    auto-cook) and the handler REPORTS 'not cooked' rather than mutating
    or cooking anything. Unlike PR-3/PR-4's GATED tools, this tool has NO
    preview_fn and NO 109-gate pending-approval shape.
    """

    def test_mcp_tool_meta_require_approval_false(self):
        import fxhoudinimcp.tools.cop_onnx_tools  # noqa: F401
        tool_map = _get_tool_map()
        assert "houdini_cop_onnx_read_pixels" in tool_map, (
            "houdini_cop_onnx_read_pixels not registered on the mcp server; "
            "hou-dev must define it (decorated with @mcp.tool) in cop_onnx_tools.py."
        )
        tool_obj = tool_map["houdini_cop_onnx_read_pixels"]
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", True) if isinstance(meta, dict) else True
        assert require_approval is False, (
            f"houdini_cop_onnx_read_pixels meta must have require_approval=False (READONLY -- "
            f"genuinely read-only; a stale node is REPORTED not cooked, never mutated). "
            f"Got meta={meta!r}."
        )


# ---------------------------------------------------------------------------
# bridge.execute call contract — houdini_cop_onnx_read_pixels
# ---------------------------------------------------------------------------

class TestCopOnnxReadPixelsBridgeContract:
    """houdini_cop_onnx_read_pixels must delegate to bridge.execute EXACTLY
    ONCE, with command 'cop_onnx_read_pixels' and the exact 8-key params
    dict from the lockedFieldContract."""

    @pytest.mark.asyncio
    async def test_exactly_one_bridge_execute_call(self, read_pixels_summary_bridge_mock):
        with patch(
            "fxhoudinimcp.server._get_bridge", return_value=read_pixels_summary_bridge_mock
        ):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_read_pixels

            ctx_mock = MagicMock()
            await houdini_cop_onnx_read_pixels(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
            )

        assert read_pixels_summary_bridge_mock.execute.call_count == 1, (
            f"houdini_cop_onnx_read_pixels must make exactly ONE bridge.execute call, "
            f"got {read_pixels_summary_bridge_mock.execute.call_count}."
        )

    @pytest.mark.asyncio
    async def test_command_string_is_cop_onnx_read_pixels(self, read_pixels_summary_bridge_mock):
        with patch(
            "fxhoudinimcp.server._get_bridge", return_value=read_pixels_summary_bridge_mock
        ):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_read_pixels

            ctx_mock = MagicMock()
            await houdini_cop_onnx_read_pixels(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
            )

        call_args = read_pixels_summary_bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "cop_onnx_read_pixels", (
            f"Expected bridge.execute('cop_onnx_read_pixels', ...) but got command={command!r}. "
            "Command string must match register_handler's first arg exactly "
            "(the 4-bug convention: command == register name == params keys == handler kwargs)."
        )

    @pytest.mark.asyncio
    async def test_params_dict_has_exact_eight_keys(self, read_pixels_summary_bridge_mock):
        with patch(
            "fxhoudinimcp.server._get_bridge", return_value=read_pixels_summary_bridge_mock
        ):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_read_pixels

            ctx_mock = MagicMock()
            await houdini_cop_onnx_read_pixels(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
                plane="output1",
                mode="roi",
                roi=[0, 0, 16, 16],
                max_pixels=2048,
                downsample=2,
                page=1,
                page_size=512,
            )

        call_args = read_pixels_summary_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        expected_keys = {
            "node_path", "plane", "mode", "roi",
            "max_pixels", "downsample", "page", "page_size",
        }
        assert set(params.keys()) == expected_keys, (
            f"params dict must be exactly {expected_keys!r}. Got keys={set(params.keys())!r}."
        )
        assert params["node_path"] == "/obj/copnet1/agent_onnx"
        assert params["plane"] == "output1"
        assert params["mode"] == "roi"
        assert params["roi"] == [0, 0, 16, 16]
        assert params["max_pixels"] == 2048
        assert params["downsample"] == 2
        assert params["page"] == 1
        assert params["page_size"] == 512

    @pytest.mark.asyncio
    async def test_defaults_forwarded_when_omitted(self, read_pixels_summary_bridge_mock):
        """plane=None, mode='summary', roi=None, max_pixels=4096,
        downsample=None, page=0, page_size=1024 are the documented
        defaults — they must be forwarded verbatim (not omitted) when the
        caller doesn't pass them."""
        with patch(
            "fxhoudinimcp.server._get_bridge", return_value=read_pixels_summary_bridge_mock
        ):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_read_pixels

            ctx_mock = MagicMock()
            await houdini_cop_onnx_read_pixels(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
            )

        call_args = read_pixels_summary_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert params.get("plane") is None, (
            f"plane must default to None. Got {params.get('plane')!r}."
        )
        assert params.get("mode") == "summary", (
            f"mode must default to 'summary'. Got {params.get('mode')!r}."
        )
        assert params.get("roi") is None, (
            f"roi must default to None. Got {params.get('roi')!r}."
        )
        assert params.get("max_pixels") == 4096, (
            f"max_pixels must default to 4096. Got {params.get('max_pixels')!r}."
        )
        assert params.get("downsample") is None, (
            f"downsample must default to None. Got {params.get('downsample')!r}."
        )
        assert params.get("page") == 0, (
            f"page must default to 0. Got {params.get('page')!r}."
        )
        assert params.get("page_size") == 1024, (
            f"page_size must default to 1024. Got {params.get('page_size')!r}."
        )


# ---------------------------------------------------------------------------
# verbatim passthrough — success, not-cooked, and error shapes
# ---------------------------------------------------------------------------

class TestCopOnnxReadPixelsResultPassthrough:
    """The wrapper must return bridge.execute's result VERBATIM in every
    shape: a summary-mode success result, the READ-ONLY 'not cooked'
    result (ok:True, cooked:False -- a normal, valid outcome, NOT an
    error), and an FR-5 error shape. The wrapper is a thin single
    bridge.execute call with NO result interpretation."""

    @pytest.mark.asyncio
    async def test_summary_result_passed_through_verbatim(self, read_pixels_summary_bridge_mock):
        expected_result = read_pixels_summary_bridge_mock.execute.return_value
        with patch(
            "fxhoudinimcp.server._get_bridge", return_value=read_pixels_summary_bridge_mock
        ):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_read_pixels

            ctx_mock = MagicMock()
            result = await houdini_cop_onnx_read_pixels(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
            )

        assert result == expected_result, (
            "Wrapper must pass bridge.execute's result through unchanged. "
            f"Expected {expected_result!r}, got {result!r}."
        )

    @pytest.mark.asyncio
    async def test_not_cooked_result_passed_through_not_treated_as_failure(
        self, read_pixels_not_cooked_bridge_mock
    ):
        """A stale/uncooked node's 'not cooked' result is ok:True,
        cooked:False -- a valid, reportable outcome. The wrapper must NOT
        raise, reinterpret, or mutate it into an error shape (mirrors the
        run_inference failed-cook URC-4 regression guard, applied here to
        the read-only not-cooked path)."""
        expected_result = read_pixels_not_cooked_bridge_mock.execute.return_value
        with patch(
            "fxhoudinimcp.server._get_bridge", return_value=read_pixels_not_cooked_bridge_mock
        ):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_read_pixels

            ctx_mock = MagicMock()
            # Must NOT raise -- a not-cooked response is a normal, valid
            # return value from bridge.execute.
            result = await houdini_cop_onnx_read_pixels(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
            )

        assert result == expected_result, (
            "A not-cooked bridge response must be returned VERBATIM -- "
            "the wrapper must not reinterpret, unwrap, or swallow it. "
            f"Expected {expected_result!r}, got {result!r}."
        )
        assert result.get("ok") is True
        assert result.get("cooked") is False

    @pytest.mark.asyncio
    async def test_error_result_passed_through_verbatim(self, read_pixels_error_bridge_mock):
        expected_result = read_pixels_error_bridge_mock.execute.return_value
        with patch(
            "fxhoudinimcp.server._get_bridge", return_value=read_pixels_error_bridge_mock
        ):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_read_pixels

            ctx_mock = MagicMock()
            result = await houdini_cop_onnx_read_pixels(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
                plane="nope",
            )

        assert result == expected_result, (
            "Wrapper must pass an FR-5 error shape through unchanged. "
            f"Expected {expected_result!r}, got {result!r}."
        )
        assert result.get("ok") is False


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

    def test_read_pixels_ctx_not_in_schema(self):
        import fxhoudinimcp.tools.cop_onnx_tools  # noqa: F401
        schema = self._get_tool_schema("houdini_cop_onnx_read_pixels")
        properties = schema.get("properties", {})
        assert "ctx" not in properties, (
            f"'ctx' must NOT appear in houdini_cop_onnx_read_pixels's input schema properties. "
            f"Got properties={list(properties.keys())!r}."
        )
