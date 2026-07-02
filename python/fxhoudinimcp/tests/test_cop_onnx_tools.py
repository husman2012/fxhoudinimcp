"""pytest wrapper tests for houdini_cop_onnx_list_models and
houdini_cop_onnx_inspect_model MCP tools (PP12-113 PR-2).

Unit: pp12-113b
testVerificationSurface: pytest-model
planSha: 22e035c1811c41e521c71c799a1c4e8bfad96e8e20799da3e153ae0009b42e7f

These tests are written BEFORE the implementation (red phase). They will
fail with ModuleNotFoundError until hou-dev implements:
  - fxhoudinimcp/tools/cop_onnx_tools.py       (the two MCP wrappers)
  - fxhoudinimcp_server/handlers/cop_onnx_handlers.py  (the handlers, not
    exercised directly by this file — hou-dev's hython-smoke covers those)

Grounded against:
  - python/fxhoudinimcp/tests/test_usd_export_tools.py (MagicMock(spec=)
    pattern; ctx-injection + _get_bridge patch shape; verbatim-passthrough
    assertion style)

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
  - @pytest.mark.asyncio on every async test (both wrappers are async
    @mcp.tool coroutines).
  - Both wrappers make EXACTLY ONE bridge.execute call — no retries, no
    secondary calls.
  - Both must be require_approval=False (ungated / READONLY per the
    locked field contract — the scratch-node cleanup in the handler is
    what keeps this tool READONLY, not the wrapper).

Locked field contract (plan pp12-113b, lockedFieldContract):
    houdini_cop_onnx_list_models(ctx, roots=None) -> dict
        -> bridge.execute('cop_onnx_list_models', {'roots': roots})

    houdini_cop_onnx_inspect_model(ctx, model_path, node_path=None) -> dict
        -> bridge.execute('cop_onnx_inspect_model',
                           {'model_path': model_path, 'node_path': node_path})

Command strings are BARE ('cop_onnx_list_models' / 'cop_onnx_inspect_model')
— NOT the 'cops.*' prefix used by the existing cop_handlers.py (grep-
confirmed no collision at plan time).
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
def list_models_bridge_mock():
    """Spec-bound bridge mock returning a minimal cop_onnx_list_models result."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "ok": True,
        "models": [
            {"path": "/models/identity.onnx", "size": 1024, "mtime": 1719800000.0},
        ],
        "roots_scanned": ["/models"],
        "missing_roots": [],
    })
    return mock


@pytest.fixture()
def inspect_model_bridge_mock():
    """Spec-bound bridge mock returning a minimal OnnxContract.to_dict() shape."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "ok": True,
        "model_path": "/models/identity.onnx",
        "inputs": [
            {"name": "input", "shape": [1, 3, "dynamic", "dynamic"], "dtype": "float32", "layout_guess": "NCHW"},
        ],
        "outputs": [
            {"name": "output", "shape": [1, 1000], "dtype": "float32", "layout_guess": "unknown"},
        ],
        "opset": None,
        "producer": None,
        "loadable": True,
        "error": None,
    })
    return mock


# ---------------------------------------------------------------------------
# Module import (PRIMARY RED GATE)
# ---------------------------------------------------------------------------

class TestCopOnnxToolsModuleImport:
    """cop_onnx_tools module must exist (PRIMARY RED GATE).

    Both tools live in tools/cop_onnx_tools.py. Until hou-dev creates the
    file, every import of this module raises ModuleNotFoundError — that
    is the red signal.
    """

    def test_module_importable(self):
        """Import fxhoudinimcp.tools.cop_onnx_tools — FAILS RED until hou-dev creates it."""
        import fxhoudinimcp.tools.cop_onnx_tools  # noqa: F401

    def test_list_models_callable_on_module(self):
        from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_list_models  # noqa: F401
        assert callable(houdini_cop_onnx_list_models), (
            "houdini_cop_onnx_list_models must be a callable (the @mcp.tool coroutine)."
        )

    def test_inspect_model_callable_on_module(self):
        from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_inspect_model  # noqa: F401
        assert callable(houdini_cop_onnx_inspect_model), (
            "houdini_cop_onnx_inspect_model must be a callable (the @mcp.tool coroutine)."
        )


# ---------------------------------------------------------------------------
# require_approval=False on both wrappers (ungated / READONLY)
# ---------------------------------------------------------------------------

class TestCopOnnxRequireApproval:
    """Both tools must have meta={'require_approval': False} — READONLY, ungated.

    The scratch-node create/destroy in the handler is what keeps
    cop_onnx_inspect_model READONLY (guaranteed cleanup in a finally);
    the wrapper-level contract is simply that no 109-gate approval is
    required for either tool.
    """

    def _get_tool_map(self):
        from fxhoudinimcp.server import mcp
        tool_map: dict = {}
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_map = mcp._tool_manager._tools
        elif hasattr(mcp, "_tools"):
            tool_map = mcp._tools
        return tool_map

    def test_list_models_require_approval_false(self):
        tool_map = self._get_tool_map()
        assert "houdini_cop_onnx_list_models" in tool_map, (
            "houdini_cop_onnx_list_models not registered on the mcp server; "
            "hou-dev must import cop_onnx_tools at server startup."
        )
        tool_obj = tool_map["houdini_cop_onnx_list_models"]
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", True) if isinstance(meta, dict) else True
        assert require_approval is False, (
            f"houdini_cop_onnx_list_models meta must have require_approval=False. Got meta={meta!r}."
        )

    def test_inspect_model_require_approval_false(self):
        tool_map = self._get_tool_map()
        assert "houdini_cop_onnx_inspect_model" in tool_map, (
            "houdini_cop_onnx_inspect_model not registered on the mcp server; "
            "hou-dev must import cop_onnx_tools at server startup."
        )
        tool_obj = tool_map["houdini_cop_onnx_inspect_model"]
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", True) if isinstance(meta, dict) else True
        assert require_approval is False, (
            f"houdini_cop_onnx_inspect_model meta must have require_approval=False. Got meta={meta!r}."
        )


# ---------------------------------------------------------------------------
# bridge.execute call contract — houdini_cop_onnx_list_models
# ---------------------------------------------------------------------------

class TestCopOnnxListModelsBridgeContract:
    """houdini_cop_onnx_list_models must delegate to bridge.execute exactly once,
    with command 'cop_onnx_list_models' and params {'roots': roots}."""

    @pytest.mark.asyncio
    async def test_command_string_is_cop_onnx_list_models(self, list_models_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=list_models_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_list_models

            ctx_mock = MagicMock()
            await houdini_cop_onnx_list_models(ctx=ctx_mock, roots=None)

        list_models_bridge_mock.execute.assert_called_once()
        call_args = list_models_bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "cop_onnx_list_models", (
            f"Expected bridge.execute('cop_onnx_list_models', ...) but got command={command!r}. "
            "Command string must match register_handler's first arg exactly (bare, not 'cops.*')."
        )

    @pytest.mark.asyncio
    async def test_params_dict_has_roots_key(self, list_models_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=list_models_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_list_models

            ctx_mock = MagicMock()
            roots = ["/models", "/other_models"]
            await houdini_cop_onnx_list_models(ctx=ctx_mock, roots=roots)

        call_args = list_models_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert "roots" in params, f"Params dict must contain 'roots'. Got params={params!r}."
        assert params["roots"] == roots, (
            f"roots must be forwarded verbatim. Expected {roots!r}, got {params['roots']!r}."
        )

    @pytest.mark.asyncio
    async def test_roots_none_default_forwarded(self, list_models_bridge_mock):
        """roots=None (the default) must be forwarded as None, not omitted or coerced."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=list_models_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_list_models

            ctx_mock = MagicMock()
            await houdini_cop_onnx_list_models(ctx=ctx_mock)

        call_args = list_models_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert "roots" in params, f"'roots' key must be present even when None. Got params={params!r}."
        assert params["roots"] is None

    @pytest.mark.asyncio
    async def test_exactly_one_bridge_execute_call(self, list_models_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=list_models_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_list_models

            ctx_mock = MagicMock()
            await houdini_cop_onnx_list_models(ctx=ctx_mock, roots=None)

        assert list_models_bridge_mock.execute.call_count == 1, (
            f"houdini_cop_onnx_list_models must make exactly ONE bridge.execute call, "
            f"got {list_models_bridge_mock.execute.call_count}."
        )

    @pytest.mark.asyncio
    async def test_result_passed_through_verbatim(self, list_models_bridge_mock):
        expected_result = list_models_bridge_mock.execute.return_value
        with patch("fxhoudinimcp.server._get_bridge", return_value=list_models_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_list_models

            ctx_mock = MagicMock()
            result = await houdini_cop_onnx_list_models(ctx=ctx_mock, roots=None)

        assert result == expected_result, (
            "Wrapper must pass bridge.execute's result through unchanged. "
            f"Expected {expected_result!r}, got {result!r}."
        )


# ---------------------------------------------------------------------------
# bridge.execute call contract — houdini_cop_onnx_inspect_model
# ---------------------------------------------------------------------------

class TestCopOnnxInspectModelBridgeContract:
    """houdini_cop_onnx_inspect_model must delegate to bridge.execute exactly
    once, with command 'cop_onnx_inspect_model' and params
    {'model_path': ..., 'node_path': ...}."""

    @pytest.mark.asyncio
    async def test_command_string_is_cop_onnx_inspect_model(self, inspect_model_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=inspect_model_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_inspect_model

            ctx_mock = MagicMock()
            await houdini_cop_onnx_inspect_model(ctx=ctx_mock, model_path="/models/identity.onnx")

        inspect_model_bridge_mock.execute.assert_called_once()
        call_args = inspect_model_bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "cop_onnx_inspect_model", (
            f"Expected bridge.execute('cop_onnx_inspect_model', ...) but got command={command!r}."
        )

    @pytest.mark.asyncio
    async def test_params_dict_has_model_path_and_node_path_keys(self, inspect_model_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=inspect_model_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_inspect_model

            ctx_mock = MagicMock()
            model_path = "/models/identity.onnx"
            await houdini_cop_onnx_inspect_model(ctx=ctx_mock, model_path=model_path)

        call_args = inspect_model_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        required_keys = {"model_path", "node_path"}
        missing = required_keys - set(params.keys())
        assert not missing, (
            f"Params dict is missing required keys: {missing!r}. Got params={params!r}."
        )
        assert params["model_path"] == model_path, (
            f"model_path must be forwarded verbatim. Expected {model_path!r}, got {params['model_path']!r}."
        )

    @pytest.mark.asyncio
    async def test_node_path_none_default_forwarded(self, inspect_model_bridge_mock):
        """node_path=None (the default, meaning 'use the scratch-node mechanism')
        must be forwarded as None, not omitted."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=inspect_model_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_inspect_model

            ctx_mock = MagicMock()
            await houdini_cop_onnx_inspect_model(ctx=ctx_mock, model_path="/models/identity.onnx")

        call_args = inspect_model_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert params.get("node_path") is None

    @pytest.mark.asyncio
    async def test_node_path_forwarded_when_provided(self, inspect_model_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=inspect_model_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_inspect_model

            ctx_mock = MagicMock()
            node_path = "/obj/copnet1/onnx1"
            await houdini_cop_onnx_inspect_model(
                ctx=ctx_mock,
                model_path="/models/identity.onnx",
                node_path=node_path,
            )

        call_args = inspect_model_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert params.get("node_path") == node_path, (
            f"node_path must be forwarded verbatim when provided. "
            f"Expected {node_path!r}, got {params.get('node_path')!r}."
        )

    @pytest.mark.asyncio
    async def test_exactly_one_bridge_execute_call(self, inspect_model_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=inspect_model_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_inspect_model

            ctx_mock = MagicMock()
            await houdini_cop_onnx_inspect_model(ctx=ctx_mock, model_path="/models/identity.onnx")

        assert inspect_model_bridge_mock.execute.call_count == 1, (
            f"houdini_cop_onnx_inspect_model must make exactly ONE bridge.execute call, "
            f"got {inspect_model_bridge_mock.execute.call_count}."
        )

    @pytest.mark.asyncio
    async def test_result_passed_through_verbatim(self, inspect_model_bridge_mock):
        expected_result = inspect_model_bridge_mock.execute.return_value
        with patch("fxhoudinimcp.server._get_bridge", return_value=inspect_model_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_inspect_model

            ctx_mock = MagicMock()
            result = await houdini_cop_onnx_inspect_model(ctx=ctx_mock, model_path="/models/identity.onnx")

        assert result == expected_result, (
            "Wrapper must pass bridge.execute's result through unchanged. "
            f"Expected {expected_result!r}, got {result!r}."
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
    """ctx must NOT appear in either tool's input schema properties.

    FastMCP injects ctx via the Context type annotation — it is NOT a
    parameter the MCP client sends.
    """

    def _get_tool_schema(self, tool_name: str) -> dict:
        from fxhoudinimcp.server import mcp
        tool_map: dict = {}
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_map = mcp._tool_manager._tools
        elif hasattr(mcp, "_tools"):
            tool_map = mcp._tools
        tool_obj = tool_map.get(tool_name)
        if tool_obj is None:
            return {}
        return getattr(tool_obj, "parameters", getattr(tool_obj, "inputSchema", {})) or {}

    def test_list_models_ctx_not_in_schema(self):
        schema = self._get_tool_schema("houdini_cop_onnx_list_models")
        properties = schema.get("properties", {})
        assert "ctx" not in properties, (
            f"'ctx' must NOT appear in houdini_cop_onnx_list_models's input schema properties. "
            f"Got properties={list(properties.keys())!r}."
        )

    def test_inspect_model_ctx_not_in_schema(self):
        schema = self._get_tool_schema("houdini_cop_onnx_inspect_model")
        properties = schema.get("properties", {})
        assert "ctx" not in properties, (
            f"'ctx' must NOT appear in houdini_cop_onnx_inspect_model's input schema properties. "
            f"Got properties={list(properties.keys())!r}."
        )


# ---------------------------------------------------------------------------
# Failure shape — FR-2/FR-5 style: verbatim passthrough of {ok: False, error}
# ---------------------------------------------------------------------------

class TestCopOnnxFailureShape:
    """Both wrappers must return {ok: False, error: str} verbatim when the
    handler (via bridge.execute) reports failure — never raise past the
    wrapper boundary."""

    @pytest.mark.asyncio
    async def test_list_models_failure_result_passthrough(self):
        failure_mock = MagicMock(spec=HoudiniBridge)
        failure_mock.execute = AsyncMock(return_value={
            "ok": False,
            "error": "roots must be a list of strings",
        })
        with patch("fxhoudinimcp.server._get_bridge", return_value=failure_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_list_models

            ctx_mock = MagicMock()
            result = await houdini_cop_onnx_list_models(ctx=ctx_mock, roots=None)

        assert isinstance(result, dict)
        assert result.get("ok") is False
        assert isinstance(result.get("error"), str)

    @pytest.mark.asyncio
    async def test_inspect_model_failure_result_passthrough(self):
        failure_mock = MagicMock(spec=HoudiniBridge)
        failure_mock.execute = AsyncMock(return_value={
            "ok": False,
            "error": "model_path does not exist: /models/missing.onnx",
        })
        with patch("fxhoudinimcp.server._get_bridge", return_value=failure_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_inspect_model

            ctx_mock = MagicMock()
            result = await houdini_cop_onnx_inspect_model(ctx=ctx_mock, model_path="/models/missing.onnx")

        assert isinstance(result, dict)
        assert result.get("ok") is False
        assert isinstance(result.get("error"), str)
