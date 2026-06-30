"""pytest wrapper tests for houdini_usd_inspect_layer and houdini_usd_validate MCP tools.

Unit: pp12-112b
testVerificationSurface: pytest-model
planSha: ac5886a927ad1d62ea2d6768414f8a907040286de888373f37dd3a1bc7a0ca6c

These tests are written BEFORE the implementation (red phase). They will fail
with ImportError until hou-dev implements:
  - fxhoudinimcp/tools/usd_export_tools.py  (the two MCP wrappers)
  - fxhoudinimcp_server/handlers/usd_export_handlers.py  (the handlers)

Grounded against:
  - python/fxhoudinimcp/tests/test_render_compare_wrapper.py  (MagicMock(spec=) pattern)
  - python/fxhoudinimcp/tests/test_render_parse_exr_wrapper.py  (async + ctx shape)

PP12-110 lessons encoded here:
  - MagicMock(spec=HoudiniBridge) — non-existent attr (e.g. .call) raises AttributeError
  - Import subject INSIDE each test, AFTER _get_bridge is patched
  - Assert PUBLIC behavior (bridge.execute cmd+params; result shape), NOT call order
  - @pytest.mark.asyncio on every async test
  - ALL FIVE params-dict keys forwarded for houdini_usd_validate (locked contract rev2)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import REAL HoudiniBridge for spec= (PP12-110 lesson: bare MagicMock() silently
# accepts .call and any other non-existent attribute, masking convention bugs).
from fxhoudinimcp.bridge import HoudiniBridge


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def inspect_bridge_mock():
    """Spec-bound bridge mock returning a minimal LayerSummary.to_dict() shape."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "ok": True,
        "default_prim": "/World/asset",
        "root_prims": ["/World"],
        "sublayers": [],
        "current_format": "in-memory",
        "has_mtlx_material": False,
    })
    return mock


@pytest.fixture()
def validate_bridge_mock():
    """Spec-bound bridge mock returning a minimal ValidationReport + mode/omitted shape (B-1)."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "ok": True,
        "verdict": "pass",
        "checks": [
            {"id": "no_world_wrapper", "passed": True, "severity": "error"},
            {"id": "default_prim_set", "passed": True, "severity": "error"},
        ],
        "wrote_files": False,
        "mode": "minimal",
        "omitted_checks": ["format", "textures"],
    })
    return mock


# ---------------------------------------------------------------------------
# UEC-1: module import (PRIMARY RED GATE)
# ---------------------------------------------------------------------------

class TestUsdExportModuleImport:
    """UEC-1: usd_export_tools module must exist (PRIMARY RED GATE).

    Both tools live in tools/usd_export_tools.py. Until hou-dev creates the file,
    every import of this module raises ImportError — this is the red signal.
    """

    def test_module_importable(self):
        """Import fxhoudinimcp.tools.usd_export_tools — FAILS RED until hou-dev creates it."""
        import fxhoudinimcp.tools.usd_export_tools  # noqa: F401

    def test_inspect_layer_callable_on_module(self):
        """houdini_usd_inspect_layer must be a callable on usd_export_tools."""
        from fxhoudinimcp.tools.usd_export_tools import houdini_usd_inspect_layer  # noqa: F401
        assert callable(houdini_usd_inspect_layer), (
            "houdini_usd_inspect_layer must be a callable (the @mcp.tool coroutine)."
        )

    def test_validate_callable_on_module(self):
        """houdini_usd_validate must be a callable on usd_export_tools."""
        from fxhoudinimcp.tools.usd_export_tools import houdini_usd_validate  # noqa: F401
        assert callable(houdini_usd_validate), (
            "houdini_usd_validate must be a callable (the @mcp.tool coroutine)."
        )


# ---------------------------------------------------------------------------
# UEC-2: FR-10 — require_approval=False on houdini_usd_inspect_layer
# ---------------------------------------------------------------------------

class TestUsdInspectLayerRequireApproval:
    """UEC-2: houdini_usd_inspect_layer must have meta={'require_approval': False} (FR-10).

    Both tools are Capability.READONLY — they read USD stage data and never mutate
    the Houdini scene. FR-10 mandates require_approval=False to bypass the 109 gate.
    """

    def test_mcp_tool_meta_require_approval_false(self):
        """@mcp.tool(meta={'require_approval': False}) on houdini_usd_inspect_layer."""
        from fxhoudinimcp.server import mcp

        tool_map: dict = {}
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_map = mcp._tool_manager._tools
        elif hasattr(mcp, "_tools"):
            tool_map = mcp._tools

        assert "houdini_usd_inspect_layer" in tool_map, (
            "houdini_usd_inspect_layer not registered on the mcp server; "
            "hou-dev must import usd_export_tools at server startup."
        )

        tool_obj = tool_map["houdini_usd_inspect_layer"]
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", True) if isinstance(meta, dict) else True

        assert require_approval is False, (
            f"houdini_usd_inspect_layer meta must have require_approval=False (FR-10). "
            f"Got meta={meta!r}."
        )


# ---------------------------------------------------------------------------
# UEC-3: FR-10 — require_approval=False on houdini_usd_validate
# ---------------------------------------------------------------------------

class TestUsdValidateRequireApproval:
    """UEC-3: houdini_usd_validate must have meta={'require_approval': False} (FR-10)."""

    def test_mcp_tool_meta_require_approval_false(self):
        """@mcp.tool(meta={'require_approval': False}) on houdini_usd_validate."""
        from fxhoudinimcp.server import mcp

        tool_map: dict = {}
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_map = mcp._tool_manager._tools
        elif hasattr(mcp, "_tools"):
            tool_map = mcp._tools

        assert "houdini_usd_validate" in tool_map, (
            "houdini_usd_validate not registered on the mcp server; "
            "hou-dev must import usd_export_tools at server startup."
        )

        tool_obj = tool_map["houdini_usd_validate"]
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", True) if isinstance(meta, dict) else True

        assert require_approval is False, (
            f"houdini_usd_validate meta must have require_approval=False (FR-10). "
            f"Got meta={meta!r}."
        )


# ---------------------------------------------------------------------------
# UEC-4: bridge.execute call contract — houdini_usd_inspect_layer
# ---------------------------------------------------------------------------

class TestUsdInspectLayerBridgeContract:
    """UEC-4: houdini_usd_inspect_layer must delegate to bridge.execute with exact contract.

    Locked field contract (plan rev2 §3):
        houdini_usd_inspect_layer(ctx: Context, node_or_layer: str) -> dict
        -> bridge.execute('usd_inspect_layer', {'node_or_layer': node_or_layer})

    Command string must be exactly 'usd_inspect_layer' (bare, NOT 'lops.inspect_usd_layer').
    Params dict must have exactly {'node_or_layer': <value>}.
    """

    @pytest.mark.asyncio
    async def test_command_string_is_usd_inspect_layer(self, inspect_bridge_mock):
        """bridge.execute must be called with command='usd_inspect_layer'."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=inspect_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_inspect_layer

            ctx_mock = MagicMock()
            await houdini_usd_inspect_layer(
                ctx=ctx_mock,
                node_or_layer="/stage/loplayer1",
            )

        inspect_bridge_mock.execute.assert_called_once()
        call_args = inspect_bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "usd_inspect_layer", (
            f"Expected bridge.execute('usd_inspect_layer', ...) but got command={command!r}. "
            "PP12-110: command string must match register_handler's first arg exactly."
        )

    @pytest.mark.asyncio
    async def test_params_dict_has_node_or_layer_key(self, inspect_bridge_mock):
        """bridge.execute params must be {'node_or_layer': <arg>}."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=inspect_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_inspect_layer

            ctx_mock = MagicMock()
            target = "/stage/loplayer1"
            await houdini_usd_inspect_layer(ctx=ctx_mock, node_or_layer=target)

        call_args = inspect_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert "node_or_layer" in params, (
            f"Params dict must contain 'node_or_layer'. Got params={params!r}."
        )
        assert params["node_or_layer"] == target, (
            f"node_or_layer must be forwarded verbatim. "
            f"Expected {target!r}, got {params['node_or_layer']!r}."
        )

    @pytest.mark.asyncio
    async def test_result_passed_through_verbatim(self, inspect_bridge_mock):
        """Wrapper must return bridge.execute's result verbatim (no transformation)."""
        expected_result = inspect_bridge_mock.execute.return_value
        with patch("fxhoudinimcp.server._get_bridge", return_value=inspect_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_inspect_layer

            ctx_mock = MagicMock()
            result = await houdini_usd_inspect_layer(ctx=ctx_mock, node_or_layer="/s/lop1")

        # Verbatim passthrough: the wrapper must return bridge.execute's result unchanged.
        # inspect_bridge_mock.execute is an AsyncMock; result is the already-awaited value.
        assert result == expected_result, (
            "Wrapper must pass bridge.execute's result through unchanged. "
            f"Expected {expected_result!r}, got {result!r}."
        )
        inspect_bridge_mock.execute.assert_called_once()


# ---------------------------------------------------------------------------
# UEC-5: bridge.execute call contract — houdini_usd_validate (ALL FIVE KEYS)
# ---------------------------------------------------------------------------

class TestUsdValidateBridgeContract:
    """UEC-5: houdini_usd_validate must delegate with ALL FIVE params-dict keys.

    Locked field contract (plan rev2 §3 — B-2 widened signature):
        houdini_usd_validate(ctx, target, out_path=None, actual_format=None,
                             texture_paths=None, checks=None) -> dict
        -> bridge.execute('usd_validate', {
               'target': target,
               'out_path': out_path,
               'actual_format': actual_format,
               'texture_paths': texture_paths,
               'checks': checks,
           })

    ALL FIVE keys must be present in the params dict, even when their values are None.
    This is required so the handler receives and can react to (e.g. checks != None → M-3).
    """

    @pytest.mark.asyncio
    async def test_command_string_is_usd_validate(self, validate_bridge_mock):
        """bridge.execute must be called with command='usd_validate'."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=validate_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_validate

            ctx_mock = MagicMock()
            await houdini_usd_validate(ctx=ctx_mock, target="/stage/lop1")

        validate_bridge_mock.execute.assert_called_once()
        call_args = validate_bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "usd_validate", (
            f"Expected bridge.execute('usd_validate', ...) but got command={command!r}."
        )

    @pytest.mark.asyncio
    async def test_params_dict_has_all_five_keys_when_defaults_used(self, validate_bridge_mock):
        """Params dict must contain all five keys even when optional args are None.

        A wrapper that only forwards non-None params would break the handler's
        ability to detect intent (e.g. checks=None vs checks=['format'] differ).
        """
        with patch("fxhoudinimcp.server._get_bridge", return_value=validate_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_validate

            ctx_mock = MagicMock()
            await houdini_usd_validate(ctx=ctx_mock, target="/stage/lop1")

        call_args = validate_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        required_keys = {"target", "out_path", "actual_format", "texture_paths", "checks"}
        missing = required_keys - set(params.keys())
        assert not missing, (
            f"Params dict is missing required keys: {missing!r}. "
            f"ALL FIVE keys must be forwarded (even when None). Got params={params!r}."
        )

    @pytest.mark.asyncio
    async def test_target_value_forwarded_correctly(self, validate_bridge_mock):
        """target param must be forwarded verbatim."""
        target = "/stage/lop_validate_test"
        with patch("fxhoudinimcp.server._get_bridge", return_value=validate_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_validate

            ctx_mock = MagicMock()
            await houdini_usd_validate(ctx=ctx_mock, target=target)

        call_args = validate_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert params.get("target") == target, (
            f"target must be forwarded verbatim. Expected {target!r}, got {params.get('target')!r}."
        )

    @pytest.mark.asyncio
    async def test_optional_params_forwarded_when_provided(self, validate_bridge_mock):
        """out_path, actual_format, texture_paths, checks are forwarded when non-None."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=validate_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_validate

            ctx_mock = MagicMock()
            await houdini_usd_validate(
                ctx=ctx_mock,
                target="/stage/lop1",
                out_path="/tmp/test.usdc",
                actual_format="usdc",
                texture_paths=["/tmp/tex.png"],
                checks=None,  # checks=None is the only valid value in v1 (M-3)
            )

        call_args = validate_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        assert params.get("out_path") == "/tmp/test.usdc", (
            f"out_path must be forwarded. Got {params.get('out_path')!r}."
        )
        assert params.get("actual_format") == "usdc", (
            f"actual_format must be forwarded. Got {params.get('actual_format')!r}."
        )
        assert params.get("texture_paths") == ["/tmp/tex.png"], (
            f"texture_paths must be forwarded. Got {params.get('texture_paths')!r}."
        )
        assert params.get("checks") is None, (
            f"checks=None must be forwarded as None. Got {params.get('checks')!r}."
        )


# ---------------------------------------------------------------------------
# UEC-6: spec-bound bridge mock — PP12-110 regression guard
# ---------------------------------------------------------------------------

class TestBridgeSpecBound:
    """UEC-6: MagicMock(spec=HoudiniBridge) — .call raises AttributeError (PP12-110 guard).

    In PP12-110, the wrapper called bridge.call(...) instead of bridge.execute(...).
    A bare MagicMock() silently accepted .call and returned a mock — masking the bug.
    A spec-bound mock (spec=HoudiniBridge) raises AttributeError because .call is
    not a real method on HoudiniBridge.

    This test proves the guard is effective: if we accidentally used .call in our
    test helper, the AttributeError would fire here.
    """

    def test_bridge_call_attribute_does_not_exist(self):
        """MagicMock(spec=HoudiniBridge).call raises AttributeError — .call is not on the class."""
        mock = MagicMock(spec=HoudiniBridge)
        with pytest.raises(AttributeError, match="call"):
            _ = mock.call  # .call must NOT exist on HoudiniBridge

    def test_bridge_execute_attribute_exists(self):
        """MagicMock(spec=HoudiniBridge).execute is accessible (execute IS on the class)."""
        mock = MagicMock(spec=HoudiniBridge)
        # This should NOT raise — execute is a real method on HoudiniBridge
        _ = mock.execute


# ---------------------------------------------------------------------------
# UEC-7: ctx not in tool input schema (ctx is injected, not a tool parameter)
# ---------------------------------------------------------------------------

class TestCtxSchemaGuard:
    """UEC-7: ctx must NOT appear in the tool's input schema properties.

    FastMCP injects ctx via the Context type annotation — it is NOT a parameter
    the MCP client sends. If ctx leaks into the JSON schema, the tool's input
    contract is wrong (clients would need to provide a ctx value, which they cannot).
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

    def test_inspect_layer_ctx_not_in_schema(self):
        """houdini_usd_inspect_layer: 'ctx' must not be a property in the tool's input schema."""
        schema = self._get_tool_schema("houdini_usd_inspect_layer")
        properties = schema.get("properties", {})
        assert "ctx" not in properties, (
            f"'ctx' must NOT appear in houdini_usd_inspect_layer's input schema properties. "
            f"FastMCP injects it; clients don't provide it. Got properties={list(properties.keys())!r}."
        )

    def test_validate_ctx_not_in_schema(self):
        """houdini_usd_validate: 'ctx' must not be a property in the tool's input schema."""
        schema = self._get_tool_schema("houdini_usd_validate")
        properties = schema.get("properties", {})
        assert "ctx" not in properties, (
            f"'ctx' must NOT appear in houdini_usd_validate's input schema properties. "
            f"Got properties={list(properties.keys())!r}."
        )


# ---------------------------------------------------------------------------
# UEC-8: failure shape — FR-2/FR-5
# ---------------------------------------------------------------------------

class TestInspectLayerFailureShape:
    """UEC-8: houdini_usd_inspect_layer must return {ok: False, error: str} on failure.

    FR-2/FR-5: any exception / bad target from the Houdini side is surfaced as
    {ok: False, error: '<reason>'} — never propagated as a Python exception past
    the handler boundary. The wrapper passes through whatever bridge.execute returns.
    """

    @pytest.mark.asyncio
    async def test_failure_result_has_ok_false_and_error_string(self):
        """When bridge returns {ok: False, error: ...}, wrapper must return it unchanged."""
        failure_mock = MagicMock(spec=HoudiniBridge)
        failure_mock.execute = AsyncMock(return_value={
            "ok": False,
            "error": "Node '/obj/geo1/box1' exists but is not a LOP node — stage() not available",
        })

        with patch("fxhoudinimcp.server._get_bridge", return_value=failure_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_inspect_layer

            ctx_mock = MagicMock()
            result = await houdini_usd_inspect_layer(ctx=ctx_mock, node_or_layer="/obj/geo1/box1")

        assert isinstance(result, dict), f"Result must be a dict, got {type(result)!r}."
        assert result.get("ok") is False, (
            f"Failure result must have ok=False. Got ok={result.get('ok')!r}."
        )
        assert "error" in result, (
            f"Failure result must contain 'error' key. Got keys={list(result.keys())!r}."
        )
        assert isinstance(result["error"], str), (
            f"'error' must be a string. Got {type(result['error'])!r}."
        )


# ---------------------------------------------------------------------------
# UEC-9: M-3 — non-None checks param -> {ok: False} from the handler
#          (wrapper forwards checks=non-None; handler rejects it)
# ---------------------------------------------------------------------------

class TestUsdValidateChecksRejection:
    """UEC-9 (M-3): houdini_usd_validate must forward checks to bridge; if checks is non-None,
    the handler returns {ok: False, error} (checks subsetting is not implemented in v1).

    The wrapper itself does NOT reject checks — it forwards all five params unchanged.
    The rejection lives in the handler (via bridge.execute). This test simulates the
    full path: wrapper forwards checks=['format'], bridge returns {ok: False, error}.
    """

    @pytest.mark.asyncio
    async def test_non_none_checks_forwarded_and_rejection_returned(self):
        """wrapper forwards checks=['format'] → bridge returns {ok:False, error}."""
        rejection_mock = MagicMock(spec=HoudiniBridge)
        rejection_mock.execute = AsyncMock(return_value={
            "ok": False,
            "error": "checks subsetting not implemented in v1; pass None",
        })

        with patch("fxhoudinimcp.server._get_bridge", return_value=rejection_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_validate

            ctx_mock = MagicMock()
            result = await houdini_usd_validate(
                ctx=ctx_mock,
                target="/stage/lop1",
                checks=["format"],
            )

        # Verify the wrapper forwarded checks (not silently stripped it)
        call_args = rejection_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert params.get("checks") == ["format"], (
            f"Wrapper must forward checks=['format'] to bridge. "
            f"Got checks={params.get('checks')!r}."
        )

        # Verify the failure result is passed through
        assert result.get("ok") is False, (
            f"Result must be {{ok: False}} when checks is non-None. Got ok={result.get('ok')!r}."
        )
        assert "error" in result, (
            f"Result must contain 'error' when checks is non-None. Got keys={list(result.keys())!r}."
        )
        error_text = result.get("error", "")
        assert "checks" in error_text.lower() or "not implemented" in error_text.lower(), (
            f"error message should mention checks or not-implemented. Got: {error_text!r}."
        )
