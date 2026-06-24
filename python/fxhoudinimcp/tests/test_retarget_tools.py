"""
test_retarget_tools.py -- spec-bound mock-bridge tests for PP12-110 PR-5 retarget tool.

PR-5 discipline (mirrors PR-4 / test_bonedeform_tools.py):
  - Tests use MagicMock(spec=HoudiniBridge) — bare MagicMock() is BANNED.
  - Regression guard asserts spec-bound mock raises AttributeError on .call.
  - _get_bridge is asserted to receive ctx (server.py:21 requires it).
  - Bridge result passes through VERBATIM — wrapper is a thin passthrough.
  - mapping=[[src, tgt]] is forwarded unchanged in the params dict.

KineFX retarget chain (plan riskNotes — authoritative):
  kinefx::rigmatchpose -> [kinefx::mappoints (only if explicit mapping)]
  -> kinefx::fullbodyik
  NOTE: "motiontransform" does NOT exist in H21.
  Connector order: setInput(0)=target, setInput(1)=source.

Expected wrapper signature (pp12-110e plan):
  @mcp.tool(meta={"require_approval": True})
  async def houdini_setup_retarget(
      ctx: Context,
      source: str,
      target: str,
      method: str = "rigmatchpose+fullbodyik",
      match_size: bool = True,
      mapping: list | None = None,
      dest: str = "/obj",
  ) -> dict[str, Any]:
      bridge = _get_bridge(ctx)
      return await bridge.execute("setup_retarget", {
          "source": source, "target": target, "method": method,
          "match_size": match_size, "mapping": mapping, "dest": dest,
      })

testVerificationSurface: pytest-model
unitId: pp12-110e
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from fxhoudinimcp.tools import kinefx_tools  # noqa: E402
from fxhoudinimcp.bridge import HoudiniBridge  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_bridge():
    bridge = MagicMock(spec=HoudiniBridge)
    bridge.execute = AsyncMock(return_value={
        "ok": True,
        "node": "/obj/mcp_setup_retarget1",
        "source": "/obj/source_skel",
        "target": "/obj/target_skel",
    })
    return bridge


@pytest.fixture()
def mock_ctx():
    return MagicMock()


@pytest.fixture()
def mock_bridge_error():
    bridge = MagicMock(spec=HoudiniBridge)
    bridge.execute = AsyncMock(return_value={
        "ok": False,
        "error": "setup_retarget: source skeleton not found at /obj/missing",
    })
    return bridge


# ---------------------------------------------------------------------------
# Spec-bound regression guard
# ---------------------------------------------------------------------------

class TestSpecBoundRejectsCall:
    def test_bridge_call_raises_attribute_error(self, mock_bridge):
        """MagicMock(spec=HoudiniBridge) must raise AttributeError on .call
        (the banned bare-MagicMock typo that PP12-110 PR-3 shipped silently)."""
        with pytest.raises(AttributeError):
            _ = mock_bridge.call


# ---------------------------------------------------------------------------
# Tool existence + async contract
# ---------------------------------------------------------------------------

class TestRetargetToolExists:
    def test_setup_retarget_exists(self):
        """houdini_setup_retarget must be a module-level attribute of kinefx_tools."""
        assert hasattr(kinefx_tools, "houdini_setup_retarget")

    def test_setup_retarget_is_async(self):
        """houdini_setup_retarget must be an async coroutine function."""
        fn = getattr(kinefx_tools, "houdini_setup_retarget", None)
        assert fn is not None
        assert asyncio.iscoroutinefunction(fn)


# ---------------------------------------------------------------------------
# MCP gate — require_approval must be True (PP12-109 gate)
# ---------------------------------------------------------------------------

class TestRetargetToolIsGated:
    def _read_mcp_tool_meta(self, tool_name):
        mcp = kinefx_tools.mcp
        tools = mcp._tool_manager._tools
        assert tool_name in tools, f"{tool_name} not registered in MCP tool registry"
        return tools[tool_name].meta or {}

    def test_setup_retarget_require_approval_true(self):
        """houdini_setup_retarget must be decorated with meta={"require_approval": True}
        so the PP12-109 gate queues it for operator approval before execution."""
        meta = self._read_mcp_tool_meta("houdini_setup_retarget")
        assert meta.get("require_approval") is True


# ---------------------------------------------------------------------------
# Bridge delegation contract
# ---------------------------------------------------------------------------

class TestSetupRetargetBridgeExecute:
    @pytest.mark.asyncio
    async def test_calls_get_bridge_with_ctx(self, mock_ctx, mock_bridge):
        """_get_bridge must be called exactly once with ctx (PP12-110 convention)."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge) as mock_get:
            await kinefx_tools.houdini_setup_retarget(
                mock_ctx,
                source="/obj/source_skel",
                target="/obj/target_skel",
            )
            mock_get.assert_called_once_with(mock_ctx)

    @pytest.mark.asyncio
    async def test_calls_execute_with_setup_retarget_command(self, mock_ctx, mock_bridge):
        """bridge.execute must be called with the command string 'setup_retarget'."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.houdini_setup_retarget(
                mock_ctx,
                source="/obj/source_skel",
                target="/obj/target_skel",
            )
        args = mock_bridge.execute.call_args
        assert args[0][0] == "setup_retarget"

    @pytest.mark.asyncio
    async def test_execute_called_once(self, mock_ctx, mock_bridge):
        """bridge.execute must be called exactly once per wrapper invocation."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.houdini_setup_retarget(
                mock_ctx,
                source="/obj/source_skel",
                target="/obj/target_skel",
            )
        mock_bridge.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_called_with_full_params_dict_defaults(self, mock_ctx, mock_bridge):
        """bridge.execute must receive the full params dict with default values filled in."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.houdini_setup_retarget(
                mock_ctx,
                source="/obj/source_skel",
                target="/obj/target_skel",
            )
        mock_bridge.execute.assert_called_once_with(
            "setup_retarget",
            {
                "source": "/obj/source_skel",
                "target": "/obj/target_skel",
                "method": "rigmatchpose+fullbodyik",
                "match_size": True,
                "mapping": None,
                "dest": "/obj",
            },
        )

    @pytest.mark.asyncio
    async def test_mapping_list_forwarded_unchanged(self, mock_ctx, mock_bridge):
        """mapping=[[src, tgt]] must be forwarded verbatim in the params dict
        (PP12-110e plan AC: explicit mapping is not pre-processed by the wrapper)."""
        explicit_mapping = [["Hips", "root"], ["Spine", "spine_01"]]
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.houdini_setup_retarget(
                mock_ctx,
                source="/obj/source_skel",
                target="/obj/target_skel",
                mapping=explicit_mapping,
            )
        _cmd, params = mock_bridge.execute.call_args[0]
        assert params["mapping"] == explicit_mapping, (
            f"mapping must be forwarded unchanged; got {params['mapping']!r}"
        )

    @pytest.mark.asyncio
    async def test_wrapper_passes_bridge_result_through_verbatim(self, mock_ctx, mock_bridge):
        """houdini_setup_retarget must return the bridge result verbatim.
        The wrapper is a thin passthrough — it must not mutate or repackage."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            result = await kinefx_tools.houdini_setup_retarget(
                mock_ctx,
                source="/obj/source_skel",
                target="/obj/target_skel",
            )
        assert result.get("ok") is True
        assert result.get("node") == "/obj/mcp_setup_retarget1"

    @pytest.mark.asyncio
    async def test_fail_loud_returns_ok_false_no_exception(self, mock_ctx, mock_bridge_error):
        """An error envelope from the bridge must be returned as-is (FR-2 fail-loud).
        The wrapper must NOT raise — it surfaces the error dict verbatim."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_error):
            result = await kinefx_tools.houdini_setup_retarget(
                mock_ctx,
                source="/obj/missing",
                target="/obj/target_skel",
            )
        assert result.get("ok") is False
        assert "error" in result and result["error"]
