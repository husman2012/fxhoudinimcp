"""
test_secondarymotion_tools.py -- spec-bound mock-bridge tests for PP12-110 PR-6 secondarymotion tool.

PR-6 discipline (mirrors PR-5 / test_retarget_tools.py):
  - Tests use MagicMock(spec=HoudiniBridge) — bare MagicMock() is BANNED.
  - Regression guard asserts spec-bound mock raises AttributeError on .call.
  - _get_bridge is asserted to receive ctx (server.py convention).
  - Bridge result passes through VERBATIM — wrapper is a thin passthrough.
  - joints=["a","b"] and params={"effect":"jiggle","stiffness":0.4} forwarded unchanged.
  - require_approval must be True (PP12-109 gate — Capability.MUTATING).

kinefx::secondarymotion node facts (plan riskNotes — authoritative):
  Inputs: ['Skeleton', 'MotionClip'] — wire ONLY input 0 (Skeleton).
  Effect menu tokens: ['lagovershoot', 'jiggle', 'spring'].
  Probe-confirmed parm names: jointgroup, effect, effectmult, lag (len2), overshoot (len2),
    stiffness (NOT jigglestiffness), jiggledamping, limit, flex, multiplier (len3),
    springconstant, mass, damping.

Expected wrapper signature (pp12-110f plan):
  @mcp.tool(meta={"require_approval": True})
  async def houdini_apply_secondarymotion(
      ctx: Context,
      node: str,
      joints: list[str] | None = None,
      params: dict | None = None,
      dest: str = "/obj",
  ) -> dict[str, Any]:
      bridge = _get_bridge(ctx)
      return await bridge.execute("apply_secondarymotion", {
          "node": node, "joints": joints, "params": params, "dest": dest,
      })

testVerificationSurface: pytest-model
unitId: pp12-110f
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
        "node": "/obj/geo1/mcp_secondarymotion1",
        "effect": "jiggle",
        "affected_joints": ["a", "b"],
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
        "error": "apply_secondarymotion: skeleton not found at /obj/missing",
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

class TestSecondarymotionToolExists:
    def test_apply_secondarymotion_exists(self):
        """houdini_apply_secondarymotion must be a module-level attribute of kinefx_tools."""
        assert hasattr(kinefx_tools, "houdini_apply_secondarymotion")

    def test_apply_secondarymotion_is_async(self):
        """houdini_apply_secondarymotion must be an async coroutine function."""
        fn = getattr(kinefx_tools, "houdini_apply_secondarymotion", None)
        assert fn is not None
        assert asyncio.iscoroutinefunction(fn)


# ---------------------------------------------------------------------------
# MCP gate — require_approval must be True (PP12-109 gate)
# ---------------------------------------------------------------------------

class TestSecondarymotionToolIsGated:
    def _read_mcp_tool_meta(self, tool_name):
        mcp = kinefx_tools.mcp
        tools = mcp._tool_manager._tools
        assert tool_name in tools, f"{tool_name} not registered in MCP tool registry"
        return tools[tool_name].meta or {}

    def test_apply_secondarymotion_require_approval_true(self):
        """houdini_apply_secondarymotion must be decorated with meta={"require_approval": True}
        so the PP12-109 gate queues it for operator approval before execution."""
        meta = self._read_mcp_tool_meta("houdini_apply_secondarymotion")
        assert meta.get("require_approval") is True


# ---------------------------------------------------------------------------
# Bridge delegation contract
# ---------------------------------------------------------------------------

class TestApplySecondarymotionBridgeExecute:
    @pytest.mark.asyncio
    async def test_calls_get_bridge_with_ctx(self, mock_ctx, mock_bridge):
        """_get_bridge must be called exactly once with ctx (PP12-110 convention)."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge) as mock_get:
            await kinefx_tools.houdini_apply_secondarymotion(
                mock_ctx,
                node="/obj/geo1/skel",
            )
            mock_get.assert_called_once_with(mock_ctx)

    @pytest.mark.asyncio
    async def test_calls_execute_with_apply_secondarymotion_command(self, mock_ctx, mock_bridge):
        """bridge.execute must be called with the command string 'apply_secondarymotion'."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.houdini_apply_secondarymotion(
                mock_ctx,
                node="/obj/geo1/skel",
            )
        args = mock_bridge.execute.call_args
        assert args[0][0] == "apply_secondarymotion"

    @pytest.mark.asyncio
    async def test_execute_called_once(self, mock_ctx, mock_bridge):
        """bridge.execute must be called exactly once per wrapper invocation."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.houdini_apply_secondarymotion(
                mock_ctx,
                node="/obj/geo1/skel",
            )
        mock_bridge.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_called_with_full_params_dict(self, mock_ctx, mock_bridge):
        """bridge.execute must receive the full params dict with node, joints, params, dest."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.houdini_apply_secondarymotion(
                mock_ctx,
                node="/obj/geo1/skel",
                joints=["a", "b"],
                params={"effect": "jiggle", "stiffness": 0.4},
                dest="/obj",
            )
        mock_bridge.execute.assert_called_once_with(
            "apply_secondarymotion",
            {
                "node": "/obj/geo1/skel",
                "joints": ["a", "b"],
                "params": {"effect": "jiggle", "stiffness": 0.4},
                "dest": "/obj",
            },
        )

    @pytest.mark.asyncio
    async def test_joints_forwarded_unchanged(self, mock_ctx, mock_bridge):
        """joints list must be forwarded verbatim — the wrapper must not pre-process it."""
        joints_sentinel = ["hip", "spine", "tail_05"]
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.houdini_apply_secondarymotion(
                mock_ctx,
                node="/obj/geo1/skel",
                joints=joints_sentinel,
            )
        _cmd, params = mock_bridge.execute.call_args[0]
        assert params["joints"] == joints_sentinel, (
            f"joints must be forwarded unchanged; got {params['joints']!r}"
        )

    @pytest.mark.asyncio
    async def test_params_dict_forwarded_unchanged(self, mock_ctx, mock_bridge):
        """params dict must be forwarded verbatim — the wrapper must not mutate it."""
        effect_params = {"effect": "jiggle", "stiffness": 0.4}
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.houdini_apply_secondarymotion(
                mock_ctx,
                node="/obj/geo1/skel",
                joints=["a", "b"],
                params=effect_params,
            )
        _cmd, call_params = mock_bridge.execute.call_args[0]
        assert call_params["params"] == effect_params, (
            f"params must be forwarded unchanged; got {call_params['params']!r}"
        )

    @pytest.mark.asyncio
    async def test_wrapper_passes_bridge_result_through_verbatim(self, mock_ctx, mock_bridge):
        """houdini_apply_secondarymotion must return the bridge result verbatim.
        The wrapper is a thin passthrough — it must not mutate or repackage."""
        sentinel_result = {
            "ok": True,
            "node": "/obj/geo1/mcp_secondarymotion1",
            "effect": "jiggle",
            "affected_joints": ["a", "b"],
        }
        mock_bridge.execute = AsyncMock(return_value=sentinel_result)
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            result = await kinefx_tools.houdini_apply_secondarymotion(
                mock_ctx,
                node="/obj/geo1/skel",
                joints=["a", "b"],
                params={"effect": "jiggle", "stiffness": 0.4},
            )
        assert result == sentinel_result, (
            f"Bridge result must pass through verbatim; got {result!r}"
        )

    @pytest.mark.asyncio
    async def test_fail_loud_returns_ok_false_no_exception(self, mock_ctx, mock_bridge_error):
        """An error envelope from the bridge must be returned as-is (FR-2 fail-loud).
        The wrapper must NOT raise — it surfaces the error dict verbatim."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_error):
            result = await kinefx_tools.houdini_apply_secondarymotion(
                mock_ctx,
                node="/obj/missing",
            )
        assert result.get("ok") is False
        assert "error" in result and result["error"]
