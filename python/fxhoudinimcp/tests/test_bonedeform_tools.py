"""
test_bonedeform_tools.py -- spec-bound mock-bridge tests for PP12-110 PR-4 bonedeform tool.

PR-4 discipline (mirrors PR-3 fix006 lessons):
  - Tests use MagicMock(spec=HoudiniBridge) — bare MagicMock() is BANNED.
  - Regression guard asserts spec-bound mock raises AttributeError on .call.
  - _get_bridge is asserted to receive ctx (server.py:21 requires it).
  - FR-2 fail-loud: error envelopes {ok: False} propagate verbatim, no raise.

bonedeform SOP input order (authoritative probed order, plan riskNotes):
  geo -> input0, rest -> input1, anim -> input2

testVerificationSurface: pytest-model
unitId: pp12-110d
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


@pytest.fixture()
def mock_bridge():
    bridge = MagicMock(spec=HoudiniBridge)
    # FR-12 contract shape (pp12-110d fix#2):
    #   skeleton: {joints, frame_range}   (not has_skin_geo — that's fbxcharacterimport's shape)
    #   validator: {cook_errors, deformed_points, has_capture_weight, note}
    bridge.execute = AsyncMock(return_value={
        "ok": True,
        "node": "/obj/geo1/mcp_bonedeform1",
        "skeleton": {"joints": 84, "frame_range": [1, 100]},
        "validator": {
            "cook_errors": [],
            "deformed_points": 105802,
            "has_capture_weight": True,
            "note": "verify-after-mutate",
        },
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
        "error": "bonedeform: captured geo not found at /obj/geo",
    })
    return bridge


class TestSpecBoundRejectsCall:
    def test_bridge_call_raises_attribute_error(self, mock_bridge):
        with pytest.raises(AttributeError):
            _ = mock_bridge.call


class TestBonedeformToolExists:
    def test_setup_bonedeform_exists(self):
        assert hasattr(kinefx_tools, "houdini_setup_bonedeform")

    def test_setup_bonedeform_is_async(self):
        fn = getattr(kinefx_tools, "houdini_setup_bonedeform", None)
        assert fn is not None
        assert asyncio.iscoroutinefunction(fn)


class TestBonedeformToolIsGated:
    def _read_mcp_tool_meta(self, tool_name):
        mcp = kinefx_tools.mcp
        tools = mcp._tool_manager._tools
        assert tool_name in tools, f"{tool_name} not registered in MCP tool registry"
        return tools[tool_name].meta or {}

    def test_setup_bonedeform_require_approval_true(self):
        meta = self._read_mcp_tool_meta("houdini_setup_bonedeform")
        assert meta.get("require_approval") is True


class TestSetupBonedeformBridgeExecute:
    @pytest.mark.asyncio
    async def test_calls_get_bridge_with_ctx(self, mock_ctx, mock_bridge):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge) as mock_get:
            await kinefx_tools.houdini_setup_bonedeform(
                mock_ctx,
                rest="/obj/rest",
                anim="/obj/anim",
                geo="/obj/geo",
                dest="/obj/d",
            )
            mock_get.assert_called_once_with(mock_ctx)

    @pytest.mark.asyncio
    async def test_calls_execute_with_setup_bonedeform_command(self, mock_ctx, mock_bridge):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.houdini_setup_bonedeform(
                mock_ctx,
                rest="/obj/rest",
                anim="/obj/anim",
                geo="/obj/geo",
                dest="/obj/d",
            )
        args = mock_bridge.execute.call_args
        assert args[0][0] == "setup_bonedeform"

    @pytest.mark.asyncio
    async def test_execute_called_with_full_params_dict(self, mock_ctx, mock_bridge):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.houdini_setup_bonedeform(
                mock_ctx,
                rest="/obj/rest",
                anim="/obj/anim",
                geo="/obj/geo",
                dest="/obj/d",
            )
        mock_bridge.execute.assert_called_once_with(
            "setup_bonedeform",
            {"rest": "/obj/rest", "anim": "/obj/anim", "geo": "/obj/geo", "dest": "/obj/d"},
        )

    @pytest.mark.asyncio
    async def test_fail_loud_returns_ok_false_no_exception(self, mock_ctx, mock_bridge_error):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_error):
            result = await kinefx_tools.houdini_setup_bonedeform(
                mock_ctx,
                rest="/obj/rest",
                anim="/obj/anim",
                geo="/obj/geo",
            )
        assert result.get("ok") is False
        assert "error" in result and result["error"]

    @pytest.mark.asyncio
    async def test_wrapper_passes_bridge_result_through_verbatim(self, mock_ctx, mock_bridge):
        """houdini_setup_bonedeform must return the bridge result verbatim (FR-12 shape).

        The wrapper is a thin passthrough; it must not mutate or repackage the
        bridge's return value.  Verifies the FR-12 return shape is preserved:
          skeleton: {joints, frame_range}
          validator: {cook_errors, deformed_points, has_capture_weight, note}
        """
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            result = await kinefx_tools.houdini_setup_bonedeform(
                mock_ctx,
                rest="/obj/rest",
                anim="/obj/anim",
                geo="/obj/geo",
                dest="/obj/d",
            )
        assert result.get("ok") is True
        # skeleton shape — FR-12: {joints, frame_range}
        skel = result.get("skeleton", {})
        assert "joints" in skel, "skeleton must have 'joints'"
        assert "frame_range" in skel, "skeleton must have 'frame_range' (not 'has_skin_geo')"
        assert "has_skin_geo" not in skel, "stale 'has_skin_geo' must not appear in setup_bonedeform skeleton"
        # validator shape — FR-12: {cook_errors, deformed_points, has_capture_weight, note}
        vld = result.get("validator", {})
        assert "cook_errors" in vld, "validator must have 'cook_errors'"
        assert "deformed_points" in vld, "validator must have 'deformed_points'"
        assert "has_capture_weight" in vld, "validator must have 'has_capture_weight'"
        assert "note" in vld, "validator must have 'note'"
        assert "ok" not in vld, "stale 'ok' key must not appear in setup_bonedeform validator"
        assert "warnings" not in vld, "stale 'warnings' key must not appear in setup_bonedeform validator"


class TestFR2HandlerFailLoud:
    @pytest.fixture()
    def mock_bridge_invalid_paths(self):
        bridge = MagicMock(spec=HoudiniBridge)
        bridge.execute = AsyncMock(return_value={
            "ok": False,
            "error": "bonedeform: rest skeleton path not found: '/obj/does_not_exist'",
        })
        return bridge

    @pytest.mark.asyncio
    async def test_invalid_rest_path_returns_ok_false(
        self, mock_ctx, mock_bridge_invalid_paths
    ):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_invalid_paths):
            result = await kinefx_tools.houdini_setup_bonedeform(
                mock_ctx,
                rest="/obj/does_not_exist",
                anim="/obj/anim",
                geo="/obj/geo",
            )
        assert result.get("ok") is False
        assert "error" in result and result["error"]
