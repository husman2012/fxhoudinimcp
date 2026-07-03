"""Spec-bound mock-bridge tests for fork F5 get_heightfield_layer_stats wrapper.

Discipline (mcp-fork-build-lessons):
  - MagicMock(spec=HoudiniBridge); bare MagicMock() is BANNED (auto-passes a non-existent .call).
  - _get_bridge is asserted to receive ctx.
  - command string == handler registration == params keys == handler kwargs invariant:
    execute() called with "geometry.get_heightfield_layer_stats" + the exact params dict.
  - READ-ONLY tool: require_approval IS False (ungated), the inverse of the mutating factories.

testVerificationSurface: pytest-model
unitId: b4-w4-f5
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

from fxhoudinimcp.tools import geometry  # noqa: E402
from fxhoudinimcp.bridge import HoudiniBridge  # noqa: E402


@pytest.fixture()
def mock_bridge():
    bridge = MagicMock(spec=HoudiniBridge)
    bridge.execute = AsyncMock(return_value={
        "node_path": "/obj/hf/heightfield1",
        "layer": "height",
        "resolution": [500, 500, 1],
        "voxel_count": 250000,
        "min": -62.9691,
        "max": 84.6028,
        "mean": 2.9157,
        "sampled": False,
        "sample_count": 250000,
    })
    return bridge


@pytest.fixture()
def mock_ctx():
    return MagicMock()


class TestSpecBoundRejectsCall:
    def test_bridge_call_raises_attribute_error(self, mock_bridge):
        with pytest.raises(AttributeError):
            _ = mock_bridge.call


class TestToolExists:
    def test_get_heightfield_layer_stats_exists(self):
        assert hasattr(geometry, "get_heightfield_layer_stats")

    def test_get_heightfield_layer_stats_is_async(self):
        fn = getattr(geometry, "get_heightfield_layer_stats", None)
        assert fn is not None
        assert asyncio.iscoroutinefunction(fn)


class TestToolIsUngated:
    def test_read_only_tool_is_ungated(self):
        # READ-ONLY tools opt out of the 109 gate (FR-10): require_approval is explicitly False.
        tools = geometry.mcp._tool_manager._tools
        assert "get_heightfield_layer_stats" in tools
        meta = tools["get_heightfield_layer_stats"].meta or {}
        assert meta.get("require_approval") is False


class TestBridgeExecute:
    @pytest.mark.asyncio
    async def test_calls_get_bridge_with_ctx(self, mock_ctx, mock_bridge):
        with patch.object(geometry, "_get_bridge", return_value=mock_bridge) as mock_get:
            await geometry.get_heightfield_layer_stats(mock_ctx, node_path="/obj/hf/heightfield1")
            mock_get.assert_called_once_with(mock_ctx)

    @pytest.mark.asyncio
    async def test_command_string_and_default_params(self, mock_ctx, mock_bridge):
        with patch.object(geometry, "_get_bridge", return_value=mock_bridge):
            await geometry.get_heightfield_layer_stats(mock_ctx, node_path="/obj/hf/heightfield1")
        mock_bridge.execute.assert_called_once_with(
            "geometry.get_heightfield_layer_stats",
            {"node_path": "/obj/hf/heightfield1", "layer": "height", "histogram_bins": 0},
        )

    @pytest.mark.asyncio
    async def test_all_params_passthrough(self, mock_ctx, mock_bridge):
        with patch.object(geometry, "_get_bridge", return_value=mock_bridge):
            await geometry.get_heightfield_layer_stats(
                mock_ctx, node_path="/obj/terrain/erode1", layer="mask", histogram_bins=8)
        mock_bridge.execute.assert_called_once_with(
            "geometry.get_heightfield_layer_stats",
            {"node_path": "/obj/terrain/erode1", "layer": "mask", "histogram_bins": 8},
        )

    @pytest.mark.asyncio
    async def test_passthrough_returns_stats(self, mock_ctx, mock_bridge):
        with patch.object(geometry, "_get_bridge", return_value=mock_bridge):
            result = await geometry.get_heightfield_layer_stats(
                mock_ctx, node_path="/obj/hf/heightfield1")
        assert result["layer"] == "height"
        assert result["voxel_count"] == 250000
        assert result["min"] < result["max"]
