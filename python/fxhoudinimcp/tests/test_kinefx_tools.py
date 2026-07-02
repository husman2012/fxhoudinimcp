"""
test_kinefx_tools.py -- spec-bound mock-bridge tests for kinefx_tools (PP12-110b/c fix006).

Fix Dispatch #006 -- bridge API: rewrites tests from bridge.call() to bridge.execute().
MagicMock(spec=HoudiniBridge) ensures bridge.call raises AttributeError -- regression guard.

testVerificationSurface: pytest-model
unitId: pp12-110c
"""
from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if os.path.abspath(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, os.path.abspath(_PKG_ROOT))

from fxhoudinimcp.tools import kinefx_tools  # noqa: E402
from fxhoudinimcp.bridge import HoudiniBridge  # noqa: E402


@pytest.fixture()
def mock_bridge():
    bridge = MagicMock(spec=HoudiniBridge)
    bridge.execute = AsyncMock(return_value={"status": "ok", "data": {}})
    return bridge


@pytest.fixture()
def mock_ctx():
    return MagicMock()


class TestKinefxToolRegistration:
    def test_kinefx_probe_require_approval_false(self):
        mcp = kinefx_tools.mcp
        tools = mcp._tool_manager._tools
        assert "kinefx_probe" in tools
        assert tools["kinefx_probe"].meta.get("require_approval") is False

    def test_query_skeleton_require_approval_false(self):
        mcp = kinefx_tools.mcp
        tools = mcp._tool_manager._tools
        assert "query_skeleton" in tools
        assert tools["query_skeleton"].meta.get("require_approval") is False

    def test_inspect_apex_require_approval_false(self):
        mcp = kinefx_tools.mcp
        tools = mcp._tool_manager._tools
        assert "inspect_apex" in tools
        assert tools["inspect_apex"].meta.get("require_approval") is False


class TestSpecBoundMockRejectsCall:
    def test_bridge_call_raises_attribute_error(self, mock_bridge):
        with pytest.raises(AttributeError):
            _ = mock_bridge.call


class TestKinefxToolBridgeExecute:
    @pytest.mark.asyncio
    async def test_kinefx_probe_calls_execute_with_node_path(self, mock_ctx, mock_bridge):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge) as mock_get:
            await kinefx_tools.kinefx_probe(mock_ctx, node_path="/obj/geo1")
            mock_get.assert_called_once_with(mock_ctx)
        mock_bridge.execute.assert_called_once_with("kinefx_probe", {"node_path": "/obj/geo1"})

    @pytest.mark.asyncio
    async def test_query_skeleton_calls_execute_with_node_and_frame(self, mock_ctx, mock_bridge):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge) as mock_get:
            await kinefx_tools.query_skeleton(mock_ctx, node_path="/obj/rig", frame=1.0)
            mock_get.assert_called_once_with(mock_ctx)
        mock_bridge.execute.assert_called_once_with("query_skeleton", {"node_path": "/obj/rig", "frame": 1.0})

    @pytest.mark.asyncio
    async def test_query_skeleton_frame_defaults_to_none(self, mock_ctx, mock_bridge):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.query_skeleton(mock_ctx, node_path="/obj/rig")
        mock_bridge.execute.assert_called_once_with("query_skeleton", {"node_path": "/obj/rig", "frame": None})

    @pytest.mark.asyncio
    async def test_inspect_apex_calls_execute_with_node_path(self, mock_ctx, mock_bridge):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge) as mock_get:
            await kinefx_tools.inspect_apex(mock_ctx, node_path="/obj/apex_graph")
            mock_get.assert_called_once_with(mock_ctx)
        mock_bridge.execute.assert_called_once_with("inspect_apex", {"node_path": "/obj/apex_graph"})

    @pytest.mark.asyncio
    async def test_bridge_returns_dict_result(self, mock_ctx, mock_bridge):
        mock_bridge.execute.return_value = {"count": 3, "joints": []}
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            result = await kinefx_tools.query_skeleton(mock_ctx, node_path="/obj/rig")
        assert result is not None


class TestKinefxToolsAreAsync:
    def test_kinefx_probe_is_coroutine_function(self):
        import asyncio
        assert asyncio.iscoroutinefunction(kinefx_tools.kinefx_probe)

    def test_query_skeleton_is_coroutine_function(self):
        import asyncio
        assert asyncio.iscoroutinefunction(kinefx_tools.query_skeleton)

    def test_inspect_apex_is_coroutine_function(self):
        import asyncio
        assert asyncio.iscoroutinefunction(kinefx_tools.inspect_apex)
