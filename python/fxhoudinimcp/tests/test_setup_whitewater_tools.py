"""Spec-bound mock-bridge tests for fork F2 setup_whitewater wrapper.

Discipline (mcp-fork-build-lessons):
  - MagicMock(spec=HoudiniBridge); bare MagicMock() is BANNED (auto-passes a non-existent .call).
  - _get_bridge is asserted to receive ctx.
  - command string == handler registration == params keys invariant: execute() called with
    "workflow.setup_whitewater" + the exact params dict.

testVerificationSurface: pytest-model
unitId: b4-w2-f2
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

from fxhoudinimcp.tools import workflows  # noqa: E402
from fxhoudinimcp.bridge import HoudiniBridge  # noqa: E402


@pytest.fixture()
def mock_bridge():
    bridge = MagicMock(spec=HoudiniBridge)
    bridge.execute = AsyncMock(return_value={
        "success": True,
        "geo_path": "/obj/whitewater_sim",
        "flip_solver_path": "/obj/whitewater_sim/flip_solver",
        "source_path": "/obj/whitewater_sim/whitewater_source",
        "solver_path": "/obj/whitewater_sim/whitewater_solver",
        "cache_path": "/obj/whitewater_sim/whitewater_cache",
        "all_nodes": [],
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
    def test_setup_whitewater_exists(self):
        assert hasattr(workflows, "setup_whitewater")

    def test_setup_whitewater_is_async(self):
        fn = getattr(workflows, "setup_whitewater", None)
        assert fn is not None
        assert asyncio.iscoroutinefunction(fn)


class TestToolIsGated:
    def test_setup_whitewater_is_mutating_by_default(self):
        tools = workflows.mcp._tool_manager._tools
        assert "setup_whitewater" in tools
        meta = tools["setup_whitewater"].meta or {}
        assert meta.get("require_approval") is not False


class TestBridgeExecute:
    @pytest.mark.asyncio
    async def test_calls_get_bridge_with_ctx(self, mock_ctx, mock_bridge):
        with patch.object(workflows, "_get_bridge", return_value=mock_bridge) as mock_get:
            await workflows.setup_whitewater(mock_ctx, source_geo="/obj/geo1")
            mock_get.assert_called_once_with(mock_ctx)

    @pytest.mark.asyncio
    async def test_command_string_and_default_params(self, mock_ctx, mock_bridge):
        with patch.object(workflows, "_get_bridge", return_value=mock_bridge):
            await workflows.setup_whitewater(mock_ctx, source_geo="/obj/geo1")
        mock_bridge.execute.assert_called_once_with(
            "workflow.setup_whitewater",
            {"source_geo": "/obj/geo1", "particle_sep": 0.2,
             "name": "whitewater_sim", "foam_amount": 1.0},
        )

    @pytest.mark.asyncio
    async def test_all_params_passthrough(self, mock_ctx, mock_bridge):
        with patch.object(workflows, "_get_bridge", return_value=mock_bridge):
            await workflows.setup_whitewater(
                mock_ctx, source_geo="/obj/splash", particle_sep=0.1,
                name="foam", foam_amount=3.0)
        mock_bridge.execute.assert_called_once_with(
            "workflow.setup_whitewater",
            {"source_geo": "/obj/splash", "particle_sep": 0.1,
             "name": "foam", "foam_amount": 3.0},
        )

    @pytest.mark.asyncio
    async def test_passthrough_returns_paths(self, mock_ctx, mock_bridge):
        with patch.object(workflows, "_get_bridge", return_value=mock_bridge):
            result = await workflows.setup_whitewater(mock_ctx)
        assert result["solver_path"] == "/obj/whitewater_sim/whitewater_solver"
        assert result["source_path"] == "/obj/whitewater_sim/whitewater_source"
        assert result["cache_path"] == "/obj/whitewater_sim/whitewater_cache"
        assert result["success"] is True
