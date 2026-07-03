"""Spec-bound mock-bridge tests for fork F3 setup_constraint_network wrapper.

Discipline (mcp-fork-build-lessons):
  - MagicMock(spec=HoudiniBridge); bare MagicMock() is BANNED (auto-passes a non-existent .call).
  - _get_bridge is asserted to receive ctx.
  - command string == handler registration == params keys invariant: execute() called with
    "workflow.setup_constraint_network" + the exact params dict.

testVerificationSurface: pytest-model
unitId: b4-w2-f3
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
        "geo_path": "/obj/constraint_sim",
        "fracture_path": "/obj/constraint_sim/fracture1",
        "constraint_props_path": "/obj/constraint_sim/constraint_props1",
        "solver_path": "/obj/constraint_sim/rbd_solver1",
        "cache_path": "/obj/constraint_sim/file_cache1",
        "constraint_type": "glue",
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
    def test_setup_constraint_network_exists(self):
        assert hasattr(workflows, "setup_constraint_network")

    def test_setup_constraint_network_is_async(self):
        fn = getattr(workflows, "setup_constraint_network", None)
        assert fn is not None
        assert asyncio.iscoroutinefunction(fn)


class TestToolIsGated:
    def test_setup_constraint_network_is_mutating_by_default(self):
        tools = workflows.mcp._tool_manager._tools
        assert "setup_constraint_network" in tools
        meta = tools["setup_constraint_network"].meta or {}
        assert meta.get("require_approval") is not False


class TestBridgeExecute:
    @pytest.mark.asyncio
    async def test_calls_get_bridge_with_ctx(self, mock_ctx, mock_bridge):
        with patch.object(workflows, "_get_bridge", return_value=mock_bridge) as mock_get:
            await workflows.setup_constraint_network(mock_ctx, geo_path="/obj/geo1")
            mock_get.assert_called_once_with(mock_ctx)

    @pytest.mark.asyncio
    async def test_command_string_and_default_params(self, mock_ctx, mock_bridge):
        with patch.object(workflows, "_get_bridge", return_value=mock_bridge):
            await workflows.setup_constraint_network(mock_ctx, geo_path="/obj/geo1")
        mock_bridge.execute.assert_called_once_with(
            "workflow.setup_constraint_network",
            {"geo_path": "/obj/geo1", "constraint_type": "glue",
             "strength": 10000.0, "ground": True, "name": "constraint_sim"},
        )

    @pytest.mark.asyncio
    async def test_all_params_passthrough(self, mock_ctx, mock_bridge):
        with patch.object(workflows, "_get_bridge", return_value=mock_bridge):
            await workflows.setup_constraint_network(
                mock_ctx, geo_path="/obj/rock", constraint_type="soft",
                strength=25.0, ground=False, name="rubble")
        mock_bridge.execute.assert_called_once_with(
            "workflow.setup_constraint_network",
            {"geo_path": "/obj/rock", "constraint_type": "soft",
             "strength": 25.0, "ground": False, "name": "rubble"},
        )

    @pytest.mark.asyncio
    async def test_passthrough_returns_paths(self, mock_ctx, mock_bridge):
        with patch.object(workflows, "_get_bridge", return_value=mock_bridge):
            result = await workflows.setup_constraint_network(mock_ctx)
        assert result["constraint_props_path"] == "/obj/constraint_sim/constraint_props1"
        assert result["solver_path"] == "/obj/constraint_sim/rbd_solver1"
        assert result["constraint_type"] == "glue"
