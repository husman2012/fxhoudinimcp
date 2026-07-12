"""Spec-bound mock-bridge tests for fork F4 bind_usd_material wrapper.

Discipline (mcp-fork-build-lessons):
  - MagicMock(spec=HoudiniBridge); bare MagicMock() is BANNED (auto-passes a non-existent .call).
  - _get_bridge is asserted to receive ctx.
  - command string == handler registration == params keys == handler kwargs invariant:
    execute() called with "materials.bind_usd_material" + the exact params dict.

testVerificationSurface: pytest-model
unitId: b4-w3-f4
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

from fxhoudinimcp.tools import materials  # noqa: E402
from fxhoudinimcp.bridge import HoudiniBridge  # noqa: E402


@pytest.fixture()
def mock_bridge():
    bridge = MagicMock(spec=HoudiniBridge)
    bridge.execute = AsyncMock(return_value={
        "assignmaterial_path": "/stage/bind_material",
        "material_prim": "/materials/redmat",
        "geo_pattern": "/asset/geo/*",
        "bound_count": 1,
        "bound_prims": ["/asset/geo/mesh_0"],
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
        "error": "geo_pattern '/nope/*' matched no prims to bind '/materials/redmat'",
    })
    return bridge


class TestSpecBoundRejectsCall:
    def test_bridge_call_raises_attribute_error(self, mock_bridge):
        with pytest.raises(AttributeError):
            _ = mock_bridge.call


class TestToolExists:
    def test_bind_usd_material_exists(self):
        assert hasattr(materials, "bind_usd_material")

    def test_bind_usd_material_is_async(self):
        fn = getattr(materials, "bind_usd_material", None)
        assert fn is not None
        assert asyncio.iscoroutinefunction(fn)


class TestToolIsGated:
    def test_bind_usd_material_is_mutating_by_default(self):
        tools = materials.mcp._tool_manager._tools
        assert "bind_usd_material" in tools
        meta = tools["bind_usd_material"].meta or {}
        assert meta.get("require_approval") is not False


class TestBridgeExecute:
    @pytest.mark.asyncio
    async def test_calls_get_bridge_with_ctx(self, mock_ctx, mock_bridge):
        with patch.object(materials, "_get_bridge", return_value=mock_bridge) as mock_get:
            await materials.bind_usd_material(
                mock_ctx, input_lop="/stage/matlib",
                geo_pattern="/asset/geo/*", material_prim="/materials/redmat")
            mock_get.assert_called_once_with(mock_ctx)

    @pytest.mark.asyncio
    async def test_command_string_and_default_name(self, mock_ctx, mock_bridge):
        with patch.object(materials, "_get_bridge", return_value=mock_bridge):
            await materials.bind_usd_material(
                mock_ctx, input_lop="/stage/matlib",
                geo_pattern="/asset/geo/*", material_prim="/materials/redmat")
        mock_bridge.execute.assert_called_once_with(
            "materials.bind_usd_material",
            {"input_lop": "/stage/matlib", "geo_pattern": "/asset/geo/*",
             "material_prim": "/materials/redmat", "name": "bind_material"},
        )

    @pytest.mark.asyncio
    async def test_all_params_passthrough(self, mock_ctx, mock_bridge):
        with patch.object(materials, "_get_bridge", return_value=mock_bridge):
            await materials.bind_usd_material(
                mock_ctx, input_lop="/stage/look", geo_pattern="/table.collections:donuts",
                material_prim="/materials/glaze", name="glaze_bind")
        mock_bridge.execute.assert_called_once_with(
            "materials.bind_usd_material",
            {"input_lop": "/stage/look", "geo_pattern": "/table.collections:donuts",
             "material_prim": "/materials/glaze", "name": "glaze_bind"},
        )

    @pytest.mark.asyncio
    async def test_passthrough_returns_binding(self, mock_ctx, mock_bridge):
        with patch.object(materials, "_get_bridge", return_value=mock_bridge):
            result = await materials.bind_usd_material(
                mock_ctx, input_lop="/stage/matlib",
                geo_pattern="/asset/geo/*", material_prim="/materials/redmat")
        assert result["assignmaterial_path"] == "/stage/bind_material"
        assert result["material_prim"] == "/materials/redmat"
        assert result["bound_count"] == 1

    @pytest.mark.asyncio
    async def test_fail_loud_returns_ok_false_no_exception(self, mock_ctx, mock_bridge_error):
        with patch.object(materials, "_get_bridge", return_value=mock_bridge_error):
            result = await materials.bind_usd_material(
                mock_ctx, input_lop="/stage/matlib",
                geo_pattern="/nope/*", material_prim="/materials/redmat")
        assert result.get("ok") is False
        assert "error" in result and result["error"]
