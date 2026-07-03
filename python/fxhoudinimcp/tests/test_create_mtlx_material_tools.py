"""Spec-bound mock-bridge tests for fork F1 create_mtlx_material wrapper.

Discipline (mcp-fork-build-lessons):
  - MagicMock(spec=HoudiniBridge); bare MagicMock() is BANNED (it auto-passes a non-existent
    .call). Regression guard asserts the spec-bound mock raises on .call.
  - _get_bridge is asserted to receive ctx (server.py requires it).
  - The command string == handler registration == params keys == handler kwargs invariant:
    verify execute() is called with "materials.create_mtlx_material" + the exact params dict.
  - Optional params (base_color/metalness/roughness/textures/normal_map) only ride the dict
    when provided.

testVerificationSurface: pytest-model
unitId: b4-w2-f1
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
        "materiallibrary_path": "/stage/rust_matlib",
        "surface_path": "/stage/rust_matlib/rust",
        "material_prim_path": "/materials/rust",
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
        "error": "Parent node not found: /stage/nope",
    })
    return bridge


class TestSpecBoundRejectsCall:
    def test_bridge_call_raises_attribute_error(self, mock_bridge):
        # A bare MagicMock would fabricate .call and ship a broken wrapper green.
        with pytest.raises(AttributeError):
            _ = mock_bridge.call


class TestToolExists:
    def test_create_mtlx_material_exists(self):
        assert hasattr(materials, "create_mtlx_material")

    def test_create_mtlx_material_is_async(self):
        fn = getattr(materials, "create_mtlx_material", None)
        assert fn is not None
        assert asyncio.iscoroutinefunction(fn)


class TestToolIsGated:
    def test_create_mtlx_material_is_mutating_by_default(self):
        # A mutating (109-gated) tool carries NO require_approval:False meta — unlike the
        # read-only tools that opt out (FR-10). Absence-or-not-False == gated.
        mcp = materials.mcp
        tools = mcp._tool_manager._tools
        assert "create_mtlx_material" in tools
        meta = tools["create_mtlx_material"].meta or {}
        assert meta.get("require_approval") is not False


class TestBridgeExecute:
    @pytest.mark.asyncio
    async def test_calls_get_bridge_with_ctx(self, mock_ctx, mock_bridge):
        with patch.object(materials, "_get_bridge", return_value=mock_bridge) as mock_get:
            await materials.create_mtlx_material(mock_ctx, name="rust")
            mock_get.assert_called_once_with(mock_ctx)

    @pytest.mark.asyncio
    async def test_command_string_and_minimal_params(self, mock_ctx, mock_bridge):
        with patch.object(materials, "_get_bridge", return_value=mock_bridge):
            await materials.create_mtlx_material(mock_ctx, name="rust")
        # command == handler registration name; minimal params carry name + the parent default.
        mock_bridge.execute.assert_called_once_with(
            "materials.create_mtlx_material",
            {"name": "rust", "parent_path": "/stage"},
        )

    @pytest.mark.asyncio
    async def test_optional_params_only_when_provided(self, mock_ctx, mock_bridge):
        with patch.object(materials, "_get_bridge", return_value=mock_bridge):
            await materials.create_mtlx_material(
                mock_ctx,
                name="rust",
                parent_path="/stage/look",
                base_color=[0.4, 0.2, 0.1],
                metalness=1.0,
                roughness=0.35,
                textures={"base_color": "$HIP/tex/albedo.exr"},
                normal_map="$HIP/tex/normal.exr",
            )
        mock_bridge.execute.assert_called_once_with(
            "materials.create_mtlx_material",
            {
                "name": "rust",
                "parent_path": "/stage/look",
                "base_color": [0.4, 0.2, 0.1],
                "metalness": 1.0,
                "roughness": 0.35,
                "textures": {"base_color": "$HIP/tex/albedo.exr"},
                "normal_map": "$HIP/tex/normal.exr",
            },
        )

    @pytest.mark.asyncio
    async def test_omitted_optionals_absent_from_params(self, mock_ctx, mock_bridge):
        # None optionals must NOT appear as keys (the handler defaults them).
        with patch.object(materials, "_get_bridge", return_value=mock_bridge):
            await materials.create_mtlx_material(mock_ctx, name="rust", metalness=1.0)
        sent = mock_bridge.execute.call_args[0][1]
        assert sent == {"name": "rust", "parent_path": "/stage", "metalness": 1.0}
        for k in ("base_color", "roughness", "textures", "normal_map"):
            assert k not in sent

    @pytest.mark.asyncio
    async def test_passthrough_verbatim(self, mock_ctx, mock_bridge):
        with patch.object(materials, "_get_bridge", return_value=mock_bridge):
            result = await materials.create_mtlx_material(mock_ctx, name="rust")
        assert result["material_prim_path"] == "/materials/rust"
        assert result["materiallibrary_path"] == "/stage/rust_matlib"
        assert result["surface_path"] == "/stage/rust_matlib/rust"

    @pytest.mark.asyncio
    async def test_fail_loud_returns_ok_false_no_exception(self, mock_ctx, mock_bridge_error):
        with patch.object(materials, "_get_bridge", return_value=mock_bridge_error):
            result = await materials.create_mtlx_material(
                mock_ctx, name="rust", parent_path="/stage/nope")
        assert result.get("ok") is False
        assert "error" in result and result["error"]
