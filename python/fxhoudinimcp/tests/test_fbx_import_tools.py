"""
test_fbx_import_tools.py -- spec-bound mock-bridge tests for PR-3 FBX import tools (pp12-110c fix006).

Fix Dispatch #006 -- bridge API:
  - Tests were using bare MagicMock() (bridge.call auto-created) -- false-green.
  - Rewritten to use MagicMock(spec=HoudiniBridge) and assert bridge.execute().
  - _get_bridge is asserted to receive ctx (server.py:21 requires it).

testVerificationSurface: pytest-model
unitId: pp12-110c
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


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def mock_bridge():
    bridge = MagicMock(spec=HoudiniBridge)
    bridge.execute = AsyncMock(return_value={
        "ok": True,
        "node": "/obj/geo1/mcp_fbxcharacterimport1",
        "skeleton": {"joints": 84, "has_skin_geo": True},
    })
    return bridge


@pytest.fixture()
def mock_bridge_anim():
    bridge = MagicMock(spec=HoudiniBridge)
    bridge.execute = AsyncMock(return_value={
        "ok": True,
        "node": "/obj/geo1/mcp_fbxanimimport1",
        "skeleton": {"joints": 84, "frame_range": [1, 90]},
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
        "error": "kinefx::fbxcharacterimport: FBX file not found: /bad/path.fbx",
    })
    return bridge


# ===========================================================================
# Tests: spec-bound mock rejects bridge.call (regression guard)
# ===========================================================================

class TestSpecBoundRejectsCall:
    def test_bridge_call_raises_attribute_error(self, mock_bridge):
        with pytest.raises(AttributeError):
            _ = mock_bridge.call

    def test_bridge_anim_call_raises_attribute_error(self, mock_bridge_anim):
        with pytest.raises(AttributeError):
            _ = mock_bridge_anim.call


# ===========================================================================
# Tests: tool existence
# ===========================================================================

class TestFbxToolsExist:
    def test_import_fbx_character_exists(self):
        assert hasattr(kinefx_tools, "houdini_import_fbx_character")

    def test_import_fbx_animation_exists(self):
        assert hasattr(kinefx_tools, "houdini_import_fbx_animation")

    def test_import_fbx_character_is_async(self):
        fn = getattr(kinefx_tools, "houdini_import_fbx_character", None)
        assert fn is not None
        assert asyncio.iscoroutinefunction(fn)

    def test_import_fbx_animation_is_async(self):
        fn = getattr(kinefx_tools, "houdini_import_fbx_animation", None)
        assert fn is not None
        assert asyncio.iscoroutinefunction(fn)


# ===========================================================================
# Tests: tool registration -- GATED (require_approval=True)
# ===========================================================================

class TestFbxToolsAreGated:
    def _read_mcp_tool_meta(self, tool_name):
        mcp = kinefx_tools.mcp
        tools = mcp._tool_manager._tools
        assert tool_name in tools, f"{tool_name} not registered in MCP tool registry"
        return tools[tool_name].meta or {}

    def test_import_fbx_character_require_approval_true(self):
        meta = self._read_mcp_tool_meta("houdini_import_fbx_character")
        assert meta.get("require_approval") is True

    def test_import_fbx_animation_require_approval_true(self):
        meta = self._read_mcp_tool_meta("houdini_import_fbx_animation")
        assert meta.get("require_approval") is True

    def test_read_only_tools_still_ungated(self):
        mcp = kinefx_tools.mcp
        tools = mcp._tool_manager._tools
        for name in ("kinefx_probe", "query_skeleton", "inspect_apex"):
            if name in tools:
                meta = tools[name].meta or {}
                assert meta.get("require_approval") is False


# ===========================================================================
# Tests: bridge.execute() routing for houdini_import_fbx_character
# ===========================================================================

class TestImportFbxCharacterBridgeExecute:

    @pytest.mark.asyncio
    async def test_calls_execute_with_import_fbx_character_command(self, mock_ctx, mock_bridge):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge) as mock_get:
            await kinefx_tools.houdini_import_fbx_character(
                mock_ctx,
                path="G:/BigMediumSmall/WorkersWelders_RIG_FBX_TPOSE.fbx",
                dest="/obj",
            )
            mock_get.assert_called_once_with(mock_ctx)
        args = mock_bridge.execute.call_args
        assert args[0][0] == "import_fbx_character"

    @pytest.mark.asyncio
    async def test_passes_path_in_params_dict(self, mock_ctx, mock_bridge):
        fbx_path = "G:/BigMediumSmall/WorkersWelders_RIG_FBX_TPOSE.fbx"
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.houdini_import_fbx_character(mock_ctx, path=fbx_path, dest="/obj")
        _, params = mock_bridge.execute.call_args[0]
        assert params["path"] == fbx_path

    @pytest.mark.asyncio
    async def test_passes_dest_in_params_dict(self, mock_ctx, mock_bridge):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.houdini_import_fbx_character(
                mock_ctx, path="G:/in/char.fbx", dest="/obj/character_rig"
            )
        _, params = mock_bridge.execute.call_args[0]
        assert params["dest"] == "/obj/character_rig"

    @pytest.mark.asyncio
    async def test_execute_called_with_full_params_dict(self, mock_ctx, mock_bridge):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.houdini_import_fbx_character(
                mock_ctx, path="G:/in/char.fbx", dest="/obj"
            )
        mock_bridge.execute.assert_called_once_with(
            "import_fbx_character", {"path": "G:/in/char.fbx", "dest": "/obj"}
        )

    @pytest.mark.asyncio
    async def test_returns_ok_true_on_success(self, mock_ctx, mock_bridge):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            result = await kinefx_tools.houdini_import_fbx_character(mock_ctx, path="G:/in/char.fbx")
        assert result is not None
        assert result.get("ok") is True

    @pytest.mark.asyncio
    async def test_returns_node_path_on_success(self, mock_ctx, mock_bridge):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            result = await kinefx_tools.houdini_import_fbx_character(mock_ctx, path="G:/in/char.fbx")
        assert "node" in result and result["node"]

    @pytest.mark.asyncio
    async def test_returns_skeleton_summary_on_success(self, mock_ctx, mock_bridge):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            result = await kinefx_tools.houdini_import_fbx_character(mock_ctx, path="G:/in/char.fbx")
        skeleton = result.get("skeleton")
        assert skeleton is not None
        assert "joints" in skeleton
        assert "has_skin_geo" in skeleton

    @pytest.mark.asyncio
    async def test_fail_loud_returns_ok_false_no_exception(self, mock_ctx, mock_bridge_error):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_error):
            result = await kinefx_tools.houdini_import_fbx_character(mock_ctx, path="/bad/path.fbx")
        assert result.get("ok") is False
        assert "error" in result and result["error"]


# ===========================================================================
# Tests: bridge.execute() routing for houdini_import_fbx_animation
# ===========================================================================

class TestImportFbxAnimationBridgeExecute:

    @pytest.mark.asyncio
    async def test_calls_execute_with_import_fbx_animation_command(self, mock_ctx, mock_bridge_anim):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_anim) as mock_get:
            await kinefx_tools.houdini_import_fbx_animation(
                mock_ctx,
                path="G:/BigMediumSmall/WorkersWelders_RIG_FBX_TPOSE.fbx",
                dest="/obj",
                cascadeur=False,
            )
            mock_get.assert_called_once_with(mock_ctx)
        args = mock_bridge_anim.execute.call_args[0]
        assert args[0] == "import_fbx_animation"

    @pytest.mark.asyncio
    async def test_passes_cascadeur_false_in_params_dict(self, mock_ctx, mock_bridge_anim):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_anim):
            await kinefx_tools.houdini_import_fbx_animation(
                mock_ctx, path="G:/in/clip.fbx", cascadeur=False
            )
        _, params = mock_bridge_anim.execute.call_args[0]
        assert params["cascadeur"] is False

    @pytest.mark.asyncio
    async def test_passes_cascadeur_true_in_params_dict(self, mock_ctx, mock_bridge_anim):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_anim):
            await kinefx_tools.houdini_import_fbx_animation(
                mock_ctx,
                path="G:/BigMediumSmall/WorkersWelders_RIG_FBX_TPOSE.fbx",
                cascadeur=True,
            )
        _, params = mock_bridge_anim.execute.call_args[0]
        assert params["cascadeur"] is True

    @pytest.mark.asyncio
    async def test_execute_called_with_full_params_dict(self, mock_ctx, mock_bridge_anim):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_anim):
            await kinefx_tools.houdini_import_fbx_animation(
                mock_ctx, path="G:/in/clip.fbx", dest="/obj", cascadeur=False
            )
        mock_bridge_anim.execute.assert_called_once_with(
            "import_fbx_animation", {"path": "G:/in/clip.fbx", "dest": "/obj", "cascadeur": False}
        )

    @pytest.mark.asyncio
    async def test_cascadeur_defaults_to_false(self, mock_ctx, mock_bridge_anim):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_anim):
            result = await kinefx_tools.houdini_import_fbx_animation(mock_ctx, path="G:/in/clip.fbx")
        assert result is not None

    @pytest.mark.asyncio
    async def test_returns_skeleton_with_frame_range(self, mock_ctx, mock_bridge_anim):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_anim):
            result = await kinefx_tools.houdini_import_fbx_animation(
                mock_ctx, path="G:/BigMediumSmall/clip.fbx", cascadeur=True,
            )
        assert result.get("ok") is True
        skeleton = result.get("skeleton")
        assert skeleton is not None
        assert "joints" in skeleton
        assert "frame_range" in skeleton

    @pytest.mark.asyncio
    async def test_fail_loud_returns_ok_false_no_exception(self, mock_ctx, mock_bridge_error):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_error):
            result = await kinefx_tools.houdini_import_fbx_animation(mock_ctx, path="/bad/path.fbx")
        assert result.get("ok") is False
        assert "error" in result and result["error"]


# ===========================================================================
# Tests: FR-12 verify-after-mutate
# ===========================================================================

class TestVerifyAfterMutate:

    @pytest.mark.asyncio
    async def test_character_import_skeleton_joints_count_is_integer(self, mock_ctx, mock_bridge):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            result = await kinefx_tools.houdini_import_fbx_character(mock_ctx, path="G:/in/char.fbx")
        joints = result.get("skeleton", {}).get("joints")
        assert isinstance(joints, int) and joints >= 0

    @pytest.mark.asyncio
    async def test_character_import_has_skin_geo_is_bool(self, mock_ctx, mock_bridge):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            result = await kinefx_tools.houdini_import_fbx_character(mock_ctx, path="G:/in/char.fbx")
        has_skin = result.get("skeleton", {}).get("has_skin_geo")
        assert isinstance(has_skin, bool)


# ===========================================================================
# Regression: FR-2 handler-side fail-loud (bridge envelope propagation)
# ===========================================================================

class TestFR2HandlerFailLoud:

    @pytest.fixture()
    def mock_bridge_invalid_dest(self):
        bridge = MagicMock(spec=HoudiniBridge)
        bridge.execute = AsyncMock(return_value={
            "ok": False, "error": "dest node not found: '/obj/does_not_exist'"
        })
        return bridge

    @pytest.fixture()
    def mock_bridge_createnode_fail(self):
        bridge = MagicMock(spec=HoudiniBridge)
        bridge.execute = AsyncMock(return_value={
            "ok": False, "error": "Failed to create node of type kinefx::fbxcharacterimport"
        })
        return bridge

    @pytest.fixture()
    def mock_bridge_geometry_fail(self):
        bridge = MagicMock(spec=HoudiniBridge)
        bridge.execute = AsyncMock(return_value={
            "ok": False, "error": "Invalid output index 1 for kinefx::fbxcharacterimport"
        })
        return bridge

    @pytest.mark.asyncio
    async def test_character_explicit_invalid_dest_returns_ok_false(
        self, mock_ctx, mock_bridge_invalid_dest
    ):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_invalid_dest):
            result = await kinefx_tools.houdini_import_fbx_character(
                mock_ctx, path="G:/in/char.fbx", dest="/obj/does_not_exist"
            )
        assert result.get("ok") is False
        assert "error" in result and result["error"]

    @pytest.mark.asyncio
    async def test_animation_explicit_invalid_dest_returns_ok_false(
        self, mock_ctx, mock_bridge_invalid_dest
    ):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_invalid_dest):
            result = await kinefx_tools.houdini_import_fbx_animation(
                mock_ctx, path="G:/in/clip.fbx", dest="/obj/does_not_exist"
            )
        assert result.get("ok") is False
        assert "error" in result and result["error"]

    @pytest.mark.asyncio
    async def test_character_createnode_failure_returns_ok_false(
        self, mock_ctx, mock_bridge_createnode_fail
    ):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_createnode_fail):
            result = await kinefx_tools.houdini_import_fbx_character(mock_ctx, path="G:/in/char.fbx")
        assert result.get("ok") is False
        assert "error" in result and result["error"]

    @pytest.mark.asyncio
    async def test_animation_createnode_failure_returns_ok_false(
        self, mock_ctx, mock_bridge_createnode_fail
    ):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_createnode_fail):
            result = await kinefx_tools.houdini_import_fbx_animation(mock_ctx, path="G:/in/clip.fbx")
        assert result.get("ok") is False
        assert "error" in result and result["error"]

    @pytest.mark.asyncio
    async def test_character_geometry_failure_returns_ok_false(
        self, mock_ctx, mock_bridge_geometry_fail
    ):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_geometry_fail):
            result = await kinefx_tools.houdini_import_fbx_character(mock_ctx, path="G:/in/char.fbx")
        assert result.get("ok") is False
        assert "error" in result and result["error"]

    @pytest.mark.asyncio
    async def test_animation_geometry_failure_returns_ok_false(
        self, mock_ctx, mock_bridge_geometry_fail
    ):
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_geometry_fail):
            result = await kinefx_tools.houdini_import_fbx_animation(mock_ctx, path="G:/in/clip.fbx")
        assert result.get("ok") is False
        assert "error" in result and result["error"]
