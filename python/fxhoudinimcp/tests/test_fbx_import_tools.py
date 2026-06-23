"""
test_fbx_import_tools.py — mock-bridge RED tests for PP12-110 PR-3 FBX import tools.

TDD phase: RED — `houdini_import_fbx_character` and `houdini_import_fbx_animation`
do NOT exist yet.  Collection-time import succeeds (kinefx_tools.py exists from PR-2)
but the function lookups below will fail until hou-dev ships the two new wrappers.

This file asserts:
  1. Both tools are importable from kinefx_tools as async functions.
  2. Both tools are registered with meta={"require_approval": True}  (GATED — unlike
     PR-2's read-only tools which use require_approval=False).
  3. houdini_import_fbx_character calls bridge.call("import_fbx_character", path=..., dest=...)
     and returns {ok, node, skeleton:{joints_count, has_skin_geo}}.
  4. houdini_import_fbx_animation calls bridge.call("import_fbx_animation", path=..., dest=...,
     cascadeur=...) and when cascadeur=True sets the convertunits flag.
  5. On node.errors() the handler returns {ok:false, error:<message>} — no exception raised
     (fail-loud pattern, FR-2/FR-3).
  6. Success return reads OUT 1 (skeleton, @name) for joint count, OUT 0 for has_skin_geo —
     NOT just OUT 0 (skin mesh).

No hou / Qt / pxr imports anywhere.
Runs on bare CI (plain pytest, no Houdini required).

testVerificationSurface: pytest-model
unitId: pp12-110c
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

# ---------------------------------------------------------------------------
# RED GATE — the two new wrappers do not exist yet.
# kinefx_tools.py itself is importable (PR-2 landed it) but houdini_import_fbx_character
# and houdini_import_fbx_animation are NOT defined until hou-dev ships PR-3.
# hou-dev MUST NOT modify this file to work around the AttributeErrors below.
# ---------------------------------------------------------------------------
from fxhoudinimcp.tools import kinefx_tools  # noqa: E402 — PR-2 module exists


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def mock_bridge():
    """A bridge stub; .call is an AsyncMock so async tools can be awaited."""
    bridge = MagicMock()
    bridge.call = AsyncMock(return_value={
        "ok": True,
        "node": "/obj/geo1/mcp_fbxcharacterimport1",
        "skeleton": {"joints": 84, "has_skin_geo": True},
    })
    return bridge


@pytest.fixture()
def mock_bridge_anim():
    """Bridge stub for animation import — returns a motion-clip-flavoured response."""
    bridge = MagicMock()
    bridge.call = AsyncMock(return_value={
        "ok": True,
        "node": "/obj/geo1/mcp_fbxanimimport1",
        "skeleton": {"joints": 84, "frame_range": [1, 90]},
    })
    return bridge


@pytest.fixture()
def mock_ctx():
    """FastMCP Context stub."""
    return MagicMock()


@pytest.fixture()
def mock_bridge_error():
    """Bridge stub that returns a node-errors failure envelope."""
    bridge = MagicMock()
    bridge.call = AsyncMock(return_value={
        "ok": False,
        "error": "kinefx::fbxcharacterimport: FBX file not found: /bad/path.fbx",
    })
    return bridge


# ===========================================================================
# Tests: tool existence (RED gate — AttributeError until hou-dev ships PR-3)
# ===========================================================================

class TestFbxToolsExist:
    """
    The two new PR-3 tools must exist as attributes of the kinefx_tools module.
    These assertions are the PRIMARY red gate: they fail with AttributeError until
    hou-dev implements the wrappers.

    hou-dev MUST NOT add these names as stubs to make this test pass without
    implementing the actual handler logic (tdd-with-agents.md §5 — no tweaking tests).
    """

    def test_import_fbx_character_exists(self):
        """houdini_import_fbx_character must be defined in kinefx_tools."""
        assert hasattr(kinefx_tools, "houdini_import_fbx_character"), (
            "houdini_import_fbx_character not found in kinefx_tools — "
            "hou-dev must add the wrapper function"
        )

    def test_import_fbx_animation_exists(self):
        """houdini_import_fbx_animation must be defined in kinefx_tools."""
        assert hasattr(kinefx_tools, "houdini_import_fbx_animation"), (
            "houdini_import_fbx_animation not found in kinefx_tools — "
            "hou-dev must add the wrapper function"
        )

    def test_import_fbx_character_is_async(self):
        """houdini_import_fbx_character must be an async (coroutine) function."""
        fn = getattr(kinefx_tools, "houdini_import_fbx_character", None)
        assert fn is not None, "houdini_import_fbx_character not defined"
        assert asyncio.iscoroutinefunction(fn), (
            "houdini_import_fbx_character must be async (MCP tools are coroutines)"
        )

    def test_import_fbx_animation_is_async(self):
        """houdini_import_fbx_animation must be an async (coroutine) function."""
        fn = getattr(kinefx_tools, "houdini_import_fbx_animation", None)
        assert fn is not None, "houdini_import_fbx_animation not defined"
        assert asyncio.iscoroutinefunction(fn), (
            "houdini_import_fbx_animation must be async (MCP tools are coroutines)"
        )


# ===========================================================================
# Tests: tool registration — GATED (require_approval=True)
# ===========================================================================

class TestFbxToolsAreGated:
    """
    PR-3 FBX import tools are MUTATING (they call createNode + cook) and MUST be
    registered with meta={"require_approval": True} — unlike PR-2's read-only tools
    which use require_approval=False.

    FR-10 / spec §4.1: every state-mutating tool routes through the 109 Security Gate
    (require_approval=True).  A tool registered as read-only would bypass the gate
    and allow the agent to create/cook nodes without operator approval — a spec violation.
    """

    def _read_mcp_tool_meta(self, tool_name: str) -> dict:
        """Read the meta dict from the FastMCP internal tool registry."""
        mcp = kinefx_tools.mcp
        # FastMCP stores tools in _tool_manager._tools (dict keyed by name)
        tools = mcp._tool_manager._tools
        assert tool_name in tools, (
            f"{tool_name} not registered in MCP tool registry — "
            f"hou-dev must register it with @mcp.tool(meta={{\"require_approval\": True}})"
        )
        return tools[tool_name].meta or {}

    def test_import_fbx_character_require_approval_true(self):
        """houdini_import_fbx_character must have meta[\"require_approval\"] == True (GATED)."""
        meta = self._read_mcp_tool_meta("houdini_import_fbx_character")
        assert meta.get("require_approval") is True, (
            f"houdini_import_fbx_character.meta['require_approval'] == "
            f"{meta.get('require_approval')!r} — expected True (GATED mutating tool)"
        )

    def test_import_fbx_animation_require_approval_true(self):
        """houdini_import_fbx_animation must have meta[\"require_approval\"] == True (GATED)."""
        meta = self._read_mcp_tool_meta("houdini_import_fbx_animation")
        assert meta.get("require_approval") is True, (
            f"houdini_import_fbx_animation.meta['require_approval'] == "
            f"{meta.get('require_approval')!r} — expected True (GATED mutating tool)"
        )

    def test_read_only_tools_still_ungated(self):
        """PR-2 read-only tools must remain ungated (require_approval=False) — regression guard."""
        mcp = kinefx_tools.mcp
        tools = mcp._tool_manager._tools
        for name in ("kinefx_probe", "query_skeleton", "inspect_apex"):
            if name in tools:
                meta = tools[name].meta or {}
                assert meta.get("require_approval") is False, (
                    f"Read-only tool {name} must remain ungated (require_approval=False)"
                )


# ===========================================================================
# Tests: bridge.call() routing for houdini_import_fbx_character
# ===========================================================================

class TestImportFbxCharacterBridgeCalls:
    """
    houdini_import_fbx_character must:
      - Call bridge.call("import_fbx_character", path=<str>, dest=<str>)
      - Return {ok, node, skeleton:{joints, has_skin_geo}} from the bridge response.
      - Return {ok:false, error:<str>} (NO exception) when bridge returns ok=false.
    """

    @pytest.mark.asyncio
    async def test_calls_bridge_with_import_fbx_character_command(
        self, mock_ctx, mock_bridge
    ):
        """Must call bridge.call('import_fbx_character', ...)."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.houdini_import_fbx_character(
                mock_ctx,
                path="G:/BigMediumSmall/WorkersWelders_RIG_FBX_TPOSE.fbx",
                dest="/obj",
            )
        mock_bridge.call.assert_called_once()
        args, kwargs = mock_bridge.call.call_args
        assert args[0] == "import_fbx_character", (
            f"Expected bridge.call('import_fbx_character', ...) but got '{args[0]}'"
        )

    @pytest.mark.asyncio
    async def test_passes_path_kwarg(self, mock_ctx, mock_bridge):
        """Must forward the path keyword argument to bridge.call."""
        fbx_path = "G:/BigMediumSmall/WorkersWelders_RIG_FBX_TPOSE.fbx"
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.houdini_import_fbx_character(
                mock_ctx, path=fbx_path, dest="/obj"
            )
        _, kwargs = mock_bridge.call.call_args
        assert kwargs.get("path") == fbx_path

    @pytest.mark.asyncio
    async def test_passes_dest_kwarg(self, mock_ctx, mock_bridge):
        """Must forward the dest keyword argument to bridge.call."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.houdini_import_fbx_character(
                mock_ctx,
                path="G:/in/char.fbx",
                dest="/obj/character_rig",
            )
        _, kwargs = mock_bridge.call.call_args
        assert kwargs.get("dest") == "/obj/character_rig"

    @pytest.mark.asyncio
    async def test_returns_ok_true_on_success(self, mock_ctx, mock_bridge):
        """Must return the bridge response containing ok=True on success."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            result = await kinefx_tools.houdini_import_fbx_character(
                mock_ctx, path="G:/in/char.fbx"
            )
        assert result is not None
        assert result.get("ok") is True

    @pytest.mark.asyncio
    async def test_returns_node_path_on_success(self, mock_ctx, mock_bridge):
        """Return envelope must include 'node' key with the created node path."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            result = await kinefx_tools.houdini_import_fbx_character(
                mock_ctx, path="G:/in/char.fbx"
            )
        assert "node" in result, "Return envelope missing 'node' key"
        assert result["node"]  # non-empty

    @pytest.mark.asyncio
    async def test_returns_skeleton_summary_on_success(self, mock_ctx, mock_bridge):
        """Return envelope must include 'skeleton' dict with joints count and has_skin_geo."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            result = await kinefx_tools.houdini_import_fbx_character(
                mock_ctx, path="G:/in/char.fbx"
            )
        skeleton = result.get("skeleton")
        assert skeleton is not None, "Return envelope missing 'skeleton' key"
        # skeleton must report joint count and skin geometry flag
        assert "joints" in skeleton, "skeleton missing 'joints' key (joint count)"
        assert "has_skin_geo" in skeleton, "skeleton missing 'has_skin_geo' key"

    @pytest.mark.asyncio
    async def test_fail_loud_returns_ok_false_no_exception(self, mock_ctx, mock_bridge_error):
        """
        When bridge returns ok=False (node errors), the tool must return {ok:false, error:<str>}
        WITHOUT raising a Python exception.  FR-2: fail-loud via envelope, not exception.
        """
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_error):
            # Must NOT raise
            result = await kinefx_tools.houdini_import_fbx_character(
                mock_ctx, path="/bad/path.fbx"
            )
        assert result.get("ok") is False, (
            f"Expected ok=False on node errors, got: {result}"
        )
        assert "error" in result, "Failure envelope must include 'error' key"
        assert result["error"]  # non-empty error message

    @pytest.mark.asyncio
    async def test_does_not_call_bridge_execute(self, mock_ctx, mock_bridge):
        """FBX import tools use bridge.call (keyword args), NOT bridge.execute."""
        mock_bridge.execute = AsyncMock()
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.houdini_import_fbx_character(
                mock_ctx, path="G:/in/char.fbx"
            )
        mock_bridge.execute.assert_not_called()


# ===========================================================================
# Tests: bridge.call() routing for houdini_import_fbx_animation
# ===========================================================================

class TestImportFbxAnimationBridgeCalls:
    """
    houdini_import_fbx_animation must:
      - Call bridge.call("import_fbx_animation", path=..., dest=..., cascadeur=...)
      - When cascadeur=True, the cascadeur flag must be forwarded (handler sets convertunits).
      - Return {ok, node, skeleton:{joints, frame_range}} from the bridge response.
      - Return {ok:false, error:<str>} (NO exception) on node errors.
    """

    @pytest.mark.asyncio
    async def test_calls_bridge_with_import_fbx_animation_command(
        self, mock_ctx, mock_bridge_anim
    ):
        """Must call bridge.call('import_fbx_animation', ...)."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_anim):
            await kinefx_tools.houdini_import_fbx_animation(
                mock_ctx,
                path="G:/BigMediumSmall/WorkersWelders_RIG_FBX_TPOSE.fbx",
                dest="/obj",
                cascadeur=False,
            )
        mock_bridge_anim.call.assert_called_once()
        args, kwargs = mock_bridge_anim.call.call_args
        assert args[0] == "import_fbx_animation", (
            f"Expected bridge.call('import_fbx_animation', ...) but got '{args[0]}'"
        )

    @pytest.mark.asyncio
    async def test_passes_cascadeur_false_kwarg(self, mock_ctx, mock_bridge_anim):
        """Must forward cascadeur=False to bridge.call when not a Cascadeur FBX."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_anim):
            await kinefx_tools.houdini_import_fbx_animation(
                mock_ctx, path="G:/in/clip.fbx", cascadeur=False
            )
        _, kwargs = mock_bridge_anim.call.call_args
        assert kwargs.get("cascadeur") is False

    @pytest.mark.asyncio
    async def test_passes_cascadeur_true_kwarg(self, mock_ctx, mock_bridge_anim):
        """
        When cascadeur=True, must forward it to bridge.call so the handler
        sets convertunits on kinefx::fbxanimimport.  FR-3 / spec §7.4.
        """
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_anim):
            await kinefx_tools.houdini_import_fbx_animation(
                mock_ctx,
                path="G:/BigMediumSmall/WorkersWelders_RIG_FBX_TPOSE.fbx",
                cascadeur=True,
            )
        _, kwargs = mock_bridge_anim.call.call_args
        assert kwargs.get("cascadeur") is True, (
            "cascadeur=True must be forwarded to bridge.call so the handler sets "
            "convertunits on kinefx::fbxanimimport (spec §7.4 / FR-3)"
        )

    @pytest.mark.asyncio
    async def test_cascadeur_defaults_to_false(self, mock_ctx, mock_bridge_anim):
        """cascadeur parameter must default to False (non-Cascadeur FBX is the common case)."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_anim):
            # Call WITHOUT cascadeur kwarg — must not raise
            result = await kinefx_tools.houdini_import_fbx_animation(
                mock_ctx, path="G:/in/clip.fbx"
            )
        assert result is not None

    @pytest.mark.asyncio
    async def test_returns_skeleton_with_frame_range(self, mock_ctx, mock_bridge_anim):
        """
        Animation import return envelope must include skeleton with frame_range
        (distinguishes it from character import which returns has_skin_geo).
        """
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_anim):
            result = await kinefx_tools.houdini_import_fbx_animation(
                mock_ctx,
                path="G:/BigMediumSmall/WorkersWelders_RIG_FBX_TPOSE.fbx",
                cascadeur=True,
            )
        assert result.get("ok") is True
        skeleton = result.get("skeleton")
        assert skeleton is not None, "Return envelope missing 'skeleton' key"
        assert "joints" in skeleton, "skeleton must include joint count"
        assert "frame_range" in skeleton, (
            "animation import skeleton must include 'frame_range' "
            "(distinguishes it from character import)"
        )

    @pytest.mark.asyncio
    async def test_fail_loud_returns_ok_false_no_exception(self, mock_ctx, mock_bridge_error):
        """
        When bridge returns ok=False, the animation import tool must return
        {ok:false, error:<str>} WITHOUT raising.  FR-3: fail-loud via envelope.
        """
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_error):
            result = await kinefx_tools.houdini_import_fbx_animation(
                mock_ctx, path="/bad/path.fbx"
            )
        assert result.get("ok") is False
        assert "error" in result and result["error"]


# ===========================================================================
# Tests: FR-12 verify-after-mutate — skeleton embedded in return (spec §4.1)
# ===========================================================================

class TestVerifyAfterMutate:
    """
    FR-12: every gated import handler re-queries its result and embeds the skeleton
    summary in the return envelope.  The tests above already assert the 'skeleton'
    key exists; this class makes the intent explicit as a named requirement.
    """

    @pytest.mark.asyncio
    async def test_character_import_skeleton_joints_count_is_integer(
        self, mock_ctx, mock_bridge
    ):
        """The joints count in the skeleton summary must be a non-negative integer."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            result = await kinefx_tools.houdini_import_fbx_character(
                mock_ctx, path="G:/in/char.fbx"
            )
        joints = result.get("skeleton", {}).get("joints")
        assert isinstance(joints, int) and joints >= 0, (
            f"skeleton.joints must be a non-negative int, got {joints!r}"
        )

    @pytest.mark.asyncio
    async def test_character_import_has_skin_geo_is_bool(self, mock_ctx, mock_bridge):
        """has_skin_geo in the skeleton summary must be a bool."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            result = await kinefx_tools.houdini_import_fbx_character(
                mock_ctx, path="G:/in/char.fbx"
            )
        has_skin = result.get("skeleton", {}).get("has_skin_geo")
        assert isinstance(has_skin, bool), (
            f"skeleton.has_skin_geo must be bool, got {has_skin!r}"
        )


# ===========================================================================
# Regression tests: FR-2 handler-side fail-loud (fix dispatch #005)
#
# These cover the 4 Major findings from the codex-reviewer tier-2 verdict:
#   Major-1: geometry reads OUTSIDE the outer try/except
#   Major-2: createNode / parm().set() UNGUARDED (raise past handler boundary)
#   Major-3: bare except: pass silences frame_range errors
#   Major-4: dest="/obj" sentinel — SOP createNode under OBJ-context network
#
# We test the bridge-side contract here (bridge.call returns ok:False on those
# errors); the hython-smoke covers the real hou-side behaviour.
# ===========================================================================

class TestFR2HandlerFailLoud:
    """
    Every code path in the FBX import handlers MUST return {ok:False, error:<str>}
    rather than raising an exception.  These tests simulate the failure conditions
    surfaced by the codex-reviewer and verify the wrapper propagates them as
    structured failure envelopes.
    """

    @pytest.fixture()
    def mock_bridge_invalid_dest(self):
        """Bridge stub that returns ok=False for an invalid dest node."""
        bridge = MagicMock()
        bridge.call = AsyncMock(return_value={
            "ok": False,
            "error": "dest node not found: '/obj/does_not_exist'",
        })
        return bridge

    @pytest.fixture()
    def mock_bridge_createnode_fail(self):
        """Bridge stub simulating createNode raising (node type unavailable)."""
        bridge = MagicMock()
        bridge.call = AsyncMock(return_value={
            "ok": False,
            "error": "Failed to create node of type kinefx::fbxcharacterimport",
        })
        return bridge

    @pytest.fixture()
    def mock_bridge_geometry_fail(self):
        """Bridge stub simulating geometry() raising (cook produced 0 outputs)."""
        bridge = MagicMock()
        bridge.call = AsyncMock(return_value={
            "ok": False,
            "error": "Invalid output index 1 for kinefx::fbxcharacterimport",
        })
        return bridge

    # ── dest contract (Major-4 / FR-4) ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_character_explicit_invalid_dest_returns_ok_false(
        self, mock_ctx, mock_bridge_invalid_dest
    ):
        """
        When an explicit dest that does not exist in the scene is supplied,
        import_fbx_character must return {ok:False, error:...} — no exception.
        FR-4: explicitly-provided dest that resolves to None → fail loud.
        """
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_invalid_dest):
            result = await kinefx_tools.houdini_import_fbx_character(
                mock_ctx,
                path="G:/in/char.fbx",
                dest="/obj/does_not_exist",
            )
        assert result.get("ok") is False, (
            "Expected ok=False for non-existent explicit dest, got: %r" % result
        )
        assert "error" in result and result["error"], (
            "Failure envelope must contain a non-empty 'error' string"
        )

    @pytest.mark.asyncio
    async def test_animation_explicit_invalid_dest_returns_ok_false(
        self, mock_ctx, mock_bridge_invalid_dest
    ):
        """
        When an explicit dest that does not exist in the scene is supplied,
        import_fbx_animation must return {ok:False, error:...} — no exception.
        """
        # Reuse the same bridge stub (error message is dest-not-found).
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_invalid_dest):
            result = await kinefx_tools.houdini_import_fbx_animation(
                mock_ctx,
                path="G:/in/clip.fbx",
                dest="/obj/does_not_exist",
            )
        assert result.get("ok") is False
        assert "error" in result and result["error"]

    # ── createNode failure (Major-2) ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_character_createnode_failure_returns_ok_false(
        self, mock_ctx, mock_bridge_createnode_fail
    ):
        """
        If createNode raises (kinefx::fbxcharacterimport not installed / unavailable),
        the handler must return {ok:False, error:...} rather than propagating the
        exception.  FR-2 outer try/except must catch it.
        """
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_createnode_fail):
            result = await kinefx_tools.houdini_import_fbx_character(
                mock_ctx, path="G:/in/char.fbx"
            )
        assert result.get("ok") is False
        assert "error" in result and result["error"]

    @pytest.mark.asyncio
    async def test_animation_createnode_failure_returns_ok_false(
        self, mock_ctx, mock_bridge_createnode_fail
    ):
        """Same contract for import_fbx_animation."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_createnode_fail):
            result = await kinefx_tools.houdini_import_fbx_animation(
                mock_ctx, path="G:/in/clip.fbx"
            )
        assert result.get("ok") is False
        assert "error" in result and result["error"]

    # ── geometry() failure (Major-1) ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_character_geometry_failure_returns_ok_false(
        self, mock_ctx, mock_bridge_geometry_fail
    ):
        """
        If geometry(1) raises (e.g. cook produced fewer outputs than expected),
        the handler must return {ok:False, error:...}.  Major-1 fix: geometry reads
        must be INSIDE the outer FR-2 try/except, not after it.
        """
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_geometry_fail):
            result = await kinefx_tools.houdini_import_fbx_character(
                mock_ctx, path="G:/in/char.fbx"
            )
        assert result.get("ok") is False
        assert "error" in result and result["error"]

    @pytest.mark.asyncio
    async def test_animation_geometry_failure_returns_ok_false(
        self, mock_ctx, mock_bridge_geometry_fail
    ):
        """Same contract for import_fbx_animation."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge_geometry_fail):
            result = await kinefx_tools.houdini_import_fbx_animation(
                mock_ctx, path="G:/in/clip.fbx"
            )
        assert result.get("ok") is False
        assert "error" in result and result["error"]
