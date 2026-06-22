"""
test_kinefx_tools.py — mock-bridge tests for kinefx_tools MCP surface (PP12-110b).

TDD phase: RED — `fxhoudinimcp.tools.kinefx_tools` does NOT exist yet.
Collection-time import fails with ModuleNotFoundError, making every test red
until hou-dev ships kinefx_tools.py.

This file asserts:
  1. The module is importable (collection gate; RED until impl lands).
  2. All 3 tools exist in the MCP tool registry with require_approval=False
     (via meta dict — FastMCP.tool() has no native require_approval kwarg).
  3. Each tool calls bridge.call() with the correct command string and keyword
     arguments (bridge.call != bridge.execute — kinefx tools use keyword args).

No hou / Qt / pxr imports anywhere.
Runs on bare CI (plain pytest, no Houdini required).

testVerificationSurface: pytest-model
unitId: pp12-110b
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if os.path.abspath(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, os.path.abspath(_PKG_ROOT))

# ---------------------------------------------------------------------------
# RED GATE — this import fails until hou-dev ships kinefx_tools.py.
# hou-dev MUST NOT modify this file to work around the import error.
# ---------------------------------------------------------------------------
from fxhoudinimcp.tools import kinefx_tools  # noqa: E402  (RED: ModuleNotFoundError)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def mock_mcp():
    """A FastMCP-like stub: captures tool registrations made via @mcp.tool()."""
    mcp = MagicMock()
    # Track registered tools by name -> Tool mock
    mcp._tool_registry: dict = {}

    def _tool_decorator(**kwargs):
        """Mimic @mcp.tool(meta={...}) — returns a decorator that records the fn."""
        meta = kwargs.get("meta", {})
        name = kwargs.get("name", None)

        def decorator(fn):
            tool_name = name or fn.__name__
            tool_mock = MagicMock()
            tool_mock.meta = meta
            tool_mock.fn = fn
            tool_mock.name = tool_name
            mcp._tool_registry[tool_name] = tool_mock
            return fn

        return decorator

    mcp.tool = _tool_decorator
    return mcp


@pytest.fixture()
def mock_bridge():
    """A bridge stub; .call is an AsyncMock so async tools can be awaited."""
    bridge = MagicMock()
    bridge.call = AsyncMock(return_value={"status": "ok", "data": {}})
    return bridge


@pytest.fixture()
def mock_ctx(mock_bridge):
    """FastMCP Context stub with a bridge injected via state."""
    ctx = MagicMock()
    # The server uses ctx.request_context.lifespan_context["bridge"] or similar.
    # We patch _get_bridge at the module level instead (see tests below).
    return ctx


# ===========================================================================
# Tests: module-level tool registration (meta["require_approval"])
# ===========================================================================

class TestKinefxToolRegistration:
    """
    Verify that each of the 3 ungated tools is registered with
    meta={"require_approval": False}.

    Because FastMCP.tool() has no native require_approval param, the impl
    MUST pass meta={"require_approval": False}; the test inspects that dict.
    """

    def _collect_registered_tools(self):
        """Re-import kinefx_tools against the mock_mcp to capture registrations."""
        # This fixture-free helper is called from each test with a fresh mcp stub.
        mcp = MagicMock()
        registry: dict = {}

        def _tool(**kwargs):
            meta = kwargs.get("meta", {})
            name = kwargs.get("name", None)

            def decorator(fn):
                tool_name = name or fn.__name__
                registry[tool_name] = {"meta": meta, "fn": fn}
                return fn

            return decorator

        mcp.tool = _tool

        # Patch the module's mcp object so @mcp.tool() calls hit our stub
        with patch.object(kinefx_tools, "mcp", mcp):
            # Force re-registration by calling the registration function if one
            # exists, or by reloading. The preferred pattern: kinefx_tools exposes
            # a register(mcp) function; if not, we reach into the module globals.
            if hasattr(kinefx_tools, "_register_tools"):
                kinefx_tools._register_tools(mcp)
            # Fallback: import already ran the decorators; registry captured above.

        return registry

    def test_kinefx_probe_require_approval_false(self):
        """kinefx_probe must have meta["require_approval"] == False."""
        mcp = kinefx_tools.mcp
        tools = mcp._tool_manager._tools  # FastMCP internal registry
        assert "kinefx_probe" in tools, "kinefx_probe not registered in MCP"
        assert tools["kinefx_probe"].meta.get("require_approval") is False

    def test_query_skeleton_require_approval_false(self):
        mcp = kinefx_tools.mcp
        tools = mcp._tool_manager._tools
        assert "query_skeleton" in tools, "query_skeleton not registered in MCP"
        assert tools["query_skeleton"].meta.get("require_approval") is False

    def test_inspect_apex_require_approval_false(self):
        mcp = kinefx_tools.mcp
        tools = mcp._tool_manager._tools
        assert "inspect_apex" in tools, "inspect_apex not registered in MCP"
        assert tools["inspect_apex"].meta.get("require_approval") is False


# ===========================================================================
# Tests: bridge.call() command routing
# ===========================================================================

class TestKinefxToolBridgeCalls:
    """
    Verify each tool calls bridge.call(command, **kwargs) with the correct
    command string and keyword arguments.

    New kinefx tools use keyword args (bridge.call) unlike existing tools
    that use positional dict args (bridge.execute).
    """

    @pytest.mark.asyncio
    async def test_kinefx_probe_calls_bridge_with_node_path(self, mock_ctx, mock_bridge):
        """kinefx_probe calls bridge.call('kinefx_probe', node_path=...)."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.kinefx_probe(mock_ctx, node_path="/obj/geo1")

        mock_bridge.call.assert_called_once()
        args, kwargs = mock_bridge.call.call_args
        assert args[0] == "kinefx_probe"
        assert kwargs.get("node_path") == "/obj/geo1"

    @pytest.mark.asyncio
    async def test_query_skeleton_calls_bridge_with_node_and_frame(self, mock_ctx, mock_bridge):
        """query_skeleton calls bridge.call('query_skeleton', node_path=..., frame=...)."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.query_skeleton(mock_ctx, node_path="/obj/rig", frame=1.0)

        mock_bridge.call.assert_called_once()
        args, kwargs = mock_bridge.call.call_args
        assert args[0] == "query_skeleton"
        assert kwargs.get("node_path") == "/obj/rig"
        assert kwargs.get("frame") == 1.0

    @pytest.mark.asyncio
    async def test_query_skeleton_frame_defaults_to_none(self, mock_ctx, mock_bridge):
        """query_skeleton frame parameter is optional (defaults to None or current frame)."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            # Should NOT raise even without frame
            await kinefx_tools.query_skeleton(mock_ctx, node_path="/obj/rig")

        mock_bridge.call.assert_called_once()

    @pytest.mark.asyncio
    async def test_inspect_apex_calls_bridge_with_node_path(self, mock_ctx, mock_bridge):
        """inspect_apex calls bridge.call('inspect_apex', node_path=...)."""
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            await kinefx_tools.inspect_apex(mock_ctx, node_path="/obj/apex_graph")

        mock_bridge.call.assert_called_once()
        args, kwargs = mock_bridge.call.call_args
        assert args[0] == "inspect_apex"
        assert kwargs.get("node_path") == "/obj/apex_graph"

    @pytest.mark.asyncio
    async def test_kinefx_probe_does_not_call_execute(self, mock_ctx, mock_bridge):
        """kinefx tools use bridge.call, NOT bridge.execute."""
        bridge = mock_bridge
        bridge.execute = AsyncMock()
        with patch.object(kinefx_tools, "_get_bridge", return_value=bridge):
            await kinefx_tools.kinefx_probe(mock_ctx, node_path="/obj/x")

        bridge.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_bridge_returns_dict_result(self, mock_ctx, mock_bridge):
        """Tool must return something derived from bridge.call result."""
        mock_bridge.call.return_value = {"status": "ok", "data": {"count": 3}}
        with patch.object(kinefx_tools, "_get_bridge", return_value=mock_bridge):
            result = await kinefx_tools.query_skeleton(mock_ctx, node_path="/obj/rig")

        assert result is not None, "Tool must return a value, not None"


# ===========================================================================
# Tests: tool functions are async
# ===========================================================================

class TestKinefxToolsAreAsync:
    """
    MCP tools in this codebase are declared async (FastMCP constraint).
    """

    def test_kinefx_probe_is_coroutine_function(self):
        import asyncio
        assert asyncio.iscoroutinefunction(kinefx_tools.kinefx_probe)

    def test_query_skeleton_is_coroutine_function(self):
        import asyncio
        assert asyncio.iscoroutinefunction(kinefx_tools.query_skeleton)

    def test_inspect_apex_is_coroutine_function(self):
        import asyncio
        assert asyncio.iscoroutinefunction(kinefx_tools.inspect_apex)
