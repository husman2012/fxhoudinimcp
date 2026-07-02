"""pytest wrapper tests for render_lint_settings MCP tool.

Verifies the MCP wrapper's public contract WITHOUT importing the
implementation — these tests run RED until hou-dev authors
tools/render_readback_tools.py.

Contract asserted (plan pp12-114c lockedFieldContract):
  - wrapper module: fxhoudinimcp.tools.render_readback_tools
  - @mcp.tool with meta={'require_approval': False}  (FR-10, READONLY, ungated)
  - signature: render_lint_settings(render_node, preset, ctx: Context) -> dict
  - calls bridge.execute('render_lint_settings', {'render_node': ..., 'preset': ...})
  - ctx-schema guard: 'ctx' NOT in the tool's FastMCP input schema properties

PP12-110 lessons applied:
  - MagicMock(spec=HoudiniBridge) so .call() / non-existent attrs raise AttributeError.
  - bridge.execute (NOT bridge.call) is the only acceptable method name.
  - ctx: Context (not Any) — the schema guard verifies FastMCP hides ctx from clients.
  - Import wrapper INSIDE each test after monkeypatching _get_bridge.

testVerificationSurface: pytest-model
unitId: pp12-114c
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the REAL HoudiniBridge for spec= (PP12-110 lesson: spec-bound mocks
# raise AttributeError on non-existent attrs like .call()).
# ---------------------------------------------------------------------------
from fxhoudinimcp.bridge import HoudiniBridge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def bridge_mock():
    """Return a spec-bound HoudiniBridge AsyncMock.

    spec=HoudiniBridge means any attribute NOT on the real class
    (e.g. .call()) immediately raises AttributeError instead of silently
    returning a new MagicMock — the PP12-110 mirror-test anti-pattern guard.
    """
    mock = MagicMock(spec=HoudiniBridge)
    # bridge.execute is async; wire it as an AsyncMock.
    mock.execute = AsyncMock(return_value={
        "render_node": "/stage/karma1",
        "preset": "nuke_safe",
        "results": [],
        "summary": {"ok": 0, "warn": 0, "error": 0},
        "ready_to_render": True,
    })
    return mock


# ---------------------------------------------------------------------------
# RL-1: module importability
# ---------------------------------------------------------------------------

class TestRenderLintSettingsModuleImport:
    """RL-1: wrapper module must exist at the specified path."""

    def test_module_importable(self):
        """Import fxhoudinimcp.tools.render_readback_tools — FAILS RED until hou-dev creates it."""
        # This assertion is the primary RED gate: the module does not exist yet.
        import fxhoudinimcp.tools.render_readback_tools  # noqa: F401


# ---------------------------------------------------------------------------
# RL-2: bridge.execute call contract
# ---------------------------------------------------------------------------

class TestRenderLintSettingsBridgeContract:
    """RL-2: wrapper delegates to bridge.execute('render_lint_settings', {...})."""

    @pytest.mark.asyncio
    async def test_calls_bridge_execute_not_bridge_call(self, bridge_mock):
        """Wrapper must call bridge.execute (not bridge.call or any other attr).

        PP12-110 lesson: bridge.call does NOT exist. spec=HoudiniBridge ensures
        that calling bridge.call() on the mock raises AttributeError.
        """
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            # Import inside test — after _get_bridge is patched.
            from fxhoudinimcp.tools.render_readback_tools import render_lint_settings

            ctx_mock = MagicMock()
            await render_lint_settings(
                render_node="/stage/karma1", preset="nuke_safe", ctx=ctx_mock
            )

        # Verify bridge.execute was called with the correct command string.
        bridge_mock.execute.assert_called_once()
        call_args = bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "render_lint_settings", (
            f"Expected bridge.execute('render_lint_settings', ...) but got command={command!r}. "
            "PP12-110: the dispatcher command string must match register_handler's first arg."
        )

    @pytest.mark.asyncio
    async def test_bridge_execute_params_contain_render_node(self, bridge_mock):
        """Wrapper must pass render_node in the params dict to bridge.execute."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_lint_settings

            ctx_mock = MagicMock()
            await render_lint_settings(
                render_node="/stage/karma1", preset="nuke_safe", ctx=ctx_mock
            )

        call_args = bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert "render_node" in params, (
            f"bridge.execute params must contain 'render_node'; got: {params!r}"
        )
        assert params["render_node"] == "/stage/karma1", (
            f"render_node must be forwarded as-is; got {params['render_node']!r}"
        )

    @pytest.mark.asyncio
    async def test_bridge_execute_params_contain_preset(self, bridge_mock):
        """Wrapper must pass preset in the params dict to bridge.execute."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_lint_settings

            ctx_mock = MagicMock()
            await render_lint_settings(
                render_node="/stage/karma1", preset="nuke_safe", ctx=ctx_mock
            )

        call_args = bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert "preset" in params, (
            f"bridge.execute params must contain 'preset'; got: {params!r}"
        )
        assert params["preset"] == "nuke_safe", (
            f"preset must be forwarded as-is; got {params['preset']!r}"
        )

    def test_bridge_call_does_not_exist(self):
        """Regression guard: bridge.call must NOT exist on HoudiniBridge.

        PP12-110 root cause: wrapper called bridge.call() which does not exist
        on the real class. A spec-bound mock catches this at test time.
        """
        mock = MagicMock(spec=HoudiniBridge)
        with pytest.raises(AttributeError):
            _ = mock.call  # .call does not exist on HoudiniBridge


# ---------------------------------------------------------------------------
# RL-3: ctx-schema guard (FastMCP hides ctx: Context from client schema)
# ---------------------------------------------------------------------------

class TestRenderLintSettingsCtxSchemaGuard:
    """RL-3: 'ctx' must NOT appear in the tool's input schema properties.

    FastMCP automatically injects ctx: Context from the MCP lifespan context
    and HIDES it from the client-visible JSON schema. If 'ctx' appears in the
    schema, the tool was authored with `ctx: Any` or a non-Context type.
    """

    def test_ctx_not_in_tool_input_schema(self):
        """'ctx' must be absent from render_lint_settings's FastMCP input schema.

        RED until hou-dev creates the wrapper with ctx: Context (not ctx: Any).
        """
        # Need the mcp server object to inspect registered tool schemas.
        from fxhoudinimcp.server import mcp  # noqa: PLC0415

        # FastMCP exposes _tool_manager or _tools depending on version.
        # Probe both attribute shapes defensively.
        tool_map: dict = {}
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_map = mcp._tool_manager._tools
        elif hasattr(mcp, "_tools"):
            tool_map = mcp._tools

        assert "render_lint_settings" in tool_map, (
            "render_lint_settings not registered on the mcp server; "
            "hou-dev must add tools/render_readback_tools.py and register it."
        )

        tool_obj = tool_map["render_lint_settings"]
        schema = tool_obj.parameters if hasattr(tool_obj, "parameters") else {}
        props = schema.get("properties", {})

        assert "ctx" not in props, (
            f"'ctx' appears in render_lint_settings input schema properties: {sorted(props.keys())}. "
            "FastMCP hides ctx: Context from client schema automatically — "
            "if 'ctx' is visible, the param was typed as ctx: Any (PP12-110 bug class)."
        )


# ---------------------------------------------------------------------------
# RL-4: default preset is 'nuke_safe'
# ---------------------------------------------------------------------------

class TestRenderLintSettingsDefaultPreset:
    """RL-4: preset parameter must default to 'nuke_safe'."""

    def test_default_preset_is_nuke_safe(self, bridge_mock):
        """Call with no preset arg; bridge must receive preset='nuke_safe'."""
        import asyncio
        import inspect

        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_lint_settings

        sig = inspect.signature(render_lint_settings)
        preset_param = sig.parameters.get("preset")
        assert preset_param is not None, (
            "render_lint_settings must have a 'preset' parameter"
        )
        assert preset_param.default == "nuke_safe", (
            f"preset default must be 'nuke_safe', got {preset_param.default!r}"
        )
