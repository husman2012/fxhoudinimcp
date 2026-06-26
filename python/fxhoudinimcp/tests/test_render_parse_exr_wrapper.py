"""pytest wrapper tests for render_parse_exr MCP tool.

Verifies the MCP wrapper's public contract WITHOUT importing the
implementation — these tests run RED until hou-dev authors
tools/render_readback_tools.py (adding render_parse_exr).

Contract asserted (plan pp12-114d lockedFieldContract):
  - wrapper module: fxhoudinimcp.tools.render_readback_tools
  - @mcp.tool with meta={'require_approval': False}  (FR-10, READONLY, ungated)
  - signature: render_parse_exr(exr_path: str, subimage: int | None, ctx: Context) -> dict
  - calls bridge.execute('render_parse_exr', {'exr_path': ..., 'subimage': ...})
  - ctx-schema guard: 'ctx' NOT in the tool's FastMCP input schema properties
  - return shape (§4.2): ExrManifest.to_dict() with exr_path key present

PP12-110 lessons applied:
  - MagicMock(spec=HoudiniBridge) so .call() / non-existent attrs raise AttributeError.
  - bridge.execute (NOT bridge.call) is the only acceptable method name.
  - ctx: Context (not Any) — the schema guard verifies FastMCP hides ctx from clients.
  - Import wrapper INSIDE each test after monkeypatching _get_bridge.

testVerificationSurface: pytest-model
unitId: pp12-114d
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
        "exr_path": "cli:/tmp/test.0001.exr",
        "is_multipart": False,
        "subimages": 1,
        "compression": "zips",
        "xres": 64,
        "yres": 64,
        "channels": [
            {"name": "R", "layer": None, "dtype": "half"},
            {"name": "G", "layer": None, "dtype": "half"},
            {"name": "B", "layer": None, "dtype": "half"},
        ],
        "crypto_layers": [],
        "metadata": {"compression": "zips"},
    })
    return mock


# ---------------------------------------------------------------------------
# RPE-1: module importability
# ---------------------------------------------------------------------------

class TestRenderParseExrModuleImport:
    """RPE-1: wrapper module must exist at the specified path and expose render_parse_exr."""

    def test_module_importable(self):
        """Import fxhoudinimcp.tools.render_readback_tools — FAILS RED until hou-dev creates it."""
        # This assertion is the primary RED gate: render_parse_exr does not exist yet.
        import fxhoudinimcp.tools.render_readback_tools  # noqa: F401

    def test_render_parse_exr_callable_on_module(self):
        """render_parse_exr must be importable as a callable from render_readback_tools."""
        from fxhoudinimcp.tools.render_readback_tools import render_parse_exr  # noqa: F401
        assert callable(render_parse_exr), (
            "render_parse_exr must be a callable (the @mcp.tool-decorated coroutine)."
        )


# ---------------------------------------------------------------------------
# RPE-2: bridge.execute call contract
# ---------------------------------------------------------------------------

class TestRenderParseExrBridgeContract:
    """RPE-2: wrapper delegates to bridge.execute('render_parse_exr', {...})."""

    @pytest.mark.asyncio
    async def test_calls_bridge_execute_not_bridge_call(self, bridge_mock):
        """Wrapper must call bridge.execute (not bridge.call or any other attr).

        PP12-110 lesson: bridge.call does NOT exist. spec=HoudiniBridge ensures
        that calling bridge.call() on the mock raises AttributeError.
        """
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            # Import inside test — after _get_bridge is patched.
            from fxhoudinimcp.tools.render_readback_tools import render_parse_exr

            ctx_mock = MagicMock()
            await render_parse_exr(
                exr_path="/tmp/test.0001.exr", subimage=None, ctx=ctx_mock
            )

        # Verify bridge.execute was called with the correct command string.
        bridge_mock.execute.assert_called_once()
        call_args = bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "render_parse_exr", (
            f"Expected bridge.execute('render_parse_exr', ...) but got command={command!r}. "
            "PP12-110: the dispatcher command string must match register_handler's first arg."
        )

    @pytest.mark.asyncio
    async def test_bridge_execute_params_contain_exr_path(self, bridge_mock):
        """Wrapper must pass exr_path in the params dict to bridge.execute."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_parse_exr

            ctx_mock = MagicMock()
            await render_parse_exr(
                exr_path="/tmp/test.0001.exr", subimage=None, ctx=ctx_mock
            )

        call_args = bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert "exr_path" in params, (
            f"bridge.execute params must contain 'exr_path'; got: {params!r}"
        )
        assert params["exr_path"] == "/tmp/test.0001.exr", (
            f"exr_path must be forwarded as-is; got {params['exr_path']!r}"
        )

    @pytest.mark.asyncio
    async def test_bridge_execute_params_contain_subimage(self, bridge_mock):
        """Wrapper must pass subimage in the params dict to bridge.execute."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_parse_exr

            ctx_mock = MagicMock()
            await render_parse_exr(
                exr_path="/tmp/test.0001.exr", subimage=2, ctx=ctx_mock
            )

        call_args = bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert "subimage" in params, (
            f"bridge.execute params must contain 'subimage'; got: {params!r}"
        )
        assert params["subimage"] == 2, (
            f"subimage must be forwarded as-is; got {params['subimage']!r}"
        )

    @pytest.mark.asyncio
    async def test_bridge_execute_subimage_none_forwarded(self, bridge_mock):
        """Wrapper must forward subimage=None (the default) to bridge.execute, not omit it."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_parse_exr

            ctx_mock = MagicMock()
            await render_parse_exr(
                exr_path="/tmp/test.0001.exr", subimage=None, ctx=ctx_mock
            )

        call_args = bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        # subimage=None must be present in params — the handler uses it for
        # hoiiotool --subimage N selection; omitting it silently ignores the arg.
        assert "subimage" in params, (
            f"'subimage' must always be forwarded to bridge.execute (even as None); "
            f"got params keys: {sorted(params.keys())}"
        )
        assert params["subimage"] is None, (
            f"subimage=None must be forwarded as None; got {params['subimage']!r}"
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
# RPE-3: ctx-schema guard (FastMCP hides ctx: Context from client schema)
# ---------------------------------------------------------------------------

class TestRenderParseExrCtxSchemaGuard:
    """RPE-3: 'ctx' must NOT appear in the tool's input schema properties.

    FastMCP automatically injects ctx: Context from the MCP lifespan context
    and HIDES it from the client-visible JSON schema. If 'ctx' appears in the
    schema, the tool was authored with `ctx: Any` or a non-Context type.
    """

    def test_ctx_not_in_tool_input_schema(self):
        """'ctx' must be absent from render_parse_exr's FastMCP input schema.

        RED until hou-dev creates the wrapper with ctx: Context (not ctx: Any).
        """
        from fxhoudinimcp.server import mcp  # noqa: PLC0415

        # FastMCP exposes _tool_manager or _tools depending on version.
        tool_map: dict = {}
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_map = mcp._tool_manager._tools
        elif hasattr(mcp, "_tools"):
            tool_map = mcp._tools

        assert "render_parse_exr" in tool_map, (
            "render_parse_exr not registered on the mcp server; "
            "hou-dev must add the tool to tools/render_readback_tools.py and "
            "ensure it is imported at server startup."
        )

        tool_obj = tool_map["render_parse_exr"]
        schema = tool_obj.parameters if hasattr(tool_obj, "parameters") else {}
        props = schema.get("properties", {})

        assert "ctx" not in props, (
            f"'ctx' appears in render_parse_exr input schema properties: {sorted(props.keys())}. "
            "FastMCP hides ctx: Context from client schema automatically — "
            "if 'ctx' is visible, the param was typed as ctx: Any (PP12-110 bug class)."
        )


# ---------------------------------------------------------------------------
# RPE-4: return shape — §4.2 ExrManifest.to_dict() keys present
# ---------------------------------------------------------------------------

class TestRenderParseExrReturnShape:
    """RPE-4: wrapper must surface the §4.2 ExrManifest.to_dict() shape."""

    @pytest.mark.asyncio
    async def test_return_value_is_dict(self, bridge_mock):
        """Return value from render_parse_exr must be a dict (the manifest dict)."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_parse_exr

            ctx_mock = MagicMock()
            result = await render_parse_exr(
                exr_path="/tmp/test.0001.exr", subimage=None, ctx=ctx_mock
            )

        assert isinstance(result, dict), (
            f"render_parse_exr must return a dict; got {type(result).__name__!r}"
        )

    @pytest.mark.asyncio
    async def test_return_contains_exr_path_key(self, bridge_mock):
        """§4.2: return dict must contain 'exr_path' key."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_parse_exr

            ctx_mock = MagicMock()
            result = await render_parse_exr(
                exr_path="/tmp/test.0001.exr", subimage=None, ctx=ctx_mock
            )

        assert "exr_path" in result, (
            f"§4.2: 'exr_path' key missing from result; keys present: {sorted(result.keys())}"
        )

    @pytest.mark.asyncio
    async def test_return_contains_is_multipart_key(self, bridge_mock):
        """§4.2: return dict must contain 'is_multipart' key."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_parse_exr

            ctx_mock = MagicMock()
            result = await render_parse_exr(
                exr_path="/tmp/test.0001.exr", subimage=None, ctx=ctx_mock
            )

        assert "is_multipart" in result, (
            f"§4.2: 'is_multipart' key missing; keys present: {sorted(result.keys())}"
        )

    @pytest.mark.asyncio
    async def test_return_contains_channels_key(self, bridge_mock):
        """§4.2: return dict must contain 'channels' key (list of channel dicts)."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_parse_exr

            ctx_mock = MagicMock()
            result = await render_parse_exr(
                exr_path="/tmp/test.0001.exr", subimage=None, ctx=ctx_mock
            )

        assert "channels" in result, (
            f"§4.2: 'channels' key missing; keys present: {sorted(result.keys())}"
        )
        assert isinstance(result.get("channels"), list), (
            f"§4.2: 'channels' must be a list; got {type(result.get('channels')).__name__!r}"
        )
