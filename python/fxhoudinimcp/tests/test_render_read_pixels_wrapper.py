"""pytest wrapper tests for render_read_pixels MCP tool.

Verifies the MCP wrapper's public contract WITHOUT importing the
implementation — these tests run RED until hou-dev authors
tools/render_readback_tools.py (adding render_read_pixels).

Contract asserted (plan pp12-114e lockedFieldContract):
  - wrapper module: fxhoudinimcp.tools.render_readback_tools
  - @mcp.tool with meta={'require_approval': False}  (FR-10, READONLY, ungated)
  - signature:
        render_read_pixels(
            source: str,
            plane: str,
            mode: str,
            roi: list | None,
            max_pixels: int,
            downsample: int,
            page: int,
            page_size: int,
            ctx: Context,          <- hidden from client schema
        ) -> dict
  - calls bridge.execute('render_read_pixels', {
        'source': ..., 'plane': ..., 'mode': ..., 'roi': ...,
        'max_pixels': ..., 'downsample': ..., 'page': ..., 'page_size': ...,
    })
  - ctx-schema guard: 'ctx' NOT in the tool's FastMCP input schema properties
  - return shape (§4.2): {plane, xres, yres, channels, dtype, mode, stats,
                           histogram, pixels, page, page_size, total_pages, truncated}

PP12-110 lessons applied:
  - MagicMock(spec=HoudiniBridge) so .call() / non-existent attrs raise AttributeError.
  - bridge.execute (NOT bridge.call) is the only acceptable method name.
  - ctx: Context (not Any) — the schema guard verifies FastMCP hides ctx from clients.
  - Import wrapper INSIDE each test after monkeypatching _get_bridge.

testVerificationSurface: pytest-model
unitId: pp12-114e
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
        "plane": "C",
        "xres": 64,
        "yres": 64,
        "channels": 3,
        "dtype": "float16",
        "mode": "summary",
        "stats": {
            "min": [0.0, 0.0, 0.0],
            "max": [1.0, 1.0, 1.0],
            "mean": [0.5, 0.5, 0.5],
            "nan_count": 0,
            "inf_count": 0,
        },
        "histogram": {
            "bins": 16,
            "counts": [[0] * 16, [0] * 16, [0] * 16],
        },
        "pixels": [],
        "page": 0,
        "page_size": 1024,
        "total_pages": 1,
        "truncated": False,
    })
    return mock


# ---------------------------------------------------------------------------
# RRP-1: module importability
# ---------------------------------------------------------------------------

class TestRenderReadPixelsModuleImport:
    """RRP-1: wrapper module must exist at the specified path and expose render_read_pixels."""

    def test_module_importable(self):
        """Import fxhoudinimcp.tools.render_readback_tools — FAILS RED until hou-dev creates it."""
        # This assertion is the primary RED gate: render_read_pixels does not exist yet.
        import fxhoudinimcp.tools.render_readback_tools  # noqa: F401

    def test_render_read_pixels_callable_on_module(self):
        """render_read_pixels must be importable as a callable from render_readback_tools."""
        from fxhoudinimcp.tools.render_readback_tools import render_read_pixels  # noqa: F401
        assert callable(render_read_pixels), (
            "render_read_pixels must be a callable (the @mcp.tool-decorated coroutine)."
        )


# ---------------------------------------------------------------------------
# RRP-2: bridge.execute call contract
# ---------------------------------------------------------------------------

class TestRenderReadPixelsBridgeContract:
    """RRP-2: wrapper delegates to bridge.execute('render_read_pixels', {...})."""

    @pytest.mark.asyncio
    async def test_calls_bridge_execute_not_bridge_call(self, bridge_mock):
        """Wrapper must call bridge.execute (not bridge.call or any other attr).

        PP12-110 lesson: bridge.call does NOT exist. spec=HoudiniBridge ensures
        that calling bridge.call() on the mock raises AttributeError.
        """
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            # Import inside test — after _get_bridge is patched.
            from fxhoudinimcp.tools.render_readback_tools import render_read_pixels

            ctx_mock = MagicMock()
            await render_read_pixels(
                source="/tmp/test.0001.exr",
                plane="C",
                mode="summary",
                roi=None,
                max_pixels=4096,
                downsample=1,
                page=0,
                page_size=1024,
                ctx=ctx_mock,
            )

        # Verify bridge.execute was called with the correct command string.
        bridge_mock.execute.assert_called_once()
        call_args = bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "render_read_pixels", (
            f"Expected bridge.execute('render_read_pixels', ...) but got command={command!r}. "
            "PP12-110: the dispatcher command string must match register_handler's first arg."
        )

    @pytest.mark.asyncio
    async def test_bridge_execute_params_contain_source(self, bridge_mock):
        """Wrapper must pass source in the params dict to bridge.execute."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_read_pixels

            ctx_mock = MagicMock()
            await render_read_pixels(
                source="/tmp/test.0001.exr",
                plane="C",
                mode="summary",
                roi=None,
                max_pixels=4096,
                downsample=1,
                page=0,
                page_size=1024,
                ctx=ctx_mock,
            )

        call_args = bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert "source" in params, (
            f"bridge.execute params must contain 'source'; got: {params!r}"
        )
        assert params["source"] == "/tmp/test.0001.exr", (
            f"source must be forwarded as-is; got {params['source']!r}"
        )

    @pytest.mark.asyncio
    async def test_bridge_execute_params_contain_all_eight_fields(self, bridge_mock):
        """Wrapper must pass all eight client params in the params dict to bridge.execute.

        The locked field contract requires:
            {source, plane, mode, roi, max_pixels, downsample, page, page_size}
        """
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_read_pixels

            ctx_mock = MagicMock()
            await render_read_pixels(
                source="/tmp/test.exr",
                plane="N",
                mode="roi",
                roi=[0, 0, 8, 8],
                max_pixels=512,
                downsample=2,
                page=1,
                page_size=256,
                ctx=ctx_mock,
            )

        call_args = bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        required_keys = {"source", "plane", "mode", "roi", "max_pixels",
                         "downsample", "page", "page_size"}
        missing = required_keys - set(params.keys())
        assert not missing, (
            f"bridge.execute params missing keys: {missing}; got params keys: {sorted(params.keys())}"
        )

    @pytest.mark.asyncio
    async def test_bridge_execute_params_values_forwarded_correctly(self, bridge_mock):
        """Every param value must be forwarded verbatim to bridge.execute."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_read_pixels

            ctx_mock = MagicMock()
            await render_read_pixels(
                source="/tmp/frame.exr",
                plane="depth",
                mode="sample",
                roi=None,
                max_pixels=2048,
                downsample=3,
                page=2,
                page_size=512,
                ctx=ctx_mock,
            )

        call_args = bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        assert params.get("source") == "/tmp/frame.exr", f"source mismatch: {params.get('source')!r}"
        assert params.get("plane") == "depth", f"plane mismatch: {params.get('plane')!r}"
        assert params.get("mode") == "sample", f"mode mismatch: {params.get('mode')!r}"
        assert params.get("roi") is None, f"roi mismatch: {params.get('roi')!r}"
        assert params.get("max_pixels") == 2048, f"max_pixels mismatch: {params.get('max_pixels')!r}"
        assert params.get("downsample") == 3, f"downsample mismatch: {params.get('downsample')!r}"
        assert params.get("page") == 2, f"page mismatch: {params.get('page')!r}"
        assert params.get("page_size") == 512, f"page_size mismatch: {params.get('page_size')!r}"

    @pytest.mark.asyncio
    async def test_roi_list_forwarded_when_provided(self, bridge_mock):
        """roi=[x0,y0,x1,y1] must be forwarded as a list (not None, not str)."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_read_pixels

            ctx_mock = MagicMock()
            await render_read_pixels(
                source="/tmp/test.exr",
                plane="C",
                mode="roi",
                roi=[0, 0, 8, 8],
                max_pixels=4096,
                downsample=1,
                page=0,
                page_size=1024,
                ctx=ctx_mock,
            )

        call_args = bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert params.get("roi") == [0, 0, 8, 8], (
            f"roi=[0,0,8,8] must be forwarded as-is; got {params.get('roi')!r}"
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
# RRP-3: ctx-schema guard (FastMCP hides ctx: Context from client schema)
# ---------------------------------------------------------------------------

class TestRenderReadPixelsCtxSchemaGuard:
    """RRP-3: 'ctx' must NOT appear in the tool's input schema properties.

    FastMCP automatically injects ctx: Context from the MCP lifespan context
    and HIDES it from the client-visible JSON schema. If 'ctx' appears in the
    schema, the tool was authored with `ctx: Any` or a non-Context type.
    """

    def test_ctx_not_in_tool_input_schema(self):
        """'ctx' must be absent from render_read_pixels's FastMCP input schema.

        RED until hou-dev creates the wrapper with ctx: Context (not ctx: Any).
        """
        from fxhoudinimcp.server import mcp  # noqa: PLC0415

        # FastMCP exposes _tool_manager or _tools depending on version.
        tool_map: dict = {}
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_map = mcp._tool_manager._tools
        elif hasattr(mcp, "_tools"):
            tool_map = mcp._tools

        assert "render_read_pixels" in tool_map, (
            "render_read_pixels not registered on the mcp server; "
            "hou-dev must add the tool to tools/render_readback_tools.py and "
            "ensure it is imported at server startup."
        )

        tool_obj = tool_map["render_read_pixels"]
        schema = tool_obj.parameters if hasattr(tool_obj, "parameters") else {}
        props = schema.get("properties", {})

        assert "ctx" not in props, (
            f"'ctx' appears in render_read_pixels input schema properties: {sorted(props.keys())}. "
            "FastMCP hides ctx: Context from client schema automatically — "
            "if 'ctx' is visible, the param was typed as ctx: Any (PP12-110 bug class)."
        )


# ---------------------------------------------------------------------------
# RRP-4: return shape — §4.2 keys present
# ---------------------------------------------------------------------------

class TestRenderReadPixelsReturnShape:
    """RRP-4: wrapper must surface the §4.2 shape."""

    @pytest.mark.asyncio
    async def test_return_value_is_dict(self, bridge_mock):
        """Return value from render_read_pixels must be a dict."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_read_pixels

            ctx_mock = MagicMock()
            result = await render_read_pixels(
                source="/tmp/test.exr",
                plane="C",
                mode="summary",
                roi=None,
                max_pixels=4096,
                downsample=1,
                page=0,
                page_size=1024,
                ctx=ctx_mock,
            )

        assert isinstance(result, dict), (
            f"render_read_pixels must return a dict; got {type(result).__name__!r}"
        )

    @pytest.mark.asyncio
    async def test_return_contains_plane_key(self, bridge_mock):
        """§4.2: return dict must contain 'plane' key."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_read_pixels

            ctx_mock = MagicMock()
            result = await render_read_pixels(
                source="/tmp/test.exr",
                plane="C",
                mode="summary",
                roi=None,
                max_pixels=4096,
                downsample=1,
                page=0,
                page_size=1024,
                ctx=ctx_mock,
            )

        assert "plane" in result, (
            f"§4.2: 'plane' key missing from result; keys present: {sorted(result.keys())}"
        )

    @pytest.mark.asyncio
    async def test_return_contains_xres_yres_keys(self, bridge_mock):
        """§4.2: return dict must contain 'xres' and 'yres' keys."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_read_pixels

            ctx_mock = MagicMock()
            result = await render_read_pixels(
                source="/tmp/test.exr",
                plane="C",
                mode="summary",
                roi=None,
                max_pixels=4096,
                downsample=1,
                page=0,
                page_size=1024,
                ctx=ctx_mock,
            )

        assert "xres" in result, (
            f"§4.2: 'xres' key missing; keys present: {sorted(result.keys())}"
        )
        assert "yres" in result, (
            f"§4.2: 'yres' key missing; keys present: {sorted(result.keys())}"
        )

    @pytest.mark.asyncio
    async def test_return_contains_pixels_key(self, bridge_mock):
        """§4.2: return dict must contain 'pixels' key (list)."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_read_pixels

            ctx_mock = MagicMock()
            result = await render_read_pixels(
                source="/tmp/test.exr",
                plane="C",
                mode="summary",
                roi=None,
                max_pixels=4096,
                downsample=1,
                page=0,
                page_size=1024,
                ctx=ctx_mock,
            )

        assert "pixels" in result, (
            f"§4.2: 'pixels' key missing; keys present: {sorted(result.keys())}"
        )
        assert isinstance(result.get("pixels"), list), (
            f"§4.2: 'pixels' must be a list; got {type(result.get('pixels')).__name__!r}"
        )

    @pytest.mark.asyncio
    async def test_return_contains_stats_key(self, bridge_mock):
        """§4.2: return dict must contain 'stats' dict with required sub-keys."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_read_pixels

            ctx_mock = MagicMock()
            result = await render_read_pixels(
                source="/tmp/test.exr",
                plane="C",
                mode="summary",
                roi=None,
                max_pixels=4096,
                downsample=1,
                page=0,
                page_size=1024,
                ctx=ctx_mock,
            )

        assert "stats" in result, (
            f"§4.2: 'stats' key missing; keys present: {sorted(result.keys())}"
        )
        stats = result.get("stats", {})
        assert isinstance(stats, dict), (
            f"§4.2: 'stats' must be a dict; got {type(stats).__name__!r}"
        )
        required_stats_keys = {"min", "max", "mean", "nan_count", "inf_count"}
        missing = required_stats_keys - set(stats.keys())
        assert not missing, (
            f"§4.2: 'stats' missing sub-keys: {missing}; got: {sorted(stats.keys())}"
        )

    @pytest.mark.asyncio
    async def test_return_contains_truncated_key(self, bridge_mock):
        """§4.2: return dict must contain 'truncated' key (bool)."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_read_pixels

            ctx_mock = MagicMock()
            result = await render_read_pixels(
                source="/tmp/test.exr",
                plane="C",
                mode="summary",
                roi=None,
                max_pixels=4096,
                downsample=1,
                page=0,
                page_size=1024,
                ctx=ctx_mock,
            )

        assert "truncated" in result, (
            f"§4.2: 'truncated' key missing; keys present: {sorted(result.keys())}"
        )
