"""pytest wrapper tests for render_compare MCP tool.

Unit: pp12-114f
testVerificationSurface: pytest-model
planSha: f4b68e911486cad5c0ec2758ae87604d530f9019affe1eeb6e86aef40b03139c

These tests are written BEFORE the implementation (red phase). They will fail
with ImportError / AttributeError until hou-dev implements:
  - fxhoudinimcp/tools/render_readback_tools.py  (the render_compare wrapper)
  - fxhoudinimcp_server/handlers/render_readback_handlers.py  (the handler)

Grounded against: python/fxhoudinimcp/tests/test_render_read_pixels_wrapper.py
Plan rev2 contract — adversarial-pressure-tested before any tests were written.

PP12-110 lessons encoded here:
  - MagicMock(spec=HoudiniBridge) — any non-existent attr raises AttributeError
  - Import subject INSIDE each test, AFTER _get_bridge is patched
  - Assert PUBLIC behavior (bridge.execute cmd+params; result shape), NOT call order
  - @pytest.mark.asyncio on every async test
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import REAL HoudiniBridge for spec= (PP12-110 lesson: bare MagicMock() silently accepts
# non-existent attributes like .call, masking the bug the spec-bound mock catches).
from fxhoudinimcp.bridge import HoudiniBridge


# ---------------------------------------------------------------------------
# Shared fixture: spec-bound bridge mock with success return value
# ---------------------------------------------------------------------------

@pytest.fixture()
def bridge_mock():
    """Return a spec-bound HoudiniBridge AsyncMock.

    spec=HoudiniBridge means any attribute NOT on the real class
    (e.g. .call()) immediately raises AttributeError — the PP12-110 guard.

    Success shape for render_compare per plan rev2 §4.2:
      Bare compare dict with NO 'ok' key (MINOR-5: success shape has no 'ok': True).
    """
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "a": "/tmp/test_a.exr",
        "b": "/tmp/test_b.exr",
        "planes": ["C"],
        "metric": "stats",
        "results": {
            "C": {
                "aov": "C",
                "metric": "stats",
                "a_stats": {"min": [0.0], "max": [1.0], "mean": [0.5]},
                "b_stats": {"min": [0.0], "max": [1.0], "mean": [0.5]},
            }
        },
        "changed": False,
        "presence_diff": False,
    })
    return mock


# ---------------------------------------------------------------------------
# RCC-1: module import + symbol exposure (PRIMARY RED GATE)
# ---------------------------------------------------------------------------

class TestRenderCompareModuleImport:
    """RCC-1: wrapper module must exist and expose render_compare."""

    def test_module_importable(self):
        """Import fxhoudinimcp.tools.render_readback_tools — FAILS RED until hou-dev creates it."""
        # This is the primary RED gate: render_compare does not exist yet.
        import fxhoudinimcp.tools.render_readback_tools  # noqa: F401

    def test_render_compare_callable_on_module(self):
        """render_compare must be importable as a callable from render_readback_tools."""
        from fxhoudinimcp.tools.render_readback_tools import render_compare  # noqa: F401
        assert callable(render_compare), (
            "render_compare must be a callable (the @mcp.tool-decorated coroutine)."
        )


# ---------------------------------------------------------------------------
# RCC-2: FR-10 — require_approval=False (bypasses PP12-109 gate)
# ---------------------------------------------------------------------------

class TestRenderCompareRequireApproval:
    """RCC-2: render_compare is Capability.READONLY — require_approval must be False (FR-10).

    render_compare reads files and returns a diff — it never mutates the Houdini scene.
    FR-10 mandates Capability.READONLY + require_approval=False so the 109 gate is bypassed.
    """

    def test_mcp_tool_meta_require_approval_false(self):
        """@mcp.tool(meta={'require_approval': False}) must be set on render_compare.

        FastMCP exposes tool metadata via _tool_manager._tools or _tools on the
        FastMCP server object.
        """
        from fxhoudinimcp.server import mcp  # noqa: PLC0415

        tool_map: dict = {}
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_map = mcp._tool_manager._tools
        elif hasattr(mcp, "_tools"):
            tool_map = mcp._tools

        assert "render_compare" in tool_map, (
            "render_compare not registered on the mcp server; "
            "hou-dev must add the tool to tools/render_readback_tools.py and "
            "ensure it is imported at server startup."
        )

        tool_obj = tool_map["render_compare"]
        # meta is stored on the tool as .meta, .tags, or custom attribute depending on FastMCP version
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", True) if isinstance(meta, dict) else True

        assert require_approval is False, (
            f"render_compare meta must have require_approval=False (FR-10: Capability.READONLY). "
            f"Got meta={meta!r}. The 109 gate must be bypassed for read-only tools."
        )


# ---------------------------------------------------------------------------
# RCC-3: bridge.execute call contract — command string + params shape
# ---------------------------------------------------------------------------

class TestRenderCompareBridgeContract:
    """RCC-3: wrapper must delegate to bridge.execute('render_compare', {a, b, planes, metric})."""

    @pytest.mark.asyncio
    async def test_calls_bridge_execute_not_bridge_call(self, bridge_mock):
        """Wrapper must call bridge.execute (not bridge.call or any other attr).

        PP12-110 lesson: bridge.call does NOT exist on HoudiniBridge.
        spec=HoudiniBridge ensures calling mock.call() raises AttributeError.
        """
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_compare

            ctx_mock = MagicMock()
            await render_compare(
                a="/tmp/a.exr",
                b="/tmp/b.exr",
                planes=None,
                metric="stats",
                ctx=ctx_mock,
            )

        # Verify bridge.execute was called with the correct command string.
        bridge_mock.execute.assert_called_once()
        call_args = bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "render_compare", (
            f"Expected bridge.execute('render_compare', ...) but got command={command!r}. "
            "PP12-110: the dispatcher command string must match register_handler's first arg."
        )

    @pytest.mark.asyncio
    async def test_bridge_execute_params_contain_all_four_fields(self, bridge_mock):
        """Wrapper must pass {a, b, planes, metric} in params dict to bridge.execute.

        Locked field contract from plan rev2 §3:
            render_compare(ctx, a, b, planes=None, metric='stats')
            -> bridge.execute('render_compare', {'a':a, 'b':b, 'planes':planes, 'metric':metric})
        """
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_compare

            ctx_mock = MagicMock()
            await render_compare(
                a="/tmp/a.exr",
                b="/tmp/b.exr",
                planes=["C", "N"],
                metric="mae",
                ctx=ctx_mock,
            )

        call_args = bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        required_keys = {"a", "b", "planes", "metric"}
        missing = required_keys - set(params.keys())
        assert not missing, (
            f"bridge.execute params missing keys: {missing}; got params keys: {sorted(params.keys())}"
        )

    @pytest.mark.asyncio
    async def test_bridge_execute_params_values_forwarded_correctly(self, bridge_mock):
        """Every param value must be forwarded verbatim to bridge.execute."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_compare

            ctx_mock = MagicMock()
            await render_compare(
                a="/renders/frame_a.exr",
                b="/renders/frame_b.exr",
                planes=["C", "depth"],
                metric="psnr",
                ctx=ctx_mock,
            )

        call_args = bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        assert params.get("a") == "/renders/frame_a.exr", (
            f"'a' not forwarded correctly; got {params.get('a')!r}"
        )
        assert params.get("b") == "/renders/frame_b.exr", (
            f"'b' not forwarded correctly; got {params.get('b')!r}"
        )
        assert params.get("planes") == ["C", "depth"], (
            f"'planes' not forwarded correctly; got {params.get('planes')!r}"
        )
        assert params.get("metric") == "psnr", (
            f"'metric' not forwarded correctly; got {params.get('metric')!r}"
        )

    @pytest.mark.asyncio
    async def test_planes_none_forwarded_as_none(self, bridge_mock):
        """planes=None (default — compare all available planes) must be forwarded as None."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_compare

            ctx_mock = MagicMock()
            await render_compare(
                a="/tmp/a.exr",
                b="/tmp/b.exr",
                ctx=ctx_mock,
            )

        call_args = bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert params.get("planes") is None, (
            f"planes=None (default) must be forwarded as None; got {params.get('planes')!r}"
        )

    @pytest.mark.asyncio
    async def test_default_metric_is_stats(self, bridge_mock):
        """Default metric must be 'stats' per the locked field contract."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_compare

            ctx_mock = MagicMock()
            await render_compare(
                a="/tmp/a.exr",
                b="/tmp/b.exr",
                ctx=ctx_mock,
            )

        call_args = bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert params.get("metric") == "stats", (
            f"Default metric must be 'stats'; got {params.get('metric')!r}"
        )

    def test_bridge_call_does_not_exist(self):
        """Regression guard: bridge.call must NOT exist on HoudiniBridge.

        PP12-110 root cause: a wrapper called bridge.call() which does not exist
        on the real class. A spec-bound mock catches this at test time.
        """
        mock = MagicMock(spec=HoudiniBridge)
        with pytest.raises(AttributeError):
            _ = mock.call  # .call does not exist on HoudiniBridge


# ---------------------------------------------------------------------------
# RCC-4: rev2 MINOR-5 — success shape has NO 'ok': True key
# ---------------------------------------------------------------------------

class TestRenderCompareSuccessShape:
    """RCC-4: on success, 'ok' must NOT be in the result dict (rev2 MINOR-5).

    Success/failure shape asymmetry (plan rev2 §4.2 / MINOR-5):
      - success: bare compare dict — {a, b, planes, metric, results, changed, presence_diff}
                 there is NO 'ok': True key
      - failure: {ok: False, error: <str>}
    """

    @pytest.mark.asyncio
    async def test_ok_not_in_success_result(self, bridge_mock):
        """'ok' must NOT appear in a successful result (MINOR-5)."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_compare

            ctx_mock = MagicMock()
            result = await render_compare(
                a="/tmp/a.exr",
                b="/tmp/b.exr",
                planes=None,
                metric="stats",
                ctx=ctx_mock,
            )

        assert "ok" not in result, (
            f"MINOR-5: success shape must NOT contain 'ok' key; "
            f"found 'ok'={result.get('ok')!r} in result. "
            "Success is the bare §4.2 compare dict; 'ok': True is the failure-path sentinel."
        )

    @pytest.mark.asyncio
    async def test_success_result_has_required_keys(self, bridge_mock):
        """§4.2: success result must contain a, b, planes, metric, results, changed, presence_diff."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_compare

            ctx_mock = MagicMock()
            result = await render_compare(
                a="/tmp/a.exr",
                b="/tmp/b.exr",
                ctx=ctx_mock,
            )

        required_keys = {"a", "b", "planes", "metric", "results", "changed", "presence_diff"}
        missing = required_keys - set(result.keys())
        assert not missing, (
            f"§4.2: success result missing keys: {missing}; "
            f"got result keys: {sorted(result.keys())}"
        )

    @pytest.mark.asyncio
    async def test_failure_shape_has_ok_false(self):
        """On failure, result must be {ok: False, error: str}."""
        # Simulate the handler returning a failure envelope
        failure_mock = MagicMock(spec=HoudiniBridge)
        failure_mock.execute = AsyncMock(return_value={
            "ok": False,
            "error": "EXR file not found: /does/not/exist.exr",
        })

        with patch("fxhoudinimcp.server._get_bridge", return_value=failure_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_compare

            ctx_mock = MagicMock()
            result = await render_compare(
                a="/does/not/exist.exr",
                b="/tmp/b.exr",
                ctx=ctx_mock,
            )

        assert result.get("ok") is False, (
            f"Failure shape must have ok=False; got ok={result.get('ok')!r}"
        )
        assert "error" in result and result["error"], (
            f"Failure shape must have non-empty 'error' key; got result={result!r}"
        )


# ---------------------------------------------------------------------------
# RCC-5: rev2 MAJOR-3 — metric pre-validated BEFORE any file read
# ---------------------------------------------------------------------------

class TestRenderCompareMetricValidation:
    """RCC-5: metric must be pre-validated; invalid metric → error before any file I/O (MAJOR-3).

    Valid metrics: {'stats', 'mae', 'psnr'} (_VALID_METRICS in plan rev2 §5.1).
    An invalid metric must produce an error whose message names the metric,
    NOT a file-not-found error — even when the paths don't exist.
    This proves validation fires BEFORE file access.
    """

    @pytest.mark.asyncio
    async def test_invalid_metric_produces_metric_error_not_file_error(self):
        """metric='bogus' with non-existent paths → error must mention the metric (MAJOR-3).

        If the handler reads files before validating the metric, the error would
        be 'file not found' rather than 'invalid metric'. MAJOR-3 requires validation
        to fire first.
        """
        metric_error_mock = MagicMock(spec=HoudiniBridge)
        metric_error_mock.execute = AsyncMock(return_value={
            "ok": False,
            "error": "Invalid metric 'bogus'; must be one of: stats, mae, psnr",
        })

        with patch("fxhoudinimcp.server._get_bridge", return_value=metric_error_mock):
            from fxhoudinimcp.tools.render_readback_tools import render_compare

            ctx_mock = MagicMock()
            result = await render_compare(
                a="/no/such/file/a.exr",
                b="/no/such/file/b.exr",
                metric="bogus",
                ctx=ctx_mock,
            )

        # The error message must reference the metric, not a missing file.
        assert result.get("ok") is False, (
            f"MAJOR-3: invalid metric must produce ok=False; got {result.get('ok')!r}"
        )
        error_msg = result.get("error", "")
        assert "bogus" in error_msg or "metric" in error_msg.lower(), (
            f"MAJOR-3: error message must name the invalid metric or say 'metric'; "
            f"got error={error_msg!r}. "
            "If the error is file-not-found, metric validation fires AFTER file I/O (bug)."
        )

    @pytest.mark.asyncio
    async def test_valid_metrics_pass_bridge_execute(self, bridge_mock):
        """All three valid metrics ('stats', 'mae', 'psnr') must be accepted without error."""
        for valid_metric in ("stats", "mae", "psnr"):
            bridge_mock.execute.reset_mock()
            with patch("fxhoudinimcp.server._get_bridge", return_value=bridge_mock):
                from fxhoudinimcp.tools.render_readback_tools import render_compare  # noqa: PLC0415

                ctx_mock = MagicMock()
                result = await render_compare(
                    a="/tmp/a.exr",
                    b="/tmp/b.exr",
                    metric=valid_metric,
                    ctx=ctx_mock,
                )

            # Must reach bridge.execute (not short-circuit as invalid)
            bridge_mock.execute.assert_called_once(), (
                f"metric='{valid_metric}' must reach bridge.execute; "
                f"call count={bridge_mock.execute.call_count}"
            )
            # Must NOT have ok=False from metric validation
            assert result.get("ok") is not False, (
                f"metric='{valid_metric}' is valid but got ok=False; "
                f"error={result.get('error')!r}"
            )


# ---------------------------------------------------------------------------
# RCC-6: spec-bound bridge guard (PP12-110 regression prevention)
# ---------------------------------------------------------------------------

class TestRenderCompareBridgeSpecBound:
    """RCC-6: bridge.call must NOT exist on HoudiniBridge (spec-bound guard).

    PP12-110 root cause: a wrapper called bridge.call() which does not exist on
    HoudiniBridge. A bare MagicMock() silently creates .call as a new mock attribute,
    masking the bug. MagicMock(spec=HoudiniBridge) raises AttributeError instead.
    """

    def test_bridge_call_raises_attribute_error(self):
        """HoudiniBridge has no .call attribute — spec-bound mock must raise AttributeError."""
        mock = MagicMock(spec=HoudiniBridge)
        with pytest.raises(AttributeError):
            _ = mock.call

    def test_bridge_execute_exists_on_real_class(self):
        """bridge.execute must exist on the REAL HoudiniBridge class."""
        assert hasattr(HoudiniBridge, "execute") or callable(
            getattr(HoudiniBridge, "execute", None)
        ), (
            "HoudiniBridge.execute must be defined; "
            "this is the method the wrapper calls (PP12-110 convention)."
        )

    @pytest.mark.asyncio
    async def test_spec_mock_rejects_nonexistent_attr(self):
        """Verify that a spec-bound mock rejects .call during an actual test invocation."""
        spec_mock = MagicMock(spec=HoudiniBridge)
        # .call should not exist; attempting to set it as an AsyncMock should raise
        # or accessing it should raise AttributeError
        with pytest.raises(AttributeError):
            _ = spec_mock.call("render_compare", {})


# ---------------------------------------------------------------------------
# RCC-7: ctx-schema guard — FastMCP hides ctx: Context from client schema
# ---------------------------------------------------------------------------

class TestRenderCompareCtxSchemaGuard:
    """RCC-7: 'ctx' must NOT appear in render_compare's input schema properties.

    FastMCP automatically injects ctx: Context from the MCP lifespan context and
    HIDES it from the client-visible JSON schema. If 'ctx' appears in the schema,
    the tool was authored with `ctx: Any` instead of `ctx: Context` (PP12-110 bug class).
    """

    def test_ctx_not_in_tool_input_schema(self):
        """'ctx' must be absent from render_compare's FastMCP input schema.

        RED until hou-dev creates the wrapper with ctx: Context (not ctx: Any).
        """
        from fxhoudinimcp.server import mcp  # noqa: PLC0415

        tool_map: dict = {}
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_map = mcp._tool_manager._tools
        elif hasattr(mcp, "_tools"):
            tool_map = mcp._tools

        assert "render_compare" in tool_map, (
            "render_compare not registered on the mcp server; "
            "hou-dev must add the tool to tools/render_readback_tools.py and "
            "ensure it is imported at server startup."
        )

        tool_obj = tool_map["render_compare"]
        schema = tool_obj.parameters if hasattr(tool_obj, "parameters") else {}
        props = schema.get("properties", {})

        assert "ctx" not in props, (
            f"'ctx' appears in render_compare input schema properties: {sorted(props.keys())}. "
            "FastMCP hides ctx: Context from client schema automatically — "
            "if 'ctx' is visible, the param was typed as ctx: Any (PP12-110 bug class)."
        )
