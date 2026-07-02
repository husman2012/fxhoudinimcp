"""pytest wrapper tests for houdini_usd_export_layer (PP12-112 PR-3, pp12-112c).

Unit: pp12-112c
testVerificationSurface: pytest-model
planSha: d29cbeba4743dff2f7006c135bd8c7ba591b6fcd7e77496b3d875ada4dcae254

These tests are written BEFORE the implementation (red phase). They will fail
with ImportError until hou-dev implements:
  - fxhoudinimcp/tools/usd_export_tools.py    (adds houdini_usd_export_layer)
  - fxhoudinimcp_server/handlers/usd_export_handlers.py
    (adds usd_export_layer handler + _preview_export_layer preview_fn)

Grounded against (per the rev3 lockedFieldContract -- ground vs SHIPPED code,
NOT spec prose):
  - python/fxhoudinimcp/tests/test_usd_export_tools.py (pp12-112b)
      the MagicMock(spec=HoudiniBridge) pattern; import-inside-test-after-patch;
      assert on PUBLIC behavior (command + params + result), not call order.
  - python/fxhoudinimcp/tools/export_tools.py (houdini_export_vat)
      the SINGLE bridge.execute(...) gated-wrapper shape:
          bridge = _get_bridge(ctx)
          return await bridge.execute("export_vat", {...})
  - python/fxhoudinimcp/tools/usd_export_tools.py (pp12-112b, existing two
      wrappers) -- the _get_bridge reference style this module already uses
      (module-level `import fxhoudinimcp.server as _fxserver; mcp = _fxserver.mcp`,
      then `_fxserver._get_bridge(ctx)` INSIDE the wrapper body so
      `patch("fxhoudinimcp.server._get_bridge", ...)` intercepts correctly).

PP12-110 lessons encoded here:
  - MagicMock(spec=HoudiniBridge) -- a non-existent attr (e.g. .call) raises
    AttributeError.  Bare MagicMock() is BANNED (masks convention bugs).
  - Import subject INSIDE each test, AFTER _get_bridge is patched.
  - Assert PUBLIC behavior (bridge.execute cmd+params; verbatim result
    passthrough), NOT call order / internal call count beyond "exactly one".

Contract under test (plan rev3 lockedFieldContract):
    houdini_usd_export_layer(ctx: Context, node: str, out_path: str,
                              flatten: bool = False,
                              default_prim: str | None = None) -> dict
    -> a SINGLE bridge.execute('usd_export_layer', {
           'node_path': node, 'out_path': out_path,
           'flatten': flatten, 'default_prim': default_prim,
       })
    meta={'require_approval': True}  (GATED -- first mutating tool in this family)

The wrapper MUST return bridge.execute's result VERBATIM -- including a
pending-approval / preview response shape (the 109 gate).  A pending/preview
response is NOT an error and must not be reinterpreted or swallowed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import REAL HoudiniBridge for spec= (PP12-110 lesson: bare MagicMock()
# silently accepts .call and any other non-existent attribute, masking
# convention bugs like calling bridge.call() instead of bridge.execute()).
from fxhoudinimcp.bridge import HoudiniBridge


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def export_layer_bridge_mock():
    """Spec-bound bridge mock returning a successful gated-export result shape."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "ok": True,
        "out_path": "$HIP/x.usdc",
        "format": "usdc",
        "actual_format": "usdc",
        "validator_post": {
            "ok": True,
            "mode": "postwrite",
            "omitted_checks": [],
            "verdict": "pass",
            "checks": [
                {"id": "no_world_wrapper", "status": "pass"},
                {"id": "default_prim_set", "status": "pass"},
            ],
            "wrote_files": False,
        },
    })
    return mock


@pytest.fixture()
def pending_approval_bridge_mock():
    """Spec-bound bridge mock returning a 109-gate pending-approval shape.

    A pending/preview response is NOT a failure -- the wrapper must pass it
    through verbatim, not reinterpret it as an error (rev3 lockedFieldContract
    FOLD plan-5: 'MUST NOT misinterpret a pending-approval / preview response
    shape as failure').
    """
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "status": "pending_approval",
        "pending_id": "pend-abc123",
        "preview": {
            "out_path": "$HIP/x.usdc",
            "resolved_format": "usdc",
            "pre_validation": {
                "ok": True,
                "mode": "preflight",
                "verdict": "pass",
            },
            "flatten": False,
            "default_prim": None,
            "no_world_wrapper": True,
        },
    })
    return mock


# ---------------------------------------------------------------------------
# ULC-1: module import + callable (PRIMARY RED GATE)
# ---------------------------------------------------------------------------

class TestUsdExportLayerModuleImport:
    """ULC-1: houdini_usd_export_layer must be a callable on usd_export_tools.

    The wrapper is added to the EXISTING usd_export_tools.py module (pp12-112b
    already shipped houdini_usd_inspect_layer + houdini_usd_validate there).
    Until hou-dev adds houdini_usd_export_layer, this import raises ImportError
    -- that is the RED signal for this file.
    """

    def test_module_importable(self):
        """usd_export_tools must remain importable (pp12-112b baseline)."""
        import fxhoudinimcp.tools.usd_export_tools  # noqa: F401

    def test_export_layer_callable_on_module(self):
        """houdini_usd_export_layer must be a callable (the @mcp.tool coroutine).

        FAILS RED until hou-dev adds the wrapper.
        """
        from fxhoudinimcp.tools.usd_export_tools import houdini_usd_export_layer  # noqa: F401
        assert callable(houdini_usd_export_layer), (
            "houdini_usd_export_layer must be a callable (the @mcp.tool coroutine)."
        )


# ---------------------------------------------------------------------------
# ULC-2: GATED -- meta require_approval=True (this is the FIRST mutating tool
#          of the usd_export family; require_approval MUST be True, unlike
#          the two pp12-112b read-only tools which are require_approval=False)
# ---------------------------------------------------------------------------

class TestUsdExportLayerRequireApproval:
    """ULC-2: houdini_usd_export_layer must have meta={'require_approval': True}.

    This is the first GATED (mutating) tool in the usd_export family -- it
    writes a USD layer to disk via Sdf.Layer.Export().  Every shipped gated
    wrapper (houdini_export_vat, houdini_export_alembic_ue, ...) sets
    require_approval=True; this tool must match that convention exactly.
    """

    def test_mcp_tool_meta_require_approval_true(self):
        """@mcp.tool(meta={'require_approval': True}) on houdini_usd_export_layer."""
        # red-4 (Minor): import usd_export_tools explicitly so the @mcp.tool
        # registration side-effect has definitely run for THIS test class,
        # rather than relying on an earlier test class's import (module-level
        # caching would make that work, but this class must be runnable
        # standalone, e.g. `pytest -k TestUsdExportLayerRequireApproval`).
        import fxhoudinimcp.tools.usd_export_tools  # noqa: F401
        from fxhoudinimcp.server import mcp

        tool_map: dict = {}
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_map = mcp._tool_manager._tools
        elif hasattr(mcp, "_tools"):
            tool_map = mcp._tools

        assert "houdini_usd_export_layer" in tool_map, (
            "houdini_usd_export_layer not registered on the mcp server; "
            "hou-dev must define it (decorated with @mcp.tool) in usd_export_tools.py."
        )

        tool_obj = tool_map["houdini_usd_export_layer"]
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", False) if isinstance(meta, dict) else False

        assert require_approval is True, (
            f"houdini_usd_export_layer meta must have require_approval=True (GATED -- "
            f"the first mutating tool in this family; PP12-109 security gate). "
            f"Got meta={meta!r}."
        )


# ---------------------------------------------------------------------------
# ULC-3: bridge.execute call contract -- SINGLE call, exact command + params
# ---------------------------------------------------------------------------

class TestUsdExportLayerBridgeContract:
    """ULC-3: houdini_usd_export_layer must delegate to bridge.execute with the
    EXACT single-call contract from the rev3 lockedFieldContract.

    Locked field contract (plan rev3):
        houdini_usd_export_layer(ctx, node, out_path, flatten=False, default_prim=None)
        -> a SINGLE bridge.execute('usd_export_layer', {
               'node_path': node, 'out_path': out_path,
               'flatten': flatten, 'default_prim': default_prim,
           })

    This is the FOLD plan-2/plan-5 fix: earlier plan revisions proposed a
    novel 3-call wrapper (pre-validate / export / post-validate all client-side).
    The adopted contract is a SINGLE bridge.execute call -- matching every
    other shipped gated wrapper (houdini_export_vat et al.) -- because the
    pre-flight validation lives in the gate's preview_fn and the post-write
    validation is INLINE in the handler, both server-side.
    """

    @pytest.mark.asyncio
    async def test_exactly_one_bridge_execute_call(self, export_layer_bridge_mock):
        """The wrapper must call bridge.execute EXACTLY ONCE (no multi-call wrapper)."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=export_layer_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_export_layer

            ctx_mock = MagicMock()
            await houdini_usd_export_layer(
                ctx=ctx_mock,
                node="/stage/sphere1",
                out_path="$HIP/x.usdc",
            )

        export_layer_bridge_mock.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_command_string_is_usd_export_layer(self, export_layer_bridge_mock):
        """bridge.execute must be called with command='usd_export_layer' (bare, not lops.*)."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=export_layer_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_export_layer

            ctx_mock = MagicMock()
            await houdini_usd_export_layer(
                ctx=ctx_mock,
                node="/stage/sphere1",
                out_path="$HIP/x.usdc",
            )

        call_args = export_layer_bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "usd_export_layer", (
            f"Expected bridge.execute('usd_export_layer', ...) but got command={command!r}. "
            "Command string must match register_handler's first arg exactly "
            "(the 4-bug convention: command == register name == params keys == handler kwargs)."
        )

    @pytest.mark.asyncio
    async def test_params_dict_has_exact_four_keys(self, export_layer_bridge_mock):
        """params dict must be EXACTLY {node_path, out_path, flatten, default_prim}."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=export_layer_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_export_layer

            ctx_mock = MagicMock()
            await houdini_usd_export_layer(
                ctx=ctx_mock,
                node="/stage/sphere1",
                out_path="$HIP/x.usdc",
                flatten=True,
                default_prim="/sphere1",
            )

        call_args = export_layer_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        expected_keys = {"node_path", "out_path", "flatten", "default_prim"}
        assert set(params.keys()) == expected_keys, (
            f"params dict must be exactly {expected_keys!r}. Got keys={set(params.keys())!r}."
        )
        assert params["node_path"] == "/stage/sphere1", (
            f"node_path must be forwarded from the 'node' arg verbatim. "
            f"Got {params['node_path']!r}."
        )
        assert params["out_path"] == "$HIP/x.usdc", (
            f"out_path must be forwarded verbatim. Got {params['out_path']!r}."
        )
        assert params["flatten"] is True, (
            f"flatten must be forwarded verbatim. Got {params['flatten']!r}."
        )
        assert params["default_prim"] == "/sphere1", (
            f"default_prim must be forwarded verbatim. Got {params['default_prim']!r}."
        )

    @pytest.mark.asyncio
    async def test_params_defaults_when_optional_args_omitted(self, export_layer_bridge_mock):
        """flatten defaults to False and default_prim defaults to None when omitted."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=export_layer_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_export_layer

            ctx_mock = MagicMock()
            await houdini_usd_export_layer(
                ctx=ctx_mock,
                node="/stage/sphere1",
                out_path="$HIP/x.usda",
            )

        call_args = export_layer_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        assert params.get("flatten") is False, (
            f"flatten must default to False. Got {params.get('flatten')!r}."
        )
        assert params.get("default_prim") is None, (
            f"default_prim must default to None. Got {params.get('default_prim')!r}."
        )


# ---------------------------------------------------------------------------
# ULC-4: verbatim passthrough -- INCLUDING a pending-approval / preview shape
#          (this is the FOLD plan-5 regression guard: the wrapper must NOT
#          misinterpret a pending-approval response as a failure)
# ---------------------------------------------------------------------------

class TestUsdExportLayerResultPassthrough:
    """ULC-4: the wrapper must return bridge.execute's result VERBATIM.

    Critically this includes the 109-gate pending-approval / preview response
    shape -- the wrapper is a thin single bridge.execute call with NO result
    interpretation.  A wrapper that checks `if not result.get("ok"): raise`
    would incorrectly treat a pending-approval response (which has no "ok"
    key at all) as a failure.  This is exactly the ambiguity the rev3 contract
    (FOLD plan-5) resolved by making the wrapper a single dumb passthrough.
    """

    @pytest.mark.asyncio
    async def test_success_result_passed_through_verbatim(self, export_layer_bridge_mock):
        """On a normal (already-approved / trusted-gate) success, result is unchanged."""
        expected_result = export_layer_bridge_mock.execute.return_value
        with patch("fxhoudinimcp.server._get_bridge", return_value=export_layer_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_export_layer

            ctx_mock = MagicMock()
            result = await houdini_usd_export_layer(
                ctx=ctx_mock,
                node="/stage/sphere1",
                out_path="$HIP/x.usdc",
            )

        assert result == expected_result, (
            "Wrapper must pass bridge.execute's result through unchanged. "
            f"Expected {expected_result!r}, got {result!r}."
        )

    @pytest.mark.asyncio
    async def test_pending_approval_result_passed_through_not_treated_as_failure(
        self, pending_approval_bridge_mock
    ):
        """A pending-approval/preview shape (the 109 gate) is NOT an error --
        the wrapper must return it verbatim, unmodified, un-raised.
        """
        expected_result = pending_approval_bridge_mock.execute.return_value
        with patch("fxhoudinimcp.server._get_bridge", return_value=pending_approval_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_export_layer

            ctx_mock = MagicMock()
            # Must NOT raise -- a pending-approval response is a normal, valid
            # return value from bridge.execute, not an exception-worthy state.
            result = await houdini_usd_export_layer(
                ctx=ctx_mock,
                node="/stage/sphere1",
                out_path="$HIP/x.usdc",
            )

        assert result == expected_result, (
            "A pending-approval bridge response must be returned VERBATIM -- "
            "the wrapper must not reinterpret, unwrap, or swallow it. "
            f"Expected {expected_result!r}, got {result!r}."
        )
        assert result.get("status") == "pending_approval", (
            "The pending_approval status key must survive the wrapper untouched."
        )
        # Regression guard: the wrapper must not have injected an 'ok' key or
        # otherwise mutated the pending-approval shape into a pass/fail shape.
        assert "ok" not in result, (
            f"Wrapper must not inject an 'ok' key into a pending-approval response "
            f"(that would imply the wrapper is interpreting/reshaping the result "
            f"rather than passing it through verbatim). Got keys={list(result.keys())!r}."
        )


# ---------------------------------------------------------------------------
# ULC-5: spec-bound bridge mock -- PP12-110 regression guard
# ---------------------------------------------------------------------------

class TestBridgeSpecBoundGuard:
    """ULC-5: MagicMock(spec=HoudiniBridge) -- .call raises AttributeError.

    In PP12-110, a wrapper called bridge.call(...) instead of bridge.execute(...).
    A bare MagicMock() silently accepted .call and returned a mock, masking the
    bug. A spec-bound mock (spec=HoudiniBridge) raises AttributeError because
    .call is not a real method on HoudiniBridge -- proving this test suite's
    bridge mocks would catch that class of bug if it recurred here.
    """

    def test_bridge_call_attribute_does_not_exist(self):
        """MagicMock(spec=HoudiniBridge).call raises AttributeError -- .call is not on the class."""
        mock = MagicMock(spec=HoudiniBridge)
        with pytest.raises(AttributeError, match="call"):
            _ = mock.call  # .call must NOT exist on HoudiniBridge

    def test_bridge_execute_attribute_exists(self):
        """MagicMock(spec=HoudiniBridge).execute is accessible (execute IS on the class)."""
        mock = MagicMock(spec=HoudiniBridge)
        _ = mock.execute  # must NOT raise


# ---------------------------------------------------------------------------
# ULC-6: ctx not in tool input schema (ctx is injected, not a tool parameter)
# ---------------------------------------------------------------------------

class TestCtxSchemaGuard:
    """ULC-6: 'ctx' must NOT appear in houdini_usd_export_layer's input schema.

    FastMCP injects ctx via the Context type annotation -- it is NOT a
    parameter the MCP client sends. Mirrors the pp12-112b UEC-7 guard.
    """

    def test_export_layer_ctx_not_in_schema(self):
        """houdini_usd_export_layer: 'ctx' must not be a property in the input schema."""
        # red-4 (Minor): explicit import so the @mcp.tool registration
        # side-effect has run for THIS class standalone (see the identical
        # rationale in TestUsdExportLayerRequireApproval above).
        import fxhoudinimcp.tools.usd_export_tools  # noqa: F401
        from fxhoudinimcp.server import mcp

        tool_map: dict = {}
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_map = mcp._tool_manager._tools
        elif hasattr(mcp, "_tools"):
            tool_map = mcp._tools
        tool_obj = tool_map.get("houdini_usd_export_layer")
        assert tool_obj is not None, "houdini_usd_export_layer not registered on the mcp server."

        schema = getattr(tool_obj, "parameters", getattr(tool_obj, "inputSchema", {})) or {}
        properties = schema.get("properties", {})
        assert "ctx" not in properties, (
            f"'ctx' must NOT appear in houdini_usd_export_layer's input schema properties. "
            f"FastMCP injects it; clients don't provide it. Got properties={list(properties.keys())!r}."
        )
