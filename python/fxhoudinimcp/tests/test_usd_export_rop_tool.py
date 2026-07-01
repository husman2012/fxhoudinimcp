"""pytest wrapper tests for houdini_usd_export_rop (PP12-112 PR-4, pp12-112d).

Unit: pp12-112d
testVerificationSurface: pytest-model
planSha: 27f1d7b8428d108f13de6cad095a4f476de5f41319d4b8972c481244cd5c67ff

These tests are written BEFORE the implementation (red phase). They will fail
with ImportError until hou-dev implements:
  - fxhoudinimcp/tools/usd_export_tools.py    (adds houdini_usd_export_rop)
  - fxhoudinimcp_server/handlers/usd_export_handlers.py
    (adds usd_export_rop handler + _preview_export_rop preview_fn)

Grounded against (per the rev3 lockedFieldContract -- ground vs SHIPPED code,
NOT spec prose):
  - python/fxhoudinimcp/tests/test_usd_export_layer_tool.py (pp12-112c, THE
      layer-for-layer template for this file)
      the MagicMock(spec=HoudiniBridge) pattern; import-inside-test-after-patch;
      assert on PUBLIC behavior (command + params + result), not call order;
      verbatim pending-approval passthrough regression guard; ctx-not-in-schema
      guard.
  - houdini/scripts/python/fxhoudinimcp_server/handlers/usd_export_handlers.py
      (pp12-112c, usd_export_layer + _preview_export_layer) -- the SAME
      preview_fn(params: dict) positional-single-arg convention; the SAME
      keyword-only handler(**params) convention; the SAME gated
      register_handler(..., Capability.MUTATING, preview_fn=..., preview_required=True)
      registration shape.
  - python/fxhoudinimcp/tools/usd_export_tools.py (pp12-112c, existing
      houdini_usd_export_layer wrapper) -- the exact _get_bridge reference
      style this module already uses (module-level
      `import fxhoudinimcp.server as _fxserver; mcp = _fxserver.mcp`,
      then `_fxserver._get_bridge(ctx)` INSIDE the wrapper body so
      `patch("fxhoudinimcp.server._get_bridge", ...)` intercepts correctly).

PP12-110 lessons encoded here:
  - MagicMock(spec=HoudiniBridge) -- a non-existent attr (e.g. .call) raises
    AttributeError.  Bare MagicMock() is BANNED (masks convention bugs).
  - Import subject INSIDE each test, AFTER _get_bridge is patched.
  - Assert PUBLIC behavior (bridge.execute cmd+params; verbatim result
    passthrough), NOT call order / internal call count beyond "exactly one".

Contract under test (plan rev3 lockedFieldContract):
    houdini_usd_export_rop(ctx: Context, lop_node: str, out_path: str,
                            frame_range: list | None = None) -> dict
    -> a SINGLE bridge.execute('usd_export_rop', {
           'lop_node': lop_node, 'out_path': out_path,
           'frame_range': frame_range,
       })
    meta={'require_approval': True}  (GATED -- the SECOND mutating tool of
    this family, after usd_export_layer)

The wrapper MUST return bridge.execute's result VERBATIM -- including a
pending-approval / preview response shape (the 109 gate). A pending/preview
response is NOT an error and must not be reinterpreted or swallowed.

Naming note (rev3 FOLD plan-11): the param is named `lop_node` (not `node` as
in PR-3's usd_export_layer) because the USD ROP renders a specific LOP node's
composed stage -- the more precise name; matches spec.md 112 section 4.1
signature houdini_usd_export_rop(lop_node, out_path, ...).
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
def export_rop_bridge_mock():
    """Spec-bound bridge mock returning a successful gated-ROP-export result shape."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "ok": True,
        "out_path": "$HIP/rop.usdc",
        "format": "usdc",
        "actual_format": "usdc",
        "validator_post": {
            "ok": True,
            "mode": "postwrite",
            "omitted_checks": [],
            "verdict": "pass",
            "checks": [
                {"id": "no_world_wrapper", "status": "pass"},
                {"id": "format_matches_ext", "status": "pass"},
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
    -- mirrors the FOLD plan-5 regression guard from usd_export_layer).
    """
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "status": "pending_approval",
        "pending_id": "pend-rop-xyz789",
        "preview": {
            "out_path": "$HIP/rop.usdc",
            "resolved_format": "usdc",
            "pre_validation": {
                "ok": True,
                "mode": "preflight",
                "verdict": "pass",
            },
            "frame_range": None,
            "driven_via": "usd ROP (/out-context)",
            "no_world_wrapper": True,
        },
    })
    return mock


# ---------------------------------------------------------------------------
# URC-1: module import + callable (PRIMARY RED GATE)
# ---------------------------------------------------------------------------

class TestUsdExportRopModuleImport:
    """URC-1: houdini_usd_export_rop must be a callable on usd_export_tools.

    The wrapper is added to the EXISTING usd_export_tools.py module (which
    already ships houdini_usd_inspect_layer, houdini_usd_validate, and
    houdini_usd_export_layer). Until hou-dev adds houdini_usd_export_rop,
    this import raises ImportError -- that is the RED signal for this file.
    """

    def test_module_importable(self):
        """usd_export_tools must remain importable (pp12-112c baseline)."""
        import fxhoudinimcp.tools.usd_export_tools  # noqa: F401

    def test_existing_export_layer_wrapper_unaffected(self):
        """houdini_usd_export_layer (pp12-112c) must still be importable --
        this new wrapper must not clobber the existing gated sibling."""
        from fxhoudinimcp.tools.usd_export_tools import houdini_usd_export_layer  # noqa: F401
        assert callable(houdini_usd_export_layer)

    def test_export_rop_callable_on_module(self):
        """houdini_usd_export_rop must be a callable (the @mcp.tool coroutine).

        FAILS RED until hou-dev adds the wrapper.
        """
        from fxhoudinimcp.tools.usd_export_tools import houdini_usd_export_rop  # noqa: F401
        assert callable(houdini_usd_export_rop), (
            "houdini_usd_export_rop must be a callable (the @mcp.tool coroutine)."
        )


# ---------------------------------------------------------------------------
# URC-2: GATED -- meta require_approval=True (the SECOND mutating tool of the
#          usd_export family, matching usd_export_layer's convention exactly)
# ---------------------------------------------------------------------------

class TestUsdExportRopRequireApproval:
    """URC-2: houdini_usd_export_rop must have meta={'require_approval': True}.

    This is the second GATED (mutating) tool in the usd_export family -- it
    drives the /out `usd` ROP to write a composed stage to disk. Every shipped
    gated wrapper (houdini_usd_export_layer, houdini_export_vat, ...) sets
    require_approval=True; this tool must match that convention exactly.
    """

    def test_mcp_tool_meta_require_approval_true(self):
        """@mcp.tool(meta={'require_approval': True}) on houdini_usd_export_rop."""
        # Explicit import so the @mcp.tool registration side-effect has
        # definitely run for THIS test class, rather than relying on an
        # earlier test class's import (module-level caching would make that
        # work, but this class must be runnable standalone, e.g.
        # `pytest -k TestUsdExportRopRequireApproval`).
        import fxhoudinimcp.tools.usd_export_tools  # noqa: F401
        from fxhoudinimcp.server import mcp

        tool_map: dict = {}
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_map = mcp._tool_manager._tools
        elif hasattr(mcp, "_tools"):
            tool_map = mcp._tools

        assert "houdini_usd_export_rop" in tool_map, (
            "houdini_usd_export_rop not registered on the mcp server; "
            "hou-dev must define it (decorated with @mcp.tool) in usd_export_tools.py."
        )

        tool_obj = tool_map["houdini_usd_export_rop"]
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", False) if isinstance(meta, dict) else False

        assert require_approval is True, (
            f"houdini_usd_export_rop meta must have require_approval=True (GATED -- "
            f"a mutating tool in this family; PP12-109 security gate). "
            f"Got meta={meta!r}."
        )


# ---------------------------------------------------------------------------
# URC-3: bridge.execute call contract -- SINGLE call, exact command + params
# ---------------------------------------------------------------------------

class TestUsdExportRopBridgeContract:
    """URC-3: houdini_usd_export_rop must delegate to bridge.execute with the
    EXACT single-call contract from the rev3 lockedFieldContract.

    Locked field contract (plan rev3):
        houdini_usd_export_rop(ctx, lop_node, out_path, frame_range=None)
        -> a SINGLE bridge.execute('usd_export_rop', {
               'lop_node': lop_node, 'out_path': out_path,
               'frame_range': frame_range,
           })

    Matches every other shipped gated wrapper (houdini_usd_export_layer,
    houdini_export_vat et al.) -- the pre-flight validation lives in the
    gate's preview_fn and the post-write validation is INLINE in the
    handler, both server-side. The wrapper is a thin single-call passthrough.
    """

    @pytest.mark.asyncio
    async def test_exactly_one_bridge_execute_call(self, export_rop_bridge_mock):
        """The wrapper must call bridge.execute EXACTLY ONCE (no multi-call wrapper)."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=export_rop_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_export_rop

            ctx_mock = MagicMock()
            await houdini_usd_export_rop(
                ctx=ctx_mock,
                lop_node="/stage/sphere1",
                out_path="$HIP/rop.usdc",
            )

        export_rop_bridge_mock.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_command_string_is_usd_export_rop(self, export_rop_bridge_mock):
        """bridge.execute must be called with command='usd_export_rop' (bare, not lops.*)."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=export_rop_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_export_rop

            ctx_mock = MagicMock()
            await houdini_usd_export_rop(
                ctx=ctx_mock,
                lop_node="/stage/sphere1",
                out_path="$HIP/rop.usdc",
            )

        call_args = export_rop_bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "usd_export_rop", (
            f"Expected bridge.execute('usd_export_rop', ...) but got command={command!r}. "
            "Command string must match register_handler's first arg exactly "
            "(the 4-bug convention: command == register name == params keys == handler kwargs)."
        )

    @pytest.mark.asyncio
    async def test_params_dict_has_exact_three_keys(self, export_rop_bridge_mock):
        """params dict must be EXACTLY {lop_node, out_path, frame_range}."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=export_rop_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_export_rop

            ctx_mock = MagicMock()
            await houdini_usd_export_rop(
                ctx=ctx_mock,
                lop_node="/stage/sphere1",
                out_path="$HIP/rop.usdc",
                frame_range=[1, 10],
            )

        call_args = export_rop_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        expected_keys = {"lop_node", "out_path", "frame_range"}
        assert set(params.keys()) == expected_keys, (
            f"params dict must be exactly {expected_keys!r}. Got keys={set(params.keys())!r}."
        )
        assert params["lop_node"] == "/stage/sphere1", (
            f"lop_node must be forwarded from the 'lop_node' arg verbatim. "
            f"Got {params['lop_node']!r}."
        )
        assert params["out_path"] == "$HIP/rop.usdc", (
            f"out_path must be forwarded verbatim. Got {params['out_path']!r}."
        )
        assert params["frame_range"] == [1, 10], (
            f"frame_range must be forwarded verbatim. Got {params['frame_range']!r}."
        )

    @pytest.mark.asyncio
    async def test_params_frame_range_defaults_to_none_when_omitted(self, export_rop_bridge_mock):
        """frame_range defaults to None when omitted (current-frame export)."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=export_rop_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_export_rop

            ctx_mock = MagicMock()
            await houdini_usd_export_rop(
                ctx=ctx_mock,
                lop_node="/stage/sphere1",
                out_path="$HIP/rop.usda",
            )

        call_args = export_rop_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        assert params.get("frame_range") is None, (
            f"frame_range must default to None. Got {params.get('frame_range')!r}."
        )


# ---------------------------------------------------------------------------
# URC-4: verbatim passthrough -- INCLUDING a pending-approval / preview shape
# ---------------------------------------------------------------------------

class TestUsdExportRopResultPassthrough:
    """URC-4: the wrapper must return bridge.execute's result VERBATIM.

    Critically this includes the 109-gate pending-approval / preview response
    shape -- the wrapper is a thin single bridge.execute call with NO result
    interpretation. A wrapper that checks `if not result.get("ok"): raise`
    would incorrectly treat a pending-approval response (which has no "ok"
    key at all) as a failure.
    """

    @pytest.mark.asyncio
    async def test_success_result_passed_through_verbatim(self, export_rop_bridge_mock):
        """On a normal (already-approved / trusted-gate) success, result is unchanged."""
        expected_result = export_rop_bridge_mock.execute.return_value
        with patch("fxhoudinimcp.server._get_bridge", return_value=export_rop_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_export_rop

            ctx_mock = MagicMock()
            result = await houdini_usd_export_rop(
                ctx=ctx_mock,
                lop_node="/stage/sphere1",
                out_path="$HIP/rop.usdc",
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
            from fxhoudinimcp.tools.usd_export_tools import houdini_usd_export_rop

            ctx_mock = MagicMock()
            # Must NOT raise -- a pending-approval response is a normal, valid
            # return value from bridge.execute, not an exception-worthy state.
            result = await houdini_usd_export_rop(
                ctx=ctx_mock,
                lop_node="/stage/sphere1",
                out_path="$HIP/rop.usdc",
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
# URC-5: spec-bound bridge mock -- PP12-110 regression guard
# ---------------------------------------------------------------------------

class TestBridgeSpecBoundGuard:
    """URC-5: MagicMock(spec=HoudiniBridge) -- .call raises AttributeError.

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
# URC-6: ctx not in tool input schema (ctx is injected, not a tool parameter)
# ---------------------------------------------------------------------------

class TestCtxSchemaGuard:
    """URC-6: 'ctx' must NOT appear in houdini_usd_export_rop's input schema.

    FastMCP injects ctx via the Context type annotation -- it is NOT a
    parameter the MCP client sends. Mirrors the pp12-112c ULC-6 guard.
    """

    def test_export_rop_ctx_not_in_schema(self):
        """houdini_usd_export_rop: 'ctx' must not be a property in the input schema."""
        # Explicit import so the @mcp.tool registration side-effect has run
        # for THIS class standalone (see the identical rationale in
        # TestUsdExportRopRequireApproval above).
        import fxhoudinimcp.tools.usd_export_tools  # noqa: F401
        from fxhoudinimcp.server import mcp

        tool_map: dict = {}
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_map = mcp._tool_manager._tools
        elif hasattr(mcp, "_tools"):
            tool_map = mcp._tools
        tool_obj = tool_map.get("houdini_usd_export_rop")
        assert tool_obj is not None, "houdini_usd_export_rop not registered on the mcp server."

        schema = getattr(tool_obj, "parameters", getattr(tool_obj, "inputSchema", {})) or {}
        properties = schema.get("properties", {})
        assert "ctx" not in properties, (
            f"'ctx' must NOT appear in houdini_usd_export_rop's input schema properties. "
            f"FastMCP injects it; clients don't provide it. Got properties={list(properties.keys())!r}."
        )
