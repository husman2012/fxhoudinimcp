"""pytest wrapper tests for houdini_mtlx_inspect + houdini_mtlx_edit
(PP12-112 PR-5, pp12-112e).

Unit: pp12-112e
testVerificationSurface: pytest-model
planSha: f1babc2821e387aade3fa83df71973c9773ece3e94e974b22ddbc697db823ae3

These tests are written BEFORE the implementation (red phase). They will fail
with ImportError until hou-dev implements:
  - fxhoudinimcp/tools/usd_export_tools.py
      (adds houdini_mtlx_inspect + houdini_mtlx_edit)
  - fxhoudinimcp_server/handlers/usd_export_handlers.py
      (adds mtlx_inspect handler, mtlx_edit handler, _preview_mtlx_edit
      preview_fn, the HAS_MTLX/_require_mtlx guard, and the
      _resolve_node/_mtlx_validate/_validate_edits_shape helpers)

Grounded against (per the rev3 lockedFieldContract -- ground vs SHIPPED code,
NOT spec prose):
  - python/fxhoudinimcp/tests/test_usd_export_rop_tool.py (pp12-112d, THE
      layer-for-layer template for this file) -- the
      MagicMock(spec=HoudiniBridge) pattern; import-inside-test-after-patch;
      assert on PUBLIC behavior (command + params + result), not call order;
      verbatim pending-approval passthrough regression guard; ctx-not-in-
      schema guard; the exact bridge-access-through-module-reference style
      (`_fxserver._get_bridge(ctx)` inside the wrapper body so
      `patch("fxhoudinimcp.server._get_bridge", ...)` intercepts correctly).
  - houdini/scripts/python/fxhoudinimcp_server/handlers/usd_export_handlers.py
      (pp12-112b usd_inspect_layer -- the READONLY template: `def
      usd_inspect_layer(*, node_or_layer: str) -> dict`, FR-2 empty guard,
      FR-5 try/except -> {ok:False,error}, `return {ok:True, **summary.to_dict()}`,
      register_handler(cmd, fn, Capability.READONLY); pp12-112d usd_export_rop
      / _preview_export_rop -- the GATED template: preview_fn(params: dict)
      -> dict positional-single-arg convention, keyword-only handler(**params),
      register_handler(..., Capability.MUTATING, preview_fn=..., preview_required=True)).
  - python/fxhoudinimcp/usd_export_model.py -- MtlxSummary dataclass
      (REUSED UNMODIFIED for the inspect result; to_dict nests validate_ok /
      validate_errors under a 'validate' sub-dict with keys 'ok'/'errors').

Contract under test (plan rev3 lockedFieldContract):
    houdini_mtlx_inspect(ctx: Context, mtlx_path_or_doc: str) -> dict
    -> a SINGLE bridge.execute('mtlx_inspect', {'mtlx_path_or_doc': ...})
    meta={'require_approval': False}  (UNGATED READONLY)

    houdini_mtlx_edit(ctx: Context, mtlx_path: str, edits: list, out_path: str) -> dict
    -> a SINGLE bridge.execute('mtlx_edit', {
           'mtlx_path': mtlx_path, 'out_path': out_path, 'edits': edits,
       })
    meta={'require_approval': True}  (GATED -- MaterialX-doc mutating write)

PP12-110 lessons encoded here:
  - MagicMock(spec=HoudiniBridge) -- a non-existent attr (e.g. .call) raises
    AttributeError.  Bare MagicMock() is BANNED (masks convention bugs).
  - Import subject INSIDE each test, AFTER _get_bridge is patched.
  - Assert PUBLIC behavior (bridge.execute cmd+params; verbatim result
    passthrough), NOT call order / internal call count beyond "exactly one".
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
def mtlx_inspect_bridge_mock():
    """Spec-bound bridge mock returning a successful mtlx_inspect result shape
    (the MtlxSummary.to_dict() shape, per usd_export_model.py)."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "ok": True,
        "nodegraphs": ["NG_main"],
        "surface_nodes": ["standard_surface1"],
        "inputs_with_abs_paths": ["basecolor_file"],
        "validate": {"ok": True, "errors": []},
    })
    return mock


@pytest.fixture()
def mtlx_edit_bridge_mock():
    """Spec-bound bridge mock returning a successful gated mtlx_edit result shape."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "ok": True,
        "out_path": "$HIP/edited.mtlx",
        "edits_applied": 1,
        "validate": {"ok": True, "errors": []},
    })
    return mock


@pytest.fixture()
def mtlx_edit_pending_approval_bridge_mock():
    """Spec-bound bridge mock returning a 109-gate pending-approval shape.

    A pending/preview response is NOT a failure -- the wrapper must pass it
    through verbatim, not reinterpret it as an error (mirrors the
    usd_export_layer / usd_export_rop regression guard).
    """
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "status": "pending_approval",
        "pending_id": "pend-mtlx-edit-xyz",
        "preview": {
            "out_path": "$HIP/edited.mtlx",
            "source": "$HIP/source.mtlx",
            "source_parseable": True,
            "edits": [{"node": "standard_surface1", "input": "base_color", "value": "0.5"}],
            "edits_shape_ok": True,
            "edits_preview": [{"node": "standard_surface1", "input": "base_color",
                                "value": "0.5", "exists": True, "ambiguous": False,
                                "current_value": "1.0"}],
            "pre_validation": {"ok": True, "errors": []},
            "no_regex": True,
        },
    })
    return mock


# ---------------------------------------------------------------------------
# MIC-1 / MEC-1: module import + callable (PRIMARY RED GATE)
# ---------------------------------------------------------------------------

class TestMtlxToolsModuleImport:
    """MIC-1/MEC-1: both wrappers must be callables on usd_export_tools.

    The wrappers are added to the EXISTING usd_export_tools.py module (which
    already ships houdini_usd_inspect_layer, houdini_usd_validate,
    houdini_usd_export_layer, houdini_usd_export_rop). Until hou-dev adds
    houdini_mtlx_inspect / houdini_mtlx_edit, these imports raise
    ImportError -- that is the RED signal for this file.
    """

    def test_module_importable(self):
        """usd_export_tools must remain importable (pp12-112b/c/d baseline)."""
        import fxhoudinimcp.tools.usd_export_tools  # noqa: F401

    def test_existing_export_rop_wrapper_unaffected(self):
        """houdini_usd_export_rop (pp12-112d) must still be importable -- this
        new pair of wrappers must not clobber the existing gated sibling."""
        from fxhoudinimcp.tools.usd_export_tools import houdini_usd_export_rop  # noqa: F401
        assert callable(houdini_usd_export_rop)

    def test_mtlx_inspect_callable_on_module(self):
        """houdini_mtlx_inspect must be a callable (the @mcp.tool coroutine).

        FAILS RED until hou-dev adds the wrapper.
        """
        from fxhoudinimcp.tools.usd_export_tools import houdini_mtlx_inspect  # noqa: F401
        assert callable(houdini_mtlx_inspect), (
            "houdini_mtlx_inspect must be a callable (the @mcp.tool coroutine)."
        )

    def test_mtlx_edit_callable_on_module(self):
        """houdini_mtlx_edit must be a callable (the @mcp.tool coroutine).

        FAILS RED until hou-dev adds the wrapper.
        """
        from fxhoudinimcp.tools.usd_export_tools import houdini_mtlx_edit  # noqa: F401
        assert callable(houdini_mtlx_edit), (
            "houdini_mtlx_edit must be a callable (the @mcp.tool coroutine)."
        )


# ---------------------------------------------------------------------------
# MIC-2 / MEC-2: meta require_approval -- inspect False (UNGATED READONLY),
#                edit True (GATED MUTATING)
# ---------------------------------------------------------------------------

class TestMtlxToolsRequireApproval:
    """MIC-2/MEC-2: houdini_mtlx_inspect must be UNGATED (require_approval=
    False, matching houdini_usd_inspect_layer's READONLY convention exactly);
    houdini_mtlx_edit must be GATED (require_approval=True, matching every
    shipped mutating wrapper: houdini_usd_export_layer, houdini_usd_export_rop).
    """

    def _tool_map(self):
        import fxhoudinimcp.tools.usd_export_tools  # noqa: F401
        from fxhoudinimcp.server import mcp

        tool_map: dict = {}
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_map = mcp._tool_manager._tools
        elif hasattr(mcp, "_tools"):
            tool_map = mcp._tools
        return tool_map

    def test_mtlx_inspect_require_approval_false(self):
        """@mcp.tool(meta={'require_approval': False}) on houdini_mtlx_inspect."""
        tool_map = self._tool_map()
        assert "houdini_mtlx_inspect" in tool_map, (
            "houdini_mtlx_inspect not registered on the mcp server; "
            "hou-dev must define it (decorated with @mcp.tool) in usd_export_tools.py."
        )
        tool_obj = tool_map["houdini_mtlx_inspect"]
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", True) if isinstance(meta, dict) else True

        assert require_approval is False, (
            f"houdini_mtlx_inspect meta must have require_approval=False (UNGATED -- "
            f"a READONLY tool per FR-10). Got meta={meta!r}."
        )

    def test_mtlx_edit_require_approval_true(self):
        """@mcp.tool(meta={'require_approval': True}) on houdini_mtlx_edit."""
        tool_map = self._tool_map()
        assert "houdini_mtlx_edit" in tool_map, (
            "houdini_mtlx_edit not registered on the mcp server; "
            "hou-dev must define it (decorated with @mcp.tool) in usd_export_tools.py."
        )
        tool_obj = tool_map["houdini_mtlx_edit"]
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", False) if isinstance(meta, dict) else False

        assert require_approval is True, (
            f"houdini_mtlx_edit meta must have require_approval=True (GATED -- "
            f"a mutating MaterialX-doc write; PP12-109 security gate). "
            f"Got meta={meta!r}."
        )


# ---------------------------------------------------------------------------
# MIC-3: houdini_mtlx_inspect bridge.execute call contract
# ---------------------------------------------------------------------------

class TestMtlxInspectBridgeContract:
    """MIC-3: houdini_mtlx_inspect must delegate to bridge.execute with the
    EXACT single-call contract from the rev3 lockedFieldContract.

    Locked field contract (plan rev3):
        houdini_mtlx_inspect(ctx, mtlx_path_or_doc)
        -> a SINGLE bridge.execute('mtlx_inspect', {'mtlx_path_or_doc': ...})
    """

    @pytest.mark.asyncio
    async def test_exactly_one_bridge_execute_call(self, mtlx_inspect_bridge_mock):
        """The wrapper must call bridge.execute EXACTLY ONCE (no multi-call wrapper)."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=mtlx_inspect_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_mtlx_inspect

            ctx_mock = MagicMock()
            await houdini_mtlx_inspect(ctx=ctx_mock, mtlx_path_or_doc="$HIP/material.mtlx")

        mtlx_inspect_bridge_mock.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_command_string_is_mtlx_inspect(self, mtlx_inspect_bridge_mock):
        """bridge.execute must be called with command='mtlx_inspect' (bare, not lops.*)."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=mtlx_inspect_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_mtlx_inspect

            ctx_mock = MagicMock()
            await houdini_mtlx_inspect(ctx=ctx_mock, mtlx_path_or_doc="$HIP/material.mtlx")

        call_args = mtlx_inspect_bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "mtlx_inspect", (
            f"Expected bridge.execute('mtlx_inspect', ...) but got command={command!r}. "
            "Command string must match register_handler's first arg exactly "
            "(the 4-bug convention: command == register name == params keys == handler kwargs)."
        )

    @pytest.mark.asyncio
    async def test_params_dict_has_exact_one_key(self, mtlx_inspect_bridge_mock):
        """params dict must be EXACTLY {mtlx_path_or_doc}."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=mtlx_inspect_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_mtlx_inspect

            ctx_mock = MagicMock()
            await houdini_mtlx_inspect(ctx=ctx_mock, mtlx_path_or_doc="$HIP/material.mtlx")

        call_args = mtlx_inspect_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        expected_keys = {"mtlx_path_or_doc"}
        assert set(params.keys()) == expected_keys, (
            f"params dict must be exactly {expected_keys!r}. Got keys={set(params.keys())!r}."
        )
        assert params["mtlx_path_or_doc"] == "$HIP/material.mtlx", (
            f"mtlx_path_or_doc must be forwarded verbatim. Got {params['mtlx_path_or_doc']!r}."
        )

    @pytest.mark.asyncio
    async def test_result_passed_through_verbatim(self, mtlx_inspect_bridge_mock):
        """The wrapper must return bridge.execute's result VERBATIM (the
        MtlxSummary.to_dict() shape -- 'nodegraphs', 'surface_nodes',
        'inputs_with_abs_paths', 'validate': {'ok', 'errors'})."""
        expected_result = mtlx_inspect_bridge_mock.execute.return_value
        with patch("fxhoudinimcp.server._get_bridge", return_value=mtlx_inspect_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_mtlx_inspect

            ctx_mock = MagicMock()
            result = await houdini_mtlx_inspect(ctx=ctx_mock, mtlx_path_or_doc="$HIP/material.mtlx")

        assert result == expected_result, (
            "Wrapper must pass bridge.execute's result through unchanged. "
            f"Expected {expected_result!r}, got {result!r}."
        )


# ---------------------------------------------------------------------------
# MEC-3: houdini_mtlx_edit bridge.execute call contract
# ---------------------------------------------------------------------------

class TestMtlxEditBridgeContract:
    """MEC-3: houdini_mtlx_edit must delegate to bridge.execute with the
    EXACT single-call contract from the rev3 lockedFieldContract.

    Locked field contract (plan rev3):
        houdini_mtlx_edit(ctx, mtlx_path, edits, out_path)
        -> a SINGLE bridge.execute('mtlx_edit', {
               'mtlx_path': mtlx_path, 'out_path': out_path, 'edits': edits,
           })
    """

    @pytest.mark.asyncio
    async def test_exactly_one_bridge_execute_call(self, mtlx_edit_bridge_mock):
        """The wrapper must call bridge.execute EXACTLY ONCE (no multi-call wrapper)."""
        edits = [{"node": "standard_surface1", "input": "base_color", "value": "0.5"}]
        with patch("fxhoudinimcp.server._get_bridge", return_value=mtlx_edit_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_mtlx_edit

            ctx_mock = MagicMock()
            await houdini_mtlx_edit(
                ctx=ctx_mock,
                mtlx_path="$HIP/source.mtlx",
                edits=edits,
                out_path="$HIP/edited.mtlx",
            )

        mtlx_edit_bridge_mock.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_command_string_is_mtlx_edit(self, mtlx_edit_bridge_mock):
        """bridge.execute must be called with command='mtlx_edit' (bare, not lops.*)."""
        edits = [{"node": "standard_surface1", "input": "base_color", "value": "0.5"}]
        with patch("fxhoudinimcp.server._get_bridge", return_value=mtlx_edit_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_mtlx_edit

            ctx_mock = MagicMock()
            await houdini_mtlx_edit(
                ctx=ctx_mock,
                mtlx_path="$HIP/source.mtlx",
                edits=edits,
                out_path="$HIP/edited.mtlx",
            )

        call_args = mtlx_edit_bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "mtlx_edit", (
            f"Expected bridge.execute('mtlx_edit', ...) but got command={command!r}. "
            "Command string must match register_handler's first arg exactly "
            "(the 4-bug convention: command == register name == params keys == handler kwargs)."
        )

    @pytest.mark.asyncio
    async def test_params_dict_has_exact_three_keys(self, mtlx_edit_bridge_mock):
        """params dict must be EXACTLY {mtlx_path, out_path, edits}."""
        edits = [{"node": "standard_surface1", "input": "base_color", "value": "0.5"}]
        with patch("fxhoudinimcp.server._get_bridge", return_value=mtlx_edit_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_mtlx_edit

            ctx_mock = MagicMock()
            await houdini_mtlx_edit(
                ctx=ctx_mock,
                mtlx_path="$HIP/source.mtlx",
                edits=edits,
                out_path="$HIP/edited.mtlx",
            )

        call_args = mtlx_edit_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        expected_keys = {"mtlx_path", "out_path", "edits"}
        assert set(params.keys()) == expected_keys, (
            f"params dict must be exactly {expected_keys!r}. Got keys={set(params.keys())!r}."
        )
        assert params["mtlx_path"] == "$HIP/source.mtlx", (
            f"mtlx_path must be forwarded verbatim. Got {params['mtlx_path']!r}."
        )
        assert params["out_path"] == "$HIP/edited.mtlx", (
            f"out_path must be forwarded verbatim. Got {params['out_path']!r}."
        )
        assert params["edits"] == edits, (
            f"edits must be forwarded verbatim (no wrapper-side transformation). "
            f"Got {params['edits']!r}."
        )


# ---------------------------------------------------------------------------
# MEC-4: verbatim passthrough -- INCLUDING a pending-approval / preview shape
# ---------------------------------------------------------------------------

class TestMtlxEditResultPassthrough:
    """MEC-4: the wrapper must return bridge.execute's result VERBATIM.

    Critically this includes the 109-gate pending-approval / preview response
    shape -- the wrapper is a thin single bridge.execute call with NO result
    interpretation. A wrapper that checks `if not result.get("ok"): raise`
    would incorrectly treat a pending-approval response (which has no "ok"
    key at all) as a failure.
    """

    @pytest.mark.asyncio
    async def test_success_result_passed_through_verbatim(self, mtlx_edit_bridge_mock):
        """On a normal (already-approved / trusted-gate) success, result is unchanged."""
        edits = [{"node": "standard_surface1", "input": "base_color", "value": "0.5"}]
        expected_result = mtlx_edit_bridge_mock.execute.return_value
        with patch("fxhoudinimcp.server._get_bridge", return_value=mtlx_edit_bridge_mock):
            from fxhoudinimcp.tools.usd_export_tools import houdini_mtlx_edit

            ctx_mock = MagicMock()
            result = await houdini_mtlx_edit(
                ctx=ctx_mock,
                mtlx_path="$HIP/source.mtlx",
                edits=edits,
                out_path="$HIP/edited.mtlx",
            )

        assert result == expected_result, (
            "Wrapper must pass bridge.execute's result through unchanged. "
            f"Expected {expected_result!r}, got {result!r}."
        )

    @pytest.mark.asyncio
    async def test_pending_approval_result_passed_through_not_treated_as_failure(
        self, mtlx_edit_pending_approval_bridge_mock
    ):
        """A pending-approval/preview shape (the 109 gate) is NOT an error --
        the wrapper must return it verbatim, unmodified, un-raised.
        """
        edits = [{"node": "standard_surface1", "input": "base_color", "value": "0.5"}]
        expected_result = mtlx_edit_pending_approval_bridge_mock.execute.return_value
        with patch(
            "fxhoudinimcp.server._get_bridge",
            return_value=mtlx_edit_pending_approval_bridge_mock,
        ):
            from fxhoudinimcp.tools.usd_export_tools import houdini_mtlx_edit

            ctx_mock = MagicMock()
            # Must NOT raise -- a pending-approval response is a normal, valid
            # return value from bridge.execute, not an exception-worthy state.
            result = await houdini_mtlx_edit(
                ctx=ctx_mock,
                mtlx_path="$HIP/source.mtlx",
                edits=edits,
                out_path="$HIP/edited.mtlx",
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
# MIC-4/MEC-5: spec-bound bridge mock -- PP12-110 regression guard
# ---------------------------------------------------------------------------

class TestBridgeSpecBoundGuard:
    """MIC-4/MEC-5: MagicMock(spec=HoudiniBridge) -- .call raises AttributeError.

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
# MIC-5/MEC-6: ctx not in tool input schema (ctx is injected, not a tool parameter)
# ---------------------------------------------------------------------------

class TestCtxSchemaGuard:
    """MIC-5/MEC-6: 'ctx' must NOT appear in either tool's input schema.

    FastMCP injects ctx via the Context type annotation -- it is NOT a
    parameter the MCP client sends. Mirrors the pp12-112c/d ctx-schema guard.
    """

    def _tool_map(self):
        import fxhoudinimcp.tools.usd_export_tools  # noqa: F401
        from fxhoudinimcp.server import mcp

        tool_map: dict = {}
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            tool_map = mcp._tool_manager._tools
        elif hasattr(mcp, "_tools"):
            tool_map = mcp._tools
        return tool_map

    def test_mtlx_inspect_ctx_not_in_schema(self):
        """houdini_mtlx_inspect: 'ctx' must not be a property in the input schema."""
        tool_map = self._tool_map()
        tool_obj = tool_map.get("houdini_mtlx_inspect")
        assert tool_obj is not None, "houdini_mtlx_inspect not registered on the mcp server."

        schema = getattr(tool_obj, "parameters", getattr(tool_obj, "inputSchema", {})) or {}
        properties = schema.get("properties", {})
        assert "ctx" not in properties, (
            f"'ctx' must NOT appear in houdini_mtlx_inspect's input schema properties. "
            f"FastMCP injects it; clients don't provide it. Got properties={list(properties.keys())!r}."
        )

    def test_mtlx_edit_ctx_not_in_schema(self):
        """houdini_mtlx_edit: 'ctx' must not be a property in the input schema."""
        tool_map = self._tool_map()
        tool_obj = tool_map.get("houdini_mtlx_edit")
        assert tool_obj is not None, "houdini_mtlx_edit not registered on the mcp server."

        schema = getattr(tool_obj, "parameters", getattr(tool_obj, "inputSchema", {})) or {}
        properties = schema.get("properties", {})
        assert "ctx" not in properties, (
            f"'ctx' must NOT appear in houdini_mtlx_edit's input schema properties. "
            f"FastMCP injects it; clients don't provide it. Got properties={list(properties.keys())!r}."
        )
