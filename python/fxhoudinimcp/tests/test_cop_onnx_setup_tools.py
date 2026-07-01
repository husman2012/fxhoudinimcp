"""pytest wrapper tests for houdini_cop_onnx_setup_node and
houdini_cop_onnx_set_provider MCP tools (PP12-113 PR-3, GATED).

Unit: pp12-113c
testVerificationSurface: pytest-model
planSha: 93096c3c0443cab7842583f6ba15abdbd56e3da289b87f6f11d02c2e9da0fa8b

These tests are written BEFORE the implementation (red phase). They will
fail with ImportError until hou-dev implements the two new wrappers on
fxhoudinimcp/tools/cop_onnx_tools.py (which already ships
houdini_cop_onnx_list_models / houdini_cop_onnx_inspect_model from PR-2)
plus the corresponding handlers on
fxhoudinimcp_server/handlers/cop_onnx_handlers.py.

Grounded against (layer-for-layer template, per plan pp12-113c
reuseSurvey):
  - python/fxhoudinimcp/tests/test_usd_export_rop_tool.py (pp12-112d) —
    THE exemplar for a GATED wrapper test: MagicMock(spec=HoudiniBridge);
    import-inside-test-after-patch; require_approval=True assertion via
    the mcp tool_map; exactly-one bridge.execute call; exact params-dict
    key set; VERBATIM passthrough of BOTH a success result AND a
    pending-approval/preview shape (never reinterpreted, never raises,
    never gains/loses an 'ok' key); ctx-not-in-schema guard;
    MagicMock(spec=HoudiniBridge) .call-raises-AttributeError PP12-110
    regression guard.
  - python/fxhoudinimcp/tests/test_cop_onnx_tools.py (pp12-113b) — the
    cop_onnx-family bare-command-string convention ('cop_onnx_*', not a
    'cops.*' prefix).
  - houdini/scripts/python/fxhoudinimcp_server/handlers/usd_export_handlers.py
    usd_export_rop + _preview_export_rop — the GATED handler-side shape
    this file's dispatcher-level contract mirrors (not exercised
    directly here; hou-dev's hython-smoke covers the handler).

PP12-110 lessons encoded here (mcp-subprocess-delegation.md /
mcp-fork-build-lessons memory):
  - MagicMock(spec=HoudiniBridge) — a non-existent attr (e.g. .call) raises
    AttributeError; a bare MagicMock() would silently accept it and mask
    the bug.
  - Import the subject module INSIDE each test, AFTER _get_bridge is
    patched (module-level import would bind the tool before the patch is
    active).
  - Assert PUBLIC behavior (bridge.execute cmd + params; verbatim result
    passthrough, INCLUDING a pending-approval shape) — NOT internal call
    order or unrelated attributes.
  - @pytest.mark.asyncio on every async test (both wrappers are async
    @mcp.tool coroutines).
  - Both wrappers make EXACTLY ONE bridge.execute call — no retries, no
    secondary calls, no result-shape interpretation.
  - Both must be require_approval=True (GATED — the first two mutating
    tools of the cop_onnx family; PP12-109 security gate). A
    pending-approval bridge response is NOT a failure and must survive
    the wrapper untouched.

Locked field contract (plan pp12-113c lockedFieldContract, "wrappers:
houdini_cop_onnx_setup_node / houdini_cop_onnx_set_provider"):

    houdini_cop_onnx_setup_node(
        ctx, parent_path, model_path, node_name='agent_onnx',
        setup_shapes=True, flip_input=None, flip_output=None,
    ) -> dict
        -> a SINGLE bridge.execute('cop_onnx_setup_node', {
               'parent_path': parent_path, 'model_path': model_path,
               'node_name': node_name, 'setup_shapes': setup_shapes,
               'flip_input': flip_input, 'flip_output': flip_output,
           })

    houdini_cop_onnx_set_provider(ctx, node_path, provider) -> dict
        -> a SINGLE bridge.execute('cop_onnx_set_provider', {
               'node_path': node_path, 'provider': provider,
           })

Command strings are BARE ('cop_onnx_setup_node' / 'cop_onnx_set_provider')
— matching the PR-2 cop_onnx family convention (not a 'cops.*' prefix).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import REAL HoudiniBridge for spec= (PP12-110 lesson: bare MagicMock()
# silently accepts .call and any other non-existent attribute, masking
# convention bugs).
from fxhoudinimcp.bridge import HoudiniBridge


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def setup_node_bridge_mock():
    """Spec-bound bridge mock returning a successful gated setup_node result."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "ok": True,
        "node_path": "/obj/copnet1/agent_onnx",
        "model_path": "/models/identity.onnx",
        "input_tensors": [
            {
                "name": "input",
                "shape": [1, 3, "dynamic", "dynamic"],
                "dtype": "float32",
                "cop_input_index": 1,
            },
        ],
        "output_tensors": [
            {
                "name": "output",
                "shape": [1, 3, "dynamic", "dynamic"],
                "dtype": "float32",
                "cop_plane": "n_output",
            },
        ],
        "warnings": [],
        "applied": True,
    })
    return mock


@pytest.fixture()
def setup_node_pending_approval_bridge_mock():
    """Spec-bound bridge mock returning a 109-gate pending-approval shape.

    A pending/preview response is NOT a failure — the wrapper must pass it
    through verbatim, not reinterpret it as an error (mirrors the
    pp12-112d URC-4 regression guard).
    """
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "status": "pending_approval",
        "pending_id": "pend-setup-node-abc123",
        "preview": {
            "action": "create cop/onnx node",
            "parent_path": "/obj/copnet1",
            "node_name": "agent_onnx",
            "model_path": "/models/identity.onnx",
            "setup_shapes": True,
            "flip_input": None,
            "flip_output": None,
            "node_will_persist": True,
            "parent_exists": True,
            "parent_is_cop_net": True,
        },
    })
    return mock


@pytest.fixture()
def set_provider_bridge_mock():
    """Spec-bound bridge mock returning a successful gated set_provider result."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "ok": True,
        "node_path": "/obj/copnet1/agent_onnx",
        "requested": "cuda",
        "available_providers": ["automatic", "cpu", "cuda", "directml"],
        "will_bind": "cuda",
        "warnings": [],
    })
    return mock


@pytest.fixture()
def set_provider_pending_approval_bridge_mock():
    """Spec-bound bridge mock returning a 109-gate pending-approval shape
    for cop_onnx_set_provider."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "status": "pending_approval",
        "pending_id": "pend-set-provider-def456",
        "preview": {
            "action": "set Execution Provider",
            "node_path": "/obj/copnet1/agent_onnx",
            "requested": "cuda",
            "available_providers": ["automatic", "cpu", "cuda", "directml"],
            "will_bind": "cuda",
            "node_exists": True,
            "node_is_onnx": True,
        },
    })
    return mock


# ---------------------------------------------------------------------------
# module import + callable (PRIMARY RED GATE)
# ---------------------------------------------------------------------------

class TestCopOnnxSetupToolsModuleImport:
    """Both new wrappers must be callables on the EXISTING cop_onnx_tools
    module (which already ships houdini_cop_onnx_list_models /
    houdini_cop_onnx_inspect_model from PR-2). Until hou-dev adds them,
    these imports raise ImportError — the RED signal for this file.
    """

    def test_module_importable(self):
        """cop_onnx_tools must remain importable (pp12-113b baseline)."""
        import fxhoudinimcp.tools.cop_onnx_tools  # noqa: F401

    def test_existing_pr2_wrappers_unaffected(self):
        """houdini_cop_onnx_list_models / houdini_cop_onnx_inspect_model
        (PR-2) must still be importable — the new wrappers must not
        clobber their existing siblings (append-only contract)."""
        from fxhoudinimcp.tools.cop_onnx_tools import (  # noqa: F401
            houdini_cop_onnx_inspect_model,
            houdini_cop_onnx_list_models,
        )
        assert callable(houdini_cop_onnx_list_models)
        assert callable(houdini_cop_onnx_inspect_model)

    def test_setup_node_callable_on_module(self):
        """houdini_cop_onnx_setup_node must be a callable (the @mcp.tool
        coroutine). FAILS RED until hou-dev adds the wrapper."""
        from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_setup_node  # noqa: F401
        assert callable(houdini_cop_onnx_setup_node), (
            "houdini_cop_onnx_setup_node must be a callable (the @mcp.tool coroutine)."
        )

    def test_set_provider_callable_on_module(self):
        """houdini_cop_onnx_set_provider must be a callable (the @mcp.tool
        coroutine). FAILS RED until hou-dev adds the wrapper."""
        from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_set_provider  # noqa: F401
        assert callable(houdini_cop_onnx_set_provider), (
            "houdini_cop_onnx_set_provider must be a callable (the @mcp.tool coroutine)."
        )


# ---------------------------------------------------------------------------
# GATED — meta require_approval=True on BOTH new wrappers
# ---------------------------------------------------------------------------

def _get_tool_map():
    from fxhoudinimcp.server import mcp
    tool_map: dict = {}
    if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
        tool_map = mcp._tool_manager._tools
    elif hasattr(mcp, "_tools"):
        tool_map = mcp._tools
    return tool_map


class TestCopOnnxSetupNodeRequireApproval:
    """houdini_cop_onnx_setup_node must have meta={'require_approval': True}.

    This is the FIRST mutating tool of the cop_onnx family — it creates a
    PERSISTENT cop/onnx node under a caller-given COP-network parent.
    Every shipped gated wrapper (houdini_usd_export_layer,
    houdini_usd_export_rop, ...) sets require_approval=True; this tool
    must match that convention exactly.
    """

    def test_mcp_tool_meta_require_approval_true(self):
        import fxhoudinimcp.tools.cop_onnx_tools  # noqa: F401
        tool_map = _get_tool_map()
        assert "houdini_cop_onnx_setup_node" in tool_map, (
            "houdini_cop_onnx_setup_node not registered on the mcp server; "
            "hou-dev must define it (decorated with @mcp.tool) in cop_onnx_tools.py."
        )
        tool_obj = tool_map["houdini_cop_onnx_setup_node"]
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", False) if isinstance(meta, dict) else False
        assert require_approval is True, (
            f"houdini_cop_onnx_setup_node meta must have require_approval=True (GATED -- "
            f"a mutating tool; PP12-109 security gate). Got meta={meta!r}."
        )


class TestCopOnnxSetProviderRequireApproval:
    """houdini_cop_onnx_set_provider must have meta={'require_approval': True}.

    This is the SECOND mutating tool of the cop_onnx family — it sets the
    node's Execution Provider parm.
    """

    def test_mcp_tool_meta_require_approval_true(self):
        import fxhoudinimcp.tools.cop_onnx_tools  # noqa: F401
        tool_map = _get_tool_map()
        assert "houdini_cop_onnx_set_provider" in tool_map, (
            "houdini_cop_onnx_set_provider not registered on the mcp server; "
            "hou-dev must define it (decorated with @mcp.tool) in cop_onnx_tools.py."
        )
        tool_obj = tool_map["houdini_cop_onnx_set_provider"]
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", False) if isinstance(meta, dict) else False
        assert require_approval is True, (
            f"houdini_cop_onnx_set_provider meta must have require_approval=True (GATED -- "
            f"a mutating tool; PP12-109 security gate). Got meta={meta!r}."
        )


# ---------------------------------------------------------------------------
# bridge.execute call contract — houdini_cop_onnx_setup_node
# ---------------------------------------------------------------------------

class TestCopOnnxSetupNodeBridgeContract:
    """houdini_cop_onnx_setup_node must delegate to bridge.execute EXACTLY
    ONCE, with command 'cop_onnx_setup_node' and the exact 6-key params
    dict from the lockedFieldContract."""

    @pytest.mark.asyncio
    async def test_exactly_one_bridge_execute_call(self, setup_node_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=setup_node_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_setup_node

            ctx_mock = MagicMock()
            await houdini_cop_onnx_setup_node(
                ctx=ctx_mock,
                parent_path="/obj/copnet1",
                model_path="/models/identity.onnx",
            )

        assert setup_node_bridge_mock.execute.call_count == 1, (
            f"houdini_cop_onnx_setup_node must make exactly ONE bridge.execute call, "
            f"got {setup_node_bridge_mock.execute.call_count}."
        )

    @pytest.mark.asyncio
    async def test_command_string_is_cop_onnx_setup_node(self, setup_node_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=setup_node_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_setup_node

            ctx_mock = MagicMock()
            await houdini_cop_onnx_setup_node(
                ctx=ctx_mock,
                parent_path="/obj/copnet1",
                model_path="/models/identity.onnx",
            )

        call_args = setup_node_bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "cop_onnx_setup_node", (
            f"Expected bridge.execute('cop_onnx_setup_node', ...) but got command={command!r}. "
            "Command string must match register_handler's first arg exactly "
            "(the 4-bug convention: command == register name == params keys == handler kwargs)."
        )

    @pytest.mark.asyncio
    async def test_params_dict_has_exact_six_keys(self, setup_node_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=setup_node_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_setup_node

            ctx_mock = MagicMock()
            await houdini_cop_onnx_setup_node(
                ctx=ctx_mock,
                parent_path="/obj/copnet1",
                model_path="/models/identity.onnx",
                node_name="my_onnx",
                setup_shapes=False,
                flip_input=True,
                flip_output=False,
            )

        call_args = setup_node_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        expected_keys = {
            "parent_path", "model_path", "node_name",
            "setup_shapes", "flip_input", "flip_output",
        }
        assert set(params.keys()) == expected_keys, (
            f"params dict must be exactly {expected_keys!r}. Got keys={set(params.keys())!r}."
        )
        assert params["parent_path"] == "/obj/copnet1"
        assert params["model_path"] == "/models/identity.onnx"
        assert params["node_name"] == "my_onnx"
        assert params["setup_shapes"] is False
        assert params["flip_input"] is True
        assert params["flip_output"] is False

    @pytest.mark.asyncio
    async def test_defaults_forwarded_when_omitted(self, setup_node_bridge_mock):
        """node_name='agent_onnx', setup_shapes=True, flip_input=None,
        flip_output=None are the documented defaults — they must be
        forwarded verbatim (not omitted) when the caller doesn't pass them."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=setup_node_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_setup_node

            ctx_mock = MagicMock()
            await houdini_cop_onnx_setup_node(
                ctx=ctx_mock,
                parent_path="/obj/copnet1",
                model_path="/models/identity.onnx",
            )

        call_args = setup_node_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert params.get("node_name") == "agent_onnx", (
            f"node_name must default to 'agent_onnx'. Got {params.get('node_name')!r}."
        )
        assert params.get("setup_shapes") is True, (
            f"setup_shapes must default to True. Got {params.get('setup_shapes')!r}."
        )
        assert params.get("flip_input") is None, (
            f"flip_input must default to None. Got {params.get('flip_input')!r}."
        )
        assert params.get("flip_output") is None, (
            f"flip_output must default to None. Got {params.get('flip_output')!r}."
        )


# ---------------------------------------------------------------------------
# bridge.execute call contract — houdini_cop_onnx_set_provider
# ---------------------------------------------------------------------------

class TestCopOnnxSetProviderBridgeContract:
    """houdini_cop_onnx_set_provider must delegate to bridge.execute
    EXACTLY ONCE, with command 'cop_onnx_set_provider' and the exact
    2-key params dict from the lockedFieldContract."""

    @pytest.mark.asyncio
    async def test_exactly_one_bridge_execute_call(self, set_provider_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=set_provider_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_set_provider

            ctx_mock = MagicMock()
            await houdini_cop_onnx_set_provider(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
                provider="cuda",
            )

        assert set_provider_bridge_mock.execute.call_count == 1, (
            f"houdini_cop_onnx_set_provider must make exactly ONE bridge.execute call, "
            f"got {set_provider_bridge_mock.execute.call_count}."
        )

    @pytest.mark.asyncio
    async def test_command_string_is_cop_onnx_set_provider(self, set_provider_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=set_provider_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_set_provider

            ctx_mock = MagicMock()
            await houdini_cop_onnx_set_provider(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
                provider="cuda",
            )

        call_args = set_provider_bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "cop_onnx_set_provider", (
            f"Expected bridge.execute('cop_onnx_set_provider', ...) but got command={command!r}."
        )

    @pytest.mark.asyncio
    async def test_params_dict_has_exact_two_keys(self, set_provider_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=set_provider_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_set_provider

            ctx_mock = MagicMock()
            await houdini_cop_onnx_set_provider(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
                provider="cuda",
            )

        call_args = set_provider_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        expected_keys = {"node_path", "provider"}
        assert set(params.keys()) == expected_keys, (
            f"params dict must be exactly {expected_keys!r}. Got keys={set(params.keys())!r}."
        )
        assert params["node_path"] == "/obj/copnet1/agent_onnx"
        assert params["provider"] == "cuda"

    @pytest.mark.asyncio
    async def test_provider_forwarded_verbatim_even_when_unavailable(self, set_provider_bridge_mock):
        """An unavailable provider request (e.g. 'tensorrt') must still be
        forwarded VERBATIM to the handler — the wrapper does NOT validate
        or pre-filter the provider string; that is the handler's job
        (never-error-on-unavailable is a HANDLER contract, not a wrapper
        short-circuit)."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=set_provider_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_set_provider

            ctx_mock = MagicMock()
            await houdini_cop_onnx_set_provider(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
                provider="tensorrt",
            )

        call_args = set_provider_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert params["provider"] == "tensorrt", (
            f"provider must be forwarded verbatim regardless of availability. "
            f"Got {params['provider']!r}."
        )


# ---------------------------------------------------------------------------
# verbatim passthrough — INCLUDING a pending-approval / preview shape
# ---------------------------------------------------------------------------

class TestCopOnnxSetupNodeResultPassthrough:
    """The wrapper must return bridge.execute's result VERBATIM. Critically
    this includes the 109-gate pending-approval / preview response shape —
    the wrapper is a thin single bridge.execute call with NO result
    interpretation. A wrapper that checks `if not result.get("ok"): raise`
    would incorrectly treat a pending-approval response (which has no "ok"
    key at all) as a failure."""

    @pytest.mark.asyncio
    async def test_success_result_passed_through_verbatim(self, setup_node_bridge_mock):
        expected_result = setup_node_bridge_mock.execute.return_value
        with patch("fxhoudinimcp.server._get_bridge", return_value=setup_node_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_setup_node

            ctx_mock = MagicMock()
            result = await houdini_cop_onnx_setup_node(
                ctx=ctx_mock,
                parent_path="/obj/copnet1",
                model_path="/models/identity.onnx",
            )

        assert result == expected_result, (
            "Wrapper must pass bridge.execute's result through unchanged. "
            f"Expected {expected_result!r}, got {result!r}."
        )

    @pytest.mark.asyncio
    async def test_pending_approval_result_passed_through_not_treated_as_failure(
        self, setup_node_pending_approval_bridge_mock
    ):
        expected_result = setup_node_pending_approval_bridge_mock.execute.return_value
        with patch(
            "fxhoudinimcp.server._get_bridge",
            return_value=setup_node_pending_approval_bridge_mock,
        ):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_setup_node

            ctx_mock = MagicMock()
            # Must NOT raise -- a pending-approval response is a normal,
            # valid return value from bridge.execute.
            result = await houdini_cop_onnx_setup_node(
                ctx=ctx_mock,
                parent_path="/obj/copnet1",
                model_path="/models/identity.onnx",
            )

        assert result == expected_result, (
            "A pending-approval bridge response must be returned VERBATIM -- "
            "the wrapper must not reinterpret, unwrap, or swallow it. "
            f"Expected {expected_result!r}, got {result!r}."
        )
        assert result.get("status") == "pending_approval"
        assert "ok" not in result, (
            f"Wrapper must not inject an 'ok' key into a pending-approval response. "
            f"Got keys={list(result.keys())!r}."
        )


class TestCopOnnxSetProviderResultPassthrough:
    """Same verbatim-passthrough contract for houdini_cop_onnx_set_provider."""

    @pytest.mark.asyncio
    async def test_success_result_passed_through_verbatim(self, set_provider_bridge_mock):
        expected_result = set_provider_bridge_mock.execute.return_value
        with patch("fxhoudinimcp.server._get_bridge", return_value=set_provider_bridge_mock):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_set_provider

            ctx_mock = MagicMock()
            result = await houdini_cop_onnx_set_provider(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
                provider="cuda",
            )

        assert result == expected_result, (
            "Wrapper must pass bridge.execute's result through unchanged. "
            f"Expected {expected_result!r}, got {result!r}."
        )

    @pytest.mark.asyncio
    async def test_pending_approval_result_passed_through_not_treated_as_failure(
        self, set_provider_pending_approval_bridge_mock
    ):
        expected_result = set_provider_pending_approval_bridge_mock.execute.return_value
        with patch(
            "fxhoudinimcp.server._get_bridge",
            return_value=set_provider_pending_approval_bridge_mock,
        ):
            from fxhoudinimcp.tools.cop_onnx_tools import houdini_cop_onnx_set_provider

            ctx_mock = MagicMock()
            result = await houdini_cop_onnx_set_provider(
                ctx=ctx_mock,
                node_path="/obj/copnet1/agent_onnx",
                provider="cuda",
            )

        assert result == expected_result, (
            "A pending-approval bridge response must be returned VERBATIM -- "
            "the wrapper must not reinterpret, unwrap, or swallow it. "
            f"Expected {expected_result!r}, got {result!r}."
        )
        assert result.get("status") == "pending_approval"
        assert "ok" not in result, (
            f"Wrapper must not inject an 'ok' key into a pending-approval response. "
            f"Got keys={list(result.keys())!r}."
        )


# ---------------------------------------------------------------------------
# spec-bound bridge mock — PP12-110 regression guard
# ---------------------------------------------------------------------------

class TestBridgeSpecBoundGuard:
    """MagicMock(spec=HoudiniBridge) — .call raises AttributeError (PP12-110 guard).

    In PP12-110, a wrapper called bridge.call(...) instead of
    bridge.execute(...). A bare MagicMock() silently accepted .call and
    returned a mock — masking the bug. A spec-bound mock raises
    AttributeError because .call is not a real method on HoudiniBridge.
    """

    def test_bridge_call_attribute_does_not_exist(self):
        mock = MagicMock(spec=HoudiniBridge)
        with pytest.raises(AttributeError, match="call"):
            _ = mock.call  # .call must NOT exist on HoudiniBridge

    def test_bridge_execute_attribute_exists(self):
        mock = MagicMock(spec=HoudiniBridge)
        _ = mock.execute  # must NOT raise — execute IS a real method


# ---------------------------------------------------------------------------
# ctx not in tool input schema (ctx is injected, not a tool parameter)
# ---------------------------------------------------------------------------

class TestCtxSchemaGuard:
    """ctx must NOT appear in either new tool's input schema properties.

    FastMCP injects ctx via the Context type annotation — it is NOT a
    parameter the MCP client sends.
    """

    def _get_tool_schema(self, tool_name: str) -> dict:
        tool_map = _get_tool_map()
        tool_obj = tool_map.get(tool_name)
        if tool_obj is None:
            return {}
        return getattr(tool_obj, "parameters", getattr(tool_obj, "inputSchema", {})) or {}

    def test_setup_node_ctx_not_in_schema(self):
        import fxhoudinimcp.tools.cop_onnx_tools  # noqa: F401
        schema = self._get_tool_schema("houdini_cop_onnx_setup_node")
        properties = schema.get("properties", {})
        assert "ctx" not in properties, (
            f"'ctx' must NOT appear in houdini_cop_onnx_setup_node's input schema properties. "
            f"Got properties={list(properties.keys())!r}."
        )

    def test_set_provider_ctx_not_in_schema(self):
        import fxhoudinimcp.tools.cop_onnx_tools  # noqa: F401
        schema = self._get_tool_schema("houdini_cop_onnx_set_provider")
        properties = schema.get("properties", {})
        assert "ctx" not in properties, (
            f"'ctx' must NOT appear in houdini_cop_onnx_set_provider's input schema properties. "
            f"Got properties={list(properties.keys())!r}."
        )
