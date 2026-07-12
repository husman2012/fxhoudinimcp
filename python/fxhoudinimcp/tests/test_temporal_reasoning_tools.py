"""pytest wrapper tests for houdini_describe_sim_events and
houdini_assert_simulation MCP tools (PP12-117 PR-2, READONLY / UNGATED).

Unit: pp12-117b
testVerificationSurface: pytest-model
planSha: 596ba5d0454501373c5b30de5841e1da66f96e2342aa058fddfef5bb1b26eb2e

These tests are written BEFORE the implementation (red phase). They will
fail with ModuleNotFoundError until hou-dev creates
fxhoudinimcp/tools/temporal_reasoning_tools.py (the two MCP wrappers) plus
the corresponding handler module
fxhoudinimcp_server/handlers/temporal_reasoning_handlers.py (not exercised
directly by this file -- that is test_temporal_reasoning_handlers.py's job).

Grounded against (layer-for-layer template, per plan pp12-117b
reuseSurvey + lockedFieldContract REVISION 2):
  - test_spatial_reasoning_tools.py (pp12-116b) -- THE exemplar for a
    READONLY (require_approval=False) wrapper test: MagicMock(spec=
    HoudiniBridge); import-inside-test-after-patch; exactly-one
    bridge.execute call; exact command string + params-dict key set;
    verbatim passthrough of the bridge's result; ctx-not-in-schema guard;
    MagicMock(spec=HoudiniBridge) .call-raises-AttributeError PP12-110
    regression guard.
  - fxhoudinimcp_server/handlers/spatial_reasoning_handlers.py (the shipped
    READONLY handler-registration convention -- not exercised directly
    here).

PP12-110 lessons encoded here (mcp-subprocess-delegation.md /
mcp-fork-build-lessons memory):
  - MagicMock(spec=HoudiniBridge) -- a non-existent attr (e.g. .call) raises
    AttributeError; a bare MagicMock() would silently accept it and mask
    the bug.
  - Import the subject module INSIDE each test, AFTER _get_bridge is
    patched (module-level import would bind the tool before the patch is
    active).
  - Assert PUBLIC behavior (bridge.execute cmd + params; verbatim result
    passthrough) -- NOT internal call order or unrelated attributes.
  - @pytest.mark.asyncio on every async test (both wrappers are async
    @mcp.tool coroutines).
  - Both wrappers make EXACTLY ONE bridge.execute call -- no retries, no
    secondary calls.
  - Both must be require_approval=False (READONLY/ungated -- the
    reversible-frame-evaluation exception, Blocker-2 of the plan's
    lockedFieldContract: assert_simulation steps+restores frames but never
    writes state).

Locked field contract (plan pp12-117b lockedFieldContract, REVISION 2):
    houdini_describe_sim_events(ctx) -> dict
        -> bridge.execute('describe_sim_events', {})

    houdini_assert_simulation(ctx, network, frame_range, assertions,
                               cook_job=None) -> dict
        -> bridge.execute('assert_simulation', {
               'network': network, 'frame_range': frame_range,
               'assertions': assertions, 'cook_job': cook_job,
           })

Command strings are BARE ('describe_sim_events' / 'assert_simulation') --
command == bridge.execute name == register_handler name (the 4-bug MCP
convention class, mcp-fork-build-lessons).
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
def describe_sim_events_bridge_mock():
    """Spec-bound bridge mock returning the exact SPEC 4.1 vocab shape."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "events": [
            {"type": "fracture", "context": "rbd", "params": {"at_frame": "int", "trigger": "str?"}},
            {"type": "emit", "context": "pop|pyro", "params": {"at_frame": "int", "rate": "float"}},
            {"type": "tear", "context": "vellum", "params": {"threshold": "float"}},
            {"type": "ignite", "context": "pyro", "params": {"at_frame": "int", "region": "str?"}},
            {"type": "keyframe", "context": "any", "params": {"node": "str", "parm": "str", "frames": "[[f,v]]"}},
        ],
        "triggers": ["stress_gt", "collision_with", "frame_eq", "field_gt"],
        "assertions": [
            "piece_count", "constraint_count", "point_count", "velocity_bounds",
            "bbox_over_time", "field_stats", "mass_conservation",
        ],
    })
    return mock


@pytest.fixture()
def assert_simulation_bridge_mock():
    """Spec-bound bridge mock returning a minimal SPEC 4.1 assert_simulation shape."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "results": [
            {"metric": "piece_count", "pass": True, "series": [[38, 1], [40, 1], [41, 73]]},
        ],
        "pass": True,
    })
    return mock


@pytest.fixture()
def assert_simulation_error_bridge_mock():
    """Spec-bound bridge mock returning a scene-resolution {ok:false,error}
    shape -- must be passed through VERBATIM, never reinterpreted."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "ok": False,
        "error": "Node not found: /obj/does_not_exist (assertion node)",
    })
    return mock


# ---------------------------------------------------------------------------
# module import + callable (PRIMARY RED GATE)
# ---------------------------------------------------------------------------

class TestTemporalReasoningToolsModuleImport:
    """The new wrappers must be callables on a NEW temporal_reasoning_tools
    module. Until hou-dev creates the file, every import raises
    ModuleNotFoundError -- that is the red signal for this file."""

    def test_module_importable(self):
        """FAILS RED until hou-dev creates fxhoudinimcp/tools/temporal_reasoning_tools.py."""
        import fxhoudinimcp.tools.temporal_reasoning_tools  # noqa: F401

    def test_describe_sim_events_callable_on_module(self):
        from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_describe_sim_events  # noqa: F401
        assert callable(houdini_describe_sim_events), (
            "houdini_describe_sim_events must be a callable (the @mcp.tool coroutine)."
        )

    def test_assert_simulation_callable_on_module(self):
        from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_assert_simulation  # noqa: F401
        assert callable(houdini_assert_simulation), (
            "houdini_assert_simulation must be a callable (the @mcp.tool coroutine)."
        )


# ---------------------------------------------------------------------------
# require_approval=False on both wrappers (READONLY / ungated)
# ---------------------------------------------------------------------------

def _get_tool_map():
    from fxhoudinimcp.server import mcp
    tool_map: dict = {}
    if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
        tool_map = mcp._tool_manager._tools
    elif hasattr(mcp, "_tools"):
        tool_map = mcp._tools
    return tool_map


class TestTemporalReasoningRequireApproval:
    """Both tools must have meta={'require_approval': False} -- READONLY,
    ungated. assert_simulation is the explicit reversible-frame-evaluation
    exception (Blocker-2 of the plan's lockedFieldContract): it steps
    hou.setFrame() to read per-frame scalars but restores the saved frame
    in a finally and never writes any node/parm/userData state -- so it
    stays READONLY despite touching the playbar."""

    def test_describe_sim_events_require_approval_false(self):
        import fxhoudinimcp.tools.temporal_reasoning_tools  # noqa: F401
        tool_map = _get_tool_map()
        assert "houdini_describe_sim_events" in tool_map, (
            "houdini_describe_sim_events not registered on the mcp server; "
            "hou-dev must import temporal_reasoning_tools at server startup "
            "(tools/__init__.py)."
        )
        tool_obj = tool_map["houdini_describe_sim_events"]
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", True) if isinstance(meta, dict) else True
        assert require_approval is False, (
            f"houdini_describe_sim_events meta must have require_approval=False. Got meta={meta!r}."
        )

    def test_assert_simulation_require_approval_false(self):
        import fxhoudinimcp.tools.temporal_reasoning_tools  # noqa: F401
        tool_map = _get_tool_map()
        assert "houdini_assert_simulation" in tool_map, (
            "houdini_assert_simulation not registered on the mcp server; "
            "hou-dev must import temporal_reasoning_tools at server startup "
            "(tools/__init__.py)."
        )
        tool_obj = tool_map["houdini_assert_simulation"]
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", True) if isinstance(meta, dict) else True
        assert require_approval is False, (
            f"houdini_assert_simulation meta must have require_approval=False (READONLY -- "
            f"a Gate-1 reversible-frame-evaluation read oracle, not a mutation). Got meta={meta!r}."
        )


# ---------------------------------------------------------------------------
# bridge.execute call contract -- houdini_describe_sim_events
# ---------------------------------------------------------------------------

class TestDescribeSimEventsBridgeContract:
    """houdini_describe_sim_events must delegate to bridge.execute EXACTLY
    ONCE, with command 'describe_sim_events' and an EMPTY params dict."""

    @pytest.mark.asyncio
    async def test_exactly_one_bridge_execute_call(self, describe_sim_events_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=describe_sim_events_bridge_mock):
            from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_describe_sim_events

            ctx_mock = MagicMock()
            await houdini_describe_sim_events(ctx=ctx_mock)

        assert describe_sim_events_bridge_mock.execute.call_count == 1, (
            f"houdini_describe_sim_events must make exactly ONE bridge.execute call, "
            f"got {describe_sim_events_bridge_mock.execute.call_count}."
        )

    @pytest.mark.asyncio
    async def test_command_string_is_describe_sim_events(self, describe_sim_events_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=describe_sim_events_bridge_mock):
            from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_describe_sim_events

            ctx_mock = MagicMock()
            await houdini_describe_sim_events(ctx=ctx_mock)

        call_args = describe_sim_events_bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "describe_sim_events", (
            f"Expected bridge.execute('describe_sim_events', ...) but got command={command!r}. "
            "Command string must match register_handler's first arg exactly "
            "(the 4-bug convention: command == register name == params keys == handler kwargs)."
        )

    @pytest.mark.asyncio
    async def test_params_dict_is_empty(self, describe_sim_events_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=describe_sim_events_bridge_mock):
            from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_describe_sim_events

            ctx_mock = MagicMock()
            await houdini_describe_sim_events(ctx=ctx_mock)

        call_args = describe_sim_events_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert params == {}, (
            f"describe_sim_events takes NO params (a pure vocab read) -- expected an "
            f"empty params dict, got params={params!r}."
        )

    @pytest.mark.asyncio
    async def test_result_passed_through_verbatim(self, describe_sim_events_bridge_mock):
        expected_result = describe_sim_events_bridge_mock.execute.return_value
        with patch("fxhoudinimcp.server._get_bridge", return_value=describe_sim_events_bridge_mock):
            from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_describe_sim_events

            ctx_mock = MagicMock()
            result = await houdini_describe_sim_events(ctx=ctx_mock)

        assert result == expected_result, (
            "Wrapper must pass bridge.execute's result through unchanged (no ok-wrapper -- "
            f"a pure vocab read cannot fail on scene state). Expected {expected_result!r}, got {result!r}."
        )
        assert len(result["events"]) == 5
        assert result["assertions"] == [
            "piece_count", "constraint_count", "point_count", "velocity_bounds",
            "bbox_over_time", "field_stats", "mass_conservation",
        ]


# ---------------------------------------------------------------------------
# bridge.execute call contract -- houdini_assert_simulation
# ---------------------------------------------------------------------------

class TestAssertSimulationBridgeContract:
    """houdini_assert_simulation must delegate to bridge.execute EXACTLY
    ONCE, with command 'assert_simulation' and the exact 4-key params dict
    from the lockedFieldContract."""

    @pytest.mark.asyncio
    async def test_exactly_one_bridge_execute_call(self, assert_simulation_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=assert_simulation_bridge_mock):
            from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_assert_simulation

            ctx_mock = MagicMock()
            await houdini_assert_simulation(
                ctx=ctx_mock,
                network="/obj/rbd_sim",
                frame_range=[1, 80],
                assertions=[{"metric": "piece_count", "expect": {"at_frame": 40, "jump_gt": 50}}],
            )

        assert assert_simulation_bridge_mock.execute.call_count == 1, (
            f"houdini_assert_simulation must make exactly ONE bridge.execute call, "
            f"got {assert_simulation_bridge_mock.execute.call_count}."
        )

    @pytest.mark.asyncio
    async def test_command_string_is_assert_simulation(self, assert_simulation_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=assert_simulation_bridge_mock):
            from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_assert_simulation

            ctx_mock = MagicMock()
            await houdini_assert_simulation(
                ctx=ctx_mock,
                network="/obj/rbd_sim",
                frame_range=[1, 80],
                assertions=[],
            )

        call_args = assert_simulation_bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "assert_simulation", (
            f"Expected bridge.execute('assert_simulation', ...) but got command={command!r}."
        )

    @pytest.mark.asyncio
    async def test_params_dict_has_exact_four_keys(self, assert_simulation_bridge_mock):
        network = "/obj/rbd_sim"
        frame_range = [1, 80]
        assertions = [
            {"metric": "piece_count", "node": "/obj/rbd_sim/wall", "expect": {"at_frame": 40, "jump_gt": 50}},
            {"metric": "velocity_bounds", "max": 250},
        ]

        with patch("fxhoudinimcp.server._get_bridge", return_value=assert_simulation_bridge_mock):
            from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_assert_simulation

            ctx_mock = MagicMock()
            await houdini_assert_simulation(
                ctx=ctx_mock,
                network=network,
                frame_range=frame_range,
                assertions=assertions,
                cook_job="cook-9a",
            )

        call_args = assert_simulation_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        expected_keys = {"network", "frame_range", "assertions", "cook_job"}
        assert set(params.keys()) == expected_keys, (
            f"params dict must be exactly {expected_keys!r}. Got keys={set(params.keys())!r}."
        )
        assert params["network"] == network
        assert params["frame_range"] == frame_range
        assert params["assertions"] == assertions
        assert params["cook_job"] == "cook-9a"

    @pytest.mark.asyncio
    async def test_cook_job_defaults_to_none_and_is_forwarded(self, assert_simulation_bridge_mock):
        """cook_job defaults to None and must be forwarded verbatim as None
        (not omitted) when the caller doesn't pass it."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=assert_simulation_bridge_mock):
            from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_assert_simulation

            ctx_mock = MagicMock()
            await houdini_assert_simulation(
                ctx=ctx_mock,
                network="/obj/rbd_sim",
                frame_range=[1, 80],
                assertions=[],
            )

        call_args = assert_simulation_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert "cook_job" in params, f"'cook_job' key must be present even when None. Got params={params!r}."
        assert params["cook_job"] is None, (
            f"'cook_job' must default to None when the caller omits it. Got {params['cook_job']!r}."
        )

    @pytest.mark.asyncio
    async def test_success_result_passed_through_verbatim(self, assert_simulation_bridge_mock):
        expected_result = assert_simulation_bridge_mock.execute.return_value
        with patch("fxhoudinimcp.server._get_bridge", return_value=assert_simulation_bridge_mock):
            from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_assert_simulation

            ctx_mock = MagicMock()
            result = await houdini_assert_simulation(
                ctx=ctx_mock,
                network="/obj/rbd_sim",
                frame_range=[1, 80],
                assertions=[{"metric": "piece_count", "expect": {"at_frame": 40, "jump_gt": 50}}],
            )

        assert result == expected_result, (
            "Wrapper must pass bridge.execute's result through unchanged (the pure "
            f"SPEC 4.1 dict -- no ok-wrapper on success). Expected {expected_result!r}, got {result!r}."
        )
        assert result["pass"] is True

    @pytest.mark.asyncio
    async def test_scene_resolution_error_result_passed_through_not_reinterpreted(
        self, assert_simulation_error_bridge_mock
    ):
        """A scene-resolution {"ok": False, "error": ...} shape must survive
        the wrapper VERBATIM -- the wrapper must not raise, must not unwrap,
        and must not drop the error string."""
        expected_result = assert_simulation_error_bridge_mock.execute.return_value
        with patch("fxhoudinimcp.server._get_bridge", return_value=assert_simulation_error_bridge_mock):
            from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_assert_simulation

            ctx_mock = MagicMock()
            result = await houdini_assert_simulation(
                ctx=ctx_mock,
                network="/obj/does_not_exist",
                frame_range=[1, 10],
                assertions=[{"metric": "point_count", "expect": {"max": 100}}],
            )

        assert result == expected_result, (
            "A scene-resolution error bridge response must be returned VERBATIM -- "
            f"the wrapper must not reinterpret, unwrap, or swallow it. Expected "
            f"{expected_result!r}, got {result!r}."
        )
        assert result["ok"] is False
        assert "does_not_exist" in result["error"]


# ---------------------------------------------------------------------------
# spec-bound bridge mock -- PP12-110 regression guard
# ---------------------------------------------------------------------------

class TestBridgeSpecBoundGuard:
    """MagicMock(spec=HoudiniBridge) -- .call raises AttributeError (PP12-110 guard).

    In PP12-110, a wrapper called bridge.call(...) instead of
    bridge.execute(...). A bare MagicMock() silently accepted .call and
    returned a mock -- masking the bug. A spec-bound mock raises
    AttributeError because .call is not a real method on HoudiniBridge.
    """

    def test_bridge_call_attribute_does_not_exist(self):
        mock = MagicMock(spec=HoudiniBridge)
        with pytest.raises(AttributeError, match="call"):
            _ = mock.call  # .call must NOT exist on HoudiniBridge

    def test_bridge_execute_attribute_exists(self):
        mock = MagicMock(spec=HoudiniBridge)
        _ = mock.execute  # must NOT raise -- execute IS a real method


# ---------------------------------------------------------------------------
# ctx not in tool input schema (ctx is injected, not a tool parameter)
# ---------------------------------------------------------------------------

class TestCtxSchemaGuard:
    """ctx must NOT appear in either new tool's input schema properties.

    FastMCP injects ctx via the Context type annotation -- it is NOT a
    parameter the MCP client sends.
    """

    def _get_tool_schema(self, tool_name: str) -> dict:
        tool_map = _get_tool_map()
        tool_obj = tool_map.get(tool_name)
        if tool_obj is None:
            return {}
        return getattr(tool_obj, "parameters", getattr(tool_obj, "inputSchema", {})) or {}

    def test_describe_sim_events_ctx_not_in_schema(self):
        import fxhoudinimcp.tools.temporal_reasoning_tools  # noqa: F401
        schema = self._get_tool_schema("houdini_describe_sim_events")
        properties = schema.get("properties", {})
        assert "ctx" not in properties, (
            f"'ctx' must NOT appear in houdini_describe_sim_events's input schema properties. "
            f"Got properties={list(properties.keys())!r}."
        )

    def test_assert_simulation_ctx_not_in_schema(self):
        import fxhoudinimcp.tools.temporal_reasoning_tools  # noqa: F401
        schema = self._get_tool_schema("houdini_assert_simulation")
        properties = schema.get("properties", {})
        assert "ctx" not in properties, (
            f"'ctx' must NOT appear in houdini_assert_simulation's input schema properties. "
            f"Got properties={list(properties.keys())!r}."
        )


# =============================================================================
# houdini_compile_timeline — PP12-117 PR-3, the MUTATING/109-gated wrapper
# (NARROWED SCOPE per plan pp12-117c lockedFieldContract, REVISION 2).
#
# TDD phase: RED — houdini_compile_timeline does NOT exist yet on
# temporal_reasoning_tools.py. Every test below imports it via a fresh
# `from fxhoudinimcp.tools.temporal_reasoning_tools import
# houdini_compile_timeline` INSIDE the test (after any bridge patch is
# active), exactly like the PR-2 classes above — an ImportError here fails
# only these new tests, not the PR-2 suite above.
#
# Grounded LAYER-FOR-LAYER on test_spatial_reasoning_tools.py's
# TestSolveLayout* classes (THE exemplar for a GATED wrapper test):
# MagicMock(spec=HoudiniBridge); import-inside-test-after-patch;
# require_approval=True via the mcp tool_map; exactly-one bridge.execute
# call; exact params-dict key set; VERBATIM passthrough of BOTH a success
# result AND a pending-approval/preview shape (never reinterpreted, never
# raises, never gains/loses an 'ok' key); apply=false is STILL gated
# (fail-safe — gate capability is per-COMMAND, not per-argument);
# ctx-not-in-schema guard.
#
# Locked field contract (plan pp12-117c lockedFieldContract, REVISION 2):
#     houdini_compile_timeline(ctx, network, events, frame_range,
#                               apply=True) -> dict
#         -> a SINGLE bridge.execute('compile_timeline', {
#                'network': network, 'events': events,
#                'frame_range': frame_range, 'apply': apply,
#            })
#
# Decorated @mcp.tool(meta={'require_approval': True}) — GATED (the FIRST
# MUTATING tool of the Temporal/Sim-Reasoning MCP member; PP12-109
# security gate).
# =============================================================================

@pytest.fixture()
def compile_timeline_bridge_mock():
    """Spec-bound bridge mock returning a successful gated compile_timeline
    result (SPEC 4.1 shape: compiled/event_graph/applied/unresolved)."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "compiled": {
            "keyframes": [{"node": "/obj/rbd_sim", "parm": "tx", "frames": [[1, 0], [10, 5]]}],
            "chop_triggers": [],
            "dop_parms": [],
        },
        "event_graph": {"nodes": 1, "edges": 0},
        "applied": True,
        "unresolved": [],
    })
    return mock


@pytest.fixture()
def compile_timeline_pending_approval_bridge_mock():
    """Spec-bound bridge mock returning a 109-gate pending-approval shape —
    a pending/preview response is NOT a failure and must survive the
    wrapper untouched (mirrors solve_layout's pending-approval guard)."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "status": "pending_approval",
        "pending_id": "pend-compile-timeline-xyz789",
        "preview": {
            "would_set_keyframes": [{"node": "/obj/rbd_sim", "parm": "tx", "frame_count": 2}],
            "unresolved": [],
            "event_graph": {"nodes": 1, "edges": 0},
        },
    })
    return mock


class TestCompileTimelineToolsModuleImport:
    """houdini_compile_timeline must be a callable on the EXISTING
    temporal_reasoning_tools module (which already ships
    houdini_describe_sim_events / houdini_assert_simulation from PR-2).
    Until hou-dev adds it, this import raises ImportError — the RED
    signal for this section."""

    def test_existing_pr2_wrappers_unaffected(self):
        """The PR-2 wrappers must still be importable — the new PR-3
        wrapper must not clobber its existing siblings (append-only
        contract)."""
        from fxhoudinimcp.tools.temporal_reasoning_tools import (  # noqa: F401
            houdini_assert_simulation,
            houdini_describe_sim_events,
        )
        assert callable(houdini_describe_sim_events)
        assert callable(houdini_assert_simulation)

    def test_compile_timeline_callable_on_module(self):
        """FAILS RED until hou-dev adds houdini_compile_timeline."""
        from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_compile_timeline  # noqa: F401
        assert callable(houdini_compile_timeline), (
            "houdini_compile_timeline must be a callable (the @mcp.tool coroutine)."
        )


class TestCompileTimelineRequireApprovalTrue:
    """houdini_compile_timeline must have meta={'require_approval': True}
    — GATED (Capability.MUTATING handler-side; the first MUTATING tool of
    the Temporal/Sim-Reasoning MCP member)."""

    def test_mcp_tool_meta_require_approval_true(self):
        import fxhoudinimcp.tools.temporal_reasoning_tools  # noqa: F401
        tool_map = _get_tool_map()
        assert "houdini_compile_timeline" in tool_map, (
            "houdini_compile_timeline not registered on the mcp server; hou-dev must "
            "define it (decorated with @mcp.tool) in temporal_reasoning_tools.py."
        )
        tool_obj = tool_map["houdini_compile_timeline"]
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", False) if isinstance(meta, dict) else False
        assert require_approval is True, (
            f"houdini_compile_timeline meta must have require_approval=True (GATED — "
            f"a mutating tool). Got meta={meta!r}."
        )


class TestCompileTimelineBridgeContract:
    """houdini_compile_timeline must delegate to bridge.execute EXACTLY
    ONCE, with command 'compile_timeline' and the exact 4-key params dict
    from the lockedFieldContract."""

    @pytest.mark.asyncio
    async def test_exactly_one_bridge_execute_call(self, compile_timeline_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=compile_timeline_bridge_mock):
            from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_compile_timeline

            ctx_mock = MagicMock()
            await houdini_compile_timeline(
                ctx=ctx_mock,
                network="/obj/rbd_sim",
                events=[{
                    "id": "kf1", "type": "keyframe", "target": "/obj/rbd_sim",
                    "params": {"parm": "tx", "frames": [[1, 0], [10, 5]]},
                }],
                frame_range=[1, 10],
            )

        assert compile_timeline_bridge_mock.execute.call_count == 1, (
            f"houdini_compile_timeline must make exactly ONE bridge.execute call, "
            f"got {compile_timeline_bridge_mock.execute.call_count}."
        )

    @pytest.mark.asyncio
    async def test_command_string_is_compile_timeline(self, compile_timeline_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=compile_timeline_bridge_mock):
            from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_compile_timeline

            ctx_mock = MagicMock()
            await houdini_compile_timeline(
                ctx=ctx_mock, network="/obj/rbd_sim", events=[], frame_range=[1, 10],
            )

        call_args = compile_timeline_bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "compile_timeline", (
            f"Expected bridge.execute('compile_timeline', ...) but got command={command!r}. "
            "Command string must match register_handler's first arg exactly "
            "(the 4-bug convention: command == register name == params keys == handler kwargs)."
        )

    @pytest.mark.asyncio
    async def test_params_dict_has_exact_four_keys(self, compile_timeline_bridge_mock):
        network = "/obj/rbd_sim"
        events = [{
            "id": "kf1", "type": "keyframe", "target": "/obj/rbd_sim",
            "params": {"parm": "tx", "frames": [[1, 0], [10, 5]]},
        }]
        frame_range = [1, 10]

        with patch("fxhoudinimcp.server._get_bridge", return_value=compile_timeline_bridge_mock):
            from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_compile_timeline

            ctx_mock = MagicMock()
            await houdini_compile_timeline(
                ctx=ctx_mock, network=network, events=events, frame_range=frame_range,
                apply=False,
            )

        call_args = compile_timeline_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        expected_keys = {"network", "events", "frame_range", "apply"}
        assert set(params.keys()) == expected_keys, (
            f"params dict must be exactly {expected_keys!r}. Got keys={set(params.keys())!r}."
        )
        assert params["network"] == network
        assert params["events"] == events
        assert params["frame_range"] == frame_range
        assert params["apply"] is False

    @pytest.mark.asyncio
    async def test_apply_defaults_to_true_and_is_forwarded(self, compile_timeline_bridge_mock):
        """apply defaults to True and must be forwarded verbatim (not
        omitted) when the caller doesn't pass it."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=compile_timeline_bridge_mock):
            from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_compile_timeline

            ctx_mock = MagicMock()
            await houdini_compile_timeline(
                ctx=ctx_mock, network="/obj/rbd_sim", events=[], frame_range=[1, 10],
            )

        call_args = compile_timeline_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert params.get("apply") is True, f"apply must default to True. Got {params.get('apply')!r}."


class TestCompileTimelineResultPassthrough:
    """The wrapper must return bridge.execute's result VERBATIM — including
    the 109-gate pending-approval/preview shape (a normal, valid return —
    never reinterpreted, never raised)."""

    @pytest.mark.asyncio
    async def test_success_result_passed_through_verbatim(self, compile_timeline_bridge_mock):
        expected_result = compile_timeline_bridge_mock.execute.return_value
        with patch("fxhoudinimcp.server._get_bridge", return_value=compile_timeline_bridge_mock):
            from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_compile_timeline

            ctx_mock = MagicMock()
            result = await houdini_compile_timeline(
                ctx=ctx_mock, network="/obj/rbd_sim", events=[], frame_range=[1, 10],
            )

        assert result == expected_result, (
            f"Wrapper must pass bridge.execute's result through unchanged. "
            f"Expected {expected_result!r}, got {result!r}."
        )

    @pytest.mark.asyncio
    async def test_pending_approval_result_passed_through_not_treated_as_failure(
        self, compile_timeline_pending_approval_bridge_mock
    ):
        expected_result = compile_timeline_pending_approval_bridge_mock.execute.return_value
        with patch(
            "fxhoudinimcp.server._get_bridge",
            return_value=compile_timeline_pending_approval_bridge_mock,
        ):
            from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_compile_timeline

            ctx_mock = MagicMock()
            result = await houdini_compile_timeline(
                ctx=ctx_mock, network="/obj/rbd_sim", events=[], frame_range=[1, 10],
            )

        assert result == expected_result, (
            "A pending-approval bridge response must be returned VERBATIM — "
            f"Expected {expected_result!r}, got {result!r}."
        )
        assert result.get("status") == "pending_approval"
        assert "ok" not in result, (
            f"Wrapper must not inject an 'ok' key into a pending-approval response. "
            f"Got keys={list(result.keys())!r}."
        )


class TestCompileTimelineApplyFalseStillGated:
    """apply=false must STILL go through bridge.execute with
    require_approval still True on the tool (gate capability is
    per-COMMAND, not per-argument — mirrors solve_layout's
    apply=false-is-still-gated contract)."""

    @pytest.mark.asyncio
    async def test_apply_false_still_makes_a_single_gated_bridge_call(self, compile_timeline_bridge_mock):
        import fxhoudinimcp.tools.temporal_reasoning_tools  # noqa: F401
        tool_map = _get_tool_map()
        tool_obj = tool_map["houdini_compile_timeline"]
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", False) if isinstance(meta, dict) else False
        assert require_approval is True, (
            "houdini_compile_timeline must remain require_approval=True regardless of "
            "the apply argument value (fail-safe — gate capability is per-COMMAND)."
        )

        with patch("fxhoudinimcp.server._get_bridge", return_value=compile_timeline_bridge_mock):
            from fxhoudinimcp.tools.temporal_reasoning_tools import houdini_compile_timeline

            ctx_mock = MagicMock()
            await houdini_compile_timeline(
                ctx=ctx_mock, network="/obj/rbd_sim", events=[], frame_range=[1, 10], apply=False,
            )
        assert compile_timeline_bridge_mock.execute.call_count == 1


class TestCompileTimelineCtxSchemaGuard:
    """ctx must NOT appear in houdini_compile_timeline's input schema
    properties."""

    def _get_tool_schema(self, tool_name: str) -> dict:
        tool_map = _get_tool_map()
        tool_obj = tool_map.get(tool_name)
        if tool_obj is None:
            return {}
        return getattr(tool_obj, "parameters", getattr(tool_obj, "inputSchema", {})) or {}

    def test_compile_timeline_ctx_not_in_schema(self):
        import fxhoudinimcp.tools.temporal_reasoning_tools  # noqa: F401
        tool_map = _get_tool_map()
        assert "houdini_compile_timeline" in tool_map, (
            "houdini_compile_timeline not registered on the mcp server; hou-dev must "
            "define it (decorated with @mcp.tool) in temporal_reasoning_tools.py. "
            "(Without this assertion, an absent tool would vacuously pass the "
            "ctx-not-in-schema check below via an empty {} schema.)"
        )
        schema = self._get_tool_schema("houdini_compile_timeline")
        properties = schema.get("properties", {})
        assert "ctx" not in properties, (
            f"'ctx' must NOT appear in houdini_compile_timeline's input schema properties. "
            f"Got properties={list(properties.keys())!r}."
        )


class TestCompileTimelineWrapperImportsNoHouOrPxr:
    """The wrapper module must import NO hou, NO pxr (CL-015) — it must
    stay importable off-DCC for this pytest suite."""

    def test_source_has_no_forbidden_imports(self):
        import fxhoudinimcp.tools.temporal_reasoning_tools as mod
        with open(mod.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        forbidden_tokens = ["import hou", "from hou", "import pxr", "from pxr"]
        for token in forbidden_tokens:
            assert token not in source, (
                f"temporal_reasoning_tools.py must not contain {token!r} (CL-015)."
            )
