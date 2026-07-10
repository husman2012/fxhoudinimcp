"""pytest wrapper tests for houdini_describe_relations and houdini_assert_scene
MCP tools (PP12-116 PR-2, READONLY / UNGATED).

Unit: pp12-116b
testVerificationSurface: pytest-model
planSha: 1e391cf7bd9173ea2219a4a186bedf7c2bc5127809f72a8df25aaa1ad3eb33a5

These tests are written BEFORE the implementation (red phase). They will
fail with ModuleNotFoundError until hou-dev creates
fxhoudinimcp/tools/spatial_reasoning_tools.py (the two MCP wrappers) plus
the corresponding handler module
fxhoudinimcp_server/handlers/spatial_reasoning_handlers.py (not exercised
directly by this file — that is test_spatial_reasoning_handlers.py's job).

Grounded against (layer-for-layer template, per plan pp12-116b
reuseSurvey + lockedFieldContract):
  - python/fxhoudinimcp/tests/test_cop_onnx_tools.py (pp12-113b) — THE
    exemplar for a READONLY (require_approval=False) wrapper test:
    MagicMock(spec=HoudiniBridge); import-inside-test-after-patch; exactly-
    one bridge.execute call; exact command string + params-dict key set;
    verbatim passthrough of the bridge's result; ctx-not-in-schema guard;
    MagicMock(spec=HoudiniBridge) .call-raises-AttributeError PP12-110
    regression guard.
  - houdini/scripts/python/fxhoudinimcp_server/handlers/cop_onnx_handlers.py
    (the shipped READONLY handler-registration convention — not exercised
    directly here).

PP12-110 lessons encoded here (mcp-subprocess-delegation.md /
mcp-fork-build-lessons memory):
  - MagicMock(spec=HoudiniBridge) — a non-existent attr (e.g. .call) raises
    AttributeError; a bare MagicMock() would silently accept it and mask
    the bug.
  - Import the subject module INSIDE each test, AFTER _get_bridge is
    patched (module-level import would bind the tool before the patch is
    active).
  - Assert PUBLIC behavior (bridge.execute cmd + params; verbatim result
    passthrough) — NOT internal call order or unrelated attributes.
  - @pytest.mark.asyncio on every async test (both wrappers are async
    @mcp.tool coroutines).
  - Both wrappers make EXACTLY ONE bridge.execute call — no retries, no
    secondary calls.
  - Both must be require_approval=False (READONLY/ungated — no 109 gate;
    solve_layout, the MUTATING/gated tool, is PR-3, out of scope here).

Locked field contract (plan pp12-116b lockedFieldContract, REVISION 2):
    houdini_describe_relations(ctx) -> dict
        -> bridge.execute('describe_relations', {})

    houdini_assert_scene(ctx, objects, transforms=None, relations=None,
                          checks=None) -> dict
        -> bridge.execute('assert_scene', {
               'objects': objects, 'transforms': transforms,
               'relations': relations, 'checks': checks,
           })

Command strings are BARE ('describe_relations' / 'assert_scene') — command
== bridge.execute name == register_handler name (the 4-bug MCP convention
class, mcp-fork-build-lessons).
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
def describe_relations_bridge_mock():
    """Spec-bound bridge mock returning the exact SPEC 4.1 vocab shape."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "relations": [
            {"name": "on_top_of", "params": {"clearance": "float, default 0.0 -- gap above the support"},
             "desc": "A rests directly on top of B, along the +Y axis."},
            {"name": "under", "params": {"clearance": "float, default 0.0 -- gap above the support"},
             "desc": "B rests on top of A (the reverse of on_top_of)."},
            {"name": "adjacent", "params": {"gap": "float, default 0.0 -- face-to-face separation",
                                             "side": "one of +x,-x,+z,-z; optional -- nearest face-pair if absent"},
             "desc": "A sits beside B on the given world-axis side, separated by gap."},
            {"name": "non_overlap", "params": {}, "desc": "A and B do not intersect in 3D (AABB non-collision)."},
            {"name": "aligned", "params": {"axis": "one of x,y,z", "edge": "one of min,center,max; default center"},
             "desc": "A and B share the same coordinate on the given axis/edge."},
            {"name": "oriented_toward", "params": {},
             "desc": "A's +Z facing direction (rotated by A.r) points at target's X-Z centroid."},
            {"name": "clearance", "params": {"min": "float -- minimum required footprint distance"},
             "desc": "A keeps at least `min` distance from every other object's footprint on X-Z."},
        ],
    })
    return mock


@pytest.fixture()
def assert_scene_bridge_mock():
    """Spec-bound bridge mock returning a minimal SPEC 4.1 assert_scene shape."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "collision": {"pairs": [], "count": 0},
        "support": {"unsupported": [], "ok": True},
        "pass": True,
    })
    return mock


@pytest.fixture()
def assert_scene_error_bridge_mock():
    """Spec-bound bridge mock returning a scene-resolution {ok:false,error}
    shape -- must be passed through VERBATIM, never reinterpreted."""
    mock = MagicMock(spec=HoudiniBridge)
    mock.execute = AsyncMock(return_value={
        "ok": False,
        "error": "Node not found: /obj/does_not_exist (id=ghost)",
    })
    return mock


# ---------------------------------------------------------------------------
# module import + callable (PRIMARY RED GATE)
# ---------------------------------------------------------------------------

class TestSpatialReasoningToolsModuleImport:
    """The new wrappers must be callables on a NEW spatial_reasoning_tools
    module. Until hou-dev creates the file, every import raises
    ModuleNotFoundError — that is the red signal for this file."""

    def test_module_importable(self):
        """FAILS RED until hou-dev creates fxhoudinimcp/tools/spatial_reasoning_tools.py."""
        import fxhoudinimcp.tools.spatial_reasoning_tools  # noqa: F401

    def test_describe_relations_callable_on_module(self):
        from fxhoudinimcp.tools.spatial_reasoning_tools import houdini_describe_relations  # noqa: F401
        assert callable(houdini_describe_relations), (
            "houdini_describe_relations must be a callable (the @mcp.tool coroutine)."
        )

    def test_assert_scene_callable_on_module(self):
        from fxhoudinimcp.tools.spatial_reasoning_tools import houdini_assert_scene  # noqa: F401
        assert callable(houdini_assert_scene), (
            "houdini_assert_scene must be a callable (the @mcp.tool coroutine)."
        )


# ---------------------------------------------------------------------------
# require_approval=False on both wrappers (READONLY / ungated — no 109 gate)
# ---------------------------------------------------------------------------

def _get_tool_map():
    from fxhoudinimcp.server import mcp
    tool_map: dict = {}
    if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
        tool_map = mcp._tool_manager._tools
    elif hasattr(mcp, "_tools"):
        tool_map = mcp._tools
    return tool_map


class TestSpatialReasoningRequireApproval:
    """Both tools must have meta={'require_approval': False} — READONLY,
    ungated. Neither reads a live bounding box mutatively (a boundingBox()
    read may trigger an implicit cook, which is READ-consistent with the
    shipped geometry.get_bounding_box precedent, not a mutation). solve_layout
    (MUTATING/gated) is PR-3, out of scope for this unit."""

    def test_describe_relations_require_approval_false(self):
        import fxhoudinimcp.tools.spatial_reasoning_tools  # noqa: F401
        tool_map = _get_tool_map()
        assert "houdini_describe_relations" in tool_map, (
            "houdini_describe_relations not registered on the mcp server; "
            "hou-dev must import spatial_reasoning_tools at server startup "
            "(tools/__init__.py)."
        )
        tool_obj = tool_map["houdini_describe_relations"]
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", True) if isinstance(meta, dict) else True
        assert require_approval is False, (
            f"houdini_describe_relations meta must have require_approval=False. Got meta={meta!r}."
        )

    def test_assert_scene_require_approval_false(self):
        import fxhoudinimcp.tools.spatial_reasoning_tools  # noqa: F401
        tool_map = _get_tool_map()
        assert "houdini_assert_scene" in tool_map, (
            "houdini_assert_scene not registered on the mcp server; "
            "hou-dev must import spatial_reasoning_tools at server startup "
            "(tools/__init__.py)."
        )
        tool_obj = tool_map["houdini_assert_scene"]
        meta = getattr(tool_obj, "meta", None) or getattr(tool_obj, "tags", None) or {}
        require_approval = meta.get("require_approval", True) if isinstance(meta, dict) else True
        assert require_approval is False, (
            f"houdini_assert_scene meta must have require_approval=False (READONLY -- "
            f"a Gate-1 read oracle, not a mutation). Got meta={meta!r}."
        )


# ---------------------------------------------------------------------------
# bridge.execute call contract — houdini_describe_relations
# ---------------------------------------------------------------------------

class TestDescribeRelationsBridgeContract:
    """houdini_describe_relations must delegate to bridge.execute EXACTLY
    ONCE, with command 'describe_relations' and an EMPTY params dict."""

    @pytest.mark.asyncio
    async def test_exactly_one_bridge_execute_call(self, describe_relations_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=describe_relations_bridge_mock):
            from fxhoudinimcp.tools.spatial_reasoning_tools import houdini_describe_relations

            ctx_mock = MagicMock()
            await houdini_describe_relations(ctx=ctx_mock)

        assert describe_relations_bridge_mock.execute.call_count == 1, (
            f"houdini_describe_relations must make exactly ONE bridge.execute call, "
            f"got {describe_relations_bridge_mock.execute.call_count}."
        )

    @pytest.mark.asyncio
    async def test_command_string_is_describe_relations(self, describe_relations_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=describe_relations_bridge_mock):
            from fxhoudinimcp.tools.spatial_reasoning_tools import houdini_describe_relations

            ctx_mock = MagicMock()
            await houdini_describe_relations(ctx=ctx_mock)

        call_args = describe_relations_bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "describe_relations", (
            f"Expected bridge.execute('describe_relations', ...) but got command={command!r}. "
            "Command string must match register_handler's first arg exactly "
            "(the 4-bug convention: command == register name == params keys == handler kwargs)."
        )

    @pytest.mark.asyncio
    async def test_params_dict_is_empty(self, describe_relations_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=describe_relations_bridge_mock):
            from fxhoudinimcp.tools.spatial_reasoning_tools import houdini_describe_relations

            ctx_mock = MagicMock()
            await houdini_describe_relations(ctx=ctx_mock)

        call_args = describe_relations_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})
        assert params == {}, (
            f"describe_relations takes NO params (a pure vocab read) -- expected an "
            f"empty params dict, got params={params!r}."
        )

    @pytest.mark.asyncio
    async def test_result_passed_through_verbatim(self, describe_relations_bridge_mock):
        expected_result = describe_relations_bridge_mock.execute.return_value
        with patch("fxhoudinimcp.server._get_bridge", return_value=describe_relations_bridge_mock):
            from fxhoudinimcp.tools.spatial_reasoning_tools import houdini_describe_relations

            ctx_mock = MagicMock()
            result = await houdini_describe_relations(ctx=ctx_mock)

        assert result == expected_result, (
            "Wrapper must pass bridge.execute's result through unchanged (no ok-wrapper -- "
            f"a pure vocab read cannot fail on scene state). Expected {expected_result!r}, got {result!r}."
        )
        assert len(result["relations"]) == 7


# ---------------------------------------------------------------------------
# bridge.execute call contract — houdini_assert_scene
# ---------------------------------------------------------------------------

class TestAssertSceneBridgeContract:
    """houdini_assert_scene must delegate to bridge.execute EXACTLY ONCE,
    with command 'assert_scene' and the exact 4-key params dict from the
    lockedFieldContract."""

    @pytest.mark.asyncio
    async def test_exactly_one_bridge_execute_call(self, assert_scene_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=assert_scene_bridge_mock):
            from fxhoudinimcp.tools.spatial_reasoning_tools import houdini_assert_scene

            ctx_mock = MagicMock()
            await houdini_assert_scene(
                ctx=ctx_mock,
                objects=[{"id": "box_a", "bbox": [1.0, 1.0, 1.0]}],
            )

        assert assert_scene_bridge_mock.execute.call_count == 1, (
            f"houdini_assert_scene must make exactly ONE bridge.execute call, "
            f"got {assert_scene_bridge_mock.execute.call_count}."
        )

    @pytest.mark.asyncio
    async def test_command_string_is_assert_scene(self, assert_scene_bridge_mock):
        with patch("fxhoudinimcp.server._get_bridge", return_value=assert_scene_bridge_mock):
            from fxhoudinimcp.tools.spatial_reasoning_tools import houdini_assert_scene

            ctx_mock = MagicMock()
            await houdini_assert_scene(
                ctx=ctx_mock,
                objects=[{"id": "box_a", "bbox": [1.0, 1.0, 1.0]}],
            )

        call_args = assert_scene_bridge_mock.execute.call_args
        command = call_args[0][0] if call_args.args else call_args[1].get("command", "")
        assert command == "assert_scene", (
            f"Expected bridge.execute('assert_scene', ...) but got command={command!r}."
        )

    @pytest.mark.asyncio
    async def test_params_dict_has_exact_four_keys(self, assert_scene_bridge_mock):
        objects = [{"id": "box_a", "bbox": [1.0, 1.0, 1.0]}]
        transforms = {"box_a": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]}}
        relations = [{"type": "non_overlap", "a": "box_a", "b": "box_b"}]
        checks = ["collision", "support"]

        with patch("fxhoudinimcp.server._get_bridge", return_value=assert_scene_bridge_mock):
            from fxhoudinimcp.tools.spatial_reasoning_tools import houdini_assert_scene

            ctx_mock = MagicMock()
            await houdini_assert_scene(
                ctx=ctx_mock,
                objects=objects,
                transforms=transforms,
                relations=relations,
                checks=checks,
            )

        call_args = assert_scene_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        expected_keys = {"objects", "transforms", "relations", "checks"}
        assert set(params.keys()) == expected_keys, (
            f"params dict must be exactly {expected_keys!r}. Got keys={set(params.keys())!r}."
        )
        assert params["objects"] == objects
        assert params["transforms"] == transforms
        assert params["relations"] == relations
        assert params["checks"] == checks

    @pytest.mark.asyncio
    async def test_optional_params_default_to_none_and_are_forwarded(self, assert_scene_bridge_mock):
        """transforms/relations/checks all default to None and must be
        forwarded verbatim as None (not omitted, not coerced to [] / {})
        when the caller doesn't pass them."""
        with patch("fxhoudinimcp.server._get_bridge", return_value=assert_scene_bridge_mock):
            from fxhoudinimcp.tools.spatial_reasoning_tools import houdini_assert_scene

            ctx_mock = MagicMock()
            await houdini_assert_scene(
                ctx=ctx_mock,
                objects=[{"id": "box_a", "bbox": [1.0, 1.0, 1.0]}],
            )

        call_args = assert_scene_bridge_mock.execute.call_args
        params = call_args[0][1] if len(call_args.args) > 1 else call_args[1].get("params", {})

        for key in ("transforms", "relations", "checks"):
            assert key in params, f"'{key}' key must be present even when None. Got params={params!r}."
            assert params[key] is None, (
                f"'{key}' must default to None when the caller omits it. Got {params[key]!r}."
            )

    @pytest.mark.asyncio
    async def test_success_result_passed_through_verbatim(self, assert_scene_bridge_mock):
        expected_result = assert_scene_bridge_mock.execute.return_value
        with patch("fxhoudinimcp.server._get_bridge", return_value=assert_scene_bridge_mock):
            from fxhoudinimcp.tools.spatial_reasoning_tools import houdini_assert_scene

            ctx_mock = MagicMock()
            result = await houdini_assert_scene(
                ctx=ctx_mock,
                objects=[{"id": "box_a", "bbox": [1.0, 1.0, 1.0]}],
            )

        assert result == expected_result, (
            "Wrapper must pass bridge.execute's result through unchanged (the pure "
            f"SPEC 4.1 dict -- no ok-wrapper on success). Expected {expected_result!r}, got {result!r}."
        )
        assert result["pass"] is True

    @pytest.mark.asyncio
    async def test_scene_resolution_error_result_passed_through_not_reinterpreted(
        self, assert_scene_error_bridge_mock
    ):
        """A scene-resolution {"ok": False, "error": ...} shape must survive
        the wrapper VERBATIM -- the wrapper must not raise, must not unwrap,
        and must not drop the error string (mirrors the pending-approval /
        failed-cook verbatim-passthrough precedent from cop_onnx)."""
        expected_result = assert_scene_error_bridge_mock.execute.return_value
        with patch("fxhoudinimcp.server._get_bridge", return_value=assert_scene_error_bridge_mock):
            from fxhoudinimcp.tools.spatial_reasoning_tools import houdini_assert_scene

            ctx_mock = MagicMock()
            result = await houdini_assert_scene(
                ctx=ctx_mock,
                objects=[{"id": "ghost", "bbox": None, "node": "/obj/does_not_exist"}],
            )

        assert result == expected_result, (
            "A scene-resolution error bridge response must be returned VERBATIM -- "
            f"the wrapper must not reinterpret, unwrap, or swallow it. Expected "
            f"{expected_result!r}, got {result!r}."
        )
        assert result["ok"] is False
        assert "does_not_exist" in result["error"]


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

    def test_describe_relations_ctx_not_in_schema(self):
        import fxhoudinimcp.tools.spatial_reasoning_tools  # noqa: F401
        schema = self._get_tool_schema("houdini_describe_relations")
        properties = schema.get("properties", {})
        assert "ctx" not in properties, (
            f"'ctx' must NOT appear in houdini_describe_relations's input schema properties. "
            f"Got properties={list(properties.keys())!r}."
        )

    def test_assert_scene_ctx_not_in_schema(self):
        import fxhoudinimcp.tools.spatial_reasoning_tools  # noqa: F401
        schema = self._get_tool_schema("houdini_assert_scene")
        properties = schema.get("properties", {})
        assert "ctx" not in properties, (
            f"'ctx' must NOT appear in houdini_assert_scene's input schema properties. "
            f"Got properties={list(properties.keys())!r}."
        )
