"""Tests for MCP tool wrappers: validate bridge delegation."""

from __future__ import annotations

# Third-party
import pytest

# Internal
from fxhoudinimcp.errors import ConnectionError as HoudiniConnectionError
from fxhoudinimcp.tools.code import execute_python
from fxhoudinimcp.tools.materials import list_materials
from fxhoudinimcp.tools.nodes import create_node
from fxhoudinimcp.tools.scene import (
    get_houdini_connection_status,
    get_scene_info,
    new_scene,
)
from fxhoudinimcp.tools.workflows import setup_pyro_sim


class TestSceneTools:
    @pytest.mark.asyncio
    async def test_get_scene_info(self, mock_ctx, mock_bridge):
        mock_bridge.execute.return_value = {"hip_file": "/tmp/test.hip"}
        result = await get_scene_info(mock_ctx)
        mock_bridge.execute.assert_called_once_with("scene.get_scene_info")
        assert result == {"hip_file": "/tmp/test.hip"}

    @pytest.mark.asyncio
    async def test_new_scene(self, mock_ctx, mock_bridge):
        mock_bridge.execute.return_value = {"created": True}
        result = await new_scene(mock_ctx, save_current=True)
        mock_bridge.execute.assert_called_once_with("scene.new_scene", {"save_current": True})

    @pytest.mark.asyncio
    async def test_connection_status_success(self, mock_ctx, mock_bridge):
        mock_bridge.base_url = "http://localhost:8100"
        mock_bridge.health_check.return_value = {"status": "ok", "pid": 123}
        result = await get_houdini_connection_status(mock_ctx)
        assert result == {
            "connected": True,
            "base_url": "http://localhost:8100",
            "health": {"status": "ok", "pid": 123},
        }

    @pytest.mark.asyncio
    async def test_connection_status_disconnect(self, mock_ctx, mock_bridge):
        mock_bridge.base_url = "http://localhost:8100"
        mock_bridge.health_check.side_effect = HoudiniConnectionError(
            "Cannot connect",
            details={"url": "http://localhost:8100"},
        )
        result = await get_houdini_connection_status(mock_ctx)
        assert result["connected"] is False
        assert result["base_url"] == "http://localhost:8100"
        assert result["details"] == {"url": "http://localhost:8100"}


class TestNodeTools:
    @pytest.mark.asyncio
    async def test_create_node_required_params(self, mock_ctx, mock_bridge):
        mock_bridge.execute.return_value = {"path": "/obj/geo1/box1"}
        result = await create_node(mock_ctx, parent_path="/obj/geo1", node_type="box")
        mock_bridge.execute.assert_called_once_with(
            "nodes.create_node",
            {"parent_path": "/obj/geo1", "node_type": "box"},
        )

    @pytest.mark.asyncio
    async def test_create_node_all_params(self, mock_ctx, mock_bridge):
        await create_node(
            mock_ctx,
            parent_path="/obj",
            node_type="geo",
            name="my_geo",
            position=[0, 0],
        )
        mock_bridge.execute.assert_called_once_with(
            "nodes.create_node",
            {"parent_path": "/obj", "node_type": "geo", "name": "my_geo", "position": [0, 0]},
        )


class TestCodeTools:
    @pytest.mark.asyncio
    async def test_execute_python_code_only(self, mock_ctx, mock_bridge):
        result = await execute_python(
            mock_ctx,
            code="print('hi')",
            justification="no dedicated tool prints to the console",
        )
        mock_bridge.execute.assert_called_once_with(
            "code.execute_python",
            {"code": "print('hi')"},
        )
        # The justification is echoed back, never forwarded to Houdini.
        assert result["justification"]

    @pytest.mark.asyncio
    async def test_execute_python_with_return(self, mock_ctx, mock_bridge):
        await execute_python(
            mock_ctx,
            code="x = 1 + 1",
            justification="no dedicated tool evaluates arbitrary Python",
            return_expression="x",
        )
        mock_bridge.execute.assert_called_once_with(
            "code.execute_python",
            {"code": "x = 1 + 1", "return_expression": "x"},
        )

    @pytest.mark.asyncio
    async def test_justification_required_in_schemas(self):
        """The schema must force clients to articulate why VEX/Python."""
        from fxhoudinimcp.server import mcp

        tools = {t.name: t for t in await mcp.list_tools()}
        for tool_name in ("execute_python", "create_wrangle"):
            schema = tools[tool_name].inputSchema
            assert "justification" in schema["required"], (
                f"{tool_name} must require a justification"
            )


class TestWorkflowTools:
    @pytest.mark.asyncio
    async def test_setup_pyro_defaults(self, mock_ctx, mock_bridge):
        await setup_pyro_sim(mock_ctx)
        mock_bridge.execute.assert_called_once_with(
            "workflow.setup_pyro_sim",
            {
                "source_geo": "/obj/geo1/sphere1",
                "container": "box",
                "res_scale": 1.0,
                "substeps": 1,
                "name": "pyro_sim",
            },
        )


class TestMaterialTools:
    @pytest.mark.asyncio
    async def test_list_materials_default(self, mock_ctx, mock_bridge):
        await list_materials(mock_ctx)
        mock_bridge.execute.assert_called_once_with(
            "materials.list_materials",
            {"root_path": "/mat"},
        )


class TestGateControlToolsAreAgentReachable:
    """ADR-0007 Phase 4 — the full gate-control surface is reachable from an MCP client.

    Phase 2 withheld set_permission_mode and approve_pending_call, assuming the in-Houdini panel
    would be the operator's route to them. The panel was never built, so every client sat in PROPOSE
    with a queue nothing could drain. Enforcement lives in the HANDLERS (mode allowlist;
    MUTATING-only approval) — the bridge carries no caller identity, so a shell-capable agent
    reached these commands regardless and the wrappers fenced only the operator.
    """

    @pytest.mark.asyncio
    async def test_gate_control_tools_are_registered(self):
        from fxhoudinimcp.server import mcp

        names = {t.name for t in await mcp.list_tools()}
        assert "set_permission_mode" in names
        assert "approve_pending_call" in names

    @pytest.mark.asyncio
    async def test_readonly_gate_introspection_tools_are_still_registered(self):
        from fxhoudinimcp.server import mcp

        names = {t.name for t in await mcp.list_tools()}
        assert "get_permission_mode" in names
        assert "list_pending_calls" in names

    @pytest.mark.asyncio
    async def test_every_gate_command_has_a_tool_wrapper(self):
        # The module docstring promises all 7 gate commands; a silently-dropped wrapper is the
        # regression that stranded the operator under Phase 2.
        import fxhoudinimcp.tools.gate as gate_tools

        for name in (
            "get_permission_mode",
            "set_permission_mode",
            "list_pending_calls",
            "approve_pending_call",
            "reject_pending_call",
            "classify_code",
            "get_audit_log",
        ):
            assert hasattr(gate_tools, name), f"missing gate tool wrapper: {name}"
