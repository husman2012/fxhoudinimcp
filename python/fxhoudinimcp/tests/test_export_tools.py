"""
test_export_tools.py -- spec-bound mock-bridge tests for PP12-111 PR-3 export tools.

PR-3 discipline (mirrors test_retarget_tools.py / test_bonedeform_tools.py):
  - Tests use MagicMock(spec=HoudiniBridge) — bare MagicMock() is BANNED.
  - Regression guard asserts spec-bound mock raises AttributeError on .call
    (the bug that shipped silently in PP12-110 PR-3 before the spec= fix).
  - _get_bridge is asserted to receive ctx (server.py convention).
  - Bridge result passes through VERBATIM — wrappers are thin passthroughs.

Both export tools are read-only and UNGATED (FR-10):
  - houdini_export_probe_versions: require_approval=False
  - houdini_export_validate_budget: require_approval=False

validate_budget is DRY-RUN ONLY — it WRITES NOTHING (wrote_files=False).

Expected wrapper signatures (pp12-111c plan):

  @mcp.tool(meta={"require_approval": False})
  async def houdini_export_probe_versions(
      ctx: Context,
      target_ue: str | None = None,
  ) -> dict[str, Any]:
      bridge = _get_bridge(ctx)
      return await bridge.execute("probe_versions", {"target_ue": target_ue})

  @mcp.tool(meta={"require_approval": False})
  async def houdini_export_validate_budget(
      ctx: Context,
      node: str,
      target: str,
      budget_preset: str | None = None,
  ) -> dict[str, Any]:
      bridge = _get_bridge(ctx)
      return await bridge.execute("validate_budget", {
          "node": node, "target": target, "budget_preset": budget_preset,
      })

testVerificationSurface: hython-smoke
unitId: pp12-111c
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from fxhoudinimcp.tools import export_tools  # noqa: E402  -- ImportError = RED
from fxhoudinimcp.bridge import HoudiniBridge  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_bridge():
    bridge = MagicMock(spec=HoudiniBridge)
    bridge.execute = AsyncMock(return_value={
        "houdini": "21.0.729",
        "labs_vat_rop": "3.0",
        "rop_alembic": True,
        "rop_fbx": True,
    })
    return bridge


@pytest.fixture()
def mock_bridge_budget():
    bridge = MagicMock(spec=HoudiniBridge)
    bridge.execute = AsyncMock(return_value={
        "verdict": "pass",
        "checks": {
            "tri_count": {"status": "pass", "value": 1200, "limit": 65000},
        },
        "wrote_files": False,
    })
    return bridge


@pytest.fixture()
def mock_ctx():
    return MagicMock()


# ---------------------------------------------------------------------------
# Spec-bound regression guard
# ---------------------------------------------------------------------------

class TestSpecBoundRejectsCall:
    def test_bridge_call_raises_attribute_error(self, mock_bridge):
        """MagicMock(spec=HoudiniBridge) must raise AttributeError on .call
        (the banned bare-MagicMock typo that PP12-110 PR-3 shipped silently).
        This guard proves spec=HoudiniBridge is active for this test suite."""
        with pytest.raises(AttributeError):
            _ = mock_bridge.call


# ---------------------------------------------------------------------------
# Tool existence + async contract
# ---------------------------------------------------------------------------

class TestExportToolsExist:
    def test_probe_versions_exists(self):
        """houdini_export_probe_versions must be a module-level attribute of export_tools."""
        assert hasattr(export_tools, "houdini_export_probe_versions")

    def test_validate_budget_exists(self):
        """houdini_export_validate_budget must be a module-level attribute of export_tools."""
        assert hasattr(export_tools, "houdini_export_validate_budget")

    def test_probe_versions_is_async(self):
        """houdini_export_probe_versions must be an async coroutine function."""
        fn = getattr(export_tools, "houdini_export_probe_versions", None)
        assert fn is not None
        assert asyncio.iscoroutinefunction(fn)

    def test_validate_budget_is_async(self):
        """houdini_export_validate_budget must be an async coroutine function."""
        fn = getattr(export_tools, "houdini_export_validate_budget", None)
        assert fn is not None
        assert asyncio.iscoroutinefunction(fn)


# ---------------------------------------------------------------------------
# MCP gate — both tools must be UNGATED (require_approval=False, FR-10)
# ---------------------------------------------------------------------------

class TestExportToolsUngated:
    def _read_mcp_tool_meta(self, tool_name: str) -> dict:
        mcp = export_tools.mcp
        tools = mcp._tool_manager._tools
        assert tool_name in tools, f"{tool_name} not registered in MCP tool registry"
        return tools[tool_name].meta or {}

    def test_probe_versions_require_approval_false(self):
        """houdini_export_probe_versions must be decorated with meta={"require_approval": False}
        — it is read-only and UNGATED (FR-10), bypassing the PP12-109 gate."""
        meta = self._read_mcp_tool_meta("houdini_export_probe_versions")
        assert meta.get("require_approval") is False

    def test_validate_budget_require_approval_false(self):
        """houdini_export_validate_budget must be decorated with meta={"require_approval": False}
        — it is a DRY-RUN read tool and UNGATED (FR-10), bypassing the PP12-109 gate."""
        meta = self._read_mcp_tool_meta("houdini_export_validate_budget")
        assert meta.get("require_approval") is False


# ---------------------------------------------------------------------------
# probe_versions bridge delegation contract
# ---------------------------------------------------------------------------

class TestProbeVersionsBridgeExecute:
    @pytest.mark.asyncio
    async def test_calls_get_bridge_with_ctx(self, mock_ctx, mock_bridge):
        """_get_bridge must be called exactly once with ctx (server.py convention)."""
        with patch.object(export_tools, "_get_bridge", return_value=mock_bridge) as mock_get:
            await export_tools.houdini_export_probe_versions(mock_ctx)
            mock_get.assert_called_once_with(mock_ctx)

    @pytest.mark.asyncio
    async def test_calls_execute_with_probe_versions_command(self, mock_ctx, mock_bridge):
        """bridge.execute must be called with the command string 'probe_versions'."""
        with patch.object(export_tools, "_get_bridge", return_value=mock_bridge):
            await export_tools.houdini_export_probe_versions(mock_ctx)
        args = mock_bridge.execute.call_args
        assert args[0][0] == "probe_versions"

    @pytest.mark.asyncio
    async def test_probe_versions_default_params(self, mock_ctx, mock_bridge):
        """bridge.execute must receive {'target_ue': None} when target_ue is not supplied."""
        with patch.object(export_tools, "_get_bridge", return_value=mock_bridge):
            await export_tools.houdini_export_probe_versions(mock_ctx)
        mock_bridge.execute.assert_called_once_with(
            "probe_versions",
            {"target_ue": None},
        )

    @pytest.mark.asyncio
    async def test_probe_versions_with_target_ue(self, mock_ctx, mock_bridge):
        """bridge.execute must receive target_ue forwarded unchanged when supplied."""
        with patch.object(export_tools, "_get_bridge", return_value=mock_bridge):
            await export_tools.houdini_export_probe_versions(mock_ctx, target_ue="5.4")
        mock_bridge.execute.assert_called_once_with(
            "probe_versions",
            {"target_ue": "5.4"},
        )

    @pytest.mark.asyncio
    async def test_probe_versions_returns_bridge_result_verbatim(self, mock_ctx, mock_bridge):
        """houdini_export_probe_versions must return the bridge result verbatim.
        The wrapper is a thin passthrough — it must not mutate or repackage."""
        with patch.object(export_tools, "_get_bridge", return_value=mock_bridge):
            result = await export_tools.houdini_export_probe_versions(mock_ctx)
        assert "houdini" in result
        assert "labs_vat_rop" in result
        assert "rop_alembic" in result
        assert "rop_fbx" in result

    @pytest.mark.asyncio
    async def test_probe_versions_execute_called_once(self, mock_ctx, mock_bridge):
        """bridge.execute must be called exactly once per wrapper invocation."""
        with patch.object(export_tools, "_get_bridge", return_value=mock_bridge):
            await export_tools.houdini_export_probe_versions(mock_ctx)
        mock_bridge.execute.assert_called_once()


# ---------------------------------------------------------------------------
# validate_budget bridge delegation contract
# ---------------------------------------------------------------------------

class TestValidateBudgetBridgeExecute:
    @pytest.mark.asyncio
    async def test_calls_get_bridge_with_ctx(self, mock_ctx, mock_bridge_budget):
        """_get_bridge must be called exactly once with ctx."""
        with patch.object(export_tools, "_get_bridge", return_value=mock_bridge_budget) as mock_get:
            await export_tools.houdini_export_validate_budget(
                mock_ctx,
                node="/obj/geo1/box1",
                target="fbx",
            )
            mock_get.assert_called_once_with(mock_ctx)

    @pytest.mark.asyncio
    async def test_calls_execute_with_validate_budget_command(self, mock_ctx, mock_bridge_budget):
        """bridge.execute must be called with the command string 'validate_budget'."""
        with patch.object(export_tools, "_get_bridge", return_value=mock_bridge_budget):
            await export_tools.houdini_export_validate_budget(
                mock_ctx,
                node="/obj/geo1/box1",
                target="fbx",
            )
        args = mock_bridge_budget.execute.call_args
        assert args[0][0] == "validate_budget"

    @pytest.mark.asyncio
    async def test_validate_budget_default_params(self, mock_ctx, mock_bridge_budget):
        """bridge.execute must receive full params dict with budget_preset=None by default."""
        with patch.object(export_tools, "_get_bridge", return_value=mock_bridge_budget):
            await export_tools.houdini_export_validate_budget(
                mock_ctx,
                node="/obj/geo1/box1",
                target="fbx",
            )
        mock_bridge_budget.execute.assert_called_once_with(
            "validate_budget",
            {
                "node": "/obj/geo1/box1",
                "target": "fbx",
                "budget_preset": None,
            },
        )

    @pytest.mark.asyncio
    async def test_validate_budget_with_preset(self, mock_ctx, mock_bridge_budget):
        """budget_preset must be forwarded unchanged when supplied."""
        with patch.object(export_tools, "_get_bridge", return_value=mock_bridge_budget):
            await export_tools.houdini_export_validate_budget(
                mock_ctx,
                node="/obj/geo1/box1",
                target="chaos_gc",
                budget_preset="hero",
            )
        _cmd, params = mock_bridge_budget.execute.call_args[0]
        assert params["budget_preset"] == "hero"
        assert params["target"] == "chaos_gc"

    @pytest.mark.asyncio
    async def test_validate_budget_returns_bridge_result_verbatim(self, mock_ctx, mock_bridge_budget):
        """houdini_export_validate_budget must return bridge result verbatim.
        The wrapper is a thin passthrough — it must not mutate or repackage."""
        with patch.object(export_tools, "_get_bridge", return_value=mock_bridge_budget):
            result = await export_tools.houdini_export_validate_budget(
                mock_ctx,
                node="/obj/geo1/box1",
                target="fbx",
            )
        assert "verdict" in result
        assert "checks" in result

    @pytest.mark.asyncio
    async def test_validate_budget_wrote_files_false(self, mock_ctx, mock_bridge_budget):
        """validate_budget is DRY-RUN only: wrote_files must be False in the result.
        This is the key guard that the handler never writes files (FR-validate-dryrun)."""
        with patch.object(export_tools, "_get_bridge", return_value=mock_bridge_budget):
            result = await export_tools.houdini_export_validate_budget(
                mock_ctx,
                node="/obj/geo1/box1",
                target="fbx",
            )
        assert result.get("wrote_files") is False, (
            f"validate_budget must be DRY-RUN only (wrote_files=False); "
            f"got wrote_files={result.get('wrote_files')!r}"
        )

    @pytest.mark.asyncio
    async def test_validate_budget_execute_called_once(self, mock_ctx, mock_bridge_budget):
        """bridge.execute must be called exactly once per wrapper invocation."""
        with patch.object(export_tools, "_get_bridge", return_value=mock_bridge_budget):
            await export_tools.houdini_export_validate_budget(
                mock_ctx,
                node="/obj/geo1/box1",
                target="fbx",
            )
        mock_bridge_budget.execute.assert_called_once()
