"""
test_sessions_tools.py — spec-bound mock-bridge tests for PP12-115b session tools.

This is the GREEN phase for list_sessions and the RED phase for the
houdini_select_session scan+resolve upgrade (by-hip + ambiguous).

Covers the public contract of:
  - houdini_list_sessions(ctx) -> dict
      Scans ports base_port..base_port+9 via HoudiniBridge.health_check().
      Live ports => session_entry via build_session_entry.
      Dead ports (ConnectionError) => skipped.
      Returns {"sessions": [...], "active_port": <int>}
      where each entry has active=True iff port == lifespan_context["active_port"].

  - houdini_select_session(ctx, port=None, hip=None) -> dict
      SCANS all ports (like list_sessions), then:
        selector = port if port is not None else hip
        target = session_model.resolve_target(sessions, selector)
      If target is None (no-match / ambiguous / dead):
        returns {"ok": False, "error": <str>, "active_port": <UNCHANGED>}
      Else:
        state["active_port"] = target
        returns {"ok": True, "session": <entry>, "active_port": target}

      By-port: select_session(port=8101) scans, resolve_target finds 8101 if live.
      By-hip:  select_session(hip="asset") resolves to whichever live port's
               hip_file contains "asset" (case-insensitive, exactly one match).

Both tools:
  - @mcp.tool(meta={"require_approval": False})  — READ-ONLY, UNGATED
  - async def tool(ctx: Context, ...)
  - _get_bridge(ctx) convention grounded (must work with new lifespan shape)

Lifespan context shape (115b):
  {
    "host": "localhost",
    "base_port": 8100,
    "active_port": 8100,       # mutable — select_session updates this
    "bridges": {},             # port -> HoudiniBridge, managed by _get_bridge
  }

Bridge mock: MagicMock(spec=HoudiniBridge) — bare MagicMock() is BANNED.

Scan mock pattern (used by select tests):
  Two live ports: 8100 (layout.hip) and 8101 (asset.hip). All others dead.

testVerificationSurface: pytest-model (pure Python) + hython-smoke (real scan)
unitId: pp12-115b
Linear: HMDNI-116
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

from fxhoudinimcp.bridge import HoudiniBridge  # noqa: E402
from fxhoudinimcp.errors import ConnectionError as HoudiniConnectionError  # noqa: E402


# ---------------------------------------------------------------------------
# Module is now GREEN — import directly (no guard needed).
# ---------------------------------------------------------------------------

from fxhoudinimcp.tools import sessions as sessions_tools  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_lifespan(
    *,
    host: str = "localhost",
    base_port: int = 8100,
    active_port: int = 8100,
) -> dict:
    """Build a 115b-shaped lifespan context dict."""
    return {
        "host": host,
        "base_port": base_port,
        "active_port": active_port,
        "bridges": {},
    }


def _health_dict(port: int, **extra) -> dict:
    """Minimal health response for a live port."""
    return {
        "status": "ok",
        "pid": 1000 + port,
        "hip_file": f"C:/scenes/scene_{port}.hip",
        "houdini_version": "21.0.729",
        **extra,
    }


def _make_ctx(lifespan: dict | None = None) -> MagicMock:
    """Build a mock MCP context whose lifespan_context is the given dict."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = lifespan or _make_lifespan()
    return ctx


def _live_bridge(port: int, **extra_health) -> MagicMock:
    """Return a spec-bound mock bridge whose health_check succeeds."""
    bridge = MagicMock(spec=HoudiniBridge)
    bridge.health_check = AsyncMock(return_value=_health_dict(port, **extra_health))
    return bridge


def _dead_bridge() -> MagicMock:
    """Return a spec-bound mock bridge whose health_check raises ConnectionError."""
    bridge = MagicMock(spec=HoudiniBridge)
    bridge.health_check = AsyncMock(
        side_effect=HoudiniConnectionError(
            "Health check failed: cannot reach Houdini",
            details={"original_error": "Connection refused"},
        )
    )
    return bridge


def _scan_two_live(base_port: int = 8100) -> "callable":
    """Return a health_check side-effect where 8100=layout.hip live,
    8101=asset.hip live, all other ports dead.

    Used as: with patch.object(HoudiniBridge, "health_check", _scan_two_live()):
    """
    async def _fake_health_check(self):
        url = self.base_url  # e.g. "http://localhost:8101"
        port = int(url.split(":")[-1])
        if port == base_port:
            return {
                "status": "ok",
                "pid": 1000 + port,
                "hip_file": "C:/scenes/layout.hip",
                "houdini_version": "21.0.729",
            }
        if port == base_port + 1:
            return {
                "status": "ok",
                "pid": 1000 + port,
                "hip_file": "C:/scenes/asset.hip",
                "houdini_version": "21.0.729",
            }
        raise HoudiniConnectionError("dead", details={"original_error": "Connection refused"})

    return _fake_health_check


# ===========================================================================
# Section 1 — Spec-bound regression guard (MagicMock(spec=) discipline)
# ===========================================================================

class TestSpecBoundRejectsCall:
    """MagicMock(spec=HoudiniBridge) must raise AttributeError on .call.

    Ensures spec= is active for this test suite (PP12-110 lesson).
    """

    def test_bridge_call_raises_attribute_error(self):
        """spec-bound bridge must raise AttributeError on .call (not .execute)."""
        bridge = _live_bridge(8100)
        with pytest.raises(AttributeError):
            _ = bridge.call


# ===========================================================================
# Section 3 — Module-level attribute existence + async contract
# ===========================================================================

class TestSessionToolsExist:
    """Both tools must be async module-level attributes of sessions_tools."""

    def test_list_sessions_exists(self):
        """houdini_list_sessions must be a module-level attribute."""
        assert hasattr(sessions_tools, "houdini_list_sessions"), (
            "houdini_list_sessions not found in fxhoudinimcp.tools.sessions"
        )

    def test_select_session_exists(self):
        """houdini_select_session must be a module-level attribute."""
        assert hasattr(sessions_tools, "houdini_select_session"), (
            "houdini_select_session not found in fxhoudinimcp.tools.sessions"
        )

    def test_list_sessions_is_async(self):
        """houdini_list_sessions must be an async coroutine function."""
        fn = getattr(sessions_tools, "houdini_list_sessions", None)
        assert fn is not None
        assert asyncio.iscoroutinefunction(fn), (
            "houdini_list_sessions must be async"
        )

    def test_select_session_is_async(self):
        """houdini_select_session must be an async coroutine function."""
        fn = getattr(sessions_tools, "houdini_select_session", None)
        assert fn is not None
        assert asyncio.iscoroutinefunction(fn), (
            "houdini_select_session must be async"
        )


# ===========================================================================
# Section 4 — MCP gate: both tools must be UNGATED (require_approval=False)
# ===========================================================================

class TestSessionToolsUngated:
    """Both tools are READ-ONLY and must be registered with require_approval=False."""

    def _read_mcp_tool_meta(self, tool_name: str) -> dict:
        from fxhoudinimcp.server import mcp
        tools = mcp._tool_manager._tools
        assert tool_name in tools, (
            f"{tool_name} not registered in MCP tool registry"
        )
        return tools[tool_name].meta or {}

    def test_list_sessions_require_approval_false(self):
        """houdini_list_sessions must have require_approval=False (READ-ONLY, UNGATED)."""
        meta = self._read_mcp_tool_meta("houdini_list_sessions")
        assert meta.get("require_approval") is False, (
            f"houdini_list_sessions must be ungated (require_approval=False); meta={meta!r}"
        )

    def test_select_session_require_approval_false(self):
        """houdini_select_session must have require_approval=False (server-side routing only)."""
        meta = self._read_mcp_tool_meta("houdini_select_session")
        assert meta.get("require_approval") is False, (
            f"houdini_select_session must be ungated (require_approval=False); meta={meta!r}"
        )


# ===========================================================================
# Section 5 — houdini_list_sessions bridge delegation contract
#
# Plan AT pinned test:
#   Mock scan of ports 8100+8101 live → 2 entries with active flag on active_port.
# ===========================================================================

class TestListSessions:
    """houdini_list_sessions — scans ports, returns sessions list + active_port."""

    @pytest.fixture()
    def ctx_two_live(self) -> MagicMock:
        """Context with 8100 as active_port; both 8100 and 8101 respond live."""
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)
        return ctx

    @pytest.mark.asyncio
    async def test_pin_two_live_sessions(self, ctx_two_live):
        """PIN (plan AT-4): scan 8100+8101 live returns 2 entries; 8100 active=True.

        Monkeypatches HoudiniBridge.health_check so ports 8100+8101 succeed
        and all others raise ConnectionError.
        """
        lifespan = ctx_two_live.request_context.lifespan_context

        async def _fake_health_check(self):
            if self.base_url in ("http://localhost:8100", "http://localhost:8101"):
                port = int(self.base_url.split(":")[-1])
                return _health_dict(port)
            raise HoudiniConnectionError(
                "dead port",
                details={"original_error": "Connection refused"},
            )

        with patch.object(HoudiniBridge, "health_check", _fake_health_check):
            result = await sessions_tools.houdini_list_sessions(ctx_two_live)

        assert "sessions" in result, f"Result must have 'sessions' key; got {result!r}"
        assert "active_port" in result, "Result must have 'active_port' key"
        sessions = result["sessions"]
        assert len(sessions) == 2, (
            f"Expected 2 live sessions (8100+8101), got {len(sessions)}"
        )
        ports = {e["port"] for e in sessions}
        assert 8100 in ports, "Port 8100 must be in sessions"
        assert 8101 in ports, "Port 8101 must be in sessions"

    @pytest.mark.asyncio
    async def test_active_port_flag_set(self, ctx_two_live):
        """Session at active_port=8100 has active=True; 8101 has active=False."""
        async def _fake_health_check(self):
            if self.base_url in ("http://localhost:8100", "http://localhost:8101"):
                port = int(self.base_url.split(":")[-1])
                return _health_dict(port)
            raise HoudiniConnectionError("dead", details={})

        with patch.object(HoudiniBridge, "health_check", _fake_health_check):
            result = await sessions_tools.houdini_list_sessions(ctx_two_live)

        sessions = result["sessions"]
        by_port = {e["port"]: e for e in sessions}
        assert by_port[8100]["active"] is True, "port 8100 must be active=True"
        assert by_port[8101]["active"] is False, "port 8101 must be active=False"

    @pytest.mark.asyncio
    async def test_dead_port_skipped(self):
        """Dead ports (ConnectionError) must be skipped — not included in sessions."""
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)

        async def _fake_health_check(self):
            if self.base_url == "http://localhost:8100":
                return _health_dict(8100)
            raise HoudiniConnectionError("dead", details={})

        with patch.object(HoudiniBridge, "health_check", _fake_health_check):
            result = await sessions_tools.houdini_list_sessions(ctx)

        sessions = result["sessions"]
        assert len(sessions) == 1, f"Only 8100 should appear; got {sessions!r}"
        assert sessions[0]["port"] == 8100

    @pytest.mark.asyncio
    async def test_all_dead_returns_empty_list(self):
        """All ports dead => sessions=[] with active_port intact."""
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)

        async def _fake_health_check(self):
            raise HoudiniConnectionError("dead", details={})

        with patch.object(HoudiniBridge, "health_check", _fake_health_check):
            result = await sessions_tools.houdini_list_sessions(ctx)

        assert result["sessions"] == []
        assert result["active_port"] == 8100

    @pytest.mark.asyncio
    async def test_result_active_port_matches_lifespan(self):
        """result['active_port'] must equal lifespan_context['active_port']."""
        lifespan = _make_lifespan(base_port=8100, active_port=8101)
        ctx = _make_ctx(lifespan)

        async def _fake_health_check(self):
            if self.base_url == "http://localhost:8101":
                return _health_dict(8101)
            raise HoudiniConnectionError("dead", details={})

        with patch.object(HoudiniBridge, "health_check", _fake_health_check):
            result = await sessions_tools.houdini_list_sessions(ctx)

        assert result["active_port"] == 8101


# ===========================================================================
# Section 6 — houdini_select_session scan+resolve contract
#
# The UPGRADED contract (hou-dev implements this — these tests are RED):
#   houdini_select_session(ctx, port=None, hip=None)
#   1. SCANs all ports (same pattern as list_sessions).
#   2. selector = port if port is not None else hip
#   3. target = session_model.resolve_target(sessions_entries, selector)
#   4. If target is None → {"ok": False, "error": <str>, "active_port": UNCHANGED}
#   5. Else → state["active_port"] = target
#              {"ok": True, "session": <entry>, "active_port": target}
#
# SCAN FIXTURE (used by all select tests):
#   Port 8100: live, hip_file="C:/scenes/layout.hip"
#   Port 8101: live, hip_file="C:/scenes/asset.hip"
#   Ports 8102-8115: dead
#
# Plan AT pinned tests (AT-5a, AT-5b, AT-5c) plus new by-hip tests (AT-5d..g).
# ===========================================================================

class TestSelectSession:
    """houdini_select_session — scan + resolve_target, update active_port on success."""

    # -----------------------------------------------------------------------
    # By-PORT tests (scan-aware: probe all ports, resolve by int match)
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_pin_select_live_port(self):
        """PIN (plan AT-5a): select port=8101 (live in scan) => active_port=8101, ok=True."""
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)

        with patch.object(HoudiniBridge, "health_check", _scan_two_live(8100)):
            result = await sessions_tools.houdini_select_session(ctx, port=8101)

        assert result.get("ok") is True, (
            f"Selecting a live port must return ok=True; got {result!r}"
        )
        assert ctx.request_context.lifespan_context["active_port"] == 8101, (
            "lifespan_context['active_port'] must be updated to 8101"
        )

    @pytest.mark.asyncio
    async def test_pin_select_returns_session_entry(self):
        """PIN (plan AT-5b): result['session'] carries the selected port's entry."""
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)

        with patch.object(HoudiniBridge, "health_check", _scan_two_live(8100)):
            result = await sessions_tools.houdini_select_session(ctx, port=8101)

        assert "session" in result, f"Result must include 'session'; got {result!r}"
        assert result["session"]["port"] == 8101

    @pytest.mark.asyncio
    async def test_pin_select_absent_port_returns_error(self):
        """PIN (plan AT-5c): selecting port=8199 (absent from scan) returns ok=False.

        CRITICAL: active_port must NOT be changed when the port is absent.
        """
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)

        with patch.object(HoudiniBridge, "health_check", _scan_two_live(8100)):
            result = await sessions_tools.houdini_select_session(ctx, port=8199)

        assert result.get("ok") is False, (
            f"Selecting an absent port must return ok=False; got {result!r}"
        )
        assert ctx.request_context.lifespan_context["active_port"] == 8100, (
            "active_port must NOT change when target port is absent from scan"
        )

    @pytest.mark.asyncio
    async def test_select_absent_port_includes_error_message(self):
        """Absent-port result must include an 'error' key with a string message."""
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)

        with patch.object(HoudiniBridge, "health_check", _scan_two_live(8100)):
            result = await sessions_tools.houdini_select_session(ctx, port=8199)

        assert "error" in result, (
            f"Absent-port result must include 'error' key; got {result!r}"
        )
        assert isinstance(result["error"], str), "'error' must be a string"

    @pytest.mark.asyncio
    async def test_select_same_port_is_idempotent(self):
        """Selecting already-active port=8100 succeeds with ok=True, no side effects."""
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)

        with patch.object(HoudiniBridge, "health_check", _scan_two_live(8100)):
            result = await sessions_tools.houdini_select_session(ctx, port=8100)

        assert result.get("ok") is True
        assert ctx.request_context.lifespan_context["active_port"] == 8100

    @pytest.mark.asyncio
    async def test_select_does_not_change_active_port_on_failure(self):
        """active_port stays unchanged after a failed select — guard against mutation."""
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)
        initial_port = lifespan["active_port"]

        with patch.object(HoudiniBridge, "health_check", _scan_two_live(8100)):
            await sessions_tools.houdini_select_session(ctx, port=9999)

        assert lifespan["active_port"] == initial_port, (
            "active_port must not change after a failed select"
        )

    @pytest.mark.asyncio
    async def test_select_result_includes_active_port_on_success(self):
        """Successful select result must carry active_port equal to the selected port."""
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)

        with patch.object(HoudiniBridge, "health_check", _scan_two_live(8100)):
            result = await sessions_tools.houdini_select_session(ctx, port=8101)

        assert "active_port" in result, (
            f"Success result must include 'active_port'; got {result!r}"
        )
        assert result["active_port"] == 8101

    @pytest.mark.asyncio
    async def test_select_result_includes_active_port_on_failure(self):
        """Failed select result must carry active_port equal to the UNCHANGED original."""
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)

        with patch.object(HoudiniBridge, "health_check", _scan_two_live(8100)):
            result = await sessions_tools.houdini_select_session(ctx, port=8199)

        assert "active_port" in result, (
            f"Failure result must include 'active_port'; got {result!r}"
        )
        assert result["active_port"] == 8100, (
            "Failure result active_port must equal original unchanged active_port"
        )

    # -----------------------------------------------------------------------
    # By-HIP tests (NEW — select by scene-name substring; RED phase)
    #
    # Scan: 8100=layout.hip, 8101=asset.hip, rest dead.
    # resolve_target(sessions, "asset") -> 8101 (exactly one match)
    # resolve_target(sessions, "nomatch") -> None (zero matches)
    # resolve_target(sessions, "layout") with TWO "layout" sessions -> None (ambiguous)
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_select_by_hip_resolves_unique_match(self):
        """AT-5d (NEW/RED): select hip='asset' resolves to port 8101 (the asset.hip port).

        This is the core new UX — select by scene name.
        select_session does NOT yet accept a hip= kwarg; this test is RED.
        """
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)

        with patch.object(HoudiniBridge, "health_check", _scan_two_live(8100)):
            result = await sessions_tools.houdini_select_session(ctx, hip="asset")

        assert result.get("ok") is True, (
            f"select hip='asset' must resolve uniquely and return ok=True; got {result!r}"
        )
        assert ctx.request_context.lifespan_context["active_port"] == 8101, (
            "active_port must be updated to 8101 (the asset.hip port)"
        )
        assert result.get("active_port") == 8101, (
            "result['active_port'] must be 8101"
        )
        assert "session" in result, (
            f"Successful by-hip select must include 'session'; got {result!r}"
        )
        assert result["session"]["port"] == 8101

    @pytest.mark.asyncio
    async def test_select_by_hip_no_match_returns_error(self):
        """AT-5e (NEW/RED): select hip='nomatch' → ok=False, active_port UNCHANGED."""
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)

        with patch.object(HoudiniBridge, "health_check", _scan_two_live(8100)):
            result = await sessions_tools.houdini_select_session(ctx, hip="nomatch")

        assert result.get("ok") is False, (
            f"select hip='nomatch' must return ok=False; got {result!r}"
        )
        assert ctx.request_context.lifespan_context["active_port"] == 8100, (
            "active_port must NOT change on no-match hip select"
        )
        assert "error" in result, (
            f"No-match result must include 'error'; got {result!r}"
        )
        assert isinstance(result["error"], str)

    @pytest.mark.asyncio
    async def test_select_by_hip_ambiguous_returns_error(self):
        """AT-5f (NEW/RED): hip='layout' matches TWO sessions => ambiguous => ok=False.

        Scan uses a two-layout fixture where both 8100 and 8101 have
        'layout' in their hip_file path. resolve_target returns None for
        two-or-more matches (ambiguous). active_port must NOT change.
        """
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)

        # Custom scan: BOTH ports have "layout" in their hip path.
        async def _two_layout_scan(self):
            port = int(self.base_url.split(":")[-1])
            if port in (8100, 8101):
                return {
                    "status": "ok",
                    "pid": 1000 + port,
                    "hip_file": f"C:/scenes/layout_{port}.hip",
                    "houdini_version": "21.0.729",
                }
            raise HoudiniConnectionError("dead", details={})

        with patch.object(HoudiniBridge, "health_check", _two_layout_scan):
            result = await sessions_tools.houdini_select_session(ctx, hip="layout")

        assert result.get("ok") is False, (
            f"Ambiguous hip='layout' (two matches) must return ok=False; got {result!r}"
        )
        assert ctx.request_context.lifespan_context["active_port"] == 8100, (
            "active_port must NOT change on ambiguous hip select"
        )
        assert "error" in result, (
            f"Ambiguous result must include 'error'; got {result!r}"
        )
        assert isinstance(result["error"], str)

    @pytest.mark.asyncio
    async def test_select_by_hip_case_insensitive(self):
        """AT-5g (NEW/RED): hip='ASSET' (uppercase) resolves to 8101 same as lowercase.

        resolve_target performs case-insensitive substring match.
        """
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)

        with patch.object(HoudiniBridge, "health_check", _scan_two_live(8100)):
            result = await sessions_tools.houdini_select_session(ctx, hip="ASSET")

        assert result.get("ok") is True, (
            f"Case-insensitive hip match must resolve to ok=True; got {result!r}"
        )
        assert ctx.request_context.lifespan_context["active_port"] == 8101


# ===========================================================================
# Section 7 — AMBIGUOUS-VS-NO-MATCH error message distinction (PP12-115c)
#
# Current behaviour: houdini_select_session returns a generic error string for
# BOTH zero-match and multi-match cases.  The v1.1 contract requires distinct
# messages so the caller can differentiate them.
#
# Expected contracts (RED until hou-dev implements resolve_with_reason):
#   no-match  → error contains "no session" or "no match" (NOT "ambiguous")
#   ambiguous → error contains "ambiguous" (NOT "no session"/"no match" alone)
#
# All tests in this class are RED because the distinction does not yet exist
# in the shipped houdini_select_session.
# ===========================================================================

class TestSelectSessionErrorDistinction:
    """houdini_select_session error message must distinguish no-match from ambiguous."""

    @pytest.mark.asyncio
    async def test_no_match_error_does_not_say_ambiguous(self):
        """AT-8a (PIN/RED): no-match error must NOT say 'ambiguous'.

        When zero sessions match the hip selector, the error must clearly
        indicate no match was found — NOT hint at multiple matches.
        """
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)

        with patch.object(HoudiniBridge, "health_check", _scan_two_live(8100)):
            result = await sessions_tools.houdini_select_session(ctx, hip="nomatch")

        assert result.get("ok") is False
        error = result.get("error", "")
        assert isinstance(error, str)
        assert "ambiguous" not in error.lower(), (
            f"No-match error must not say 'ambiguous'; got error={error!r}"
        )

    @pytest.mark.asyncio
    async def test_ambiguous_error_says_ambiguous(self):
        """AT-8b (PIN/RED): ambiguous (multi-match) error MUST say 'ambiguous'.

        When two or more sessions match the hip selector, the error must say
        'ambiguous' so the caller knows to use a more specific selector.

        This test is RED because current select_session returns the same generic
        "no live session matches" for both zero-match and multi-match cases.
        """
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)

        # Two sessions both containing "layout" — ambiguous match.
        async def _two_layout_scan(self):
            port = int(self.base_url.split(":")[-1])
            if port in (8100, 8101):
                return {
                    "status": "ok",
                    "pid": 1000 + port,
                    "hip_file": f"C:/scenes/layout_{port}.hip",
                    "houdini_version": "21.0.729",
                }
            raise HoudiniConnectionError("dead", details={})

        with patch.object(HoudiniBridge, "health_check", _two_layout_scan):
            result = await sessions_tools.houdini_select_session(ctx, hip="layout")

        assert result.get("ok") is False
        error = result.get("error", "")
        assert isinstance(error, str)
        assert "ambiguous" in error.lower(), (
            f"Ambiguous error must contain 'ambiguous'; got error={error!r}"
        )

    @pytest.mark.asyncio
    async def test_no_match_error_is_informative(self):
        """AT-8c: no-match error message is a non-empty informative string.

        Must contain enough information to diagnose the problem (e.g. the
        selector value or 'no session').
        """
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)

        with patch.object(HoudiniBridge, "health_check", _scan_two_live(8100)):
            result = await sessions_tools.houdini_select_session(ctx, hip="zzznomatchzzz")

        error = result.get("error", "")
        assert len(error) > 0, "No-match error message must not be empty"
        # Must reference the selector or 'no session'
        assert "zzznomatchzzz" in error or "no" in error.lower(), (
            f"No-match error must reference selector or 'no session'; got {error!r}"
        )

    @pytest.mark.asyncio
    async def test_ambiguous_error_is_informative(self):
        """AT-8d: ambiguous error message is a non-empty informative string.

        Must say 'ambiguous' and ideally reference the selector.
        """
        lifespan = _make_lifespan(base_port=8100, active_port=8100)
        ctx = _make_ctx(lifespan)

        async def _two_layout_scan(self):
            port = int(self.base_url.split(":")[-1])
            if port in (8100, 8101):
                return {
                    "status": "ok",
                    "pid": 1000 + port,
                    "hip_file": f"C:/scenes/layout_{port}.hip",
                    "houdini_version": "21.0.729",
                }
            raise HoudiniConnectionError("dead", details={})

        with patch.object(HoudiniBridge, "health_check", _two_layout_scan):
            result = await sessions_tools.houdini_select_session(ctx, hip="layout")

        error = result.get("error", "")
        assert len(error) > 0, "Ambiguous error message must not be empty"
        assert "ambiguous" in error.lower(), (
            f"Ambiguous error must contain 'ambiguous'; got {error!r}"
        )


# ===========================================================================
# Section 8 — PID PINNING: select_session records pid; list_sessions reports
#             stale_pid flag when pid has drifted (PP12-115c)
#
# v1.1 lifespan shape adds active_pid: int|None.
# On select_session success: state["active_pid"] = session_entry["pid"]
# On list_sessions: each entry gets "active_pid_stale": bool
#
# These tests are RED because neither active_pid tracking nor the stale flag
# exist in the current implementation.
# ===========================================================================

def _make_lifespan_v11(
    *,
    host: str = "localhost",
    base_port: int = 8100,
    active_port: int = 8100,
    active_pid: int | None = None,
) -> dict:
    """Build a v1.1-shaped lifespan context dict (adds active_pid)."""
    return {
        "host": host,
        "base_port": base_port,
        "active_port": active_port,
        "active_pid": active_pid,
        "bridges": {},
    }


class TestSelectSessionPidPinning:
    """select_session must record pid on success (v1.1 — RED until hou-dev implements)."""

    @pytest.mark.asyncio
    async def test_select_success_records_pid_in_lifespan(self):
        """AT-9a (PIN/RED): successful select stores pid in lifespan['active_pid'].

        After a successful houdini_select_session(port=8101), the lifespan
        context must have active_pid == the pid from the selected session entry.
        """
        lifespan = _make_lifespan_v11(base_port=8100, active_port=8100, active_pid=None)
        ctx = _make_ctx(lifespan)

        with patch.object(HoudiniBridge, "health_check", _scan_two_live(8100)):
            result = await sessions_tools.houdini_select_session(ctx, port=8101)

        assert result.get("ok") is True, f"Expected ok=True; got {result!r}"
        # pid for port 8101 in _scan_two_live: 1000 + 8101 = 9101
        expected_pid = 1000 + 8101
        assert lifespan.get("active_pid") == expected_pid, (
            f"lifespan['active_pid'] must be {expected_pid} after successful select; "
            f"got {lifespan.get('active_pid')!r}"
        )

    @pytest.mark.asyncio
    async def test_select_failure_does_not_change_active_pid(self):
        """AT-9b (PIN/RED): failed select must NOT change lifespan['active_pid'].

        An absent-port failure must leave active_pid intact (same guard as
        the active_port invariant — no partial state mutation).
        """
        initial_pid = 5000
        lifespan = _make_lifespan_v11(base_port=8100, active_port=8100, active_pid=initial_pid)
        ctx = _make_ctx(lifespan)

        with patch.object(HoudiniBridge, "health_check", _scan_two_live(8100)):
            result = await sessions_tools.houdini_select_session(ctx, port=9999)

        assert result.get("ok") is False, f"Expected ok=False; got {result!r}"
        assert lifespan.get("active_pid") == initial_pid, (
            f"Failed select must not change active_pid; "
            f"expected {initial_pid}, got {lifespan.get('active_pid')!r}"
        )

    @pytest.mark.asyncio
    async def test_select_by_hip_success_records_pid(self):
        """AT-9c (PIN/RED): successful by-hip select also records pid in active_pid."""
        lifespan = _make_lifespan_v11(base_port=8100, active_port=8100, active_pid=None)
        ctx = _make_ctx(lifespan)

        with patch.object(HoudiniBridge, "health_check", _scan_two_live(8100)):
            result = await sessions_tools.houdini_select_session(ctx, hip="asset")

        assert result.get("ok") is True, f"Expected ok=True; got {result!r}"
        # asset.hip is on 8101 in _scan_two_live: pid = 1000 + 8101 = 9101
        expected_pid = 1000 + 8101
        assert lifespan.get("active_pid") == expected_pid, (
            f"lifespan['active_pid'] must be {expected_pid} after by-hip select; "
            f"got {lifespan.get('active_pid')!r}"
        )


class TestListSessionsStalePid:
    """houdini_list_sessions must report active_pid_stale flag (v1.1 — RED)."""

    @pytest.mark.asyncio
    async def test_list_sessions_reports_stale_pid_when_drifted(self):
        """AT-10a (PIN/RED): list_sessions includes active_pid_stale=True when pid drifted.

        Scenario: we selected port 8100 when its pid was 5000.
        The live scan now returns pid=9999 for port 8100 (Houdini restarted).
        houdini_list_sessions must set active_pid_stale=True in the result.
        """
        lifespan = _make_lifespan_v11(
            base_port=8100, active_port=8100, active_pid=5000,
        )
        ctx = _make_ctx(lifespan)

        async def _stale_pid_scan(self):
            port = int(self.base_url.split(":")[-1])
            if port == 8100:
                return {
                    "status": "ok",
                    "pid": 9999,   # drifted from recorded 5000
                    "hip_file": "C:/scenes/layout.hip",
                    "houdini_version": "21.0.729",
                }
            raise HoudiniConnectionError("dead", details={})

        with patch.object(HoudiniBridge, "health_check", _stale_pid_scan):
            result = await sessions_tools.houdini_list_sessions(ctx)

        assert "active_pid_stale" in result, (
            f"houdini_list_sessions v1.1 must include 'active_pid_stale'; got keys={list(result.keys())}"
        )
        assert result["active_pid_stale"] is True, (
            f"Pid drift 5000→9999 must set active_pid_stale=True; got {result['active_pid_stale']!r}"
        )

    @pytest.mark.asyncio
    async def test_list_sessions_reports_stale_pid_false_when_stable(self):
        """AT-10b (PIN/RED): list_sessions includes active_pid_stale=False when pid stable.

        Scenario: live pid on port 8100 still matches recorded active_pid=9100.
        """
        # pid from _scan_two_live for port 8100: 1000 + 8100 = 9100
        lifespan = _make_lifespan_v11(
            base_port=8100, active_port=8100, active_pid=9100,
        )
        ctx = _make_ctx(lifespan)

        with patch.object(HoudiniBridge, "health_check", _scan_two_live(8100)):
            result = await sessions_tools.houdini_list_sessions(ctx)

        assert "active_pid_stale" in result, (
            f"houdini_list_sessions v1.1 must include 'active_pid_stale'; got keys={list(result.keys())}"
        )
        assert result["active_pid_stale"] is False, (
            f"Stable pid 9100==9100 must set active_pid_stale=False; got {result['active_pid_stale']!r}"
        )
