"""
test_session_model.py — pure-logic tests for PP12-115b session_model.py.

This is the RED phase: fxhoudinimcp.session_model does NOT exist yet.
Import will fail with ModuleNotFoundError => all tests in this file are RED.

Covers the public contract of:
  - build_session_entry(port, health_dict) -> dict
      Maps a health probe result + port into a clean session entry.
      Carries: port, pid, hip_file, houdini_version, and any additive
      fields (gate_mode) present in the health dict.
      Does NOT carry: 'status' key (omitted).
  - mark_active(sessions, active_port) -> list[dict]
      Returns a new list with 'active' True on the entry whose port ==
      active_port, False on all others.
  - resolve_target(sessions, target) -> int | None
      target is an int  -> return matching port, or None if absent.
      target is a str   -> case-insensitive substring match on hip_file.
                           Return port if EXACTLY ONE match.
                           Return None if ZERO or MULTIPLE matches (ambiguous).

TDD phase: RED
testVerificationSurface: pytest-model
unitId: pp12-115b
Linear: HMDNI-116

No hou / Qt / pxr imports anywhere in this file.
Runs under plain pytest headless (off-DCC, no Houdini install required).
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — allow running standalone as well as via pytest.
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import pytest

# ---------------------------------------------------------------------------
# Module under test — does NOT exist yet; import failure = RED.
# ---------------------------------------------------------------------------
from fxhoudinimcp.session_model import (  # noqa: E402
    build_session_entry,
    mark_active,
    resolve_target,
)


# ===========================================================================
# Shared fixtures / helpers
# ===========================================================================

def _health(port: int = 8100, **extra) -> dict:
    """Minimal health-check dict returned by HoudiniBridge.health_check()."""
    return {
        "status": "ok",
        "pid": 1000 + port,
        "hip_file": f"C:/scenes/scene_{port}.hip",
        "houdini_version": "21.0.729",
        **extra,
    }


def _two_sessions() -> list[dict]:
    """A canonical two-session list used across multiple tests."""
    return [
        build_session_entry(8100, _health(8100, gate_mode="propose")),
        build_session_entry(8101, _health(8101)),
    ]


# ===========================================================================
# Section 1 — build_session_entry(port, health_dict) -> dict
#
# Locked contract (plan pp12-115b acceptanceTests):
#   build_session_entry(
#       8101,
#       {'status':'ok','pid':222,'hip_file':'C:/x/asset.hip',
#        'houdini_version':'21.0.729','gate_mode':'propose'}
#   ) == {
#       'port':8101, 'pid':222, 'hip_file':'C:/x/asset.hip',
#       'houdini_version':'21.0.729', 'gate_mode':'propose'
#   }
#
# - 'status' is OMITTED from the result.
# - All other health dict keys are CARRIED.
# - 'port' is ADDED from the port argument.
# - 'gate_mode' only appears in result if present in health dict.
# ===========================================================================

class TestBuildSessionEntry:
    """build_session_entry — locked field shape, status omitted, port added."""

    def test_full_with_gate_mode(self):
        """PIN (plan AT-1): full health dict with gate_mode maps to session entry.

        Input:  port=8101, health={status,pid,hip_file,houdini_version,gate_mode}
        Output: {port,pid,hip_file,houdini_version,gate_mode}  — no 'status'.
        """
        health = {
            "status": "ok",
            "pid": 222,
            "hip_file": "C:/x/asset.hip",
            "houdini_version": "21.0.729",
            "gate_mode": "propose",
        }
        result = build_session_entry(8101, health)
        assert result == {
            "port": 8101,
            "pid": 222,
            "hip_file": "C:/x/asset.hip",
            "houdini_version": "21.0.729",
            "gate_mode": "propose",
        }, f"Expected pinned result, got {result!r}"

    def test_status_omitted(self):
        """'status' key must NOT appear in the session entry."""
        entry = build_session_entry(8100, _health(8100))
        assert "status" not in entry, (
            f"'status' must be omitted from session entry; keys={list(entry.keys())}"
        )

    def test_port_is_added(self):
        """'port' must equal the port argument, regardless of health dict content."""
        entry = build_session_entry(8102, _health(8100))  # health port differs
        assert entry["port"] == 8102

    def test_pid_carried(self):
        """'pid' from health dict is carried into the entry."""
        entry = build_session_entry(8100, _health(8100))
        assert entry["pid"] == 9100  # _health(8100) -> pid = 1000 + 8100 = 9100

    def test_pid_carried_value(self):
        """'pid' exact value from health dict is preserved."""
        health = {**_health(8100), "pid": 42}
        entry = build_session_entry(8100, health)
        assert entry["pid"] == 42

    def test_hip_file_carried(self):
        """'hip_file' from health dict is carried into the entry."""
        health = {**_health(8100), "hip_file": "D:/renders/forest.hip"}
        entry = build_session_entry(8100, health)
        assert entry["hip_file"] == "D:/renders/forest.hip"

    def test_houdini_version_carried(self):
        """'houdini_version' from health dict is carried into the entry."""
        health = {**_health(8100), "houdini_version": "21.5.100"}
        entry = build_session_entry(8100, health)
        assert entry["houdini_version"] == "21.5.100"

    def test_gate_mode_carried_when_present(self):
        """'gate_mode' is carried when present in the health dict."""
        health = {**_health(8100), "gate_mode": "propose"}
        entry = build_session_entry(8100, health)
        assert entry.get("gate_mode") == "propose"

    def test_gate_mode_absent_when_not_in_health(self):
        """'gate_mode' must NOT appear in the entry when absent from health dict."""
        health = _health(8100)  # no gate_mode
        assert "gate_mode" not in health
        entry = build_session_entry(8100, health)
        assert "gate_mode" not in entry, (
            "'gate_mode' must be absent from entry when not in health dict"
        )

    def test_pending_count_carried_when_present(self):
        """Additive health fields beyond gate_mode are also carried (e.g. pending_count)."""
        health = {**_health(8100), "gate_mode": "propose", "pending_count": 3}
        entry = build_session_entry(8100, health)
        assert entry.get("pending_count") == 3

    def test_result_is_dict(self):
        """build_session_entry returns a dict."""
        entry = build_session_entry(8100, _health(8100))
        assert isinstance(entry, dict)

    def test_different_ports_produce_independent_entries(self):
        """Two calls with different ports produce independent entries."""
        e1 = build_session_entry(8100, _health(8100))
        e2 = build_session_entry(8101, _health(8101))
        assert e1["port"] == 8100
        assert e2["port"] == 8101
        assert e1 is not e2


# ===========================================================================
# Section 2 — mark_active(sessions, active_port) -> list[dict]
#
# Locked contract (plan pp12-115b acceptanceTests):
#   mark_active([{'port':8100,...}, {'port':8101,...}], 8101)
#   -> [{'port':8100,...,'active':False}, {'port':8101,...,'active':True}]
#
# - Returns a new list (does not mutate input).
# - Sets 'active'=True on the entry whose port == active_port.
# - Sets 'active'=False on all other entries.
# - If active_port is not in the list, all entries get active=False.
# ===========================================================================

class TestMarkActive:
    """mark_active — active flag set on matching port; False on others."""

    def test_pin_mark_active(self):
        """PIN (plan AT-2): mark_active sets active=True on 8101, False on 8100."""
        sessions = _two_sessions()
        result = mark_active(sessions, 8101)
        ports = {e["port"]: e["active"] for e in result}
        assert ports[8101] is True, f"port 8101 must be active=True, got {ports[8101]!r}"
        assert ports[8100] is False, f"port 8100 must be active=False, got {ports[8100]!r}"

    def test_single_session_becomes_active(self):
        """A list of one session gets active=True when its port matches."""
        sessions = [build_session_entry(8100, _health(8100))]
        result = mark_active(sessions, 8100)
        assert result[0]["active"] is True

    def test_all_others_are_false(self):
        """All entries except the active one have active=False."""
        sessions = [
            build_session_entry(8100, _health(8100)),
            build_session_entry(8101, _health(8101)),
            build_session_entry(8102, _health(8102)),
        ]
        result = mark_active(sessions, 8101)
        for e in result:
            if e["port"] == 8101:
                assert e["active"] is True
            else:
                assert e["active"] is False

    def test_returns_new_list(self):
        """mark_active returns a new list object, not the input list."""
        sessions = _two_sessions()
        result = mark_active(sessions, 8100)
        assert result is not sessions

    def test_does_not_mutate_input(self):
        """mark_active does not modify the input session dicts."""
        sessions = _two_sessions()
        # capture state before
        had_active = ["active" in e for e in sessions]
        mark_active(sessions, 8101)
        after = ["active" in e for e in sessions]
        # if original dicts didn't have 'active', they still shouldn't
        # (or if they did, value should be unchanged — either way, a new list is returned)
        assert had_active == after, "Input session dicts must not be mutated"

    def test_absent_port_all_false(self):
        """When active_port not in sessions, all entries get active=False."""
        sessions = _two_sessions()
        result = mark_active(sessions, 9999)
        assert all(e["active"] is False for e in result)

    def test_empty_sessions(self):
        """mark_active with an empty list returns an empty list."""
        result = mark_active([], 8100)
        assert result == []

    def test_returns_list(self):
        """mark_active always returns a list."""
        result = mark_active(_two_sessions(), 8100)
        assert isinstance(result, list)

    def test_active_field_type_is_bool(self):
        """'active' field must be a Python bool, not an int or truthy."""
        result = mark_active(_two_sessions(), 8100)
        for e in result:
            assert isinstance(e["active"], bool), (
                f"'active' must be bool, got {type(e['active']).__name__!r} "
                f"for port {e['port']}"
            )

    def test_preserves_all_other_fields(self):
        """mark_active must not drop existing fields from entries."""
        sessions = [
            {**build_session_entry(8100, _health(8100, gate_mode="propose")), "hip_file": "scene.hip"},
        ]
        result = mark_active(sessions, 8100)
        assert result[0].get("hip_file") == "scene.hip"
        assert result[0].get("gate_mode") == "propose"


# ===========================================================================
# Section 3 — resolve_target(sessions, target) -> int | None
#
# Locked contract (plan pp12-115b acceptanceTests):
#
# Integer target:
#   resolve_target(sessions, 8101) -> 8101   if port 8101 is in sessions
#   resolve_target(sessions, 9999) -> None    if port 9999 is not in sessions
#
# String target (hip_file case-insensitive substring):
#   resolve_target(sessions, 'asset') -> port  if EXACTLY ONE entry's
#       hip_file contains 'asset' (case-insensitive)
#   resolve_target(sessions, 'scene') -> None  if ZERO matches
#   resolve_target(sessions, 'hip')   -> None  if MULTIPLE matches (ambiguous)
#
# CRITICAL: Zero match AND multiple match both return None.
#   The agent must call list_sessions to disambiguate; never silently pick first.
# ===========================================================================

class TestResolveTarget:
    """resolve_target — int lookup by port, str lookup by hip_file substring."""

    def _sessions(self) -> list[dict]:
        return [
            build_session_entry(8100, {
                **_health(8100), "hip_file": "C:/scenes/terrain.hip",
            }),
            build_session_entry(8101, {
                **_health(8101), "hip_file": "C:/work/asset_hero.hip",
            }),
        ]

    # -----------------------------------------------------------------------
    # Integer target — port lookup
    # -----------------------------------------------------------------------

    def test_pin_int_present(self):
        """PIN (plan AT-3a): resolve_target(sessions, 8101) -> 8101 when present."""
        sessions = self._sessions()
        result = resolve_target(sessions, 8101)
        assert result == 8101, f"Expected 8101, got {result!r}"

    def test_pin_int_absent(self):
        """PIN (plan AT-3b): resolve_target(sessions, 9999) -> None when absent."""
        result = resolve_target(self._sessions(), 9999)
        assert result is None, f"Expected None for absent port, got {result!r}"

    def test_int_matches_first_port(self):
        """resolve_target finds the first port (8100) by integer."""
        result = resolve_target(self._sessions(), 8100)
        assert result == 8100

    def test_int_empty_list_returns_none(self):
        """resolve_target on empty list always returns None."""
        assert resolve_target([], 8100) is None

    def test_int_returns_int_or_none(self):
        """resolve_target on integer target returns an int (port) or None."""
        result = resolve_target(self._sessions(), 8100)
        assert result is None or isinstance(result, int)

    # -----------------------------------------------------------------------
    # String target — hip_file substring match
    # -----------------------------------------------------------------------

    def test_pin_str_single_match(self):
        """PIN (plan AT-3c): resolve_target(sessions, 'asset') -> 8101 (single match)."""
        result = resolve_target(self._sessions(), "asset")
        assert result == 8101, f"Expected 8101 for 'asset', got {result!r}"

    def test_pin_str_no_match_returns_none(self):
        """PIN (plan AT-3d): resolve_target(sessions, 'nonexistent') -> None (zero matches)."""
        result = resolve_target(self._sessions(), "nonexistent")
        assert result is None, f"Expected None for no-match substring, got {result!r}"

    def test_pin_str_ambiguous_returns_none(self):
        """PIN (plan AT-3e): resolve_target(sessions, 'hip') -> None (multiple matches).

        CRITICAL anti-pattern: must NOT silently return the first match.
        Both hip_file values contain '.hip' — multiple match = None.
        """
        # Both entries have '.hip' in their hip_file paths.
        result = resolve_target(self._sessions(), ".hip")
        assert result is None, (
            f"Ambiguous match must return None, not the first match; got {result!r}"
        )

    def test_str_case_insensitive_upper(self):
        """Case-insensitive: 'ASSET' matches 'asset_hero.hip'."""
        result = resolve_target(self._sessions(), "ASSET")
        assert result == 8101

    def test_str_case_insensitive_mixed(self):
        """Case-insensitive: 'Asset' matches 'asset_hero.hip'."""
        result = resolve_target(self._sessions(), "Asset")
        assert result == 8101

    def test_str_case_insensitive_terrain(self):
        """Case-insensitive: 'TERRAIN' matches 'terrain.hip'."""
        result = resolve_target(self._sessions(), "TERRAIN")
        assert result == 8100

    def test_str_partial_match_single(self):
        """Partial substring 'hero' matches exactly one hip_file."""
        result = resolve_target(self._sessions(), "hero")
        assert result == 8101

    def test_str_partial_match_multi_is_none(self):
        """Partial substring matching multiple hip_files returns None (ambiguous)."""
        sessions = [
            build_session_entry(8100, {**_health(8100), "hip_file": "C:/work/asset_A.hip"}),
            build_session_entry(8101, {**_health(8101), "hip_file": "C:/work/asset_B.hip"}),
        ]
        result = resolve_target(sessions, "asset")
        assert result is None, (
            f"Multiple matches on 'asset' must return None; got {result!r}"
        )

    def test_str_exact_hip_filename(self):
        """Full hip_file path works as substring (exact match = single match = ok)."""
        result = resolve_target(self._sessions(), "C:/work/asset_hero.hip")
        assert result == 8101

    def test_str_returns_int_port_on_match(self):
        """resolve_target returns an int (port) on a string single match."""
        result = resolve_target(self._sessions(), "terrain")
        assert isinstance(result, int), f"Expected int port, got {type(result).__name__!r}"

    def test_str_empty_string_multi_is_none(self):
        """Empty string matches all hip_files (multiple match) -> None."""
        result = resolve_target(self._sessions(), "")
        assert result is None, (
            "Empty string matches all hip_files — ambiguous — must return None"
        )

    def test_str_empty_sessions(self):
        """String lookup on empty list returns None (zero match)."""
        assert resolve_target([], "asset") is None


# ===========================================================================
# Section 4 — hou-free import verification (CL-015)
#
# session_model.py must import with zero hou/Qt/pxr at module top-level.
# ===========================================================================

class TestHouFreeImport:
    """Confirm session_model carries no hou/Qt/pxr dependency (CL-015)."""

    def test_module_importable_without_hou(self):
        """session_model must load under plain Python with no Houdini installed."""
        import fxhoudinimcp.session_model as sm
        assert sm is not None

    def test_hou_not_imported(self):
        """session_model must not reference 'import hou' at top-level."""
        import fxhoudinimcp.session_model as sm
        import inspect
        import re
        source = inspect.getsource(sm)
        source_no_comments = re.sub(r"#.*", "", source)
        assert "import hou" not in source_no_comments, (
            "session_model.py must not import hou (CL-015 — pure-logic boundary)"
        )
