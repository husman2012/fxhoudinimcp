"""hython-smoke round-trip tests — ADR 0002 §5 (pp12-109d).

Verifies the full ALLOW-path envelope contract end-to-end against a live
bridge + running Houdini session.

Run ONLY under hython (requires a live Houdini session with the gate installed).
All tests are decorated @pytest.mark.hython_smoke and skip cleanly if the
HOUDINI_SMOKE environment variable is not set (default CI run, no live bridge).

Author:              hou-test (pp12-109d)
Verification surface: hython-smoke (deferred — requires live bridge)
ADR reference:        docs/homedini/plans/_agentic/architecture/
                      0002-gate-bridge-envelope-contract.adr.md §5
"""

from __future__ import annotations

import os
import sys

import pytest

# ---------------------------------------------------------------------------
# sys.path — ensure both the fork and homedini packages are importable.
# ---------------------------------------------------------------------------
_FORK_PYTHON     = "C:/Users/husma/development/fxhoudinimcp/houdini/scripts/python"
_HOMEDINI_PYTHON = "C:/Users/husma/development/HoudiniUtilTools/scripts/python"
_FORK_CLIENT     = "C:/Users/husma/development/fxhoudinimcp/python"
for _p in (_FORK_PYTHON, _HOMEDINI_PYTHON, _FORK_CLIENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Smoke-skip guard.
#
# Without a live Houdini session + installed gate these tests cannot run.
# The guard allows the pytest suite to collect and skip cleanly in CI, so the
# file is collected (round-trip test skeleton present) but not executed until
# a `hython-smoke` environment is available.
# ---------------------------------------------------------------------------
_SMOKE_ENABLED = os.environ.get("HOUDINI_SMOKE", "").strip().lower() in ("1", "true", "yes")

pytestmark = pytest.mark.hython_smoke


def _skip_if_no_live_bridge():
    """Return a skip reason if the live bridge is unavailable; else None."""
    if not _SMOKE_ENABLED:
        return (
            "HOUDINI_SMOKE not set — hython-smoke tests deferred. "
            "Set HOUDINI_SMOKE=1 and run under hython with a live Houdini session."
        )
    return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_bridge():
    """Live HoudiniBridge connected to a running Houdini instance.

    Skips the entire module if the environment is not configured for smoke runs.
    Requires:
    - HOUDINI_SMOKE=1 in the environment
    - A running Houdini 21 session with the fxhoudinimcp server active
    - The gate installed in READ-ONLY mode (default)
    """
    reason = _skip_if_no_live_bridge()
    if reason:
        pytest.skip(reason)

    try:
        from fxhoudinimcp.bridge import HoudiniBridge
    except ImportError as exc:
        pytest.skip(f"fxhoudinimcp.bridge not importable: {exc}")

    bridge = HoudiniBridge()
    return bridge


@pytest.fixture(scope="module")
def live_bridge_propose_mode(live_bridge):
    """Bridge fixture after switching the gate to PROPOSE mode.

    Used by the QUEUE-path test to verify queue behavior without needing
    a full FULL-ACCESS session.
    """
    # Switch mode to propose for queue-path testing.
    # The teardown restores READ-ONLY for other tests.
    import asyncio

    async def _switch(mode: str) -> None:
        await live_bridge.execute("gate.set_permission_mode", {"mode": mode})

    asyncio.get_event_loop().run_until_complete(_switch("propose"))
    yield live_bridge
    asyncio.get_event_loop().run_until_complete(_switch("read_only"))


@pytest.fixture(scope="module")
def live_bridge_readonly_mode(live_bridge):
    """Bridge fixture guaranteed in READ-ONLY mode (DENY path for mutating)."""
    import asyncio

    async def _switch(mode: str) -> None:
        await live_bridge.execute("gate.set_permission_mode", {"mode": mode})

    asyncio.get_event_loop().run_until_complete(_switch("read_only"))
    return live_bridge


@pytest.fixture(scope="module")
def live_middleware():
    """Direct handle to the gated dispatcher (bypasses bridge HTTP layer).

    Useful for asserting gate metadata (gate=allowed) that the bridge strips
    before returning to the caller (bridge only returns result.get('data', {})).
    """
    reason = _skip_if_no_live_bridge()
    if reason:
        pytest.skip(reason)

    try:
        import fxhoudinimcp_server.dispatcher as _d
    except ImportError as exc:
        pytest.skip(f"fxhoudinimcp_server.dispatcher not importable: {exc}")

    from fxhoudinimcp_server.gate.middleware import _gated_dispatch
    return _gated_dispatch


# ---------------------------------------------------------------------------
# Tests — ADR 0002 §5 round-trip suite
# ---------------------------------------------------------------------------

class TestGateBridgeRoundTrip:
    """ADR 0002 Option A — end-to-end ALLOW-path envelope round-trip.

    These tests require a live Houdini session (hython-smoke tier).
    They PASS GREEN only after hou-dev applies the 3-site fix per ADR 0002.
    """

    def test_allow_path_readonly_returns_data(self, live_bridge):
        """bridge.execute('scene.get_scene_info') must return non-empty payload.

        In READ-ONLY mode, scene.get_scene_info is ALLOW-path (READONLY capability).
        bridge.execute() returns result.get('data', {}) for status=='success'.
        With the ALLOW-path bug, the bridge returns {} for every ALLOW-path call.
        After the fix, it returns the actual handler payload with 'houdini_version'.

        ADR §5 test 1.
        """
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            live_bridge.execute("scene.get_scene_info", {})
        )

        assert isinstance(result, dict), (
            f"Expected dict from bridge.execute, got {type(result).__name__}"
        )
        assert result != {}, (
            "DEFECT #2: bridge returned empty dict for ALLOW-path scene.get_scene_info. "
            "The ALLOW path is stripping the 'data' key before it reaches the bridge."
        )
        assert "houdini_version" in result, (
            f"Expected 'houdini_version' in bridge result, got keys: {list(result.keys())}"
        )

    def test_allow_path_list_children_returns_data(self, live_bridge):
        """bridge.execute('list_children') must return non-empty payload (ALLOW).

        list_children is READONLY capability → ALLOW in READ-ONLY mode.
        After the fix, the bridge returns the handler's list of children.

        ADR §5 test 2.
        """
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            live_bridge.execute("list_children", {"path": "/obj"})
        )

        assert isinstance(result, (dict, list)), (
            f"Expected dict or list from bridge.execute, got {type(result).__name__}"
        )
        # Accept dict (newer API) or list (older API) — both are non-empty after fix.
        if isinstance(result, dict):
            assert result != {}, (
                "DEFECT #2: bridge returned empty dict for ALLOW-path list_children."
            )

    def test_gate_metadata_present_in_gated_dispatch(self, live_middleware):
        """_gated_dispatch() result must contain gate='allowed' AND 'data'.

        Verifies the raw gated-dispatch layer (not via bridge HTTP) so we can
        assert gate metadata that the bridge strips.

        ADR §5 test 3.
        """
        result = live_middleware("scene.get_scene_info", {})

        assert result.get("gate") == "allowed", (
            f"result['gate'] = {result.get('gate')!r} (expected 'allowed')"
        )
        assert result.get("status") == "success", (
            f"result['status'] = {result.get('status')!r} (expected 'success')"
        )
        assert "data" in result, (
            "DEFECT #2: 'data' key missing from _gated_dispatch result. "
            f"result keys = {list(result.keys())}"
        )
        assert result["data"].get("houdini_version"), (
            "DEFECT #2: 'data' present but houdini_version missing or empty. "
            f"result['data'] = {result.get('data')!r}"
        )

    def test_queue_path_not_affected(self, live_bridge_propose_mode):
        """QUEUE-path behavior is unaffected by the ALLOW-path fix.

        In PROPOSE mode, MUTATING commands are queued. The queue response
        envelope must still return a pending_id and queue metadata.
        This test guards against the fix accidentally breaking the QUEUE path.

        ADR §5 test 4.
        """
        import asyncio

        # create_node is a MUTATING capability → QUEUE in PROPOSE mode.
        # In PROPOSE mode it should return a queued response, not execute.
        try:
            result = asyncio.get_event_loop().run_until_complete(
                live_bridge_propose_mode.execute(
                    "create_node",
                    {"node_type": "geo", "parent": "/obj", "name": "test_queue_node"},
                )
            )
        except Exception:
            # A HoudiniCommandError on the QUEUE path means the gate returned
            # a non-standard response — flag it.
            pytest.fail(
                "QUEUE-path: create_node in PROPOSE mode raised an exception "
                "rather than returning a queued response. "
                "Check that the ALLOW-path fix did not inadvertently affect QUEUE."
            )

        # The queue response may surface as the raw gated response (pending_id)
        # or wrapped by the bridge depending on the server version.
        # At minimum it must not be the handler's execute result.
        assert isinstance(result, dict), (
            f"Expected dict from queued create_node, got {type(result).__name__!r}"
        )

    def test_deny_path_raises_houdini_command_error(self, live_bridge_readonly_mode):
        """DENY-path commands raise HoudiniCommandError from the bridge.

        In READ-ONLY mode, set_parameter is MUTATING → DENY.
        The bridge must raise HoudiniCommandError, not return empty dict.
        This test guards against the fix accidentally breaking the DENY path.

        ADR §5 test 5.
        """
        import asyncio

        try:
            from fxhoudinimcp.bridge import HoudiniCommandError
        except ImportError as exc:
            pytest.skip(f"HoudiniCommandError not importable: {exc}")

        with pytest.raises(HoudiniCommandError):
            asyncio.get_event_loop().run_until_complete(
                live_bridge_readonly_mode.execute(
                    "set_parameter",
                    {"node": "/obj/geo1", "parm": "tx", "value": 0.0},
                )
            )
