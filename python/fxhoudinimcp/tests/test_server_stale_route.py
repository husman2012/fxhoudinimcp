"""test_server_stale_route.py — RED tests for refuse-to-route-if-stale (PP12-115d).

This is the FIRST-EVER direct test of fxhoudinimcp.server._get_bridge.  Prior
suites (test_bonedeform_tools.py, test_export_tools.py, etc.) only patch
*module-local* `_get_bridge` re-exports on tool modules — none of them
exercises server.py's own `_get_bridge` implementation.

Covers the public contract added by ADR-0006 (refuse-to-route-if-stale):

  _probe_pid(host, port, timeout=_PROBE_TIMEOUT) -> int | None
      A sync mcp.health probe (mirrors fxhoudinimcp_server.startup._query_health's
      MECHANISM, but lives in the fxhoudinimcp package because the two packages
      cannot import each other).  Returns the live pid on a well-formed health
      response; returns None on ANY failure (connection refused, timeout,
      non-dict body, malformed JSON, missing/non-int 'pid').  NEVER RAISES.

  _get_bridge(ctx) enforcement guard
      RUNTIME path only (the legacy {"bridge": <mock>} short-circuit is
      UNCHANGED and stays first).  Gated STRICTLY on
      state.get("active_pid") is not None:
        - active_pid is None            -> zero probe, zero enforcement,
                                            route exactly as before.
        - live pid == active_pid        -> pass; lazy-create the bridge if
                                            not already in state["bridges"];
                                            return it.
        - live pid != active_pid        -> raise fxhoudinimcp.errors.
                                            ConnectionError (details.reason
                                            == "drift"); NO bridge lazily
                                            created for the stale port.
        - probe returns None (gone/unreachable) -> raise ConnectionError
                                            (details.reason == "unreachable");
                                            fail-closed.

Public-contract assertions only (return values, raised type + details,
call-counts/call-args on the probe) — never asserting internals beyond what
the locked contract itself specifies (the no-probe-when-active_pid-None
invariant IS part of the public contract, so its call-count assertion is
warranted, not a mirror-test violation).

At RED time, neither `_probe_pid` nor the `_get_bridge` guard exist in
server.py — every test in this file is expected to fail with an AttributeError
or ImportError until hou-dev implements PP12-115d.

testVerificationSurface: pytest-model
unitId: pp12-115d
planSha: 481d4cd2e50214ac89626f90b0402ddd0fa865bf8464f21a141581e8df30e115
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from fxhoudinimcp import server as server_mod  # noqa: E402
from fxhoudinimcp.errors import ConnectionError as HoudiniConnectionError  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _runtime_state(
    *,
    host: str = "localhost",
    base_port: int = 8100,
    active_port: int = 8100,
    active_pid: int | None = None,
    bridges: dict | None = None,
) -> dict:
    """Build a 115b/c/d-shaped runtime lifespan state dict."""
    return {
        "host": host,
        "base_port": base_port,
        "active_port": active_port,
        "active_pid": active_pid,
        "bridges": {} if bridges is None else bridges,
    }


def _make_ctx(state: dict) -> MagicMock:
    """Build a mock MCP context whose lifespan_context is the given dict."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = state
    return ctx


def _normalize_probe_call(call_args) -> tuple:
    """Normalize a `_probe_pid` mock call (positional-or-kwargs) into a
    (host, port, timeout) triple, verifying each value lands IN ITS ROLE
    rather than merely being present somewhere in the call. Supports either
    `_probe_pid(host, port, timeout)` positionally, or `_probe_pid(host=...,
    port=..., timeout=...)` via kwargs (or a mix of the two)."""
    args, kwargs = call_args
    names = ("host", "port", "timeout")
    values: dict = {}
    for i, name in enumerate(names):
        if name in kwargs:
            values[name] = kwargs[name]
        elif i < len(args):
            values[name] = args[i]
        else:
            raise AssertionError(
                f"_probe_pid call is missing the {name!r} argument "
                f"(positional or kwarg); call={call_args!r}"
            )
    return values["host"], values["port"], values["timeout"]


def _health_response(pid: int | None = 4242, **extra) -> MagicMock:
    """Build a mock urllib response object whose .read() yields an mcp.health body."""
    body = {"status": "ok", "houdini_version": "21.0.729", **extra}
    if pid is not None:
        body["pid"] = pid
    response = MagicMock()
    response.read.return_value = json.dumps(body).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


# ===========================================================================
# Section (a)+(b) — _probe_pid: sync mcp.health probe, never raises
# ===========================================================================

class TestProbePidSuccess:
    """(a) _probe_pid returns the live int pid on a well-formed mcp.health response."""

    def test_probe_pid_returns_pid_on_success(self):
        response = _health_response(pid=54321)
        with patch(
            "fxhoudinimcp.server.urllib.request.urlopen", return_value=response
        ) as mock_urlopen:
            result = server_mod._probe_pid("localhost", 8100, timeout=0.5)
        assert result == 54321, f"_probe_pid must return the live pid; got {result!r}"

        # The mock must be MEANINGFUL: assert the probe actually hit the right
        # endpoint, not merely that *some* urlopen call happened to return our
        # canned response. Mirrors fxhoudinimcp_server.startup._query_health's
        # request-building mechanism (POST, form-encoded ["mcp.health",[],{}]
        # body, to http://{host}:{port}/api).
        mock_urlopen.assert_called_once()
        call_args, call_kwargs = mock_urlopen.call_args
        assert call_args, "_probe_pid must call urlopen with a positional Request"
        request = call_args[0]

        url = getattr(request, "full_url", None) or getattr(request, "_full_url", None)
        assert url is not None, "the first positional arg to urlopen must be a Request"
        assert "8100" in url, f"the probed URL must contain the target port; got {url!r}"
        assert "/api" in url, f"the probed URL must hit the /api endpoint; got {url!r}"

        body = getattr(request, "data", None)
        assert body is not None, "the Request must carry a POST body"
        assert b"mcp.health" in body, (
            f"the POST body must encode the mcp.health RPC call; got {body!r}"
        )

        assert call_kwargs.get("timeout") == 0.5, (
            f"_probe_pid must pass its own timeout through to urlopen; "
            f"got kwargs={call_kwargs!r}"
        )


class TestProbePidNeverRaises:
    """(b) _probe_pid returns None (NEVER raises) on every failure mode."""

    def test_probe_pid_connection_refused_returns_none(self):
        import urllib.error

        with patch(
            "fxhoudinimcp.server.urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            result = server_mod._probe_pid("localhost", 8199, timeout=0.5)
        assert result is None, "connection-refused must return None, not raise"

    def test_probe_pid_timeout_returns_none(self):
        with patch(
            "fxhoudinimcp.server.urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            result = server_mod._probe_pid("localhost", 8199, timeout=0.5)
        assert result is None, "timeout must return None, not raise"

    def test_probe_pid_non_dict_body_returns_none(self):
        response = MagicMock()
        response.read.return_value = json.dumps([1, 2, 3]).encode("utf-8")
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        with patch("fxhoudinimcp.server.urllib.request.urlopen", return_value=response):
            result = server_mod._probe_pid("localhost", 8100, timeout=0.5)
        assert result is None, "a non-dict JSON body (e.g. a list) must return None"

    def test_probe_pid_malformed_json_returns_none(self):
        response = MagicMock()
        response.read.return_value = b"{not valid json::"
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        with patch("fxhoudinimcp.server.urllib.request.urlopen", return_value=response):
            result = server_mod._probe_pid("localhost", 8100, timeout=0.5)
        assert result is None, "malformed/undecodable JSON must return None"

    def test_probe_pid_missing_pid_key_returns_none(self):
        response = _health_response(pid=None)  # omits 'pid' entirely
        with patch("fxhoudinimcp.server.urllib.request.urlopen", return_value=response):
            result = server_mod._probe_pid("localhost", 8100, timeout=0.5)
        assert result is None, "a health dict missing the 'pid' key must return None"

    def test_probe_pid_non_int_pid_returns_none(self):
        response = _health_response(pid="not-an-int")
        with patch("fxhoudinimcp.server.urllib.request.urlopen", return_value=response):
            result = server_mod._probe_pid("localhost", 8100, timeout=0.5)
        assert result is None, "a non-int 'pid' value must return None"

    def test_probe_pid_read_raises_returns_none(self):
        """The try/except must wrap the read()+decode step, not just urlopen()
        itself -- a response whose .read() raises must still return None,
        never propagate the exception."""
        response = MagicMock()
        response.read.side_effect = OSError("read failed")
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        with patch("fxhoudinimcp.server.urllib.request.urlopen", return_value=response):
            result = server_mod._probe_pid("localhost", 8100, timeout=0.5)
        assert result is None, "a response.read() failure must return None, not raise"


# ===========================================================================
# Section (c) — active_pid is None: ZERO enforcement, ZERO probe
# ===========================================================================

class TestGetBridgeNoBaselineNoProbe:
    """(c) active_pid=None -> _get_bridge returns the bridge and NEVER probes.

    This is the single most load-bearing invariant in the whole unit: it
    keeps single-session (never-select_session) use byte-identical to before
    115d, with zero added latency.  The assertion is on CALL-COUNT, not just
    the result — because the contract explicitly requires "zero probe", not
    merely "does not raise".
    """

    def test_active_pid_none_returns_bridge_without_probing(self):
        state = _runtime_state(active_port=8100, active_pid=None, bridges={})
        ctx = _make_ctx(state)

        with patch.object(server_mod, "_probe_pid") as mock_probe:
            bridge = server_mod._get_bridge(ctx)

        assert mock_probe.call_count == 0, (
            "active_pid=None must NEVER call _probe_pid; "
            f"got {mock_probe.call_count} call(s)"
        )
        assert bridge is not None
        assert state["bridges"][8100] is bridge


# ===========================================================================
# Section (d) — matching live pid + active_port NOT yet in bridges: lazy-create
# ===========================================================================

class TestGetBridgeMatchingPidLazyCreates:
    """(d) active_pid set + active_port absent from bridges + matching probe
    -> probes, passes, LAZILY CREATES the bridge, returns it, no raise.

    This is the real selected-non-base path: select_session updates
    active_port/active_pid but does NOT itself create a bridge entry.
    """

    def test_matching_pid_lazily_creates_and_returns_bridge(self):
        state = _runtime_state(
            host="localhost", active_port=8101, active_pid=9101, bridges={}
        )
        ctx = _make_ctx(state)

        assert 8101 not in state["bridges"], "precondition: port not yet in bridges"

        with patch.object(server_mod, "_probe_pid", return_value=9101) as mock_probe:
            bridge = server_mod._get_bridge(ctx)

        mock_probe.assert_called_once()
        # The probe must be called with the ACTIVE port (8101) IN THE PORT
        # ROLE -- not merely present somewhere in the call (a guard that
        # probes the wrong port, or passes 8101 in the host slot, must fail).
        # Reuse the role-normalizer that case (h) uses (codex red-review R2).
        host, port, timeout = _normalize_probe_call(mock_probe.call_args)
        assert port == 8101, (
            "_probe_pid must be called with the ACTIVE port (8101) in the port "
            f"role, not base_port or any other value; call={mock_probe.call_args!r}"
        )
        assert host == "localhost", (
            f"_probe_pid must be called with state['host']; call={mock_probe.call_args!r}"
        )
        assert timeout == server_mod._PROBE_TIMEOUT, (
            f"_probe_pid must be called with _PROBE_TIMEOUT; call={mock_probe.call_args!r}"
        )
        assert 8101 in state["bridges"], (
            "a matching probe must lazily create the bridge for active_port"
        )
        assert state["bridges"][8101] is bridge


# ===========================================================================
# Section (e) — drifted pid: raise BEFORE lazy-create (bridges unchanged)
# ===========================================================================

class TestGetBridgeDriftedPidRaises:
    """(e) drifted live pid -> raise ConnectionError(reason='drift') and the
    stale port's bridge is NEVER lazily created (raise precedes lazy-create)."""

    def test_drifted_pid_raises_connection_error_with_drift_reason(self):
        state = _runtime_state(
            host="localhost", active_port=8100, active_pid=5000, bridges={}
        )
        ctx = _make_ctx(state)

        with patch.object(server_mod, "_probe_pid", return_value=9999):
            with pytest.raises(HoudiniConnectionError) as excinfo:
                server_mod._get_bridge(ctx)

        details = excinfo.value.details
        assert details.get("port") == 8100
        assert details.get("expected_pid") == 5000
        assert details.get("live_pid") == 9999
        assert details.get("reason") == "drift"

    def test_drifted_pid_does_not_lazily_create_bridge(self):
        state = _runtime_state(
            host="localhost", active_port=8100, active_pid=5000, bridges={}
        )
        ctx = _make_ctx(state)

        with patch.object(server_mod, "_probe_pid", return_value=9999):
            with pytest.raises(HoudiniConnectionError):
                server_mod._get_bridge(ctx)

        assert 8100 not in state["bridges"], (
            "a drift-refuse must NOT lazily create a bridge for the stale port; "
            f"bridges={state['bridges']!r}"
        )


# ===========================================================================
# Section (f) — probe returns None (gone/unreachable): fail-closed
# ===========================================================================

class TestGetBridgeUnreachableRaises:
    """(f) probe returns None -> raise ConnectionError(reason='unreachable')
    (fail-closed — an inconclusive probe is treated as gone)."""

    def test_probe_none_raises_connection_error_with_unreachable_reason(self):
        state = _runtime_state(
            host="localhost", active_port=8100, active_pid=5000, bridges={}
        )
        ctx = _make_ctx(state)

        with patch.object(server_mod, "_probe_pid", return_value=None):
            with pytest.raises(HoudiniConnectionError) as excinfo:
                server_mod._get_bridge(ctx)

        details = excinfo.value.details
        assert details.get("port") == 8100
        assert details.get("expected_pid") == 5000
        assert details.get("live_pid") is None
        assert details.get("reason") == "unreachable"

    def test_probe_none_does_not_lazily_create_bridge(self):
        state = _runtime_state(
            host="localhost", active_port=8100, active_pid=5000, bridges={}
        )
        ctx = _make_ctx(state)

        with patch.object(server_mod, "_probe_pid", return_value=None):
            with pytest.raises(HoudiniConnectionError):
                server_mod._get_bridge(ctx)

        assert 8100 not in state["bridges"]


# ===========================================================================
# Section (g) — legacy {"bridge": <mock>} short-circuit: FIRST, untouched
# ===========================================================================

class TestGetBridgeLegacyShortCircuit:
    """(g) the legacy {"bridge": <mock>} lifespan-context shape returns the
    mock unmodified regardless of any other state keys present, and the
    probe is NEVER called (invariant 2 — legacy path first + untouched)."""

    def test_legacy_shape_returns_mock_bridge_unmodified(self):
        legacy_bridge = MagicMock(name="legacy-bridge")
        ctx = _make_ctx({"bridge": legacy_bridge})

        with patch.object(server_mod, "_probe_pid") as mock_probe:
            result = server_mod._get_bridge(ctx)

        assert result is legacy_bridge
        assert mock_probe.call_count == 0, (
            "the legacy short-circuit must never invoke the probe"
        )

    def test_legacy_shape_ignores_other_state_keys(self):
        """Even if other 115-shaped keys are present, 'bridge' short-circuits first."""
        legacy_bridge = MagicMock(name="legacy-bridge")
        state = {
            "bridge": legacy_bridge,
            "host": "localhost",
            "active_port": 8100,
            "active_pid": 9999,  # would normally trigger enforcement
            "bridges": {},
        }
        ctx = _make_ctx(state)

        with patch.object(server_mod, "_probe_pid") as mock_probe:
            result = server_mod._get_bridge(ctx)

        assert result is legacy_bridge
        assert mock_probe.call_count == 0, (
            "presence of 'bridge' must short-circuit before any active_pid check"
        )


# ===========================================================================
# Section (h) — probe is invoked with state['host'] AND timeout==_PROBE_TIMEOUT
# ===========================================================================

class TestGetBridgeProbeWiring:
    """(h) the borrowed timeout + host are at least wired through to the probe."""

    def test_probe_called_with_host_and_probe_timeout_constant(self):
        state = _runtime_state(
            host="myhost.example", active_port=8100, active_pid=5000, bridges={}
        )
        ctx = _make_ctx(state)

        with patch.object(server_mod, "_probe_pid", return_value=5000) as mock_probe:
            server_mod._get_bridge(ctx)

        mock_probe.assert_called_once()
        host, port, timeout = _normalize_probe_call(mock_probe.call_args)
        assert host == "myhost.example", (
            f"_probe_pid must be called with state['host'] AS THE HOST ARG; "
            f"got host={host!r}; call={mock_probe.call_args!r}"
        )
        assert port == 8100, (
            f"_probe_pid must be called with the active port AS THE PORT ARG; "
            f"got port={port!r}; call={mock_probe.call_args!r}"
        )
        assert timeout == server_mod._PROBE_TIMEOUT, (
            f"_probe_pid must be called with the _PROBE_TIMEOUT module constant "
            f"AS THE TIMEOUT ARG; got timeout={timeout!r}; call={mock_probe.call_args!r}"
        )


# ===========================================================================
# Section (i) — DOCUMENTED RESIDUAL R1: pidless selection -> no enforcement
# ===========================================================================

class TestGetBridgePidlessSelectionResidual:
    """(i) a 'selected' session whose active_pid is None (pidless health) ->
    _get_bridge does NOT enforce (pins the KNOWN narrow-contract behaviour;
    ADR-0006 contract residual R1).  Identical assertion shape to case (c) —
    kept as its own test to pin the residual explicitly, independent of the
    "never selected at all" scenario."""

    def test_pidless_selected_session_skips_enforcement(self):
        # A session was "selected" (active_port != base_port, distinguishing
        # this from a truly-untouched lifespan) but active_pid is None because
        # the health entry at select-time lacked a numeric 'pid' key.
        state = _runtime_state(
            host="localhost", base_port=8100, active_port=8101, active_pid=None,
            bridges={},
        )
        ctx = _make_ctx(state)

        with patch.object(server_mod, "_probe_pid") as mock_probe:
            bridge = server_mod._get_bridge(ctx)

        assert mock_probe.call_count == 0, (
            "a selected-but-pidless session must not trigger enforcement "
            "(R1 — documented residual)"
        )
        assert bridge is not None
        assert state["bridges"][8101] is bridge
