"""Tests for Houdini-side startup health checks."""

from __future__ import annotations

# Built-in
import os
import sys
import threading
import time
from unittest.mock import MagicMock

# Third-party
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "houdini", "scripts", "python"),
)

from fxhoudinimcp_server import startup  # noqa: E402


@pytest.fixture(autouse=True)
def reset_startup_state(monkeypatch):
    monkeypatch.setattr(startup, "_server_started", False)
    monkeypatch.setattr(startup, "_port", 8100)
    # Reset _starting guard; raising=False so this works before impl adds the attribute.
    monkeypatch.setattr(startup, "_starting", False, raising=False)


def test_wait_for_current_process_health_accepts_current_pid(monkeypatch):
    monkeypatch.setattr(
        startup,
        "_query_health",
        lambda port: {
            "status": "ok",
            "pid": os.getpid(),
            "houdini_version": "21.0.631",
        },
    )

    health = startup._wait_for_current_process_health(8100)

    assert health is not None
    assert health["pid"] == os.getpid()


def test_ensure_running_restarts_when_cached_state_is_stale(monkeypatch):
    calls = []
    monkeypatch.setattr(startup, "_server_started", True)
    monkeypatch.setattr(
        startup,
        "_wait_for_current_process_health",
        lambda port, timeout_seconds=0.5: None,
    )
    monkeypatch.setattr(startup, "start", lambda: calls.append("start"))

    startup.ensure_running()

    assert calls == ["start"]


def test_ensure_running_keeps_live_server(monkeypatch):
    calls = []
    monkeypatch.setattr(startup, "_server_started", True)
    monkeypatch.setattr(
        startup,
        "_wait_for_current_process_health",
        lambda port, timeout_seconds=0.5: {"pid": os.getpid()},
    )
    monkeypatch.setattr(startup, "start", lambda: calls.append("start"))

    startup.ensure_running()

    assert calls == []


# ---------------------------------------------------------------------------
# RED tests for fxstartup-async-health
# All four tests below will FAIL against the current startup.py because:
#   - _is_graphical_session() does not exist → AttributeError on monkeypatch.setattr
#   - _starting flag does not exist (fixture uses raising=False; start() itself has no guard)
#   - start() has no graphical branch; it always calls _wait_for_current_process_health
#     synchronously on the calling thread
# ---------------------------------------------------------------------------

# --- Shared stub helpers ---


def _stub_houdini_imports(monkeypatch):
    """Stub all Houdini-only imports that start() does inside itself."""
    # hwebserver — needed by `import hwebserver` inside start()
    fake_hwebserver = MagicMock()
    fake_hwebserver.run = MagicMock()
    monkeypatch.setitem(sys.modules, "hwebserver", fake_hwebserver)

    # handlers and hwebserver_app — imported as side-effect modules inside start()
    fake_handlers = MagicMock()
    fake_hwebserver_app = MagicMock()
    monkeypatch.setitem(sys.modules, "fxhoudinimcp_server.handlers", fake_handlers)
    monkeypatch.setitem(
        sys.modules, "fxhoudinimcp_server.hwebserver_app", fake_hwebserver_app
    )

    # gate.install_gate — called inside start()
    fake_gate_module = MagicMock()
    fake_gate_module.install_gate = MagicMock()
    monkeypatch.setitem(sys.modules, "fxhoudinimcp_server.gate", fake_gate_module)
    # Also patch the imported name directly so the already-cached import path works
    monkeypatch.setattr(
        "fxhoudinimcp_server.gate",
        fake_gate_module,
        raising=False,
    )

    return fake_hwebserver


def test_graphical_async_returns_promptly_and_flag_flips_after_wait(monkeypatch):
    """In a graphical session, start() must return BEFORE the health wait completes.

    Observable contract:
      - start() returns in < 2 s (does not block on the wait)
      - _server_started is still False immediately after start() returns
        (the wait hasn't finished yet)
      - after the daemon thread completes the (mocked) wait, _server_started
        flips to True
    """
    _stub_houdini_imports(monkeypatch)

    # _is_graphical_session() → True (graphical Houdini session)
    monkeypatch.setattr(startup, "_is_graphical_session", lambda: True, raising=False)

    # Controlled wait: blocks until we release it from the test
    wait_gate = threading.Event()
    wait_completed = threading.Event()

    def _controlled_wait(port, timeout_seconds=15.0):
        wait_gate.wait(timeout=5.0)  # block until test releases
        wait_completed.set()
        return {"pid": os.getpid(), "status": "ok", "houdini_version": "21.0.631"}

    monkeypatch.setattr(
        startup, "_wait_for_current_process_health", _controlled_wait
    )

    # Call start() — in the graphical branch it must return immediately
    t_start = time.monotonic()
    startup.start()
    elapsed = time.monotonic() - t_start

    # start() must have returned BEFORE the health wait gate was released
    assert elapsed < 2.0, (
        f"start() blocked for {elapsed:.2f}s in a graphical session — "
        "it should have returned promptly (daemon thread does the wait)"
    )

    # _server_started must still be False — the background thread hasn't finished
    assert startup._server_started is False, (
        "_server_started should remain False until the background wait completes"
    )

    # Now release the gate and give the daemon thread time to finish
    wait_gate.set()
    wait_completed.wait(timeout=5.0)
    time.sleep(0.05)  # allow the thread to write _server_started

    assert startup._server_started is True, (
        "_server_started must flip to True after the background wait succeeds"
    )


def test_graphical_async_health_failure_does_not_raise(monkeypatch):
    """In a graphical session, a health-wait FAILURE (returns None) must NOT raise.

    Observable contract:
      - start() returns without raising any exception
      - _server_started stays False (health never proved)
    """
    _stub_houdini_imports(monkeypatch)

    monkeypatch.setattr(startup, "_is_graphical_session", lambda: True, raising=False)

    # Health wait returns None — server never answered
    monkeypatch.setattr(
        startup,
        "_wait_for_current_process_health",
        lambda port, timeout_seconds=15.0: None,
    )

    # Must NOT raise
    startup.start()

    # _server_started must remain False — health failed
    assert startup._server_started is False, (
        "_server_started must stay False when graphical-mode health wait returns None"
    )


def test_hython_sync_raises_on_health_none(monkeypatch):
    """In a hython (headless) session, start() keeps the SYNCHRONOUS contract.

    When _wait_for_current_process_health returns None, start() must raise
    RuntimeError — the same behaviour as the current synchronous implementation.

    RED: raises=True so the test fails until hou-dev adds _is_graphical_session
    to startup.py (AttributeError on monkeypatch.setattr before impl lands).
    """
    _stub_houdini_imports(monkeypatch)

    # raising=True: this will raise AttributeError until hou-dev adds
    # _is_graphical_session to startup.py, keeping this test RED.
    monkeypatch.setattr(startup, "_is_graphical_session", lambda: False, raising=True)

    monkeypatch.setattr(
        startup,
        "_wait_for_current_process_health",
        lambda port, timeout_seconds=15.0: None,
    )

    with pytest.raises(RuntimeError, match="hwebserver did not answer mcp.health"):
        startup.start()

    assert startup._server_started is False


def test_hython_sync_success_sets_server_started(monkeypatch):
    """In a hython (headless) session, a successful health response sets _server_started.

    Observable contract:
      - start() returns without raising
      - _server_started is True after return

    RED: raising=True so the test fails until hou-dev adds _is_graphical_session
    to startup.py (AttributeError on monkeypatch.setattr before impl lands).
    """
    _stub_houdini_imports(monkeypatch)

    # raising=True: AttributeError until hou-dev adds _is_graphical_session.
    monkeypatch.setattr(startup, "_is_graphical_session", lambda: False, raising=True)

    monkeypatch.setattr(
        startup,
        "_wait_for_current_process_health",
        lambda port, timeout_seconds=15.0: {
            "pid": os.getpid(),
            "status": "ok",
            "houdini_version": "21.0.631",
        },
    )

    startup.start()

    assert startup._server_started is True


def test_reentrance_guard_skips_hwebserver_run(monkeypatch):
    """While a start() is mid-warmup (_starting=True), a second start() must not
    call hwebserver.run again.

    Observable contract:
      - When _starting is True (set externally, simulating mid-warmup),
        start() returns immediately without running hwebserver.run

    RED: _starting flag is not yet checked by start(); hwebserver.run will be
    called because there is no re-entrancy guard in the current implementation.
    """
    fake_hwebserver = _stub_houdini_imports(monkeypatch)

    monkeypatch.setattr(startup, "_is_graphical_session", lambda: False, raising=False)
    # Stub the wait so start() can reach its completion path without blocking
    monkeypatch.setattr(
        startup,
        "_wait_for_current_process_health",
        lambda port, timeout_seconds=15.0: {
            "pid": os.getpid(),
            "status": "ok",
            "houdini_version": "21.0.631",
        },
    )
    # Simulate the state mid-way through a first start() call
    monkeypatch.setattr(startup, "_starting", True, raising=False)

    startup.start()

    # hwebserver.run must NOT have been called — the guard bailed out early.
    # RED: current start() has no _starting check, so .run IS called.
    fake_hwebserver.run.assert_not_called()


def test_start_graphical_thread_start_failure_clears_starting(monkeypatch):
    """CODEX-F1 regression: if t.start() raises, _starting must be False (not wedged).

    Observable contract:
      - start() re-raises the exception from Thread.start()
      - startup._starting is False after the exception (not permanently True)
    """
    _stub_houdini_imports(monkeypatch)

    monkeypatch.setattr(startup, "_is_graphical_session", lambda: True, raising=False)

    # Make Thread.start() raise before the worker ever runs
    def _bad_thread_start(self):
        raise RuntimeError("OS thread limit reached")

    monkeypatch.setattr(startup.threading.Thread, "start", _bad_thread_start)

    # start() must re-raise the RuntimeError from t.start()
    with pytest.raises(RuntimeError, match="OS thread limit reached"):
        startup.start()

    # _starting must be cleared — no permanent wedge
    assert startup._starting is False
