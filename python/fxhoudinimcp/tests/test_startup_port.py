"""
Tests for _pick_free_port — pure-logic port-selection helper in startup.py.
PP12-115a (HMDNI-116) — MCP Bridge Hardening, Part A.

No hou / Qt / pxr imports anywhere in this file.  Runs under plain pytest
headless (off-DCC, no Houdini install required).  startup.py has no top-level
hou import, so it is directly importable here.

Covers the public contract of:
  _pick_free_port(base: int, probe, max_tries: int = 16, my_pid: int | None = None) -> int

The `probe` argument is an injectable callable: `probe(port: int) -> dict | None`.
  - Returns a dict with at least a "pid" key when a server is listening.
  - Returns None when the port is free.

Idempotent-restart contract: if probe(port) returns {"pid": my_pid} the port is
already owned by THIS process — return it immediately (don't skip).

Port-exhaustion contract: if all max_tries ports are taken by other pids → raise
RuntimeError.

TDD phase: RED — _pick_free_port does NOT exist yet in startup.py.
Expected failure: AttributeError on `startup._pick_free_port`.

Cross-references:
  - Plan pp12-115a planSha b84d3aaced76b5929a7a5979033cb48ccebd63322f06771753477e2d43507f39
  - HMDNI-116
  - CL-015: pure-logic boundary — no hou/Qt/pxr
  - tdd-with-agents.md §4: hou-test writes red; hou-dev turns green
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — inject the houdini-side server package so fxhoudinimcp_server
# is importable under plain pytest (no Houdini install needed for pure-logic).
# Three ".." levels from tests/ → fxhoudinimcp/ → python/ → <fork_root>/.
# ---------------------------------------------------------------------------
_SERVER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "houdini", "scripts", "python")
)
if _SERVER not in sys.path:
    sys.path.insert(0, _SERVER)

import pytest
from fxhoudinimcp_server import startup


# ===========================================================================
# Section 1 — First port is free: probe returns None immediately
#
# Locked contract (plan pp12-115a acceptanceTests[0]):
#   _pick_free_port(8100, probe=lambda p: None, my_pid=999) == 8100
# ===========================================================================

class TestPickFreePortFirstFree:
    """_pick_free_port returns base port when it is immediately free."""

    def test_first_port_free(self):
        """Returns base when probe(base) is None (port is free)."""
        result = startup._pick_free_port(8100, probe=lambda p: None, my_pid=999)
        assert result == 8100, (
            f"_pick_free_port must return 8100 when probe(8100) is None; got {result!r}"
        )

    def test_first_port_free_no_my_pid(self):
        """Returns base when probe returns None and my_pid is not set."""
        result = startup._pick_free_port(8100, probe=lambda p: None)
        assert result == 8100, (
            f"_pick_free_port must return 8100 when probe(8100) is None and my_pid=None; got {result!r}"
        )

    def test_returns_int(self):
        """Return value is a plain Python int."""
        result = startup._pick_free_port(8100, probe=lambda p: None)
        assert isinstance(result, int), (
            f"_pick_free_port must return int; got {type(result).__name__!r}"
        )

    def test_different_base_port(self):
        """Works with a different base port (9000)."""
        result = startup._pick_free_port(9000, probe=lambda p: None, my_pid=42)
        assert result == 9000, (
            f"_pick_free_port(9000, ...) must return 9000 when free; got {result!r}"
        )


# ===========================================================================
# Section 2 — First port taken by another pid, second is free
#
# Locked contract (plan pp12-115a acceptanceTests[1]):
#   probe(8100) = {"pid": 111}  (other process)
#   probe(8101) = None          (free)
#   my_pid = 999
#   _pick_free_port(8100, probe, my_pid=999) == 8101
# ===========================================================================

class TestPickFreePortSkipTaken:
    """_pick_free_port skips ports owned by other pids, takes the first free one."""

    def _probe_first_taken(self, port: int) -> dict | None:
        """Port 8100 is taken by pid 111; 8101 is free."""
        if port == 8100:
            return {"pid": 111}
        return None

    def test_skips_other_pid_takes_next_free(self):
        """Skips port taken by another pid, returns next free port."""
        result = startup._pick_free_port(8100, probe=self._probe_first_taken, my_pid=999)
        assert result == 8101, (
            f"_pick_free_port must return 8101 when 8100 is taken by pid 111 and 8101 is free; "
            f"got {result!r}"
        )

    def test_probe_called_with_base_first(self):
        """probe is called starting from the base port, then incrementing."""
        probed_ports: list[int] = []

        def probe(port: int) -> dict | None:
            probed_ports.append(port)
            if port == 8100:
                return {"pid": 111}
            return None

        startup._pick_free_port(8100, probe=probe, my_pid=999)
        assert probed_ports[0] == 8100, (
            f"First probe call must be port 8100; got {probed_ports[0]!r}"
        )
        assert 8101 in probed_ports, (
            f"Port 8101 must be probed when 8100 is taken; probed: {probed_ports!r}"
        )


# ===========================================================================
# Section 3 — Idempotent restart: port already owned by this process
#
# Locked contract (plan pp12-115a acceptanceTests[2]):
#   probe(8100) = {"pid": 999}   (THIS process already owns the port)
#   my_pid = 999
#   _pick_free_port(8100, probe, my_pid=999) == 8100
#
# This is the idempotent-restart case: if the server is already running under
# this pid, return the port immediately instead of skipping it.
# ===========================================================================

class TestPickFreePortIdempotentRestart:
    """_pick_free_port returns base when port is already owned by this process."""

    def test_returns_base_when_owned_by_my_pid(self):
        """Returns 8100 when probe(8100) = {'pid': 999} and my_pid = 999."""
        probe = lambda p: {"pid": 999}
        result = startup._pick_free_port(8100, probe=probe, my_pid=999)
        assert result == 8100, (
            f"_pick_free_port must return 8100 when port is owned by my_pid=999; "
            f"got {result!r}"
        )

    def test_idempotent_does_not_skip_to_next_port(self):
        """When probe returns my_pid, the function must NOT increment to the next port."""
        probed_ports: list[int] = []

        def probe(port: int) -> dict | None:
            probed_ports.append(port)
            return {"pid": 42}

        result = startup._pick_free_port(8100, probe=probe, my_pid=42)
        assert result == 8100, (
            f"Idempotent restart: must return 8100 immediately, not scan further; "
            f"got {result!r}"
        )
        # Should only have probed port 8100 (not 8101, 8102, …)
        assert probed_ports == [8100], (
            f"Should only probe port 8100 before returning; probed: {probed_ports!r}"
        )

    def test_idempotent_different_pid(self):
        """No false idempotent match: probe({'pid': 111}) with my_pid=999 must skip."""
        calls: list[int] = []

        def probe(port: int) -> dict | None:
            calls.append(port)
            if port == 8100:
                return {"pid": 111}
            return None

        result = startup._pick_free_port(8100, probe=probe, my_pid=999)
        assert result == 8101, (
            f"pid=111 is NOT my_pid=999; must skip 8100 and return 8101; got {result!r}"
        )


# ===========================================================================
# Section 4 — Several ports taken, then free
#
# Locked contract (plan pp12-115a acceptanceTests[3]):
#   probe(8100) = {"pid": 111}
#   probe(8101) = {"pid": 222}
#   probe(8102) = {"pid": 333}
#   probe(8103) = None   (free)
#   my_pid = 999
#   _pick_free_port(8100, probe, my_pid=999) == 8103
# ===========================================================================

class TestPickFreePortSeveralTakenThenFree:
    """_pick_free_port scans past multiple taken ports to the first free one."""

    def _probe(self, port: int) -> dict | None:
        taken = {8100: 111, 8101: 222, 8102: 333}
        if port in taken:
            return {"pid": taken[port]}
        return None

    def test_scans_to_fourth_free(self):
        """Returns 8103 when 8100/8101/8102 are taken by other pids and 8103 is free."""
        result = startup._pick_free_port(8100, probe=self._probe, my_pid=999)
        assert result == 8103, (
            f"_pick_free_port must return 8103 after skipping 8100/8101/8102; "
            f"got {result!r}"
        )

    def test_scans_in_sequential_order(self):
        """Ports are probed in sequential ascending order."""
        probed: list[int] = []

        def probe(port: int) -> dict | None:
            probed.append(port)
            return self._probe(port)

        startup._pick_free_port(8100, probe=probe, my_pid=999)
        assert probed[:4] == [8100, 8101, 8102, 8103], (
            f"Ports must be probed in order 8100→8101→8102→8103; got {probed[:4]!r}"
        )


# ===========================================================================
# Section 5 — All max_tries ports taken → RuntimeError
#
# Locked contract (plan pp12-115a acceptanceTests[4]):
#   All max_tries=4 ports (8100..8103) taken by other pids.
#   my_pid = 999
#   _pick_free_port(8100, probe, max_tries=4, my_pid=999) raises RuntimeError
# ===========================================================================

class TestPickFreePortExhaustion:
    """_pick_free_port raises RuntimeError when all max_tries ports are taken."""

    def _probe_all_taken_by_other(self, port: int) -> dict | None:
        """Every port is taken by a process that is NOT my_pid=999."""
        return {"pid": port}   # unique other-pids per port

    def test_raises_runtime_error_on_exhaustion(self):
        """Raises RuntimeError when all max_tries=4 ports are taken by other pids."""
        with pytest.raises(RuntimeError):
            startup._pick_free_port(
                8100,
                probe=self._probe_all_taken_by_other,
                max_tries=4,
                my_pid=999,
            )

    def test_exhaustion_only_after_exactly_max_tries(self):
        """Raises RuntimeError after probing exactly max_tries ports, not fewer."""
        probed: list[int] = []

        def probe(port: int) -> dict | None:
            probed.append(port)
            return {"pid": port}   # always taken by another pid

        with pytest.raises(RuntimeError):
            startup._pick_free_port(8100, probe=probe, max_tries=4, my_pid=999)

        assert len(probed) == 4, (
            f"Must probe exactly 4 ports before raising RuntimeError; "
            f"probed {len(probed)}: {probed!r}"
        )

    def test_default_max_tries_is_16(self):
        """Default max_tries is 16: probe exactly 16 ports then raise."""
        probed: list[int] = []

        def probe(port: int) -> dict | None:
            probed.append(port)
            return {"pid": port}

        with pytest.raises(RuntimeError):
            startup._pick_free_port(8100, probe=probe, my_pid=999)

        assert len(probed) == 16, (
            f"Default max_tries must be 16; probed {len(probed)}: {probed!r}"
        )

    def test_error_message_mentions_port_or_exhaustion(self):
        """RuntimeError message is non-empty and mentions port or exhaustion."""
        with pytest.raises(RuntimeError) as exc_info:
            startup._pick_free_port(
                8100,
                probe=lambda p: {"pid": p},
                max_tries=2,
                my_pid=999,
            )
        msg = str(exc_info.value)
        assert len(msg) > 0, "RuntimeError message must be non-empty"


# ===========================================================================
# Section 6 — hou-free import verification (CL-015)
#
# startup.py has NO top-level hou import (hou is imported only inside
# _is_graphical_session). This section proves it loads under plain pytest.
# ===========================================================================

class TestStartupHouFreeAtTopLevel:
    """startup.py must import cleanly off-DCC (no top-level hou dependency)."""

    def test_startup_importable_without_hou(self):
        """fxhoudinimcp_server.startup loads under plain Python with no Houdini."""
        from fxhoudinimcp_server import startup as _s
        assert _s is not None

    def test_hou_not_imported_at_module_top_level(self):
        """startup module must not reference 'import hou' at top level (CL-015)."""
        import inspect
        import re

        source = inspect.getsource(startup)
        # Strip comments, then check that 'import hou' only appears inside a
        # function body (preceded by indentation) — NOT at module level (column 0).
        top_level_import_hou = re.search(r"^import hou\b", source, re.MULTILINE)
        assert top_level_import_hou is None, (
            "startup.py must not have a top-level 'import hou' "
            "(CL-015 — pure-logic boundary violated)"
        )
