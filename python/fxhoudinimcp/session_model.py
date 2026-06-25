"""session_model.py — pure-logic multi-session discovery and routing (PP12-115b).

This module is intentionally import-free of hou, Qt (PySide6), and pxr so
it remains pytest-able off-DCC (CL-015 invariant).

Public API
----------
build_session_entry(port, health) -> dict
    Convert a raw health_check() response into a canonical session entry.

mark_active(entries, active_port) -> list[dict]
    Return a new list with each entry annotated with active=True/False.

resolve_target(sessions, selector) -> int | None
    Resolve a port-int or hip-file substring selector to a port, or None
    if the selector is ambiguous (zero or multiple matches).
"""

from __future__ import annotations

_CARRIED_KEYS = ("pid", "hip_file", "houdini_version", "gate_mode", "pending_count")


def build_session_entry(port: int, health: dict) -> dict:
    """Convert a raw health_check() response into a canonical session entry.

    The 'status' key is intentionally omitted from the entry; 'port' is
    added.  Optional keys (gate_mode, pending_count) are carried only when
    present in the health dict.

    Args:
        port:   The port this session is running on.
        health: The dict returned by HoudiniBridge.health_check().

    Returns:
        A dict with at minimum {"port": <int>, "pid": ..., "hip_file": ...,
        "houdini_version": ...} plus any optional keys that were present.
    """
    entry: dict = {"port": port}
    for key in _CARRIED_KEYS:
        if key in health:
            entry[key] = health[key]
    return entry


def mark_active(entries: list[dict], active_port: int) -> list[dict]:
    """Return a new list with each entry annotated with an 'active' bool.

    Does NOT mutate the input list or any of its dicts.

    Args:
        entries:     List of session entry dicts (as produced by build_session_entry).
        active_port: The currently selected port.

    Returns:
        A new list where each entry is a copy of the original with
        'active' set to True iff entry["port"] == active_port.
    """
    return [{**e, "active": bool(e["port"] == active_port)} for e in entries]


def resolve_target(sessions: list[dict], selector: int | str | None) -> int | None:
    """Resolve a selector to a port number, or None if ambiguous/absent.

    Rules:
    - int selector: exact port match — returns the port if a session with
      that port exists in the list, else None.
    - str selector: case-insensitive substring match on 'hip_file' —
      returns the port ONLY when exactly ONE session matches.
      Zero matches -> None.  Two or more matches -> None (ambiguous).
      Empty string matches every session; ambiguous when multiple sessions exist.
    - Any other type: None.

    Args:
        sessions: List of session entry dicts.
        selector: An int port number or str hip_file substring.

    Returns:
        The matched port (int) or None.
    """
    if isinstance(selector, int):
        for s in sessions:
            if s["port"] == selector:
                return selector
        return None

    if isinstance(selector, str):
        needle = selector.lower()
        matches = [
            s for s in sessions
            if "hip_file" in s and needle in s["hip_file"].lower()
        ]
        if len(matches) == 1:
            return matches[0]["port"]
        return None

    return None
