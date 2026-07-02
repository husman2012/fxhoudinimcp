"""session_model.py — pure-logic multi-session discovery and routing (PP12-115b/c).

This module is intentionally import-free of hou, Qt (PySide6), and pxr so
it remains pytest-able off-DCC (CL-015 invariant).

Public API
----------
build_session_entry(port, health) -> dict
    Convert a raw health_check() response into a canonical session entry.

mark_active(entries, active_port) -> list[dict]
    Return a new list with each entry annotated with active=True/False.

resolve_with_reason(sessions, selector) -> tuple[int | None, str]
    Resolve a port-int or hip-file substring selector to a (port, reason) tuple.
    reason is one of: "ok", "no_match", "ambiguous", "no_selector".

resolve_target(sessions, selector) -> int | None
    Thin delegate to resolve_with_reason; returns the port only.
    Preserved for backward compatibility with existing callers and tests.

active_pid_stale(sessions, active_port, active_pid) -> bool
    True when the live pid on active_port has drifted from the recorded pid.
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


def resolve_with_reason(
    sessions: list[dict], selector: int | str | None
) -> tuple[int | None, str]:
    """Resolve a selector to a (port, reason) tuple.

    Returns a 2-tuple whose first element is the matched port (int) or None,
    and whose second element is one of the reason strings below.

    Reason strings:
        "ok"          — exactly one match found; port is the matched port.
        "no_match"    — zero sessions matched the selector.
        "ambiguous"   — two or more sessions matched a string selector.
        "no_selector" — selector is None or not an int/str.

    Rules:
        - int selector: exact port match.
          Match found  -> (port, "ok").
          No match     -> (None, "no_match").
        - str selector: case-insensitive substring match on 'hip_file'.
          Exactly one match -> (port, "ok").
          Zero matches      -> (None, "no_match").
          Two+ matches      -> (None, "ambiguous").
        - None or other type: (None, "no_selector").

    Args:
        sessions: List of session entry dicts.
        selector: An int port number, str hip_file substring, or None.

    Returns:
        A (port | None, reason_str) 2-tuple.
    """
    if selector is None:
        return (None, "no_selector")

    if isinstance(selector, int):
        for s in sessions:
            if s.get("port") == selector:
                return (selector, "ok")
        return (None, "no_match")

    if isinstance(selector, str):
        needle = selector.lower()
        matches = [
            s for s in sessions
            if needle in str(s.get("hip_file") or "").lower()
        ]
        if len(matches) == 1:
            return (matches[0]["port"], "ok")
        if len(matches) == 0:
            return (None, "no_match")
        return (None, "ambiguous")

    # Any other type (list, dict, float, …)
    return (None, "no_selector")


def resolve_target(sessions: list[dict], selector: int | str | None) -> int | None:
    """Resolve a selector to a port number, or None if ambiguous/absent.

    Delegates to resolve_with_reason and returns the port element only.

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
    return resolve_with_reason(sessions, selector)[0]


def active_pid_stale(
    sessions: list[dict], active_port: int, active_pid: int | None
) -> bool:
    """Return True when the active session's live pid has drifted or the port is gone.

    Use this to detect a Houdini restart on the active port: if Houdini crashed
    and restarted it gets a new OS pid, so the live pid will differ from the one
    recorded at select-time.  Also returns True when the active port is absent
    from the live sessions list — a gone port means the selection died and the
    agent must re-select.

    Rules:
        - active_pid is None -> False  (no baseline to compare against).
        - active_port absent from sessions -> True (selection died/gone — re-select).
          Note: a matched entry whose "pid" key is missing also returns True
          because None (missing) != active_pid.
        - live session pid != active_pid -> True  (drift detected).
        - live session pid == active_pid -> False (stable).

    Args:
        sessions:    Live session entries from the current scan.
        active_port: The port recorded as the active selection.
        active_pid:  The pid recorded when the session was selected, or None.

    Returns:
        bool — True iff drift is detectable and confirmed, or the port is gone.
    """
    if active_pid is None:
        return False
    for s in sessions:
        if s.get("port") == active_port:
            return bool(s.get("pid") != active_pid)
    return True   # active_port no longer live -> the selection died/gone -> stale
