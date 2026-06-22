"""Gate middleware: wraps dispatcher.dispatch() with security gate logic.

install_gate() swaps dispatcher.dispatch for _gated_dispatch exactly once
(idempotent — double-wrap guarded via _is_gated sentinel + hou.session stash).

Architecture notes:
- GATE singleton is stashed on hou.session._fxhoudinimcp_gate, NOT on this
  module, so it survives importlib.reload() (which re-executes the module body
  but never touches hou.session).  See CL-005/CL-011.
- Gate commands bypass the outer hdefereval marshal in _gated_dispatch via
  _ORIGINAL_DISPATCH.  The gate command handlers run directly so that approve/
  reject paths do not create a worker that blocks waiting for main-thread that
  is already servicing the outer dispatch — the rev-1 deadlock pattern.
- PendingQueue thunks are bare lambdas: lambda h=handler, p=captured: h(**p).
  No nested dispatcher.dispatch call, no worker/join.  See rev-1 deadlock fix.
- No off-main-thread hou.* access.  See CL-016.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
from typing import Any, Callable

log = logging.getLogger("fxhoudinimcp_server.gate.middleware")

# ---------------------------------------------------------------------------
# The exact set of gate command names that bypass the gate (exact-match only).
# R4E: startswith("gate.") is NOT used — a leading-space typo must not bypass.
# ---------------------------------------------------------------------------

_GATE_COMMANDS: frozenset[str] = frozenset({
    "gate.get_permission_mode",
    "gate.set_permission_mode",
    "gate.list_pending_calls",
    "gate.approve_pending_call",
    "gate.reject_pending_call",
    "gate.classify_code",
    "gate.get_audit_log",
})

# Keys whose values contain operator-supplied code that the classifier must see.
# D5 fix: covers all injection surfaces across all 12 CODE_EXEC commands.
_CODE_KEYS: tuple[str, ...] = (
    "code",
    "command",
    "expression",
    "vex_code",
    "snippet",
    "python",
    "return_expression",
)


def _mode_str(mode) -> str:
    """Normalize a Mode enum value to the underscore form for external responses.

    The pure-core Mode.READ_ONLY.value is "read-only" (hyphen, PEP-8 enum
    value convention).  The smoke harness and FastMCP tools surface mode names
    with underscores ("read_only") for consistency with Python identifier style.
    This helper is the bridge between the two representations so the pure core
    stays unmodified.
    """
    return mode.value.replace("-", "_")


def _extract_code(params: dict[str, Any]) -> str | None:
    """Extract operator-supplied code from params for classifier input.

    Returns the first non-empty string value found under any key in _CODE_KEYS,
    or None if params contains no code payload.
    """
    for key in _CODE_KEYS:
        val = params.get(key)
        if val and isinstance(val, str):
            return val
    return None


def _get_gate():
    """Return the GATE singleton from hou.session, or None if not installed."""
    try:
        import hou  # noqa: PLC0415
        return getattr(hou.session, "_fxhoudinimcp_gate", None)
    except ImportError:
        return None


def _set_gate(gate_obj) -> None:
    """Stash GATE on hou.session so it survives importlib.reload()."""
    try:
        import hou  # noqa: PLC0415
        hou.session._fxhoudinimcp_gate = gate_obj
    except ImportError:
        # hython with no session support — fall through; gate lives on module only
        pass


def _build_gate(config_path: str):
    """Build a new GateInstance (wrapping pure-core objects) from config_path."""
    from homedini.dcc.mcp_gate.config import load_config
    from homedini.dcc.mcp_gate.pending_queue import PendingQueue
    from homedini.dcc.mcp_gate.audit import AuditLog

    cfg = load_config(config_path)
    # Expand $HOUDINI_USER_PREF_DIR in audit_log path.
    audit_path = cfg.audit_log
    if "$HOUDINI_USER_PREF_DIR" in audit_path:
        pref_dir = os.environ.get("HOUDINI_USER_PREF_DIR", "")
        audit_path = audit_path.replace("$HOUDINI_USER_PREF_DIR", pref_dir)

    return _GateInstance(
        config=cfg,
        queue=PendingQueue(ttl_seconds=cfg.queue_ttl_seconds),
        audit=AuditLog(path=audit_path),
    )


class _GateInstance:
    """Holds the runtime gate state: config, pending queue, and audit log."""

    def __init__(self, config, queue, audit):
        self.config = config   # GateConfig
        self.queue = queue     # PendingQueue
        self.audit = audit     # AuditLog


def _cap_from_dispatcher(command: str):
    """Return pure-core Capability matching the fork dispatcher's Capability for command.

    Converts the fork Capability enum to pure-core Capability by .value string.
    Both enums share identical string values ("readonly"/"mutating"/"code_exec").
    Returns pure-core Capability.MUTATING as safe default if undeclared.
    """
    from homedini.dcc.mcp_gate.gate_model import Capability as CoreCap

    import fxhoudinimcp_server.dispatcher as _d
    fork_cap = _d.capability_of(command)
    if fork_cap is None:
        return CoreCap.MUTATING
    return CoreCap(fork_cap.value)


def _gated_dispatch(command: str, params: dict[str, Any]) -> dict[str, Any]:
    """Security-gated replacement for dispatcher.dispatch().

    Gate commands bypass directly to _ORIGINAL_DISPATCH (no hdefereval marshal).
    All other commands are evaluated through the pure-core decision pipeline.
    """
    import fxhoudinimcp_server.dispatcher as _d

    # --- Gate command bypass (exact-match only, R4E) ---
    if command in _GATE_COMMANDS:
        # Gate commands run via _ORIGINAL_DISPATCH to avoid nested hdefereval.
        # _ORIGINAL_DISPATCH wraps handler results in {"status":"success","data":handler_result}.
        # We surface the handler result directly (it already has gate=allowed, status, data keys)
        # so callers see a flat response: {gate, status, data, ...}.
        orig = getattr(_d, "_ORIGINAL_DISPATCH", None)
        if orig is None:
            return {"status": "error", "error": {"code": "GATE_NOT_INSTALLED", "message": "Gate not properly installed"}}
        outer = orig(command, params)
        # Unwrap the dispatcher envelope: surface the inner handler dict directly.
        if isinstance(outer, dict) and outer.get("status") == "success" and "data" in outer:
            inner = outer["data"]
            if isinstance(inner, dict):
                result = dict(inner)
                # Preserve timing_ms from the outer envelope if helpful.
                if "timing_ms" in outer:
                    result.setdefault("timing_ms", outer["timing_ms"])
                return result
        # Dispatch-level error (e.g. timeout, unknown command) — return as-is.
        return outer

    gate = _get_gate()
    if gate is None:
        # Fail closed: gate not installed -> deny everything non-gate.
        log.error("_gated_dispatch: GATE singleton missing — denying command %r (fail-closed)", command)
        return {
            "gate": "denied",
            "status": "denied",
            "reason": "Security gate not initialized — all commands denied until install_gate() is called.",
        }

    from homedini.dcc.mcp_gate.gate_model import (
        Mode, Capability as CoreCap, AuditEvent, Decision,
    )
    from homedini.dcc.mcp_gate.policy import decide
    from homedini.dcc.mcp_gate.classifier import classify_python, classify_hscript

    cfg = gate.config

    # --- Resolve capability ---
    capability = _cap_from_dispatcher(command)

    # --- Classify code payload (if present) ---
    code = _extract_code(params)
    if code is not None:
        classification = classify_python(code, cfg.danger_classes)
    else:
        # Non-code commands: produce a benign classification.
        from homedini.dcc.mcp_gate.gate_model import Classification, Severity
        classification = Classification(
            danger=False,
            classes=[],
            severity=Severity.NONE,
            reasons=[],
        )

    # --- Policy decision ---
    decision = decide(cfg.mode, capability, classification)

    # --- Act on decision ---
    if decision == Decision.ALLOW:
        # Emit audit event then call original dispatch.
        _emit_audit(gate, command, params, "allowed", classification)
        orig = getattr(_d, "_ORIGINAL_DISPATCH", None)
        if orig is None:
            log.error("_gated_dispatch: _ORIGINAL_DISPATCH missing on allow — failing closed")
            return {"gate": "denied", "status": "denied", "reason": "Gate internal error: original dispatch missing"}
        outer = orig(command, params)
        # Unwrap the dispatcher envelope ({"status":"success","data":handler_result})
        # so callers see the handler's keys directly + gate=allowed.
        # This matches the smoke harness expectations:
        #   info.get("gate") == "allowed" AND "houdini_version" in info (item3/4).
        if isinstance(outer, dict) and outer.get("status") == "success" and "data" in outer:
            inner = outer["data"]
            if isinstance(inner, dict):
                # ADR-0002 Option A: preserve the {status, gate, data} envelope so
                # bridge.execute()'s result.get("data", {}) resolves to the handler
                # payload rather than {}.  The old dict(inner) flattening dropped
                # the "data" key, breaking every non-gate tool over the live bridge.
                result = {"status": "success", "gate": "allowed", "data": inner}
                if "timing_ms" in outer:
                    result["timing_ms"] = outer["timing_ms"]
                return result
        # Non-standard response (e.g. error or timeout) — surface as-is + gate.
        outer["gate"] = "allowed"
        return outer

    elif decision == Decision.DENY:
        _emit_audit(gate, command, params, "denied", classification)
        mode_str = _mode_str(cfg.mode)
        return {
            "gate": "denied",
            "status": "denied",
            "reason": (
                f"Command '{command}' denied in mode '{cfg.mode.value}'. "
                f"Capability: {capability.value}."
            ),
            "mode": mode_str,
            "capability": capability.value,
        }

    elif decision == Decision.QUEUE:
        # Capture handler and params for the bare thunk — no nested dispatch.
        handler = _d._handler_registry.get(command)
        if handler is None:
            return {
                "status": "error",
                "error": {"code": "UNKNOWN_COMMAND", "message": f"No handler for: {command}"},
            }
        captured = dict(params)
        # Bare thunk: lambda with default captures, NO nested dispatcher.dispatch.
        # This is the rev-1 deadlock fix — no worker/join nesting.
        thunk: Callable = lambda h=handler, p=captured: h(**p)  # noqa: E731

        pending_id = gate.queue.add(
            tool=command,
            capability=capability,
            classification=classification,
            code=code or "",
            run_thunk=thunk,
        )
        _emit_audit(gate, command, params, "queued", classification, pending_id=pending_id)
        return {
            "gate": "queued",
            "status": "pending_approval",
            "pending_id": pending_id,
            "command": command,
            "capability": capability.value,
            "mode": _mode_str(cfg.mode),
            "classification": {
                "danger": classification.danger,
                "classes": classification.classes,
                "severity": classification.severity.name,
                "reasons": classification.reasons,
            },
            "message": (
                f"Command '{command}' queued for approval (mode={cfg.mode.value}, "
                f"cap={capability.value}). Use gate.approve_pending_call or "
                f"gate.reject_pending_call with pending_id={pending_id!r}."
            ),
        }

    else:
        # Unknown decision variant — fail closed.
        log.error("_gated_dispatch: unexpected Decision %r for command %r — denying", decision, command)
        return {"gate": "denied", "status": "denied", "reason": f"Unexpected policy decision: {decision!r}"}


def _emit_audit(gate, command: str, params: dict, event_type: str, classification, pending_id: str | None = None) -> None:
    """Append an AuditEvent to the gate's audit log.  Errors are logged, not raised.

    AuditEvent expects a 'classification' field (serialised dict) and
    a 'code_sha256' field (hex digest or None) — NOT raw danger/classes/etc.
    """
    try:
        import hashlib
        from homedini.dcc.mcp_gate.gate_model import AuditEvent
        # Compute SHA-256 of the code payload if present (FR-7: never store raw code).
        code = _extract_code(params)
        code_sha256 = hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest() if code else None
        # Serialise the Classification into the format AuditEvent expects.
        cl_dict: dict = {
            "danger": classification.danger,
            "classes": list(classification.classes),
            "severity": (
                classification.severity.name
                if hasattr(classification.severity, "name")
                else str(classification.severity)
            ),
            "reasons": list(classification.reasons),
        }
        ev = AuditEvent(
            ts=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            event=event_type,
            tool=command,
            capability=str(_cap_from_dispatcher(command).value),
            mode=gate.config.mode.value,
            classification=cl_dict,
            code_sha256=code_sha256,
            pending_id=pending_id,
        )
        gate.audit.append(ev)
    except Exception as exc:  # noqa: BLE001
        log.warning("_emit_audit: failed to write audit event for %r: %s", command, exc)


def _warn_undeclared_commands() -> None:
    """FR-11 backstop: log a warning for any registered command with no declared capability.

    Iterates dispatcher.list_commands() and warns for any command where
    dispatcher.capability_of(cmd) returns None.  Such commands will be treated
    as MUTATING (fail-closed per ADR §3.3.4), but the missing declaration is
    a configuration smell worth surfacing at install time.
    """
    try:
        import fxhoudinimcp_server.dispatcher as _d
        if not (hasattr(_d, "list_commands") and hasattr(_d, "capability_of")):
            return
        for cmd in _d.list_commands():
            if _d.capability_of(cmd) is None:
                log.warning(
                    "install_gate: command %r has no declared capability — "
                    "will be treated as MUTATING (fail-closed)",
                    cmd,
                )
    except Exception as exc:  # noqa: BLE001
        log.warning("install_gate: _warn_undeclared_commands failed: %s", exc)


def install_gate() -> None:
    """Install the security gate on dispatcher.dispatch(), idempotent.

    Loads all fxhoudinimcp_server handlers (if not already loaded), then
    wraps dispatcher.dispatch with _gated_dispatch.  Safe to call multiple
    times — idempotent via the _is_gated sentinel (R4C).

    The GATE singleton is stashed on hou.session so it survives reload().
    Config is read from $HOUDINI_USER_PREF_DIR/mcp_gate/mcp_gate.json.

    Re-install ordering (ADR §3.4.4 D4):
    1. RESTORE existing GATE from hou.session FIRST — preserves the live
       PendingQueue across a Python-module reload.
    2. Check wrap idempotency SECOND and separately — these two are independent.
    3. Register gate handlers, warn undeclared commands, wire wrap only on
       first install.
    """
    import fxhoudinimcp_server.dispatcher as _d

    # Ensure all command handlers are registered before the gate wraps dispatch.
    # This makes install_gate() callable from standalone tests (the smoke harness)
    # without a separate 'from fxhoudinimcp_server import handlers' call.
    try:
        from fxhoudinimcp_server import handlers  # noqa: F401
    except ImportError as exc:
        log.warning("install_gate: could not import handlers: %s", exc)

    # Step 1: RESTORE GATE from hou.session FIRST (ADR §3.4.4 D4).
    # This must happen before the idempotency check so that a re-install after
    # a module reload re-attaches the live PendingQueue rather than discarding it.
    existing = _get_gate()
    if existing is not None:
        gate = existing  # REUSE — preserves the live PendingQueue
        log.debug("install_gate: re-attaching existing GATE from hou.session (queue preserved)")
    else:
        # First install — build a fresh gate and stash it on hou.session.
        pref_dir = os.environ.get("HOUDINI_USER_PREF_DIR", "")
        config_path = os.path.join(pref_dir, "mcp_gate", "mcp_gate.json")
        gate = _build_gate(config_path)
        _set_gate(gate)
        log.debug("install_gate: built fresh GATE from config %r", config_path)

    # Step 2: Wrap idempotency is SEPARATE from the GATE restore (R4C).
    # Check only after GATE is resolved so we never drop the live queue.
    if getattr(_d.dispatch, "_is_gated", False) is True:
        log.debug(
            "install_gate: gate wrap already applied (idempotent no-op); queue preserved"
        )
        return

    # Step 3: Register gate handlers, backstop-warn undeclared commands, wire wrap.
    _register_gate_handlers(_d, gate)
    _warn_undeclared_commands()

    # Preserve _ORIGINAL_DISPATCH before wrapping.
    _d._ORIGINAL_DISPATCH = _d.dispatch

    # Install the gated wrapper.
    _d.dispatch = _gated_dispatch
    _gated_dispatch._is_gated = True  # type: ignore[attr-defined]

    log.info(
        "install_gate: security gate installed (mode=%s)",
        gate.config.mode.value,
    )


# ---------------------------------------------------------------------------
# Gate command handlers
# ---------------------------------------------------------------------------

def _register_gate_handlers(_d, gate_ref) -> None:
    """Register the 7 gate control commands into the dispatcher.

    These are registered with READONLY capability so they are never themselves
    gated (they bypass via _GATE_COMMANDS frozenset in _gated_dispatch).
    Thunks use a late-binding lambda to _get_gate() so that if the gate
    singleton is replaced (e.g. in tests), handlers see the live object.
    """
    from homedini.dcc.mcp_gate.gate_model import Mode

    # --- gate.get_permission_mode ---
    def _get_permission_mode() -> dict:
        g = _get_gate()
        if g is None:
            return {"status": "error", "error": "Gate not installed"}
        return {"gate": "allowed", "status": "success", "data": {"mode": _mode_str(g.config.mode)}}

    # --- gate.set_permission_mode ---
    def _set_permission_mode(mode: str) -> dict:
        g = _get_gate()
        if g is None:
            return {"status": "error", "error": "Gate not installed"}
        try:
            new_mode = Mode(mode)
        except ValueError:
            return {
                "gate": "allowed",
                "status": "error",
                "error": f"Unknown mode {mode!r}. Valid: {[m.value for m in Mode]}",
            }
        from dataclasses import replace
        g.config = replace(g.config, mode=new_mode)
        log.info("gate.set_permission_mode: mode changed to %r", new_mode.value)
        return {"gate": "allowed", "status": "success", "data": {"mode": _mode_str(g.config.mode)}}

    # --- gate.list_pending_calls ---
    def _list_pending_calls() -> dict:
        g = _get_gate()
        if g is None:
            return {"status": "error", "error": "Gate not installed"}
        pending = g.queue.list()
        return {"gate": "allowed", "status": "success", "data": {"pending": pending}}

    # --- gate.approve_pending_call ---
    def _approve_pending_call(pending_id: str) -> dict:
        g = _get_gate()
        if g is None:
            return {"status": "error", "error": "Gate not installed"}
        # First check: is the pending_id known (raises KeyError if not)?
        try:
            raw_handler_result = g.queue.approve(pending_id)
        except KeyError:
            return {
                "gate": "allowed",
                "status": "error",
                "error": f"No pending call with id {pending_id!r} (expired or already handled).",
            }
        except Exception as exc:  # noqa: BLE001 — handler itself raised
            log.error("gate.approve_pending_call: handler raised: %s", exc, exc_info=True)
            wrapped = {
                "status": "error",
                "gate": "approved",
                "error": {
                    "code": type(exc).__name__,
                    "message": str(exc),
                },
            }
            return wrapped

        # Wrap the raw handler result the same way dispatcher._execute would.
        if isinstance(raw_handler_result, dict) and "status" in raw_handler_result:
            # Handler already returned a status-keyed dict — pass through.
            wrapped = dict(raw_handler_result)
        else:
            wrapped = {"status": "success", "data": raw_handler_result}

        # Emit approved audit event.
        g2 = _get_gate()
        if g2 is not None:
            try:
                from homedini.dcc.mcp_gate.gate_model import Severity, Classification
                cl = Classification(danger=False, classes=[], severity=Severity.NONE, reasons=[])
                _emit_audit_raw(g2, pending_id, "approved", cl)
            except Exception as exc2:  # noqa: BLE001
                log.warning("gate.approve_pending_call: audit emit failed: %s", exc2)

        wrapped["gate"] = "approved"
        wrapped["status"] = wrapped.get("status", "success")
        return wrapped

    # --- gate.reject_pending_call ---
    def _reject_pending_call(pending_id: str, reason: str = "") -> dict:
        g = _get_gate()
        if g is None:
            return {"status": "error", "error": "Gate not installed"}
        try:
            g.queue.reject(pending_id, reason)
        except KeyError:
            return {
                "gate": "allowed",
                "status": "error",
                "error": f"No pending call with id {pending_id!r} (expired or already handled).",
            }
        return {"gate": "allowed", "status": "rejected", "pending_id": pending_id}

    # --- gate.classify_code ---
    def _classify_code(code: str, language: str = "python") -> dict:
        g = _get_gate()
        if g is None:
            return {"status": "error", "error": "Gate not installed"}
        from homedini.dcc.mcp_gate.classifier import classify_python, classify_hscript
        if language == "hscript":
            cl = classify_hscript(code)
        else:
            cl = classify_python(code, g.config.danger_classes)
        return {
            "gate": "allowed",
            "status": "success",
            "data": {
                "classification": {
                    "danger": cl.danger,
                    "classes": cl.classes,
                    "severity": cl.severity.name,
                    "reasons": cl.reasons,
                    "capability": "code_exec",
                },
            },
        }

    # --- gate.get_audit_log ---
    def _get_audit_log(n: int = 50) -> dict:
        g = _get_gate()
        if g is None:
            return {"status": "error", "error": "Gate not installed"}
        entries = g.audit.tail(n)
        return {"gate": "allowed", "status": "success", "data": {"entries": entries}}

    # Register gate commands with explicit capability declarations.
    # gate.set_permission_mode is MUTATING — it changes operator-intent state
    # (the gate enforcement mode).  All others are READONLY introspection.
    # All 7 bypass the gate itself via _GATE_COMMANDS frozenset in _gated_dispatch.
    from fxhoudinimcp_server.dispatcher import Capability as ForkCap, register_handler
    for name, fn, cap in [
        ("gate.get_permission_mode",  _get_permission_mode,  ForkCap.READONLY),
        ("gate.set_permission_mode",  _set_permission_mode,  ForkCap.MUTATING),  # FIX-3: operator-intent state change
        ("gate.list_pending_calls",   _list_pending_calls,   ForkCap.READONLY),
        ("gate.approve_pending_call", _approve_pending_call, ForkCap.READONLY),
        ("gate.reject_pending_call",  _reject_pending_call,  ForkCap.READONLY),
        ("gate.classify_code",        _classify_code,        ForkCap.READONLY),
        ("gate.get_audit_log",        _get_audit_log,        ForkCap.READONLY),
    ]:
        register_handler(name, fn, capability=cap)


def _emit_audit_raw(gate, pending_id: str, event_type: str, classification) -> None:
    """Emit an audit event with a pending_id reference (approve/reject path)."""
    try:
        from homedini.dcc.mcp_gate.gate_model import AuditEvent
        cl_dict: dict = {
            "danger": classification.danger,
            "classes": list(classification.classes),
            "severity": (
                classification.severity.name
                if hasattr(classification.severity, "name")
                else str(classification.severity)
            ),
            "reasons": list(classification.reasons),
        }
        ev = AuditEvent(
            ts=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            event=event_type,
            tool="<approved>",
            capability="",
            mode=gate.config.mode.value,
            classification=cl_dict,
            code_sha256=None,
            pending_id=pending_id,
        )
        gate.audit.append(ev)
    except Exception as exc:  # noqa: BLE001
        log.warning("_emit_audit_raw: failed: %s", exc)
