"""Main-thread dispatch mechanism for executing hou.* calls safely.

Houdini requires all hou.* API calls to run on the main thread.
hwebserver handlers run on worker threads, so we use
hdefereval.executeInMainThreadWithResult() to marshal calls
to the main thread and block until they complete.

Preview-hook extension (ADR 0005 rev2, pp12-111g):
  register_handler() now accepts optional ``preview_fn`` and
  ``preview_required`` kwargs.  ``preview_fn(params: dict) -> dict`` is
  called by the gate middleware (_run_preview in middleware.py) before
  queuing; the registry is stashed on hou.session._fxhoudinimcp_preview_registry
  so it survives importlib.reload() (ADV-003 / CL-005).
"""

from __future__ import annotations

# Built-in
import logging
import threading
import time
import traceback
from enum import Enum
from typing import Any, Callable, Optional

# Third-party (hdefereval is only available in graphical Houdini sessions)
try:
    import hdefereval
    HAS_HDEFEREVAL = True
except ImportError:
    HAS_HDEFEREVAL = False

logger = logging.getLogger(__name__)

###### Constants

_COMMAND_TIMEOUT = 120  # seconds

# Registry of command name -> handler function
_handler_registry: dict[str, Callable] = {}


class Capability(str, Enum):
    """Security capability tier for each registered MCP handler.

    READONLY   — pure introspection/query; no scene mutation, no code execution.
    MUTATING   — scene-mutating but does NOT compile/run operator-supplied code.
                 Default (fail-closed) when register_handler is called without
                 an explicit capability arg.
    CODE_EXEC  — compiles or runs operator-supplied code (exec/eval/setExpression/
                 cook-of-injected-content). Always queued below TRUSTED.
    """
    READONLY = "readonly"
    MUTATING = "mutating"
    CODE_EXEC = "code_exec"


# Registry of command name -> declared capability (populated by register_handler)
_capability_registry: dict[str, Capability] = {}


def _preview_registry() -> dict[str, dict]:
    """Return the reload-stable preview registry stashed on hou.session.

    The preview registry maps command name ->
        {"preview_fn": Callable | None, "preview_required": bool}.

    The registry is stored on ``hou.session._fxhoudinimcp_preview_registry``
    rather than as a bare module global so that importlib.reload() of this
    module does NOT reset it — stale registrations would silently lose their
    preview hooks after a hot-reload (CL-005 / ADV-003).

    When the session attribute is absent (first call, or after a gate reset
    that calls ``delattr(hou.session, '_fxhoudinimcp_preview_registry')``),
    the new dict is seeded from ``_preview_registry_fallback``.  The fallback
    is populated by every ``register_handler()`` call, so module-import-time
    registrations survive gate resets in the same interpreter session (e.g.
    in hython smoke tests that call ``_reset_gate()`` between checks).
    """
    # Import hou lazily so this module stays importable in plain pytest contexts
    # that mock hou (test-fixture-conventions.md §2.3).
    try:
        import hou  # type: ignore[import-untyped]
    except ImportError:
        # Off-DCC test fallback: return a module-level dict as best-effort.
        return _preview_registry_fallback
    reg = getattr(hou.session, "_fxhoudinimcp_preview_registry", None)
    if reg is None:
        # Seed from the module-level fallback so module-import-time registrations
        # survive a gate reset (delattr wipes the session attr; the fallback keeps
        # the registrations alive across resets in the same interpreter session).
        reg = dict(_preview_registry_fallback)
        hou.session._fxhoudinimcp_preview_registry = reg
    return reg


# Module-level fallback that shadows hou.session across gate resets.
# Populated by register_handler() alongside the session registry.
# When hou is importable, this is a secondary store used only to seed a
# freshly-created session registry.  When hou is not importable (off-DCC
# plain pytest), it IS the primary registry (_preview_registry returns it
# directly via the ImportError path above).
_preview_registry_fallback: dict[str, dict] = {}


def register_handler(
    command: str,
    handler: Callable,
    capability: Capability = Capability.MUTATING,
    preview_fn: Optional[Callable[[dict], dict]] = None,
    preview_required: bool = False,
) -> None:
    """Register a handler function for a command name.

    Args:
        command:          Dotted command name (e.g. "scene.get_scene_info")
        handler:          Function to call with **params
        capability:       Security tier for the gate. Defaults to MUTATING
                          (fail-closed) — a handler that omits the argument is
                          treated as mutating, never silently allowed as readonly.
                          Only CODE_EXEC and READONLY need explicit declarations;
                          MUTATING is the safe default.
        preview_fn:       Optional ``(params: dict) -> dict`` callable invoked
                          by the gate middleware before queuing a call.  Runs on
                          the main thread via hdefereval (CL-016).  When None the
                          handler has no preview hook (preview=None in the queue
                          entry).
        preview_required: When True *and* preview_fn raises or times out, the
                          gate DENIES the call rather than queuing it without a
                          preview (ADV-007).  Ignored when preview_fn is None.
    """
    _handler_registry[command] = handler
    _capability_registry[command] = capability
    # Build the preview record once; write it to BOTH stores so that the
    # module-level fallback always mirrors the session registry.  This is the
    # dual-write discipline required for gate-reset survival (pp12-111g B2/B3):
    # the smoke-test helper deletes hou.session._fxhoudinimcp_preview_registry
    # between checks, and the next _preview_registry() call re-seeds from
    # _preview_registry_fallback (see that function's docstring).
    _preview_record: dict = {
        "preview_fn": preview_fn,
        "preview_required": preview_required,
    }
    # Primary store: reload-stable hou.session dict (ADV-003 / CL-005).
    _preview_registry()[command] = _preview_record
    # Shadow store: module-level fallback so registrations survive a session-
    # attr deletion (e.g. _reset_gate in hython smoke tests).
    _preview_registry_fallback[command] = _preview_record


def capability_of(command: str) -> Capability | None:
    """Return the declared capability tier for *command*, or None if undeclared."""
    return _capability_registry.get(command)


def preview_of(command: str) -> dict:
    """Return the preview registration dict for *command*.

    Returns a dict with keys:
        ``preview_fn``       -- Callable[[dict], dict] or None
        ``preview_required`` -- bool (True = DENY on raise/timeout; False = degrade)

    Returns ``{"preview_fn": None, "preview_required": False}`` when the command
    is unregistered or has no preview hook.  Callers (middleware) should treat a
    missing registration the same as an explicitly-None ``preview_fn`` -- no
    preview runs, the gate queues normally.
    """
    return _preview_registry().get(command, {"preview_fn": None, "preview_required": False})


def list_commands() -> list[str]:
    """Return all registered command names."""
    return sorted(_handler_registry.keys())


def dispatch(command: str, params: dict[str, Any]) -> dict[str, Any]:
    """Execute a command on the main thread and return the result.

    This is called from hwebserver worker threads. It uses
    hdefereval.executeInMainThreadWithResult() to safely execute
    hou.* calls on the main thread.

    Args:
        command: The command name to execute
        params: Parameters to pass to the handler

    Returns:
        A response dict with "status", "data"/"error", and "timing_ms" keys.
    """
    handler = _handler_registry.get(command)
    if handler is None:
        return {
            "status": "error",
            "error": {
                "code": "UNKNOWN_COMMAND",
                "message": f"No handler registered for command: {command}",
                "available_commands": list_commands(),
            },
        }

    start_time = time.time()

    def _execute():
        try:
            result = handler(**params)
            return {"status": "success", "data": result}
        except Exception as e:
            return {
                "status": "error",
                "error": {
                    "code": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                },
            }

    try:
        if HAS_HDEFEREVAL:
            # Run hdefereval call in a worker thread so we can enforce a timeout
            container: dict[str, Any] = {}

            def _run():
                try:
                    container["result"] = hdefereval.executeInMainThreadWithResult(_execute)
                except Exception as exc:
                    container["error"] = exc
                    container["tb"] = traceback.format_exc()

            worker = threading.Thread(target=_run, daemon=True)
            worker.start()
            worker.join(timeout=_COMMAND_TIMEOUT)

            if worker.is_alive():
                logger.error(
                    "Command '%s' timed out after %s seconds", command, _COMMAND_TIMEOUT
                )
                result = {
                    "status": "error",
                    "error": {
                        "code": "TIMEOUT",
                        "message": (
                            f"Command '{command}' did not complete within "
                            f"{_COMMAND_TIMEOUT} seconds."
                        ),
                    },
                }
            elif "error" in container:
                result = {
                    "status": "error",
                    "error": {
                        "code": "DISPATCH_ERROR",
                        "message": f"Failed to dispatch to main thread: {container['error']}",
                        "traceback": container.get("tb", ""),
                    },
                }
            else:
                result = container["result"]
        else:
            # Fallback for hython (single-threaded, no hdefereval needed)
            result = _execute()
    except Exception as e:
        result = {
            "status": "error",
            "error": {
                "code": "DISPATCH_ERROR",
                "message": f"Failed to dispatch to main thread: {e}",
                "traceback": traceback.format_exc(),
            },
        }

    result["timing_ms"] = round((time.time() - start_time) * 1000, 2)
    return result
