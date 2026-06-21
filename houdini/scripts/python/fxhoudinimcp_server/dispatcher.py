"""Main-thread dispatch mechanism for executing hou.* calls safely.

Houdini requires all hou.* API calls to run on the main thread.
hwebserver handlers run on worker threads, so we use
hdefereval.executeInMainThreadWithResult() to marshal calls
to the main thread and block until they complete.
"""

from __future__ import annotations

# Built-in
import logging
import threading
import time
import traceback
from enum import Enum
from typing import Any, Callable

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


def register_handler(
    command: str,
    handler: Callable,
    capability: Capability = Capability.MUTATING,
) -> None:
    """Register a handler function for a command name.

    Args:
        command:    Dotted command name (e.g. "scene.get_scene_info")
        handler:    Function to call with **params
        capability: Security tier for the gate. Defaults to MUTATING (fail-closed) —
                    a handler that omits the argument is treated as mutating, never
                    silently allowed as readonly.  Only CODE_EXEC and READONLY need
                    explicit declarations; MUTATING is the safe default.
    """
    _handler_registry[command] = handler
    _capability_registry[command] = capability


def capability_of(command: str) -> Capability | None:
    """Return the declared capability tier for *command*, or None if undeclared."""
    return _capability_registry.get(command)


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
