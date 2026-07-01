"""MCP wrappers: houdini_cop_onnx_list_models, houdini_cop_onnx_inspect_model.

Both are READ-ONLY, UNGATED (require_approval=False, Capability.READONLY
handler-side) — the Copernicus-ONNX inspection surface (PP12-113 PR-2).

houdini_cop_onnx_list_models   — enumerate .onnx files under configured or
                                  given filesystem roots (path/size/mtime).
houdini_cop_onnx_inspect_model — read a model's input/output tensor
                                  contract (names/shapes/dtypes) via the
                                  Houdini-side temp-node-Setup-Shapes
                                  mechanism (cop_onnx_handlers.py); the
                                  handler GUARANTEES scratch-node cleanup
                                  in a finally, which is what keeps this
                                  tool READONLY/ungated despite creating a
                                  transient node — see the handler
                                  module's docstring for the full mechanism
                                  + the Phase-0 findings that shape it.

Each wrapper delegates to the correspondingly named handler registered on
the Houdini side via bridge.execute. No domain logic lives here.

Contract: imports NO hou, NO pxr, NO onnx — this module must be importable
off-DCC for the wrapper pytest suite (CL-015).

PP12-113 / pp12-113b (houdini_cop_onnx_list_models, houdini_cop_onnx_inspect_model)
"""
from __future__ import annotations

from mcp.server.fastmcp import Context

import fxhoudinimcp.server as _fxserver

# mcp is used by the @mcp.tool() decorator at module import time.
mcp = _fxserver.mcp


@mcp.tool(meta={"require_approval": False})
async def houdini_cop_onnx_list_models(
    ctx: Context,
    roots: "list | None" = None,
) -> dict:
    """Enumerate .onnx files under the given (or default) filesystem roots.

    Filesystem-metadata-only — no Houdini node is created, no model is
    parsed. A missing root is noted in missing_roots and never raises.

    Returns::

        {
            "ok": True,
            "models": [{"path": str, "size": int, "mtime": float}, ...],
            "roots_scanned": [str, ...],
            "missing_roots": [str, ...],
        }

    or an error shape on failure::

        {"ok": False, "error": "<reason>"}

    Args:
        ctx: MCP lifespan context — injected by FastMCP; hidden from client schema.
        roots: Optional list of Houdini-expandable root paths to scan for
            .onnx files. Defaults to the handler's built-in defaults
            (``$HIP/models``, ``$HOUDINI_USER_PREF_DIR/onnx``) when None.
    """
    # Access _get_bridge through the module reference so that
    # `patch("fxhoudinimcp.server._get_bridge", ...)` intercepts it correctly
    # in tests (a local import would cache the original function object).
    bridge = _fxserver._get_bridge(ctx)
    return await bridge.execute(
        "cop_onnx_list_models",
        {"roots": roots},
    )


@mcp.tool(meta={"require_approval": False})
async def houdini_cop_onnx_inspect_model(
    ctx: Context,
    model_path: str,
    node_path: "str | None" = None,
) -> dict:
    """Read an .onnx model's input/output tensor contract.

    Returns the OnnxContract.to_dict() shape merged with ok=True on
    success::

        {
            "ok": True,
            "model_path": str,
            "inputs": [...], "outputs": [...],
            "opset": int | None, "producer": str | None,
            "loadable": bool, "error": str | None,
        }

    or an error shape on failure::

        {"ok": False, "error": "<reason>"}

    Args:
        ctx: MCP lifespan context — injected by FastMCP; hidden from client schema.
        model_path: Path to the .onnx file (Houdini-expandable).
        node_path: Optional existing cop/onnx node path to inspect
            in-place instead of the handler's scratch-node mechanism.
    """
    bridge = _fxserver._get_bridge(ctx)
    return await bridge.execute(
        "cop_onnx_inspect_model",
        {"model_path": model_path, "node_path": node_path},
    )
