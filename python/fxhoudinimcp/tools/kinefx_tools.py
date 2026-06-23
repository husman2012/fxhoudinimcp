"""MCP tool wrappers for the KineFX/APEX handlers (PP12-110 PR-2 + PR-3).

PR-2 (read-only): kinefx_probe, query_skeleton, inspect_apex — inspect
existing cooked geometry / APEX graph state without creating nodes or writing.

PR-3 (mutating / gated): houdini_import_fbx_character, houdini_import_fbx_animation --
create + cook kinefx::fbxcharacterimport / fbxanimimport nodes, then return a
skeleton summary.

All five tools call the bridge using the canonical convention:
    bridge = _get_bridge(ctx)
    return await bridge.execute("<command>", {<params dict>})

This matches every other tool in this codebase (e.g. tools/nodes.py).
bridge.call() does NOT exist on HoudiniBridge -- never use it.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context

from fxhoudinimcp.server import mcp, _get_bridge


@mcp.tool(meta={"require_approval": False})
async def kinefx_probe(ctx: Context, node_path: str = "/obj") -> dict[str, Any]:
    """Probe which KineFX/APEX node types exist at or under node_path.

    Returns a dict with Houdini version and a boolean map of
    the 7 canonical KineFX/APEX node types.

    Args:
        ctx: MCP request context (injected by FastMCP).
        node_path: Scene path to inspect (default ``/obj``).
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute("kinefx_probe", {"node_path": node_path})


@mcp.tool(meta={"require_approval": False})
async def query_skeleton(
    ctx: Context,
    node_path: str,
    frame: float | None = None,
) -> dict[str, Any]:
    """Read joint hierarchy and transforms from a cooked skeleton SOP.

    Serialises the skeleton using the section 7.3 JSON shape::

        {"count": N, "joints": [{"name": ..., "parent": ..., "rest": {...}}, ...]}

    Args:
        ctx: MCP request context (injected by FastMCP).
        node_path: Path to the skeleton SOP node.
        frame: Optional frame number to sample; uses current frame if None.
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute("query_skeleton", {"node_path": node_path, "frame": frame})


@mcp.tool(meta={"require_approval": False})
async def inspect_apex(ctx: Context, node_path: str) -> dict[str, Any]:
    """Summarise an APEX graph node -- nodes, wires, control count.

    Returns::

        {
            "nodes": [{"name": ..., "node_type": ..., "ports": [...]}, ...],
            "wires": [{"src": ..., "dst": ...}, ...],
            "control_count": <int>
        }

    Args:
        ctx: MCP request context (injected by FastMCP).
        node_path: Path to the APEX graph node.
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute("inspect_apex", {"node_path": node_path})


# ---------------------------------------------------------------------------
# PR-3 -- GATED mutating import tools (require_approval=True)
# ---------------------------------------------------------------------------

@mcp.tool(meta={"require_approval": True})
async def houdini_import_fbx_character(
    ctx: Context,
    path: str,
    dest: str = "/obj",
) -> dict[str, Any]:
    """Import an FBX character rig via kinefx::fbxcharacterimport (GATED -- mutating).

    Creates a ``kinefx::fbxcharacterimport`` node under *dest*, cooks it, and
    returns a skeleton summary for verify-after-mutate (FR-12).

    Capability: MUTATING -- routed through the PP12-109 security gate.

    Returns::

        {
            "ok": True,
            "node": "<created node path>",
            "skeleton": {
                "joints": <int>,       # joint count from OUT 1 (deformation skeleton)
                "has_skin_geo": <bool> # True when OUT 0 (skin mesh) has points
            }
        }

    On cook error::

        {"ok": False, "error": "<node error messages>"}

    Args:
        ctx: MCP request context (injected by FastMCP).
        path: Absolute path to the ``.fbx`` file on disk.
        dest: Houdini scene path under which to create the import node
              (default ``/obj``).
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute("import_fbx_character", {"path": path, "dest": dest})


@mcp.tool(meta={"require_approval": True})
async def houdini_import_fbx_animation(
    ctx: Context,
    path: str,
    dest: str = "/obj",
    cascadeur: bool = False,
) -> dict[str, Any]:
    """Import an FBX animation clip via kinefx::fbxanimimport (Cascadeur first-class, GATED).

    Creates a ``kinefx::fbxanimimport`` node under *dest*, cooks it, and returns
    a skeleton summary for verify-after-mutate (FR-12).

    When *cascadeur* is ``True``, the ``convertunits`` parm is set on the
    import node to handle Cascadeur's non-standard unit conventions (FR-3,
    confirmed present via hython probe 2026-06-22).

    Capability: MUTATING -- routed through the PP12-109 security gate.

    Returns::

        {
            "ok": True,
            "node": "<created node path>",
            "skeleton": {
                "joints": <int>,
                "frame_range": [<start_frame>, <end_frame>]  # when readable
            }
        }

    On cook error::

        {"ok": False, "error": "<node error messages>"}

    Args:
        ctx: MCP request context (injected by FastMCP).
        path: Absolute path to the ``.fbx`` file on disk.
        dest: Houdini scene path under which to create the import node
              (default ``/obj``).
        cascadeur: When ``True``, sets ``convertunits`` parm for Cascadeur FBX
                   files.  Default ``False``.
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute(
        "import_fbx_animation", {"path": path, "dest": dest, "cascadeur": cascadeur}
    )
