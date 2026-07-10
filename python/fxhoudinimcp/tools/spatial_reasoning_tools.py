"""MCP wrappers: houdini_describe_relations, houdini_assert_scene.

Both are READ-ONLY, UNGATED (require_approval=False, Capability.READONLY
handler-side) -- the Gate-1 read surface of the Spatial-Reasoning MCP
member (PP12-116 PR-2). solve_layout (the MUTATING/gated tool) is PR-3,
out of scope here.

houdini_describe_relations -- the anti-hallucination vocabulary tool: a
                               pure delegate to
                               spatial_reasoning_model.describe_relations()
                               over the bridge (no params).
houdini_assert_scene       -- the Gate-1 spatial oracle: reads each
                               object's world-space bounding box +
                               placement from the live scene (resolving
                               both ObjNode and SopNode paths), derives the
                               pure model's inputs, and returns its exact
                               SPEC 4.1 assertion dict.

Each wrapper delegates to the correspondingly named handler registered on
the Houdini side via bridge.execute. No domain logic lives here.

Contract: imports NO hou, NO pxr -- this module must be importable
off-DCC for the wrapper pytest suite (CL-015).

PP12-116 / pp12-116b (houdini_describe_relations, houdini_assert_scene --
                       PR-2 of member 116)
"""
from __future__ import annotations

from mcp.server.fastmcp import Context

import fxhoudinimcp.server as _fxserver

# mcp is used by the @mcp.tool() decorator at module import time.
mcp = _fxserver.mcp


@mcp.tool(meta={"require_approval": False})
async def houdini_describe_relations(ctx: Context) -> dict:
    """Return the relation vocabulary -- the anti-hallucination reference
    for every relation type assert_scene/solve_layout accept.

    READ-ONLY / UNGATED -- a pure vocabulary read that cannot fail on scene
    state.

    Returns::

        {
            "relations": [
                {"name": str, "params": {...}, "desc": str},
                ...
            ]
        }

    Args:
        ctx: MCP lifespan context -- injected by FastMCP; hidden from client schema.
    """
    # Access _get_bridge through the module reference so that
    # `patch("fxhoudinimcp.server._get_bridge", ...)` intercepts it correctly
    # in tests (a local import would cache the original function object).
    bridge = _fxserver._get_bridge(ctx)
    return await bridge.execute("describe_relations", {})


@mcp.tool(meta={"require_approval": False})
async def houdini_assert_scene(
    ctx: Context,
    objects: list,
    transforms: "dict | None" = None,
    relations: "list | None" = None,
    checks: "list | None" = None,
) -> dict:
    """The Gate-1 spatial oracle: assert collision/support/navigability/
    clearance/relations over a scene, resolving object placement + bbox
    from the live scene where the caller doesn't supply it. READ-ONLY /
    UNGATED -- a bounding-box read may trigger an implicit cook, which is
    READ-consistent with the shipped geometry.get_bounding_box precedent,
    not a mutation.

    Returns the pure model's exact SPEC 4.1 assertion dict VERBATIM on
    success (no added 'ok' key -- 'pass' is the assertion result, not a
    tool-execution flag), or {"ok": False, "error": "<reason>"} on a
    scene-resolution failure (a missing/unresolvable node). A caller-
    contract error (an unknown relation type / missing reference /
    out-of-set param) propagates as the dispatcher's standard error
    envelope, not a normal return value.

    A SINGLE bridge.execute call -- the wrapper performs no result
    interpretation and returns bridge.execute's result VERBATIM.

    Args:
        ctx: MCP lifespan context -- injected by FastMCP; hidden from client schema.
        objects: List of {"id", "bbox": [w,d,h]|None, "fixed"?, "node"?}
            wire dicts. bbox is read from the live scene (via node) when
            None.
        transforms: Optional {"id": {"t": [x,y,z], "r": [rx,ry,rz]}} dict.
            When an object's id is present, its transform is used verbatim
            (no hou read); otherwise it is read from the live scene (via
            node) as the world-AABB center + world rotation.
        relations: Optional list of relation wire dicts to check.
        checks: Optional list of check names (e.g. ["collision",
            "support"]). Defaults to ["collision", "support"] handler-side
            when None.
    """
    bridge = _fxserver._get_bridge(ctx)
    return await bridge.execute(
        "assert_scene",
        {
            "objects": objects,
            "transforms": transforms,
            "relations": relations,
            "checks": checks,
        },
    )
