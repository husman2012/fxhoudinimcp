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


# ---------------------------------------------------------------------------
# PR-4 -- GATED bonedeform wiring tool (require_approval=True)
# ---------------------------------------------------------------------------

@mcp.tool(meta={"require_approval": True})
async def houdini_setup_bonedeform(
    ctx: Context,
    rest: str,
    anim: str,
    geo: str,
    dest: str = "/obj",
) -> dict[str, Any]:
    """Wire a bonedeform SOP from rest skeleton, animated skeleton, and captured skin geo (GATED).

    Creates a ``bonedeform`` SOP under *dest*, wires the three inputs in the
    authoritative probe order, cooks, and returns a verify-after-mutate envelope
    (FR-12).

    Input wiring (authoritative probed order — plan riskNotes):
      * input 0 -- ``geo``  (Geometry to Deform)
      * input 1 -- ``rest`` (Rest Point Transforms)
      * input 2 -- ``anim`` (Deform Point Transforms)

    Capability: MUTATING -- routed through the PP12-109 security gate.

    Returns::

        {
            "ok": True,
            "node": "<created node path>",
            "skeleton": {
                "joints": <int>,          # point count from anim node @name attrib
                "frame_range": [s, e]     # playbar range, or null if unreadable
            },
            "validator": {
                "cook_errors": [],
                "deformed_points": <int>, # point count from bd.geometry()
                "has_capture_weight": <bool>,  # read from INPUT geo, not bd output
                "note": "verify-after-mutate"
            }
        }

    On cook error::

        {"ok": False, "error": "<node error messages>"}

    Args:
        ctx: MCP request context (injected by FastMCP).
        rest: Houdini scene path to the rest skeleton SOP node.
        anim: Houdini scene path to the animated skeleton SOP node.
        geo:  Houdini scene path to the captured skin geometry SOP node.
        dest: Houdini scene path under which to create the bonedeform node
              (default ``/obj``).
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute(
        "setup_bonedeform",
        {"rest": rest, "anim": anim, "geo": geo, "dest": dest},
    )


# ---------------------------------------------------------------------------
# PR-5 -- GATED retarget wiring tool (require_approval=True)
# ---------------------------------------------------------------------------

@mcp.tool(meta={"require_approval": True})
async def houdini_setup_retarget(
    ctx: Context,
    source: str,
    target: str,
    method: str = "rigmatchpose+fullbodyik",
    match_size: bool = True,
    mapping: list | None = None,
    dest: str = "/obj",
) -> dict[str, Any]:
    """Wire a KineFX retarget chain from source skeleton to target skeleton (GATED).

    Creates a ``kinefx::rigmatchpose`` → ``kinefx::fullbodyik`` network under
    *dest* (or with an intermediate ``kinefx::mappoints`` when an explicit
    *mapping* is provided), cooks the chain, and returns a verify-after-mutate
    envelope (FR-12).

    KineFX retarget chain (H21 authoritative — live probed 2026-06-23):
      * By-name path (no mapping): ``rigmatchpose → fullbodyik``
        (fullbodyik.mapusing = "matchattrib", attribtomatch = "name")
      * Explicit-mapping path: ``rigmatchpose → mappoints → fullbodyik``
        (fullbodyik.mapusing = "mappingattrib")

    Connector order (authoritative):
      * setInput(0) = Target Skeleton
      * setInput(1) = Source Skeleton

    Capability: MUTATING -- routed through the PP12-109 security gate.

    Returns::

        {
            "ok": True,
            "retarget_node": "<fullbodyik node path>",
            "target_skeleton": {
                "joints": <int>,           # joint count from @name attrib
                "frame_range": [s, e]      # playbar range, or null if unreadable
            },
            "validator": {
                "unmapped_target_joints": [...],  # target joints with no mapping
                "cook_errors": [],
                "note": "verify-after-mutate"
            }
        }

    On cook error::

        {"ok": False, "error": "<node error messages>"}

    Args:
        ctx: MCP request context (injected by FastMCP).
        source: Houdini scene path to the source skeleton SOP node.
        target: Houdini scene path to the target skeleton SOP node.
        method: Retarget method string (default ``"rigmatchpose+fullbodyik"``).
        match_size: When ``True``, the rigmatchpose node will attempt to match
                    skeleton scale (default ``True``).
        mapping: Optional list of ``[source_joint, target_joint]`` pairs.
                 When provided, inserts a ``kinefx::mappoints`` node and sets
                 ``fullbodyik.mapusing`` to ``"mappingattrib"``.
                 When ``None``, uses the by-name automatic path.
        dest: Houdini scene path under which to create the retarget network
              (default ``/obj``).
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute(
        "setup_retarget",
        {
            "source": source,
            "target": target,
            "method": method,
            "match_size": match_size,
            "mapping": mapping,
            "dest": dest,
        },
    )


# ---------------------------------------------------------------------------
# PR-6 -- GATED secondary motion tool (require_approval=True)
# ---------------------------------------------------------------------------

@mcp.tool(meta={"require_approval": True})
async def houdini_apply_secondarymotion(
    ctx: Context,
    node: str,
    joints: list | None = None,
    params: dict | None = None,
    dest: str = "/obj",
) -> dict[str, Any]:
    """Apply kinefx::secondarymotion to a skeleton SOP (GATED -- mutating).

    Wires a single ``kinefx::secondarymotion`` SOP onto a skeleton node,
    sets ``jointgroup`` from the joints list and effect parameters from the
    pass-through params dict (real probe-confirmed parm names), cooks it so
    the chosen effect (lagovershoot / jiggle / spring) is applied WITHOUT
    a simulation, and returns ``{ok, node, affected_joints, frame_range}``.

    Capability: MUTATING -- routed through the PP12-109 security gate.

    Returns::

        {
            "ok": True,
            "node": "<created node path>",
            "affected_joints": <int>,       # count of joints affected
            "frame_range": [s, e],          # playbar range
            "ignored_params": [...]         # only present when non-empty
        }

    On error::

        {"ok": False, "error": "<message>"}

    Args:
        ctx: MCP request context (injected by FastMCP).
        node: Houdini scene path to the skeleton SOP node to attach to.
        joints: Optional list of joint names to apply secondarymotion to.
                When None or empty, applies to ALL joints.
        params: Optional dict of parm-name → value for the secondarymotion
                node.  Keys are real Houdini parm names (probe-confirmed):
                ``effect``, ``effectmult``, ``lag``, ``overshoot``,
                ``stiffness``, ``jiggledamping``, ``limit``, ``flex``,
                ``multiplier``, ``springconstant``, ``mass``, ``damping``.
                Scalar values are broadcast to multi-component parmTuples.
                Unknown keys are silently collected into ``ignored_params``.
        dest: Houdini scene path under which to create the secondarymotion
              node (default ``/obj``).
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute(
        "apply_secondarymotion",
        {"node": node, "joints": joints, "params": params, "dest": dest},
    )
