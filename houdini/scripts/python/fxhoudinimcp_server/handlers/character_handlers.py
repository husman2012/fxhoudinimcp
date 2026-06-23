"""KineFX / APEX handlers for fxhoudinimcp (PP12-110 PR-2 + PR-3).

PR-2 (read-only): kinefx_probe, query_skeleton, inspect_apex — inspect
existing cooked geometry / APEX graph state without creating nodes or writing.

PR-3 (mutating / gated): import_fbx_character, import_fbx_animation —
create + cook kinefx::fbxcharacter/animimport nodes, then return a
skeleton summary.  Registered with Capability.MUTATING.

FR-2 (fail-loud) contract for MUTATING handlers:
    NO code path may raise past the handler boundary.  Every failure MUST
    return {"ok": False, "error": "<msg>"}.  An outer try/except Exception
    wraps the entire mutating body after param validation.

FR-4 (dest contract) for MUTATING handlers:
    - If dest is explicitly provided and hou.node(dest) returns None → fail loud.
    - If dest is omitted (defaults to "/obj") OR resolves to an OBJ-context
      network manager (hou.objNodeTypeCategory), create/reuse a geo container
      and place the SOP import node inside it.  This is required because
      kinefx::fbxcharacterimport and kinefx::fbxanimimport are SOP types and
      cannot be placed directly under an OBJ-context network.
"""

from __future__ import annotations

import logging
import hou
from fxhoudinimcp_server.dispatcher import Capability, register_handler

_log = logging.getLogger(__name__)

# ── kinefx_model reuse (PR-1) — sys.path bootstrap for Houdini load ─────────
# fxhoudinimcp (MCP-client) lives in <fork>/python/, which is NOT on Houdini's
# Python path.  Compute fork root as 5 dirs above handlers/, then append
# /python so importlib can locate the package.
import os as _os, sys as _sys
_PY = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..", "python")
)
if _PY not in _sys.path:
    _sys.path.insert(0, _PY)
from fxhoudinimcp import kinefx_model

# ── type-resolution helper (reused from graph_handlers — FR-1) ───────────────
from fxhoudinimcp_server.handlers.graph_handlers import _resolve_node_type

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The 7 FR-1 KineFX/APEX node types whose build-level availability is probed.
# These are SOP-context types; all resolve via hou.sopNodeTypeCategory().
_KINEFX_NODE_TYPES = [
    "kinefx::fbxcharacterimport",
    "kinefx::fbxanimimport",
    "bonedeform",
    "rigmatchpose",
    "motiontransform",
    "kinefx::secondarymotion",
    "apex::autorigcomponent",
]


def _get_node(node_path: str) -> hou.Node:
    node = hou.node(node_path)
    if node is None:
        raise ValueError(f"Node not found: {node_path!r}")
    return node


def _resolve_sop_parent(dest: str | None, geo_name: str) -> hou.Node:
    """Return a SOP-context parent node suitable for FBX import nodes.

    Rules (FR-4 dest contract):
      * If *dest* was explicitly provided by the caller (not the sentinel "/obj"
        default) and hou.node(dest) is None → raise ValueError so the outer
        FR-2 envelope can return {ok: False, error: ...}.
      * If *dest* is None or "/obj" (the default sentinel) → the caller did not
        specify a destination.  Create or reuse a geo container under /obj so
        the SOP import node has a valid SOP-context parent.
      * If *dest* resolves to an OBJ-category network (hou.objNodeTypeCategory)
        → same treatment: create/reuse a geo container inside it.  Placing a
        SOP type directly under an OBJ manager raises hou.OperationFailed.
      * Otherwise *dest* is assumed to be a valid SOP-context parent (e.g.
        /obj/geo1) — return it directly.

    The sentinel value "/obj" is the module-level default for both handlers
    and signals "caller did not supply a dest" rather than "caller explicitly
    wants /obj as the parent".

    Args:
        dest: the raw dest string from params (None → not supplied).
        geo_name: name for the geo node created as a SOP container
                  (e.g. "mcp_fbx_char" or "mcp_fbx_anim").

    Returns:
        A hou.Node whose category is SOP-context (suitable for createNode of
        kinefx::fbxcharacterimport / kinefx::fbxanimimport).

    Raises:
        ValueError: when dest was explicitly provided but does not exist in the
                    current scene.
    """
    _DEFAULT_DEST = "/obj"

    if dest is None or dest == _DEFAULT_DEST:
        # Caller did not supply a destination — use /obj and wrap in a geo node.
        obj_net = hou.node("/obj")
        if obj_net is None:
            raise ValueError("Scene /obj network not found — is a Houdini session active?")
        # Reuse an existing geo node of the same name, or create a fresh one.
        existing = obj_net.node(geo_name)
        return existing if existing is not None else obj_net.createNode("geo", geo_name)

    # Caller supplied an explicit dest — it MUST exist.
    parent = hou.node(dest)
    if parent is None:
        raise ValueError(f"dest node not found: {dest!r}")

    # If dest is an OBJ-category network, wrap in a geo container.
    if parent.childTypeCategory() == hou.objNodeTypeCategory():
        existing = parent.node(geo_name)
        return existing if existing is not None else parent.createNode("geo", geo_name)

    # dest is a valid SOP-context parent (e.g. /obj/my_geo).
    return parent


# ---------------------------------------------------------------------------
# kinefx_probe
# ---------------------------------------------------------------------------

def kinefx_probe(node_path: str = "/obj") -> dict:
    """Report which KineFX/APEX node types resolve in the installed Houdini build.

    This is a build-level type-availability check — it tests whether each of
    the 7 FR-1 node types is installed and resolvable via the SOP node-type
    registry, NOT whether any instances exist under a given scene path.

    Args:
        node_path: Scene path hint (unused for type-probe; reserved for future
                   scoped probes). Defaults to "/obj".

    Returns::
        {
            "houdini": "<version>",
            "nodes": {
                "kinefx::fbxcharacterimport": <bool>,
                ...
            }
        }

    A type that is absent returns False alongside its queried name so the
    caller can see exactly what was checked (fail-loud, not silent skip).
    """
    sop_category = hou.sopNodeTypeCategory()
    presence: dict[str, bool] = {}
    for type_name in _KINEFX_NODE_TYPES:
        resolved = _resolve_node_type(sop_category, type_name)
        presence[type_name] = resolved is not None

    return {
        "houdini": hou.applicationVersionString(),
        "nodes": presence,
    }


# ---------------------------------------------------------------------------
# query_skeleton
# ---------------------------------------------------------------------------

def query_skeleton(node_path: str, frame: float | None = None) -> dict:
    """Read joint hierarchy + transforms from a cooked skeleton SOP node.

    Reads the geometry at the given node path (optionally at *frame*) and
    serialises it using the PR-1 pure model (kinefx_model.derive_parents,
    kinefx_model.pack_trs, kinefx_model.skeleton_to_json).

    Does NOT cook the node; reads existing cooked geometry only.

    Args:
        node_path: Absolute path to the cooked skeleton SOP node.
        frame: Optional frame number to sample at; uses current cooked geo if None.

    Returns the §7.3 JSON shape::
        {
            "count": <int>,
            "joints": [
                {
                    "name": <str>,
                    "parent": <str|null>,
                    "rest": {
                        "t": [tx, ty, tz],
                        "r": [qx, qy, qz, qw],
                        "s": [sx, sy, sz]
                    }
                },
                ...
            ]
        }
    """
    node = _get_node(node_path)

    # Optionally sample at a specific frame
    if frame is not None:
        geo = node.geometryAtFrame(float(frame))
    else:
        geo = node.geometry()

    if geo is None:
        raise ValueError(f"No cooked geometry at {node_path!r}")

    # Read joint names from @name point attribute
    name_attr = geo.findPointAttrib("name")
    if name_attr is None:
        raise ValueError(f"No @name point attribute at {node_path!r} — not a skeleton SOP?")

    names = [pt.attribValue("name") for pt in geo.points()]

    # Read bone/parent connectivity — stored in the @boneCapturePath or
    # inferred from the rig hierarchy.  For KineFX skeletons the parent
    # relationship is encoded in the point connectivity (edges).
    # We build the edge list as (parent_idx, child_idx) pairs from prim edges.
    edges: list[tuple[str, str]] = []
    for prim in geo.prims():
        verts = prim.vertices()
        if len(verts) >= 2:
            pi0 = verts[0].point().number()
            pi1 = verts[1].point().number()
            if 0 <= pi0 < len(names) and 0 <= pi1 < len(names):
                # Convention: prim goes parent→child
                edges.append((names[pi0], names[pi1]))

    parent_map = kinefx_model.derive_parents(edges, names)

    # Read world transforms per joint point (4x4 matrix from @transform)
    transform_attr = geo.findPointAttrib("transform")

    joints: list[kinefx_model.Joint] = []
    for pt in geo.points():
        name = pt.attribValue("name")
        parent = parent_map.get(name)

        if transform_attr is not None:
            raw = pt.attribValue("transform")
            # hou returns a flat 9-element (3×3) or 16-element (4×4) tuple;
            # pack_trs expects a row-major 4×4 (16 elements)
            if len(raw) == 16:
                t, r, s = kinefx_model.pack_trs(raw)
            else:
                # 3×3 rotation only — pad to 4×4 using point world position
                # for the translation row (pt.position() gives world XYZ).
                pos = pt.position()
                flat = (
                    list(raw[0:3]) + [0.0]
                    + list(raw[3:6]) + [0.0]
                    + list(raw[6:9]) + [0.0]
                    + [pos[0], pos[1], pos[2], 1.0]
                )
                t, r, s = kinefx_model.pack_trs(flat)
        else:
            t, r, s = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0)

        joints.append(kinefx_model.Joint(name=name, parent=parent, rest=(t, r, s)))

    skeleton = kinefx_model.Skeleton(joints=joints)
    return kinefx_model.skeleton_to_json(skeleton)


# ---------------------------------------------------------------------------
# inspect_apex
# ---------------------------------------------------------------------------

def inspect_apex(node_path: str) -> dict:
    """Return a summary of an APEX graph node.

    Walks the APEX graph stored at node_path and returns an ApexGraphSummary
    JSON blob::

        {
            "nodes": [{"name": ..., "node_type": ..., "ports": [...]}, ...],
            "wires": [{"src": ..., "dst": ...}, ...],
            "control_count": <int>
        }

    Does NOT cook or modify the APEX graph.

    Args:
        node_path: Absolute path to the APEX SOP node.
    """
    node = _get_node(node_path)

    # APEX graphs are stored in the geometry as a special attribute.
    # We walk the geometry to build the summary.
    geo = node.geometry()
    if geo is None:
        raise ValueError(f"No cooked geometry at {node_path!r}")

    apex_nodes: list[kinefx_model.ApexNodeSummary] = []
    wires: list[dict] = []
    control_count = 0

    # APEX geometry uses detail attributes to store graph data.
    # The node graph is accessible via hou.apex (when available) or through
    # the packed prim / detail attrib approach.
    #
    # For the purposes of this read-only tool we enumerate what we can from
    # the geometry attributes and packed prims.
    try:
        import hou.apex as hou_apex  # type: ignore[import]
        graph = hou_apex.Graph(geo)
        for n in graph.nodes():
            ports = [p.name() for p in n.ports()]
            node_type = n.typeName() if hasattr(n, "typeName") else str(type(n).__name__)
            summary = kinefx_model.ApexNodeSummary(
                name=n.name(),
                node_type=node_type,
                ports=ports,
            )
            apex_nodes.append(summary)
            if "control" in node_type.lower():
                control_count += 1

        for wire in graph.wires():
            wires.append({"src": str(wire.src()), "dst": str(wire.dst())})

    except (ImportError, AttributeError):
        # Fallback: read from detail string attrib "__apex_graph" if present
        attrib = geo.findGlobalAttrib("__apex_graph")
        if attrib:
            import json as _json
            raw = _json.loads(geo.attribValue("__apex_graph"))
            for n in raw.get("nodes", []):
                apex_nodes.append(kinefx_model.ApexNodeSummary(
                    name=n.get("name", ""),
                    node_type=n.get("type", ""),
                    ports=n.get("ports", []),
                ))
            wires = raw.get("wires", [])
            control_count = raw.get("control_count", 0)
        else:
            # Return a minimal stub so the caller knows we queried but found nothing
            return kinefx_model.ApexGraphSummary(nodes=[], wires=[], control_count=0).to_dict()

    graph_summary = kinefx_model.ApexGraphSummary(
        nodes=apex_nodes,
        wires=wires,
        control_count=control_count,
    )
    return graph_summary.to_dict()


# ---------------------------------------------------------------------------
# import_fbx_character
# ---------------------------------------------------------------------------

def import_fbx_character(path: str, dest: str = None) -> dict:
    """Import an FBX character rig via kinefx::fbxcharacterimport.

    Creates the import node under *dest*, sets the FBX file path, cooks,
    and returns a skeleton summary built from the node's outputs:

    * OUT 0 — skin mesh  (has_skin_geo = True when it has points)
    * OUT 1 — deformation skeleton, 84 joints with @name attr  (the one we count)

    FR-2 (fail-loud): the ENTIRE mutating body is wrapped in an outer
    try/except Exception so that NO code path raises past the handler boundary.
    createNode, parm().set(), cook(), geometry() reads, and all other Houdini
    calls return {"ok": False, "error": "<msg>"} on any exception.

    FR-4 (dest contract): if dest is explicitly provided but does not exist in
    the scene, returns {"ok": False, "error": "dest node not found: '<dest>'"}.
    If dest is omitted (default) or is the OBJ network manager, the SOP import
    node is placed inside a geo container.

    FR-12 (verify-after-mutate): skeleton summary is embedded in the return
    envelope so the caller can inspect what was imported without a second query.

    Args:
        path: Absolute file-system path to the FBX file.
        dest: Optional scene node path for the SOP parent. Defaults to None
              (auto-creates a geo container under /obj).

    Returns::

        {
            "ok": True,
            "node": "<node path>",
            "skeleton": {
                "joints": <int>,       # point count at OUT 1 (@name attr)
                "has_skin_geo": <bool> # OUT 0 has at least one point
            }
        }

    or on error::

        {"ok": False, "error": "<message>"}
    """
    # dest=None means "caller did not supply"; "/obj" is the legacy default —
    # both are handled by _resolve_sop_parent which wraps them in a geo node.

    if not path:
        # param validation failures are the ONE category allowed to raise (they
        # are programming errors, not Houdini operation failures).
        raise ValueError("import_fbx_character requires 'path'")

    # ── FR-2 outer envelope — wraps ALL Houdini mutation + geometry reads ─────
    try:
        parent = _resolve_sop_parent(dest, "mcp_fbx_char")

        # Guard parm() result: returns None on SDK drift (FR-2 Major-2).
        imp = parent.createNode("kinefx::fbxcharacterimport", "mcp_fbxcharacterimport")
        fbxfile_parm = imp.parm("fbxfile")
        if fbxfile_parm is None:
            return {"ok": False, "error": "parm 'fbxfile' not found on kinefx::fbxcharacterimport — SDK version mismatch?"}
        fbxfile_parm.set(path)

        try:
            imp.cook(force=True)
        except Exception as exc:
            errs = imp.errors()
            err_msg = "\n".join(errs) if errs else str(exc)
            return {"ok": False, "error": err_msg}

        errs = imp.errors()
        if errs:
            return {"ok": False, "error": "\n".join(errs)}

        # OUT 1 — deformation skeleton with @name point attribute (84 joints for WorkersWelders)
        # hou.SopNode.geometry(output_index) is the correct API; geometryAtOutput() does not exist.
        # These reads are INSIDE the outer try/except (FR-2 Major-1): a malformed FBX with
        # fewer than 2 outputs raises hou.OperationFailed, which is caught below.
        skel_geo = imp.geometry(1)
        joint_count = 0
        if skel_geo is not None:
            name_attr = skel_geo.findPointAttrib("name")
            if name_attr is not None:
                joint_count = skel_geo.intrinsicValue("pointcount")

        # OUT 0 — skin mesh; has_skin_geo is True when it has at least one point
        skin_geo = imp.geometry(0)
        has_skin_geo = bool(
            skin_geo is not None and skin_geo.intrinsicValue("pointcount") > 0
        )

        return {
            "ok": True,
            "node": imp.path(),
            "skeleton": {"joints": joint_count, "has_skin_geo": has_skin_geo},
        }

    except Exception as exc:
        # FR-2: catch everything from createNode / parm.set / geometry reads.
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# import_fbx_animation
# ---------------------------------------------------------------------------

def import_fbx_animation(path: str, dest: str = None, cascadeur: bool = False) -> dict:
    """Import FBX animation via kinefx::fbxanimimport (Cascadeur first-class).

    Creates the import node under *dest*, sets the FBX file path, optionally
    sets ``convertunits`` for Cascadeur FBX files, and cooks.

    * OUT 0 (single output) — skeleton with @name attr + baked animation

    FR-2 (fail-loud): the ENTIRE mutating body is wrapped in an outer
    try/except Exception so that NO code path raises past the handler boundary.
    createNode, parm().set(), cook(), geometry() reads, and frame-range parm
    reads all return {"ok": False, "error": "<msg>"} on any exception.

    FR-3 (Cascadeur): when ``cascadeur=True``, sets the ``convertunits`` parm
    (confirmed present on kinefx::fbxanimimport via hython probe, 2026-06-22).

    FR-4 (dest contract): if dest is explicitly provided but does not exist,
    returns {"ok": False, "error": "dest node not found: '<dest>'"}.
    When dest is omitted/default, wraps in a geo container (SOP-context parent).

    FR-12 (verify-after-mutate): skeleton summary embedded in return envelope.

    Args:
        path: Absolute file-system path to the FBX file.
        dest: Optional scene node path for the SOP parent. Defaults to None
              (auto-creates a geo container under /obj).
        cascadeur: When True, sets the ``convertunits`` parm for Cascadeur FBX.

    Returns::

        {
            "ok": True,
            "node": "<node path>",
            "skeleton": {
                "joints": <int>,
                "frame_range": [<start>, <end>]  # when parms are readable
            }
        }

    or on error::

        {"ok": False, "error": "<message>"}
    """
    # dest=None means "caller did not supply"; "/obj" is the legacy default —
    # both are handled by _resolve_sop_parent which wraps them in a geo node.

    if not path:
        # param validation failures are the ONE category allowed to raise.
        raise ValueError("import_fbx_animation requires 'path'")

    # ── FR-2 outer envelope — wraps ALL Houdini mutation + geometry reads ─────
    try:
        parent = _resolve_sop_parent(dest, "mcp_fbx_anim")

        # Guard parm() result: returns None on SDK drift (FR-2 Major-2).
        imp = parent.createNode("kinefx::fbxanimimport", "mcp_fbxanimimport")
        fbxfile_parm = imp.parm("fbxfile")
        if fbxfile_parm is None:
            return {"ok": False, "error": "parm 'fbxfile' not found on kinefx::fbxanimimport — SDK version mismatch?"}
        fbxfile_parm.set(path)

        # FR-3: Cascadeur convertunits flag (parm may be absent on older builds).
        if cascadeur:
            cu_parm = imp.parm("convertunits")
            if cu_parm is not None:
                cu_parm.set(1)

        try:
            imp.cook(force=True)
        except Exception as exc:
            errs = imp.errors()
            err_msg = "\n".join(errs) if errs else str(exc)
            return {"ok": False, "error": err_msg}

        errs = imp.errors()
        if errs:
            return {"ok": False, "error": "\n".join(errs)}

        # OUT 0 — single output: skeleton with @name + baked animation data.
        # INSIDE the outer try/except (FR-2 Major-1).
        geo = imp.geometry()
        joint_count = 0
        if geo is not None:
            name_attr = geo.findPointAttrib("name")
            if name_attr is not None:
                joint_count = geo.intrinsicValue("pointcount")

        skeleton: dict = {"joints": joint_count}

        # Attempt to read frame range from FBX import parms (optional — absent = skip).
        # FR-2 Major-3 fix: narrow the bare except to specific Houdini exception types
        # and log a warning so the omission is observable rather than silent.
        try:
            start_parm = imp.parm("animationstartframe")
            end_parm = imp.parm("animationendframe")
            if start_parm is not None and end_parm is not None:
                skeleton["frame_range"] = [int(start_parm.eval()), int(end_parm.eval())]
        except (hou.OperationFailed, hou.Error, RuntimeError) as exc:
            _log.warning("import_fbx_animation: could not read frame range parms: %s", exc)

        return {"ok": True, "node": imp.path(), "skeleton": skeleton}

    except Exception as exc:
        # FR-2: catch everything from createNode / parm.set / cook / geometry reads.
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Registration — ALL three read-only + TWO new mutating
# ---------------------------------------------------------------------------

register_handler("kinefx_probe", kinefx_probe, Capability.READONLY)
register_handler("query_skeleton", query_skeleton, Capability.READONLY)
register_handler("inspect_apex", inspect_apex, Capability.READONLY)
register_handler("import_fbx_character", import_fbx_character, Capability.MUTATING)
register_handler("import_fbx_animation", import_fbx_animation, Capability.MUTATING)
