"""KineFX / APEX read-only handlers for fxhoudinimcp (PP12-110 PR-2).

All three handlers are READONLY — they inspect existing cooked geometry and
APEX graph state without creating nodes, cooking, or writing anything.

Registered via register_handler(..., Capability.READONLY).
hdefereval.executeInMainThreadWithResult is called inside the dispatcher,
not here — these are plain Python functions.
"""

from __future__ import annotations

import hou
from fxhoudinimcp_server.dispatcher import Capability, register_handler

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


# ---------------------------------------------------------------------------
# kinefx_probe
# ---------------------------------------------------------------------------

def kinefx_probe(params: dict) -> dict:
    """Report which KineFX/APEX node types resolve in the installed Houdini build.

    This is a build-level type-availability check — it tests whether each of
    the 7 FR-1 node types is installed and resolvable via the SOP node-type
    registry, NOT whether any instances exist under a given scene path.

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

def query_skeleton(params: dict) -> dict:
    """Read joint hierarchy + transforms from a cooked skeleton SOP node.

    Reads the geometry at the given node path (optionally at *frame*) and
    serialises it using the PR-1 pure model (kinefx_model.derive_parents,
    kinefx_model.pack_trs, kinefx_model.skeleton_to_json).

    Does NOT cook the node; reads existing cooked geometry only.

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
    node_path: str = params.get("node_path")
    if not node_path:
        raise ValueError("query_skeleton requires 'node_path'")

    frame = params.get("frame")
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

def inspect_apex(params: dict) -> dict:
    """Return a summary of an APEX graph node.

    Walks the APEX graph stored at node_path and returns an ApexGraphSummary
    JSON blob::

        {
            "nodes": [{"name": ..., "node_type": ..., "ports": [...]}, ...],
            "wires": [{"src": ..., "dst": ...}, ...],
            "control_count": <int>
        }

    Does NOT cook or modify the APEX graph.
    """
    node_path: str = params.get("node_path")
    if not node_path:
        raise ValueError("inspect_apex requires 'node_path'")

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
# Registration — ALL three are READONLY
# ---------------------------------------------------------------------------

register_handler("kinefx_probe", kinefx_probe, Capability.READONLY)
register_handler("query_skeleton", query_skeleton, Capability.READONLY)
register_handler("inspect_apex", inspect_apex, Capability.READONLY)
