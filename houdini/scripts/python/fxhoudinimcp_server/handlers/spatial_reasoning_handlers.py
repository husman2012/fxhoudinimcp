"""Handlers: describe_relations, assert_scene.

Both are READ-ONLY, UNGATED (Capability.READONLY) -- the Gate-1 read surface
of the Spatial-Reasoning MCP member (PP12-116 PR-2).

describe_relations -- a PURE delegate (no hou) to
                       spatial_reasoning_model.describe_relations() -- the
                       exact SPEC 4.1 relation-vocabulary wire shape,
                       verbatim.
assert_scene        -- the Gate-1 spatial oracle. Resolves each wire
                       object's WORLD-SPACE axis-aligned bounding box +
                       placement from the LIVE scene via hou (both
                       hou.ObjNode and hou.SopNode paths), derives the pure
                       model's inputs (t = the WORLD AABB CENTER -- NEVER
                       worldTransform().extractTranslates(), the node
                       origin/pivot; bbox = the WORLD extents mapped Y-up;
                       r = the node's world rotation in degrees), builds the
                       pure-model ObjectSpec/RelationSpec via the model's
                       module-private _object_from_dict/_relation_from_dict
                       (there is NO ObjectSpec.from_dict), and delegates ALL
                       geometry math to the byte-frozen PR-1
                       spatial_reasoning_model.assert_scene.

COORDINATE FRAME (BINDING, plan pp12-116b lockedFieldContract revision 2,
Blocker-3): the model treats transforms[id]['t'] (its `pos`) as the
object's WORLD-SPACE AABB CENTER -- _collision_check/_support_check
(spatial_reasoning_model.py:500-516) use pos[id] +/- half[id], i.e. pos IS
the box center, NOT the node origin. The handler's hou-read `t` MUST
therefore be the world-AABB CENTER, computed by transforming the EIGHT
corners of the SOP-local bounding box by the node's worldTransform() and
taking the component-wise min/max of the transformed corners -- a rotation-
correct world AABB, not the local bbox nudged by the origin.
bbox=(w, d, h) are the world extents mapped w=x-extent, d=z-extent
(depth), h=y-extent (vertical), per the pinned Y-up frame
(spatial_reasoning_model.py module docstring).

GEOMETRY RESOLUTION (Blocker-2): node.geometry() exists ONLY on
hou.SopNode. The spec's example paths (e.g. /obj/table) are OBJ paths, so
an hou.ObjNode is resolved via displayNode().geometry() instead. A node
that is neither a SopNode nor an ObjNode is a scene-resolution error.

WORLD-TRANSFORM SOURCE (fix-cycle 2, codex-adversarial-reviewer Blocker):
hou.SopNode has NO worldTransform() of its own -- it is hou.ObjNode-only
(verified against the shipped Houdini 21.0.729 hou.py). A SopNode's world
placement is resolved via its CONTAINING OBJECT, found by walking
node.parent() up to the nearest hou.ObjNode ancestor; that ancestor's
worldTransform() is used. The ObjNode branch uses the ObjNode itself as
the transform source. An unresolvable ancestor (node.parent() walks to
None before reaching an ObjNode) is a scene-resolution error.

PER-OBJECT RESOLUTION TABLE (Major-1): transform (t/r) and bbox are
resolved INDEPENDENTLY per wire object -- a caller-supplied value is used
verbatim (no hou read for that piece); an omitted value is read from the
live scene (requiring a resolvable node). The model REQUIRES a transform
for every object id, so transform_final is always resolved (caller or hou)
before the model call.

SUCCESS vs ERROR BOUNDARY (Blocker-4): the SCENE-RESOLUTION phase (all hou
reads: hou.node, isinstance/displayNode, geo.boundingBox, worldTransform)
is wrapped so a hou.OperationFailed / AttributeError / resolution hard
error degrades to {"ok": False, "error": "<reason>"} WITHOUT raising --
mirroring the shipped cop_onnx read-tool {ok:false,error} convention. The
MODEL CALL is effectively outside that guarantee: the narrow except clause
below catches ONLY (hou.OperationFailed, AttributeError), so a
caller-contract ValueError raised by the model's own constructors
(_relation_from_dict -- an unknown relation type) or by
spatial_reasoning_model.assert_scene itself PROPAGATES as a normal
exception, which the dispatcher (fxhoudinimcp_server.dispatcher.dispatch)
turns into its standard error envelope -- it is NEVER folded into
{"ok": False}.

This is a Gate-1 STATIC approximation (the AABB absorbs rotation into its
extents while r drives oriented_toward facing) -- consistent with spec
Sections 6-FR-B/10-R2 "static approximation, not an RBD settle".

Grounded LAYER-FOR-LAYER on:
  - handlers/cop_onnx_handlers.py -- the shipped READONLY handler-
    registration convention (keyword-only def fn(*, ...) +
    register_handler(cmd, fn, Capability.READONLY)) and the sys.path
    bootstrap needed to reach the fxhoudinimcp package from hython.
  - handlers/geometry_handlers.py's _get_sop_geo / _get_bounding_box --
    the shipped bbox-read discipline (SOP-only; this handler EXTENDS it
    with the ObjNode->displayNode branch + world-transform composition
    the /obj paths and the model's world-center contract require).

Cross-references
-----------------
Plan pp12-116b lockedFieldContract (BINDING, revision 2)
docs/portfolio_projects/PP12_houdini_mcp_agentic_bridge/
  116_mcp_spatial_reasoning_surface/spec.md Sections 4.1, 4.3, 5, 6, 7
CL-015 (extended): only *_model.py stays hou/Qt/pxr-free; this handler MAY
  (and does) import hou -- it is the hou-facing layer PR-2 ships.
CL-016: hou reads happen on the dispatcher's main-thread marshal
  (hdefereval); this handler does not itself call hdefereval.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# sys.path bootstrap -- 5 levels up from this file reaches the fork root;
# +/python adds the FastMCP-side fxhoudinimcp package (spatial_reasoning_model
# lives there) so the import below resolves when this module is imported
# from hython. Mirrors cop_onnx_handlers.py exactly.
#
#  __file__: .../fxhoudinimcp/houdini/scripts/python/fxhoudinimcp_server/handlers/spatial_reasoning_handlers.py
#   1 up -> .../handlers/
#   2 up -> .../fxhoudinimcp_server/
#   3 up -> .../python/
#   4 up -> .../scripts/
#   5 up -> .../houdini/
#   6 up -> .../fxhoudinimcp/             (fork root)
#  +/python -> .../fxhoudinimcp/python/
# ---------------------------------------------------------------------------
import os as _os
import sys as _sys

_PY = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..", "python")
)
if _PY not in _sys.path:
    _sys.path.insert(0, _PY)

try:
    import hou  # noqa: E402  (hython / Houdini-side interpreter only)
except ImportError:  # pragma: no cover -- off-DCC import guard
    hou = None  # type: ignore[assignment]

from fxhoudinimcp_server.dispatcher import Capability, register_handler  # noqa: E402
from fxhoudinimcp import spatial_reasoning_model as _model  # noqa: E402


###### describe_relations (FR-C, pure delegate)

def describe_relations() -> dict:
    """Return the model's exact SPEC 4.1 relation-vocabulary dict, verbatim.

    A PURE delegate -- touches NO hou. Cannot fail on scene state, so there
    is no ok-wrapper on the return.
    """
    return _model.describe_relations()


register_handler("describe_relations", describe_relations, Capability.READONLY)


###### assert_scene (FR-B / Gate-1) -- geometry resolution helpers

def _transform_point(x: float, y: float, z: float, matrix) -> tuple:
    """Transform a local point by a world/affine 4x4 matrix (row-vector convention).

    Uses matrix.at(row, col) directly rather than constructing a new
    hou.Vector3 and relying on Vector3.__mul__(Matrix4) -- avoids any
    dependency on how a Vector3 instance is constructed, and matches
    hou.Matrix4's documented .at(row, col) element accessor.
    """
    nx = x * matrix.at(0, 0) + y * matrix.at(1, 0) + z * matrix.at(2, 0) + matrix.at(3, 0)
    ny = x * matrix.at(0, 1) + y * matrix.at(1, 1) + z * matrix.at(2, 1) + matrix.at(3, 1)
    nz = x * matrix.at(0, 2) + y * matrix.at(1, 2) + z * matrix.at(2, 2) + matrix.at(3, 2)
    return (nx, ny, nz)


def _world_aabb_center_and_extents(node) -> tuple:
    """Resolve a node to (t, r, bbox) via a TRUE world-space AABB.

    t    -- [cx, cy, cz], the WORLD AABB CENTER (never
            worldTransform().extractTranslates(), which is the node
            origin/pivot -- Blocker-3).
    r    -- [rx, ry, rz] world rotation, degrees (worldTransform().
            extractRotates(), default rotation order -- Major-2).
    bbox -- (w, d, h) world extents mapped Y-up: w=x-extent, d=z-extent,
            h=y-extent.

    Resolves geometry for BOTH hou.SopNode (node.geometry()) and
    hou.ObjNode (node.displayNode().geometry() -- node.geometry() does not
    exist on hou.ObjNode -- Blocker-2). Raises hou.OperationFailed for any
    resolution failure (no geometry, an ObjNode with no display node, a
    SopNode with no containing hou.ObjNode ancestor, or a node that is
    neither a SopNode nor an ObjNode).

    WORLD-TRANSFORM SOURCE (fix-cycle 2 -- codex-adversarial-reviewer
    Blocker): hou.SopNode has NO worldTransform() of its own -- only
    hou.ObjNode does. A SOP's world placement is its CONTAINING OBJECT's
    worldTransform(), found by walking node.parent() up to the nearest
    hou.ObjNode ancestor. The ObjNode branch's transform node is the
    ObjNode itself; the SopNode branch's transform node is that resolved
    OBJ ancestor -- captured here as `xform_node` so a single
    `xform_node.worldTransform()` call downstream is correct for both
    branches.
    """
    if isinstance(node, hou.SopNode):
        geo = node.geometry()
        xform_node = node.parent()
        while xform_node is not None and not isinstance(xform_node, hou.ObjNode):
            xform_node = xform_node.parent()
        if xform_node is None:
            raise hou.OperationFailed(
                f"SopNode has no containing ObjNode: {node.path()}"
            )
    elif isinstance(node, hou.ObjNode):
        display_node = node.displayNode()
        if display_node is None:
            raise hou.OperationFailed(
                f"ObjNode has no display node (empty subnet?): {node.path()}"
            )
        geo = display_node.geometry()
        xform_node = node
    else:
        raise hou.OperationFailed(
            f"Node is neither a SopNode nor an ObjNode: {node.path()}"
        )

    if geo is None:
        raise hou.OperationFailed(f"Node has no geometry: {node.path()}")

    lbb = geo.boundingBox()
    wt = xform_node.worldTransform()

    minv = lbb.minvec()
    maxv = lbb.maxvec()

    world_corners = [
        _transform_point(x, y, z, wt)
        for x in (minv[0], maxv[0])
        for y in (minv[1], maxv[1])
        for z in (minv[2], maxv[2])
    ]
    world_min = [min(c[i] for c in world_corners) for i in range(3)]
    world_max = [max(c[i] for c in world_corners) for i in range(3)]

    t = [
        (world_min[0] + world_max[0]) / 2.0,
        (world_min[1] + world_max[1]) / 2.0,
        (world_min[2] + world_max[2]) / 2.0,
    ]
    ext = [world_max[i] - world_min[i] for i in range(3)]
    bbox = (ext[0], ext[2], ext[1])  # w=x-extent, d=z-extent, h=y-extent

    r = list(wt.extractRotates())

    return t, r, bbox


def _resolve_object(obj_wire: dict, transforms_wire: dict) -> tuple:
    """Resolve one wire object's (bbox_final, transform_final) per the
    per-object resolution table (Major-1): transform and bbox are resolved
    INDEPENDENTLY -- a caller-supplied value is used verbatim (no hou read
    for that piece); an omitted value is read from the live scene (which
    requires a resolvable node -- else a scene-resolution error).

    Returns (bbox_final: tuple, transform_final: dict{t, r}). Raises
    hou.OperationFailed on any hard error (no node given when one is
    needed; the node does not resolve).
    """
    oid = obj_wire["id"]
    node_path = obj_wire.get("node") or ""
    wire_bbox = obj_wire.get("bbox")
    has_transform = oid in transforms_wire
    has_bbox = wire_bbox is not None

    read_t = read_r = read_bbox = None
    if not has_transform or not has_bbox:
        if not node_path:
            missing = []
            if not has_transform:
                missing.append("transform")
            if not has_bbox:
                missing.append("bbox")
            raise hou.OperationFailed(
                f"Object {oid!r}: cannot resolve missing {'/'.join(missing)} "
                f"without a node path (supply transforms[{oid!r}] and/or "
                f"bbox, or a resolvable node)"
            )
        node = hou.node(node_path)
        if node is None:
            raise hou.OperationFailed(
                f"Node not found: {node_path} (object id={oid!r})"
            )
        read_t, read_r, read_bbox = _world_aabb_center_and_extents(node)

    if has_transform:
        wire_transform = transforms_wire[oid]
        t_final = list(wire_transform["t"])
        r_final = list(wire_transform.get("r", [0.0, 0.0, 0.0]))
    else:
        t_final = read_t
        r_final = read_r

    bbox_final = tuple(wire_bbox) if has_bbox else read_bbox

    return bbox_final, {"t": t_final, "r": r_final}


def assert_scene(
    *,
    objects: list,
    transforms: "dict | None" = None,
    relations: "list | None" = None,
    checks: "list | None" = None,
) -> dict:
    """The Gate-1 spatial oracle: resolve each object's world placement +
    bbox from the live scene (or the caller-supplied override), build the
    pure-model specs, and delegate ALL geometry math to the byte-frozen
    spatial_reasoning_model.assert_scene.

    Returns the model's exact SPEC 4.1 dict verbatim on success -- NO
    added 'ok' key ('pass' is the assertion result, not a tool-execution
    flag). A scene-resolution failure (hou.OperationFailed / AttributeError
    / a resolution hard error) returns {"ok": False, "error": "<reason>"}
    WITHOUT raising. A caller-contract error (an unknown relation type /
    missing reference / out-of-set param, raised by the model's own
    constructors or by assert_scene itself) PROPAGATES -- it is never
    caught into {"ok": False}.
    """
    transforms_wire = transforms or {}

    try:
        resolved_objects = []
        resolved_transforms: dict = {}
        for obj_wire in objects:
            oid = obj_wire["id"]
            bbox_final, transform_final = _resolve_object(obj_wire, transforms_wire)
            resolved_objects.append(_model._object_from_dict({
                "id": oid,
                "bbox": bbox_final,
                "fixed": obj_wire.get("fixed", False),
                "t": transform_final["t"],
                "r": transform_final["r"],
                "node": obj_wire.get("node", ""),
            }))
            resolved_transforms[oid] = transform_final

        relation_specs = [_model._relation_from_dict(rd) for rd in (relations or [])]
    except (hou.OperationFailed, AttributeError) as exc:
        return {"ok": False, "error": str(exc)}

    return _model.assert_scene(resolved_objects, resolved_transforms, relation_specs, checks)


register_handler("assert_scene", assert_scene, Capability.READONLY)
