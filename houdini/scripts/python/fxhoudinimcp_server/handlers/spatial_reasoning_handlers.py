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


###### solve_layout (FR-A, MUTATING/GATED) -- PP12-116 PR-3
#
# Grounded LAYER-FOR-LAYER on the SHIPPED gated exemplar
# (cop_onnx_handlers.py: _preview_setup_node + cop_onnx_setup_node +
# register_handler(..., Capability.MUTATING, preview_fn=..., preview_
# required=True)) and this module's OWN PR-2 world-AABB resolution
# (_world_aabb_center_and_extents, BYTE-FROZEN, untouched below).
#
# COORDINATE FRAME + MODEL CONTRACT (plan pp12-116c lockedFieldContract,
# revision 2): spatial_reasoning_model.solve_layout(spec) returns
# transforms for ALL objects (fixed ones unchanged at their input t/r);
# t is the WORLD-AABB CENTER, r the WORLD rotation degrees -- identical
# to assert_scene's contract above. Construction/validation errors RAISE
# ValueError (propagates); a feasible-but-unsatisfiable spec returns
# solved:false + unsatisfied[] (never raises).
#
# APPLY MAPPING (the load-bearing write-side correctness, B1/B2/B3/M1/M2/
# M5 of the locked contract):
#   (B1) an OBJ node's t/r PARMS are LOCAL (relative to the parent
#        chain) while the model's t/r are WORLD. Apply is SCOPED to OBJ
#        nodes whose parentAndSubnetTransform() is IDENTITY (the standard
#        /obj-level case, where LOCAL == WORLD) -- a non-identity parent
#        is a scene-resolution error / preview DENY, never silently
#        corrupted.
#   (B2) setting r rotates about the pivot, moving an off-center object's
#        world center by the pivot-to-center radius -- NOT a small
#        residual. Apply order is ROTATE-THEN-REREAD-THEN-TRANSLATE:
#        (1) set parmTuple('r'); (2) RE-READ the POST-ROTATION world-AABB
#        center via _world_aabb_center_and_extents (BYTE-FROZEN);
#        (3) set parmTuple('t') to cur_objt + (solved_t -
#        post_rotation_center). See _apply_solved_transforms.
#   (B3) two objects sharing one OBJ ancestor (e.g. two SOPs in one geo)
#        fight over a single t/r pair -- ALL movable objects must resolve
#        to DISTINCT OBJ nodes, and a movable OBJ must not coincide with
#        a FIXED object's OBJ. See _check_movable_targets.
#   (M2) _world_aabb_center_and_extents returns (t, r, bbox), NEVER the
#        node -- _resolve_obj_node (new, below) is the helper that
#        resolves the actual MUTABLE OBJ node for the apply to write.
#   (M5) the whole multi-object apply PRE-VALIDATES every movable object
#        (resolvable + distinct-OBJ + identity-parent) BEFORE any parm
#        write, and runs inside exactly ONE hou.undos.group(...) context
#        (never per-object) -- see _apply_solved_transforms.
#
# GATED (Capability.MUTATING + a positional _preview_solve_layout that
# pre-validates and raises -> gate DENY + preview_required=True).
# apply=false is STILL gated (fail-safe -- gate capability is per-COMMAND,
# not per-argument).

def _resolve_obj_node(node):
    """Resolve *node* to the mutable hou.ObjNode solve_layout's apply
    writes parmTuple('t'/'r') on: an hou.ObjNode resolves to itself; a
    hou.SopNode resolves to its containing OBJ ancestor (walking
    node.parent() up to the nearest hou.ObjNode). Raises
    hou.OperationFailed when no OBJ ancestor is found, or when node is
    neither a SopNode nor an ObjNode.

    Distinct from _world_aabb_center_and_extents (BYTE-FROZEN, untouched
    by this PR): that helper returns (t, r, bbox) for READING a node's
    world placement; this helper returns the actual mutable OBJ NODE
    solve_layout's apply WRITES to (M2 fold, plan pp12-116c
    lockedFieldContract revision 2).
    """
    if isinstance(node, hou.ObjNode):
        return node
    if isinstance(node, hou.SopNode):
        parent = node.parent()
        while parent is not None and not isinstance(parent, hou.ObjNode):
            parent = parent.parent()
        if parent is None:
            raise hou.OperationFailed(
                f"SopNode has no containing ObjNode: {node.path()}"
            )
        return parent
    raise hou.OperationFailed(
        f"Node is neither a SopNode nor an ObjNode: {node.path()}"
    )


def _resolve_obj_nodes_for_objects(objects_wire: list) -> dict:
    """Resolve {oid: (wire_node, obj_node, is_fixed)} for every solve_layout
    wire object via _resolve_obj_node -- the mutable-node resolution shared
    by _preview_solve_layout (cheap validation only, no bbox/geometry read)
    and solve_layout's own apply-time pre-validation (M5: the dispatcher's
    direct-call surface bypasses the 109-gate preview_fn entirely, so the
    SAME check must ALSO run inside the real apply).

    wire_node is the object's OWN wire node (hou.node(node_path) --
    unresolved-to-OBJ) -- retained (fix-cycle 3, B1) so a subsequent
    apply-time re-read of the object's world-AABB center can consult the
    SAME geometry source used for the object's INITIAL resolution, rather
    than the (possibly different) geometry of the resolved OBJ node's own
    displayNode(). obj_node is the mutable hou.ObjNode _resolve_obj_node
    resolves wire_node to -- the node solve_layout's apply WRITES
    parmTuple('t'/'r') on.

    Raises hou.OperationFailed on a missing node path or an unresolvable
    node (propagated from hou.node() / _resolve_obj_node()).
    """
    resolved: dict = {}
    for obj_wire in objects_wire:
        oid = obj_wire["id"]
        node_path = obj_wire.get("node") or ""
        if not node_path:
            raise hou.OperationFailed(
                f"Object {oid!r}: solve_layout requires a node path for every object"
            )
        wire_node = hou.node(node_path)
        if wire_node is None:
            raise hou.OperationFailed(
                f"Node not found: {node_path} (object id={oid!r})"
            )
        resolved[oid] = (
            wire_node,
            _resolve_obj_node(wire_node),
            bool(obj_wire.get("fixed", False)),
        )
    return resolved


def _is_ancestor_or_descendant(path_a: str, path_b: str) -> bool:
    """True iff *path_a* and *path_b* are the SAME OBJ path, or one is an
    ANCESTOR of the other in the node hierarchy (B2, fix-cycle 3,
    codex-adversarial-reviewer verified Blocker): moving an ancestor OBJ
    transforms every descendant OBJ along with it (Houdini's parent-child
    transform inheritance), so an ancestor/descendant pair can no more be
    independently placed than two objects sharing the exact same OBJ node.
    Uses `/`-boundary-safe path-prefix containment (`q == p` or
    `q.startswith(p + "/")` or `p.startswith(q + "/")`) so "/obj/subnet1"
    is NOT flagged as an ancestor of "/obj/subnet10" (a naive
    `startswith(p)` without the "/" boundary would falsely match)."""
    if path_a == path_b:
        return True
    if path_b.startswith(path_a + "/"):
        return True
    if path_a.startswith(path_b + "/"):
        return True
    return False


def _check_movable_targets(resolved: dict) -> None:
    """Validate the apply-time invariants over an already-resolved
    {oid: (wire_node, obj_node, is_fixed)} dict (B1/B2/B3, plan pp12-116c
    lockedFieldContract revision 2): every MOVABLE (non-fixed) object's
    OBJ node must have an IDENTITY parentAndSubnetTransform() (else its
    LOCAL t/r parms silently diverge from its WORLD t/r -- B1), and every
    movable OBJ node must be ANCESTRY-DISTINCT (neither the same node, nor
    an ancestor, nor a descendant -- B2) from every other movable's AND
    every fixed object's OBJ node (moving one OBJ silently drags along any
    OBJ nested beneath it, and two OBJs sharing one ancestor/descendant
    relationship -- like two OBJs at the same path -- cannot be
    independently placed -- B3).

    Raises hou.OperationFailed (-> gate DENY / apply scene-resolution
    error) on any violation. FIXED objects are exempt from the
    identity-parent check (they are never written).
    """
    movable_paths: dict = {}
    fixed_paths: dict = {}
    for oid, (_wire_node, obj_node, is_fixed) in resolved.items():
        path = obj_node.path()
        if is_fixed:
            fixed_paths[oid] = path
            continue
        if obj_node.parentAndSubnetTransform() != hou.Matrix4(1):
            raise hou.OperationFailed(
                f"Object {oid!r}: OBJ node {path!r} has a non-identity "
                f"parentAndSubnetTransform() -- solve_layout's apply is scoped "
                f"to identity-parent OBJ nodes"
            )
        for other_oid, other_path in movable_paths.items():
            if _is_ancestor_or_descendant(other_path, path):
                raise hou.OperationFailed(
                    f"Objects {other_oid!r} ({other_path!r}) and {oid!r} "
                    f"({path!r}) resolve to the same or an ancestor/descendant "
                    f"OBJ node -- cannot be independently placed"
                )
        movable_paths[oid] = path

    for oid, path in movable_paths.items():
        for fixed_oid, fixed_path in fixed_paths.items():
            if _is_ancestor_or_descendant(fixed_path, path):
                raise hou.OperationFailed(
                    f"Movable object {oid!r} ({path!r}) shares the same or an "
                    f"ancestor/descendant OBJ node with fixed object "
                    f"{fixed_oid!r} ({fixed_path!r}) -- cannot be independently "
                    f"placed"
                )


def _apply_solved_transforms(transforms: dict, resolved: dict) -> None:
    """Move each non-fixed object's OBJ node so its world-AABB center lands
    at transforms[oid]['t'] under transforms[oid]['r'] -- via
    ROTATE-THEN-REREAD-THEN-TRANSLATE (B2, plan pp12-116c
    lockedFieldContract revision 2): (1) set the rotation parm FIRST;
    (2) RE-READ the POST-ROTATION world-AABB center via
    _world_aabb_center_and_extents (BYTE-FROZEN, unmodified) -- via the
    object's OWN WIRE NODE, NOT the resolved OBJ node (B1, fix-cycle 3):
    the object's INITIAL resolution (solve_layout, before this apply runs)
    always reads its world-AABB center/bbox via the wire node, so the
    post-rotation re-read MUST consult that SAME geometry source -- an
    OBJ's displayNode() geometry can differ from a wire SopNode that is
    some OTHER SOP under the same OBJ, and reading the wrong one computes
    a wrong post-rotation center and corrupts the translate delta even
    when the object needs zero net movement; (3) set the translation parm
    to cur_objt + (solved_t - post_rotation_center) -- correct regardless
    of the object's local geometry offset from its pivot. FIXED objects
    are NEVER written. The whole multi-object apply runs inside exactly
    ONE hou.undos.group(...) context (M5 atomicity -- entered once for the
    WHOLE apply, never per-object), so a mid-apply exception propagates
    out of this single context and is caught by the caller's
    scene-resolution try/except boundary.
    """
    with hou.undos.group("solve_layout apply"):
        for oid, (wire_node, obj_node, is_fixed) in resolved.items():
            if is_fixed:
                continue
            solved_t = transforms[oid]["t"]
            solved_r = transforms[oid]["r"]

            obj_node.parmTuple("r").set(tuple(solved_r))

            post_t, _post_r, _post_bbox = _world_aabb_center_and_extents(wire_node)

            cur_objt = obj_node.parmTuple("t").eval()
            new_t = tuple(cur_objt[i] + (solved_t[i] - post_t[i]) for i in range(3))
            obj_node.parmTuple("t").set(new_t)


def _preview_solve_layout(params: dict) -> dict:
    """Return the 109-gate approval payload for solve_layout WITHOUT
    solving or mutating. CHEAP validation only (M4: stays well under the
    ~30s preview timeout) -- resolves every wire object's mutable OBJ node
    and validates the identity-parent + distinct-OBJ invariants (M1-DENY:
    an invalid apply target is DENIED at the gate, not merely flagged).
    Does NOT invoke spatial_reasoning_model.solve_layout and does NOT read
    any node's geometry/bounding box.

    Called POSITIONALLY by the gate middleware as ``preview_fn(params)`` --
    a single ``params: dict`` argument, NOT ``**params`` (matches
    _preview_setup_node's convention).

    Returns:
        {"would_move": [<non-fixed object ids>], "apply": <echoed apply>,
         "note": <human-readable summary>}

    Raises:
        hou.OperationFailed: on an unresolvable node, two movable objects
            sharing one OBJ ancestor, a movable object sharing a fixed
            object's OBJ, or a movable OBJ with a non-identity
            parentAndSubnetTransform() -- the gate DENIES the call.
    """
    objects = params.get("objects", [])
    apply = params.get("apply", True)

    resolved = _resolve_obj_nodes_for_objects(objects)
    _check_movable_targets(resolved)

    would_move = [oid for oid, (_, _, is_fixed) in resolved.items() if not is_fixed]
    return {
        "would_move": would_move,
        "apply": apply,
        "note": (
            f"solve_layout will move {len(would_move)} object(s)"
            if apply else
            "apply=false -- no scene mutation"
        ),
    }


def solve_layout(
    *,
    objects: list,
    relations: list,
    bounds: "dict | None" = None,
    apply: bool = True,
    max_iters: int = 200,
) -> dict:
    """Solve relations->positions for a scene and, when apply=true, MOVE
    each non-fixed node so its world-AABB center lands at the solved
    center under the solved rotation. GATED (Capability.MUTATING, PP12-109
    security gate) -- see _preview_solve_layout / register_handler below.

    STEPS (plan pp12-116c lockedFieldContract, "tool -- solve_layout"):
    (1) resolve each wire object {id, node, bbox?, fixed?} into an
    ObjectSpec: bbox = the caller's if given else the hou world-AABB
    extents (_world_aabb_center_and_extents, PR-2 reuse); t/r = the
    object's CURRENT world-AABB center/rotation, ALWAYS read via the node
    (solve_layout's wire schema has no caller-transform override, unlike
    assert_scene's separate `transforms` param). (2) build a LayoutSpec.
    (3) delegate the solve to the byte-frozen
    spatial_reasoning_model.solve_layout (a caller-contract ValueError --
    e.g. an unknown relation type -- PROPAGATES, never folded into
    {ok:false}). (4) if apply=true: PRE-VALIDATE every movable object
    (resolvable + distinct-OBJ + identity-parent, _check_movable_targets)
    BEFORE any parm write, then apply via
    ROTATE-THEN-REREAD-THEN-TRANSLATE inside a single hou.undos.group
    (_apply_solved_transforms). (5) return the model's result dict
    VERBATIM. A scene-resolution failure (an unresolvable node, a
    non-identity parent, a shared OBJ ancestor) returns
    {"ok": False, "error": "<reason>"} WITHOUT raising -- at either the
    initial-resolution phase or the apply phase.
    """
    try:
        resolved_objects = []
        resolved_nodes: dict = {}
        for obj_wire in objects:
            oid = obj_wire["id"]
            node_path = obj_wire.get("node") or ""
            if not node_path:
                raise hou.OperationFailed(
                    f"Object {oid!r}: solve_layout requires a node path for every object"
                )
            node = hou.node(node_path)
            if node is None:
                raise hou.OperationFailed(
                    f"Node not found: {node_path} (object id={oid!r})"
                )

            t, r, bbox_from_node = _world_aabb_center_and_extents(node)
            wire_bbox = obj_wire.get("bbox")
            bbox_final = tuple(wire_bbox) if wire_bbox is not None else bbox_from_node
            is_fixed = bool(obj_wire.get("fixed", False))

            resolved_objects.append(_model._object_from_dict({
                "id": oid,
                "bbox": bbox_final,
                "fixed": is_fixed,
                "t": list(t),
                "r": list(r),
                "node": node_path,
            }))

            resolved_nodes[oid] = (node, _resolve_obj_node(node), is_fixed)

        relation_specs = [_model._relation_from_dict(rd) for rd in (relations or [])]
    except (hou.OperationFailed, AttributeError) as exc:
        return {"ok": False, "error": str(exc)}

    layout_spec = _model.LayoutSpec(
        objects=resolved_objects,
        relations=relation_specs,
        bounds=bounds,
        max_iters=max_iters,
    )
    result = _model.solve_layout(layout_spec)

    if apply:
        try:
            _check_movable_targets(resolved_nodes)
            _apply_solved_transforms(result["transforms"], resolved_nodes)
        except (hou.OperationFailed, AttributeError) as exc:
            return {"ok": False, "error": str(exc)}

    return result


register_handler(
    "solve_layout",
    solve_layout,
    Capability.MUTATING,
    preview_fn=_preview_solve_layout,
    preview_required=True,
)
