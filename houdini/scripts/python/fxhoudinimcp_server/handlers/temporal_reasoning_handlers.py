"""Handlers: describe_sim_events, assert_simulation.

Both are READ-ONLY, UNGATED (Capability.READONLY) -- the Gate-1 read surface
of the Temporal/Sim-Reasoning MCP member (PP12-117 PR-2).

describe_sim_events -- a PURE delegate (no hou) to
                       temporal_reasoning_model.describe_sim_events() -- the
                       exact SPEC 4.1 {events, triggers, assertions}
                       vocabulary shape, verbatim.
assert_simulation    -- the Gate-1 TEMPORAL oracle. Across an inclusive
                       frame_range, reads each requested assertion's
                       per-frame SCALAR value off the LIVE sim (from the
                       assertion's own `node` or the network) by stepping
                       hou.setFrame() + reading geometry, pre-reduces to one
                       scalar per frame, builds one [[frame,value]...]
                       series PER ASSERTION (in input order), and delegates
                       ALL pass/fail math to the byte-frozen PR-1
                       temporal_reasoning_model.evaluate_assertions.

REVERSIBLE-FRAME-EVALUATION EXCEPTION (Blocker-2, plan pp12-117b
lockedFieldContract revision 2): assert_simulation steps hou.setFrame() to
read per-frame scalars but SAVES the pre-call frame and RESTORES it in a
`finally` (both the success path and any error path raised mid-stepping) --
it NEVER calls animation.set_frame/step_simulation/reset_simulation, NEVER
writes a cache, and NEVER writes any node/parm/userData. This is why it
stays Capability.READONLY / require_approval=False despite touching the
playbar -- an implicit DOP-cook cache populated as a side effect of reading
geometry is treated as NON-PERSISTENT evaluation cache (READ-consistent
with the shipped assert_scene/get_bounding_box implicit-cook precedent),
never a mutation.

METRIC -> SOURCE table (BINDING, the load-bearing pin; per-assertion
source; ground on SHIPPED readers; a missing required attribute is raised
AS hou.OperationFailed so the WHOLE call degrades to one {ok:false,error},
never a partial result -- Major-10):
  piece_count       -- DETERMINISTIC precedence: (1) count of unique
                       NON-EMPTY primitive `name` attribute values, if the
                       `name` prim attribute exists; else (2) unique
                       non-empty POINT `name` attribute values, if that
                       point attribute exists; else (3)
                       geo.intrinsicValue('primitivecount') (Major-8).
  point_count       -- geo.intrinsicValue('pointcount').
  velocity_bounds   -- max over points of |`v` point attribute|; 0.0 if no
                       points; a MISSING `v` attribute raises
                       hou.OperationFailed.
  bbox_over_time    -- the WORLD-AABB max extent: a LOCAL duplicate of
                       spatial_reasoning_handlers._world_aabb_center_and_extents's
                       semantics (byte-identical algorithm -- ObjNode via
                       displayNode().geometry() + the ObjNode's own
                       worldTransform(); SopNode via its containing ObjNode
                       ancestor's worldTransform(); the 8 local-bbox corners
                       transformed to world space) -- scalar =
                       max(w, d, h) of the world extents (Major-9). See
                       _transform_point's docstring for why this is
                       duplicated (_world_aabb_extents) rather than
                       cross-imported: importing the shared function would
                       close over spatial_reasoning_handlers' OWN `hou`
                       binding, which is fixed at THAT module's first
                       import in a pytest session and never rebound --
                       a cross-test-module hou-mock leak verified during
                       PR-2 implementation when this file's tests ran
                       alongside test_spatial_reasoning_handlers.py.
  mass_conservation -- the SUM of a `mass` point attribute over the
                       geometry; a MISSING `mass` attribute raises
                       hou.OperationFailed (never silently substitutes
                       point count).

DEFERRED metrics (Blocker-4/5 -- no shipped aggregate reader in PR-2):
field_stats and constraint_count each return a structured
{'ok': False, 'error': '<metric> unsupported in PR-2: ...'} BEFORE any node
resolution or frame stepping -- degrading the WHOLE call, never raising
through the dispatcher.

GEOMETRY RESOLUTION (mirrors spatial_reasoning_handlers.py Blocker-2):
node.geometry() exists ONLY on hou.SopNode. An hou.ObjNode is resolved via
displayNode().geometry() instead. A node that is neither is a
scene-resolution error.

PER-ASSERTION SOURCE ROUTING (Blocker-3): each assertion resolves its own
source_path = assertion.get('node') or network -- an assertion's own `node`
(when present) is the read source instead of `network`. Two assertions on
the SAME metric but DIFFERENT nodes build TWO independent series in INPUT
order. Distinct source nodes are resolved ONCE (cached) before any frame
stepping.

EXPECT NORMALIZATION (Blocker-6): when an assertion has no `expect` key,
its top-level predicate/support keys ({max, min, jump_gt, tolerance,
max_gt, eq, at_frame}) are folded into an `expect` dict -- e.g. the SPEC
4.1 example {"metric": "velocity_bounds", "max": 250} normalizes to
expect={"max": 250}. BOTH `expect` and a top-level predicate key present is
a caller-contract error -- raises ValueError naming the conflict (NOT
folded into {ok:false} -- see the success/error boundary below).

FRAME_RANGE VALIDATION (Major-7): validated BEFORE any hou read --
exactly two ints (bool rejected), start <= end (a single-frame [f,f] is
allowed). assertions=[] performs NO frame stepping at all and returns
temporal_reasoning_model.evaluate_assertions([]) verbatim
(pass:true, results:[]).

cook_job (documented-unavailable): a non-null cook_job returns
{'ok': False, 'error': 'cook_job reuse unavailable (115 cook registry not
built); omit cook_job to read the current synchronous sim state'} -- this
check runs BEFORE frame_range validation (the 115 cook-registry surface is
simply not built yet; this is the only PR-2 handling for it).

SUCCESS vs ERROR BOUNDARY (mirrors spatial_reasoning_handlers.py
Blocker-4): the SCENE-RESOLUTION + READ phase (hou.node,
isinstance/displayNode, geometry reads, world-AABB, hou.setFrame) is
wrapped so a hou.OperationFailed / AttributeError degrades to
{"ok": False, "error": "<reason>"} WITHOUT raising, and the saved frame is
restored in a `finally` regardless. The MODEL CALL
(temporal_reasoning_model.evaluate_assertions) is OUTSIDE that guard: a
caller-contract ValueError raised by frame_range validation, the
expect-normalization conflict, or the model's own metric/series/expect
validation (e.g. an unknown assertion metric) PROPAGATES as a normal
exception, which the dispatcher (fxhoudinimcp_server.dispatcher.dispatch)
turns into its standard error envelope -- it is NEVER folded into
{"ok": False}. Concretely: an assertion whose metric is not one of the five
locally-readable metrics above (and not one of the two deferred metrics) is
never read at all -- it is passed straight through to
temporal_reasoning_model.evaluate_assertions with an empty series, and the
model's own "unknown assertion metric" check raises ValueError before ever
inspecting that series.

Grounded LAYER-FOR-LAYER on:
  - handlers/spatial_reasoning_handlers.py -- the shipped READONLY
    handler-registration convention, the sys.path bootstrap needed to
    reach the fxhoudinimcp package from hython, the
    {ok:false,error} / narrow-except / model-ValueError-propagates
    boundary, and _world_aabb_center_and_extents's algorithm (its
    SEMANTICS are duplicated here as _world_aabb_extents/
    _transform_point rather than cross-imported -- see
    _transform_point's docstring for the test-isolation reason;
    cross-handler private-helper imports ARE otherwise an established
    convention in this codebase, e.g. character_handlers.py imports
    graph_handlers._resolve_node_type, but that convention does not hold
    here because of the hou-mock-staleness failure mode this duplication
    avoids).
  - handlers/geometry_handlers.py -- the shipped prim-name /
    point-name / intrinsicValue idiom this module's piece_count/
    point_count readers extend.

Cross-references
-----------------
Plan pp12-117b lockedFieldContract (BINDING, revision 2 -- folds ALL 11
  codex-adversarial-reviewer findings, 6 Blockers + 5 Majors)
docs/portfolio_projects/PP12_houdini_mcp_agentic_bridge/
  117_mcp_temporal_sim_reasoning_surface/spec.md Sections 4.1, 4.3, 6, 7, 9
CL-015 (extended): only *_model.py stays hou/Qt/pxr-free; this handler MAY
  (and does) import hou -- it is the hou-facing layer PR-2 ships.
CL-016: hou reads happen on the dispatcher's main-thread marshal
  (hdefereval); this handler does not itself call hdefereval.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# sys.path bootstrap -- 5 levels up from this file reaches the fork root;
# +/python adds the FastMCP-side fxhoudinimcp package (temporal_reasoning_model
# lives there) so the import below resolves when this module is imported
# from hython. Mirrors spatial_reasoning_handlers.py exactly.
#
#  __file__: .../fxhoudinimcp/houdini/scripts/python/fxhoudinimcp_server/handlers/temporal_reasoning_handlers.py
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
from fxhoudinimcp import temporal_reasoning_model as _model  # noqa: E402


###### describe_sim_events (FR-C, pure delegate)

def describe_sim_events() -> dict:
    """Return the model's exact SPEC 4.1 {events, triggers, assertions}
    vocabulary dict, verbatim.

    A PURE delegate -- touches NO hou. Cannot fail on scene state.
    """
    return _model.describe_sim_events()


register_handler("describe_sim_events", describe_sim_events, Capability.READONLY)


###### assert_simulation (FR-B / Gate-1) -- constants + helpers

# The 7 keys a caller may supply at TOP LEVEL of an assertion dict instead
# of a nested `expect` dict (Blocker-6). Mirrors
# temporal_reasoning_model._PREDICATE_KEYS | _SUPPORT_KEYS exactly.
_PREDICATE_AND_SUPPORT_KEYS = (
    "max", "min", "jump_gt", "tolerance", "max_gt", "eq", "at_frame",
)

# The five metrics this PR-2 handler knows how to READ off the live scene
# (the METRIC -> SOURCE table). Any OTHER metric string -- including the
# two DEFERRED ones below, which are intercepted earlier -- is never read;
# it is passed straight through to the model so the model's own
# unknown-metric ValueError fires (never folded into {ok:false}).
_LOCALLY_READABLE_METRICS = frozenset({
    "piece_count", "point_count", "velocity_bounds", "bbox_over_time",
    "mass_conservation",
})

# DEFERRED metrics (Blocker-4/5) -- no shipped aggregate reader in PR-2.
# Checked BEFORE any node resolution or frame stepping; degrades the WHOLE
# call to this exact structured message.
_DEFERRED_METRICS = {
    "field_stats": "field_stats unsupported in PR-2: no shipped DOP aggregate field-stat reader",
    "constraint_count": "constraint_count unsupported in PR-2: no shipped active-constraint reader",
}


def _validate_frame_range(frame_range) -> tuple:
    """Validate frame_range BEFORE any hou read (Major-7).

    Must be exactly a two-element list of ints (bool rejected), start <=
    end (a single-frame [f, f] is allowed). Raises ValueError otherwise.
    """
    if not isinstance(frame_range, list) or len(frame_range) != 2:
        raise ValueError(
            f"frame_range must be a list of exactly two ints [start, end], "
            f"got {frame_range!r}"
        )
    start, end = frame_range
    if isinstance(start, bool) or not isinstance(start, int):
        raise ValueError(f"frame_range start must be an int (not bool), got {start!r}")
    if isinstance(end, bool) or not isinstance(end, int):
        raise ValueError(f"frame_range end must be an int (not bool), got {end!r}")
    if start > end:
        raise ValueError(
            f"frame_range start must be <= end, got start={start!r} end={end!r}"
        )
    return start, end


def _normalize_assertion(a: dict) -> dict:
    """Normalize one wire assertion dict (Blocker-6): fold a top-level
    predicate/support key set into `expect` when `expect` is absent; raise
    ValueError naming the conflict when BOTH are present.

    Returns {"metric": ..., "expect": {...}, "node": <str|None>} -- the
    only three fields the read phase + the model call need. `node` is the
    per-assertion source override (Blocker-3); None means "fall back to
    `network`".
    """
    top_level = {k: a[k] for k in _PREDICATE_AND_SUPPORT_KEYS if k in a}
    has_expect = "expect" in a
    if has_expect and top_level:
        raise ValueError(
            f"assertion for metric {a.get('metric')!r} has BOTH an 'expect' dict "
            f"and top-level predicate key(s) {sorted(top_level)!r} -- supply one "
            f"or the other, not both"
        )
    expect = a["expect"] if has_expect else top_level
    return {"metric": a.get("metric"), "expect": expect, "node": a.get("node")}


def _resolve_source_node(path: str, cache: dict):
    """Resolve *path* to a hou.Node, caching by path so a distinct source
    referenced by multiple assertions is resolved only once. Raises
    hou.OperationFailed when the node does not exist."""
    if path in cache:
        return cache[path]
    node = hou.node(path)
    if node is None:
        raise hou.OperationFailed(f"Node not found: {path}")
    cache[path] = node
    return node


def _read_geometry(node):
    """Resolve *node*'s CURRENT geometry -- hou.SopNode via .geometry(),
    hou.ObjNode via displayNode().geometry() (node.geometry() does not
    exist on hou.ObjNode). Raises hou.OperationFailed on any resolution
    failure (no display node, no geometry, or a node that is neither a
    SopNode nor an ObjNode)."""
    if isinstance(node, hou.SopNode):
        geo = node.geometry()
    elif isinstance(node, hou.ObjNode):
        display_node = node.displayNode()
        if display_node is None:
            raise hou.OperationFailed(
                f"ObjNode has no display node (empty subnet?): {node.path()}"
            )
        geo = display_node.geometry()
    else:
        raise hou.OperationFailed(
            f"Node is neither a SopNode nor an ObjNode: {node.path()}"
        )
    if geo is None:
        raise hou.OperationFailed(f"Node has no geometry: {node.path()}")
    return geo


def _read_piece_count(geo) -> float:
    """piece_count precedence (Major-8, DETERMINISTIC -- FIX-PASS
    codex-reviewer Major-1): precedence is keyed on NON-EMPTY VALUES
    actually present, NOT on attribute existence. A `name` attribute
    (primitive or point) that EXISTS but whose every value is "" must
    NOT win the precedence -- it must fall through to the next source.
    1. unique non-empty PRIMITIVE `name` attribute values, if any are
       non-empty
    2. else unique non-empty POINT `name` attribute values, if any are
       non-empty
    3. else geo.intrinsicValue('primitivecount')
    """
    if geo.findPrimAttrib("name") is not None:
        names = {p.attribValue("name") for p in geo.prims()}
        names = {n for n in names if n not in (None, "")}
        if names:
            return float(len(names))
    if geo.findPointAttrib("name") is not None:
        names = {pt.attribValue("name") for pt in geo.points()}
        names = {n for n in names if n not in (None, "")}
        if names:
            return float(len(names))
    return float(geo.intrinsicValue("primitivecount"))


def _read_point_count(geo) -> float:
    return float(geo.intrinsicValue("pointcount"))


def _read_velocity_bounds(geo) -> float:
    """max over points of |`v` point attribute|; 0.0 if no points. A
    MISSING `v` attribute raises hou.OperationFailed (Major-10) -- but
    ONLY when there ARE points to require it on (FIX-PASS codex-reviewer
    Major-2): the "0.0 if no points" case is checked FIRST, so an empty
    source with no `v` attribute at all returns 0.0 rather than
    degrading the whole call."""
    pts = geo.points()
    if not pts:
        return 0.0
    if geo.findPointAttrib("v") is None:
        raise hou.OperationFailed(
            "velocity_bounds requires a 'v' point attribute, none found"
        )
    max_mag = 0.0
    for pt in pts:
        vx, vy, vz = pt.attribValue("v")
        mag = (vx * vx + vy * vy + vz * vz) ** 0.5
        if mag > max_mag:
            max_mag = mag
    return float(max_mag)


def _transform_point(x: float, y: float, z: float, matrix) -> tuple:
    """Transform a local point by a world/affine 4x4 matrix (row-vector
    convention). Byte-identical to
    spatial_reasoning_handlers._transform_point -- duplicated here
    (deliberately NOT imported) so this module's isinstance(node,
    hou.SopNode/hou.ObjNode) checks below always resolve against THIS
    module's own `hou` binding, rebound fresh on every reload of this
    module. Importing the function from spatial_reasoning_handlers instead
    would close over THAT module's `hou` name, which is bound ONCE at
    spatial_reasoning_handlers' own first import in a pytest session and
    never rebound afterward -- a cross-test-module hou-mock leak that
    produces spurious "Node is neither a SopNode nor an ObjNode" failures
    when this file's tests run alongside test_spatial_reasoning_handlers.py
    in the same session (verified during PR-2 implementation)."""
    nx = x * matrix.at(0, 0) + y * matrix.at(1, 0) + z * matrix.at(2, 0) + matrix.at(3, 0)
    ny = x * matrix.at(0, 1) + y * matrix.at(1, 1) + z * matrix.at(2, 1) + matrix.at(3, 1)
    nz = x * matrix.at(0, 2) + y * matrix.at(1, 2) + z * matrix.at(2, 2) + matrix.at(3, 2)
    return (nx, ny, nz)


def _world_aabb_extents(node) -> tuple:
    """Resolve *node*'s WORLD-space AABB extents (w, d, h) -- byte-identical
    semantics to spatial_reasoning_handlers._world_aabb_center_and_extents
    (Major-9; the world-AABB center is not needed here, only the extents,
    so this returns just the extents tuple). See _transform_point's
    docstring for why this is a local duplicate rather than a cross-module
    import.

    Resolves geometry for BOTH hou.SopNode (node.geometry()) and
    hou.ObjNode (node.displayNode().geometry()). Raises
    hou.OperationFailed for any resolution failure (no geometry, an
    ObjNode with no display node, a SopNode with no containing hou.ObjNode
    ancestor, or a node that is neither a SopNode nor an ObjNode).
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

    ext = [world_max[i] - world_min[i] for i in range(3)]
    return (ext[0], ext[2], ext[1])  # w=x-extent, d=z-extent, h=y-extent


def _read_bbox_over_time(node) -> float:
    """The world-AABB max extent (Major-9) -- max(w, d, h) of the world
    extents resolved by _world_aabb_extents."""
    bbox = _world_aabb_extents(node)
    return float(max(bbox))


def _read_mass_conservation(geo) -> float:
    """The SUM of a `mass` point attribute over the geometry. A MISSING
    `mass` attribute raises hou.OperationFailed (Major-10) -- never
    silently substitutes point count."""
    if geo.findPointAttrib("mass") is None:
        raise hou.OperationFailed(
            "mass_conservation requires a 'mass' point attribute, none found"
        )
    total = 0.0
    for pt in geo.points():
        total += float(pt.attribValue("mass"))
    return total


def _read_metric(metric: str, node) -> float:
    """Read ONE metric's current-frame scalar off *node*. Only called for
    metrics in _LOCALLY_READABLE_METRICS -- bbox_over_time resolves its own
    geometry via _read_bbox_over_time; the other four share _read_geometry.
    """
    if metric == "bbox_over_time":
        return _read_bbox_over_time(node)
    geo = _read_geometry(node)
    if metric == "piece_count":
        return _read_piece_count(geo)
    if metric == "point_count":
        return _read_point_count(geo)
    if metric == "velocity_bounds":
        return _read_velocity_bounds(geo)
    if metric == "mass_conservation":
        return _read_mass_conservation(geo)
    # Unreachable: callers only invoke this for metrics in
    # _LOCALLY_READABLE_METRICS, which covers exactly the four branches
    # above (plus bbox_over_time handled first).
    raise hou.OperationFailed(f"No read implementation for metric {metric!r}")


def assert_simulation(
    *,
    network: str,
    frame_range: list,
    assertions: list,
    cook_job: "str | None" = None,
) -> dict:
    """The Gate-1 temporal oracle: step an inclusive frame range, read each
    requested assertion's per-frame scalar off its resolved source node,
    build one series per assertion (input order), and delegate ALL
    pass/fail math to the byte-frozen temporal_reasoning_model.evaluate_assertions.

    Steps (locked contract order):
      (0) cook_job non-null -> the documented {ok:false} -- BEFORE
          frame_range is even validated.
      (1) VALIDATE frame_range. assertions=[] -> NO frame stepping,
          returns evaluate_assertions([]) verbatim.
      (2) NORMALIZE each assertion's expect (Blocker-6); resolve
          source_path = assertion.get('node') or network per assertion
          (Blocker-3).
      (2b) a DEFERRED metric (field_stats/constraint_count) -> the
          unsupported {ok:false} for the WHOLE call, before any node
          resolution.
      (3) resolve each distinct source node used by a locally-readable
          assertion; an unresolvable node -> {ok:false,error}.
      (4) SAVE hou.frame(); step the inclusive range, reading each
          locally-readable assertion's metric off its resolved source.
      (5) RESTORE the saved frame in a finally (success or error).
      (6) delegate [{metric, series, expect}...] (in input order,
          including any NOT-locally-readable metric with an empty series
          so the MODEL's own unknown-metric check fires) to
          temporal_reasoning_model.evaluate_assertions; return its exact
          {results, pass} VERBATIM.

    The MODEL CALL is OUTSIDE the scene-resolution guard: a caller-contract
    ValueError (an unknown metric, a malformed expect, the
    normalize-conflict, a malformed frame_range) PROPAGATES as the
    dispatcher's standard error envelope -- NEVER folded into {ok:false}.
    """
    if cook_job is not None:
        return {
            "ok": False,
            "error": (
                "cook_job reuse unavailable (115 cook registry not built); "
                "omit cook_job to read the current synchronous sim state"
            ),
        }

    start, end = _validate_frame_range(frame_range)

    # FIX-PASS (codex-reviewer Major-3): ONLY the genuine empty-list case
    # short-circuits to the no-stepping evaluate_assertions([]) shape. A
    # non-list `assertions` (e.g. None) must NOT be treated as "falsy
    # therefore empty" -- it is delegated straight to the model so the
    # model's OWN caller-contract ValueError (evaluate_assertions raises
    # on a non-list) propagates untouched, rather than being silently
    # folded into a passing empty-assertions result.
    if assertions == []:
        return _model.evaluate_assertions([])
    if not isinstance(assertions, list):
        return _model.evaluate_assertions(assertions)

    normalized = [_normalize_assertion(a) for a in assertions]

    for a in normalized:
        if a["metric"] in _DEFERRED_METRICS:
            return {"ok": False, "error": _DEFERRED_METRICS[a["metric"]]}

    readable_indices = [
        i for i, a in enumerate(normalized) if a["metric"] in _LOCALLY_READABLE_METRICS
    ]

    series_by_assertion: list = [[] for _ in normalized]

    if readable_indices:
        try:
            node_cache: dict = {}
            for i in readable_indices:
                source_path = normalized[i]["node"] or network
                _resolve_source_node(source_path, node_cache)

            saved_frame = hou.frame()
            try:
                for f in range(start, end + 1):
                    hou.setFrame(f)
                    for i in readable_indices:
                        a = normalized[i]
                        source_path = a["node"] or network
                        node = node_cache[source_path]
                        value = _read_metric(a["metric"], node)
                        series_by_assertion[i].append([f, value])
            finally:
                hou.setFrame(saved_frame)
        except (hou.OperationFailed, AttributeError) as exc:
            return {"ok": False, "error": str(exc)}

    model_assertions = [
        {"metric": a["metric"], "series": series_by_assertion[i], "expect": a["expect"]}
        for i, a in enumerate(normalized)
    ]
    return _model.evaluate_assertions(model_assertions)


register_handler("assert_simulation", assert_simulation, Capability.READONLY)


###### compile_timeline (FR-A, MUTATING/GATED) -- PP12-117 PR-3
#
# ADDITIVE to this module (plan pp12-117c lockedFieldContract, revision 2)
# -- the PR-2 READONLY handlers above are BYTE-UNCHANGED.
#
# Grounded LAYER-FOR-LAYER on the SHIPPED gated exemplar
# (spatial_reasoning_handlers.py: solve_layout + _preview_solve_layout +
# register_handler(..., Capability.MUTATING, preview_fn=..., preview_
# required=True)) and animation_handlers.py's shipped _set_keyframes
# (node_path, parm_name, keyframes=[{frame,value}]) apply surface
# (Major-8, direct-import below).
#
# PIPELINE (locked contract "handler: compile_timeline"; REVISION 3 --
# operator decision 2026-07-11, B2 ATOMICITY GUARANTEE CLARIFIED: DROPS the
# round-2/round-3 bespoke `_delete_keyframe` rollback (a data-loss Blocker --
# it deleted a pre-existing keyframe the call had merely OVERWRITTEN instead
# of restoring its prior value) in favor of native-undo-only recovery; M3
# unconditional network resolution + B1-residual canonical network
# comparison from the round-3 fix-pass are UNCHANGED):
#   (1) resolve `network` ONCE, UNCONDITIONALLY, via _resolve_network --
#       BEFORE anything else, regardless of `events` (even events=[]).
#       `network` empty/falsy OR hou.node(network) is None ->
#       {"ok": False, "error": ...} IMMEDIATELY, WITHOUT raising
#       (scene-resolution boundary; contract step 1). The resolved
#       node's OWN canonical .path() -- NEVER the raw `network` string --
#       is what every target is scope-compared against from here on
#       (ROUND-3 B1-residual: closes both the non-canonical-network
#       false-reject and the empty-network unscoped-wildcard hole).
#   (2) call temporal_reasoning_model._compile_events(events, network) --
#       a model ValueError on a malformed event (unknown type, a cyclic/
#       dangling/duplicate causes graph) PROPAGATES, never folded into
#       {ok:false}.
#   (3) PREFLIGHT-VALIDATE every compiled entry (Blocker-1/2/3/4) BEFORE
#       any write: the target node CANONICALLY resolves (hou.node) AND
#       its OWN resolved .path() is the RESOLVED `network` node's
#       canonical path OR under `<canonical_network> + '/'` (Blocker-1 --
#       compared via the RESOLVED TARGET's canonical path against the
#       RESOLVED NETWORK's canonical path, NEVER either raw string,
#       closing a '..' path-traversal escape) AND the parm exists on it
#       AND is NOT LOCKED (Blocker-2 -- `parm is not None` alone is not a
#       writability check) AND the frames payload is well-formed. A
#       failing entry moves to unresolved (its SOURCE EVENT ID) and is
#       DROPPED from the applied set -- BEFORE any write (all-or-none).
#   (4) if apply=True: ATOMICALLY apply ONLY the validated set inside ONE
#       hou.undos.group, converting frames [[f,v]...] -> [{frame,value}...]
#       (Blocker-1, keyframe-conversion) and calling the SHIPPED
#       _set_keyframes (Major-8, direct-imported below) once per validated
#       entry. apply=False makes ZERO _set_keyframes calls -- NO mutation.
#       REVISION 3 (B2, CLARIFIED): if ANY entry's _set_keyframes call
#       raises mid-apply, the handler STOPS immediately and degrades the
#       WHOLE call to {ok:false, error:'partial apply -- the gated call
#       was interrupted mid-write; use Undo to revert'} -- it does NOT
#       attempt a bespoke compensating rollback. The single
#       `hou.undos.group` the whole apply ran inside is the operator's
#       recovery mechanism (one native Undo reverts the entire partial
#       write); the residual (a rare TOCTOU can leave a partial write) is
#       ACCEPTED + documented as recoverable that way, appropriate for a
#       gated, undo-wrapped, operator-approved tool.
#   (5) return {compiled, event_graph, applied, unresolved} -- unresolved
#       is the UNION of compile_plan's own unresolved (type-inferred/
#       threshold/causally-impossible) + any preflight-dropped ids.
#
# GATED: register_handler('compile_timeline', compile_timeline,
#     Capability.MUTATING, preview_fn=_preview_compile_timeline,
#     preview_required=True). apply=false is STILL gated (fail-safe --
#     gate capability is per-COMMAND, not per-argument, mirroring
#     solve_layout).
#
# _preview_compile_timeline(params: dict) -> dict (POSITIONAL, Major-6)
# runs the SAME unconditional network resolution (step 1) + the SAME
# read-only preflight validation apply does -- NO writes -- so its
# would-apply/unresolved split matches EXACTLY what apply would do.
# A raise inside the preview (e.g. a malformed event) -> the 109 gate
# DENIES (mirrors _preview_solve_layout's raise->DENY contract).

# Major-8: DIRECT-IMPORT the shipped keyframe-apply MODULE (not the bare
# name) so every call resolves `_set_keyframes` via ATTRIBUTE LOOKUP AT
# CALL TIME against `animation_handlers`, not a name copied once at THIS
# module's own import time -- correct regardless of which module happens
# to be re-imported fresh in a given test/reload cycle (a name-copied
# `from ... import _set_keyframes` would keep pointing at a stale object
# if only animation_handlers, and not this module, is later re-imported).
# Ground the calling convention on the SHIPPED signature (node_path: str,
# parm_name: str, keyframes: list[{frame, value, ...}]).
from fxhoudinimcp_server.handlers import animation_handlers as _anim_handlers  # noqa: E402


def _current_hou():
    """Return the CURRENTLY-installed `hou` module from sys.modules,
    resolved FRESH on every call -- rather than trusting THIS module's own
    import-time-bound `hou` name (bound once, at whichever moment this
    module was last (re)imported).

    Why this matters here specifically: compile_timeline's own red-test
    suite exercises the REAL dispatcher (`_dispatch`) directly against
    ALREADY-registered handlers, without forcing a fresh reimport of this
    module before every call (a sibling fixture, mock_set_keyframes, pops
    this module from sys.modules so a LATER explicit reimport works
    correctly, but several tests dispatch straight through without one).
    A module-import-time-bound `hou` name would then resolve to whichever
    mock object was active the last time this module happened to be
    reimported -- stale relative to the CURRENT test's own mock. A fresh
    sys.modules lookup sidesteps that entirely, and is a harmless no-op in
    a real Houdini session (its `hou` module is never swapped out from
    under a live process)."""
    import sys as _sys_mod
    return _sys_mod.modules["hou"]


def _apply_exception_types() -> tuple:
    """FIX-PASS (codex-reviewer Blocker-2): the exception types the
    apply-phase except clause catches -- hou.OperationFailed always, and
    hou.PermissionError TOO when it exists AND is a genuine exception
    class (hou.PermissionError is NOT a subclass of hou.OperationFailed
    in this Houdini build, so a write that slips past preflight onto a
    parm Houdini itself refuses must still degrade to {ok:false} rather
    than escape as an uncaught exception). Defensively excludes a
    non-exception stand-in -- e.g. an auto-vivified MagicMock attribute
    under an off-DCC mock `hou` that never defines PermissionError --
    so building this tuple never itself raises TypeError while an
    unrelated exception is being matched."""
    _hou = _current_hou()
    types = [_hou.OperationFailed, AttributeError]
    perm = getattr(_hou, "PermissionError", None)
    if isinstance(perm, type) and issubclass(perm, BaseException):
        types.append(perm)
    return tuple(types)


def _target_in_network_scope(canonical_target: str, canonical_network: str) -> bool:
    """True iff *canonical_target* IS *canonical_network* itself or a
    DESCENDANT under `canonical_network + '/'` -- a '/'-boundary-safe
    prefix check (Blocker-4). A path that merely STARTS WITH
    *canonical_network* WITHOUT the '/' separator (e.g.
    '/obj/rbd_sim_other' vs network '/obj/rbd_sim') is NOT in scope --
    mirrors spatial_reasoning_handlers._is_ancestor_or_descendant's
    boundary-safety discipline.

    BOTH arguments MUST already be CANONICAL resolved node paths (each
    run through hou.node(...).path()) -- NEVER a raw caller-supplied
    string on EITHER side:

    FIX-PASS (codex-reviewer Blocker-1, SECURITY, round-2): the TARGET
    side -- a raw string like '/obj/rbd_sim/../cam_escape' string-passes
    a raw-prefix check but resolves OUTSIDE the network once Houdini
    normalizes the '..' component.

    FIX-PASS (round-3, B1-residual): the NETWORK side -- comparing
    against the RAW caller-supplied `network` string (a) false-rejects a
    perfectly valid target when `network` is supplied in a non-canonical
    form (e.g. a trailing slash: 'net/' + '/' double-slash never matches
    a real child path), and (b) an empty/unresolvable `network` string
    makes `target.startswith('' + '/')` True for virtually ANY absolute
    path -- an unscoped-write wildcard. The caller (_preflight_validate)
    resolves `network` to its OWN canonical path via _resolve_network
    ONCE, up front, and passes THAT resolved canonical path here."""
    return (
        canonical_target == canonical_network
        or canonical_target.startswith(canonical_network + "/")
    )


def _resolve_network(network: "str | None"):
    """Resolve `network` to its OWN hou.Node ONCE, UNCONDITIONALLY --
    contract step 1 (ROUND-3 FIX-PASS M3 + B1-residual). Returns the
    resolved hou.Node, or None when `network` is empty/falsy or does not
    resolve to any real scene node -- callers of this function degrade
    the WHOLE call to {"ok": False, "error": ...} on None, regardless of
    `events` (even events=[]; M3 -- the resolution is NOT conditioned on
    whether any compiled entry's target also fails to resolve).

    Every target's scope comparison uses THIS node's canonical
    `.path()` (see _target_in_network_scope) -- never the raw `network`
    string (B1-residual)."""
    if not network:
        return None
    return _current_hou().node(network)


def _frames_well_formed(frames) -> bool:
    """Defensive well-formedness check on a compiled entry's frames
    payload -- a non-empty list of [frame, value] pairs, both scalar
    int/float (bool excluded). compile_plan already guarantees a
    non-empty list for anything it compiles, but the apply-time preflight
    never trusts the pure layer's output blindly for a live write."""
    if not isinstance(frames, list) or not frames:
        return False
    for item in frames:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return False
        f, v = item
        if isinstance(f, bool) or not isinstance(f, (int, float)):
            return False
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False
    return True


def _preflight_validate(compiled_by_id: dict, canonical_network: str) -> tuple:
    """PREFLIGHT-VALIDATE every {event_id: {node,parm,frames}} entry
    BEFORE any write (Blocker-3/4): the target node CANONICALLY resolves
    (hou.node) AND its OWN resolved .path() is *canonical_network* OR
    under `canonical_network + '/'` (FIX-PASS Blocker-1 + ROUND-3
    B1-residual -- compared via the RESOLVED TARGET node's canonical
    path against the ALREADY-RESOLVED, ALREADY-CANONICAL network path
    the caller computed via _resolve_network -- see
    _target_in_network_scope's docstring) AND the parm exists on it AND
    is NOT LOCKED (FIX-PASS Blocker-2 -- `parm is not None` alone is not
    a writability check) AND the frames payload is well-formed. NEVER
    writes.

    Returns (validated: dict{event_id: {node,parm,frames}},
    dropped_ids: set[str]) -- entries failing ANY check are dropped
    (never applied) and their event id is returned in `dropped_ids` so
    the caller folds it into `unresolved`. Callers are responsible for
    the SEPARATE, UNCONDITIONAL, up-front `network`-itself resolution
    (_resolve_network) -- this function never re-checks `network`
    resolvability itself; it trusts the caller's already-resolved
    canonical path.
    """
    _hou = _current_hou()
    validated: dict = {}
    dropped: set = set()
    for eid, entry in compiled_by_id.items():
        target = entry["node"]
        parm_name = entry["parm"]
        frames = entry["frames"]

        if not _frames_well_formed(frames):
            dropped.add(eid)
            continue

        node = _hou.node(target)
        if node is None:
            dropped.add(eid)
            continue

        # Blocker-1 + ROUND-3 B1-residual: compare the TARGET node's OWN
        # CANONICAL path (node.path()) against the caller's ALREADY-
        # RESOLVED canonical network path -- never a raw string on
        # either side. A path-traversal target like
        # '/obj/rbd_sim/../cam_escape' string-passes a raw-string prefix
        # check but resolves OUTSIDE the gated network once Houdini
        # normalizes the '..' component.
        if not _target_in_network_scope(node.path(), canonical_network):
            dropped.add(eid)
            continue

        parm = node.parm(parm_name)
        if parm is None:
            dropped.add(eid)
            continue

        # Blocker-2 (NON-ATOMIC apply): a LOCKED parm passes a bare
        # `parm is not None` check but must NEVER enter the write phase
        # -- all-or-none, only provably-settable entries are validated
        # here, BEFORE _apply_validated_keyframes writes anything.
        # `is True` (not a bare truthiness check): a real hou.Parm.isLocked()
        # returns a genuine bool, but an UNCONFIGURED mock parm (the
        # normal-case test fixture, which never sets isLocked.return_value)
        # auto-vivifies isLocked() into a fresh, generically-truthy
        # MagicMock -- a bare `if parm.isLocked():` would misclassify
        # EVERY unlocked parm as locked. `is True` matches ONLY a literal
        # boolean True, which a real hou.Parm and the LOCKED-node test
        # fixture both return explicitly, while an unconfigured mock's
        # auto-vivified MagicMock never satisfies.
        if parm.isLocked() is True:
            dropped.add(eid)
            continue

        validated[eid] = entry
    return validated, dropped


def _apply_validated_keyframes(validated: dict) -> None:
    """ATOMICALLY apply the ALREADY-preflight-validated {event_id:
    {node,parm,frames}} set inside exactly ONE hou.undos.group -- for
    each entry, convert frames [[f,v]...] -> [{frame,value}...]
    (Blocker-1) and call the SHIPPED _set_keyframes once per entry
    (Major-8). Every entry here was pre-validated (node/parm exist,
    in-scope, well-formed frames) -- so a mid-apply failure can only be
    a genuine TOCTOU race (a parm locked by a callback strictly AFTER
    preflight observed isLocked() -> False, or any other genuine
    setKeyframe error).

    REVISION 3 (operator decision 2026-07-11 -- SUPERSEDES the round-2/
    round-3 bespoke `_delete_keyframe` compensating-rollback approach;
    it was the source of a data-loss Blocker: deleting a keyframe the
    call had merely OVERWRITTEN instead of restoring its PRIOR value).
    The atomicity guarantee for this GATED tool is now: the ENTIRE
    validated set is applied inside exactly ONE `hou.undos.group(...)`
    (a single native undo step the operator reverts with one Undo); on
    ANY mid-apply raise, this function does NOT attempt to undo already-
    written entries itself -- it re-raises the ORIGINAL exception for
    the caller (compile_timeline) to degrade the WHOLE call to the
    documented {ok:false, error:'partial apply -- ... use Undo to
    revert'} shape. Recovery of a partial write is the operator's native
    Undo, not a bespoke compensating delete.

    The exception is caught and re-raised INSIDE this function (rather
    than left to propagate naturally out of the `with` block) because a
    MagicMock-backed `hou.undos.group(...)` (the off-DCC test double)
    auto-vivifies `__exit__` to a truthy MagicMock, which would SILENTLY
    SUPPRESS an exception that propagates through the `with` block's own
    exception machinery. Catching it inside the block (so the `with`
    exits NORMALLY) and re-raising immediately after sidesteps that
    off-DCC mock footgun while behaving identically against a real
    `hou.undos.group`, which does not suppress exceptions either."""
    _hou = _current_hou()
    caught: list = []
    with _hou.undos.group("compile_timeline apply"):
        try:
            for entry in validated.values():
                converted = [{"frame": f, "value": v} for f, v in entry["frames"]]
                _anim_handlers._set_keyframes(entry["node"], entry["parm"], converted)
        except Exception as exc:  # noqa: BLE001 -- re-raised verbatim below
            caught.append(exc)
    if caught:
        raise caught[0]


def _preview_compile_timeline(params: dict) -> dict:
    """Return the 109-gate approval payload for compile_timeline WITHOUT
    applying or mutating (Major-6). Resolves `network` UNCONDITIONALLY
    up front (ROUND-3 M3 -- the SAME _resolve_network step apply
    performs, BEFORE compile_plan even runs) AND runs compile_plan
    (pure) AND the SAME read-only preflight validation apply performs --
    NO writes -- so the preview's would-apply/unresolved split EXACTLY
    matches what apply would do.

    Called POSITIONALLY by the gate middleware as ``preview_fn(params)`` --
    a single ``params: dict`` argument (matches _preview_solve_layout's
    convention).

    Returns:
        {"would_set_keyframes": [{"node", "parm", "frame_count"}, ...],
         "unresolved": [...], "event_graph": {"nodes", "edges"}}
        OR, when `network` itself is unresolvable (ROUND-3 M3 --
        checked UNCONDITIONALLY, regardless of `events`, even
        events=[]): {"ok": False, "error": ...} -- the SAME terminal
        shape compile_timeline (apply) returns for the SAME input, so
        the preview never diverges from what apply would do (Major-6).

    Raises:
        ValueError: a malformed event (propagated from
            temporal_reasoning_model._compile_events) -- the gate DENIES
            the call (mirrors _preview_solve_layout's raise->DENY).
    """
    network = params["network"]
    events = params.get("events", [])

    # ROUND-3 FIX-PASS (M3): contract step 1 -- resolve `network`
    # UNCONDITIONALLY, before compile_plan even runs. An unresolvable
    # `network` degrades the WHOLE call regardless of `events` (even
    # events=[] -- there is then no per-entry target to trigger a
    # node-resolution failure of its own, so this check must not depend
    # on one).
    network_node = _resolve_network(network)
    if network_node is None:
        return {"ok": False, "error": f"Node not found: {network}"}
    canonical_network = network_node.path()

    internal = _model._compile_events(events, network)  # a ValueError -> gate DENY

    validated, dropped_ids = _preflight_validate(
        internal["compiled_by_id"], canonical_network
    )

    would_set_keyframes = [
        {"node": entry["node"], "parm": entry["parm"], "frame_count": len(entry["frames"])}
        for entry in validated.values()
    ]
    unresolved = list(internal["unresolved_ids"]) + [
        eid for eid in dropped_ids if eid not in internal["unresolved_ids"]
    ]

    return {
        "would_set_keyframes": would_set_keyframes,
        "unresolved": unresolved,
        "event_graph": internal["event_graph"],
    }


def compile_timeline(
    *,
    network: str,
    events: list,
    frame_range: list,
    apply: bool = True,
) -> dict:
    """Compile an agent-authored event-timeline into concrete Houdini
    KEYFRAMES on the EXISTING sim network -- scoped, preflight-validated,
    and GATED (Capability.MUTATING, PP12-109 security gate).

    STEPS (plan pp12-117c lockedFieldContract "REVISION 3 -- B2 ATOMICITY
    GUARANTEE, CLARIFIED"; M3 unconditional network resolution and
    B1-residual canonical network comparison from the round-3 fix-pass are
    UNCHANGED):
      (1) resolve `network` ONCE, UNCONDITIONALLY, via _resolve_network --
          BEFORE anything else, regardless of `events` (even events=[]).
          Unresolvable (empty/falsy `network`, or hou.node(network) is
          None) -> {"ok": False, "error": ...} IMMEDIATELY, WITHOUT
          raising -- the scene-resolution boundary case, distinct from
          an ordinary per-entry unresolved report.
      (2) delegate the pure translation to
          temporal_reasoning_model._compile_events(events, network) -- a
          caller-contract ValueError (an unknown event type, a malformed
          field, a cyclic/dangling/duplicate causes graph) PROPAGATES,
          NEVER folded into {ok:false}.
      (3) PREFLIGHT-VALIDATE every compiled entry (node CANONICALLY
          resolves + in-scope of the RESOLVED, CANONICAL network path +
          parm exists + parm NOT LOCKED + well-formed frames,
          Blocker-1/Blocker-2/B1-residual) BEFORE any write; a failing
          entry moves to `unresolved` and is dropped from the applied
          set.
      (4) if apply=True: ATOMICALLY apply ONLY the validated set (see
          _apply_validated_keyframes) inside ONE hou.undos.group.
          REVISION 3: on ANY mid-apply write failure, the handler STOPS
          and degrades the WHOLE call to {ok:false, error:'partial apply
          -- the gated call was interrupted mid-write; use Undo to
          revert'} -- NO bespoke rollback; recovery is the operator's
          native Undo on the single undo-group. apply=False makes ZERO
          _set_keyframes calls.
      (5) return {compiled, event_graph, applied, unresolved} -- unresolved
          is the union of compile_plan's own unresolved
          (type-inferred/threshold/causally-impossible) plus any
          preflight-dropped ids.

    The MODEL CALL (temporal_reasoning_model._compile_events) is OUTSIDE
    the scene-resolution guard -- its ValueError propagates as the
    dispatcher's standard error envelope, mirroring
    spatial_reasoning_handlers.solve_layout / assert_simulation's
    established success/error boundary.
    """
    # ROUND-3 FIX-PASS (M3): contract step 1 -- resolve `network`
    # UNCONDITIONALLY, before the model translation even runs. An
    # unresolvable `network` degrades the WHOLE call regardless of
    # `events` (even events=[]).
    network_node = _resolve_network(network)
    if network_node is None:
        return {"ok": False, "error": f"Node not found: {network}"}
    canonical_network = network_node.path()

    internal = _model._compile_events(events, network)  # a ValueError PROPAGATES

    # Preflight validation NEVER writes and NEVER raises the apply-exception
    # types (it only drops non-settable entries into `dropped_ids`) -- so it
    # stays OUTSIDE the apply try/except below. Only the apply phase itself
    # (_apply_validated_keyframes) can raise a genuine mid-apply TOCTOU
    # failure.
    validated, dropped_ids = _preflight_validate(
        internal["compiled_by_id"], canonical_network
    )
    if apply:
        try:
            _apply_validated_keyframes(validated)
        except Exception:
            # REVISION 3 (B2 ATOMICITY GUARANTEE, CLARIFIED, round-3
            # re-review fix-pass): the locked contract requires catching
            # ANY mid-apply raise -- not merely the narrow
            # _apply_exception_types() tuple (hou.OperationFailed,
            # AttributeError, hou.PermissionError when genuine). A
            # RuntimeError (or any other exception) from a genuine
            # mid-apply TOCTOU failure -- e.g. an unexpected internal
            # error inside setKeyframe -- must degrade the WHOLE call
            # identically to the narrower types; there is no bespoke
            # rollback to invoke, so no exception type distinction
            # matters here. NO bespoke rollback -- the entire apply
            # already ran inside ONE hou.undos.group; the operator
            # recovers a partial write via a single native Undo. Return
            # the EXACT documented partial-apply error string (never the
            # raw exception text).
            return {
                "ok": False,
                "error": (
                    "partial apply — the gated call was interrupted "
                    "mid-write; use Undo to revert"
                ),
            }

    unresolved = list(internal["unresolved_ids"]) + [
        eid for eid in dropped_ids if eid not in internal["unresolved_ids"]
    ]
    compiled = {
        "keyframes": [
            {"node": entry["node"], "parm": entry["parm"], "frames": entry["frames"]}
            for entry in validated.values()
        ],
        "chop_triggers": [],
        "dop_parms": [],
    }

    return {
        "compiled": compiled,
        "event_graph": internal["event_graph"],
        "applied": apply,
        "unresolved": unresolved,
    }


register_handler(
    "compile_timeline",
    compile_timeline,
    Capability.MUTATING,
    preview_fn=_preview_compile_timeline,
    preview_required=True,
)
