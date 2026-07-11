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
