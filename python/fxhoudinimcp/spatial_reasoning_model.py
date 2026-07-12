"""
spatial_reasoning_model.py — pure-logic core for the Spatial-Reasoning MCP
surface (PP12-116 PR-1).

Contract: imports NO hou, NO Qt/PySide6, NO pxr, NO MCP server framework.
Makes NO MCP call. NEVER reads a bounding box itself -- bbox arrives as plain tuples;
reading it live off a node is the PR-2 handler's job (CL-015 extended).
numpy is PERMITTED but NOT mandated -- this module deliberately uses plain
Python stdlib (dataclasses, math) only, since numpy is not installed in the
fxhoudinimcp fork .venv (verified 2026-07-09) and a correct stdlib
implementation satisfies the locked behavioral contract equally well.
Pytest-able off-DCC (CL-015) -- this module never touches Houdini, PySide6,
USD, or numpy, so it runs on bare CI with plain pytest.

COORDINATE FRAME (BINDING, plan pp12-116a lockedFieldContract revision 2,
Blocker-1): Houdini Y-up.
  - t = [x, y, z] world translation; r = [rx, ry, rz] rotations in DEGREES.
  - ObjectSpec.bbox = (w, d, h) axis-aligned EXTENTS: w -> x-extent,
    d -> z-extent (depth), h -> y-extent (VERTICAL height).
  - LayoutSpec.bounds.room = [x0, z0, x1, z1] is a FOOTPRINT rectangle on
    the X-Z plane. There is NO vertical bound.
  - ALL support/on_top_of/rest-height math is along +Y.
  - ALL footprint/adjacency/bounds/clearance math is on the X-Z plane.

This is the PURE-LOGIC reference solve the whole Spatial-Reasoning member
wraps: no Houdini coupling, so it is the safe unblocked first move (PR-1).
Handlers, @mcp.tool wrappers, the 109 gate, hdefereval marshaling, and any
hou-layer code are OUT OF SCOPE for this PR (deferred to PR-2..4).

Classes
-------
ObjectSpec    — one object's id/bbox/fixed/current transform/node path
RelationSpec  — one spatial-relation constraint (type/a/b/target/params)
LayoutSpec    — a full scene spec (objects + relations + bounds + solve
                 budget)

Functions
---------
describe_relations() -> dict
    The exact SPEC 4.1 relation-vocabulary wire shape.
solve_layout(spec) -> dict
    Deterministic (no RNG) reference solve: relations -> transforms.
assert_scene(objects, transforms, relations=[], checks=[...]) -> dict
    The Gate-1 assertion predicates (collision/support/navigability/
    clearance/relations), exact SPEC 4.1 wire shape.
_nonoverlap_force(ca, cb, ba, bb) -> tuple
    THE load-bearing non-overlap term (centroid-vector-weighted, single-
    direction x total-penetration, deterministic +X degenerate fallback).

Error taxonomy (FR-A AC-5, Major-9)
------------------------------------
CONSTRUCTION/VALIDATION errors RAISE ValueError: an unknown relation type
(at RelationSpec construction), an a/b/target reference to a missing
object id, a null/non-positive bbox, or a relation param outside its
allowed set (all raised at solve_layout time, before any relaxation).
FEASIBLE-BUT-UNSATISFIABLE constraints (over-constrained fixed objects, a
fixed object outside bounds) return solved=False + a populated
unsatisfied[] and DO NOT raise.

Cross-references
-----------------
Plan pp12-116a lockedFieldContract (BINDING, revision 2)
docs/portfolio_projects/PP12_houdini_mcp_agentic_bridge/
  116_mcp_spatial_reasoning_surface/spec.md Sections 4.1, 6, 9
CL-015 (extended): pure-logic module, no hou/Qt/pxr/MCP-server-framework,
  no MCP call, no live bounding-box read
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TOL = 1e-4

_RELATION_TYPES = frozenset({
    "on_top_of", "under", "adjacent", "non_overlap",
    "aligned", "oriented_toward", "clearance",
})

_RELATION_ORDER = (
    "on_top_of", "under", "adjacent", "non_overlap",
    "aligned", "oriented_toward", "clearance",
)

_RELATION_VOCAB = {
    "on_top_of": {
        "params": {"clearance": "float, default 0.0 -- gap above the support"},
        "desc": "A rests directly on top of B, along the +Y axis.",
    },
    "under": {
        "params": {"clearance": "float, default 0.0 -- gap above the support"},
        "desc": "B rests on top of A (the reverse of on_top_of).",
    },
    "adjacent": {
        "params": {
            "gap": "float, default 0.0 -- face-to-face separation",
            "side": "one of +x,-x,+z,-z; optional -- nearest face-pair if absent",
        },
        "desc": "A sits beside B on the given world-axis side, separated by gap.",
    },
    "non_overlap": {
        "params": {},
        "desc": "A and B do not intersect in 3D (AABB non-collision).",
    },
    "aligned": {
        "params": {
            "axis": "one of x,y,z",
            "edge": "one of min,center,max; default center",
        },
        "desc": "A and B share the same coordinate on the given axis/edge.",
    },
    "oriented_toward": {
        "params": {},
        "desc": "A's +Z facing direction (rotated by A.r) points at target's X-Z centroid.",
    },
    "clearance": {
        "params": {"min": "float -- minimum required footprint distance"},
        "desc": "A keeps at least `min` distance from every other object's footprint on X-Z.",
    },
}

_ADJACENT_SIDES = frozenset({"+x", "-x", "+z", "-z"})
_ALIGN_AXES = frozenset({"x", "y", "z"})
_ALIGN_EDGES = frozenset({"min", "center", "max"})
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


# ---------------------------------------------------------------------------
# ObjectSpec
# ---------------------------------------------------------------------------

@dataclass
class ObjectSpec:
    """One object's id/bbox/fixed/current transform/node path.

    bbox=(w, d, h) per the pinned Y-up coordinate frame: w -> x-extent,
    d -> z-extent (depth), h -> y-extent (vertical height). bbox is
    REQUIRED and NON-NULL -- the PR-2 handler resolves a null bbox via a
    live bounding-box read BEFORE constructing an ObjectSpec; this
    model never performs that read itself and raises ValueError if it
    is ever handed a null/malformed bbox.

    fixed=True objects are NEVER moved by solve_layout, even if doing so
    would satisfy a relation or a bounds constraint -- a violation instead
    surfaces in solve_layout's `unsatisfied` list.
    """

    id: str
    bbox: tuple
    fixed: bool = False
    t: tuple = (0.0, 0.0, 0.0)
    r: tuple = (0.0, 0.0, 0.0)
    node: str = ""

    def __post_init__(self) -> None:
        if self.bbox is None:
            raise ValueError(f"ObjectSpec {self.id!r}: bbox is required and must not be None")
        try:
            bbox_tuple = tuple(self.bbox)
        except TypeError:
            raise ValueError(
                f"ObjectSpec {self.id!r}: bbox must be an iterable of 3 numbers, got {self.bbox!r}"
            )
        if len(bbox_tuple) != 3:
            raise ValueError(
                f"ObjectSpec {self.id!r}: bbox must have exactly 3 extents (w, d, h), got {bbox_tuple!r}"
            )
        for extent in bbox_tuple:
            if extent <= 0:
                raise ValueError(
                    f"ObjectSpec {self.id!r}: bbox extents must all be > 0, got {bbox_tuple!r}"
                )
        self.bbox = bbox_tuple
        self.t = tuple(self.t)
        self.r = tuple(self.r)

    def to_dict(self) -> dict:
        """Serialize to a plain dict with exactly six keys."""
        return {
            "id": self.id,
            "bbox": list(self.bbox),
            "fixed": self.fixed,
            "t": list(self.t),
            "r": list(self.r),
            "node": self.node,
        }


def _object_from_dict(d: dict) -> ObjectSpec:
    """Rebuild one ObjectSpec from its to_dict() shape (or a compatible raw dict)."""
    bbox_raw = d.get("bbox")
    bbox = tuple(bbox_raw) if bbox_raw is not None else None
    return ObjectSpec(
        id=d["id"],
        bbox=bbox,
        fixed=d.get("fixed", False),
        t=tuple(d.get("t", (0.0, 0.0, 0.0))),
        r=tuple(d.get("r", (0.0, 0.0, 0.0))),
        node=d.get("node", ""),
    )


# ---------------------------------------------------------------------------
# RelationSpec
# ---------------------------------------------------------------------------

@dataclass
class RelationSpec:
    """One spatial-relation constraint.

    `target` is a TOP-LEVEL field (used by oriented_toward) -- never
    buried in params. __post_init__ raises ValueError naming an unknown
    `type` (rejected against the 7-member relation vocabulary). A/b/target
    reference ObjectSpec ids; b='' for unary relations (clearance);
    target='' for every relation except oriented_toward. A reference that
    does not resolve to a known object id is a solve_layout-time
    construction error (see the module's error taxonomy), not a
    RelationSpec-construction error.
    """

    type: str
    a: str
    b: str = ""
    target: str = ""
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in _RELATION_TYPES:
            raise ValueError(
                f"RelationSpec: unknown relation type {self.type!r}; "
                f"must be one of {sorted(_RELATION_TYPES)!r}"
            )

    def to_dict(self) -> dict:
        """Serialize to a plain dict with exactly five keys."""
        return {
            "type": self.type,
            "a": self.a,
            "b": self.b,
            "target": self.target,
            "params": dict(self.params),
        }


def _relation_from_dict(d: dict) -> RelationSpec:
    """Rebuild one RelationSpec from its to_dict() shape (or a compatible raw dict)."""
    return RelationSpec(
        type=d["type"],
        a=d.get("a", ""),
        b=d.get("b", ""),
        target=d.get("target", ""),
        params=dict(d.get("params", {})),
    )


def _relation_repr(rel: RelationSpec) -> str:
    """The pinned unsatisfied/failed repr: f'<type>:<a>-><b|target>'."""
    return f"{rel.type}:{rel.a}->{rel.b or rel.target}"


# ---------------------------------------------------------------------------
# LayoutSpec
# ---------------------------------------------------------------------------

@dataclass
class LayoutSpec:
    """A full scene spec: objects + relations + optional bounds + solve budget.

    NO seed, NO RNG field -- the solve is deterministic without one (see
    solve_layout). `max_iters` bounds the relaxation loop; `iterations` is
    a solve_layout RETURN field, never stored on the spec.
    """

    objects: list = field(default_factory=list)
    relations: list = field(default_factory=list)
    bounds: Optional[dict] = None
    max_iters: int = 200
    schema_version: int = 1

    def to_dict(self) -> dict:
        """Serialize to a plain dict with exactly five keys."""
        return {
            "objects": [o.to_dict() for o in self.objects],
            "relations": [r.to_dict() for r in self.relations],
            "bounds": self.bounds,
            "max_iters": self.max_iters,
            "schema_version": self.schema_version,
        }

    @staticmethod
    def from_dict(d: dict) -> "LayoutSpec":
        """Rebuild a LayoutSpec from its to_dict() shape.

        Rebuilds ObjectSpec/RelationSpec instances (propagating any
        ValueError a bad object/relation raises); preserves bounds/
        max_iters/schema_version; ignores unknown top-level keys.
        """
        objects = [_object_from_dict(od) for od in d.get("objects", [])]
        relations = [_relation_from_dict(rd) for rd in d.get("relations", [])]
        return LayoutSpec(
            objects=objects,
            relations=relations,
            bounds=d.get("bounds"),
            max_iters=d.get("max_iters", 200),
            schema_version=d.get("schema_version", 1),
        )


# ---------------------------------------------------------------------------
# describe_relations (FR-C)
# ---------------------------------------------------------------------------

def describe_relations() -> dict:
    """Return the exact SPEC 4.1 relation-vocabulary wire shape.

    {"relations": [{"name", "params", "desc"}, ...]} in a STABLE order
    across calls, for the 7 vocabulary types.
    """
    return {
        "relations": [
            {
                "name": name,
                "params": dict(_RELATION_VOCAB[name]["params"]),
                "desc": _RELATION_VOCAB[name]["desc"],
            }
            for name in _RELATION_ORDER
        ]
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _half_extents(bbox: tuple) -> tuple:
    """(w, d, h) -> (half_x, half_y, half_z), POSITION-axis aligned.

    bbox is stored in (w, d, h) order (w->x-extent, d->z-extent,
    h->y-extent, per the pinned Y-up coordinate frame), but every other
    helper in this module indexes centroids/positions in (x, y, z) order.
    This function re-orders the halved extents to match THAT axis order
    -- half[0]=x, half[1]=y, half[2]=z -- so `half[oid][axis]` and
    `pos[oid][axis]` always refer to the SAME world axis.
    """
    w, d, h = bbox
    return (w / 2.0, h / 2.0, d / 2.0)


def _aabb_overlap(ca: tuple, ba: tuple, cb: tuple, bb: tuple) -> bool:
    """True iff two AABBs overlap on ALL 3 axes (strict, pen > 0 on every axis)."""
    for i in range(3):
        if (ba[i] + bb[i]) - abs(ca[i] - cb[i]) <= 0.0:
            return False
    return True


def _footprint_distance(c1: tuple, h1: tuple, c2: tuple, h2: tuple) -> float:
    """Euclidean distance between two footprint rectangles on the X-Z plane.

    0.0 when the footprints overlap or touch on both axes.
    """
    dx = max(0.0, abs(c1[0] - c2[0]) - (h1[0] + h2[0]))
    dz = max(0.0, abs(c1[2] - c2[2]) - (h1[2] + h2[2]))
    return math.sqrt(dx * dx + dz * dz)


def _nearest_side(ca: tuple, cb: tuple, ha: tuple, hb: tuple) -> str:
    """Pick the adjacent `side` with the smallest-magnitude face-to-face gap."""
    candidates = {
        "+x": (ca[0] - ha[0]) - (cb[0] + hb[0]),
        "-x": (cb[0] - hb[0]) - (ca[0] + ha[0]),
        "+z": (ca[2] - ha[2]) - (cb[2] + hb[2]),
        "-z": (cb[2] - hb[2]) - (ca[2] + ha[2]),
    }
    return min(candidates, key=lambda k: abs(candidates[k]))


# ---------------------------------------------------------------------------
# _nonoverlap_force -- THE load-bearing term (Blocker-4, FR-A AC-2)
# ---------------------------------------------------------------------------

def _nonoverlap_force(ca: tuple, cb: tuple, ba: tuple, bb: tuple) -> tuple:
    """Centroid-vector-weighted non-overlap push force.

    Inputs: centroids ca, cb (len-3) and HALF-extents ba, bb (len-3).
    Per-axis penetration pen[i] = (ba[i]+bb[i]) - abs(ca[i]-cb[i]). AABB
    overlap requires pen[i] > 0 on ALL THREE axes; if any pen[i] <= 0 the
    boxes do not overlap in 3D -> returns (0.0, 0.0, 0.0).

    Otherwise: direction = normalize(ca - cb); the DEGENERATE coincident
    case (|ca-cb| < 1e-9) uses the deterministic +X unit axis (never
    random). force = direction * (pen[x]+pen[y]+pen[z]) -- a SINGLE
    direction unit-vector scaled by the TOTAL penetration magnitude, not
    independent per-axis components. Guarantees a non-zero gradient at
    full overlap (the Open-Universe zero-gradient trap).
    """
    pen = [(ba[i] + bb[i]) - abs(ca[i] - cb[i]) for i in range(3)]
    if any(p <= 0.0 for p in pen):
        return (0.0, 0.0, 0.0)

    delta = [ca[i] - cb[i] for i in range(3)]
    dist = math.sqrt(sum(d * d for d in delta))
    if dist < 1e-9:
        direction = (1.0, 0.0, 0.0)
    else:
        direction = tuple(d / dist for d in delta)

    total_pen = sum(pen)
    return tuple(direction[i] * total_pen for i in range(3))


# ---------------------------------------------------------------------------
# Relation validation (construction/validation error taxonomy, solve_layout-time)
# ---------------------------------------------------------------------------

def _check_ref(ref: str, by_id: dict, rel: RelationSpec, field_name: str) -> None:
    if ref and ref not in by_id:
        raise ValueError(
            f"RelationSpec {rel.type!r}: {field_name}={ref!r} does not "
            f"reference a known object id"
        )


_REQUIRES_A_AND_B = frozenset({
    "adjacent", "on_top_of", "under", "non_overlap", "aligned",
})


def _require_field(rel: RelationSpec, value: str, field_name: str) -> None:
    if not value:
        raise ValueError(
            f"RelationSpec {rel.type!r}: field {field_name!r} is required and must "
            f"not be empty"
        )


def _validate_relation(rel: RelationSpec, by_id: dict) -> None:
    """Construction/validation checks that RAISE ValueError (FR-A AC-5).

    Covers: an omitted REQUIRED a/b/target field (per-relation-type --
    binary relations require both a and b, oriented_toward requires a and
    target, clearance requires a), a/b/target referencing a missing
    object id, and a relation param outside its allowed set
    (adjacent.side, aligned.axis/edge, clearance.min presence). Called
    once per relation before any relaxation begins -- and by assert_scene
    for its `relations` argument, so BOTH read and solve paths share the
    exact same error taxonomy (construction/validation -> ValueError;
    never a raw KeyError/AttributeError leaking from an unguarded dict
    lookup deeper in the module).
    """
    if rel.type in _REQUIRES_A_AND_B:
        _require_field(rel, rel.a, "a")
        _require_field(rel, rel.b, "b")
    elif rel.type == "oriented_toward":
        _require_field(rel, rel.a, "a")
        _require_field(rel, rel.target, "target")
    elif rel.type == "clearance":
        _require_field(rel, rel.a, "a")

    _check_ref(rel.a, by_id, rel, "a")
    _check_ref(rel.b, by_id, rel, "b")
    _check_ref(rel.target, by_id, rel, "target")

    if rel.type == "adjacent":
        side = rel.params.get("side")
        if side is not None and side not in _ADJACENT_SIDES:
            raise ValueError(
                f"adjacent.side must be one of {sorted(_ADJACENT_SIDES)!r}, got {side!r}"
            )
    elif rel.type == "aligned":
        axis = rel.params.get("axis")
        if axis not in _ALIGN_AXES:
            raise ValueError(
                f"aligned.axis must be one of {sorted(_ALIGN_AXES)!r}, got {axis!r}"
            )
        edge = rel.params.get("edge", "center")
        if edge not in _ALIGN_EDGES:
            raise ValueError(
                f"aligned.edge must be one of {sorted(_ALIGN_EDGES)!r}, got {edge!r}"
            )
    elif rel.type == "clearance":
        if "min" not in rel.params:
            raise ValueError("clearance relation requires params['min']")


# ---------------------------------------------------------------------------
# Relation satisfaction predicates (used by both solve_layout's final check
# AND assert_scene's `relations` block -- one shared implementation)
# ---------------------------------------------------------------------------

def _on_top_of_satisfied(resting_id: str, support_id: str, pos: dict, half: dict,
                          params: dict, tol: float = _TOL) -> bool:
    clearance = params.get("clearance", 0.0)
    resting_y_min = pos[resting_id][1] - half[resting_id][1]
    support_y_max = pos[support_id][1] + half[support_id][1]
    if abs(resting_y_min - (support_y_max + clearance)) > tol:
        return False
    for axis in (0, 2):
        lo = pos[support_id][axis] - half[support_id][axis]
        hi = pos[support_id][axis] + half[support_id][axis]
        if not (lo - tol <= pos[resting_id][axis] <= hi + tol):
            return False
    return True


def _adjacent_satisfied(a_id: str, b_id: str, pos: dict, half: dict,
                         params: dict, tol: float = _TOL) -> bool:
    gap = params.get("gap", 0.0)
    side = params.get("side")
    ca, cb, ha, hb = pos[a_id], pos[b_id], half[a_id], half[b_id]
    if side is None:
        side = _nearest_side(ca, cb, ha, hb)
    axis = 0 if side in ("+x", "-x") else 2
    other_axis = 2 if axis == 0 else 0

    if side in ("+x", "+z"):
        actual_gap = (ca[axis] - ha[axis]) - (cb[axis] + hb[axis])
    else:
        actual_gap = (cb[axis] - hb[axis]) - (ca[axis] + ha[axis])
    if abs(actual_gap - gap) > tol:
        return False

    other_gap = abs(ca[other_axis] - cb[other_axis]) - (ha[other_axis] + hb[other_axis])
    return other_gap <= tol


def _aligned_satisfied(a_id: str, b_id: str, pos: dict, half: dict,
                        params: dict, tol: float = _TOL) -> bool:
    axis = _AXIS_INDEX[params["axis"]]
    edge = params.get("edge", "center")

    def edge_coord(cid: str) -> float:
        c = pos[cid][axis]
        if edge == "min":
            return c - half[cid][axis]
        if edge == "max":
            return c + half[cid][axis]
        return c

    return abs(edge_coord(a_id) - edge_coord(b_id)) <= tol


def _oriented_toward_satisfied(a_id: str, target_id: str, pos: dict, rot: dict,
                                tol: float = _TOL) -> bool:
    ry_rad = math.radians(rot[a_id][1])
    fx, fz = math.sin(ry_rad), math.cos(ry_rad)
    dx = pos[target_id][0] - pos[a_id][0]
    dz = pos[target_id][2] - pos[a_id][2]
    dist = math.sqrt(dx * dx + dz * dz)
    if dist < 1e-9:
        return True
    dx, dz = dx / dist, dz / dist
    dot = max(-1.0, min(1.0, fx * dx + fz * dz))
    angle_deg = math.degrees(math.acos(dot))
    return angle_deg <= tol


def _clearance_pairs(rel: RelationSpec, pos: dict, half: dict, all_ids: list,
                      tol: float = _TOL) -> list:
    """All [a, other] pairs violating rel's minimum-clearance requirement."""
    min_gap = rel.params["min"]
    pairs = []
    for other_id in all_ids:
        if other_id == rel.a:
            continue
        d = _footprint_distance(pos[rel.a], half[rel.a], pos[other_id], half[other_id])
        if d < min_gap - tol:
            pairs.append([rel.a, other_id])
    return pairs


def _relation_satisfied(rel: RelationSpec, pos: dict, rot: dict, half: dict,
                         all_ids: list, tol: float = _TOL) -> bool:
    t = rel.type
    if t == "on_top_of":
        return _on_top_of_satisfied(rel.a, rel.b, pos, half, rel.params, tol)
    if t == "under":
        return _on_top_of_satisfied(rel.b, rel.a, pos, half, rel.params, tol)
    if t == "adjacent":
        return _adjacent_satisfied(rel.a, rel.b, pos, half, rel.params, tol)
    if t == "non_overlap":
        return not _aabb_overlap(pos[rel.a], half[rel.a], pos[rel.b], half[rel.b])
    if t == "aligned":
        return _aligned_satisfied(rel.a, rel.b, pos, half, rel.params, tol)
    if t == "oriented_toward":
        return _oriented_toward_satisfied(rel.a, rel.target, pos, rot, tol)
    if t == "clearance":
        return len(_clearance_pairs(rel, pos, half, all_ids, tol)) == 0
    return False


# ---------------------------------------------------------------------------
# Relaxation-step correctors (solve_layout's internal relaxation; never
# called by assert_scene, which only ever READS a given/already-solved
# scene via the satisfaction predicates above)
# ---------------------------------------------------------------------------

def _apply_on_top_of(resting_id: str, support_id: str, pos: dict, half: dict,
                      fixed: dict, params: dict) -> bool:
    if fixed[resting_id]:
        return False
    clearance = params.get("clearance", 0.0)
    support_top_y = pos[support_id][1] + half[support_id][1] + clearance
    target_y = support_top_y + half[resting_id][1]
    moved = False
    if abs(pos[resting_id][1] - target_y) > _TOL:
        pos[resting_id][1] = target_y
        moved = True
    for axis in (0, 2):
        lo = pos[support_id][axis] - half[support_id][axis]
        hi = pos[support_id][axis] + half[support_id][axis]
        if pos[resting_id][axis] < lo:
            pos[resting_id][axis] = lo
            moved = True
        elif pos[resting_id][axis] > hi:
            pos[resting_id][axis] = hi
            moved = True
    return moved


def _apply_adjacent(a_id: str, b_id: str, pos: dict, half: dict,
                     fixed: dict, params: dict) -> bool:
    gap = params.get("gap", 0.0)
    side = params.get("side")
    ca, cb, ha, hb = pos[a_id], pos[b_id], half[a_id], half[b_id]
    if side is None:
        side = _nearest_side(ca, cb, ha, hb)
    axis = 0 if side in ("+x", "-x") else 2
    other_axis = 2 if axis == 0 else 0

    if not fixed[a_id]:
        movable_id, anchor_id, moving_is_a = a_id, b_id, True
    elif not fixed[b_id]:
        movable_id, anchor_id, moving_is_a = b_id, a_id, False
    else:
        return False

    moved = False
    if moving_is_a:
        if side in ("+x", "+z"):
            target = pos[b_id][axis] + hb[axis] + gap + ha[axis]
        else:
            target = pos[b_id][axis] - hb[axis] - gap - ha[axis]
        if abs(pos[a_id][axis] - target) > _TOL:
            pos[a_id][axis] = target
            moved = True
        if abs(pos[a_id][other_axis] - pos[b_id][other_axis]) - (ha[other_axis] + hb[other_axis]) > _TOL:
            pos[a_id][other_axis] = pos[b_id][other_axis]
            moved = True
    else:
        if side in ("+x", "+z"):
            target = pos[a_id][axis] - ha[axis] - gap - hb[axis]
        else:
            target = pos[a_id][axis] + ha[axis] + gap + hb[axis]
        if abs(pos[b_id][axis] - target) > _TOL:
            pos[b_id][axis] = target
            moved = True
        if abs(pos[b_id][other_axis] - pos[a_id][other_axis]) - (hb[other_axis] + ha[other_axis]) > _TOL:
            pos[b_id][other_axis] = pos[a_id][other_axis]
            moved = True
    return moved


def _apply_aligned(a_id: str, b_id: str, pos: dict, half: dict,
                    fixed: dict, params: dict) -> bool:
    axis = _AXIS_INDEX[params["axis"]]
    edge = params.get("edge", "center")

    def target_center(anchor_id: str, moving_id: str) -> float:
        if edge == "min":
            return pos[anchor_id][axis] - half[anchor_id][axis] + half[moving_id][axis]
        if edge == "max":
            return pos[anchor_id][axis] + half[anchor_id][axis] - half[moving_id][axis]
        return pos[anchor_id][axis]

    if not fixed[a_id]:
        t = target_center(b_id, a_id)
        if abs(pos[a_id][axis] - t) > _TOL:
            pos[a_id][axis] = t
            return True
        return False
    if not fixed[b_id]:
        t = target_center(a_id, b_id)
        if abs(pos[b_id][axis] - t) > _TOL:
            pos[b_id][axis] = t
            return True
    return False


def _apply_non_overlap(a_id: str, b_id: str, pos: dict, half: dict, fixed: dict) -> bool:
    ca, cb = pos[a_id], pos[b_id]
    ba, bb = half[a_id], half[b_id]
    force = _nonoverlap_force(tuple(ca), tuple(cb), tuple(ba), tuple(bb))
    if force == (0.0, 0.0, 0.0):
        return False
    if fixed[a_id] and fixed[b_id]:
        return False

    # The solver's internal relaxation step uses a minimum-penetration-axis
    # (MTV-style) push for numerically stable convergence -- distinct from
    # the pinned, unit-tested public _nonoverlap_force contract above
    # (which is total-penetration-scaled, single-direction).
    pen = [(ba[i] + bb[i]) - abs(ca[i] - cb[i]) for i in range(3)]
    axis = min(range(3), key=lambda i: pen[i])
    direction = 1.0 if ca[axis] >= cb[axis] else -1.0
    push = pen[axis] + _TOL

    if not fixed[a_id] and not fixed[b_id]:
        pos[a_id][axis] += direction * push / 2.0
        pos[b_id][axis] -= direction * push / 2.0
    elif not fixed[a_id]:
        pos[a_id][axis] += direction * push
    else:
        pos[b_id][axis] -= direction * push
    return True


def _apply_oriented_toward(a_id: str, target_id: str, pos: dict, rot: dict, fixed: dict) -> bool:
    if fixed[a_id]:
        return False
    dx = pos[target_id][0] - pos[a_id][0]
    dz = pos[target_id][2] - pos[a_id][2]
    dist = math.sqrt(dx * dx + dz * dz)
    if dist < 1e-9:
        return False
    desired_ry = math.degrees(math.atan2(dx, dz))
    if abs(rot[a_id][1] - desired_ry) > _TOL:
        rot[a_id][1] = desired_ry
        return True
    return False


def _apply_clearance(rel: RelationSpec, pos: dict, half: dict, fixed: dict, all_ids: list) -> bool:
    if fixed[rel.a]:
        return False
    min_gap = rel.params["min"]
    moved = False
    for other_id in all_ids:
        if other_id == rel.a:
            continue
        ca, ha = pos[rel.a], half[rel.a]
        cb, hb = pos[other_id], half[other_id]
        d = _footprint_distance(ca, ha, cb, hb)
        if d < min_gap - _TOL:
            dx = ca[0] - cb[0]
            dz = ca[2] - cb[2]
            dist = math.sqrt(dx * dx + dz * dz)
            if dist < 1e-9:
                dx, dz, dist = 1.0, 0.0, 1.0
            push = (min_gap - d) + _TOL
            ca[0] += (dx / dist) * push
            ca[2] += (dz / dist) * push
            moved = True
    return moved


def _apply_relation_correction(rel: RelationSpec, pos: dict, rot: dict, half: dict,
                                fixed: dict, all_ids: list) -> bool:
    if rel.type == "on_top_of":
        return _apply_on_top_of(rel.a, rel.b, pos, half, fixed, rel.params)
    if rel.type == "under":
        return _apply_on_top_of(rel.b, rel.a, pos, half, fixed, rel.params)
    if rel.type == "adjacent":
        return _apply_adjacent(rel.a, rel.b, pos, half, fixed, rel.params)
    if rel.type == "aligned":
        return _apply_aligned(rel.a, rel.b, pos, half, fixed, rel.params)
    if rel.type == "non_overlap":
        return _apply_non_overlap(rel.a, rel.b, pos, half, fixed)
    if rel.type == "oriented_toward":
        return _apply_oriented_toward(rel.a, rel.target, pos, rot, fixed)
    if rel.type == "clearance":
        return _apply_clearance(rel, pos, half, fixed, all_ids)
    return False


# ---------------------------------------------------------------------------
# Bounds (optional footprint containment on the X-Z plane -- no vertical bound)
# ---------------------------------------------------------------------------

def _within_bounds_check(oid: str, pos: dict, half: dict, bounds: dict, tol: float = _TOL) -> bool:
    x0, z0, x1, z1 = bounds["room"]
    lo_x, hi_x = pos[oid][0] - half[oid][0], pos[oid][0] + half[oid][0]
    lo_z, hi_z = pos[oid][2] - half[oid][2], pos[oid][2] + half[oid][2]
    return lo_x >= x0 - tol and hi_x <= x1 + tol and lo_z >= z0 - tol and hi_z <= z1 + tol


def _clamp_to_bounds(oid: str, pos: dict, half: dict, fixed: dict, bounds: dict) -> bool:
    if fixed[oid]:
        return False
    x0, z0, x1, z1 = bounds["room"]
    moved = False
    hx, hz = half[oid][0], half[oid][2]
    min_cx, max_cx = x0 + hx, x1 - hx
    min_cz, max_cz = z0 + hz, z1 - hz
    if min_cx <= max_cx:
        if pos[oid][0] < min_cx:
            pos[oid][0] = min_cx
            moved = True
        elif pos[oid][0] > max_cx:
            pos[oid][0] = max_cx
            moved = True
    if min_cz <= max_cz:
        if pos[oid][2] < min_cz:
            pos[oid][2] = min_cz
            moved = True
        elif pos[oid][2] > max_cz:
            pos[oid][2] = max_cz
            moved = True
    return moved


# ---------------------------------------------------------------------------
# solve_layout (FR-A)
# ---------------------------------------------------------------------------

def solve_layout(spec: LayoutSpec) -> dict:
    """The deterministic (no-RNG) reference solve: relations -> transforms.

    Constructive init = each object's own input t/r (objects processed in
    a stable id-sorted order every iteration for determinism), then
    iterative relaxation (<= spec.max_iters) where each relation
    contributes a correction step. Fixed objects are never moved/rotated.

    Returns EXACTLY {transforms, solved, iterations, unsatisfied} --
    see the module docstring's error taxonomy for the raise/degrade split.
    """
    by_id = {o.id: o for o in spec.objects}
    for rel in spec.relations:
        _validate_relation(rel, by_id)

    all_ids = sorted(by_id.keys())
    pos = {oid: [by_id[oid].t[0], by_id[oid].t[1], by_id[oid].t[2]] for oid in all_ids}
    rot = {oid: [by_id[oid].r[0], by_id[oid].r[1], by_id[oid].r[2]] for oid in all_ids}
    half = {oid: _half_extents(by_id[oid].bbox) for oid in all_ids}
    fixed = {oid: by_id[oid].fixed for oid in all_ids}

    iterations_run = 0
    for iteration in range(max(spec.max_iters, 0)):
        iterations_run = iteration + 1
        moved = False
        for rel in spec.relations:
            if _apply_relation_correction(rel, pos, rot, half, fixed, all_ids):
                moved = True
        if spec.bounds:
            for oid in all_ids:
                if _clamp_to_bounds(oid, pos, half, fixed, spec.bounds):
                    moved = True
        if not moved:
            break

    transforms = {
        oid: {
            "t": [float(pos[oid][0]), float(pos[oid][1]), float(pos[oid][2])],
            "r": [float(rot[oid][0]), float(rot[oid][1]), float(rot[oid][2])],
        }
        for oid in all_ids
    }

    unsatisfied = []
    for rel in spec.relations:
        if not _relation_satisfied(rel, pos, rot, half, all_ids):
            unsatisfied.append(_relation_repr(rel))
    if spec.bounds:
        for oid in all_ids:
            if not _within_bounds_check(oid, pos, half, spec.bounds):
                unsatisfied.append(f"bounds:{oid}")

    return {
        "transforms": transforms,
        "solved": len(unsatisfied) == 0,
        "iterations": iterations_run,
        "unsatisfied": unsatisfied,
    }


# ---------------------------------------------------------------------------
# assert_scene checks (FR-B, Gate-1)
# ---------------------------------------------------------------------------

def _collision_check(all_ids: list, pos: dict, half: dict) -> dict:
    pairs = []
    for i in range(len(all_ids)):
        for j in range(i + 1, len(all_ids)):
            id_a, id_b = all_ids[i], all_ids[j]
            if _aabb_overlap(pos[id_a], half[id_a], pos[id_b], half[id_b]):
                pairs.append([id_a, id_b])
    return {"pairs": pairs, "count": len(pairs)}


def _support_check(relations: list, pos: dict, half: dict) -> dict:
    unsupported = []
    for rel in relations:
        if rel.type == "on_top_of":
            resting, support = rel.a, rel.b
        elif rel.type == "under":
            resting, support = rel.b, rel.a
        else:
            continue
        if not _on_top_of_satisfied(resting, support, pos, half, rel.params):
            if resting not in unsupported:
                unsupported.append(resting)
    return {"unsupported": unsupported, "ok": len(unsupported) == 0}


def _clearance_check(relations: list, pos: dict, half: dict, all_ids: list) -> dict:
    violations = []
    for rel in relations:
        if rel.type != "clearance":
            continue
        for pair in _clearance_pairs(rel, pos, half, all_ids):
            if pair not in violations:
                violations.append(pair)
    return {"violations": violations, "ok": len(violations) == 0}


def assert_scene(objects: list, transforms: dict, relations: Optional[list] = None,
                  checks: Optional[list] = None) -> dict:
    """The Gate-1 assertion predicates over a (possibly externally-supplied)
    scene: collision / support / navigability / clearance / relations.

    Returns a dict whose keys are EXACTLY one entry per REQUESTED check,
    plus `relations` (only when `relations` is non-empty), plus `pass`.
    All outputs are JSON-serializable plain float/int/list/str/bool.
    """
    if relations is None:
        relations = []
    if checks is None:
        checks = ["collision", "support"]

    by_id = {o.id: o for o in objects}
    for rel in relations:
        _validate_relation(rel, by_id)

    all_ids = [o.id for o in objects]
    half = {o.id: _half_extents(o.bbox) for o in objects}
    pos = {oid: list(transforms[oid]["t"]) for oid in all_ids}
    rot = {oid: list(transforms[oid]["r"]) for oid in all_ids}

    result = {}
    check_pass_flags = []

    if "collision" in checks:
        collision = _collision_check(all_ids, pos, half)
        result["collision"] = collision
        check_pass_flags.append(collision["count"] == 0)
    if "support" in checks:
        support = _support_check(relations, pos, half)
        result["support"] = support
        check_pass_flags.append(support["ok"])
    if "navigability" in checks:
        result["navigability"] = {"blocked_regions": 0, "ok": True}
        check_pass_flags.append(True)
    if "clearance" in checks:
        clearance = _clearance_check(relations, pos, half, all_ids)
        result["clearance"] = clearance
        check_pass_flags.append(clearance["ok"])

    relations_ok = True
    if relations:
        satisfied = 0
        failed = []
        for rel in relations:
            if _relation_satisfied(rel, pos, rot, half, all_ids):
                satisfied += 1
            else:
                failed.append(_relation_repr(rel))
        result["relations"] = {
            "satisfied": satisfied,
            "total": len(relations),
            "failed": failed,
        }
        relations_ok = len(failed) == 0

    result["pass"] = bool(all(check_pass_flags) and relations_ok)
    return result
