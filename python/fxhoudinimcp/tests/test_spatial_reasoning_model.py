"""
Tests for spatial_reasoning_model.py — pure-logic core (PP12-116 PR-1).

TDD phase: RED — spatial_reasoning_model.py does NOT exist yet.
Expected failure: ModuleNotFoundError on import (collection error for the
whole file — see plan pp12-116a acceptanceTests).

Contract (plan pp12-116a lockedFieldContract, REVISION 2 — folds all 14
codex-adversarial-reviewer findings, 4 Blockers + 8 Majors + 2 Minors):
imports NO hou, NO Qt/PySide6, NO pxr, NO FastMCP. Makes NO MCP call. NEVER
reads a bounding box itself (bbox arrives as plain tuples — reading it is
the PR-2 handler's job, CL-015 extended). numpy is PERMITTED but NOT
mandated — a correct stdlib-only implementation is equally acceptable, and
this suite asserts ONLY the behavioral contract, never a numpy-API usage.
Plain Python, pytest-able off-DCC (CL-015).

COORDINATE FRAME (BINDING, rev-2 Blocker-1 — SUPERSEDES the pre-fold spec
§4.1 sketch): Houdini Y-up.
  - t = [x, y, z] world translation; r = [rx, ry, rz] rotations in DEGREES.
  - ObjectSpec.bbox = (w, d, h) axis-aligned EXTENTS: w -> x-extent,
    d -> z-extent (depth), h -> y-extent (VERTICAL height).
  - LayoutSpec.bounds.room = [x0, z0, x1, z1] is a FOOTPRINT rectangle on
    the X-Z plane. There is NO vertical bound.
  - ALL support/on_top_of/rest-height math is along +Y.
  - ALL footprint/adjacency/bounds/clearance math is on the X-Z plane.

Covers the public contract of:
  - ObjectSpec    (dataclass) — id, bbox, fixed=False, t, r, node; bbox
                                 REQUIRED/NON-NULL; to_dict()
  - RelationSpec  (dataclass) — type, a, b='', target='', params={};
                                 __post_init__ raises on unknown type;
                                 to_dict()
  - LayoutSpec    (dataclass) — objects, relations, bounds=None,
                                 max_iters=200, schema_version=1;
                                 to_dict()/from_dict() — NO seed/iterations
                                 key
  - describe_relations() -> dict           (FR-C)  — exact SPEC 4.1 shape
  - solve_layout(spec: LayoutSpec) -> dict (FR-A)  — deterministic, no RNG;
                                                       error taxonomy split
  - _nonoverlap_force(ca, cb, ba, bb) -> tuple      — THE load-bearing term
  - assert_scene(objects, transforms, relations=[], checks=[...]) -> dict
                                              (FR-B, Gate-1) — exact 4.1 keys

Cross-references:
  - Plan pp12-116a lockedFieldContract (BINDING, revision 2)
  - docs/portfolio_projects/PP12_houdini_mcp_agentic_bridge/
    116_mcp_spatial_reasoning_surface/spec.md §4.1, §6, §9
  - tdd-with-agents.md §4: hou-test writes red; hou-dev turns green
  - CL-015 (extended): pure-logic module — no hou/Qt/pxr/FastMCP, no MCP
    call, no get_bounding_box call, no bbox read of its own

Out of scope for PR-1 (deferred to PR-2..4, per plan Scope guard): MCP
registration, the 109 security gate, hdefereval marshaling, the
handlers/tools layer, a live get_bounding_box read. This suite asserts
NONE of those.

NOTE on Hypothesis: the fork .venv does not have the `hypothesis` package
installed (verified 2026-07-09) and this suite deliberately does not add
it as a new dependency mid-PR. The round-trip / determinism PROPERTY
coverage that a Hypothesis strategy would provide is instead delivered via
`@pytest.mark.parametrize` over a representative set of hand-authored
LayoutSpec configurations (see `_sample_specs()` / TestRoundTripAndDeterminismProperties)
— structurally the same "assert the invariant holds across many inputs"
discipline, without the new dependency.
"""

from __future__ import annotations

import json
import sys
import os

# ---------------------------------------------------------------------------
# Path bootstrap — allow running standalone and via pytest.
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_PKG_ROOT))

import pytest

# ---------------------------------------------------------------------------
# The module under test.
# This import MUST fail (ModuleNotFoundError) in the RED phase because
# spatial_reasoning_model.py does not exist yet. Every test below is
# therefore expected to report as a single collection error, not an
# individual per-test failure, until hou-dev lands the green implementation.
# ---------------------------------------------------------------------------
from fxhoudinimcp.spatial_reasoning_model import (
    ObjectSpec,
    RelationSpec,
    LayoutSpec,
    describe_relations,
    solve_layout,
    assert_scene,
    _nonoverlap_force,
)


# ===========================================================================
# Shared helper — recursive plain-JSON-type checker.
#
# Catches a numpy-scalar / hou-object leak WITHOUT importing numpy (numpy is
# not installed in this venv, and must not be a hard test dependency either
# — the contract requires numpy be PERMITTED, not MANDATED, so this helper
# asserts strict Python types (`type(x) is float`, not `isinstance`), which
# rejects numpy.float64 / numpy.int64 / a bare tuple exactly as it rejects
# any other non-plain type, since numpy scalars and tuples have a DIFFERENT
# `type()` than the plain builtins even when they subclass them.
# ===========================================================================

def _assert_plain_json_types(obj):
    """Recursively assert every value is a plain JSON-safe Python type."""
    if isinstance(obj, dict):
        assert type(obj) is dict, f"expected a plain dict, got {type(obj)!r}"
        for k, v in obj.items():
            assert type(k) is str, f"dict key must be a plain str, got {type(k)!r}"
            _assert_plain_json_types(v)
    elif isinstance(obj, list):
        assert type(obj) is list, f"expected a plain list, got {type(obj)!r}"
        for item in obj:
            _assert_plain_json_types(item)
    elif obj is None:
        pass
    elif isinstance(obj, bool):
        assert type(obj) is bool, f"expected a plain bool, got {type(obj)!r}"
    elif isinstance(obj, int):
        assert type(obj) is int, f"expected a plain int (no numpy int64 leak), got {type(obj)!r}"
    elif isinstance(obj, float):
        assert type(obj) is float, f"expected a plain float (no numpy float64 leak), got {type(obj)!r}"
    elif isinstance(obj, str):
        assert type(obj) is str, f"expected a plain str, got {type(obj)!r}"
    else:
        pytest.fail(f"Non-JSON-plain type leaked into output: {type(obj)!r} (value={obj!r})")


# ===========================================================================
# Section 1 — ObjectSpec dataclass
#
# Locked contract:
#   ObjectSpec(id: str, bbox: tuple, fixed: bool = False,
#              t: tuple = (0,0,0), r: tuple = (0,0,0), node: str = '')
#   bbox=(w,d,h) is REQUIRED/NON-NULL. Non-positive extent or a
#   non-3-length bbox raises ValueError.
#   to_dict() -> {id, bbox:[w,d,h], fixed, t:[x,y,z], r:[rx,ry,rz], node}.
# ===========================================================================

class TestObjectSpec:
    """ObjectSpec — locked fields, defaults, bbox validation, to_dict()."""

    def test_minimal_construction_and_defaults(self):
        o = ObjectSpec(id="a", bbox=(1.0, 2.0, 3.0))
        assert o.id == "a"
        assert tuple(o.bbox) == (1.0, 2.0, 3.0)
        assert o.fixed is False
        assert tuple(o.t) == (0.0, 0.0, 0.0)
        assert tuple(o.r) == (0.0, 0.0, 0.0)
        assert o.node == ""

    def test_fixed_and_transform_can_be_set(self):
        o = ObjectSpec(
            id="a", bbox=(1.0, 1.0, 1.0), fixed=True,
            t=(1.0, 2.0, 3.0), r=(0.0, 90.0, 0.0), node="/obj/geo1",
        )
        assert o.fixed is True
        assert tuple(o.t) == (1.0, 2.0, 3.0)
        assert tuple(o.r) == (0.0, 90.0, 0.0)
        assert o.node == "/obj/geo1"

    def test_to_dict_exact_keys(self):
        o = ObjectSpec(id="a", bbox=(1.0, 2.0, 3.0), fixed=True, t=(1.0, 1.0, 1.0), r=(0.0, 0.0, 0.0), node="/x")
        d = o.to_dict()
        assert set(d.keys()) == {"id", "bbox", "fixed", "t", "r", "node"}, (
            f"Unexpected keys in ObjectSpec.to_dict(): {set(d.keys())!r}"
        )

    def test_to_dict_bbox_t_r_are_lists(self):
        o = ObjectSpec(id="a", bbox=(1.0, 2.0, 3.0))
        d = o.to_dict()
        assert d["bbox"] == [1.0, 2.0, 3.0]
        assert d["t"] == [0.0, 0.0, 0.0]
        assert d["r"] == [0.0, 0.0, 0.0]

    def test_null_bbox_raises_value_error(self):
        """bbox is REQUIRED and NON-NULL — the model must never silently
        accept a null bbox (the PR-2 handler resolves it via
        get_bounding_box before construction; the model never sees null,
        but IF it does, it raises)."""
        with pytest.raises(ValueError):
            ObjectSpec(id="a", bbox=None)

    def test_non_3_length_bbox_raises(self):
        with pytest.raises(ValueError):
            ObjectSpec(id="a", bbox=(1.0, 2.0))
        with pytest.raises(ValueError):
            ObjectSpec(id="a", bbox=(1.0, 2.0, 3.0, 4.0))

    def test_zero_extent_raises(self):
        with pytest.raises(ValueError):
            ObjectSpec(id="a", bbox=(0.0, 1.0, 1.0))

    def test_negative_extent_raises(self):
        with pytest.raises(ValueError):
            ObjectSpec(id="a", bbox=(1.0, -1.0, 1.0))

    def test_is_json_serialisable(self):
        o = ObjectSpec(id="a", bbox=(1.0, 2.0, 3.0))
        json.dumps(o.to_dict())


# ===========================================================================
# Section 2 — RelationSpec dataclass
#
# Locked contract:
#   RelationSpec(type: str, a: str, b: str = '', target: str = '',
#                params: dict = field(default_factory=dict))
#   `target` is a TOP-LEVEL field (oriented_toward), not buried in params.
#   __post_init__ raises ValueError naming an unknown `type`.
#   to_dict() -> {type, a, b, target, params}.
# ===========================================================================

class TestRelationSpec:
    """RelationSpec — locked fields, type validation, target-as-top-level, to_dict()."""

    def test_minimal_construction_and_defaults(self):
        r = RelationSpec(type="non_overlap", a="x")
        assert r.type == "non_overlap"
        assert r.a == "x"
        assert r.b == ""
        assert r.target == ""
        assert r.params == {}

    def test_target_is_a_top_level_field(self):
        r = RelationSpec(type="oriented_toward", a="rug", target="door")
        assert r.target == "door"
        assert "target" not in r.params

    def test_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError):
            RelationSpec(type="teleport_to", a="x")

    def test_empty_string_type_raises(self):
        with pytest.raises(ValueError):
            RelationSpec(type="", a="x")

    def test_all_seven_relation_types_are_constructible(self):
        for rel_type in (
            "on_top_of", "under", "adjacent", "non_overlap",
            "aligned", "oriented_toward", "clearance",
        ):
            RelationSpec(type=rel_type, a="a", b="b", target="c")

    def test_to_dict_exact_keys(self):
        r = RelationSpec(type="adjacent", a="a", b="b", params={"side": "+x"})
        d = r.to_dict()
        assert set(d.keys()) == {"type", "a", "b", "target", "params"}, (
            f"Unexpected keys in RelationSpec.to_dict(): {set(d.keys())!r}"
        )

    def test_default_params_dict_is_independent_per_instance(self):
        """A bare mutable default ({} as a class attribute) would leak
        across instances -- the locked contract requires
        field(default_factory=dict)."""
        r1 = RelationSpec(type="non_overlap", a="a")
        r2 = RelationSpec(type="non_overlap", a="b")
        r1.params["side"] = "+x"
        assert r2.params == {}, (
            "RelationSpec.params default must be independent per instance "
            "(field(default_factory=dict), not a shared mutable default)"
        )

    def test_is_json_serialisable(self):
        r = RelationSpec(type="clearance", a="a", params={"min": 0.5})
        json.dumps(r.to_dict())


# ===========================================================================
# Section 3 — LayoutSpec dataclass
#
# Locked contract:
#   LayoutSpec(objects: list = field(default_factory=list),
#              relations: list = field(default_factory=list),
#              bounds: dict|None = None, max_iters: int = 200,
#              schema_version: int = 1)
#   to_dict() -> {objects, relations, bounds, max_iters, schema_version}
#     -- NO seed, NO iterations key.
#   from_dict rebuilds ObjectSpec/RelationSpec; preserves bounds/max_iters/
#   schema_version; ignores unknown top-level keys. NO RNG field.
# ===========================================================================

class TestLayoutSpec:
    """LayoutSpec — defaults, mutable-default independence, to_dict()/from_dict()."""

    def test_defaults(self):
        s = LayoutSpec()
        assert s.objects == []
        assert s.relations == []
        assert s.bounds is None
        assert s.max_iters == 200
        assert s.schema_version == 1

    def test_default_lists_are_independent_per_instance(self):
        s1 = LayoutSpec()
        s2 = LayoutSpec()
        s1.objects.append(ObjectSpec(id="x", bbox=(1.0, 1.0, 1.0)))
        assert s2.objects == [], (
            "LayoutSpec.objects default must be independent per instance "
            "(field(default_factory=list), not a shared mutable default)"
        )

    def test_to_dict_exact_keys(self):
        s = LayoutSpec(
            objects=[ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))],
            relations=[RelationSpec(type="clearance", a="a", params={"min": 1.0})],
        )
        d = s.to_dict()
        assert set(d.keys()) == {"objects", "relations", "bounds", "max_iters", "schema_version"}, (
            f"Unexpected keys in LayoutSpec.to_dict(): {set(d.keys())!r}"
        )

    def test_to_dict_has_no_seed_or_iterations_key(self):
        """iterations is a solve_layout RETURN field, not stored on the
        spec; there is no RNG field either."""
        s = LayoutSpec()
        d = s.to_dict()
        assert "seed" not in d
        assert "iterations" not in d
        assert "rng_seed" not in d

    def test_to_dict_objects_and_relations_are_lists_of_dicts(self):
        s = LayoutSpec(
            objects=[ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))],
            relations=[RelationSpec(type="clearance", a="a", params={"min": 1.0})],
        )
        d = s.to_dict()
        assert isinstance(d["objects"], list) and isinstance(d["objects"][0], dict)
        assert isinstance(d["relations"], list) and isinstance(d["relations"][0], dict)

    def test_from_dict_rebuilds_objects_and_relations(self):
        raw = {
            "objects": [{"id": "a", "bbox": [1.0, 1.0, 1.0], "fixed": False, "t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0], "node": ""}],
            "relations": [{"type": "clearance", "a": "a", "b": "", "target": "", "params": {"min": 1.0}}],
            "bounds": None,
            "max_iters": 200,
            "schema_version": 1,
        }
        s = LayoutSpec.from_dict(raw)
        assert len(s.objects) == 1
        assert isinstance(s.objects[0], ObjectSpec)
        assert s.objects[0].id == "a"
        assert len(s.relations) == 1
        assert isinstance(s.relations[0], RelationSpec)
        assert s.relations[0].type == "clearance"

    def test_from_dict_preserves_bounds_max_iters_schema_version(self):
        raw = {
            "objects": [], "relations": [],
            "bounds": {"room": [0.0, 0.0, 10.0, 10.0]},
            "max_iters": 50,
            "schema_version": 1,
        }
        s = LayoutSpec.from_dict(raw)
        assert s.bounds == {"room": [0.0, 0.0, 10.0, 10.0]}
        assert s.max_iters == 50
        assert s.schema_version == 1

    def test_from_dict_ignores_unknown_top_level_keys(self):
        raw = {
            "objects": [], "relations": [], "bounds": None,
            "max_iters": 200, "schema_version": 1, "future_field": 123,
        }
        s = LayoutSpec.from_dict(raw)  # must not raise
        assert s.max_iters == 200

    def test_from_dict_with_bad_object_bbox_raises(self):
        raw = {
            "objects": [{"id": "a", "bbox": None, "fixed": False, "t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0], "node": ""}],
            "relations": [], "bounds": None, "max_iters": 200, "schema_version": 1,
        }
        with pytest.raises(ValueError):
            LayoutSpec.from_dict(raw)

    def test_from_dict_with_unknown_relation_type_raises(self):
        raw = {
            "objects": [{"id": "a", "bbox": [1.0, 1.0, 1.0], "fixed": False, "t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0], "node": ""}],
            "relations": [{"type": "not_a_real_relation", "a": "a", "b": "", "target": "", "params": {}}],
            "bounds": None, "max_iters": 200, "schema_version": 1,
        }
        with pytest.raises(ValueError):
            LayoutSpec.from_dict(raw)

    def test_is_json_serialisable(self):
        s = LayoutSpec(objects=[ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))])
        json.dumps(s.to_dict())


# ===========================================================================
# Section 4 — describe_relations() -> dict (FR-C)
#
# Locked contract:
#   Returns the EXACT SPEC 4.1 dict: {relations:[{name,params,desc}, ...]}
#   in STABLE order for the 7 types on_top_of, under, adjacent,
#   non_overlap, aligned, oriented_toward, clearance.
# ===========================================================================

class TestDescribeRelations:
    """describe_relations — the exact FR-C wire shape."""

    def test_returns_relations_key_only(self):
        result = describe_relations()
        assert set(result.keys()) == {"relations"}, (
            f"describe_relations() must return exactly {{'relations': [...]}}, got keys {set(result.keys())!r}"
        )

    def test_all_seven_relation_types_present(self):
        result = describe_relations()
        names = {r["name"] for r in result["relations"]}
        assert names == {
            "on_top_of", "under", "adjacent", "non_overlap",
            "aligned", "oriented_toward", "clearance",
        }, f"describe_relations() must list exactly the 7 vocabulary types, got {names!r}"

    def test_each_entry_has_name_params_desc_keys(self):
        result = describe_relations()
        for entry in result["relations"]:
            assert set(entry.keys()) == {"name", "params", "desc"}, (
                f"Unexpected keys in a describe_relations() entry: {set(entry.keys())!r}"
            )
            assert isinstance(entry["desc"], str) and entry["desc"], (
                "each relation entry needs a non-empty one-line desc string"
            )

    def test_order_is_stable_across_calls(self):
        order1 = [e["name"] for e in describe_relations()["relations"]]
        order2 = [e["name"] for e in describe_relations()["relations"]]
        assert order1 == order2, "describe_relations() order must be STABLE across calls"

    def test_is_json_serialisable(self):
        json.dumps(describe_relations())


# ===========================================================================
# Section 5 — _nonoverlap_force(ca, cb, ba, bb) -> tuple
#
# THE load-bearing term (Blocker-4, FR-A AC-2). Locked contract:
#   pen[i] = (ba[i]+bb[i]) - abs(ca[i]-cb[i]).
#   Overlap requires pen[i] > 0 on ALL THREE axes; any pen[i] <= 0 ->
#   (0.0, 0.0, 0.0).
#   Otherwise: dir = normalize(ca - cb); degenerate coincident
#   (|ca-cb| < 1e-9) -> dir = +X unit axis, DETERMINISTIC never random.
#   force = dir * (pen[x]+pen[y]+pen[z]) -- a SINGLE direction scaled by
#   TOTAL penetration, NOT independent per-axis components.
# ===========================================================================

class TestNonoverlapForce:
    """_nonoverlap_force — pinned coincident/separated/partial-overlap cases."""

    def test_coincident_identical_boxes_gives_deterministic_plus_x_nonzero_force(self):
        ca = (0.0, 0.0, 0.0)
        cb = (0.0, 0.0, 0.0)
        ba = (0.5, 0.5, 0.5)
        bb = (0.5, 0.5, 0.5)
        force = _nonoverlap_force(ca, cb, ba, bb)
        assert len(force) == 3
        fx, fy, fz = force
        assert fx > 0.0, (
            f"two coincident identical boxes must separate along +X deterministically, got force={force!r}"
        )
        assert fy == 0.0
        assert fz == 0.0

    def test_coincident_force_magnitude_equals_total_penetration(self):
        """pen = (0.5+0.5)-0 = 1.0 per axis -> total penetration 3.0."""
        force = _nonoverlap_force((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        assert abs(force[0] - 3.0) < 1e-9, (
            f"coincident-box force magnitude must equal the TOTAL penetration (3.0), got {force!r}"
        )

    def test_separated_on_one_axis_of_three_gives_zero_force(self):
        """Overlapping on x,y but separated on z -> (0,0,0) (NOT a 3D overlap)."""
        ca = (0.0, 0.0, 0.0)
        cb = (0.0, 0.0, 5.0)
        ba = (0.5, 0.5, 0.5)
        bb = (0.5, 0.5, 0.5)
        force = _nonoverlap_force(ca, cb, ba, bb)
        assert force == (0.0, 0.0, 0.0), (
            f"boxes overlapping on x,y but separated on z must yield zero force "
            f"(this is the Open-Universe zero-gradient trap check), got {force!r}"
        )

    def test_fully_separated_boxes_give_zero_force(self):
        force = _nonoverlap_force((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        assert force == (0.0, 0.0, 0.0)

    def test_touching_exactly_at_boundary_gives_zero_force(self):
        """pen == 0 exactly is NOT an overlap -- pen must be STRICTLY > 0 on all 3 axes."""
        ca = (0.0, 0.0, 0.0)
        cb = (1.0, 0.0, 0.0)  # half-extent 0.5 boxes touching exactly at x=0.5
        ba = (0.5, 0.5, 0.5)
        bb = (0.5, 0.5, 0.5)
        force = _nonoverlap_force(ca, cb, ba, bb)
        assert force == (0.0, 0.0, 0.0)

    def test_partial_overlap_direction_points_from_b_to_a(self):
        """A non-degenerate overlap's force direction is the (ca-cb) vector, normalized."""
        ca = (1.0, 0.0, 0.0)
        cb = (0.0, 0.0, 0.0)
        ba = (0.6, 0.6, 0.6)
        bb = (0.6, 0.6, 0.6)
        force = _nonoverlap_force(ca, cb, ba, bb)
        assert force[0] > 0.0, "force must push A away from B along +X when A is offset in +X from B"
        assert abs(force[1]) < 1e-9
        assert abs(force[2]) < 1e-9

    def test_force_is_a_3_tuple(self):
        force = _nonoverlap_force((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        assert isinstance(force, tuple)
        assert len(force) == 3


# ===========================================================================
# Section 6 — solve_layout(spec) -> dict (FR-A) — functional behavior
#
# Locked contract: DETERMINISTIC (no RNG). Returns EXACTLY
#   {transforms:{id:{t:[x,y,z], r:[rx,ry,rz]}}, solved: bool,
#    iterations: int, unsatisfied:[<repr>]}.
# ===========================================================================

def _make_furniture_scene():
    """table (fixed) + vase (on_top_of table) + lamp (adjacent to vase,
    non_overlap with table). Exercises on_top_of + adjacent + non_overlap
    in a single solve, in the pinned Y-up frame."""
    table = ObjectSpec(id="table", bbox=(2.0, 1.0, 1.0), fixed=True, t=(0.0, 0.0, 0.0))
    vase = ObjectSpec(id="vase", bbox=(0.4, 0.4, 0.6), fixed=False, t=(0.0, 2.0, 0.0))
    lamp = ObjectSpec(id="lamp", bbox=(0.3, 0.3, 0.5), fixed=False, t=(3.0, 2.0, 3.0))
    relations = [
        RelationSpec(type="on_top_of", a="vase", b="table"),
        RelationSpec(type="adjacent", a="lamp", b="vase", params={"side": "+x", "gap": 0.1}),
        RelationSpec(type="non_overlap", a="lamp", b="table"),
    ]
    return LayoutSpec(objects=[table, vase, lamp], relations=relations, max_iters=200)


class TestSolveLayoutFunctional:
    """solve_layout — the reference solve, verified end-to-end via assert_scene."""

    def test_furniture_scene_solves_true_with_exact_return_keys(self):
        spec = _make_furniture_scene()
        result = solve_layout(spec)
        assert set(result.keys()) == {"transforms", "solved", "iterations", "unsatisfied"}, (
            f"Unexpected keys in solve_layout() return: {set(result.keys())!r}"
        )
        assert result["solved"] is True
        assert result["unsatisfied"] == []
        assert isinstance(result["iterations"], int)
        assert set(result["transforms"].keys()) == {"table", "vase", "lamp"}
        for oid in ("table", "vase", "lamp"):
            entry = result["transforms"][oid]
            assert set(entry.keys()) == {"t", "r"}, f"transforms[{oid!r}] must have exactly {{t, r}}, got {set(entry.keys())!r}"
            assert len(entry["t"]) == 3
            assert len(entry["r"]) == 3

    def test_fixed_object_transform_is_unchanged(self):
        """A fixed=True object's solved t/r == its input t/r ALWAYS."""
        spec = _make_furniture_scene()
        result = solve_layout(spec)
        assert result["transforms"]["table"]["t"] == [0.0, 0.0, 0.0]
        assert result["transforms"]["table"]["r"] == [0.0, 0.0, 0.0]

    def test_furniture_scene_passes_assert_scene_collision_support_relations(self):
        """End-to-end verification: the solved layout is collision-free,
        support-satisfied, and every declared relation is satisfied."""
        spec = _make_furniture_scene()
        result = solve_layout(spec)
        scene_result = assert_scene(
            spec.objects, result["transforms"], relations=spec.relations,
            checks=["collision", "support"],
        )
        assert scene_result["collision"]["count"] == 0, (
            f"solved furniture scene must be collision-free, got {scene_result['collision']!r}"
        )
        assert scene_result["support"]["ok"] is True
        assert scene_result["support"]["unsupported"] == []
        assert scene_result["relations"]["failed"] == [], (
            f"every declared relation must be satisfied after solve, failed={scene_result['relations']['failed']!r}"
        )
        assert scene_result["relations"]["satisfied"] == scene_result["relations"]["total"] == 3
        assert scene_result["pass"] is True
        _assert_plain_json_types(scene_result)

    def test_solve_layout_output_is_plain_json_safe(self):
        spec = _make_furniture_scene()
        result = solve_layout(spec)
        _assert_plain_json_types(result)
        json.dumps(result)

    def test_solve_layout_is_deterministic_across_repeated_calls(self):
        """DETERMINISM: no RNG anywhere -- identical input yields identical output."""
        spec_dict = _make_furniture_scene().to_dict()
        r1 = solve_layout(LayoutSpec.from_dict(spec_dict))
        r2 = solve_layout(LayoutSpec.from_dict(spec_dict))
        assert r1 == r2, "solve_layout must be deterministic (no RNG) -- identical spec must yield identical result"


# ===========================================================================
# Section 7 — solve_layout error taxonomy (FR-A AC-5, Major-9)
#
# CONSTRUCTION/VALIDATION errors RAISE ValueError: unknown relation type,
# a/b/target referencing a missing object id, null/non-positive bbox, a
# param outside its allowed set.
# FEASIBLE-BUT-UNSATISFIABLE constraints return solved:false +
# unsatisfied[] and DO NOT raise.
# ===========================================================================

class TestSolveLayoutErrorTaxonomy:
    """The two-sided error taxonomy: construction raises, infeasibility degrades."""

    def test_over_constrained_fixed_overlap_returns_unsatisfied_not_raise(self):
        """Two FIXED, heavily-overlapping objects with a non_overlap relation
        between them can never be resolved (neither may move) -- this is
        FEASIBLE-BUT-UNSATISFIABLE, not a construction error."""
        a = ObjectSpec(id="a", bbox=(2.0, 2.0, 2.0), fixed=True, t=(0.0, 0.0, 0.0))
        b = ObjectSpec(id="b", bbox=(2.0, 2.0, 2.0), fixed=True, t=(0.5, 0.0, 0.5))
        rel = RelationSpec(type="non_overlap", a="a", b="b")
        spec = LayoutSpec(objects=[a, b], relations=[rel])
        result = solve_layout(spec)  # must NOT raise
        assert result["solved"] is False
        assert any(u.startswith("non_overlap:") for u in result["unsatisfied"]), (
            f"expected an unsatisfied non_overlap entry, got {result['unsatisfied']!r}"
        )

    def test_unknown_relation_type_raises_at_relationspec_construction(self):
        with pytest.raises(ValueError):
            RelationSpec(type="not_a_real_relation", a="a", b="b")

    def test_relation_a_referencing_missing_object_raises(self):
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))
        rel = RelationSpec(type="non_overlap", a="ghost", b="a")
        spec = LayoutSpec(objects=[a], relations=[rel])
        with pytest.raises(ValueError):
            solve_layout(spec)

    def test_relation_b_referencing_missing_object_raises(self):
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))
        rel = RelationSpec(type="non_overlap", a="a", b="ghost")
        spec = LayoutSpec(objects=[a], relations=[rel])
        with pytest.raises(ValueError):
            solve_layout(spec)

    def test_relation_target_referencing_missing_object_raises(self):
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))
        rel = RelationSpec(type="oriented_toward", a="a", target="ghost_target")
        spec = LayoutSpec(objects=[a], relations=[rel])
        with pytest.raises(ValueError):
            solve_layout(spec)

    def test_invalid_adjacent_side_param_raises(self):
        """A param outside its allowed set (adjacent.side not in
        {'+x','-x','+z','-z'}) is a CONSTRUCTION ValueError, never an
        unsatisfied entry."""
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(5.0, 0.0, 0.0))
        with pytest.raises(ValueError):
            rel = RelationSpec(type="adjacent", a="a", b="b", params={"side": "diagonal"})
            spec = LayoutSpec(objects=[a, b], relations=[rel])
            solve_layout(spec)

    def test_invalid_aligned_axis_param_raises(self):
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(5.0, 0.0, 0.0))
        with pytest.raises(ValueError):
            rel = RelationSpec(type="aligned", a="a", b="b", params={"axis": "w"})
            spec = LayoutSpec(objects=[a, b], relations=[rel])
            solve_layout(spec)

    def test_fixed_object_outside_bounds_reports_unsatisfied_not_raise(self):
        outside = ObjectSpec(id="outside", bbox=(1.0, 1.0, 1.0), fixed=True, t=(-50.0, 0.0, 0.0))
        spec = LayoutSpec(objects=[outside], relations=[], bounds={"room": [0.0, 0.0, 10.0, 10.0]})
        result = solve_layout(spec)
        assert result["solved"] is False
        assert any(u.startswith("bounds:") and "outside" in u for u in result["unsatisfied"]), (
            f"expected a bounds:outside unsatisfied entry, got {result['unsatisfied']!r}"
        )


# ===========================================================================
# Section 8 — the pinned Y-up COORDINATE FRAME (Blocker-1)
# ===========================================================================

class TestCoordinateFrame:
    """Pins the BINDING Y-up frame: support/on_top_of moves along Y;
    footprint/bounds/adjacency/clearance operate on the X-Z plane only."""

    def test_on_top_of_places_resting_object_above_along_y(self):
        table = ObjectSpec(id="table", bbox=(2.0, 2.0, 1.0), fixed=True, t=(0.0, 0.0, 0.0))
        vase = ObjectSpec(id="vase", bbox=(0.4, 0.4, 0.6), fixed=False, t=(0.0, 10.0, 0.0))
        rel = RelationSpec(type="on_top_of", a="vase", b="table")
        spec = LayoutSpec(objects=[table, vase], relations=[rel])
        result = solve_layout(spec)
        assert result["solved"] is True
        vase_y = result["transforms"]["vase"]["t"][1]
        table_top_y = table.t[1] + table.bbox[2] / 2.0
        expected_vase_y = table_top_y + vase.bbox[2] / 2.0
        assert abs(vase_y - expected_vase_y) < 1e-3, (
            "on_top_of must position the resting object so its y_min meets "
            f"the support's y_max along +Y (Y-up frame): expected vase.t.y "
            f"~= {expected_vase_y!r}, got {vase_y!r} -- a Z-up implementation "
            "would fail this."
        )

    def test_bounds_room_is_an_x_z_footprint_with_no_vertical_bound(self):
        """A very tall object, centered inside the room footprint, must NOT
        be flagged as a bounds violation -- bounds.room has no y (height)
        component."""
        tall = ObjectSpec(id="tall", bbox=(1.0, 1.0, 100.0), fixed=True, t=(5.0, 0.0, 5.0))
        spec = LayoutSpec(objects=[tall], relations=[], bounds={"room": [0.0, 0.0, 10.0, 10.0]})
        result = solve_layout(spec)
        assert result["solved"] is True
        assert result["unsatisfied"] == [], (
            "bounds.room is an X-Z footprint rectangle with NO vertical "
            "bound -- a tall object centered inside the footprint must "
            f"never be a bounds violation, got {result['unsatisfied']!r}"
        )


# ===========================================================================
# Section 9 — assert_scene(objects, transforms, relations=[], checks=[...])
#              -> dict (FR-B, Gate-1)
#
# Locked contract: keys are EXACTLY one entry per REQUESTED check, plus
# `relations` (when relations given), plus `pass`.
# ===========================================================================

class TestAssertScene:
    """assert_scene — exact key shapes per requested checks, default behavior."""

    def _two_objects_far_apart(self):
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0), t=(0.0, 0.0, 0.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(5.0, 0.0, 5.0))
        transforms = {
            "a": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "b": {"t": [5.0, 0.0, 5.0], "r": [0.0, 0.0, 0.0]},
        }
        return [a, b], transforms

    def test_default_checks_are_collision_and_support(self):
        objects, transforms = self._two_objects_far_apart()
        result = assert_scene(objects, transforms)
        assert set(result.keys()) == {"collision", "support", "pass"}, (
            f"default checks/relations must yield exactly {{collision, support, pass}}, got {set(result.keys())!r}"
        )

    def test_no_relations_key_when_relations_empty(self):
        objects, transforms = self._two_objects_far_apart()
        result = assert_scene(objects, transforms, relations=[])
        assert "relations" not in result

    def test_relations_key_present_when_relations_given(self):
        objects, transforms = self._two_objects_far_apart()
        rel = RelationSpec(type="non_overlap", a="a", b="b")
        result = assert_scene(objects, transforms, relations=[rel])
        assert "relations" in result
        assert set(result["relations"].keys()) == {"satisfied", "total", "failed"}

    def test_only_requested_checks_appear(self):
        objects, transforms = self._two_objects_far_apart()
        result = assert_scene(objects, transforms, checks=["collision"])
        assert set(result.keys()) == {"collision", "pass"}
        assert "support" not in result
        assert "navigability" not in result
        assert "clearance" not in result

    def test_collision_shape_and_detects_overlap(self):
        a = ObjectSpec(id="a", bbox=(2.0, 2.0, 2.0), t=(0.0, 0.0, 0.0))
        b = ObjectSpec(id="b", bbox=(2.0, 2.0, 2.0), t=(0.5, 0.5, 0.5))
        transforms = {
            "a": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "b": {"t": [0.5, 0.5, 0.5], "r": [0.0, 0.0, 0.0]},
        }
        result = assert_scene([a, b], transforms, checks=["collision"])
        assert set(result["collision"].keys()) == {"pairs", "count"}
        assert result["collision"]["count"] == 1
        assert any(set(p) == {"a", "b"} for p in result["collision"]["pairs"])
        assert result["pass"] is False

    def test_support_shape_and_flags_unsupported(self):
        table = ObjectSpec(id="table", bbox=(2.0, 1.0, 2.0), t=(0.0, 0.0, 0.0))
        floating_vase = ObjectSpec(id="vase", bbox=(0.4, 0.4, 0.6), t=(0.0, 50.0, 0.0))
        transforms = {
            "table": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "vase": {"t": [0.0, 50.0, 0.0], "r": [0.0, 0.0, 0.0]},
        }
        rel = RelationSpec(type="on_top_of", a="vase", b="table")
        result = assert_scene([table, floating_vase], transforms, relations=[rel], checks=["support"])
        assert set(result["support"].keys()) == {"unsupported", "ok"}
        assert "vase" in result["support"]["unsupported"]
        assert result["support"]["ok"] is False
        assert result["pass"] is False

    def test_navigability_shape_when_requested(self):
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0), t=(0.0, 0.0, 0.0))
        transforms = {"a": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]}}
        result = assert_scene([a], transforms, checks=["navigability"])
        assert set(result.keys()) == {"navigability", "pass"}
        assert set(result["navigability"].keys()) == {"blocked_regions", "ok"}

    def test_output_is_plain_json_safe_across_all_checks(self):
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0), t=(0.0, 0.0, 0.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(5.0, 0.0, 5.0))
        transforms = {
            "a": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "b": {"t": [5.0, 0.0, 5.0], "r": [0.0, 0.0, 0.0]},
        }
        rel = RelationSpec(type="non_overlap", a="a", b="b")
        result = assert_scene(
            [a, b], transforms, relations=[rel],
            checks=["collision", "support", "navigability", "clearance"],
        )
        _assert_plain_json_types(result)
        json.dumps(result)


class TestAssertSceneClearance:
    """clearance is its OWN output key -- never silently folded into
    navigability (Major-6 / Blocker-1)."""

    def test_clearance_is_its_own_key_not_folded_into_navigability(self):
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0), t=(0.0, 0.0, 0.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(5.0, 0.0, 5.0))
        transforms = {
            "a": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "b": {"t": [5.0, 0.0, 5.0], "r": [0.0, 0.0, 0.0]},
        }
        rel = RelationSpec(type="clearance", a="a", params={"min": 0.5})
        result = assert_scene([a, b], transforms, relations=[rel], checks=["collision", "clearance"])
        assert "clearance" in result
        assert "navigability" not in result
        assert set(result["clearance"].keys()) == {"violations", "ok"}
        assert result["clearance"]["ok"] is True
        assert result["clearance"]["violations"] == []

    def test_clearance_violation_when_too_close(self):
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0), t=(0.0, 0.0, 0.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(1.05, 0.0, 0.0))
        transforms = {
            "a": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "b": {"t": [1.05, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
        }
        rel = RelationSpec(type="clearance", a="a", params={"min": 5.0})
        result = assert_scene([a, b], transforms, relations=[rel], checks=["clearance"])
        assert result["clearance"]["ok"] is False
        assert any(set(p) == {"a", "b"} for p in result["clearance"]["violations"])


# ===========================================================================
# Section 10 — RELATION SEMANTICS (Major-7): on_top_of / under / non_overlap
#              / oriented_toward / aligned direct pins (via assert_scene's
#              `relations` output, no solve_layout involved)
# ===========================================================================

class TestRelationSemanticsDirect:
    """Direct assert_scene(relations=[...]) checks for individual relation
    equations against a pre-set (already-transformed) scene."""

    def test_on_top_of_satisfied_when_geometrically_resting(self):
        table = ObjectSpec(id="table", bbox=(2.0, 2.0, 1.0), t=(0.0, 0.0, 0.0))
        vase = ObjectSpec(id="vase", bbox=(0.4, 0.4, 0.6), t=(0.0, 0.8, 0.0))
        transforms = {
            "table": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "vase": {"t": [0.0, 0.8, 0.0], "r": [0.0, 0.0, 0.0]},
        }
        rel = RelationSpec(type="on_top_of", a="vase", b="table")
        result = assert_scene([table, vase], transforms, relations=[rel], checks=["collision"])
        assert result["relations"]["satisfied"] == 1
        assert result["relations"]["failed"] == []

    def test_on_top_of_fails_when_floating(self):
        table = ObjectSpec(id="table", bbox=(2.0, 1.0, 2.0), t=(0.0, 0.0, 0.0))
        vase = ObjectSpec(id="vase", bbox=(0.4, 0.4, 0.6), t=(0.0, 5.0, 0.0))
        transforms = {
            "table": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "vase": {"t": [0.0, 5.0, 0.0], "r": [0.0, 0.0, 0.0]},
        }
        rel = RelationSpec(type="on_top_of", a="vase", b="table")
        result = assert_scene([table, vase], transforms, relations=[rel], checks=["collision"])
        assert result["relations"]["satisfied"] == 0
        assert result["relations"]["failed"] == ["on_top_of:vase->table"]

    def test_under_is_on_top_of_reversed(self):
        """under(A,B) is defined as B on_top_of A -- 'table under vase'
        means vase rests on table."""
        table = ObjectSpec(id="table", bbox=(2.0, 2.0, 1.0), t=(0.0, 0.0, 0.0))
        vase = ObjectSpec(id="vase", bbox=(0.4, 0.4, 0.6), t=(0.0, 0.8, 0.0))
        transforms = {
            "table": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "vase": {"t": [0.0, 0.8, 0.0], "r": [0.0, 0.0, 0.0]},
        }
        rel = RelationSpec(type="under", a="table", b="vase")
        result = assert_scene([table, vase], transforms, relations=[rel], checks=["collision"])
        assert result["relations"]["satisfied"] == 1
        assert result["relations"]["failed"] == []

    def test_non_overlap_satisfied_when_separated(self):
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0), t=(0.0, 0.0, 0.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(10.0, 0.0, 10.0))
        transforms = {
            "a": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "b": {"t": [10.0, 0.0, 10.0], "r": [0.0, 0.0, 0.0]},
        }
        rel = RelationSpec(type="non_overlap", a="a", b="b")
        result = assert_scene([a, b], transforms, relations=[rel], checks=["collision"])
        assert result["relations"]["satisfied"] == 1

    def test_non_overlap_fails_when_colliding(self):
        a = ObjectSpec(id="a", bbox=(2.0, 2.0, 2.0), t=(0.0, 0.0, 0.0))
        b = ObjectSpec(id="b", bbox=(2.0, 2.0, 2.0), t=(0.5, 0.5, 0.5))
        transforms = {
            "a": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "b": {"t": [0.5, 0.5, 0.5], "r": [0.0, 0.0, 0.0]},
        }
        rel = RelationSpec(type="non_overlap", a="a", b="b")
        result = assert_scene([a, b], transforms, relations=[rel], checks=["collision"])
        assert result["relations"]["satisfied"] == 0
        assert result["relations"]["failed"] == ["non_overlap:a->b"]

    def test_oriented_toward_satisfied_when_facing_target(self):
        """rug at r=(0,0,0) faces +Z; door directly in +Z -> satisfied."""
        rug = ObjectSpec(id="rug", bbox=(1.0, 1.0, 0.1), t=(0.0, 0.0, 0.0), r=(0.0, 0.0, 0.0))
        door = ObjectSpec(id="door", bbox=(0.5, 0.2, 2.0), t=(0.0, 0.0, 5.0))
        transforms = {
            "rug": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "door": {"t": [0.0, 0.0, 5.0], "r": [0.0, 0.0, 0.0]},
        }
        rel = RelationSpec(type="oriented_toward", a="rug", target="door")
        result = assert_scene([rug, door], transforms, relations=[rel], checks=["collision"])
        assert result["relations"]["satisfied"] == 1
        assert result["relations"]["failed"] == []

    def test_oriented_toward_fails_when_not_facing_target(self):
        """rug still faces +Z; door is off to -X -> ~90 degrees off, fails
        for any reasonable tol."""
        rug = ObjectSpec(id="rug", bbox=(1.0, 1.0, 0.1), t=(0.0, 0.0, 0.0), r=(0.0, 0.0, 0.0))
        door = ObjectSpec(id="door", bbox=(0.5, 0.2, 2.0), t=(-5.0, 0.0, 0.0))
        transforms = {
            "rug": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "door": {"t": [-5.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
        }
        rel = RelationSpec(type="oriented_toward", a="rug", target="door")
        result = assert_scene([rug, door], transforms, relations=[rel], checks=["collision"])
        assert result["relations"]["satisfied"] == 0
        assert result["relations"]["failed"] == ["oriented_toward:rug->door"]

    def test_aligned_center_satisfied_on_matching_axis(self):
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0), t=(2.0, 0.0, 0.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(2.0, 5.0, 9.0))
        transforms = {
            "a": {"t": [2.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "b": {"t": [2.0, 5.0, 9.0], "r": [0.0, 0.0, 0.0]},
        }
        rel = RelationSpec(type="aligned", a="a", b="b", params={"axis": "x", "edge": "center"})
        result = assert_scene([a, b], transforms, relations=[rel], checks=["collision"])
        assert result["relations"]["satisfied"] == 1

    def test_aligned_fails_on_mismatched_axis(self):
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0), t=(2.0, 0.0, 0.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(2.0, 5.0, 9.0))
        transforms = {
            "a": {"t": [2.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "b": {"t": [2.0, 5.0, 9.0], "r": [0.0, 0.0, 0.0]},
        }
        rel = RelationSpec(type="aligned", a="a", b="b", params={"axis": "z", "edge": "center"})
        result = assert_scene([a, b], transforms, relations=[rel], checks=["collision"])
        assert result["relations"]["satisfied"] == 0
        assert result["relations"]["failed"] == ["aligned:a->b"]


# ===========================================================================
# Section 11 — round-trip + determinism, over a representative input set
#              (property-style coverage; Hypothesis is not installed in
#              this venv, see module docstring)
# ===========================================================================

def _sample_specs():
    """Representative LayoutSpec configurations spanning: empty, a single
    fixed object, a two-object on_top_of pair, and the full furniture
    scene with bounds -- used as the input space for round-trip and
    determinism property checks."""
    return [
        LayoutSpec(),
        LayoutSpec(objects=[ObjectSpec(id="solo", bbox=(1.0, 1.0, 1.0), fixed=True)]),
        LayoutSpec(
            objects=[
                ObjectSpec(id="table", bbox=(2.0, 1.0, 2.0), fixed=True),
                ObjectSpec(id="vase", bbox=(0.4, 0.4, 0.6), t=(0.0, 3.0, 0.0)),
            ],
            relations=[RelationSpec(type="on_top_of", a="vase", b="table")],
        ),
        LayoutSpec(
            objects=[
                ObjectSpec(id="table", bbox=(2.0, 1.0, 1.0), fixed=True, t=(0.0, 0.0, 0.0)),
                ObjectSpec(id="vase", bbox=(0.4, 0.4, 0.6), t=(0.0, 2.0, 0.0)),
                ObjectSpec(id="lamp", bbox=(0.3, 0.3, 0.5), t=(3.0, 2.0, 3.0)),
            ],
            relations=[
                RelationSpec(type="on_top_of", a="vase", b="table"),
                RelationSpec(type="adjacent", a="lamp", b="vase", params={"side": "+x", "gap": 0.1}),
                RelationSpec(type="non_overlap", a="lamp", b="table"),
            ],
            bounds={"room": [-10.0, -10.0, 10.0, 10.0]},
            max_iters=50,
        ),
    ]


class TestRoundTripAndDeterminismProperties:
    """Parametrized property-style coverage over the round-trip and
    determinism invariants (stands in for a Hypothesis strategy)."""

    @pytest.mark.parametrize("spec", _sample_specs())
    def test_from_dict_to_dict_round_trips_exact_keys(self, spec):
        rebuilt = LayoutSpec.from_dict(spec.to_dict())
        assert rebuilt.to_dict() == spec.to_dict(), (
            "LayoutSpec.from_dict(spec.to_dict()) must round-trip to an "
            "identical to_dict() output"
        )

    @pytest.mark.parametrize("spec", _sample_specs())
    def test_solve_layout_is_deterministic_for_every_sample_spec(self, spec):
        d = spec.to_dict()
        r1 = solve_layout(LayoutSpec.from_dict(d))
        r2 = solve_layout(LayoutSpec.from_dict(d))
        assert r1 == r2, f"solve_layout must be deterministic for spec {d!r}"


# ===========================================================================
# Section 12 — Module purity (CL-015 extended): no hou/Qt/pxr/FastMCP
#              imports, no MCP call, no get_bounding_box call.
# ===========================================================================

class TestModulePurity:
    """spatial_reasoning_model.py must import NO hou, Qt/PySide6, pxr, or
    FastMCP, and must never call get_bounding_box or make an MCP call
    (CL-015 extended / plan Scope guard, Major-12)."""

    def test_source_has_no_forbidden_imports_or_calls(self):
        import fxhoudinimcp.spatial_reasoning_model as mod
        with open(mod.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        forbidden_tokens = [
            "import hou",
            "from hou",
            "import PySide6",
            "import PySide2",
            "import pxr",
            "from pxr",
            "import fastmcp",
            "from fastmcp",
            "FastMCP",
            "get_bounding_box",
            "mcp__fxhoudini",
        ]
        for token in forbidden_tokens:
            assert token not in source, (
                f"spatial_reasoning_model.py must not contain {token!r} "
                f"(CL-015 extended / PR-1 scope guard)"
            )


# ===========================================================================
# Section 13 — FIX-CYCLE RED (pp12-116a, error-taxonomy completeness).
#
# Codex cross-vendor review found a real Major: the BINDING error-taxonomy
# split in the plan's lockedFieldContract -- "CONSTRUCTION/VALIDATION
# errors RAISE ValueError naming the bad value -- unknown relation type,
# an a/b/target referencing a missing object id, a null/non-positive
# bbox, a param outside its allowed set" -- leaks on TWO classes of
# malformed input the existing suite never pinned:
#
#   (1) an OMITTED required relation ref (b/target left at its "" default
#       for a binary/target-requiring relation type) is a construction
#       error exactly like a GHOST ref (b="ghost") -- but currently the
#       impl's ref-presence check (`if ref and ref not in by_id`) treats
#       an empty string as "no ref to check" and skips it, so the
#       omission is never caught at validation time; it instead surfaces
#       as a raw KeyError('') deep in the relaxation loop.
#   (2) assert_scene's `relations=[...]` argument is NEVER run through
#       the same construction/validation gate solve_layout runs
#       (`_validate_relation`) -- an out-of-set param (e.g.
#       aligned.axis="bad") or a missing required param (clearance with
#       no params["min"]) reaches the relation-satisfaction predicates
#       directly and raises a raw KeyError instead of the contract's
#       ValueError.
#
# Both are the SAME taxonomy clause; a green fix must close them
# uniformly (both solve_layout AND assert_scene) rather than patching
# one call site.
# ===========================================================================

class TestErrorTaxonomyCompleteness:
    """Pins the error-taxonomy split against the two malformed-input
    classes above: an omitted required ref is a construction ValueError
    (never a KeyError), and assert_scene's relations argument is
    validated exactly like solve_layout's."""

    # -- (1) omitted required ref -> ValueError, not KeyError ------------

    def test_adjacent_missing_b_raises_valueerror_not_keyerror(self):
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(5.0, 0.0, 0.0))
        rel = RelationSpec(type="adjacent", a="a")  # b omitted (defaults to "")
        spec = LayoutSpec(objects=[a, b], relations=[rel])
        with pytest.raises(ValueError):
            solve_layout(spec)

    def test_on_top_of_missing_b_raises_valueerror_not_keyerror(self):
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(5.0, 0.0, 0.0))
        rel = RelationSpec(type="on_top_of", a="a")  # b omitted
        spec = LayoutSpec(objects=[a, b], relations=[rel])
        with pytest.raises(ValueError):
            solve_layout(spec)

    def test_non_overlap_missing_b_raises_valueerror_not_keyerror(self):
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(5.0, 0.0, 0.0))
        rel = RelationSpec(type="non_overlap", a="a")  # b omitted
        spec = LayoutSpec(objects=[a, b], relations=[rel])
        with pytest.raises(ValueError):
            solve_layout(spec)

    def test_aligned_missing_b_raises_valueerror_not_keyerror(self):
        """b omitted with an otherwise-valid axis param -- isolates the
        missing-ref gap from the (already-pinned) axis-validation gap."""
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(5.0, 0.0, 0.0))
        rel = RelationSpec(type="aligned", a="a", params={"axis": "x"})  # b omitted
        spec = LayoutSpec(objects=[a, b], relations=[rel])
        with pytest.raises(ValueError):
            solve_layout(spec)

    def test_oriented_toward_missing_target_raises_valueerror_not_keyerror(self):
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(5.0, 0.0, 0.0))
        rel = RelationSpec(type="oriented_toward", a="a")  # target omitted
        spec = LayoutSpec(objects=[a, b], relations=[rel])
        with pytest.raises(ValueError):
            solve_layout(spec)

    # -- sanity: do NOT over-constrain -- these must stay GREEN ----------

    def test_clearance_valid_unary_does_not_raise(self):
        """clearance is unary (b='' is correct, not an omission) -- a
        valid clearance relation with its required min param present
        must solve cleanly, never raise."""
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(5.0, 0.0, 0.0))
        rel = RelationSpec(type="clearance", a="a", params={"min": 1.0})
        spec = LayoutSpec(objects=[a, b], relations=[rel])
        result = solve_layout(spec)  # must NOT raise
        assert result["solved"] is True

    def test_ghost_ref_still_raises_valueerror(self):
        """Contrast case: a genuinely-PRESENT ghost ref (b='ghost', not
        omitted) already raised ValueError before this fix cycle --
        confirms the fix does not regress the existing ghost-ref path."""
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))
        rel = RelationSpec(type="non_overlap", a="a", b="ghost")
        spec = LayoutSpec(objects=[a], relations=[rel])
        with pytest.raises(ValueError):
            solve_layout(spec)

    # -- (2) out-of-set / missing param -> ValueError on BOTH paths ------

    def test_aligned_axis_bad_raises_valueerror_via_solve_layout(self):
        """Already correct pre-fix (aligned.axis is validated by
        _validate_relation) -- pinned here alongside its assert_scene
        counterpart so the two paths are asserted side by side."""
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(5.0, 0.0, 0.0))
        rel = RelationSpec(type="aligned", a="a", b="b", params={"axis": "bad"})
        spec = LayoutSpec(objects=[a, b], relations=[rel])
        with pytest.raises(ValueError):
            solve_layout(spec)

    def test_aligned_axis_bad_raises_valueerror_via_assert_scene(self):
        """The real gap: assert_scene's relations=[...] argument bypasses
        _validate_relation entirely, so an out-of-set aligned.axis
        currently raises a raw KeyError instead of ValueError."""
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(5.0, 0.0, 0.0))
        rel = RelationSpec(type="aligned", a="a", b="b", params={"axis": "bad"})
        transforms = {
            "a": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "b": {"t": [5.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
        }
        with pytest.raises(ValueError):
            assert_scene([a, b], transforms, relations=[rel], checks=["collision"])

    def test_aligned_edge_bad_raises_valueerror_via_solve_layout(self):
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(5.0, 0.0, 0.0))
        rel = RelationSpec(type="aligned", a="a", b="b", params={"axis": "x", "edge": "bad"})
        spec = LayoutSpec(objects=[a, b], relations=[rel])
        with pytest.raises(ValueError):
            solve_layout(spec)

    def test_clearance_missing_min_raises_valueerror_via_assert_scene(self):
        """Sonnet flag #1: clearance requires params['min']; passed
        directly to assert_scene (bypassing solve_layout's
        _validate_relation gate), the missing key currently raises a raw
        KeyError instead of the contract's ValueError."""
        a = ObjectSpec(id="a", bbox=(1.0, 1.0, 1.0))
        b = ObjectSpec(id="b", bbox=(1.0, 1.0, 1.0), t=(5.0, 0.0, 0.0))
        rel = RelationSpec(type="clearance", a="a", params={})  # min missing
        transforms = {
            "a": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "b": {"t": [5.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
        }
        with pytest.raises(ValueError):
            assert_scene([a, b], transforms, relations=[rel], checks=["collision"])
