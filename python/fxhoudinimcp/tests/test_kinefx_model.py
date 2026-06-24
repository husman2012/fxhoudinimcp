"""
Tests for kinefx_model.py — pure-logic layer.

No hou / Qt / pxr imports anywhere in this file.  Runs under plain pytest
headless (off-DCC, no Houdini install required).

Covers the public contract of:
  - Joint, Skeleton, MotionClip, RetargetMap, ApexNodeSummary,
    ApexGraphSummary, KinefxRequest, OpResult   (dataclasses)
  - derive_parents(edges, names)                (edge-list → parent mapping)
  - pack_trs(matrix4)                           (4×4 nested list → TRS tuple)
  - skeleton_to_json(skeleton)                  (Skeleton → JSON-able dict)
  - validate_mapping(retarget_map, src, tgt)    (RetargetMap validity check)

Test strategy: example-based assertions on the public observable behavior.
Property tests (Hypothesis) are not added in this RED pass to keep the
dependency surface minimal; they are a hou-test follow-up if needed.

TDD phase: RED — this file is authored BEFORE kinefx_model.py exists.
All tests should fail with ImportError / ModuleNotFoundError on first run.
"""

from __future__ import annotations

import json
import math
import sys
import os

# ---------------------------------------------------------------------------
# Path bootstrap — allow running as a standalone script as well as via pytest.
# This mirrors the pattern in wedge_model tests.
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_PKG_ROOT))

import pytest

# ---------------------------------------------------------------------------
# The module under test — does NOT exist yet; all tests below will be RED.
# ---------------------------------------------------------------------------
from fxhoudinimcp.kinefx_model import (
    Joint,
    Skeleton,
    MotionClip,
    RetargetMap,
    ApexNodeSummary,
    ApexGraphSummary,
    KinefxRequest,
    OpResult,
    derive_parents,
    pack_trs,
    skeleton_to_json,
    validate_mapping,
    unmapped_target_joints,  # pp12-110e — RED: does not exist yet
)


# ===========================================================================
# Helpers
# ===========================================================================

def _identity_4x4() -> list[list[float]]:
    """4x4 identity matrix as nested Python lists (as hou.Matrix4 would export)."""
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _translation_4x4(tx: float, ty: float, tz: float) -> list[list[float]]:
    """4x4 matrix encoding a pure translation, as nested Python lists."""
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [tx,  ty,  tz,  1.0],
    ]


def _trs_dict(t=(0.0, 0.0, 0.0), r=(0.0, 0.0, 0.0, 1.0), s=(1.0, 1.0, 1.0)) -> dict:
    """Build a TRS dict in the §7.3 JSON shape."""
    return {"t": list(t), "r": list(r), "s": list(s)}


# ===========================================================================
# Section 1 — derive_parents(edges, names)
#
# Public contract (spec §7.4 + FR-11):
#   edges  : list of (parent_name, child_name) string pairs extracted from the
#             KineFX skeleton's bone/point connectivity
#   names  : ordered list of all joint names in the skeleton
#   returns: dict[str, str | None]  — maps every joint name to its parent name;
#             root joints (no parent) map to None.
# ===========================================================================

class TestDeriveParents:
    """derive_parents — edge-list → parent mapping."""

    def test_single_root_has_none_parent(self):
        """A joint that appears in names but not as a child in any edge is a root."""
        edges = []
        names = ["Hips"]
        parents = derive_parents(edges, names)
        assert parents["Hips"] is None

    def test_linear_chain(self):
        """Hips → Spine → Neck → Head: each joint gets the correct parent."""
        edges = [("Hips", "Spine"), ("Spine", "Neck"), ("Neck", "Head")]
        names = ["Hips", "Spine", "Neck", "Head"]
        parents = derive_parents(edges, names)
        assert parents["Hips"] is None, "Hips is the root"
        assert parents["Spine"] == "Hips"
        assert parents["Neck"] == "Spine"
        assert parents["Head"] == "Neck"

    def test_fork_gives_both_children_the_same_parent(self):
        """Hips → {LeftUpLeg, RightUpLeg}: both children must map to Hips."""
        edges = [("Hips", "LeftUpLeg"), ("Hips", "RightUpLeg")]
        names = ["Hips", "LeftUpLeg", "RightUpLeg"]
        parents = derive_parents(edges, names)
        assert parents["Hips"] is None
        assert parents["LeftUpLeg"] == "Hips"
        assert parents["RightUpLeg"] == "Hips"

    def test_all_names_present_in_result(self):
        """Every name in the names list must appear as a key in the output dict."""
        edges = [("root", "child_a"), ("child_a", "leaf")]
        names = ["root", "child_a", "leaf"]
        parents = derive_parents(edges, names)
        assert set(parents.keys()) == set(names)

    def test_multi_root_skeleton(self):
        """Two disconnected chains each have their own root (parent=None)."""
        # spine chain and prop chain — disconnected
        edges = [("Hips", "Spine"), ("PropRoot", "PropTip")]
        names = ["Hips", "Spine", "PropRoot", "PropTip"]
        parents = derive_parents(edges, names)
        assert parents["Hips"] is None
        assert parents["Spine"] == "Hips"
        assert parents["PropRoot"] is None
        assert parents["PropTip"] == "PropRoot"

    def test_edge_order_does_not_affect_result(self):
        """Shuffling the edge list should not change the parent mapping."""
        edges_ordered = [("Hips", "Spine"), ("Spine", "Head")]
        edges_reversed = [("Spine", "Head"), ("Hips", "Spine")]
        names = ["Hips", "Spine", "Head"]
        p_ordered = derive_parents(edges_ordered, names)
        p_reversed = derive_parents(edges_reversed, names)
        assert p_ordered == p_reversed


# ===========================================================================
# Section 2 — pack_trs(matrix4)
#
# Public contract (spec §7.4 + FR-11):
#   matrix4: a 4×4 affine matrix as nested Python lists — the shape produced
#             when hou.Matrix4 values are converted to plain Python before
#             passing to the pure layer.  NOT a hou.Matrix4 object.
#   returns: tuple of (translate, rotate, scale)
#             translate: (tx, ty, tz)      floats
#             rotate:    (rx, ry, rz, rw) or (rx, ry, rz)  floats
#             scale:     (sx, sy, sz)      floats
# ===========================================================================

class TestPackTrs:
    """pack_trs — 4×4 nested-list matrix → (translate, rotate, scale) tuple."""

    def test_identity_gives_zero_translate(self):
        """Identity matrix → translation is (0, 0, 0)."""
        t, r, s = pack_trs(_identity_4x4())
        assert pytest.approx(list(t), abs=1e-6) == [0.0, 0.0, 0.0]

    def test_identity_gives_unit_scale(self):
        """Identity matrix → scale is (1, 1, 1)."""
        t, r, s = pack_trs(_identity_4x4())
        assert pytest.approx(list(s), abs=1e-6) == [1.0, 1.0, 1.0]

    def test_pure_translation_extracts_correctly(self):
        """A matrix with only a translation sets tx/ty/tz and keeps s=(1,1,1)."""
        m = _translation_4x4(3.0, 7.0, -2.5)
        t, r, s = pack_trs(m)
        assert pytest.approx(t[0], abs=1e-5) == 3.0
        assert pytest.approx(t[1], abs=1e-5) == 7.0
        assert pytest.approx(t[2], abs=1e-5) == -2.5
        assert pytest.approx(list(s), abs=1e-5) == [1.0, 1.0, 1.0]

    def test_returns_three_components(self):
        """pack_trs must return a 3-tuple: (translate, rotate, scale)."""
        result = pack_trs(_identity_4x4())
        assert len(result) == 3

    def test_translate_component_has_three_elements(self):
        """Translate component must have exactly 3 elements (x, y, z)."""
        t, r, s = pack_trs(_identity_4x4())
        assert len(t) == 3

    def test_scale_component_has_three_elements(self):
        """Scale component must have exactly 3 elements (sx, sy, sz)."""
        t, r, s = pack_trs(_identity_4x4())
        assert len(s) == 3

    def test_rotate_component_has_three_or_four_elements(self):
        """Rotate is either Euler (3) or quaternion (4) — both are valid."""
        t, r, s = pack_trs(_identity_4x4())
        assert len(r) in (3, 4)

    def test_input_is_plain_python_not_hou_matrix(self):
        """pack_trs accepts nested Python lists, confirming no hou import is needed."""
        # If this test can import and call pack_trs without hou, the pure
        # contract holds.  The test body itself is the evidence.
        m = _identity_4x4()
        # Must not raise
        t, r, s = pack_trs(m)
        assert t is not None


# ===========================================================================
# Section 3 — Joint dataclass
#
# Public contract (spec §7.3 + §7.4):
#   Joint(name, parent, rest, anim=None)
#     name   : str
#     parent : str | None   (None for roots)
#     rest   : dict         TRS dict {"t":[...], "r":[...], "s":[...]}
#     anim   : dict | None  animated TRS, present only when a frame was requested
#
#   .to_dict() / asdict():   produces the §7.3 JSON joint shape;
#                            "anim" key is omitted when anim is None.
# ===========================================================================

class TestJointDataclass:
    """Joint dataclass — fields + serialization."""

    def _make_joint(self, **kwargs):
        defaults = dict(
            name="Hips",
            parent=None,
            rest=_trs_dict(t=(0.0, 1.02, 0.0)),
        )
        defaults.update(kwargs)
        return Joint(**defaults)

    def test_root_joint_has_none_parent(self):
        """A root joint's parent field is None."""
        j = self._make_joint(name="Hips", parent=None)
        assert j.parent is None

    def test_child_joint_stores_parent_name(self):
        """A non-root joint stores its parent's name as a string."""
        j = self._make_joint(name="Spine", parent="Hips")
        assert j.parent == "Hips"

    def test_to_dict_contains_name_parent_rest(self):
        """Serialized joint includes name, parent, and rest keys."""
        j = self._make_joint()
        d = j.to_dict()
        assert "name" in d
        assert "parent" in d
        assert "rest" in d

    def test_to_dict_omits_anim_when_none(self):
        """'anim' key must be absent from the dict when anim=None."""
        j = self._make_joint(anim=None)
        d = j.to_dict()
        assert "anim" not in d

    def test_to_dict_includes_anim_when_present(self):
        """'anim' key must be present in the dict when an animated TRS is set."""
        anim_trs = _trs_dict(t=(0.1, 1.05, 0.02))
        j = self._make_joint(anim=anim_trs)
        d = j.to_dict()
        assert "anim" in d

    def test_to_dict_rest_has_t_r_s_keys(self):
        """rest dict must contain 't', 'r', and 's' keys per §7.3."""
        j = self._make_joint(rest=_trs_dict(t=(0.0, 1.02, 0.0)))
        d = j.to_dict()
        assert set(d["rest"].keys()) >= {"t", "r", "s"}

    def test_to_dict_is_json_serializable(self):
        """The dict produced by to_dict must be json.dumps-able without error."""
        j = self._make_joint()
        d = j.to_dict()
        # Must not raise
        json.dumps(d)

    def test_root_joint_parent_is_null_in_json(self):
        """Root joint's parent serializes to JSON null (i.e. Python None)."""
        j = self._make_joint(name="Hips", parent=None)
        d = j.to_dict()
        raw = json.loads(json.dumps(d))
        assert raw["parent"] is None


# ===========================================================================
# Section 4 — Skeleton dataclass + skeleton_to_json
#
# Public contract (spec §7.3 + §7.4):
#   Skeleton(joints: list[Joint])
#     .to_dict()  → {"count": N, "joints": [...]}
#
#   skeleton_to_json(skeleton: Skeleton) → dict  (same shape as to_dict)
#     Also acceptable: skeleton_to_json returns a JSON string (both are tested).
# ===========================================================================

class TestSkeletonDataclass:
    """Skeleton dataclass — construction, joint count, round-trip."""

    def _make_skeleton(self, n: int = 3) -> Skeleton:
        joints = [
            Joint(
                name=f"Joint{i}",
                parent=f"Joint{i-1}" if i > 0 else None,
                rest=_trs_dict(t=(float(i), 0.0, 0.0)),
            )
            for i in range(n)
        ]
        return Skeleton(joints=joints)

    def test_skeleton_stores_correct_joint_count(self):
        """Skeleton.joint_count (or len(joints)) returns the number of joints added."""
        skel = self._make_skeleton(n=5)
        count = getattr(skel, "joint_count", None) or len(skel.joints)
        assert count == 5

    def test_to_dict_count_matches_joints(self):
        """to_dict()['count'] matches the number of joints in the skeleton."""
        skel = self._make_skeleton(n=7)
        d = skel.to_dict()
        assert d["count"] == 7

    def test_to_dict_joints_list_length(self):
        """to_dict()['joints'] is a list of the same length as the joints."""
        n = 4
        skel = self._make_skeleton(n=n)
        d = skel.to_dict()
        assert len(d["joints"]) == n

    def test_round_trip_preserves_joint_names(self):
        """from_dict(to_dict()) restores all joint names in the same order."""
        skel = self._make_skeleton(n=3)
        d = skel.to_dict()
        skel2 = Skeleton.from_dict(d)
        names_orig = [j.name for j in skel.joints]
        names_rt   = [j.name for j in skel2.joints]
        assert names_orig == names_rt

    def test_round_trip_preserves_parent_links(self):
        """from_dict(to_dict()) restores parent relationships correctly."""
        skel = self._make_skeleton(n=3)
        d = skel.to_dict()
        skel2 = Skeleton.from_dict(d)
        parents_orig = [j.parent for j in skel.joints]
        parents_rt   = [j.parent for j in skel2.joints]
        assert parents_orig == parents_rt

    def test_to_dict_is_json_serializable(self):
        """The skeleton dict must be json.dumps-able without error."""
        skel = self._make_skeleton(n=2)
        json.dumps(skel.to_dict())


class TestSkeletonToJson:
    """skeleton_to_json helper — matches §7.3 JSON shape."""

    def _make_simple_skeleton(self) -> Skeleton:
        joints = [
            Joint(name="Hips",  parent=None,    rest=_trs_dict(t=(0.0, 1.02, 0.0))),
            Joint(name="Spine", parent="Hips",  rest=_trs_dict(t=(0.0, 1.22, 0.0))),
        ]
        return Skeleton(joints=joints)

    def test_result_contains_count_and_joints_keys(self):
        """skeleton_to_json output must have 'count' and 'joints' top-level keys."""
        skel = self._make_simple_skeleton()
        result = skeleton_to_json(skel)
        # Accept either a dict or a JSON string
        if isinstance(result, str):
            result = json.loads(result)
        assert "count" in result
        assert "joints" in result

    def test_count_equals_joint_list_length(self):
        """'count' in the output equals len('joints')."""
        skel = self._make_simple_skeleton()
        result = skeleton_to_json(skel)
        if isinstance(result, str):
            result = json.loads(result)
        assert result["count"] == len(result["joints"])

    def test_first_joint_is_root(self):
        """The first joint's parent should be null (Hips is the root)."""
        skel = self._make_simple_skeleton()
        result = skeleton_to_json(skel)
        if isinstance(result, str):
            result = json.loads(result)
        assert result["joints"][0]["parent"] is None

    def test_second_joint_has_parent_hips(self):
        """Spine's parent is 'Hips' in the serialized output."""
        skel = self._make_simple_skeleton()
        result = skeleton_to_json(skel)
        if isinstance(result, str):
            result = json.loads(result)
        spine = next(j for j in result["joints"] if j["name"] == "Spine")
        assert spine["parent"] == "Hips"

    def test_rest_trs_shape(self):
        """Each joint's rest dict contains 't', 'r', and 's' keys."""
        skel = self._make_simple_skeleton()
        result = skeleton_to_json(skel)
        if isinstance(result, str):
            result = json.loads(result)
        for j in result["joints"]:
            assert "t" in j["rest"]
            assert "r" in j["rest"]
            assert "s" in j["rest"]

    def test_t_is_three_element_list(self):
        """The 't' component of rest must be a 3-element list."""
        skel = self._make_simple_skeleton()
        result = skeleton_to_json(skel)
        if isinstance(result, str):
            result = json.loads(result)
        for j in result["joints"]:
            assert len(j["rest"]["t"]) == 3

    def test_output_is_json_serializable(self):
        """skeleton_to_json output must be json.dumps-able (if dict) or valid JSON (if str)."""
        skel = self._make_simple_skeleton()
        result = skeleton_to_json(skel)
        if isinstance(result, dict):
            json.dumps(result)
        else:
            json.loads(result)  # must parse without error


# ===========================================================================
# Section 5 — MotionClip dataclass
#
# Public contract (spec §7.4 + §9.1 "sample-don't-dump"):
#   MotionClip(joint_count, frame_range, frames=None)
#     joint_count  : int
#     frame_range  : tuple[int, int]   (start, end)
#     frames       : list[...] | None  per-frame TRS data; may be None/empty
#
#   .to_dict() always includes joint_count + frame_range.
#   .to_dict() includes per-frame TRS only when frames is not None / non-empty.
#
#   This is the "sample-don't-dump" discipline — mirrors fxhoudinimcp's
#   geometry summarizers that avoid dumping every point by default.
# ===========================================================================

class TestMotionClip:
    """MotionClip — sample-don't-dump serialization contract."""

    def test_frame_range_always_serialized(self):
        """to_dict() must always include 'frame_range' regardless of frames."""
        clip = MotionClip(joint_count=64, frame_range=(1, 90))
        d = clip.to_dict()
        assert "frame_range" in d

    def test_joint_count_always_serialized(self):
        """to_dict() must always include 'joint_count' or 'count'."""
        clip = MotionClip(joint_count=64, frame_range=(1, 90))
        d = clip.to_dict()
        has_count = "joint_count" in d or "count" in d
        assert has_count

    def test_per_frame_trs_absent_when_frames_none(self):
        """When no explicit frame data is provided, per-frame TRS is NOT in the dict."""
        clip = MotionClip(joint_count=64, frame_range=(1, 90), frames=None)
        d = clip.to_dict()
        # Neither "frames" nor "per_frame" should appear at the top level
        has_frame_data = "frames" in d or "per_frame" in d
        assert not has_frame_data, (
            "MotionClip.to_dict() must omit per-frame TRS by default "
            "(sample-don't-dump discipline)"
        )

    def test_per_frame_trs_present_when_frames_provided(self):
        """When frames data is explicitly provided, to_dict() includes it."""
        frame_data = [{"Hips": {"t": [0, 1, 0], "r": [0, 0, 0, 1], "s": [1, 1, 1]}}]
        clip = MotionClip(joint_count=1, frame_range=(1, 1), frames=frame_data)
        d = clip.to_dict()
        has_frame_data = "frames" in d or "per_frame" in d
        assert has_frame_data, (
            "MotionClip.to_dict() must include per-frame TRS when frames is explicitly set"
        )

    def test_frame_range_contains_start_and_end(self):
        """frame_range serializes as a 2-element structure (start, end)."""
        clip = MotionClip(joint_count=10, frame_range=(5, 45))
        d = clip.to_dict()
        fr = d["frame_range"]
        # Accept list, tuple, or dict with start/end keys
        if isinstance(fr, (list, tuple)):
            assert len(fr) == 2
            assert fr[0] == 5
            assert fr[1] == 45
        else:
            assert fr.get("start") == 5 or fr.get(0) == 5

    def test_to_dict_is_json_serializable(self):
        """to_dict() output must be json.dumps-able."""
        clip = MotionClip(joint_count=30, frame_range=(1, 60))
        json.dumps(clip.to_dict())


# ===========================================================================
# Section 6 — RetargetMap + validate_mapping
#
# Public contract (spec §7.3 + §7.4 + FR-6):
#   RetargetMap(pairs: list[tuple[str, str]])
#     pairs: list of (source_joint, target_joint) string pairs
#
#   validate_mapping(retarget_map, source_skeleton, target_skeleton) -> list[str]
#     Returns a list of error / warning strings.
#     Empty list → valid mapping.
#     Non-empty list → one string per problem found.
#
#   Validated behaviors:
#     - A source joint name not in source_skeleton → error entry
#     - A target joint name not in target_skeleton → error entry
#     - A fully valid mapping → empty list
# ===========================================================================

class TestRetargetMap:
    """RetargetMap dataclass — construction and basic field access."""

    def test_retarget_map_stores_pairs(self):
        """RetargetMap stores the source→target joint pairs."""
        pairs = [("Hips", "root"), ("Spine", "spine_01")]
        rm = RetargetMap(pairs=pairs)
        assert rm.pairs == pairs

    def test_empty_pairs_is_valid(self):
        """A RetargetMap with no pairs is constructible (degenerate case)."""
        rm = RetargetMap(pairs=[])
        assert rm.pairs == []


class TestValidateMapping:
    """validate_mapping — rejects absent joints, accepts valid mappings."""

    def _make_skeleton_names(self, *names: str) -> Skeleton:
        """Create a minimal Skeleton with joints for the given names."""
        joints = [Joint(name=n, parent=None, rest=_trs_dict()) for n in names]
        return Skeleton(joints=joints)

    def test_valid_mapping_returns_empty_list(self):
        """A mapping where all source and target names exist returns no errors."""
        src = self._make_skeleton_names("Hips", "Spine", "LeftUpLeg")
        tgt = self._make_skeleton_names("root", "spine_01", "thigh_l")
        rm = RetargetMap(pairs=[
            ("Hips", "root"),
            ("Spine", "spine_01"),
            ("LeftUpLeg", "thigh_l"),
        ])
        errors = validate_mapping(rm, src, tgt)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_absent_source_joint_is_flagged(self):
        """If a source joint in a pair doesn't exist in source_skeleton, report it."""
        src = self._make_skeleton_names("Hips")         # "Ghost" is absent
        tgt = self._make_skeleton_names("root")
        rm = RetargetMap(pairs=[("Ghost", "root")])
        errors = validate_mapping(rm, src, tgt)
        assert len(errors) >= 1
        # At least one error must mention "Ghost" or "source"
        combined = " ".join(errors).lower()
        assert "ghost" in combined or "source" in combined

    def test_absent_target_joint_is_flagged(self):
        """If a target joint in a pair doesn't exist in target_skeleton, report it."""
        src = self._make_skeleton_names("Hips")
        tgt = self._make_skeleton_names("root")        # "NoSuchJoint" is absent
        rm = RetargetMap(pairs=[("Hips", "NoSuchJoint")])
        errors = validate_mapping(rm, src, tgt)
        assert len(errors) >= 1
        combined = " ".join(errors).lower()
        assert "nosuchjoint" in combined or "target" in combined

    def test_empty_mapping_against_any_skeleton_is_valid(self):
        """An empty RetargetMap validates against any skeleton without errors."""
        src = self._make_skeleton_names("Hips", "Spine")
        tgt = self._make_skeleton_names("root", "spine_01")
        rm = RetargetMap(pairs=[])
        errors = validate_mapping(rm, src, tgt)
        assert errors == []

    def test_multiple_errors_reported_for_multiple_bad_pairs(self):
        """Each invalid pair contributes at least one error string."""
        src = self._make_skeleton_names("Hips")
        tgt = self._make_skeleton_names("root")
        rm = RetargetMap(pairs=[
            ("BadSource1", "root"),
            ("BadSource2", "root"),
        ])
        errors = validate_mapping(rm, src, tgt)
        assert len(errors) >= 2

    def test_returns_list(self):
        """validate_mapping always returns a list (never None or a string)."""
        src = self._make_skeleton_names("Hips")
        tgt = self._make_skeleton_names("root")
        rm = RetargetMap(pairs=[("Hips", "root")])
        result = validate_mapping(rm, src, tgt)
        assert isinstance(result, list)


# ===========================================================================
# Section 7 — Remaining dataclasses: ApexNodeSummary, ApexGraphSummary,
#             KinefxRequest, OpResult
#
# Contract: they are constructible, carry their fields, and serialize
# to a dict that is json.dumps-able.  The full Apex serialization logic
# is tested via the real `inspect_apex` integration (hython-smoke); here
# we only verify the pure-layer contract.
# ===========================================================================

class TestApexNodeSummary:
    """ApexNodeSummary — construction and serialization."""

    def test_apex_node_summary_constructible(self):
        """ApexNodeSummary is constructible with name, type, and ports."""
        node = ApexNodeSummary(name="build_rig", node_type="apex::autorigcomponent",
                               ports=["in_skel", "out_rig"])
        assert node.name == "build_rig"

    def test_to_dict_contains_name_and_type(self):
        """to_dict() includes at least 'name' and 'type' (or 'node_type') keys."""
        node = ApexNodeSummary(name="build_rig", node_type="apex::autorigcomponent",
                               ports=["in_skel", "out_rig"])
        d = node.to_dict()
        has_type = "type" in d or "node_type" in d
        assert "name" in d
        assert has_type

    def test_to_dict_is_json_serializable(self):
        node = ApexNodeSummary(name="x", node_type="apex::x", ports=[])
        json.dumps(node.to_dict())


class TestApexGraphSummary:
    """ApexGraphSummary — construction and serialization."""

    def test_apex_graph_summary_constructible(self):
        """ApexGraphSummary is constructible with nodes, wires, and control_count."""
        graph = ApexGraphSummary(
            nodes=[ApexNodeSummary(name="n1", node_type="t1", ports=[])],
            wires=[("n1.out", "n2.in")],
            control_count=3,
        )
        assert graph.control_count == 3

    def test_to_dict_has_nodes_wires_control_count(self):
        """to_dict() contains 'nodes', 'wires', and 'control_count'."""
        graph = ApexGraphSummary(nodes=[], wires=[], control_count=0)
        d = graph.to_dict()
        assert "nodes" in d
        assert "wires" in d
        has_count = "control_count" in d or "controls" in d
        assert has_count

    def test_to_dict_is_json_serializable(self):
        graph = ApexGraphSummary(nodes=[], wires=[], control_count=0)
        json.dumps(graph.to_dict())


class TestKinefxRequest:
    """KinefxRequest — construction and field access."""

    def test_kinefx_request_constructible(self):
        """KinefxRequest is constructible with a tool name and a params dict."""
        req = KinefxRequest(tool="houdini_query_skeleton",
                            params={"node": "/obj/geo1", "frame": 1})
        assert req.tool == "houdini_query_skeleton"

    def test_params_accessible(self):
        req = KinefxRequest(tool="houdini_kinefx_probe", params={})
        assert req.params == {}


class TestOpResult:
    """OpResult — construction and serialization."""

    def test_success_result(self):
        """OpResult can represent a successful operation."""
        result = OpResult(ok=True, data={"node": "/obj/geo1/skel"})
        assert result.ok is True

    def test_failure_result(self):
        """OpResult can represent a failed operation with an error message."""
        result = OpResult(ok=False, error="Node not found: /obj/missing")
        assert result.ok is False

    def test_to_dict_contains_ok_key(self):
        """to_dict() must contain an 'ok' key."""
        result = OpResult(ok=True)
        d = result.to_dict()
        assert "ok" in d

    def test_to_dict_is_json_serializable(self):
        result = OpResult(ok=True, data={"count": 42})
        json.dumps(result.to_dict())


# ===========================================================================
# Section 8 — hou-free import verification (CL-015 + FR-11)
#
# kinefx_model.py must import with zero hou/Qt/pxr at module top-level.
# This test proves it by confirming the module loads under plain pytest
# (no Houdini environment).
# ===========================================================================

class TestHouFreeImport:
    """Confirm kinefx_model.py carries no hou/Qt/pxr dependency."""

    def test_module_importable_without_hou(self):
        """kinefx_model must load under plain Python with no hou installed."""
        # If we reached this class, the import at the top of the file succeeded.
        # That is sufficient proof — no hou/Qt/pxr was required to import.
        import fxhoudinimcp.kinefx_model as km
        assert km is not None

    def test_hou_not_in_kinefx_model_imports(self):
        """kinefx_model module must not reference 'hou' as a top-level import."""
        import fxhoudinimcp.kinefx_model as km
        import inspect
        source = inspect.getsource(km)
        # Strip comments before checking (# import hou is fine as a comment)
        import re
        source_no_comments = re.sub(r"#.*", "", source)
        assert "import hou" not in source_no_comments, (
            "kinefx_model.py must not import hou (CL-015 / FR-11)"
        )


# ===========================================================================
# Section 9 — Edge-case hardening (F1/F2/F5 fix-pass additions)
#
# These tests were added in the diff-scoped hardening pass (pp12-110a fix-pass)
# to close tier-3 review flags:
#   F1  — derive_parents cycle detection (raises ValueError)
#   F2  — derive_parents double-parent detection (raises ValueError)
#   F5  — validate_mapping duplicate-pair + one-to-many detection
#
# Each test exercises one flag's contract exactly.
# The existing 63 tests above are not modified.
# ===========================================================================

class TestDeriveParentsEdgeCases:
    """Edge-case hardening for derive_parents — F1 (cycle) and F2 (double-parent)."""

    def test_cycle_raises_value_error(self):
        """A->B->A cycle must raise ValueError mentioning 'cycle'."""
        # A is parent of B, B is parent of A — a two-node cycle.
        edges = [("A", "B"), ("B", "A")]
        names = ["A", "B"]
        with pytest.raises(ValueError, match="cycle"):
            derive_parents(edges, names)

    def test_double_parent_raises_value_error(self):
        """A joint listed as a child in two edges must raise ValueError."""
        # Both ("Root", "Child") and ("Other", "Child") claim Child as a child.
        edges = [("Root", "Child"), ("Other", "Child")]
        names = ["Root", "Other", "Child"]
        with pytest.raises(ValueError, match="multiple parents"):
            derive_parents(edges, names)


class TestValidateMappingEdgeCases:
    """Edge-case hardening for validate_mapping — F5 (duplicates + one-to-many)."""

    def _make_skeleton_names(self, *names: str) -> Skeleton:
        joints = [Joint(name=n, parent=None, rest=_trs_dict()) for n in names]
        return Skeleton(joints=joints)

    def test_duplicate_pair_is_flagged(self):
        """The same (src, tgt) pair appearing twice must produce an error entry."""
        src = self._make_skeleton_names("Hips", "Spine")
        tgt = self._make_skeleton_names("root", "spine_01")
        rm = RetargetMap(pairs=[
            ("Hips", "root"),
            ("Hips", "root"),  # exact duplicate
        ])
        errors = validate_mapping(rm, src, tgt)
        assert len(errors) >= 1
        combined = " ".join(errors).lower()
        assert "duplicate" in combined or "hips" in combined

    def test_one_to_many_target_is_flagged(self):
        """Two distinct sources mapping to the same target must produce an error."""
        src = self._make_skeleton_names("Hips", "Spine")
        tgt = self._make_skeleton_names("root")
        rm = RetargetMap(pairs=[
            ("Hips",  "root"),  # first claimant
            ("Spine", "root"),  # second claimant — same target
        ])
        errors = validate_mapping(rm, src, tgt)
        assert len(errors) >= 1
        combined = " ".join(errors).lower()
        assert "root" in combined or "multiple" in combined or "source" in combined


# ===========================================================================
# Section 10 — unmapped_target_joints (pp12-110e RED phase)
#
# Public contract (plan pp12-110e decomposition):
#   unmapped_target_joints(
#       mapping_pairs: list[list[str]],   # [[src, tgt], ...] — the tool's raw JSON param
#       target_joint_names: list[str],    # ordered list of all target skeleton joint names
#   ) -> list[str]
#
#   Returns the subset of target_joint_names NOT appearing as the 2nd element
#   of any pair in mapping_pairs, preserving the order of target_joint_names.
#
#   Rationale: used by setup_retarget handler to identify target joints that
#   have no explicit mapping so they can be passed to kinefx::mappoints
#   (the unmapped joints are left for the full-body-IK solver to handle).
#
# These tests are RED — unmapped_target_joints does not exist yet in kinefx_model.py.
# They will fail with ImportError on the first run (the import at line 57 of this
# file already includes unmapped_target_joints, which is absent from the module).
# ===========================================================================

class TestUnmappedTargetJoints:
    """unmapped_target_joints — returns target joints not covered by any mapping pair."""

    def test_partial_mapping_returns_uncovered_targets(self):
        """Only the first two targets are mapped; tail_05 must be returned."""
        mapping_pairs = [["Hips", "root"], ["Spine", "spine_01"]]
        target_joint_names = ["root", "spine_01", "tail_05"]
        result = unmapped_target_joints(mapping_pairs, target_joint_names)
        assert result == ["tail_05"], (
            f"Expected ['tail_05'], got {result!r}"
        )

    def test_full_mapping_returns_empty_list(self):
        """Every target joint is covered by exactly one pair — result is empty."""
        mapping_pairs = [["A", "root"], ["B", "spine_01"], ["C", "tail_05"]]
        target_joint_names = ["root", "spine_01", "tail_05"]
        result = unmapped_target_joints(mapping_pairs, target_joint_names)
        assert result == [], (
            f"Expected [], got {result!r}"
        )

    def test_empty_mapping_returns_all_targets(self):
        """No pairs means every target joint is unmapped."""
        mapping_pairs = []
        target_joint_names = ["root", "spine_01"]
        result = unmapped_target_joints(mapping_pairs, target_joint_names)
        assert result == ["root", "spine_01"], (
            f"Expected ['root', 'spine_01'], got {result!r}"
        )

    def test_order_preserved_from_target_joint_names(self):
        """Unmapped joints must be returned in the order they appear in target_joint_names,
        not in mapping-pair order or sorted order."""
        # Mapping covers the middle element only; result must preserve original order.
        mapping_pairs = [["X", "b"]]
        target_joint_names = ["c", "b", "a"]   # unmapped: c and a, in that order
        result = unmapped_target_joints(mapping_pairs, target_joint_names)
        assert result == ["c", "a"], (
            f"Expected ['c', 'a'] (target_joint_names order), got {result!r}"
        )
