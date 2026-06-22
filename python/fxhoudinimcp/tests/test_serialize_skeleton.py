"""
test_serialize_skeleton.py — §7.3 JSON shape verification (pure pytest, off-DCC).

Exercises the public contract of:
  - derive_parents(edges, names) -> dict[str, str | None]
  - pack_trs(matrix4)            -> (t_tuple, r_tuple, s_tuple)
  - Joint, Skeleton              -> .to_dict()
  - skeleton_to_json(skeleton)   -> dict conforming to §7.3

NO hou / Qt / pxr imports anywhere.
Runs on bare CI (plain pytest, no Houdini required).

TDD phase: GREEN-ON-ARRIVAL — PR-1 ships kinefx_model.py; this file validates
its public observable output against the §7.3 contract.

testVerificationSurface: pytest-model
unitId: pp12-110b
"""

from __future__ import annotations

import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — allows running as standalone and via pytest testpaths.
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if os.path.abspath(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, os.path.abspath(_PKG_ROOT))

import pytest

from fxhoudinimcp.kinefx_model import (
    Joint,
    Skeleton,
    derive_parents,
    pack_trs,
    skeleton_to_json,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _identity_4x4() -> list[list[float]]:
    """4×4 identity matrix (row-major, translation in row 3 = [0, 0, 0, 1])."""
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _translation_4x4(tx: float, ty: float, tz: float) -> list[list[float]]:
    """4×4 matrix with the given world-space translation in row 3."""
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [tx,  ty,  tz,  1.0],
    ]


def _trs_dict(
    t: tuple[float, float, float] = (0.0, 0.0, 0.0),
    r: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    s: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    return {"t": list(t), "r": list(r), "s": list(s)}


# ===========================================================================
# Synthetic skeleton fixture
#
#   Hips (root)
#   ├── Spine
#   │   └── Head
#   ├── LeftUpLeg   (fork)
#   └── RightUpLeg  (fork)
#
# This exercises: root (parent=null), chain (Spine→Head), fork (two children
# of the same parent) — the three structural cases §7.3 requires a test to cover.
# ===========================================================================

_NAMES = ["Hips", "Spine", "Head", "LeftUpLeg", "RightUpLeg"]

# Edge list: (parent_name, child_name)
_EDGES = [
    ("Hips",  "Spine"),
    ("Spine", "Head"),
    ("Hips",  "LeftUpLeg"),
    ("Hips",  "RightUpLeg"),
]

# Rest matrices — identity for all joints (TRS = [0,0,0], [0,0,0,1], [1,1,1])
_REST_MATRICES = {name: _identity_4x4() for name in _NAMES}


def _build_skeleton() -> Skeleton:
    """Construct a Skeleton from the synthetic bone table above."""
    parents = derive_parents(_EDGES, _NAMES)
    joints = []
    for name in _NAMES:
        t, r, s = pack_trs(_REST_MATRICES[name])
        rest = _trs_dict(t, r, s)
        joints.append(Joint(name=name, parent=parents[name], rest=rest))
    return Skeleton(joints=joints)


# ===========================================================================
# Tests: derive_parents
# ===========================================================================

class TestDeriveParents:

    def test_root_has_null_parent(self):
        parents = derive_parents(_EDGES, _NAMES)
        assert parents["Hips"] is None, "Root joint must map to None"

    def test_chain_parent_correct(self):
        parents = derive_parents(_EDGES, _NAMES)
        assert parents["Spine"] == "Hips"
        assert parents["Head"] == "Spine"

    def test_fork_parents_correct(self):
        parents = derive_parents(_EDGES, _NAMES)
        assert parents["LeftUpLeg"] == "Hips"
        assert parents["RightUpLeg"] == "Hips"

    def test_all_names_present(self):
        parents = derive_parents(_EDGES, _NAMES)
        assert set(parents.keys()) == set(_NAMES)

    def test_cycle_raises(self):
        cycle_edges = [("A", "B"), ("B", "A")]
        with pytest.raises((ValueError, Exception)):
            derive_parents(cycle_edges, ["A", "B"])

    def test_double_parent_raises(self):
        bad_edges = [("A", "C"), ("B", "C")]
        with pytest.raises((ValueError, Exception)):
            derive_parents(bad_edges, ["A", "B", "C"])


# ===========================================================================
# Tests: pack_trs
# ===========================================================================

class TestPackTrs:

    def test_identity_matrix_gives_zero_translation(self):
        t, r, s = pack_trs(_identity_4x4())
        assert list(t) == pytest.approx([0.0, 0.0, 0.0])

    def test_identity_matrix_gives_unit_quaternion(self):
        t, r, s = pack_trs(_identity_4x4())
        # Identity rotation: quaternion (0,0,0,1) or normalised equivalent
        qx, qy, qz, qw = r
        norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_identity_matrix_gives_unit_scale(self):
        t, r, s = pack_trs(_identity_4x4())
        assert list(s) == pytest.approx([1.0, 1.0, 1.0])

    def test_translation_extracted_from_row3(self):
        mat = _translation_4x4(3.0, 5.0, 7.0)
        t, r, s = pack_trs(mat)
        assert list(t) == pytest.approx([3.0, 5.0, 7.0])

    def test_returns_three_components(self):
        result = pack_trs(_identity_4x4())
        assert len(result) == 3
        t, r, s = result
        assert len(t) == 3
        assert len(r) == 4
        assert len(s) == 3


# ===========================================================================
# Tests: Joint.to_dict
# ===========================================================================

class TestJointToDict:

    def test_root_joint_dict_has_null_parent(self):
        t, r, s = pack_trs(_identity_4x4())
        j = Joint(name="Hips", parent=None, rest=_trs_dict(t, r, s))
        d = j.to_dict()
        assert d["parent"] is None

    def test_child_joint_dict_has_string_parent(self):
        t, r, s = pack_trs(_identity_4x4())
        j = Joint(name="Spine", parent="Hips", rest=_trs_dict(t, r, s))
        d = j.to_dict()
        assert d["parent"] == "Hips"

    def test_joint_dict_has_name_key(self):
        t, r, s = pack_trs(_identity_4x4())
        j = Joint(name="Head", parent="Spine", rest=_trs_dict(t, r, s))
        assert "name" in j.to_dict()

    def test_joint_dict_has_rest_with_t_r_s(self):
        t, r, s = pack_trs(_identity_4x4())
        j = Joint(name="Spine", parent="Hips", rest=_trs_dict(t, r, s))
        rest = j.to_dict()["rest"]
        assert "t" in rest
        assert "r" in rest
        assert "s" in rest

    def test_anim_key_absent_when_not_provided(self):
        """§7.3: 'anim' key MUST be absent when no frame is provided."""
        t, r, s = pack_trs(_identity_4x4())
        j = Joint(name="Hips", parent=None, rest=_trs_dict(t, r, s))
        d = j.to_dict()
        assert "anim" not in d, "'anim' must be absent when Joint.anim is None"


# ===========================================================================
# Tests: Skeleton.to_dict and skeleton_to_json (§7.3 shape)
# ===========================================================================

class TestSkeletonToJson:

    def test_count_equals_joint_count(self):
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        assert out["count"] == 5

    def test_joints_is_list(self):
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        assert isinstance(out["joints"], list)

    def test_joints_length_matches_count(self):
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        assert len(out["joints"]) == out["count"]

    def test_every_joint_has_name(self):
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        for j in out["joints"]:
            assert "name" in j, f"Missing 'name' key in joint: {j}"

    def test_every_joint_has_parent_key(self):
        """'parent' key must be present for all joints (value None for root)."""
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        for j in out["joints"]:
            assert "parent" in j, f"Missing 'parent' key in joint: {j}"

    def test_every_joint_has_rest_with_t_r_s(self):
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        for j in out["joints"]:
            rest = j["rest"]
            assert "t" in rest and "r" in rest and "s" in rest, (
                f"rest dict missing t/r/s keys in joint {j['name']}: {rest}"
            )

    def test_root_parent_is_null(self):
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        root = next(j for j in out["joints"] if j["name"] == "Hips")
        assert root["parent"] is None

    def test_chain_parents_correct(self):
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        by_name = {j["name"]: j for j in out["joints"]}
        assert by_name["Spine"]["parent"] == "Hips"
        assert by_name["Head"]["parent"] == "Spine"

    def test_fork_parents_correct(self):
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        by_name = {j["name"]: j for j in out["joints"]}
        assert by_name["LeftUpLeg"]["parent"] == "Hips"
        assert by_name["RightUpLeg"]["parent"] == "Hips"

    def test_no_anim_key_when_no_frame(self):
        """§7.3: 'anim' must be absent when skeleton was built without frame data."""
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        for j in out["joints"]:
            assert "anim" not in j, (
                f"'anim' key must not appear when no frame supplied; found in {j['name']}"
            )

    def test_output_is_json_serializable(self):
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        # Must round-trip through json.dumps without error
        serialized = json.dumps(out)
        reloaded = json.loads(serialized)
        assert reloaded["count"] == 5

    def test_top_level_keys_are_count_and_joints(self):
        """§7.3 envelope: top-level keys are exactly 'count' and 'joints'."""
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        assert set(out.keys()) == {"count", "joints"}

    def test_t_r_s_are_lists(self):
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        for j in out["joints"]:
            rest = j["rest"]
            assert isinstance(rest["t"], list)
            assert isinstance(rest["r"], list)
            assert isinstance(rest["s"], list)

    def test_t_has_3_elements(self):
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        for j in out["joints"]:
            assert len(j["rest"]["t"]) == 3

    def test_r_has_4_elements(self):
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        for j in out["joints"]:
            assert len(j["rest"]["r"]) == 4

    def test_s_has_3_elements(self):
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        for j in out["joints"]:
            assert len(j["rest"]["s"]) == 3

    def test_identity_rest_has_zero_translation(self):
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        for j in out["joints"]:
            t = j["rest"]["t"]
            assert t == pytest.approx([0.0, 0.0, 0.0]), (
                f"Expected zero translation for identity rest on {j['name']}, got {t}"
            )

    def test_identity_rest_has_unit_scale(self):
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        for j in out["joints"]:
            s = j["rest"]["s"]
            assert s == pytest.approx([1.0, 1.0, 1.0]), (
                f"Expected unit scale for identity rest on {j['name']}, got {s}"
            )

    def test_identity_rest_quaternion_is_normalised(self):
        skeleton = _build_skeleton()
        out = skeleton_to_json(skeleton)
        for j in out["joints"]:
            qx, qy, qz, qw = j["rest"]["r"]
            norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
            assert norm == pytest.approx(1.0, abs=1e-6), (
                f"Quaternion not normalised in {j['name']}: norm={norm}"
            )

    def test_skeleton_to_json_delegates_to_to_dict(self):
        """skeleton_to_json(s) must equal s.to_dict()."""
        skeleton = _build_skeleton()
        assert skeleton_to_json(skeleton) == skeleton.to_dict()
