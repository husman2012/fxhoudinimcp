"""
test_node_plan.py — RED phase for pp12-110g

Pure-logic pytest: asserts node_plan(tool, args) returns the correct
{node_type, inputs, key_parms} for each of the 5 shipped gated KineFX
character tools.

No hou / Qt / pxr imports anywhere in this file (CL-015).
Runs under plain pytest headless (off-DCC, no Houdini install required).

RED: node_plan does not exist in kinefx_model.py.
All tests will fail with ImportError / AttributeError on first run.

tdd-with-agents.md §4 — this file is authored BEFORE any implementation
exists.  hou-dev MUST NOT modify this file.
"""

from __future__ import annotations

import sys
import os

# ---------------------------------------------------------------------------
# Path bootstrap — allows running as a standalone script AND via pytest.
# Same pattern as test_kinefx_model.py in this package.
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if os.path.abspath(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, os.path.abspath(_PKG_ROOT))

import pytest

# ---------------------------------------------------------------------------
# Import under test — node_plan does NOT exist yet.
# This import will raise ImportError on first run (RED gate confirmed).
# ---------------------------------------------------------------------------
from fxhoudinimcp.kinefx_model import node_plan  # noqa: E402  (expected ImportError)


# ===========================================================================
# Contract specification (grounded from character_handlers.py)
#
# node_plan(tool: str, args: dict) -> dict with keys:
#   node_type : str   — the Houdini node type string that this tool creates
#   inputs    : list  — ordered input labels; [] when the tool has no wired inputs
#   key_parms : dict  — the most important editable parameters the tool sets
#
# Node types are ground-truth from the handler bodies:
#   import_fbx_character  → "kinefx::fbxcharacterimport"   (line 412 character_handlers.py)
#   import_fbx_animation  → "kinefx::fbxanimimport"        (line 516)
#   setup_bonedeform      → "bonedeform"                    (line 631)
#   setup_retarget        → chain: rigmatchpose → fullbodyik (explicit-mapping: + mappoints)
#   apply_secondarymotion → "kinefx::secondarymotion"       (line 936)
# ===========================================================================


# ---------------------------------------------------------------------------
# Helper — build a minimal args dict for each tool
# ---------------------------------------------------------------------------

def _fbx_char_args() -> dict:
    return {"fbxfile": "/tmp/char.fbx", "node": "/obj/geo1"}


def _fbx_anim_args() -> dict:
    return {"fbxfile": "/tmp/anim.fbx", "target_node": "/obj/geo1/skel"}


def _bonedeform_args() -> dict:
    return {
        "geo_node": "/obj/geo1/rest_geo",
        "rest_node": "/obj/geo1/rest_skel",
        "anim_node": "/obj/geo1/anim_skel",
    }


def _retarget_args_byname() -> dict:
    return {
        "source_node": "/obj/source_skel",
        "target_node": "/obj/target_skel",
        "mapping_method": "by_name",
    }


def _retarget_args_explicit() -> dict:
    return {
        "source_node": "/obj/source_skel",
        "target_node": "/obj/target_skel",
        "mapping_method": "explicit",
        "mapping_pairs": [["Hips", "root"]],
    }


def _secondarymotion_args() -> dict:
    return {
        "node": "/obj/geo1/skel",
        "effect": 0.5,
        "joint_group": "secondary_joints",
    }


# ===========================================================================
# Tool: import_fbx_character
# ===========================================================================

class TestNodePlanImportFbxCharacter:
    """node_plan returns the correct descriptor for import_fbx_character."""

    def test_node_type_is_kinefx_fbxcharacterimport(self):
        """node_plan for import_fbx_character must return kinefx::fbxcharacterimport."""
        result = node_plan("import_fbx_character", _fbx_char_args())
        assert result["node_type"] == "kinefx::fbxcharacterimport", (
            f"Expected 'kinefx::fbxcharacterimport', got {result.get('node_type')!r}"
        )

    def test_inputs_is_empty_list(self):
        """import_fbx_character creates a standalone node with no wired inputs."""
        result = node_plan("import_fbx_character", _fbx_char_args())
        assert result["inputs"] == [], (
            "import_fbx_character node has no wired inputs — inputs must be []"
        )

    def test_key_parms_contains_fbxfile(self):
        """key_parms must include 'fbxfile' reflecting the source file path."""
        result = node_plan("import_fbx_character", _fbx_char_args())
        assert "fbxfile" in result["key_parms"], (
            f"key_parms must include 'fbxfile'; got keys {list(result.get('key_parms', {}).keys())}"
        )

    def test_key_parms_fbxfile_matches_arg(self):
        """key_parms['fbxfile'] must equal the fbxfile argument passed in."""
        args = _fbx_char_args()
        result = node_plan("import_fbx_character", args)
        assert result["key_parms"]["fbxfile"] == args["fbxfile"]

    def test_result_has_three_required_keys(self):
        """Result dict must contain node_type, inputs, and key_parms."""
        result = node_plan("import_fbx_character", _fbx_char_args())
        assert "node_type" in result
        assert "inputs" in result
        assert "key_parms" in result


# ===========================================================================
# Tool: import_fbx_animation
# ===========================================================================

class TestNodePlanImportFbxAnimation:
    """node_plan returns the correct descriptor for import_fbx_animation."""

    def test_node_type_is_kinefx_fbxanimimport(self):
        """node_plan for import_fbx_animation must return kinefx::fbxanimimport."""
        result = node_plan("import_fbx_animation", _fbx_anim_args())
        assert result["node_type"] == "kinefx::fbxanimimport", (
            f"Expected 'kinefx::fbxanimimport', got {result.get('node_type')!r}"
        )

    def test_inputs_is_empty_list(self):
        """import_fbx_animation creates a standalone animation node with no wired inputs."""
        result = node_plan("import_fbx_animation", _fbx_anim_args())
        assert result["inputs"] == []

    def test_key_parms_contains_fbxfile(self):
        """key_parms must include 'fbxfile'."""
        result = node_plan("import_fbx_animation", _fbx_anim_args())
        assert "fbxfile" in result["key_parms"]

    def test_key_parms_fbxfile_matches_arg(self):
        """key_parms['fbxfile'] must equal the fbxfile arg."""
        args = _fbx_anim_args()
        result = node_plan("import_fbx_animation", args)
        assert result["key_parms"]["fbxfile"] == args["fbxfile"]

    def test_result_has_three_required_keys(self):
        result = node_plan("import_fbx_animation", _fbx_anim_args())
        assert "node_type" in result
        assert "inputs" in result
        assert "key_parms" in result


# ===========================================================================
# Tool: setup_bonedeform
# ===========================================================================

class TestNodePlanSetupBonedeform:
    """node_plan returns the correct descriptor for setup_bonedeform."""

    def test_node_type_is_bonedeform(self):
        """node_plan for setup_bonedeform must return 'bonedeform'."""
        result = node_plan("setup_bonedeform", _bonedeform_args())
        assert result["node_type"] == "bonedeform", (
            f"Expected 'bonedeform', got {result.get('node_type')!r}"
        )

    def test_inputs_has_three_entries(self):
        """bonedeform takes 3 wired inputs: input0=geo, input1=rest, input2=anim."""
        result = node_plan("setup_bonedeform", _bonedeform_args())
        assert len(result["inputs"]) == 3, (
            f"Expected 3 inputs for bonedeform, got {len(result['inputs'])}"
        )

    def test_input_order_geo_rest_anim(self):
        """Input labels must be ordered [geo/geo_node, rest/rest_node, anim/anim_node].
        The exact label names may vary (geo vs geo_node) but the COUNT and ORDER must be
        geo-first, rest-second, anim-third."""
        result = node_plan("setup_bonedeform", _bonedeform_args())
        inputs = result["inputs"]
        # Verify by checking that relevant keywords appear in the right positions
        assert len(inputs) >= 3
        # First input should relate to geo
        assert "geo" in str(inputs[0]).lower() or "geometry" in str(inputs[0]).lower(), (
            f"Input 0 should reference geo, got: {inputs[0]!r}"
        )
        # Second input should relate to rest
        assert "rest" in str(inputs[1]).lower(), (
            f"Input 1 should reference rest, got: {inputs[1]!r}"
        )
        # Third input should relate to anim
        assert "anim" in str(inputs[2]).lower(), (
            f"Input 2 should reference anim, got: {inputs[2]!r}"
        )

    def test_result_has_three_required_keys(self):
        result = node_plan("setup_bonedeform", _bonedeform_args())
        assert "node_type" in result
        assert "inputs" in result
        assert "key_parms" in result


# ===========================================================================
# Tool: setup_retarget
#
# setup_retarget creates a chain of nodes, not a single node:
#   by_name path:   kinefx::rigmatchpose → kinefx::fullbodyik
#   explicit path:  kinefx::rigmatchpose → kinefx::mappoints → kinefx::fullbodyik
#
# The node_plan must communicate this chain.  The contract allows either:
#   (a) node_type as a composite string (e.g. "kinefx::rigmatchpose+kinefx::fullbodyik")
#   (b) node_type as a list of node type strings
#   (c) node_type as the FINAL node in the chain ("kinefx::fullbodyik")
#       with key_parms["chain"] or a "nodes" key listing all in order
#
# These tests assert the MINIMUM contract regardless of form:
#   - "kinefx::rigmatchpose" must appear somewhere in the output
#   - "kinefx::fullbodyik" must appear somewhere in the output
#   - For explicit mapping: "kinefx::mappoints" must also appear
# ===========================================================================

class TestNodePlanSetupRetarget:
    """node_plan returns the correct descriptor for setup_retarget (chain of nodes)."""

    def _result_as_string(self, result: dict) -> str:
        """Flatten the entire result dict to a string for presence checks."""
        import json
        return json.dumps(result)

    def test_rigmatchpose_appears_in_result_byname(self):
        """The rigmatchpose node type must appear in the by_name path output."""
        result = node_plan("setup_retarget", _retarget_args_byname())
        flat = self._result_as_string(result)
        assert "rigmatchpose" in flat, (
            f"'kinefx::rigmatchpose' must appear in node_plan for setup_retarget; got: {flat}"
        )

    def test_fullbodyik_appears_in_result_byname(self):
        """The fullbodyik node type must appear in the by_name path output."""
        result = node_plan("setup_retarget", _retarget_args_byname())
        flat = self._result_as_string(result)
        assert "fullbodyik" in flat, (
            f"'kinefx::fullbodyik' must appear in node_plan for setup_retarget; got: {flat}"
        )

    def test_explicit_path_includes_mappoints(self):
        """For explicit mapping, kinefx::mappoints must appear in the chain."""
        result = node_plan("setup_retarget", _retarget_args_explicit())
        flat = self._result_as_string(result)
        assert "mappoints" in flat, (
            f"'kinefx::mappoints' must appear in node_plan for explicit-mapping retarget; got: {flat}"
        )

    def test_inputs_contains_source_and_target(self):
        """setup_retarget wires source and target skeletons as inputs."""
        result = node_plan("setup_retarget", _retarget_args_byname())
        assert len(result["inputs"]) >= 2, (
            "setup_retarget requires at least 2 inputs (source and target skeletons)"
        )

    def test_result_has_three_required_keys(self):
        result = node_plan("setup_retarget", _retarget_args_byname())
        assert "node_type" in result
        assert "inputs" in result
        assert "key_parms" in result


# ===========================================================================
# Tool: apply_secondarymotion
# ===========================================================================

class TestNodePlanApplySecondarymotion:
    """node_plan returns the correct descriptor for apply_secondarymotion."""

    def test_node_type_is_kinefx_secondarymotion(self):
        """node_plan for apply_secondarymotion must return 'kinefx::secondarymotion'."""
        result = node_plan("apply_secondarymotion", _secondarymotion_args())
        assert result["node_type"] == "kinefx::secondarymotion", (
            f"Expected 'kinefx::secondarymotion', got {result.get('node_type')!r}"
        )

    def test_inputs_contains_skeleton_input(self):
        """apply_secondarymotion wires the skeleton node as input 0."""
        result = node_plan("apply_secondarymotion", _secondarymotion_args())
        assert len(result["inputs"]) >= 1, (
            "apply_secondarymotion must have at least 1 input (the skeleton node)"
        )

    def test_key_parms_contains_effect_or_strength(self):
        """key_parms must include the effect/strength parameter that controls intensity."""
        result = node_plan("apply_secondarymotion", _secondarymotion_args())
        # Accept 'effect', 'strength', or 'scale' as the parameter name
        has_effect_key = any(
            k in result.get("key_parms", {})
            for k in ("effect", "strength", "scale", "effectscale")
        )
        assert has_effect_key, (
            f"key_parms must include an effect/strength/scale key; "
            f"got keys: {list(result.get('key_parms', {}).keys())}"
        )

    def test_key_parms_contains_joint_group_or_group(self):
        """key_parms must include a joint group / group parameter."""
        result = node_plan("apply_secondarymotion", _secondarymotion_args())
        has_group_key = any(
            k in result.get("key_parms", {})
            for k in ("jointgroup", "joint_group", "group", "joints")
        )
        assert has_group_key, (
            f"key_parms must include a joint group key; "
            f"got keys: {list(result.get('key_parms', {}).keys())}"
        )

    def test_result_has_three_required_keys(self):
        result = node_plan("apply_secondarymotion", _secondarymotion_args())
        assert "node_type" in result
        assert "inputs" in result
        assert "key_parms" in result


# ===========================================================================
# Cross-tool: unknown tool raises ValueError
# ===========================================================================

class TestNodePlanUnknownTool:
    """node_plan raises ValueError for an unrecognised tool name."""

    def test_unknown_tool_raises_value_error(self):
        """Passing an unrecognised tool name must raise ValueError."""
        with pytest.raises((ValueError, KeyError)):
            node_plan("totally_unknown_tool_xyz", {})

    def test_apex_graph_plan_out_of_scope(self):
        """apex_graph_plan is H22-blocked (ADR-0003) and must not be in node_plan."""
        # The result of this call may be ValueError OR a dict explicitly
        # stating it's unsupported.  We just confirm it does NOT silently
        # succeed with a valid kinefx node type.
        try:
            result = node_plan("apex_graph_plan", {})
            # If it returns without raising, it must not claim a valid kinefx node_type
            node_type = result.get("node_type", "")
            assert node_type == "" or "unsupported" in str(node_type).lower() or "blocked" in str(node_type).lower(), (
                "apex_graph_plan is H22-blocked; node_plan must not return a valid node_type for it"
            )
        except (ValueError, KeyError, NotImplementedError):
            pass  # Raising is acceptable — it is out of scope


# ===========================================================================
# CL-015 hou-free importability verification
# ===========================================================================

class TestNodePlanHouFreeImport:
    """node_plan must be callable with zero hou/Qt/pxr dependency."""

    def test_node_plan_callable_without_hou(self):
        """Calling node_plan must not require hou to be installed."""
        # If we reached this class, the import at the top of the file succeeded
        # without hou — that IS the proof.  This test also verifies the callable
        # signature is correct (not just importable).
        result = node_plan("apply_secondarymotion", _secondarymotion_args())
        assert result is not None

    def test_hou_not_imported_by_kinefx_model(self):
        """kinefx_model.py must not import hou at the module top-level (CL-015)."""
        import fxhoudinimcp.kinefx_model as km
        import inspect
        import re
        source = inspect.getsource(km)
        source_no_comments = re.sub(r"#[^\n]*", "", source)
        assert "import hou" not in source_no_comments, (
            "kinefx_model.py must not import hou (CL-015 / invariant 15)"
        )
