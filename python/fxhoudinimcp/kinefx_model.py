"""Pure-logic KineFX / APEX model layer. No hou / Qt / pxr imports."""

from __future__ import annotations

import math
from dataclasses import dataclass  # F7: removed unused 'field' import
from typing import Any


# ---------------------------------------------------------------------------
# Joint
# ---------------------------------------------------------------------------

@dataclass
class Joint:
    """One joint in a KineFX skeleton with rest-pose and optional animated TRS."""

    name: str
    parent: str | None
    rest: dict  # {"t": [...], "r": [...], "s": [...]}
    anim: dict | None = None  # animated TRS; None when no frame was requested

    def to_dict(self) -> dict:
        """Serialize to the §7.3 JSON joint shape; 'anim' key omitted when None."""
        d: dict[str, Any] = {
            "name": self.name,
            "parent": self.parent,
            "rest": self.rest,
        }
        if self.anim is not None:
            d["anim"] = self.anim
        return d

    @staticmethod
    def from_dict(d: dict) -> "Joint":
        """Restore a Joint from its serialized dict."""
        return Joint(
            name=d["name"],
            parent=d.get("parent"),
            rest=d["rest"],
            anim=d.get("anim"),
        )


# ---------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------

@dataclass
class Skeleton:
    """Ordered list of joints forming a KineFX skeleton."""

    joints: list  # list[Joint]

    @property
    def joint_count(self) -> int:
        """Number of joints in this skeleton."""
        return len(self.joints)

    def to_dict(self) -> dict:
        """Serialize to {'count': N, 'joints': [...]} shape per §7.3."""
        return {
            "count": len(self.joints),
            "joints": [j.to_dict() for j in self.joints],
        }

    @staticmethod
    def from_dict(d: dict) -> "Skeleton":
        """Restore a Skeleton from its serialized dict."""
        joints = [Joint.from_dict(j) for j in d.get("joints", [])]
        return Skeleton(joints=joints)


# ---------------------------------------------------------------------------
# MotionClip
# ---------------------------------------------------------------------------

@dataclass
class MotionClip:
    """Metadata for a KineFX motion clip; per-frame TRS is opt-in (sample-don't-dump)."""

    joint_count: int
    frame_range: tuple  # (start, end)
    frames: list | None = None  # per-frame TRS data; None = omit from dict

    def to_dict(self) -> dict:
        """Serialize clip metadata; per-frame data included only when frames is set."""
        d: dict[str, Any] = {
            "joint_count": self.joint_count,
            "frame_range": list(self.frame_range),
        }
        if self.frames is not None:
            d["frames"] = self.frames
        return d


# ---------------------------------------------------------------------------
# RetargetMap
# ---------------------------------------------------------------------------

@dataclass
class RetargetMap:
    """Source→target joint name pairs for animation retargeting."""

    pairs: list  # list[tuple[str, str]]

    def to_dict(self) -> dict:
        """Serialize to a list-of-pairs representation."""
        return {"pairs": [list(p) for p in self.pairs]}


# ---------------------------------------------------------------------------
# ApexNodeSummary
# ---------------------------------------------------------------------------

@dataclass
class ApexNodeSummary:
    """Summary of a single APEX graph node — name, type, and port list."""

    name: str
    node_type: str
    ports: list  # list[str]

    def to_dict(self) -> dict:
        """Serialize to {'name': ..., 'type': ..., 'ports': [...]}."""
        return {
            "name": self.name,
            "type": self.node_type,
            "ports": list(self.ports),
        }


# ---------------------------------------------------------------------------
# ApexGraphSummary
# ---------------------------------------------------------------------------

@dataclass
class ApexGraphSummary:
    """High-level summary of an APEX rig graph: nodes, wires, and control count."""

    nodes: list  # list[ApexNodeSummary]
    wires: list  # list[tuple[str, str]] e.g. ("n1.out", "n2.in")
    control_count: int

    def to_dict(self) -> dict:
        """Serialize to {'nodes': [...], 'wires': [...], 'control_count': N}."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "wires": [list(w) for w in self.wires],
            "control_count": self.control_count,
        }


# ---------------------------------------------------------------------------
# KinefxRequest
# ---------------------------------------------------------------------------

@dataclass
class KinefxRequest:
    """A KineFX MCP tool invocation request — tool name and parameter dict."""

    tool: str
    params: dict


# ---------------------------------------------------------------------------
# OpResult
# ---------------------------------------------------------------------------

@dataclass
class OpResult:
    """Result envelope for a KineFX MCP operation."""

    ok: bool
    data: Any = None
    error: str | None = None

    def to_dict(self) -> dict:
        """Serialize to {'ok': bool, 'data': ..., 'error': ...}."""
        d: dict[str, Any] = {"ok": self.ok}
        if self.data is not None:
            d["data"] = self.data
        if self.error is not None:
            d["error"] = self.error
        return d


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def derive_parents(
    edges: list,  # list[tuple[str, str]] — (parent_name, child_name)
    names: list,  # list[str] — all joint names in order
) -> dict:  # dict[str, str | None]
    """Map every joint name to its parent name; roots map to None.

    Args:
        edges: list of (parent_name, child_name) pairs extracted from the
               skeleton's bone connectivity.
        names: ordered list of all joint names in the skeleton.

    Returns:
        A dict keyed by every name in ``names``; value is the parent name
        string, or None for root joints (those that never appear as a child).

    Raises:
        ValueError: if a joint appears as the child in more than one edge
                    (double-parent — invalid KineFX skeleton).
        ValueError: if a cycle is detected in the parent chain (e.g. A->B->A).
    """
    # Build a child-to-parent lookup from the edge list.
    # F1 + F2: detect double-parent and cycle, raise ValueError on both.
    child_to_parent: dict[str, str] = {}
    for parent_name, child_name in edges:
        # F2: double-parent detection — a joint can have at most one parent.
        if child_name in child_to_parent:
            raise ValueError(
                f"derive_parents: joint '{child_name}' has multiple parents"
            )
        child_to_parent[child_name] = parent_name

    # F1: cycle detection — walk each child's ancestor chain; a joint that
    # appears more than once in the chain means a cycle exists.
    for start in child_to_parent:
        visited: set[str] = set()
        current = start
        while current in child_to_parent:
            if current in visited:
                raise ValueError(
                    f"derive_parents: cycle detected at '{current}'"
                )
            visited.add(current)
            current = child_to_parent[current]

    # For each joint, look up its parent; roots are absent from the lookup.
    return {n: child_to_parent.get(n) for n in names}


def pack_trs(
    matrix4: list,  # list[list[float]] — 4x4 affine matrix, row-major
) -> tuple:  # (translate, rotate, scale)
    """Decompose a 4x4 affine matrix into (translate, rotate, scale) components.

    The matrix is in row-major form as produced when ``hou.Matrix4`` values
    are converted to plain Python before reaching the pure layer.  Translation
    occupies the last row (row index 3, columns 0-2).

    **Input convention (F3 / F4 / Opus-F1):**
    - Row-major layout: ``matrix4[row][col]``.
    - Translation is in row 3 (``matrix4[3][0..2]``), NOT in column 3.
      This matches the ``hou.Matrix4`` storage convention used throughout
      the fxhoudinimcp bridge.  The caller (character_handlers.py and any
      PR-2 code) MUST honor this row-major / row-3-translation layout.
    - The upper-left 3x3 sub-matrix is assumed to encode scale * rotation
      for an **orthonormal** (right-angle axes) matrix — i.e. the three
      column vectors of the upper-left 3x3 must be mutually perpendicular
      after normalisation.  Sheared or degenerate matrices (where any column
      has near-zero length, ``< 1e-12``) will produce non-unit quaternions
      or a zero scale component; the caller is responsible for ensuring the
      input is a valid rigid-body transform.

    Args:
        matrix4: 4x4 nested list of floats in row-major order.

    Returns:
        A 3-tuple ``(translate, rotate, scale)`` where:
        - translate: (tx, ty, tz) floats
        - rotate:    (rx, ry, rz, rw) quaternion floats -- may not be unit
                     if the input contains shear or a degenerate column.
        - scale:     (sx, sy, sz) floats -- a degenerate column (length
                     near-zero) results in sx/sy/sz == 0.0 for that axis.
    """
    # Translation is in the last row, first three columns.
    tx, ty, tz = matrix4[3][0], matrix4[3][1], matrix4[3][2]

    # The 3x3 upper-left submatrix encodes scale * rotation.
    # Column vectors are the basis vectors; their lengths are the scale factors.
    col0 = (matrix4[0][0], matrix4[1][0], matrix4[2][0])
    col1 = (matrix4[0][1], matrix4[1][1], matrix4[2][1])
    col2 = (matrix4[0][2], matrix4[1][2], matrix4[2][2])

    sx = math.sqrt(col0[0]**2 + col0[1]**2 + col0[2]**2)
    sy = math.sqrt(col1[0]**2 + col1[1]**2 + col1[2]**2)
    sz = math.sqrt(col2[0]**2 + col2[1]**2 + col2[2]**2)

    # Normalise the basis vectors to isolate the rotation matrix.
    def _safe_div(v: float, d: float) -> float:
        return v / d if d > 1e-12 else 0.0

    r00 = _safe_div(col0[0], sx); r10 = _safe_div(col0[1], sx); r20 = _safe_div(col0[2], sx)
    r01 = _safe_div(col1[0], sy); r11 = _safe_div(col1[1], sy); r21 = _safe_div(col1[2], sy)
    r02 = _safe_div(col2[0], sz); r12 = _safe_div(col2[1], sz); r22 = _safe_div(col2[2], sz)

    # Convert the 3x3 rotation matrix to a unit quaternion (x, y, z, w).
    # Shepperd's method -- numerically stable across all rotation matrices.
    trace = r00 + r11 + r22
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (r21 - r12) * s
        qy = (r02 - r20) * s
        qz = (r10 - r01) * s
    elif r00 > r11 and r00 > r22:
        s = 2.0 * math.sqrt(1.0 + r00 - r11 - r22)
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
    elif r11 > r22:
        s = 2.0 * math.sqrt(1.0 + r11 - r00 - r22)
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + r22 - r00 - r11)
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s

    return (tx, ty, tz), (qx, qy, qz, qw), (sx, sy, sz)


def skeleton_to_json(skeleton: "Skeleton") -> dict:
    """Serialize a Skeleton to the §7.3 JSON dict shape.

    Args:
        skeleton: a Skeleton instance to serialize.

    Returns:
        A dict with 'count' and 'joints' keys, suitable for json.dumps.
    """
    return skeleton.to_dict()


def unmapped_target_joints(
    mapping_pairs: list,  # list[list[str]] -- [[src, tgt], ...] raw from MCP param
    target_joint_names: list,  # list[str] -- ordered list of all target joint names
) -> list:  # list[str] -- target joints NOT appearing as 2nd element of any pair
    """Return target joint names not covered by any mapping pair.

    Preserves the order of target_joint_names.  Only the 2nd element of each
    pair (the target joint) is checked; the source joint (1st element) is ignored.

    Args:
        mapping_pairs: list of [source_joint, target_joint] pairs as forwarded
                       from the MCP wrapper params dict (raw JSON lists).
                       May be None or empty -- both mean "no mapping provided".
        target_joint_names: ordered list of all joint names in the target skeleton.

    Returns:
        An ordered list of target joint names that have no corresponding
        mapping entry.  Empty list means all targets are covered.
    """
    mapped = {pair[1] for pair in (mapping_pairs or [])}
    return [name for name in target_joint_names if name not in mapped]


def affected_target_joints(
    requested_joints: list,  # list[str] -- joints caller wants secondarymotion applied to
    actual_joint_names: list,  # list[str] -- all joint names present in the skeleton
) -> list:  # list[str] -- requested joints that exist in the skeleton
    """Return the subset of requested_joints that exist in actual_joint_names.

    Preserves the order of requested_joints (NOT the order of actual_joint_names).
    Joints absent from the skeleton are silently dropped.

    Args:
        requested_joints: joints the caller asked to apply secondarymotion to.
                          May be None or empty -- both mean "no specific joints".
        actual_joint_names: ordered list of all joint names in the target skeleton.

    Returns:
        An ordered list (by requested_joints order) of joint names that exist
        in actual_joint_names.  Empty list means no requested joint exists in
        the skeleton.
    """
    actual_set = set(actual_joint_names)
    return [name for name in (requested_joints or []) if name in actual_set]


def validate_mapping(
    retarget_map: "RetargetMap",
    source_skeleton: "Skeleton",
    target_skeleton: "Skeleton",
) -> list:  # list[str] — empty on success
    """Validate that all joint names in a RetargetMap exist in their respective skeletons.

    Also detects duplicate source->target pairs and one-to-many target mappings
    (the same target joint claimed by more than one source joint).

    Args:
        retarget_map:     the source->target joint pair mapping to validate.
        source_skeleton:  the skeleton providing source joint names.
        target_skeleton:  the skeleton providing target joint names.

    Returns:
        A list of error strings -- one per invalid pair or structural problem.
        Empty list means valid.
    """
    src_names = {j.name for j in source_skeleton.joints}
    tgt_names = {j.name for j in target_skeleton.joints}
    errors: list[str] = []

    # F5: tracking sets for duplicate-pair and one-to-many detection.
    seen_pairs: set[tuple[str, str]] = set()
    target_claimed_by: dict[str, str] = {}  # target_joint -> first source that claimed it

    for src_joint, tgt_joint in retarget_map.pairs:
        pair = (src_joint, tgt_joint)

        # F5: duplicate pair detection -- same (src, tgt) listed more than once.
        if pair in seen_pairs:
            errors.append(
                f"Duplicate mapping pair: '{src_joint}' -> '{tgt_joint}'."
            )
            continue
        seen_pairs.add(pair)

        if src_joint not in src_names:
            errors.append(
                f"Source joint '{src_joint}' not found in source skeleton."
            )
        if tgt_joint not in tgt_names:
            errors.append(
                f"Target joint '{tgt_joint}' not found in target skeleton."
            )

        # F5: one-to-many target detection -- same target claimed by two sources.
        if tgt_joint in target_claimed_by and target_claimed_by[tgt_joint] != src_joint:
            errors.append(
                f"Target joint '{tgt_joint}' is mapped from multiple sources: "
                f"'{target_claimed_by[tgt_joint]}' and '{src_joint}'."
            )
        else:
            target_claimed_by[tgt_joint] = src_joint

    return errors


# ---------------------------------------------------------------------------
# node_plan — pure-logic node-creation plan for each KineFX mutating tool
# ---------------------------------------------------------------------------

def node_plan(tool: str, args: dict) -> dict:
    """Return {node_type, inputs, key_parms} for each KineFX mutating tool.

    Pure-logic: no hou / Qt / pxr imports (CL-015).
    This is consumed by preview_fn hooks so the §4.3 approval payload can
    show the operator exactly what node(s) will be created before the gate
    opens.

    Args:
        tool: one of the five KineFX mutating tool names registered in
              character_handlers.py.
        args: the kwargs dict the handler will receive (same dict the
              preview_fn receives).

    Returns:
        A dict with keys:
            node_type  — primary Houdini node type name
            inputs     — list of input slot labels (positional order)
            key_parms  — dict of parameter names to example values from args

    Raises:
        ValueError: if *tool* is not one of the five supported mutating tools.
    """
    if tool == "import_fbx_character":
        return {
            "node_type": "kinefx::fbxcharacterimport",
            "inputs": [],
            "key_parms": {"fbxfile": args.get("path", args.get("fbxfile", ""))},
        }
    elif tool == "import_fbx_animation":
        return {
            "node_type": "kinefx::fbxanimimport",
            "inputs": [],
            "key_parms": {"fbxfile": args.get("path", args.get("fbxfile", ""))},
        }
    elif tool == "setup_bonedeform":
        return {
            "node_type": "bonedeform",
            "inputs": ["geo", "rest", "anim"],
            "key_parms": {},
        }
    elif tool == "setup_retarget":
        mapping_method = args.get("mapping_method", args.get("method", "by_name"))
        if mapping_method == "explicit" or (args.get("mapping") is not None):
            chain = ["kinefx::rigmatchpose", "kinefx::mappoints", "kinefx::fullbodyik"]
        else:
            chain = ["kinefx::rigmatchpose", "kinefx::fullbodyik"]
        return {
            "node_type": "kinefx::fullbodyik",
            "inputs": ["source_node", "target_node"],
            "key_parms": {
                "chain": chain,
                "mapping_method": mapping_method,
            },
        }
    elif tool == "apply_secondarymotion":
        _params = args.get("params") or {}
        if not isinstance(_params, dict):
            _params = {}
        return {
            "node_type": "kinefx::secondarymotion",
            "inputs": ["skeleton_sop"],
            "key_parms": {
                "effect": _params.get("effect", args.get("effect", 0.5)),
                "joint_group": args.get("joints") or "",
            },
        }
    else:
        raise ValueError(f"node_plan: unknown or unsupported tool {tool!r}")
