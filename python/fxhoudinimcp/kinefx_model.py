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
