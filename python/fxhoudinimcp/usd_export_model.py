"""
usd_export_model.py — pure-logic dataclasses for the fxhoudinimcp USD/MaterialX
export pipeline (PP12-112 PR-1).

Contract: imports NO hou, NO Qt/PySide6, NO pxr, NO numpy, NO MaterialX.
Plain Python stdlib only.  Pytest-able off-DCC (CL-015).

Classes
-------
LayerSummary    — snapshot of a USD layer's structural properties
MtlxSummary     — snapshot of a MaterialX document's key properties
DisciplineCheck — a single USD discipline validation check result
ValidationReport — aggregated USD export validation verdict + checks
ExportRequest   — caller-facing USD export request

Key fixes (from lockedFieldContract rev2)
------------------------------------------
B-1: MtlxSummary.validate_errors uses field(default_factory=list),
     NOT a bare class-level default [] — two instances must not share the list.
M-4: DisciplineCheck.__post_init__ raises ValueError on status values
     other than 'pass', 'warn', 'fail'.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# LayerSummary
# ---------------------------------------------------------------------------

@dataclass
class LayerSummary:
    """Snapshot of a USD layer's structural properties.

    Fields
    ------
    default_prim       : str | None     — layer's defaultPrim opinion (None if unset)
    root_prims         : list[str]      — top-level prim paths in the layer
    sublayers          : list[str]      — sublayer stack paths
    current_format     : str            — detected format string ('usda','usdc','usdz')
    has_mtlx_material  : bool           — True if a MaterialX material binding is present
    """

    default_prim: Optional[str]
    root_prims: list
    sublayers: list
    current_format: str
    has_mtlx_material: bool

    def to_dict(self) -> dict:
        """Serialize to a plain dict with exactly five keys."""
        return {
            "default_prim": self.default_prim,
            "root_prims": list(self.root_prims),
            "sublayers": list(self.sublayers),
            "current_format": self.current_format,
            "has_mtlx_material": self.has_mtlx_material,
        }


# ---------------------------------------------------------------------------
# MtlxSummary
# ---------------------------------------------------------------------------

@dataclass
class MtlxSummary:
    """Snapshot of a MaterialX document's key properties.

    Fields
    ------
    nodegraphs             : list[str]  — names of NodeGraph elements found
    surface_nodes          : list[str]  — surface-shader node names found
    inputs_with_abs_paths  : list[str]  — input names whose file values are absolute paths
    validate_ok            : bool       — True if pxr MaterialX validation passed
    validate_errors        : list[str]  — validation error messages (empty if ok)
                                          B-1: must use field(default_factory=list)

    to_dict() shape
    ---------------
    {
        "nodegraphs": [...],
        "surface_nodes": [...],
        "inputs_with_abs_paths": [...],
        "validate": {
            "ok": <bool>,
            "errors": [...]
        }
    }
    Note: 'validate_ok' and 'validate_errors' are NESTED under the 'validate' key.
    """

    nodegraphs: list
    surface_nodes: list
    inputs_with_abs_paths: list
    validate_ok: bool
    # B-1: must use field(default_factory=list) — bare = [] is a mutable-default bug
    validate_errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict.

        The 'validate_ok' and 'validate_errors' fields are nested under
        a 'validate' sub-dict with keys 'ok' and 'errors'.
        """
        return {
            "nodegraphs": list(self.nodegraphs),
            "surface_nodes": list(self.surface_nodes),
            "inputs_with_abs_paths": list(self.inputs_with_abs_paths),
            "validate": {
                "ok": self.validate_ok,
                "errors": list(self.validate_errors),
            },
        }


# ---------------------------------------------------------------------------
# DisciplineCheck
# ---------------------------------------------------------------------------

_VALID_STATUSES = frozenset({"pass", "warn", "fail"})


@dataclass
class DisciplineCheck:
    """A single USD discipline validation check result.

    Fields
    ------
    id     : str   — unique check identifier (e.g. 'no_world_wrapper')
    status : str   — exactly one of 'pass', 'warn', 'fail'
    msg    : str   — short human-readable message (empty string when not applicable)

    Raises
    ------
    ValueError  (M-4)
        Raised at construction time if *status* is not 'pass', 'warn', or 'fail'.

    to_dict() shape
    ---------------
    Empty msg  -> {'id': ..., 'status': ...}         (2 keys — 'msg' omitted)
    Non-empty  -> {'id': ..., 'status': ..., 'msg': ...}  (3 keys)
    """

    id: str
    status: str
    msg: str = ""

    def __post_init__(self) -> None:
        """M-4: validate status at construction time."""
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"DisciplineCheck.status must be one of {sorted(_VALID_STATUSES)!r}, "
                f"got {self.status!r}"
            )

    def to_dict(self) -> dict:
        """Serialize to a dict; omit 'msg' key when msg is an empty string."""
        d: dict = {"id": self.id, "status": self.status}
        if self.msg:
            d["msg"] = self.msg
        return d


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Aggregated USD export validation verdict.

    Fields
    ------
    verdict     : str                   — 'pass', 'warn', or 'fail'
    checks      : list[DisciplineCheck] — ordered list of individual check results
    wrote_files : bool                  — True if USD files were written (postwrite mode)

    Class methods
    -------------
    from_checks(checks, wrote_files=False)
        Construct a ValidationReport by aggregating a list of DisciplineCheck
        instances (fail > warn > pass; empty list -> 'pass').
    """

    verdict: str
    checks: list
    wrote_files: bool = False

    @classmethod
    def from_checks(
        cls,
        checks: list,
        wrote_files: bool = False,
    ) -> "ValidationReport":
        """Aggregate *checks* into a ValidationReport.

        Severity precedence: fail > warn > pass.
        An empty checks list produces verdict='pass'.

        Args:
            checks:      list of DisciplineCheck instances.
            wrote_files: propagated to the returned report's wrote_files field.

        Returns:
            ValidationReport with the aggregated verdict and the original checks list.
        """
        verdict = "pass"
        for check in checks:
            if check.status == "fail":
                verdict = "fail"
            elif check.status == "warn" and verdict != "fail":
                verdict = "warn"
        return cls(verdict=verdict, checks=checks, wrote_files=wrote_files)

    def to_dict(self) -> dict:
        """Serialize to a plain dict with exactly three top-level keys."""
        return {
            "verdict": self.verdict,
            "checks": [c.to_dict() for c in self.checks],
            "wrote_files": self.wrote_files,
        }


# ---------------------------------------------------------------------------
# ExportRequest
# ---------------------------------------------------------------------------

@dataclass
class ExportRequest:
    """Caller-facing USD export request.

    Fields
    ------
    node         : str       — Houdini LOP node path to export
    out_path     : str       — output file path (may include $HIP-style expandables)
    flatten      : bool      — True to flatten the stage to a single layer
    default_prim : str | None — override for the layer's defaultPrim opinion
    """

    node: str
    out_path: str
    flatten: bool = False
    default_prim: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to a plain dict with all four fields present."""
        return {
            "node": self.node,
            "out_path": self.out_path,
            "flatten": self.flatten,
            "default_prim": self.default_prim,
        }

    @staticmethod
    def from_dict(d: dict) -> "ExportRequest":
        """Reconstruct an ExportRequest from a serialized dict."""
        return ExportRequest(
            node=d["node"],
            out_path=d["out_path"],
            flatten=d.get("flatten", False),
            default_prim=d.get("default_prim"),
        )
