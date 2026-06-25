"""
export_model.py — pure-logic dataclasses for the fxhoudinimcp export pipeline.

Contract: imports NO hou, NO Qt/PySide6, NO pxr. Plain Python stdlib only.
Pytest-able off-DCC (CL-015).

Classes
-------
BudgetCheck     — one budget validation check result
BudgetReport    — aggregated budget verdict + list of checks
VersionTriple   — houdini/labs_vat/ue compatibility strings
ExportManifest  — full export sidecar (FR-8 §7.3)
ExportRequest   — caller-facing request to the export handler

Functions
---------
verdict_from_checks(checks) -> str   — aggregate checks -> "pass"/"warn"/"fail"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_STATUSES = {"pass", "warn", "fail"}


def verdict_from_checks(checks: list) -> str:  # list[BudgetCheck] -> str
    """Aggregate a list of BudgetCheck results into a single verdict string.

    Precedence (highest wins):
      "fail"  — any check has status "fail"
      "warn"  — any check has status "warn" (and none are "fail")
      "pass"  — all checks are "pass" (or the list is empty)

    Args:
        checks: list of BudgetCheck instances.

    Returns:
        One of "fail", "warn", or "pass".

    Raises:
        ValueError: if any check has a status not in {"pass", "warn", "fail"}.
    """
    result = "pass"
    for check in checks:
        if check.status not in _VALID_STATUSES:
            raise ValueError(
                f"Unknown status {check.status!r}; must be one of {sorted(_VALID_STATUSES)}"
            )
        if check.status == "fail":
            result = "fail"
        elif check.status == "warn" and result != "fail":
            result = "warn"
    return result


# ---------------------------------------------------------------------------
# BudgetCheck
# ---------------------------------------------------------------------------

@dataclass
class BudgetCheck:
    """A single budget validation check result.

    Fields
    ------
    id      : str            — unique check identifier
    status  : str            — "pass", "warn", or "fail"
    value   : float | None   — observed metric value (omitted when None)
    limit   : float | None   — threshold/limit (omitted when None)
    msg     : str | None     — short human-readable message (omitted when None)
    detail  : str | None     — extended detail text (omitted when None)
    """

    id: str
    status: str
    value: Optional[float] = None
    limit: Optional[float] = None
    msg: Optional[str] = None
    detail: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to a dict, omitting None-valued optional fields."""
        d: dict = {"id": self.id, "status": self.status}
        if self.value is not None:
            d["value"] = self.value
        if self.limit is not None:
            d["limit"] = self.limit
        if self.msg is not None:
            d["msg"] = self.msg
        if self.detail is not None:
            d["detail"] = self.detail
        return d

    @staticmethod
    def from_dict(d: dict) -> "BudgetCheck":
        """Reconstruct a BudgetCheck from a serialized dict."""
        return BudgetCheck(
            id=d["id"],
            status=d["status"],
            value=d.get("value"),
            limit=d.get("limit"),
            msg=d.get("msg"),
            detail=d.get("detail"),
        )


# ---------------------------------------------------------------------------
# BudgetReport
# ---------------------------------------------------------------------------

@dataclass
class BudgetReport:
    """Aggregated budget validation result.

    Fields
    ------
    verdict     : str              — "pass", "warn", or "fail"
    checks      : list[BudgetCheck]
    wrote_files : bool             — True if files were written during the check
    """

    verdict: str
    checks: list = field(default_factory=list)
    wrote_files: bool = False

    def to_dict(self) -> dict:
        """Serialize to a dict including verdict, checks, and wrote_files."""
        return {
            "verdict": self.verdict,
            "checks": [c.to_dict() for c in self.checks],
            "wrote_files": self.wrote_files,
        }

    @staticmethod
    def from_dict(d: dict) -> "BudgetReport":
        """Reconstruct a BudgetReport from a serialized dict.

        Raises
        ------
        ValueError
            If the dict does not contain a "checks" key.
        """
        if "checks" not in d:
            raise ValueError(
                "BudgetReport.from_dict: dict is missing required key 'checks'"
            )
        return BudgetReport(
            verdict=d["verdict"],
            checks=[BudgetCheck.from_dict(c) for c in d["checks"]],
            wrote_files=d.get("wrote_files", False),
        )


# ---------------------------------------------------------------------------
# VersionTriple
# ---------------------------------------------------------------------------

@dataclass
class VersionTriple:
    """Houdini / Labs VAT / UE compatibility version strings.

    Fields
    ------
    houdini  : str           — Houdini version string, e.g. "21.0.456"
    labs_vat : str | None    — SideFX Labs VAT version string (omitted when None)
    ue       : str | None    — Unreal Engine version string (omitted when None)
    verdict  : str           — compatibility verdict; defaults to "warn"
    notes    : list[str]     — explanatory notes (empty by default)
    """

    houdini: str
    labs_vat: Optional[str] = None
    ue: Optional[str] = None
    verdict: str = "warn"
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a dict, omitting labs_vat and ue when None."""
        d: dict = {
            "houdini": self.houdini,
            "verdict": self.verdict,
            "notes": list(self.notes),
        }
        if self.labs_vat is not None:
            d["labs_vat"] = self.labs_vat
        if self.ue is not None:
            d["ue"] = self.ue
        return d

    @staticmethod
    def from_dict(d: dict) -> "VersionTriple":
        """Reconstruct a VersionTriple from a serialized dict."""
        return VersionTriple(
            houdini=d["houdini"],
            labs_vat=d.get("labs_vat"),
            ue=d.get("ue"),
            verdict=d.get("verdict", "warn"),
            notes=list(d.get("notes", [])),
        )


# ---------------------------------------------------------------------------
# ExportManifest
# ---------------------------------------------------------------------------

@dataclass
class ExportManifest:
    """Full export sidecar per FR-8 §7.3.

    Fields
    ------
    tool            : str                        — exporter tool name
    args            : dict                       — exporter arguments
    out_paths       : list[str]                  — output file paths
    version_triple  : VersionTriple              — version compatibility info
    validator       : BudgetReport | dict        — budget validation result
    schema_version  : int                        — sidecar schema version (carried on round-trip)
    """

    tool: str
    args: dict
    out_paths: list
    version_triple: "VersionTriple"
    validator: object  # BudgetReport | dict
    schema_version: int = 1

    def to_dict(self) -> dict:
        """Serialize to a dict including schema_version."""
        if isinstance(self.validator, BudgetReport):
            validator_dict = self.validator.to_dict()
        else:
            validator_dict = dict(self.validator)

        return {
            "schema_version": self.schema_version,
            "tool": self.tool,
            "args": dict(self.args),
            "out_paths": list(self.out_paths),
            "version_triple": self.version_triple.to_dict(),
            "validator": validator_dict,
        }

    @staticmethod
    def from_dict(d: dict) -> "ExportManifest":
        """Reconstruct an ExportManifest from a serialized dict.

        The schema_version is CARRIED from the dict (CX-003: not reset to a
        class-level default), so a v2 dict round-trips as v2.

        Raises
        ------
        ValueError
            If the validator dict is present but does not contain a "checks"
            key (CX-004: fail-loud on malformed validator).
        """
        validator_raw = d.get("validator", {})
        if isinstance(validator_raw, dict):
            # Must have "checks" key to reconstruct as BudgetReport (CX-004)
            if "checks" not in validator_raw:
                raise ValueError(
                    "ExportManifest.from_dict: validator dict is missing required "
                    "key 'checks'; cannot reconstruct as BudgetReport"
                )
            validator = BudgetReport.from_dict(validator_raw)
        else:
            # Already a BudgetReport instance
            validator = validator_raw

        return ExportManifest(
            tool=d["tool"],
            args=dict(d.get("args", {})),
            out_paths=list(d.get("out_paths", [])),
            version_triple=VersionTriple.from_dict(d["version_triple"]),
            validator=validator,
            schema_version=d.get("schema_version", 1),
        )


# ---------------------------------------------------------------------------
# VAT export mode helpers
# ---------------------------------------------------------------------------

VAT_EXPORT_MODES: dict = {"soft": 0, "rigid": 1, "fluid": 2, "sprite": 3}


def vat_mode_from_export_type(export_type: str) -> int:
    """Convert a VAT export type name to the ROP mode integer (parm 'mode').

    Accepts any case and leading/trailing whitespace.
    Valid values: 'soft'->0, 'rigid'->1, 'fluid'->2, 'sprite'->3.

    Raises:
        ValueError: if export_type is not a recognized VAT mode name.
    """
    key = export_type.strip().lower()
    if key not in VAT_EXPORT_MODES:
        raise ValueError(f"unknown VAT export_type: {export_type!r}")
    return VAT_EXPORT_MODES[key]


# ---------------------------------------------------------------------------
# Frame-range helpers (PR-5 — Alembic-UE + FBX bake)
# ---------------------------------------------------------------------------

def resolve_frame_range(
    frame_range: object,
    default_start: int,
    default_end: int,
    default_inc: int = 1,
) -> tuple:
    """Resolve a caller-supplied frame_range into a (start, end, inc) tuple of ints.

    Args:
        frame_range:   None  -> use defaults.
                       list of 2 ints -> (start, end, default_inc).
                       list of 3 ints -> (start, end, inc).
                       Anything else  -> raises ValueError.
        default_start: int used when frame_range is None.
        default_end:   int used when frame_range is None.
        default_inc:   int used when frame_range is None or a 2-element list.

    Returns:
        (start, end, inc) — a tuple of exactly 3 ints.

    Raises:
        ValueError: for non-None non-list inputs, empty lists, lists with 1
                    element, lists with 4+ elements, or any element that cannot
                    be coerced to int.
    """
    if frame_range is None:
        return (int(default_start), int(default_end), int(default_inc))

    if not isinstance(frame_range, list):
        raise ValueError(
            f"frame_range must be None or a list, got {type(frame_range).__name__!r}: {frame_range!r}"
        )

    length = len(frame_range)
    if length < 2 or length > 3:
        raise ValueError(
            f"frame_range list must have 2 or 3 elements, got {length}: {frame_range!r}"
        )

    try:
        start = int(frame_range[0])
        end = int(frame_range[1])
        inc = int(frame_range[2]) if length == 3 else int(default_inc)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"frame_range elements must be coercible to int: {frame_range!r}"
        ) from exc

    return (start, end, inc)


def alembic_packed_transform_value(deforming: bool) -> int:
    """Return the rop_alembic 'packed_transform' parameter value for the given mode.

    Houdini rop_alembic packed_transform parm:
        0 = Deform Geometry  (write per-frame vertex positions; use when deforming=True)
        1 = Transform Geometry (write transform only; use when deforming=False)

    Args:
        deforming: True -> deform geometry (0); False -> transform only (1).

    Returns:
        int 0 or 1 (plain int, NOT bool — isinstance(result, bool) is False).
    """
    # Use integer literals explicitly so the return type is int, not bool.
    # bool is a subclass of int in Python: `True == 1` but `isinstance(True, bool)` is True.
    # The test asserts isinstance(result, bool) is False, so we must not return True/False.
    return 0 if deforming else 1


# ---------------------------------------------------------------------------
# ExportRequest
# ---------------------------------------------------------------------------

@dataclass
class ExportRequest:
    """Caller-facing request to the export handler.

    Fields
    ------
    node            : str    — Houdini node path
    target          : str    — export target identifier (e.g. "fbx", "usd")
    out_path_or_dir : str    — output file path or directory
    params          : dict   — optional extra parameters (defaults to {})
    """

    node: str
    target: str
    out_path_or_dir: str
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a dict (run_budget_checks removed in v2 contract)."""
        return {
            "node": self.node,
            "target": self.target,
            "out_path_or_dir": self.out_path_or_dir,
            "params": dict(self.params),
        }

    @staticmethod
    def from_dict(d: dict) -> "ExportRequest":
        """Reconstruct an ExportRequest from a serialized dict."""
        return ExportRequest(
            node=d["node"],
            target=d["target"],
            out_path_or_dir=d["out_path_or_dir"],
            params=dict(d.get("params", {})),
        )


# ---------------------------------------------------------------------------
# pp12-111f helpers — pure-logic, no hou / Qt / pxr
# ---------------------------------------------------------------------------


def gc_export_refusal(check) -> Optional[dict]:
    """Return None if check passes; return a refusal dict if check fails (FR-7).

    Used by export_chaos_gc to gate on check_gc_sequential BEFORE any hou
    mutation.  When the check passes the handler proceeds normally.  When the
    check fails the handler returns this dict immediately (no ROP created, no
    .abc written).

    Args:
        check: a BudgetCheck instance returned by budget_rules.check_gc_sequential.

    Returns:
        None if check.status == "pass".
        dict {"ok": False, "wrote_files": False, "error": <message>} on fail.
    """
    if check.status == "pass":
        return None
    # Build error string from check.detail (always set by check_gc_sequential on fail).
    error_msg = check.detail if check.detail else "GC piece IDs are not contiguous"
    if check.msg:
        error_msg = f"{error_msg} {check.msg}"
    return {"ok": False, "error": error_msg, "wrote_files": False}


def niagara_normalize_output(out_path: str) -> str:
    """Normalise out_path to always carry a .hbjson extension (idempotent).

    The labs::niagara_rop writes a Houdini Bgeo JSON (.hbjson) file.
    This helper ensures the caller-supplied path always ends in .hbjson
    regardless of whether they supplied no extension, .hbjson, or some
    other extension.

    Args:
        out_path: the caller-supplied output path string.

    Returns:
        out_path with extension replaced (or appended) to .hbjson.

    Examples:
        "x/out"        -> "x/out.hbjson"   (no extension: append)
        "x/out.hbjson" -> "x/out.hbjson"   (idempotent)
        "x/out.bgeo"   -> "x/out.hbjson"   (other extension: replace)
    """
    import os
    if out_path.endswith(".hbjson"):
        return out_path
    root, _ = os.path.splitext(out_path)
    return root + ".hbjson"
