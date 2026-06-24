"""
Tests for export_model.py — pure-logic layer.
PP12-111a round 2 — asserts the LOCKED contract from the approved plan and spec §7.3.

No hou / Qt / pxr imports anywhere in this file.  Runs under plain pytest
headless (off-DCC, no Houdini install required).

Covers the public contract of:
  - BudgetCheck           (dataclass)  — id, status, value, limit, msg, detail
  - BudgetReport          (dataclass)  — verdict (STORED), checks, wrote_files
  - verdict_from_checks   (function)   — fail>warn>pass; raises ValueError on unknown
  - VersionTriple         (dataclass)  — COMPATIBILITY triple: houdini, labs_vat, ue, verdict, notes
  - ExportManifest        (dataclass)  — FR-8 §7.3 sidecar: tool, args, out_paths (list), version_triple, validator
  - ExportRequest         (dataclass)  — node, target (required), out_path_or_dir, params

TDD phase: RED — locked contract does NOT match the current (wrong) export_model.py.
VersionTriple(houdini=...) raises TypeError against the current major= constructor.
ExportRequest(target=...) raises TypeError against the current impl (no target field).
ExportManifest(tool=...) raises TypeError against the current impl (wrong field set).
"""

from __future__ import annotations

import json
import sys
import os

# ---------------------------------------------------------------------------
# Path bootstrap — allow running as a standalone script as well as via pytest.
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_PKG_ROOT))

import pytest

# ---------------------------------------------------------------------------
# The module under test.
# ---------------------------------------------------------------------------
from fxhoudinimcp.export_model import (
    BudgetCheck,
    BudgetReport,
    VersionTriple,
    ExportManifest,
    ExportRequest,
    verdict_from_checks,
)


# ===========================================================================
# Section 1 — BudgetCheck dataclass
#
# Locked contract (plan pp12-111a dispatch):
#   BudgetCheck(id: str, status: str, value=None, limit=None, msg=None, detail=None)
#   Field name is `msg`, NOT `message`.
#   Minimal: BudgetCheck("tris", "warn").to_dict() == {"id": "tris", "status": "warn"}
#   Optional fields (value, limit, msg, detail) are OMITTED from dict when None.
# ===========================================================================

class TestBudgetCheck:
    """BudgetCheck dataclass — locked field names, serialization, optional omission."""

    def test_minimal_construction(self):
        """BudgetCheck is constructible with just id and status."""
        bc = BudgetCheck(id="tris", status="warn")
        assert bc.id == "tris"
        assert bc.status == "warn"

    def test_minimal_dict_has_exactly_two_keys(self):
        """BudgetCheck('tris','warn').to_dict() == {'id':'tris','status':'warn'} — exactly 2 keys."""
        bc = BudgetCheck("tris", "warn")
        d = bc.to_dict()
        assert d == {"id": "tris", "status": "warn"}, (
            f"Minimal BudgetCheck.to_dict() must equal {{'id':'tris','status':'warn'}}, got {d!r}"
        )

    def test_msg_field_name_not_message(self):
        """The human-note field is named 'msg', not 'message'."""
        bc = BudgetCheck(id="tris", status="pass", msg="Near budget")
        assert hasattr(bc, "msg"), "BudgetCheck must have a 'msg' field (not 'message')"
        assert bc.msg == "Near budget"

    def test_detail_field_present(self):
        """BudgetCheck must have a 'detail' field."""
        bc = BudgetCheck(id="tris", status="warn", detail="See node /obj/geo1")
        assert hasattr(bc, "detail"), "BudgetCheck must have a 'detail' field"
        assert bc.detail == "See node /obj/geo1"

    def test_full_construction(self):
        """BudgetCheck accepts all optional fields including msg and detail."""
        bc = BudgetCheck(
            id="triangle_count",
            status="warn",
            value=85_000,
            limit=100_000,
            msg="Approaching budget",
            detail="mesh has 85k tris; reduce subdivisions",
        )
        assert bc.value == 85_000
        assert bc.limit == 100_000
        assert bc.msg == "Approaching budget"
        assert bc.detail == "mesh has 85k tris; reduce subdivisions"

    def test_to_dict_omits_none_value(self):
        """'value' key must be absent from dict when value=None."""
        bc = BudgetCheck(id="tri_count", status="pass", value=None)
        d = bc.to_dict()
        assert "value" not in d

    def test_to_dict_omits_none_limit(self):
        """'limit' key must be absent from dict when limit=None."""
        bc = BudgetCheck(id="tri_count", status="pass", limit=None)
        d = bc.to_dict()
        assert "limit" not in d

    def test_to_dict_omits_none_msg(self):
        """'msg' key must be absent from dict when msg=None."""
        bc = BudgetCheck(id="tri_count", status="pass", msg=None)
        d = bc.to_dict()
        assert "msg" not in d

    def test_to_dict_omits_none_detail(self):
        """'detail' key must be absent from dict when detail=None."""
        bc = BudgetCheck(id="tri_count", status="pass", detail=None)
        d = bc.to_dict()
        assert "detail" not in d

    def test_to_dict_includes_msg_when_set(self):
        """'msg' key present in dict when msg is not None."""
        bc = BudgetCheck(id="tri_count", status="fail", msg="Over budget by 15%")
        d = bc.to_dict()
        assert "msg" in d
        assert d["msg"] == "Over budget by 15%"

    def test_to_dict_includes_detail_when_set(self):
        """'detail' key present in dict when detail is not None."""
        bc = BudgetCheck(id="tri_count", status="warn", detail="tris=85k, limit=100k")
        d = bc.to_dict()
        assert "detail" in d
        assert d["detail"] == "tris=85k, limit=100k"

    def test_to_dict_includes_value_when_set(self):
        """'value' key present in dict when value is not None."""
        bc = BudgetCheck(id="tri_count", status="pass", value=42_000)
        d = bc.to_dict()
        assert d["value"] == 42_000

    def test_to_dict_includes_limit_when_set(self):
        """'limit' key present in dict when limit is not None."""
        bc = BudgetCheck(id="tri_count", status="warn", limit=100_000)
        d = bc.to_dict()
        assert d["limit"] == 100_000

    def test_to_dict_is_json_serializable(self):
        """The dict produced by to_dict must be json.dumps-able without error."""
        bc = BudgetCheck(
            id="lod_count", status="warn", value=3, limit=4,
            msg="High LOD count", detail="check LOD node at /obj/lod1",
        )
        json.dumps(bc.to_dict())

    def test_round_trip_full(self):
        """from_dict(to_dict()) restores all fields for a fully-populated BudgetCheck."""
        bc = BudgetCheck(
            id="vertex_count",
            status="warn",
            value=75_000,
            limit=100_000,
            msg="Near vertex limit",
            detail="reduce verts in /obj/hero",
        )
        rt = BudgetCheck.from_dict(bc.to_dict())
        assert rt.id == bc.id
        assert rt.status == bc.status
        assert rt.value == bc.value
        assert rt.limit == bc.limit
        assert rt.msg == bc.msg
        assert rt.detail == bc.detail

    def test_round_trip_minimal(self):
        """from_dict(to_dict()) restores a minimal BudgetCheck (id + status only)."""
        bc = BudgetCheck(id="material_count", status="pass")
        rt = BudgetCheck.from_dict(bc.to_dict())
        assert rt.id == bc.id
        assert rt.status == bc.status
        assert rt.value is None
        assert rt.limit is None
        assert rt.msg is None
        assert rt.detail is None

    def test_no_message_field(self):
        """BudgetCheck must NOT have a 'message' field — field was renamed to 'msg'.

        This is the shape-lock that makes the round-1 wrong impl RED.
        """
        bc = BudgetCheck(id="x", status="pass")
        assert not hasattr(bc, "message"), (
            "BudgetCheck must NOT have a 'message' field; "
            "the locked contract uses 'msg' (dispatch verbatim)"
        )


# ===========================================================================
# Section 2 — BudgetReport dataclass
#
# Locked contract:
#   BudgetReport(verdict: str, checks: list[BudgetCheck], wrote_files: bool = False)
#   'verdict' is STORED on the dataclass, not derived on access.
#   wrote_files defaults to False.
# ===========================================================================

class TestBudgetReport:
    """BudgetReport dataclass — verdict stored, checks, wrote_files default."""

    def _make_checks(self) -> list:
        return [
            BudgetCheck(id="tri_count", status="pass", value=50_000, limit=100_000),
            BudgetCheck(id="mat_count", status="warn", value=18, limit=20),
        ]

    def test_verdict_field_stored(self):
        """BudgetReport must accept and store a 'verdict' field."""
        report = BudgetReport(verdict="warn", checks=self._make_checks())
        assert report.verdict == "warn", (
            f"BudgetReport.verdict must be 'warn', got {report.verdict!r}"
        )

    def test_verdict_pass(self):
        """BudgetReport stores verdict='pass'."""
        report = BudgetReport(verdict="pass", checks=[])
        assert report.verdict == "pass"

    def test_verdict_fail(self):
        """BudgetReport stores verdict='fail'."""
        report = BudgetReport(
            verdict="fail",
            checks=[BudgetCheck(id="x", status="fail")],
        )
        assert report.verdict == "fail"

    def test_wrote_files_defaults_false(self):
        """BudgetReport.wrote_files must default to False."""
        report = BudgetReport(verdict="pass", checks=[])
        assert report.wrote_files is False, (
            f"BudgetReport.wrote_files must default to False, got {report.wrote_files!r}"
        )

    def test_stores_checks_list(self):
        """BudgetReport stores the list of BudgetCheck objects."""
        checks = self._make_checks()
        report = BudgetReport(verdict="warn", checks=checks)
        assert len(report.checks) == 2

    def test_to_dict_contains_verdict_and_checks_and_wrote_files(self):
        """to_dict() must contain 'verdict', 'checks', and 'wrote_files' keys."""
        report = BudgetReport(verdict="warn", checks=self._make_checks())
        d = report.to_dict()
        assert "verdict" in d, "BudgetReport.to_dict() must include 'verdict'"
        assert "checks" in d
        assert "wrote_files" in d

    def test_to_dict_verdict_stored(self):
        """to_dict()['verdict'] equals the stored verdict string."""
        report = BudgetReport(verdict="fail", checks=[])
        d = report.to_dict()
        assert d["verdict"] == "fail"

    def test_to_dict_wrote_files_false_by_default(self):
        """to_dict()['wrote_files'] is False when the field was not explicitly set."""
        report = BudgetReport(verdict="pass", checks=[])
        d = report.to_dict()
        assert d["wrote_files"] is False

    def test_to_dict_wrote_files_true_when_set(self):
        """to_dict()['wrote_files'] is True when explicitly passed True."""
        report = BudgetReport(verdict="pass", checks=[], wrote_files=True)
        d = report.to_dict()
        assert d["wrote_files"] is True

    def test_to_dict_checks_is_list_of_dicts(self):
        """to_dict()['checks'] is a list of dicts (not BudgetCheck objects)."""
        report = BudgetReport(verdict="warn", checks=self._make_checks())
        d = report.to_dict()
        assert isinstance(d["checks"], list)
        for item in d["checks"]:
            assert isinstance(item, dict)

    def test_to_dict_is_json_serializable(self):
        """to_dict() output must be json.dumps-able without error."""
        report = BudgetReport(
            verdict="warn", checks=self._make_checks(), wrote_files=True
        )
        json.dumps(report.to_dict())

    def test_round_trip(self):
        """from_dict(to_dict()) restores BudgetReport fields correctly."""
        checks = self._make_checks()
        report = BudgetReport(verdict="warn", checks=checks, wrote_files=True)
        rt = BudgetReport.from_dict(report.to_dict())
        assert rt.verdict == "warn"
        assert len(rt.checks) == len(checks)
        assert rt.wrote_files is True
        assert rt.checks[0].id == checks[0].id
        assert rt.checks[1].status == checks[1].status

    def test_round_trip_empty_checks(self):
        """from_dict(to_dict()) round-trips a BudgetReport with no checks."""
        report = BudgetReport(verdict="pass", checks=[])
        rt = BudgetReport.from_dict(report.to_dict())
        assert rt.verdict == "pass"
        assert rt.checks == []
        assert rt.wrote_files is False


# ===========================================================================
# Section 3 — verdict_from_checks(checks) -> str
#
# Locked contract — five pinned tests from dispatch:
#   all-pass  → "pass"
#   one-warn  → "warn"
#   one-fail  → "fail"
#   empty     → "pass"
#   status="error" → pytest.raises(ValueError)
# ===========================================================================

class TestVerdictFromChecks:
    """verdict_from_checks — five pinned tests + additional precedence coverage."""

    def test_empty_list_returns_pass(self):
        """PIN 1: No checks → verdict is 'pass'."""
        result = verdict_from_checks([])
        assert result == "pass", (
            f"verdict_from_checks([]) must return 'pass', got {result!r}"
        )

    def test_all_pass_returns_pass(self):
        """PIN 2: All 'pass' checks → verdict is 'pass'."""
        checks = [
            BudgetCheck(id="a", status="pass"),
            BudgetCheck(id="b", status="pass"),
        ]
        result = verdict_from_checks(checks)
        assert result == "pass"

    def test_any_warn_no_fail_returns_warn(self):
        """PIN 3: One 'warn' among 'pass' checks → verdict is 'warn'."""
        checks = [
            BudgetCheck(id="a", status="pass"),
            BudgetCheck(id="b", status="warn"),
        ]
        result = verdict_from_checks(checks)
        assert result == "warn"

    def test_any_fail_returns_fail(self):
        """PIN 4: One 'fail' among 'warn' and 'pass' checks → verdict is 'fail'."""
        checks = [
            BudgetCheck(id="a", status="pass"),
            BudgetCheck(id="b", status="warn"),
            BudgetCheck(id="c", status="fail"),
        ]
        result = verdict_from_checks(checks)
        assert result == "fail"

    def test_unknown_status_error_raises_value_error(self):
        """PIN 5 (CX-002): status='error' must raise ValueError.

        The current impl silently aggregates unknown statuses as 'pass'.
        This pinned test proves the locked contract requires ValueError.
        """
        checks = [BudgetCheck(id="tri_count", status="error")]
        with pytest.raises(ValueError):
            verdict_from_checks(checks)

    def test_fail_beats_warn(self):
        """'fail' has higher precedence than 'warn' — mixed list returns 'fail'."""
        checks = [
            BudgetCheck(id="x", status="warn"),
            BudgetCheck(id="y", status="fail"),
        ]
        result = verdict_from_checks(checks)
        assert result == "fail"

    def test_single_fail_check(self):
        """A list of one 'fail' check returns 'fail'."""
        checks = [BudgetCheck(id="tri_count", status="fail", value=200_000, limit=100_000)]
        result = verdict_from_checks(checks)
        assert result == "fail"

    def test_single_warn_check(self):
        """A list of one 'warn' check returns 'warn'."""
        checks = [BudgetCheck(id="lod_count", status="warn", value=3, limit=4)]
        result = verdict_from_checks(checks)
        assert result == "warn"

    def test_single_pass_check(self):
        """A list of one 'pass' check returns 'pass'."""
        checks = [BudgetCheck(id="mat_count", status="pass", value=5, limit=20)]
        result = verdict_from_checks(checks)
        assert result == "pass"

    def test_returns_string(self):
        """verdict_from_checks always returns a str, never None or a bool."""
        result = verdict_from_checks([])
        assert isinstance(result, str)

    def test_unknown_status_typo_raises(self):
        """A typo like 'passs' also raises ValueError."""
        checks = [BudgetCheck(id="tri_count", status="passs")]
        with pytest.raises(ValueError):
            verdict_from_checks(checks)

    def test_unknown_status_future_value_raises(self):
        """A plausible-future status value 'info' raises ValueError."""
        checks = [BudgetCheck(id="a", status="info")]
        with pytest.raises(ValueError):
            verdict_from_checks(checks)

    def test_empty_string_status_raises(self):
        """An empty-string status raises ValueError (not silently 'pass')."""
        checks = [BudgetCheck(id="b", status="")]
        with pytest.raises(ValueError):
            verdict_from_checks(checks)

    def test_known_statuses_still_work_after_hardening(self):
        """Known statuses must still work correctly after ValueError gate is added."""
        assert verdict_from_checks([BudgetCheck(id="a", status="pass")]) == "pass"
        assert verdict_from_checks([BudgetCheck(id="b", status="warn")]) == "warn"
        assert verdict_from_checks([BudgetCheck(id="c", status="fail")]) == "fail"


# ===========================================================================
# Section 4 — VersionTriple dataclass
#
# Locked contract (COMPATIBILITY triple, NOT semver):
#   VersionTriple(houdini: str, labs_vat: str|None=None, ue: str|None=None,
#                 verdict: str="warn", notes: list[str]=field(default_factory=list))
#   Constructor: VersionTriple(houdini="21.0.456", ...)
#   Default: verdict="warn", notes==[]
#   The old shape VersionTriple(major=21) must NOT be valid.
# ===========================================================================

class TestVersionTriple:
    """VersionTriple — COMPATIBILITY triple with houdini= constructor."""

    def test_houdini_constructor(self):
        """VersionTriple is constructed with houdini=str, not major=int."""
        vt = VersionTriple(houdini="21.0.456")
        assert vt.houdini == "21.0.456"

    def test_default_verdict_is_warn(self):
        """VersionTriple.verdict defaults to 'warn'."""
        vt = VersionTriple(houdini="21.0.456")
        assert vt.verdict == "warn", (
            f"VersionTriple.verdict must default to 'warn', got {vt.verdict!r}"
        )

    def test_default_notes_is_empty_list(self):
        """VersionTriple.notes defaults to []."""
        vt = VersionTriple(houdini="21.0.456")
        assert vt.notes == [], (
            f"VersionTriple.notes must default to [], got {vt.notes!r}"
        )

    def test_labs_vat_optional(self):
        """VersionTriple.labs_vat is None by default."""
        vt = VersionTriple(houdini="21.0.456")
        assert vt.labs_vat is None

    def test_ue_optional(self):
        """VersionTriple.ue is None by default."""
        vt = VersionTriple(houdini="21.0.456")
        assert vt.ue is None

    def test_full_construction(self):
        """VersionTriple is constructible with all fields."""
        vt = VersionTriple(
            houdini="21.0.456",
            labs_vat="3.0.1",
            ue="5.4.0",
            verdict="ok",
            notes=["Labs VAT >= 3.0 required"],
        )
        assert vt.houdini == "21.0.456"
        assert vt.labs_vat == "3.0.1"
        assert vt.ue == "5.4.0"
        assert vt.verdict == "ok"
        assert vt.notes == ["Labs VAT >= 3.0 required"]

    def test_old_major_constructor_invalid(self):
        """VersionTriple(major=21) must raise TypeError — old semver shape is GONE.

        This is the primary shape-lock that makes the round-1 wrong impl RED.
        """
        with pytest.raises(TypeError):
            VersionTriple(major=21)  # type: ignore[call-arg]

    def test_to_dict_contains_houdini(self):
        """to_dict() always includes 'houdini' key."""
        vt = VersionTriple(houdini="21.0.729")
        d = vt.to_dict()
        assert "houdini" in d
        assert d["houdini"] == "21.0.729"

    def test_to_dict_contains_verdict(self):
        """to_dict() includes 'verdict' key."""
        vt = VersionTriple(houdini="21.0.456")
        d = vt.to_dict()
        assert "verdict" in d
        assert d["verdict"] == "warn"

    def test_to_dict_contains_notes(self):
        """to_dict() includes 'notes' list."""
        vt = VersionTriple(houdini="21.0.456", notes=["check 1"])
        d = vt.to_dict()
        assert "notes" in d
        assert d["notes"] == ["check 1"]

    def test_to_dict_omits_none_labs_vat(self):
        """'labs_vat' key absent from dict when labs_vat=None."""
        vt = VersionTriple(houdini="21.0.456")
        d = vt.to_dict()
        assert "labs_vat" not in d

    def test_to_dict_omits_none_ue(self):
        """'ue' key absent from dict when ue=None."""
        vt = VersionTriple(houdini="21.0.456")
        d = vt.to_dict()
        assert "ue" not in d

    def test_to_dict_includes_labs_vat_when_set(self):
        """'labs_vat' present in dict when set."""
        vt = VersionTriple(houdini="21.0.456", labs_vat="3.0.1")
        d = vt.to_dict()
        assert d["labs_vat"] == "3.0.1"

    def test_to_dict_includes_ue_when_set(self):
        """'ue' present in dict when set."""
        vt = VersionTriple(houdini="21.0.456", ue="5.4.0")
        d = vt.to_dict()
        assert d["ue"] == "5.4.0"

    def test_to_dict_is_json_serializable(self):
        """to_dict() must be json.dumps-able."""
        vt = VersionTriple(
            houdini="21.0.456", labs_vat="3.0.1", ue="5.4.0",
            verdict="ok", notes=["all good"],
        )
        json.dumps(vt.to_dict())

    def test_round_trip_full(self):
        """from_dict(to_dict()) round-trips a full VersionTriple."""
        vt = VersionTriple(
            houdini="21.0.456",
            labs_vat="3.0.1",
            ue="5.4.0",
            verdict="ok",
            notes=["note A", "note B"],
        )
        rt = VersionTriple.from_dict(vt.to_dict())
        assert rt.houdini == "21.0.456"
        assert rt.labs_vat == "3.0.1"
        assert rt.ue == "5.4.0"
        assert rt.verdict == "ok"
        assert rt.notes == ["note A", "note B"]

    def test_round_trip_minimal(self):
        """from_dict(to_dict()) round-trips a minimal VersionTriple."""
        vt = VersionTriple(houdini="21.0.456")
        rt = VersionTriple.from_dict(vt.to_dict())
        assert rt.houdini == "21.0.456"
        assert rt.labs_vat is None
        assert rt.ue is None
        assert rt.verdict == "warn"
        assert rt.notes == []

    def test_no_major_field(self):
        """VersionTriple must NOT have a 'major' field — semver shape is gone."""
        vt = VersionTriple(houdini="21.0.456")
        assert not hasattr(vt, "major"), (
            "VersionTriple must NOT have a 'major' field; "
            "locked contract uses houdini: str"
        )


# ===========================================================================
# Section 5 — ExportManifest dataclass (FR-8 §7.3 sidecar)
#
# Locked contract:
#   ExportManifest(
#       tool: str,
#       args: dict,
#       out_paths: list[str],         # LIST, e.g. ["hero.fbx","hero_pos.exr",...]
#       version_triple: VersionTriple,
#       validator: BudgetReport|dict,
#   )
#   to_dict() includes top-level schema_version (int).
#   out_paths is a LIST (not a single str).
#   from_dict(to_dict()) == m  (round-trip lossless).
#   schema_version CARRIED (v2 dict stays v2).
#   Malformed validator dict (lacking "checks") => from_dict raises ValueError.
# ===========================================================================

class TestExportManifest:
    """ExportManifest — FR-8 §7.3 sidecar shape, out_paths as list, schema_version carry."""

    def _make_version_triple(self) -> VersionTriple:
        return VersionTriple(
            houdini="21.0.729",
            labs_vat="3.0.1",
            ue="5.4.0",
            verdict="ok",
            notes=["all versions compatible"],
        )

    def _make_validator(self) -> BudgetReport:
        return BudgetReport(
            verdict="pass",
            checks=[
                BudgetCheck(id="tri_count", status="pass", value=60_000, limit=100_000),
            ],
            wrote_files=True,
        )

    def _make_manifest(self) -> ExportManifest:
        return ExportManifest(
            tool="labs_vat",
            args={"frame_range": [1, 100], "fps": 24},
            out_paths=["hero.fbx", "hero_pos.exr", "hero_rot.exr", "hero.vat.json"],
            version_triple=self._make_version_triple(),
            validator=self._make_validator(),
        )

    def test_construction(self):
        """ExportManifest is constructible with the locked field set."""
        m = self._make_manifest()
        assert m.tool == "labs_vat"
        assert isinstance(m.args, dict)

    def test_out_paths_is_list(self):
        """ExportManifest.out_paths must be a list, not a single string."""
        m = self._make_manifest()
        assert isinstance(m.out_paths, list), (
            f"ExportManifest.out_paths must be a list; got {type(m.out_paths)!r}"
        )

    def test_out_paths_contents(self):
        """out_paths contains the 4 expected output files."""
        m = self._make_manifest()
        assert "hero.fbx" in m.out_paths
        assert "hero_pos.exr" in m.out_paths
        assert "hero_rot.exr" in m.out_paths
        assert "hero.vat.json" in m.out_paths

    def test_to_dict_contains_schema_version(self):
        """to_dict() MUST include a top-level 'schema_version' int key."""
        m = self._make_manifest()
        d = m.to_dict()
        assert "schema_version" in d, (
            "ExportManifest.to_dict() must include 'schema_version'"
        )
        assert isinstance(d["schema_version"], int)

    def test_to_dict_contains_locked_keys(self):
        """to_dict() contains tool, args, out_paths, version_triple, validator."""
        m = self._make_manifest()
        d = m.to_dict()
        assert "tool" in d
        assert "args" in d
        assert "out_paths" in d
        assert "version_triple" in d
        assert "validator" in d

    def test_to_dict_out_paths_is_list(self):
        """to_dict()['out_paths'] is a list."""
        m = self._make_manifest()
        d = m.to_dict()
        assert isinstance(d["out_paths"], list)

    def test_to_dict_no_old_fields(self):
        """to_dict() must NOT contain old wrong field names: houdini_version, node_path, export_path."""
        m = self._make_manifest()
        d = m.to_dict()
        assert "houdini_version" not in d, (
            "ExportManifest.to_dict() must not have 'houdini_version'; "
            "locked contract uses 'version_triple'"
        )
        assert "node_path" not in d, "ExportManifest must not have 'node_path'"
        assert "export_path" not in d, "ExportManifest must not have 'export_path'"

    def test_to_dict_is_json_serializable(self):
        """to_dict() must be json.dumps-able without error."""
        m = self._make_manifest()
        json.dumps(m.to_dict())

    def test_round_trip(self):
        """from_dict(to_dict()) == m (round-trip lossless)."""
        m = self._make_manifest()
        rt = ExportManifest.from_dict(m.to_dict())
        assert rt.tool == m.tool
        assert rt.args == m.args
        assert rt.out_paths == m.out_paths
        assert rt.version_triple.houdini == "21.0.729"
        assert rt.version_triple.verdict == "ok"
        assert isinstance(rt.validator, BudgetReport)
        assert rt.validator.verdict == "pass"
        assert rt.validator.wrote_files is True
        assert len(rt.validator.checks) == 1

    def test_schema_version_carried_v2(self):
        """schema_version=2 in a dict must survive from_dict (CX-003).

        This pins the CX-003 fix: a forward-compat dict must NOT be silently
        downgraded to the class default.
        """
        d = self._make_manifest().to_dict()
        d["schema_version"] = 2
        rt = ExportManifest.from_dict(d)
        d_out = rt.to_dict()
        assert d_out["schema_version"] == 2, (
            f"schema_version=2 was lost in round-trip; got {d_out['schema_version']!r}. "
            "from_dict must carry the stored schema_version."
        )

    def test_schema_version_carried_v1(self):
        """schema_version=1 also survives round-trip (regression guard)."""
        m = self._make_manifest()
        d_in = m.to_dict()
        assert d_in["schema_version"] == 1  # baseline
        rt = ExportManifest.from_dict(d_in)
        assert rt.to_dict()["schema_version"] == 1

    def test_malformed_validator_missing_checks_raises(self):
        """from_dict raises ValueError when validator dict lacks 'checks' (CX-004 / F-001).

        This pins the CX-004 fix: fail-loud at from_dict(), not silently at use time.
        """
        d = self._make_manifest().to_dict()
        d["validator"] = {"wrote_files": True}   # missing "checks"
        with pytest.raises(ValueError):
            ExportManifest.from_dict(d)

    def test_malformed_validator_entirely_wrong_raises(self):
        """A validator dict with no recognized keys also raises ValueError."""
        d = self._make_manifest().to_dict()
        d["validator"] = {"unknown_key": 42}
        with pytest.raises(ValueError):
            ExportManifest.from_dict(d)

    def test_well_formed_validator_still_works(self):
        """A well-formed validator dict (has 'checks') does NOT raise (regression guard)."""
        d = self._make_manifest().to_dict()
        d["validator"] = {"verdict": "pass", "checks": [], "wrote_files": False}
        result = ExportManifest.from_dict(d)
        assert result.validator.wrote_files is False

    def test_wrong_constructor_fields_raise(self):
        """Constructing ExportManifest with old fields (houdini_version=, node_path=) raises TypeError.

        This is the shape-lock that makes the round-1 wrong impl RED.
        """
        with pytest.raises(TypeError):
            ExportManifest(  # type: ignore[call-arg]
                houdini_version=VersionTriple(houdini="21.0.456"),
                labs_vat_version=VersionTriple(houdini="3.0.1"),
                ue_version=None,
                budget_report=BudgetReport(verdict="pass", checks=[]),
                export_path="/tmp/hero.fbx",
                node_path="/obj/hero",
            )


# ===========================================================================
# Section 6 — ExportRequest dataclass
#
# Locked contract:
#   ExportRequest(node: str, target: str, out_path_or_dir: str, params: dict={})
#   'target' is required; valid values: vat|alembic_ue|fbx|niagara|chaos_gc
#   params defaults to {} (not missing, not None)
#   target is preserved in round-trip
# ===========================================================================

class TestExportRequest:
    """ExportRequest — target required, params defaults {}, locked field set."""

    def _make_request(self, target: str = "fbx", params: dict | None = None) -> ExportRequest:
        kw: dict = dict(
            node="/obj/hero_character",
            target=target,
            out_path_or_dir="/tmp/exports/",
        )
        if params is not None:
            kw["params"] = params
        return ExportRequest(**kw)

    def test_construction_with_target(self):
        """ExportRequest is constructible with node, target, out_path_or_dir."""
        req = self._make_request(target="fbx")
        assert req.node == "/obj/hero_character"
        assert req.target == "fbx"
        assert req.out_path_or_dir == "/tmp/exports/"

    def test_target_required(self):
        """ExportRequest without target raises TypeError — target is a required field."""
        with pytest.raises(TypeError):
            ExportRequest(  # type: ignore[call-arg]
                node="/obj/hero",
                out_path_or_dir="/tmp/exports/",
                # target intentionally omitted
            )

    def test_params_defaults_empty_dict(self):
        """ExportRequest.params defaults to {} (empty dict, not None or missing)."""
        req = self._make_request()
        assert req.params == {}, (
            f"ExportRequest.params must default to {{}}, got {req.params!r}"
        )

    def test_params_settable(self):
        """ExportRequest.params can be set to a non-empty dict."""
        req = self._make_request(params={"frame_range": [1, 100], "fps": 24})
        assert req.params["fps"] == 24

    def test_target_vat(self):
        """ExportRequest target='vat' is accepted."""
        req = self._make_request(target="vat")
        assert req.target == "vat"

    def test_target_alembic_ue(self):
        """ExportRequest target='alembic_ue' is accepted."""
        req = self._make_request(target="alembic_ue")
        assert req.target == "alembic_ue"

    def test_target_niagara(self):
        """ExportRequest target='niagara' is accepted."""
        req = self._make_request(target="niagara")
        assert req.target == "niagara"

    def test_target_chaos_gc(self):
        """ExportRequest target='chaos_gc' is accepted."""
        req = self._make_request(target="chaos_gc")
        assert req.target == "chaos_gc"

    def test_to_dict_contains_target(self):
        """to_dict() includes 'target' key."""
        req = self._make_request(target="vat")
        d = req.to_dict()
        assert "target" in d, "ExportRequest.to_dict() must include 'target'"
        assert d["target"] == "vat"

    def test_to_dict_contains_node(self):
        """to_dict() includes 'node' key."""
        req = self._make_request()
        d = req.to_dict()
        assert "node" in d
        assert d["node"] == "/obj/hero_character"

    def test_to_dict_contains_out_path_or_dir(self):
        """to_dict() includes 'out_path_or_dir' key."""
        req = self._make_request()
        d = req.to_dict()
        assert "out_path_or_dir" in d

    def test_to_dict_contains_params(self):
        """to_dict() includes 'params' key."""
        req = self._make_request()
        d = req.to_dict()
        assert "params" in d

    def test_to_dict_no_old_fields(self):
        """to_dict() must NOT contain old wrong field names: node_path, export_path, houdini_version."""
        req = self._make_request()
        d = req.to_dict()
        assert "node_path" not in d, (
            "ExportRequest.to_dict() must not have 'node_path'; "
            "locked contract uses 'node'"
        )
        assert "export_path" not in d, (
            "ExportRequest.to_dict() must not have 'export_path'; "
            "locked contract uses 'out_path_or_dir'"
        )
        assert "houdini_version" not in d
        assert "run_budget_checks" not in d

    def test_to_dict_is_json_serializable(self):
        """to_dict() must be json.dumps-able without error."""
        req = self._make_request(params={"fps": 24})
        json.dumps(req.to_dict())

    def test_round_trip_target_preserved(self):
        """target survives from_dict(to_dict()) round-trip."""
        req = self._make_request(target="vat", params={"frame_range": [1, 100]})
        rt = ExportRequest.from_dict(req.to_dict())
        assert rt.node == req.node
        assert rt.target == "vat"
        assert rt.out_path_or_dir == req.out_path_or_dir
        assert rt.params == {"frame_range": [1, 100]}

    def test_round_trip_params_empty(self):
        """params={} survives round-trip."""
        req = self._make_request()
        rt = ExportRequest.from_dict(req.to_dict())
        assert rt.params == {}

    def test_wrong_constructor_fields_raise(self):
        """Constructing ExportRequest with old fields (node_path=, export_path=) raises TypeError.

        This is the shape-lock that makes the round-1 wrong impl RED.
        """
        with pytest.raises(TypeError):
            ExportRequest(  # type: ignore[call-arg]
                node_path="/obj/hero",
                export_path="/tmp/hero.fbx",
                houdini_version=VersionTriple(houdini="21.0.456"),
                labs_vat_version=VersionTriple(houdini="3.0.1"),
            )


# ===========================================================================
# Section 7 — hou-free import verification (CL-015)
#
# export_model.py must import with zero hou/Qt/pxr at module top-level.
# This test proves it by confirming the module loads under plain pytest
# (no Houdini environment).
# ===========================================================================

class TestHouFreeImport:
    """Confirm export_model.py carries no hou/Qt/pxr dependency."""

    def test_module_importable_without_hou(self):
        """export_model must load under plain Python with no hou installed."""
        import fxhoudinimcp.export_model as em
        assert em is not None

    def test_hou_not_in_export_model_imports(self):
        """export_model module must not reference 'hou' as a top-level import."""
        import fxhoudinimcp.export_model as em
        import inspect
        import re
        source = inspect.getsource(em)
        source_no_comments = re.sub(r"#.*", "", source)
        assert "import hou" not in source_no_comments, (
            "export_model.py must not import hou (CL-015 — pure-logic boundary)"
        )


# ===========================================================================
# Section 8 — vat_mode_from_export_type(export_type: str) -> int
#
# PP12-111d (PR-4) — new pure helper in export_model.py.
#
# Locked contract:
#   vat_mode_from_export_type("soft")   == 0   (VAT ROP mode int, parm 'mode')
#   vat_mode_from_export_type("rigid")  == 1
#   vat_mode_from_export_type("fluid")  == 2
#   vat_mode_from_export_type("sprite") == 3
#
#   Case-normalised: "Soft" -> 0, " SOFT " -> 0 (strip + casefold)
#   Unknown string raises ValueError.
#
#   Optional: export_model exposes VAT_EXPORT_MODES mapping with 4 keys.
#
# TDD phase: RED — vat_mode_from_export_type does NOT exist yet in export_model.
# Expected failure: ImportError on import of the symbol.
#
# Cross-references:
#   - Plan pp12-111d §7.1: GROUNDED FIELD TABLE, parm 'mode' (int 0/1/2/3)
#   - CL-015: pure-logic module, no hou/Qt/pxr
#   - tdd-with-agents.md §4: hou-test writes red; hou-dev turns green
# ===========================================================================

class TestVatModeFromExportType:
    """vat_mode_from_export_type — string -> ROP mode int mapping, case/space normalised."""

    @pytest.fixture(autouse=True)
    def _import_subject(self):
        """Import vat_mode_from_export_type from export_model (must fail at RED phase)."""
        from fxhoudinimcp.export_model import vat_mode_from_export_type
        self._fn = vat_mode_from_export_type

    # -----------------------------------------------------------------------
    # Four canonical mappings (the locked enum: mode parm 0..3)
    # -----------------------------------------------------------------------

    def test_soft_returns_0(self):
        """vat_mode_from_export_type('soft') == 0 (Soft Body, ROP mode=0)."""
        assert self._fn("soft") == 0, (
            "vat_mode_from_export_type('soft') must return 0 (Labs VAT ROP mode=0 Soft Body)"
        )

    def test_rigid_returns_1(self):
        """vat_mode_from_export_type('rigid') == 1 (Rigid Body, ROP mode=1)."""
        assert self._fn("rigid") == 1, (
            "vat_mode_from_export_type('rigid') must return 1 (Labs VAT ROP mode=1 Rigid Body)"
        )

    def test_fluid_returns_2(self):
        """vat_mode_from_export_type('fluid') == 2 (Fluid, ROP mode=2)."""
        assert self._fn("fluid") == 2, (
            "vat_mode_from_export_type('fluid') must return 2 (Labs VAT ROP mode=2 Fluid)"
        )

    def test_sprite_returns_3(self):
        """vat_mode_from_export_type('sprite') == 3 (Sprite, ROP mode=3)."""
        assert self._fn("sprite") == 3, (
            "vat_mode_from_export_type('sprite') must return 3 (Labs VAT ROP mode=3 Sprite)"
        )

    # -----------------------------------------------------------------------
    # Return-type contract
    # -----------------------------------------------------------------------

    def test_returns_int(self):
        """Return value must be a plain Python int, not a str or float."""
        result = self._fn("soft")
        assert isinstance(result, int), (
            f"vat_mode_from_export_type must return int, got {type(result).__name__!r}"
        )

    # -----------------------------------------------------------------------
    # Case / whitespace normalisation
    # -----------------------------------------------------------------------

    def test_title_case_soft_returns_0(self):
        """Case-normalised: 'Soft' -> 0 (title-case input accepted)."""
        assert self._fn("Soft") == 0, (
            "vat_mode_from_export_type must case-normalise: 'Soft' must return 0"
        )

    def test_upper_case_with_spaces_returns_0(self):
        """Case+whitespace normalised: ' SOFT ' -> 0 (stripped + casefolded)."""
        assert self._fn(" SOFT ") == 0, (
            "vat_mode_from_export_type must strip + casefold: ' SOFT ' must return 0"
        )

    def test_mixed_case_rigid(self):
        """'RIGID' -> 1 (upper-case variant accepted)."""
        assert self._fn("RIGID") == 1, (
            "vat_mode_from_export_type must case-normalise: 'RIGID' must return 1"
        )

    def test_mixed_case_fluid(self):
        """'Fluid' -> 2 (title-case variant accepted)."""
        assert self._fn("Fluid") == 2

    def test_mixed_case_sprite(self):
        """'SPRITE' -> 3 (upper-case variant accepted)."""
        assert self._fn("SPRITE") == 3

    def test_leading_trailing_whitespace_rigid(self):
        """'  rigid  ' -> 1 (whitespace stripped)."""
        assert self._fn("  rigid  ") == 1

    # -----------------------------------------------------------------------
    # Unknown input raises ValueError
    # -----------------------------------------------------------------------

    def test_unknown_string_raises_value_error(self):
        """Unknown export_type raises ValueError (not a silent default)."""
        with pytest.raises(ValueError):
            self._fn("bogus")

    def test_empty_string_raises_value_error(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError):
            self._fn("")

    def test_partial_match_raises(self):
        """'sof' (partial of 'soft') raises ValueError -- no prefix matching."""
        with pytest.raises(ValueError):
            self._fn("sof")

    def test_numeric_string_raises(self):
        """'0' raises ValueError -- numeric strings are not accepted."""
        with pytest.raises(ValueError):
            self._fn("0")

    def test_none_like_string_raises(self):
        """'none' raises ValueError -- unsupported mode name."""
        with pytest.raises(ValueError):
            self._fn("none")

    # -----------------------------------------------------------------------
    # Optional: VAT_EXPORT_MODES mapping exposed at module level
    # -----------------------------------------------------------------------

    def test_vat_export_modes_exposed_with_4_keys(self):
        """export_model exposes VAT_EXPORT_MODES dict-like with 4 canonical keys.

        Optional -- but if present it must have exactly the 4 mode names as keys.
        """
        import fxhoudinimcp.export_model as em
        if not hasattr(em, "VAT_EXPORT_MODES"):
            pytest.skip("VAT_EXPORT_MODES not present (optional attribute)")
        modes = em.VAT_EXPORT_MODES
        assert len(modes) == 4, (
            f"VAT_EXPORT_MODES must have 4 keys (soft/rigid/fluid/sprite), "
            f"got {len(modes)}: {list(modes.keys())}"
        )
        for name in ("soft", "rigid", "fluid", "sprite"):
            assert name in modes, (
                f"VAT_EXPORT_MODES missing key {name!r}; present: {list(modes.keys())}"
            )
