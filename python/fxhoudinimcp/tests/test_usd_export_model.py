"""
Tests for usd_export_model.py — pure-logic layer (PP12-112 PR-1).

TDD phase: RED — usd_export_model.py does NOT exist yet.
Expected failure: ModuleNotFoundError on import.

Contract: imports NO hou, NO Qt/PySide6, NO pxr, NO numpy, NO MaterialX.
Plain Python stdlib only.  Runs under the fork .venv with plain pytest
headless (off-DCC, no Houdini install required).

Covers the public contract of:
  - LayerSummary      (dataclass)  -- default_prim, root_prims, sublayers,
                                       current_format, has_mtlx_material, to_dict()
  - MtlxSummary       (dataclass)  -- nodegraphs, surface_nodes,
                                       inputs_with_abs_paths, validate_ok,
                                       validate_errors (default_factory=list), to_dict()
  - DisciplineCheck   (dataclass)  -- id, status ('pass'|'warn'|'fail'), msg='',
                                       __post_init__ raises ValueError on invalid status,
                                       to_dict()
  - ValidationReport  (dataclass)  -- verdict ('pass'|'warn'|'fail'), checks, wrote_files=False,
                                       to_dict(), classmethod from_checks(checks, wrote_files=False)
  - ExportRequest     (dataclass)  -- node, out_path, flatten=False, default_prim=None,
                                       to_dict() / from_dict()

Key fixes under test:
  B-1: MtlxSummary.validate_errors uses field(default_factory=list),
       NOT a bare default [] — two instances must not share the list.
  M-4: DisciplineCheck.__post_init__ raises ValueError on statuses other than
       'pass', 'warn', 'fail' (e.g. 'warning', 'FAIL', 'ok').

Cross-references:
  - Plan pp12-112a lockedFieldContract rev2 (BINDING)
  - tdd-with-agents.md §4: hou-test writes red; hou-dev turns green
  - CL-015: pure-logic module, no hou/Qt/pxr
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
# usd_export_model.py does not exist yet.
# ---------------------------------------------------------------------------
from fxhoudinimcp.usd_export_model import (
    LayerSummary,
    MtlxSummary,
    DisciplineCheck,
    ValidationReport,
    ExportRequest,
)


# ===========================================================================
# Section 1 — LayerSummary dataclass
# ===========================================================================

class TestLayerSummary:
    """LayerSummary dataclass — field names, types, and to_dict() shape."""

    def test_minimal_construction(self):
        """LayerSummary is constructible with all required fields."""
        ls = LayerSummary(
            default_prim="/asset",
            root_prims=["/asset"],
            sublayers=[],
            current_format="usdc",
            has_mtlx_material=False,
        )
        assert ls.default_prim == "/asset"
        assert ls.root_prims == ["/asset"]
        assert ls.sublayers == []
        assert ls.current_format == "usdc"
        assert ls.has_mtlx_material is False

    def test_default_prim_can_be_none(self):
        """default_prim is optional — None is a valid value."""
        ls = LayerSummary(
            default_prim=None,
            root_prims=[],
            sublayers=[],
            current_format="usda",
            has_mtlx_material=False,
        )
        assert ls.default_prim is None

    def test_to_dict_keys(self):
        """to_dict() returns exactly the five expected keys."""
        ls = LayerSummary(
            default_prim="/geo",
            root_prims=["/geo", "/mtl"],
            sublayers=["sub.usda"],
            current_format="usda",
            has_mtlx_material=True,
        )
        d = ls.to_dict()
        assert set(d.keys()) == {
            "default_prim",
            "root_prims",
            "sublayers",
            "current_format",
            "has_mtlx_material",
        }, f"Unexpected keys in LayerSummary.to_dict(): {set(d.keys())!r}"

    def test_to_dict_values_round_trip(self):
        """to_dict() preserves all field values faithfully."""
        ls = LayerSummary(
            default_prim="/geo",
            root_prims=["/geo", "/mtl"],
            sublayers=["sub.usda"],
            current_format="usda",
            has_mtlx_material=True,
        )
        d = ls.to_dict()
        assert d["default_prim"] == "/geo"
        assert d["root_prims"] == ["/geo", "/mtl"]
        assert d["sublayers"] == ["sub.usda"]
        assert d["current_format"] == "usda"
        assert d["has_mtlx_material"] is True

    def test_to_dict_none_default_prim(self):
        """to_dict() with default_prim=None preserves None."""
        ls = LayerSummary(
            default_prim=None,
            root_prims=[],
            sublayers=[],
            current_format="usdc",
            has_mtlx_material=False,
        )
        d = ls.to_dict()
        assert d["default_prim"] is None

    def test_to_dict_is_json_serialisable(self):
        """LayerSummary.to_dict() must be serialisable by json.dumps without error."""
        ls = LayerSummary(
            default_prim="/asset",
            root_prims=["/asset"],
            sublayers=[],
            current_format="usdc",
            has_mtlx_material=False,
        )
        # Must not raise
        json.dumps(ls.to_dict())


# ===========================================================================
# Section 2 — MtlxSummary dataclass
# ===========================================================================

class TestMtlxSummary:
    """MtlxSummary dataclass — field names, nested to_dict(), and B-1 default safety."""

    def test_minimal_construction(self):
        """MtlxSummary is constructible with all required positional fields."""
        ms = MtlxSummary(
            nodegraphs=["NG_base"],
            surface_nodes=["mtlxstandard_surface1"],
            inputs_with_abs_paths=[],
            validate_ok=True,
        )
        assert ms.nodegraphs == ["NG_base"]
        assert ms.surface_nodes == ["mtlxstandard_surface1"]
        assert ms.inputs_with_abs_paths == []
        assert ms.validate_ok is True

    def test_validate_errors_default_is_empty_list(self):
        """validate_errors defaults to [] when not supplied."""
        ms = MtlxSummary(
            nodegraphs=[],
            surface_nodes=[],
            inputs_with_abs_paths=[],
            validate_ok=True,
        )
        assert ms.validate_errors == []

    # B-1: mutable-default isolation
    def test_validate_errors_default_factory_isolation(self):
        """B-1: two MtlxSummary instances must NOT share the same validate_errors list.

        A bare class-level default (= []) would cause all instances to share one
        list object.  field(default_factory=list) creates a fresh list per instance.
        """
        ms1 = MtlxSummary(nodegraphs=[], surface_nodes=[], inputs_with_abs_paths=[], validate_ok=True)
        ms2 = MtlxSummary(nodegraphs=[], surface_nodes=[], inputs_with_abs_paths=[], validate_ok=True)
        ms1.validate_errors.append("err")
        assert ms2.validate_errors == [], (
            "B-1: validate_errors must use field(default_factory=list); "
            "mutating ms1.validate_errors must NOT affect ms2.validate_errors"
        )

    def test_to_dict_top_level_keys(self):
        """to_dict() has top-level keys: nodegraphs, surface_nodes, inputs_with_abs_paths, validate."""
        ms = MtlxSummary(
            nodegraphs=["NG_a"],
            surface_nodes=["surf1"],
            inputs_with_abs_paths=["/abs/tex.png"],
            validate_ok=False,
            validate_errors=["Missing nodegraph binding"],
        )
        d = ms.to_dict()
        assert set(d.keys()) == {
            "nodegraphs",
            "surface_nodes",
            "inputs_with_abs_paths",
            "validate",
        }, f"Unexpected top-level keys in MtlxSummary.to_dict(): {set(d.keys())!r}"

    def test_to_dict_validate_nested_shape(self):
        """to_dict()['validate'] is a dict with keys 'ok' and 'errors'."""
        ms = MtlxSummary(
            nodegraphs=[],
            surface_nodes=[],
            inputs_with_abs_paths=[],
            validate_ok=True,
            validate_errors=[],
        )
        d = ms.to_dict()
        assert "validate" in d
        v = d["validate"]
        assert isinstance(v, dict), f"to_dict()['validate'] must be a dict, got {type(v)!r}"
        assert set(v.keys()) == {"ok", "errors"}, (
            f"to_dict()['validate'] must have keys 'ok' and 'errors', got {set(v.keys())!r}"
        )

    def test_to_dict_validate_ok_and_errors(self):
        """to_dict()['validate']['ok'] matches validate_ok; ['errors'] matches validate_errors."""
        errors = ["No surface material found"]
        ms = MtlxSummary(
            nodegraphs=[],
            surface_nodes=[],
            inputs_with_abs_paths=[],
            validate_ok=False,
            validate_errors=errors,
        )
        d = ms.to_dict()
        assert d["validate"]["ok"] is False
        assert d["validate"]["errors"] == errors

    def test_to_dict_is_json_serialisable(self):
        """MtlxSummary.to_dict() must be serialisable by json.dumps without error."""
        ms = MtlxSummary(
            nodegraphs=["NG_base"],
            surface_nodes=["mtlxstandard_surface1"],
            inputs_with_abs_paths=[],
            validate_ok=True,
        )
        json.dumps(ms.to_dict())

    def test_to_dict_lists_preserved(self):
        """List fields are preserved faithfully in the serialised dict."""
        ms = MtlxSummary(
            nodegraphs=["NG_a", "NG_b"],
            surface_nodes=["surf1"],
            inputs_with_abs_paths=["C:/tex/albedo.png"],
            validate_ok=True,
        )
        d = ms.to_dict()
        assert d["nodegraphs"] == ["NG_a", "NG_b"]
        assert d["surface_nodes"] == ["surf1"]
        assert d["inputs_with_abs_paths"] == ["C:/tex/albedo.png"]


# ===========================================================================
# Section 3 — DisciplineCheck dataclass
# ===========================================================================

class TestDisciplineCheck:
    """DisciplineCheck dataclass — status validation (M-4) and to_dict() contract."""

    def test_pass_status_is_valid(self):
        """DisciplineCheck with status='pass' constructs without error."""
        dc = DisciplineCheck(id="no_world_wrapper", status="pass")
        assert dc.status == "pass"

    def test_warn_status_is_valid(self):
        """DisciplineCheck with status='warn' constructs without error."""
        dc = DisciplineCheck(id="abs_texture_paths", status="warn", msg="2 absolute paths found")
        assert dc.status == "warn"

    def test_fail_status_is_valid(self):
        """DisciplineCheck with status='fail' constructs without error."""
        dc = DisciplineCheck(id="default_prim_set", status="fail")
        assert dc.status == "fail"

    def test_msg_defaults_to_empty_string(self):
        """msg field defaults to '' when not supplied."""
        dc = DisciplineCheck(id="no_world_wrapper", status="pass")
        assert dc.msg == ""

    # M-4: status validation at construction time
    def test_invalid_status_warning_raises(self):
        """M-4: 'warning' is not a valid status — must raise ValueError at construction."""
        with pytest.raises(ValueError):
            DisciplineCheck(id="some_check", status="warning")

    def test_invalid_status_uppercase_fail_raises(self):
        """M-4: 'FAIL' (uppercase) is not a valid status — must raise ValueError."""
        with pytest.raises(ValueError):
            DisciplineCheck(id="some_check", status="FAIL")

    def test_invalid_status_ok_raises(self):
        """M-4: 'ok' is not a valid status — must raise ValueError."""
        with pytest.raises(ValueError):
            DisciplineCheck(id="some_check", status="ok")

    def test_invalid_status_empty_string_raises(self):
        """M-4: empty string is not a valid status — must raise ValueError."""
        with pytest.raises(ValueError):
            DisciplineCheck(id="some_check", status="")

    def test_to_dict_without_msg(self):
        """to_dict() with empty msg returns only {'id', 'status'} — 2 keys."""
        dc = DisciplineCheck(id="no_world_wrapper", status="pass")
        d = dc.to_dict()
        assert d == {"id": "no_world_wrapper", "status": "pass"}, (
            f"DisciplineCheck.to_dict() with empty msg must equal "
            f"{{'id':'no_world_wrapper','status':'pass'}}, got {d!r}"
        )

    def test_to_dict_with_msg(self):
        """to_dict() with non-empty msg returns {'id', 'status', 'msg'} — 3 keys."""
        dc = DisciplineCheck(id="default_prim_set", status="fail", msg="defaultPrim is not set")
        d = dc.to_dict()
        assert set(d.keys()) == {"id", "status", "msg"}, (
            f"DisciplineCheck.to_dict() with msg must have exactly 3 keys, got {set(d.keys())!r}"
        )
        assert d["msg"] == "defaultPrim is not set"

    def test_to_dict_empty_msg_omits_msg_key(self):
        """When msg='', the 'msg' key is absent from to_dict() output."""
        dc = DisciplineCheck(id="no_world_wrapper", status="pass", msg="")
        d = dc.to_dict()
        assert "msg" not in d, (
            f"to_dict() with empty msg must omit 'msg' key, but got keys {set(d.keys())!r}"
        )

    def test_to_dict_is_json_serialisable(self):
        """DisciplineCheck.to_dict() must be serialisable by json.dumps without error."""
        dc = DisciplineCheck(id="abs_texture_paths", status="warn", msg="found 1 abs path")
        json.dumps(dc.to_dict())


# ===========================================================================
# Section 4 — ValidationReport dataclass
# ===========================================================================

class TestValidationReport:
    """ValidationReport dataclass — construction, from_checks(), and to_dict()."""

    def _make_checks(self):
        return [
            DisciplineCheck(id="no_world_wrapper", status="pass"),
            DisciplineCheck(id="default_prim_set", status="pass"),
        ]

    def test_minimal_construction(self):
        """ValidationReport is constructible with verdict and checks."""
        checks = self._make_checks()
        vr = ValidationReport(verdict="pass", checks=checks)
        assert vr.verdict == "pass"
        assert vr.checks == checks

    def test_wrote_files_defaults_to_false(self):
        """wrote_files defaults to False."""
        vr = ValidationReport(verdict="pass", checks=[])
        assert vr.wrote_files is False

    def test_wrote_files_can_be_set_true(self):
        """wrote_files can be explicitly set to True."""
        vr = ValidationReport(verdict="pass", checks=[], wrote_files=True)
        assert vr.wrote_files is True

    # from_checks classmethod
    def test_from_checks_all_pass(self):
        """from_checks with all-pass checks returns verdict='pass'."""
        checks = [
            DisciplineCheck(id="no_world_wrapper", status="pass"),
            DisciplineCheck(id="default_prim_set", status="pass"),
        ]
        vr = ValidationReport.from_checks(checks)
        assert vr.verdict == "pass", (
            f"All-pass checks must produce verdict='pass', got {vr.verdict!r}"
        )
        assert vr.checks is checks

    def test_from_checks_with_warn_gives_warn(self):
        """from_checks with one warn and rest pass returns verdict='warn'."""
        checks = [
            DisciplineCheck(id="no_world_wrapper", status="pass"),
            DisciplineCheck(id="abs_texture_paths", status="warn", msg="1 abs path"),
        ]
        vr = ValidationReport.from_checks(checks)
        assert vr.verdict == "warn", (
            f"warn check in list must bubble to verdict='warn', got {vr.verdict!r}"
        )

    def test_from_checks_with_fail_gives_fail(self):
        """from_checks with a fail check returns verdict='fail' regardless of others."""
        checks = [
            DisciplineCheck(id="no_world_wrapper", status="pass"),
            DisciplineCheck(id="abs_texture_paths", status="warn"),
            DisciplineCheck(id="default_prim_set", status="fail"),
        ]
        vr = ValidationReport.from_checks(checks)
        assert vr.verdict == "fail", (
            f"fail check must dominate: verdict must be 'fail', got {vr.verdict!r}"
        )

    def test_from_checks_empty_list_gives_pass(self):
        """from_checks([]) returns verdict='pass' (no failures)."""
        vr = ValidationReport.from_checks([])
        assert vr.verdict == "pass"

    def test_from_checks_wrote_files_default_false(self):
        """from_checks without wrote_files kwarg gives wrote_files=False."""
        vr = ValidationReport.from_checks([])
        assert vr.wrote_files is False

    def test_from_checks_wrote_files_kwarg(self):
        """from_checks(checks, wrote_files=True) propagates to the report."""
        vr = ValidationReport.from_checks([], wrote_files=True)
        assert vr.wrote_files is True

    # to_dict
    def test_to_dict_keys(self):
        """to_dict() returns exactly {'verdict', 'checks', 'wrote_files'}."""
        vr = ValidationReport(verdict="pass", checks=[], wrote_files=False)
        d = vr.to_dict()
        assert set(d.keys()) == {"verdict", "checks", "wrote_files"}, (
            f"Unexpected to_dict() keys: {set(d.keys())!r}"
        )

    def test_to_dict_checks_serialised(self):
        """to_dict()['checks'] is a list of dicts produced by DisciplineCheck.to_dict()."""
        checks = [DisciplineCheck(id="no_world_wrapper", status="pass")]
        vr = ValidationReport(verdict="pass", checks=checks, wrote_files=False)
        d = vr.to_dict()
        assert isinstance(d["checks"], list)
        assert len(d["checks"]) == 1
        assert isinstance(d["checks"][0], dict)
        assert d["checks"][0] == {"id": "no_world_wrapper", "status": "pass"}

    def test_to_dict_is_json_serialisable(self):
        """ValidationReport.to_dict() must be serialisable by json.dumps without error."""
        checks = [
            DisciplineCheck(id="no_world_wrapper", status="pass"),
            DisciplineCheck(id="default_prim_set", status="fail", msg="not set"),
        ]
        vr = ValidationReport.from_checks(checks)
        json.dumps(vr.to_dict())

    def test_to_dict_verdict_matches_field(self):
        """to_dict()['verdict'] matches the stored verdict field."""
        vr = ValidationReport(verdict="warn", checks=[], wrote_files=False)
        assert vr.to_dict()["verdict"] == "warn"


# ===========================================================================
# Section 5 — ExportRequest dataclass
# ===========================================================================

class TestExportRequest:
    """ExportRequest dataclass — field names, defaults, to_dict/from_dict round-trip."""

    def test_minimal_construction(self):
        """ExportRequest is constructible with node and out_path only."""
        er = ExportRequest(node="/stage/lop_export", out_path="/tmp/out.usdc")
        assert er.node == "/stage/lop_export"
        assert er.out_path == "/tmp/out.usdc"

    def test_flatten_defaults_to_false(self):
        """flatten defaults to False."""
        er = ExportRequest(node="/n", out_path="/o.usda")
        assert er.flatten is False

    def test_default_prim_defaults_to_none(self):
        """default_prim defaults to None."""
        er = ExportRequest(node="/n", out_path="/o.usda")
        assert er.default_prim is None

    def test_optional_fields_can_be_set(self):
        """flatten and default_prim can be overridden."""
        er = ExportRequest(node="/n", out_path="/o.usda", flatten=True, default_prim="/asset")
        assert er.flatten is True
        assert er.default_prim == "/asset"

    def test_to_dict_contains_all_four_fields(self):
        """to_dict() contains all four fields regardless of default values."""
        er = ExportRequest(node="/n", out_path="/o.usda")
        d = er.to_dict()
        assert "node" in d
        assert "out_path" in d
        assert "flatten" in d
        assert "default_prim" in d

    def test_to_dict_values(self):
        """to_dict() round-trips all four field values."""
        er = ExportRequest(
            node="/stage/my_lop",
            out_path="$HIP/exports/asset.usda",
            flatten=True,
            default_prim="/asset",
        )
        d = er.to_dict()
        assert d["node"] == "/stage/my_lop"
        assert d["out_path"] == "$HIP/exports/asset.usda"
        assert d["flatten"] is True
        assert d["default_prim"] == "/asset"

    def test_to_dict_defaults(self):
        """to_dict() preserves False and None for default fields."""
        er = ExportRequest(node="/n", out_path="/o.usdc")
        d = er.to_dict()
        assert d["flatten"] is False
        assert d["default_prim"] is None

    def test_to_dict_is_json_serialisable(self):
        """ExportRequest.to_dict() must be serialisable by json.dumps without error."""
        er = ExportRequest(node="/n", out_path="/o.usdc", flatten=True, default_prim="/geo")
        json.dumps(er.to_dict())

    def test_from_dict_round_trip(self):
        """ExportRequest.from_dict(er.to_dict()) reconstructs an equal object (if from_dict exists)."""
        er = ExportRequest(
            node="/stage/my_lop",
            out_path="$HIP/out.usda",
            flatten=True,
            default_prim="/asset",
        )
        d = er.to_dict()
        if not hasattr(ExportRequest, "from_dict"):
            pytest.skip("from_dict is optional — not yet implemented")
        er2 = ExportRequest.from_dict(d)
        assert er2.node == er.node
        assert er2.out_path == er.out_path
        assert er2.flatten == er.flatten
        assert er2.default_prim == er.default_prim


# ===========================================================================
# Section 6 — Cross-dataclass JSON serializability (integration)
# ===========================================================================

class TestJsonSerializability:
    """All dataclasses produce nested dicts that json.dumps accepts without error."""

    def test_full_validation_report_json(self):
        """A ValidationReport with mixed statuses serialises to JSON without error."""
        checks = [
            DisciplineCheck(id="no_world_wrapper", status="pass"),
            DisciplineCheck(id="default_prim_set", status="fail", msg="missing"),
            DisciplineCheck(id="format_matches_ext", status="warn"),
        ]
        vr = ValidationReport.from_checks(checks, wrote_files=True)
        payload = json.dumps(vr.to_dict())
        assert len(payload) > 0

    def test_full_mtlx_summary_json(self):
        """A MtlxSummary with errors serialises to JSON without error."""
        ms = MtlxSummary(
            nodegraphs=["NG_pbr"],
            surface_nodes=["mtlxstandard_surface1"],
            inputs_with_abs_paths=["C:/tex/albedo.png"],
            validate_ok=False,
            validate_errors=["binding missing"],
        )
        payload = json.dumps(ms.to_dict())
        assert len(payload) > 0
