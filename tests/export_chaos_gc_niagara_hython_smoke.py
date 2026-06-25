"""
Hython smoke tests for export_chaos_gc and export_niagara MCP tools.
pp12-111f — PP12-111 PR-6.

Run with:
    hython tests/export_chaos_gc_niagara_hython_smoke.py

Three checks driven via the REAL dispatcher (not direct handler calls).
  Check 1 — Chaos-GC CONTIGUOUS: dispatch("export_chaos_gc", ...) with
             unreal_gc_piece = 0,1,2 → ok True, .abc written,
             unreal_gc_* attrib survives alembic round-trip.
  Check 2 — Chaos-GC NON-CONTIGUOUS FR-7: dispatch("export_chaos_gc", ...) with
             unreal_gc_piece = 0,2 (gap) → ok False, error names gap, NO .abc written.
  Check 3 — Niagara: dispatch("export_niagara", ...) with P/v/age/id points
             → ok True, .hbjson written.

Packed-alembic lesson (PR-5 / FR-4): custom attribs on alembic-loaded geo are
visible ONLY after an unpack SOP. All round-trip checks add an unpack node.
"""

from __future__ import annotations

import os
import sys
import json
import tempfile

# ---------------------------------------------------------------------------
# Path bootstrap — must run from fxhoudinimcp repo root.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PKG_PYTHON = os.path.join(_REPO_ROOT, "python")
_HOUDINI_HANDLERS = os.path.join(_REPO_ROOT, "houdini", "scripts", "python")
for _p in [_REPO_ROOT, _PKG_PYTHON, _HOUDINI_HANDLERS]:
    _abs = os.path.abspath(_p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

import hou


# ---------------------------------------------------------------------------
# Dispatcher accessor — imports the REAL dispatcher (not a mock).
# The RED-state note: fxhoudinimcp_server is the fork package; importing
# export_handlers side-registers the handlers so dispatch() knows about them.
# ---------------------------------------------------------------------------

def _get_dispatcher():
    from fxhoudinimcp_server.handlers import export_handlers  # noqa: F401  # side-registers
    from fxhoudinimcp_server.dispatcher import dispatch
    return dispatch


def _data(result):
    """Unwrap the dispatcher envelope {status, data, timing_ms} -> the handler dict.

    dispatch() wraps a handler's {ok, ...} return under a 'data' key; assertions
    must read the handler payload, not the envelope. Mirrors the PR-5 smoke.
    """
    if isinstance(result, dict) and "data" in result:
        return result.get("data", result)
    return result


# ---------------------------------------------------------------------------
# Geometry builder helpers
# ---------------------------------------------------------------------------

def _build_gc_geo(obj_node, piece_ids):
    """Build a SOP geo with unreal_gc_piece prim attrib set to piece_ids values."""
    geo_node = obj_node.createNode("geo", "gc_geo")
    box = geo_node.createNode("box", "box1")  # default box = 6 quad prims
    # Wrangle sets unreal_gc_piece on each prim cycling through piece_ids
    wrangle = geo_node.createNode("attribwrangle", "set_gc_piece")
    wrangle.setInput(0, box)
    pieces_str = "{" + ", ".join(str(p) for p in piece_ids) + "}"
    wrangle.parm("class").set(1)  # prim
    wrangle.parm("snippet").set(
        f"int pieces[] = {pieces_str};\n"
        f"int idx = @primnum % len(pieces);\n"
        f"i@unreal_gc_piece = pieces[idx];\n"
        f"s@name = sprintf('piece%d', pieces[idx]);\n"
    )
    wrangle.setRenderFlag(True)
    # displayNode() must return the wrangle (it carries unreal_gc_piece); the
    # handler reads geometry off the geo container's DISPLAY SOP.
    wrangle.setDisplayFlag(True)
    wrangle.cook(force=True)
    return geo_node


def _build_niagara_geo(obj_node):
    """Build a SOP geo with P, v, age, id point attribs for Niagara export."""
    geo_node = obj_node.createNode("geo", "niagara_geo")
    line = geo_node.createNode("line", "line_points")
    line.parm("points").set(4)  # 4 points along the line
    wrangle = geo_node.createNode("attribwrangle", "set_particle_attribs")
    wrangle.setInput(0, line)
    wrangle.parm("class").set(2)  # point
    wrangle.parm("snippet").set(
        "v@v = set(@ptnum * 0.1, 0, 0);\n"
        "f@age = @ptnum * 0.25;\n"
        "i@id = @ptnum;\n"
    )
    wrangle.setRenderFlag(True)
    wrangle.setDisplayFlag(True)
    wrangle.cook(force=True)
    return geo_node


# ---------------------------------------------------------------------------
# Smoke check classes
# ---------------------------------------------------------------------------

class CheckGcContiguous:
    """
    Chaos-GC contiguous smoke — unreal_gc_piece = [0, 1, 2].

    Expects:
      - dispatch result: ok=True
      - .abc file exists on disk
      - unreal_gc_* attrib survives alembic round-trip (after unpack SOP)
    """

    _abc_path: str = ""
    _result: dict = {}
    _verify_obj: hou.ObjNode | None = None

    def setup(self, tmp_dir: str):
        dispatch = _get_dispatcher()
        obj = hou.node("/obj")
        self._geo_node = _build_gc_geo(obj, [0, 1, 2])
        self._abc_path = os.path.join(tmp_dir, "gc_contiguous.abc")
        geo_path = self._geo_node.path()

        self._result = dispatch("export_chaos_gc", {
            "node_path": geo_path,
            "out_abc": self._abc_path,
        })

    def teardown(self):
        if self._geo_node and self._geo_node.isValid():
            self._geo_node.destroy()
        if self._verify_obj and self._verify_obj.isValid():
            self._verify_obj.destroy()

    def check_ok_true(self):
        ok = _data(self._result).get("ok")
        assert ok is True, f"expected ok=True, got {self._result!r}"

    def check_abc_exists(self):
        assert os.path.isfile(self._abc_path), (
            f"expected .abc at {self._abc_path!r} but file not found"
        )

    def check_gc_attrib_round_trips(self):
        """Load the .abc via alembic SOP + unpack SOP, confirm unreal_gc_piece present."""
        obj = hou.node("/obj")
        verify_geo = obj.createNode("geo", "verify_gc_contiguous")
        self._verify_obj = verify_geo

        abc_sop = verify_geo.createNode("alembic", "read_abc")
        abc_sop.parm("fileName").set(self._abc_path)
        abc_sop.cook(force=True)

        # Critical lesson (PR-5/FR-4): custom attribs only visible after unpack
        unpack_sop = verify_geo.createNode("unpack", "unpack_abc")
        unpack_sop.setInput(0, abc_sop)
        unpack_sop.cook(force=True)

        geo = unpack_sop.geometry()
        prim_attrib_names = [a.name() for a in geo.primAttribs()]
        gc_attribs = [n for n in prim_attrib_names if n.startswith("unreal_gc_")]
        assert gc_attribs, (
            f"expected unreal_gc_* prim attrib after unpack, found: {prim_attrib_names}"
        )


class CheckGcNonContiguous:
    """
    Chaos-GC non-contiguous FR-7 smoke — unreal_gc_piece = [0, 2] (gap at 1).

    Expects:
      - dispatch result: ok=False
      - error string names the gap
      - NO .abc file written
    """

    _abc_path: str = ""
    _result: dict = {}

    def setup(self, tmp_dir: str):
        dispatch = _get_dispatcher()
        obj = hou.node("/obj")
        self._geo_node = _build_gc_geo(obj, [0, 2])
        self._abc_path = os.path.join(tmp_dir, "gc_noncontiguous.abc")
        geo_path = self._geo_node.path()

        self._result = dispatch("export_chaos_gc", {
            "node_path": geo_path,
            "out_abc": self._abc_path,
        })

    def teardown(self):
        if self._geo_node and self._geo_node.isValid():
            self._geo_node.destroy()

    def check_ok_false(self):
        ok = _data(self._result).get("ok")
        assert ok is False, f"expected ok=False for non-contiguous pieces, got {self._result!r}"

    def check_error_names_gap(self):
        error = _data(self._result).get("error", "")
        assert "gap" in error.lower() or "non-contiguous" in error.lower() or "1" in error, (
            f"error string must mention the gap; got {error!r}"
        )

    def check_no_abc_written(self):
        assert not os.path.isfile(self._abc_path), (
            f"export_chaos_gc must NOT write .abc on FR-7 refusal; found file at {self._abc_path!r}"
        )


class CheckNiagara:
    """
    Niagara smoke — P/v/age/id point geo.

    Expects:
      - dispatch result: ok=True
      - .hbjson file exists on disk
    """

    _hbjson_path: str = ""
    _result: dict = {}

    def setup(self, tmp_dir: str):
        dispatch = _get_dispatcher()
        obj = hou.node("/obj")
        self._geo_node = _build_niagara_geo(obj)
        self._hbjson_path = os.path.join(tmp_dir, "niagara_out.hbjson")
        geo_path = self._geo_node.path()

        self._result = dispatch("export_niagara", {
            "node_path": geo_path,
            "out_path": self._hbjson_path,
        })

    def teardown(self):
        if self._geo_node and self._geo_node.isValid():
            self._geo_node.destroy()

    def check_ok_true(self):
        ok = _data(self._result).get("ok")
        assert ok is True, f"expected ok=True, got {self._result!r}"

    def check_hbjson_exists(self):
        assert os.path.isfile(self._hbjson_path), (
            f"expected .hbjson at {self._hbjson_path!r} but file not found"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_check(label: str, checker, tmp_dir: str) -> bool:
    """Run all check_* methods on a checker instance. Returns True if all pass."""
    print(f"\n{'='*60}")
    print(f"SMOKE: {label}")
    print(f"{'='*60}")
    passed = True
    try:
        checker.setup(tmp_dir)
    except Exception as exc:
        print(f"  SETUP ERROR: {exc}")
        return False

    check_methods = [
        name for name in dir(checker)
        if name.startswith("check_") and callable(getattr(checker, name))
    ]
    for mname in sorted(check_methods):
        try:
            getattr(checker, mname)()
            print(f"  PASS  {mname}")
        except AssertionError as exc:
            print(f"  FAIL  {mname}: {exc}")
            passed = False
        except Exception as exc:
            print(f"  ERROR {mname}: {type(exc).__name__}: {exc}")
            passed = False

    try:
        checker.teardown()
    except Exception as exc:
        print(f"  TEARDOWN WARNING: {exc}")

    return passed


def main():
    with tempfile.TemporaryDirectory(prefix="pp12_111f_smoke_") as tmp_dir:
        results = {}
        results["gc_contiguous"] = _run_check(
            "Chaos-GC CONTIGUOUS (pieces 0,1,2)",
            CheckGcContiguous(),
            tmp_dir,
        )
        results["gc_non_contiguous_fr7"] = _run_check(
            "Chaos-GC NON-CONTIGUOUS FR-7 (pieces 0,2 — gap at 1)",
            CheckGcNonContiguous(),
            tmp_dir,
        )
        results["niagara"] = _run_check(
            "Niagara (P/v/age/id points)",
            CheckNiagara(),
            tmp_dir,
        )

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_passed = False

    if all_passed:
        print("\nAll smoke checks PASSED.")
        sys.exit(0)
    else:
        print("\nOne or more smoke checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
