"""bind_usd_material_hython_smoke.py — headless Houdini smoke for fork F4.

Dual-mode (mirrors create_mtlx_material_hython_smoke.py):
  - Standalone:  hython bind_usd_material_hython_smoke.py
  - pytest:      @pytest.mark.hython_smoke tests SKIP when hou is not importable.

All calls go through the REAL dispatcher path — dispatch("materials.bind_usd_material",
{params}) -> handler(**params).

Covers:
  1. Positive: on a stage with a Mesh prim + a published Material prim, binds by pattern and
     the USD material:binding is actually authored on the mesh prim (the load-bearing assertion,
     read via UsdShade.MaterialBindingAPI).
  2. Negative (missing material): a nonexistent material_prim -> status=error.
  3. Negative (empty pattern match): a pattern matching no prims -> status=error.

testVerificationSurface: hython-smoke
unitId: b4-w3-f4
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PKG_PYTHON = os.path.join(_REPO_ROOT, "python")
_HOUDINI_HANDLERS = os.path.join(_REPO_ROOT, "houdini", "scripts", "python")
for _p in [_REPO_ROOT, _PKG_PYTHON, _HOUDINI_HANDLERS]:
    _abs = os.path.abspath(_p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

try:
    import pytest  # type: ignore[import-not-found]
    _PYTEST_AVAILABLE = True
except ImportError:
    _PYTEST_AVAILABLE = False

    class _SkipIfDecorator:
        def __call__(self, condition, reason=""):
            def decorator(fn_or_cls):
                return fn_or_cls
            return decorator

    class _MarkNS:
        skipif = _SkipIfDecorator()

        def __getattr__(self, name):
            def decorator_factory(*args, **kwargs):
                def decorator(fn_or_cls):
                    return fn_or_cls
                return decorator
            return decorator_factory

    class pytest:  # type: ignore[no-redef]
        mark = _MarkNS()

        @staticmethod
        def fail(msg: str) -> None:
            raise AssertionError(msg)

        @staticmethod
        def skip(msg: str = "") -> None:
            raise RuntimeError(f"SKIP: {msg}")

try:
    import hou  # type: ignore[import-not-found]
    HOU_AVAILABLE = True
except ImportError:
    HOU_AVAILABLE = False

if _PYTEST_AVAILABLE:
    pytestmark = pytest.mark.hython_smoke


def _get_dispatcher():
    try:
        from fxhoudinimcp_server.handlers import material_handlers  # noqa: F401
        from fxhoudinimcp_server.dispatcher import dispatch
        return dispatch
    except ImportError as exc:
        raise ImportError(
            "dispatcher or material_handlers not importable — run under hython"
        ) from exc


def _build_test_stage(matlib_name="matlib"):
    """Build /obj box -> /stage sopimport (Mesh prim) -> materiallibrary (Material prim).

    Returns (matlib_path, mesh_pattern, material_prim).
    """
    og = hou.node("/obj").createNode("geo", "box_geo")
    for c in og.children():
        c.destroy()
    box = og.createNode("box", "box1")
    box.setDisplayFlag(True)

    stage = hou.node("/stage")
    sopimport = stage.createNode("sopimport", "import_box")
    sopimport.parm("soppath").set(box.path())
    matlib = stage.createNode("materiallibrary", matlib_name)
    matlib.setInput(0, sopimport)
    matlib.createNode("mtlxstandard_surface", "redmat")
    matlib.setDisplayFlag(True)
    prefix = matlib.parm("matpathprefix").eval() if matlib.parm("matpathprefix") else "/materials/"
    material_prim = prefix.rstrip("/") + "/redmat"
    return matlib.path(), "/import_box/*", material_prim


def _binding_authored(node_path: str, material_prim: str) -> list:
    """Return the list of prim paths whose direct material:binding targets material_prim."""
    from pxr import UsdShade
    node = hou.node(node_path)
    if node is None:
        return []
    stage = node.stage()
    if stage is None:
        return []
    bound = []
    for prim in stage.Traverse():
        rel = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel()
        if rel and material_prim in [str(t) for t in rel.GetTargets()]:
            bound.append(str(prim.GetPath()))
    return bound


@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestBindUsdMaterialSmoke:

    def setup_method(self):
        hou.hipFile.clear(suppress_save_prompt=True)
        self._dispatch = _get_dispatcher()

    def test_binds_material_to_geo_prim(self):
        matlib, pattern, mat = _build_test_stage()
        r = self._dispatch("materials.bind_usd_material",
                           {"input_lop": matlib, "geo_pattern": pattern,
                            "material_prim": mat, "name": "bind1"})
        assert r.get("status") == "success", f"bind failed: {r}"
        d = r["data"]
        assert hou.node(d["assignmaterial_path"]) is not None
        assert d["bound_count"] > 0, f"handler reported no bound prims: {d}"
        # The REAL value: USD material:binding actually authored on the mesh prim.
        bound = _binding_authored(d["assignmaterial_path"], mat)
        assert bound, f"no material:binding for {mat} on {d['assignmaterial_path']}'s stage"

    def test_missing_material_returns_error(self):
        matlib, pattern, _mat = _build_test_stage()
        r = self._dispatch("materials.bind_usd_material",
                           {"input_lop": matlib, "geo_pattern": pattern,
                            "material_prim": "/materials/nonexistent", "name": "bind_bad_mat"})
        assert r.get("status") == "error", f"expected error for missing material, got {r}"

    def test_empty_pattern_match_returns_error(self):
        matlib, _pattern, mat = _build_test_stage()
        r = self._dispatch("materials.bind_usd_material",
                           {"input_lop": matlib, "geo_pattern": "/nope/*",
                            "material_prim": mat, "name": "bind_bad_pat"})
        assert r.get("status") == "error", f"expected error for empty match, got {r}"

    def test_typo_pattern_with_preexisting_binding_errors(self):
        # Regression for the false-pass BLOCKER: a stage that ALREADY binds the material
        # upstream must still error on a typo'd pattern (the bind must prove the pattern matched).
        matlib, pattern, mat = _build_test_stage()
        r1 = self._dispatch("materials.bind_usd_material",
                            {"input_lop": matlib, "geo_pattern": pattern,
                             "material_prim": mat, "name": "bind_ok"})
        assert r1.get("status") == "success", f"first bind failed: {r1}"
        # Second bind off the FIRST bind's output (material already bound upstream) with a typo.
        r2 = self._dispatch("materials.bind_usd_material",
                            {"input_lop": r1["data"]["assignmaterial_path"], "geo_pattern": "/nope/*",
                             "material_prim": mat, "name": "bind_typo"})
        assert r2.get("status") == "error", f"typo pattern with pre-existing binding false-passed: {r2}"

    def test_non_material_prim_errors(self):
        matlib, pattern, _mat = _build_test_stage()
        # /import_box/mesh_0 is a Mesh, not a UsdShade.Material.
        r = self._dispatch("materials.bind_usd_material",
                           {"input_lop": matlib, "geo_pattern": pattern,
                            "material_prim": "/import_box/mesh_0", "name": "bind_nonmat"})
        assert r.get("status") == "error", f"expected error binding a non-material prim, got {r}"


def main() -> int:
    if not HOU_AVAILABLE:
        print("ERROR: hou not importable — run with hython, not python.")
        return 1
    failures: list[str] = []
    print("\n--- bind_usd_material smoke (fork F4) ---")
    print("Using: dispatch() -> handler(**params) [real dispatcher path]")
    try:
        dispatch = _get_dispatcher()
    except ImportError as exc:
        print(f"  FAIL  dispatcher/material_handlers import: {exc}")
        return 1

    print("\n  [1] binds material -> geo prim; USD material:binding authored")
    try:
        hou.hipFile.clear(suppress_save_prompt=True)
        matlib, pattern, mat = _build_test_stage()
        r = dispatch("materials.bind_usd_material",
                     {"input_lop": matlib, "geo_pattern": pattern, "material_prim": mat, "name": "bind1"})
        if r.get("status") != "success":
            failures.append(f"positive failed: {r}")
            print(f"    FAIL  {r}")
        else:
            d = r["data"]
            bound = _binding_authored(d["assignmaterial_path"], mat)
            ok = hou.node(d["assignmaterial_path"]) is not None and d["bound_count"] > 0 and bool(bound)
            if ok:
                print(f"    PASS  bound_count={d['bound_count']} bound_prims={bound}")
            else:
                failures.append(f"binding checks: bound_count={d.get('bound_count')} bound={bound}")
                print(f"    FAIL  bound_count={d.get('bound_count')} bound={bound}")
    except Exception as exc:
        failures.append(f"positive raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    print("\n  [2] missing material -> status=error")
    try:
        hou.hipFile.clear(suppress_save_prompt=True)
        matlib, pattern, _mat = _build_test_stage()
        r = dispatch("materials.bind_usd_material",
                     {"input_lop": matlib, "geo_pattern": pattern,
                      "material_prim": "/materials/nonexistent", "name": "bind_bad_mat"})
        if r.get("status") == "error":
            print("    PASS  status=error")
        else:
            failures.append(f"missing material should be error, got {r}")
            print(f"    FAIL  {r}")
    except Exception as exc:
        failures.append(f"missing-material raised (should be wrapped): {exc}")
        print(f"    FAIL  raised: {exc}")

    print("\n  [3] empty pattern match -> status=error")
    try:
        hou.hipFile.clear(suppress_save_prompt=True)
        matlib, _pattern, mat = _build_test_stage()
        r = dispatch("materials.bind_usd_material",
                     {"input_lop": matlib, "geo_pattern": "/nope/*", "material_prim": mat, "name": "bind_bad_pat"})
        if r.get("status") == "error":
            print("    PASS  status=error")
        else:
            failures.append(f"empty match should be error, got {r}")
            print(f"    FAIL  {r}")
    except Exception as exc:
        failures.append(f"empty-match raised (should be wrapped): {exc}")
        print(f"    FAIL  raised: {exc}")

    print("\n  [4] typo pattern with pre-existing upstream binding -> status=error (BLOCKER regression)")
    try:
        hou.hipFile.clear(suppress_save_prompt=True)
        matlib, pattern, mat = _build_test_stage()
        r1 = dispatch("materials.bind_usd_material",
                      {"input_lop": matlib, "geo_pattern": pattern, "material_prim": mat, "name": "bind_ok"})
        r2 = dispatch("materials.bind_usd_material",
                      {"input_lop": r1["data"]["assignmaterial_path"], "geo_pattern": "/nope/*",
                       "material_prim": mat, "name": "bind_typo"})
        if r1.get("status") == "success" and r2.get("status") == "error":
            print("    PASS  first bind ok; typo bind errors despite pre-existing binding")
        else:
            failures.append(f"BLOCKER regression: r1={r1.get('status')} r2={r2.get('status')}")
            print(f"    FAIL  r1={r1.get('status')} r2={r2.get('status')}")
    except Exception as exc:
        failures.append(f"BLOCKER-regression raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    print("\n  [5] non-material prim as material_prim -> status=error")
    try:
        hou.hipFile.clear(suppress_save_prompt=True)
        matlib, pattern, _mat = _build_test_stage()
        r = dispatch("materials.bind_usd_material",
                     {"input_lop": matlib, "geo_pattern": pattern,
                      "material_prim": "/import_box/mesh_0", "name": "bind_nonmat"})
        if r.get("status") == "error":
            print("    PASS  status=error")
        else:
            failures.append(f"non-material prim should error, got {r}")
            print(f"    FAIL  {r}")
    except Exception as exc:
        failures.append(f"non-material raised (should be wrapped): {exc}")
        print(f"    FAIL  raised: {exc}")

    print()
    if failures:
        print(f"SMOKE FAILED — {len(failures)} failure(s):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("SMOKE PASSED — bind_usd_material authors a real USD material:binding (dispatcher path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
