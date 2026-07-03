"""setup_constraint_network_hython_smoke.py — headless Houdini smoke for fork F3.

Dual-mode (mirrors create_mtlx_material_hython_smoke.py):
  - Standalone:  hython setup_constraint_network_hython_smoke.py
  - pytest:      @pytest.mark.hython_smoke tests SKIP when hou is not importable.

All calls go through the REAL dispatcher path — dispatch("workflow.setup_constraint_network",
{params}) -> handler(**params).

Covers:
  1. Positive (glue): builds fracture + rbdconstraintproperties + rbdbulletsolver + filecache;
     the constraint-properties node cooks to a REAL constraint network (prims > 0 — the polylines
     between fractured pieces), constrainttype='glue', glue_strength set. The solver cooks to
     simulated geometry (points > 0) after advancing frames — the sim actually runs.
  2. Soft variant: constraint_type='soft' sets soft_stiffness.
  3. Negative: an invalid constraint_type -> status=error (the handler's ValueError, wrapped).

testVerificationSurface: hython-smoke
unitId: b4-w2-f3
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
        from fxhoudinimcp_server.handlers import workflow_handlers  # noqa: F401
        from fxhoudinimcp_server.dispatcher import dispatch
        return dispatch
    except ImportError as exc:
        raise ImportError(
            "dispatcher or workflow_handlers not importable — run under hython"
        ) from exc


def _make_source(name: str = "rock") -> str:
    """A small solid box source to fracture. Returns the object path."""
    geo = hou.node("/obj").createNode("geo", name)
    for c in geo.children():
        c.destroy()
    box = geo.createNode("box", "box1")
    box.parmTuple("size").set((2, 2, 2))
    box.parm("ty").set(4)
    box.setDisplayFlag(True)
    return geo.path()


def _base(t: str) -> str:
    parts = t.split("::")
    return parts[0] if parts[1:] and parts[-1][:1].isdigit() else t


@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestSetupConstraintNetworkSmoke:

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        hou.playbar.setFrameRange(1, 48)
        hou.setFrame(1)
        cls._dispatch = _get_dispatcher()

    def test_glue_builds_constraint_network_and_cooks(self):
        src = _make_source("rock_glue")
        r = self._dispatch("workflow.setup_constraint_network",
                           {"geo_path": src, "constraint_type": "glue",
                            "strength": 5000.0, "name": "cn_glue"})
        assert r.get("status") == "success", f"setup failed: {r}"
        d = r["data"]
        cp = hou.node(d["constraint_props_path"])
        assert cp is not None and _base(cp.type().name()) == "rbdconstraintproperties"
        assert cp.parm("constrainttype").evalAsString() == "glue"
        assert abs(cp.parm("glue_strength").eval() - 5000.0) < 1e-3
        # The constraint network is real geometry: prims (polylines) between pieces.
        con_geo = cp.geometry()
        assert con_geo is not None and len(con_geo.prims()) > 0, "no constraint primitives built"
        # The sim cooks to simulated geometry after advancing frames.
        hou.setFrame(15)
        sim_geo = hou.node(d["solver_path"]).geometry()
        assert sim_geo is not None and len(sim_geo.points()) > 0, "solver produced no geometry"

    def test_soft_sets_stiffness(self):
        src = _make_source("rock_soft")
        r = self._dispatch("workflow.setup_constraint_network",
                           {"geo_path": src, "constraint_type": "soft",
                            "strength": 20.0, "name": "cn_soft"})
        assert r.get("status") == "success", f"soft setup failed: {r}"
        cp = hou.node(r["data"]["constraint_props_path"])
        assert cp.parm("constrainttype").evalAsString() == "soft"
        assert abs(cp.parm("soft_stiffness").eval() - 20.0) < 1e-3

    def test_bad_constraint_type_returns_error(self):
        src = _make_source("rock_bad")
        r = self._dispatch("workflow.setup_constraint_network",
                           {"geo_path": src, "constraint_type": "bogus", "name": "cn_bad"})
        assert r.get("status") == "error", f"expected error for bad type, got {r}"


def main() -> int:
    if not HOU_AVAILABLE:
        print("ERROR: hou not importable — run with hython, not python.")
        return 1
    failures: list[str] = []
    print("\n--- setup_constraint_network smoke (fork F3) ---")
    print("Using: dispatch() -> handler(**params) [real dispatcher path]")
    try:
        dispatch = _get_dispatcher()
    except ImportError as exc:
        print(f"  FAIL  dispatcher/workflow_handlers import: {exc}")
        return 1
    hou.hipFile.clear(suppress_save_prompt=True)
    hou.playbar.setFrameRange(1, 48)
    hou.setFrame(1)

    print("\n  [1] glue: builds constraint network + cooks + simulates")
    try:
        src = _make_source("rock_glue")
        r = dispatch("workflow.setup_constraint_network",
                     {"geo_path": src, "constraint_type": "glue",
                      "strength": 5000.0, "name": "cn_glue"})
        if r.get("status") != "success":
            failures.append(f"glue setup failed: {r}")
            print(f"    FAIL  {r}")
        else:
            d = r["data"]
            cp = hou.node(d["constraint_props_path"])
            con_geo = cp.geometry()
            n_con = len(con_geo.prims()) if con_geo else 0
            ctype = cp.parm("constrainttype").evalAsString()
            strength = cp.parm("glue_strength").eval()
            hou.setFrame(15)
            sim_geo = hou.node(d["solver_path"]).geometry()
            n_pts = len(sim_geo.points()) if sim_geo else 0
            ok = (_base(cp.type().name()) == "rbdconstraintproperties" and ctype == "glue"
                  and abs(strength - 5000.0) < 1e-3 and n_con > 0 and n_pts > 0)
            if ok:
                print(f"    PASS  type={ctype} strength={strength} constraints={n_con} sim_pts={n_pts}")
            else:
                failures.append(f"glue checks: type={ctype} strength={strength} con={n_con} pts={n_pts}")
                print(f"    FAIL  type={ctype} strength={strength} con={n_con} pts={n_pts}")
    except Exception as exc:
        failures.append(f"glue raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    print("\n  [2] soft: sets soft_stiffness")
    try:
        src = _make_source("rock_soft")
        r = dispatch("workflow.setup_constraint_network",
                     {"geo_path": src, "constraint_type": "soft", "strength": 20.0, "name": "cn_soft"})
        if r.get("status") != "success":
            failures.append(f"soft setup failed: {r}")
            print(f"    FAIL  {r}")
        else:
            cp = hou.node(r["data"]["constraint_props_path"])
            ok = cp.parm("constrainttype").evalAsString() == "soft" and abs(cp.parm("soft_stiffness").eval() - 20.0) < 1e-3
            print(f"    {'PASS' if ok else 'FAIL'}  soft_stiffness={cp.parm('soft_stiffness').eval()}")
            if not ok:
                failures.append("soft stiffness not set")
    except Exception as exc:
        failures.append(f"soft raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    print("\n  [3] negative: bad constraint_type -> status=error")
    try:
        src = _make_source("rock_bad")
        r = dispatch("workflow.setup_constraint_network",
                     {"geo_path": src, "constraint_type": "bogus", "name": "cn_bad"})
        if r.get("status") == "error":
            print("    PASS  status=error")
        else:
            failures.append(f"bad type should be error, got {r}")
            print(f"    FAIL  {r}")
    except Exception as exc:
        failures.append(f"negative raised (should be wrapped): {exc}")
        print(f"    FAIL  raised: {exc}")

    print()
    if failures:
        print(f"SMOKE FAILED — {len(failures)} failure(s):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("SMOKE PASSED — setup_constraint_network builds+cooks a real constraint network (dispatcher path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
