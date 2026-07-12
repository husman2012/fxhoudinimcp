"""setup_flip_sim_hython_smoke.py — headless Houdini smoke for the FLIP-factory FIX.

Dual-mode (mirrors create_mtlx_material_hython_smoke.py):
  - Standalone:  hython setup_flip_sim_hython_smoke.py
  - pytest:      @pytest.mark.hython_smoke tests SKIP when hou is not importable.

Regression guard for the degraded-FLIP fix (2026-07-03): the factory used to create a
flipsource/fluidtank that don't exist in the DOP context and were never wired, so the FLIP
object had NO fluid and the sim emitted 0 particles. The fix sources the flipobject via its
`soppath` parm. This smoke proves the sim EMITS: object_count > 0 and imported particles > 0
after stepping.

All calls go through the REAL dispatcher path — dispatch("workflow.setup_flip_sim", {params}).

testVerificationSurface: hython-smoke
unitId: b4-w2-flipfix
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
        raise ImportError("dispatcher/workflow_handlers not importable — run under hython") from exc


def _make_source() -> str:
    geo = hou.node("/obj").createNode("geo", "flipsrc")
    for c in geo.children():
        c.destroy()
    sph = geo.createNode("sphere", "sphere1")
    sph.parm("type").set("polymesh")
    sph.parm("ty").set(3)
    sph.setDisplayFlag(True)
    return sph.path()


def _cook_and_count(dop_path: str, geo_path: str, to_frame: int = 12) -> tuple:
    """Step the sim and return (object_count, imported_particle_count)."""
    dop = hou.node(dop_path)
    for f in range(1, to_frame + 1):
        hou.setFrame(f)
        dop.cook(force=True)
    sim = dop.simulation()
    n_obj = len(sim.objects()) if sim and sim.objects() is not None else 0
    imp = hou.node(geo_path + "/dop_import1")
    n_pts = 0
    if imp is not None:
        g = imp.geometry()
        n_pts = len(g.points()) if g else 0
    return n_obj, n_pts


@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestSetupFlipSimEmits:

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        hou.playbar.setFrameRange(1, 24)
        hou.setFrame(1)
        cls._dispatch = _get_dispatcher()
        cls._src = _make_source()

    def test_flip_object_is_sourced_via_soppath(self):
        r = self._dispatch("workflow.setup_flip_sim",
                           {"source_geo": self._src, "particle_sep": 0.15, "name": "flip_emit"})
        assert r.get("status") == "success", f"setup failed: {r}"
        d = r["data"]
        flipobj = hou.node(d["dop_path"] + "/flipobject1")
        assert flipobj is not None, "flipobject not created"
        # THE fix: soppath points at the source (not the empty default) so there is fluid.
        assert flipobj.parm("soppath").eval() not in ("", "./particlefluidobject/defaultfluid"), \
            "flipobject soppath not sourced — the sim would emit nothing"

    def test_sim_emits_particles(self):
        r = self._dispatch("workflow.setup_flip_sim",
                           {"source_geo": self._src, "particle_sep": 0.15, "name": "flip_emit2"})
        assert r.get("status") == "success", f"setup failed: {r}"
        d = r["data"]
        n_obj, n_pts = _cook_and_count(d["dop_path"], d["geo_path"])
        assert n_obj > 0, "FLIP object never registered (object_count 0 — the old degraded bug)"
        assert n_pts > 0, "FLIP sim emitted no particles (the old degraded bug)"


def main() -> int:
    if not HOU_AVAILABLE:
        print("ERROR: hou not importable — run with hython, not python.")
        return 1
    failures: list[str] = []
    print("\n--- setup_flip_sim EMISSION smoke (FLIP-factory fix) ---")
    print("Using: dispatch() -> handler(**params) [real dispatcher path]")
    try:
        dispatch = _get_dispatcher()
    except ImportError as exc:
        print(f"  FAIL  dispatcher import: {exc}")
        return 1
    hou.hipFile.clear(suppress_save_prompt=True)
    hou.playbar.setFrameRange(1, 24)
    hou.setFrame(1)
    src = _make_source()

    print("\n  [1] flipobject sourced via soppath")
    try:
        r = dispatch("workflow.setup_flip_sim",
                     {"source_geo": src, "particle_sep": 0.15, "name": "flip_emit"})
        d = r["data"]
        sp = hou.node(d["dop_path"] + "/flipobject1").parm("soppath").eval()
        ok = sp not in ("", "./particlefluidobject/defaultfluid")
        print(f"    {'PASS' if ok else 'FAIL'}  soppath={sp!r}")
        if not ok:
            failures.append("soppath not sourced")
    except Exception as exc:
        failures.append(f"[1] raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    print("\n  [2] sim EMITS particles (object_count > 0, imported particles > 0)")
    try:
        r = dispatch("workflow.setup_flip_sim",
                     {"source_geo": src, "particle_sep": 0.15, "name": "flip_emit2"})
        d = r["data"]
        n_obj, n_pts = _cook_and_count(d["dop_path"], d["geo_path"])
        ok = n_obj > 0 and n_pts > 0
        print(f"    {'PASS' if ok else 'FAIL'}  object_count={n_obj} particles={n_pts}")
        if not ok:
            failures.append(f"no emission: object_count={n_obj} particles={n_pts}")
    except Exception as exc:
        failures.append(f"[2] raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    print()
    if failures:
        print(f"SMOKE FAILED — {len(failures)} failure(s):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("SMOKE PASSED — setup_flip_sim now EMITS a real FLIP sim (dispatcher path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
