"""setup_whitewater_hython_smoke.py — headless Houdini smoke for fork F2.

Dual-mode (mirrors setup_constraint_network_hython_smoke.py):
  - Standalone:  hython setup_whitewater_hython_smoke.py
  - pytest:      @pytest.mark.hython_smoke tests SKIP when hou is not importable.

All calls go through the REAL dispatcher path — dispatch("workflow.setup_whitewater",
{params}) -> handler(**params).

Covers:
  1. Positive: builds the FLIP -> fluidcompress -> whitewatersource -> whitewatersolver
     pipeline; after advancing a few frames the whitewater SOLVER cooks to
     WHITEWATER PARTICLES > 0 (foam/spray/bubbles actually emit — the load-bearing assertion).
  2. foam_amount variant: a different emission amount still builds + emits.
  3. Negative: a nonexistent source_geo -> status=error (the handler's ValueError, wrapped).

testVerificationSurface: hython-smoke
unitId: b4-w2-f2
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


def _make_source(name: str = "splash_src") -> str:
    """A small solid sphere source that becomes fluid. Returns the object path."""
    geo = hou.node("/obj").createNode("geo", name)
    for c in geo.children():
        c.destroy()
    sph = geo.createNode("sphere", "sphere1")
    sph.parm("type").set(2)  # polygon
    sph.parmTuple("rad").set((0.8, 0.8, 0.8))
    xf = geo.createNode("xform", "xform1")
    xf.setInput(0, sph)
    xf.parmTuple("t").set((0, 1.5, 0))
    xf.setDisplayFlag(True)
    return geo.path()


def _max_whitewater(solver_node, frames=(2, 3, 4)) -> int:
    """Cook the whitewater solver sequentially and return the max particle count."""
    best = 0
    for f in frames:
        hou.setFrame(f)
        g = solver_node.geometry()
        if g is not None:
            best = max(best, len(g.points()))
    return best


@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestSetupWhitewaterSmoke:

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        hou.playbar.setFrameRange(1, 48)
        hou.setFrame(1)
        cls._dispatch = _get_dispatcher()

    def test_builds_and_emits_whitewater(self):
        src = _make_source("splash_a")
        r = self._dispatch("workflow.setup_whitewater",
                           {"source_geo": src, "particle_sep": 0.2,
                            "name": "ww_a", "foam_amount": 2.0})
        assert r.get("status") == "success", f"setup failed: {r}"
        d = r["data"]
        for key in ("geo_path", "source_path", "solver_path", "cache_path"):
            assert hou.node(d[key]) is not None, f"missing node {key}: {d.get(key)}"
        # The whitewater solver emits foam/spray/bubble particles after a few frames.
        wws = hou.node(d["solver_path"])
        n = _max_whitewater(wws)
        assert n > 0, f"whitewater solver produced no particles (max over frames = {n})"

    def test_foam_amount_variant_emits(self):
        src = _make_source("splash_b")
        r = self._dispatch("workflow.setup_whitewater",
                           {"source_geo": src, "particle_sep": 0.2,
                            "name": "ww_b", "foam_amount": 5.0})
        assert r.get("status") == "success", f"setup failed: {r}"
        wws = hou.node(r["data"]["solver_path"])
        assert _max_whitewater(wws) > 0, "foam_amount variant produced no whitewater"

    def test_bad_source_geo_returns_error(self):
        r = self._dispatch("workflow.setup_whitewater",
                           {"source_geo": "/obj/nonexistent", "name": "ww_bad"})
        assert r.get("status") == "error", f"expected error for bad source, got {r}"


def main() -> int:
    if not HOU_AVAILABLE:
        print("ERROR: hou not importable — run with hython, not python.")
        return 1
    failures: list[str] = []
    print("\n--- setup_whitewater smoke (fork F2) ---")
    print("Using: dispatch() -> handler(**params) [real dispatcher path]")
    try:
        dispatch = _get_dispatcher()
    except ImportError as exc:
        print(f"  FAIL  dispatcher/workflow_handlers import: {exc}")
        return 1
    hou.hipFile.clear(suppress_save_prompt=True)
    hou.playbar.setFrameRange(1, 48)
    hou.setFrame(1)

    print("\n  [1] builds FLIP -> whitewatersource -> whitewatersolver + emits particles")
    try:
        src = _make_source("splash_a")
        r = dispatch("workflow.setup_whitewater",
                     {"source_geo": src, "particle_sep": 0.2, "name": "ww_a", "foam_amount": 2.0})
        if r.get("status") != "success":
            failures.append(f"setup failed: {r}")
            print(f"    FAIL  {r}")
        else:
            d = r["data"]
            missing = [k for k in ("geo_path", "source_path", "solver_path", "cache_path")
                       if hou.node(d.get(k, "")) is None]
            wws = hou.node(d["solver_path"])
            n = _max_whitewater(wws)
            ok = not missing and n > 0
            if ok:
                print(f"    PASS  whitewater particles (max over f2-4) = {n}")
            else:
                failures.append(f"checks: missing={missing} whitewater_pts={n}")
                print(f"    FAIL  missing={missing} whitewater_pts={n}")
    except Exception as exc:
        failures.append(f"positive raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    print("\n  [2] foam_amount variant emits")
    try:
        src = _make_source("splash_b")
        r = dispatch("workflow.setup_whitewater",
                     {"source_geo": src, "particle_sep": 0.2, "name": "ww_b", "foam_amount": 5.0})
        if r.get("status") != "success":
            failures.append(f"foam variant setup failed: {r}")
            print(f"    FAIL  {r}")
        else:
            wws = hou.node(r["data"]["solver_path"])
            n = _max_whitewater(wws)
            print(f"    {'PASS' if n > 0 else 'FAIL'}  whitewater particles = {n}")
            if n <= 0:
                failures.append("foam variant produced no whitewater")
    except Exception as exc:
        failures.append(f"foam variant raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    print("\n  [3] negative: bad source_geo -> status=error")
    try:
        r = dispatch("workflow.setup_whitewater",
                     {"source_geo": "/obj/nonexistent", "name": "ww_bad"})
        if r.get("status") == "error":
            print("    PASS  status=error")
        else:
            failures.append(f"bad source should be error, got {r}")
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
    print("SMOKE PASSED — setup_whitewater builds+cooks a real whitewater sim (dispatcher path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
