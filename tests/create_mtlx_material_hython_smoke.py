"""create_mtlx_material_hython_smoke.py — headless Houdini smoke for fork F1.

Dual-mode (mirrors bonedeform_hython_smoke.py):
  - Standalone:  hython create_mtlx_material_hython_smoke.py
  - pytest:      @pytest.mark.hython_smoke tests SKIP when hou is not importable.

All calls go through the REAL dispatcher path — dispatch("materials.create_mtlx_material",
{params}) -> handler(**params) — the authoritative production path (a direct handler({dict})
call would MISS the handler(**params) convention).

Covers:
  1. Positive: builds a materiallibrary + mtlxstandard_surface, and — the real value — the
     materiallibrary AUTO-PUBLISHES a USD Material prim at /materials/<name> on the stage
     (verified by traversing the materiallibrary's composed stage for a UsdShade.Material at
     the returned material_prim_path). Base parms (base_color/metalness/roughness) applied.
  2. Textures + normal map: wired via setNamedInput/mtlxnormalmap without raising (the texture
     files need not exist for the nodes to build).
  3. Negative: a bogus parent_path returns status=error (the handler's ValueError, wrapped by
     the dispatcher) — no unhandled exception.

testVerificationSurface: hython-smoke
unitId: b4-w2-f1
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
    """Import material_handlers to fire register_handler(), then return dispatch."""
    try:
        from fxhoudinimcp_server.handlers import material_handlers  # noqa: F401
        from fxhoudinimcp_server.dispatcher import dispatch
        return dispatch
    except ImportError as exc:
        raise ImportError(
            "dispatcher or material_handlers not importable — run under hython"
        ) from exc


def _material_prim_published(matlib_path: str, prim_path: str) -> bool:
    """True iff a UsdShade.Material prim exists at prim_path on the materiallibrary's stage."""
    from pxr import UsdShade
    matlib = hou.node(matlib_path)
    if matlib is None:
        return False
    stage = matlib.stage()
    if stage is None:
        return False
    prim = stage.GetPrimAtPath(prim_path)
    return bool(prim) and prim.IsValid() and prim.IsA(UsdShade.Material)


@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestCreateMtlxMaterialSmoke:

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._dispatch = _get_dispatcher()

    def test_publishes_material_prim_with_base_parms(self):
        result = self._dispatch(
            "materials.create_mtlx_material",
            {"name": "smoke_mtlx", "parent_path": "/stage",
             "base_color": [0.4, 0.2, 0.1], "metalness": 1.0, "roughness": 0.3},
        )
        assert result.get("status") == "success", f"create_mtlx_material failed: {result}"
        data = result["data"]
        assert data["material_prim_path"] == "/materials/smoke_mtlx"
        assert hou.node(data["materiallibrary_path"]) is not None
        assert hou.node(data["surface_path"]) is not None
        # The REAL value: the Material prim actually published on the stage.
        assert _material_prim_published(data["materiallibrary_path"], data["material_prim_path"]), (
            "materiallibrary did not publish a UsdShade.Material at "
            f"{data['material_prim_path']!r} — the mtlx surface was not collected"
        )
        # base parms applied on the surface node.
        surf = hou.node(data["surface_path"])
        assert abs(surf.parm("metalness").eval() - 1.0) < 1e-6
        assert abs(surf.parm("specular_roughness").eval() - 0.3) < 1e-6

    def test_textures_and_normal_wire_without_raising(self):
        result = self._dispatch(
            "materials.create_mtlx_material",
            {"name": "smoke_tex", "parent_path": "/stage",
             "textures": {"base_color": "$HIP/tex/albedo.exr"},
             "normal_map": "$HIP/tex/normal.exr"},
        )
        assert result.get("status") == "success", f"textured create failed: {result}"
        data = result["data"]
        matlib = hou.node(data["materiallibrary_path"])
        # Base type match — createNode("mtlxnormalmap") resolves to the versioned
        # mtlxnormalmap::2.0, so strip the ::version before comparing.
        child_bases = {c.type().name().split("::")[0] for c in matlib.children()}
        assert "mtlxUsdUVTexture" in child_bases, "texture node not created"
        assert "mtlxnormalmap" in child_bases, "normalmap node not created"

    def test_bogus_parent_returns_status_error(self):
        result = self._dispatch(
            "materials.create_mtlx_material",
            {"name": "smoke_bad", "parent_path": "/stage/does_not_exist/deep"},
        )
        assert result.get("status") == "error", f"expected error for bogus parent, got {result}"


def main() -> int:
    if not HOU_AVAILABLE:
        print("ERROR: hou not importable — run with hython, not python.")
        return 1
    failures: list[str] = []
    print("\n--- create_mtlx_material smoke (fork F1) ---")
    print("Using: dispatch() -> handler(**params) [real dispatcher path]")
    try:
        dispatch = _get_dispatcher()
    except ImportError as exc:
        print(f"  FAIL  dispatcher/material_handlers import: {exc}")
        return 1
    hou.hipFile.clear(suppress_save_prompt=True)

    print("\n  [1] positive: publishes /materials/smoke_mtlx with base parms")
    try:
        r = dispatch("materials.create_mtlx_material",
                     {"name": "smoke_mtlx", "parent_path": "/stage",
                      "base_color": [0.4, 0.2, 0.1], "metalness": 1.0, "roughness": 0.3})
        if r.get("status") != "success":
            failures.append(f"positive failed: {r}")
            print(f"    FAIL  {r}")
        else:
            d = r["data"]
            pub = _material_prim_published(d["materiallibrary_path"], d["material_prim_path"])
            if d["material_prim_path"] == "/materials/smoke_mtlx" and pub:
                print(f"    PASS  prim={d['material_prim_path']}  published={pub}  surf={d['surface_path']}")
            else:
                failures.append(f"prim path/publish wrong: path={d['material_prim_path']} published={pub}")
                print(f"    FAIL  path={d['material_prim_path']} published={pub}")
    except Exception as exc:
        failures.append(f"positive raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    print("\n  [2] textures + normal map wire without raising")
    try:
        r = dispatch("materials.create_mtlx_material",
                     {"name": "smoke_tex", "parent_path": "/stage",
                      "textures": {"base_color": "$HIP/tex/albedo.exr"},
                      "normal_map": "$HIP/tex/normal.exr"})
        if r.get("status") != "success":
            failures.append(f"textured failed: {r}")
            print(f"    FAIL  {r}")
        else:
            matlib = hou.node(r["data"]["materiallibrary_path"])
            bases = {c.type().name().split("::")[0] for c in matlib.children()}
            ok = "mtlxUsdUVTexture" in bases and "mtlxnormalmap" in bases
            print(f"    {'PASS' if ok else 'FAIL'}  child types include texture+normalmap: {ok}")
            if not ok:
                failures.append(f"texture/normal nodes missing: {bases}")
    except Exception as exc:
        failures.append(f"textured raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    print("\n  [3] negative: bogus parent -> status=error, no exception")
    try:
        r = dispatch("materials.create_mtlx_material",
                     {"name": "smoke_bad", "parent_path": "/stage/does_not_exist/deep"})
        if r.get("status") == "error":
            print(f"    PASS  status=error")
        else:
            failures.append(f"bogus parent should be error, got {r}")
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
    print("SMOKE PASSED — create_mtlx_material publishes + wires + fails loud (dispatcher path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
