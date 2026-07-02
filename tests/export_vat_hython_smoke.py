"""
export_vat_hython_smoke.py -- dispatcher-based headless smoke for PP12-111 PR-4 VAT bake.

Dual-mode (mirrors export_hython_smoke.py pattern exactly):
  - Standalone:  hython tests/export_vat_hython_smoke.py  (requires hython; pytest NOT required)
  - pytest:      collected as a test file; tests SKIP automatically when hou is not
                 importable (bare CI / off-DCC).

Covers (dispatcher-based -- same real production path as all other PP12 smokes):
  1. export_vat (box animated fixture, export_type='soft', frame_range=[1,5]):
     - ok=True
     - mesh artifact (.fbx) exists on disk
     - textures list has at least pos + rot entries on disk
     - sidecar (.export.json) exists and parses to dict with
         tool == "houdini_export_vat"
         version_triple present with non-null houdini str

NOTE: All calls go through the REAL dispatcher path:
    dispatch("export_vat", {"node": path, "out_dir": dir, ...})
This calls handler(**params), which is the authoritative production path.

Handler registration: importing export_handlers fires register_handler() for
export_vat -- exactly as the real server does at startup.

At RED phase: dispatch("export_vat", ...) fails with "unknown command export_vat"
because the handler does not yet exist.  This is the intended RED failure.

testVerificationSurface: hython-smoke
unitId: pp12-111d
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# Path bootstrap -- resolves fxhoudinimcp packages in both hython + pytest.
# Mirrors export_hython_smoke.py exactly (3-path bootstrap).
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PKG_PYTHON = os.path.join(_REPO_ROOT, "python")
_HOUDINI_HANDLERS = os.path.join(_REPO_ROOT, "houdini", "scripts", "python")
for _p in [_REPO_ROOT, _PKG_PYTHON, _HOUDINI_HANDLERS]:
    _abs = os.path.abspath(_p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

# ---------------------------------------------------------------------------
# pytest availability guard -- Houdini's hython does NOT ship pytest.
# Provides no-op shims so the file is valid in both environments.
# ---------------------------------------------------------------------------
try:
    import pytest  # type: ignore[import-not-found]
    _PYTEST_AVAILABLE = True
except ImportError:
    _PYTEST_AVAILABLE = False

    class _SkipIfDecorator:
        """No-op replacement for pytest.mark.skipif (hython standalone mode)."""
        def __call__(self, condition, reason=""):
            def decorator(fn_or_cls):
                return fn_or_cls
            return decorator

    class _MarkNS:
        """Minimal namespace shim for pytest.mark (hython standalone mode)."""
        skipif = _SkipIfDecorator()

        def __getattr__(self, name):
            def decorator_factory(*args, **kwargs):
                def decorator(fn_or_cls):
                    return fn_or_cls
                return decorator
            return decorator_factory

    class pytest:  # type: ignore[no-redef]
        """Minimal pytest shim for hython standalone execution."""
        mark = _MarkNS()

        @staticmethod
        def fail(msg: str) -> None:
            raise AssertionError(msg)

        @staticmethod
        def skip(msg: str = "") -> None:
            raise RuntimeError(f"SKIP: {msg}")

# ---------------------------------------------------------------------------
# hou availability guard
# ---------------------------------------------------------------------------
try:
    import hou  # type: ignore[import-not-found]
    HOU_AVAILABLE = True
except ImportError:
    HOU_AVAILABLE = False

# pytestmark is only meaningful when collected by pytest.
if _PYTEST_AVAILABLE:
    pytestmark = pytest.mark.hython_smoke


# ===========================================================================
# Dispatcher helper -- the REAL production path.
#
# Importing export_handlers fires register_handler() for export_vat,
# exactly as the real Houdini server does at startup.
# Without this import, available_commands is EMPTY and every dispatch()
# call returns "unknown command export_vat".
# ===========================================================================

def _get_dispatcher():
    """Import and return the dispatcher, registering export handlers.

    Importing export_handlers fires register_handler() for export_vat
    (and the PR-3 handlers probe_versions + validate_budget) -- exactly
    as the real server does at startup.
    """
    try:
        from fxhoudinimcp_server.handlers import export_handlers  # noqa: F401
        from fxhoudinimcp_server.dispatcher import dispatch
        return dispatch
    except ImportError as exc:
        raise ImportError(
            "dispatcher or export_handlers not importable -- "
            "run this smoke under hython with the bootstrap paths set"
        ) from exc


def _data(result):
    """Extract handler result from dispatcher envelope.

    dispatch() returns {"status": "success", "data": {...}, "timing_ms": N}.
    Assertions must be against the UNWRAPPED data dict, not the envelope.
    """
    return result.get("data", result)


# ===========================================================================
# Fixture builder helper
#
# Builds a minimal animated SOP fixture:
#   geo (geo1)
#     box (box1)   -- static box geometry
#     wrangle      -- @P.y += @Frame * 0.1; (1 keyframe of motion per frame)
#
# This gives the VAT ROP something to bake over frame_range=[1, 5].
# ===========================================================================

def _build_animated_fixture(parent_name: str = "geo_vat_smoke"):
    """Create an animated box fixture in /obj and return the SOP path.

    Returns:
        tuple[hou.Node, str]: (geo node to destroy later, SOP output path)
    """
    geo = hou.node("/obj").createNode("geo", parent_name)
    box = geo.createNode("box", "box1")
    wr = geo.createNode("attribwrangle", "anim_wrangle")
    wr.setFirstInput(box)
    # Point-class wrangle: move points up by Frame * 0.1 each frame
    wr.parm("class").set(0)   # 0 = point class
    wr.parm("snippet").set("@P.y += @Frame * 0.1;")
    wr.cook(force=True)
    return geo, wr.path()


# ===========================================================================
# Smoke test class
# ===========================================================================

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available -- requires hython")
class TestExportVatBake:
    """export_vat dispatcher bake smoke: animated box, export_type='soft', frames 1-5."""

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._dispatch = _get_dispatcher()
        cls._out_dir = tempfile.mkdtemp(prefix="vat_smoke_")
        cls._geo, cls._sop_path = _build_animated_fixture("geo_vat_smoke")
        # Fire the bake once and cache the result for all tests in this class.
        raw = cls._dispatch(
            "export_vat",
            {
                "node": cls._sop_path,
                "out_dir": cls._out_dir,
                "export_type": "soft",
                "frame_range": [1, 5],
            },
        )
        cls._result = _data(raw)

    @classmethod
    def teardown_class(cls):
        try:
            cls._geo.destroy()
        except Exception:
            pass
        # Leave out_dir for inspection; tmpdir is cleaned by the OS.

    # -----------------------------------------------------------------------
    # 1. Envelope: ok=True
    # -----------------------------------------------------------------------

    def test_result_is_dict(self):
        """dispatch('export_vat', ...) must return a dict."""
        assert isinstance(self._result, dict), (
            f"export_vat result must be a dict, got {type(self._result).__name__!r}"
        )

    def test_result_ok_true(self):
        """Result ok field must be True (bake succeeded without error)."""
        ok = self._result.get("ok")
        assert ok is True, (
            f"export_vat result['ok'] must be True; got {ok!r}\n"
            f"full result: {self._result}"
        )

    # -----------------------------------------------------------------------
    # 2. Mesh artifact (.fbx) exists on disk
    # -----------------------------------------------------------------------

    def test_mesh_path_present(self):
        """Result contains a 'mesh' key with the .fbx file path."""
        assert "mesh" in self._result, (
            f"export_vat result must contain 'mesh' key; "
            f"keys present: {sorted(self._result.keys())}"
        )

    def test_mesh_file_exists(self):
        """The .fbx mesh file referenced in result['mesh'] exists on disk."""
        mesh_path = self._result.get("mesh", "")
        assert mesh_path, "result['mesh'] must be a non-empty path string"
        assert os.path.isfile(mesh_path), (
            f"Mesh file does not exist on disk: {mesh_path!r}\n"
            f"out_dir contents: {os.listdir(self._out_dir)}"
        )

    def test_mesh_has_fbx_extension(self):
        """The mesh file must have a .fbx extension."""
        mesh_path = self._result.get("mesh", "")
        assert mesh_path.lower().endswith(".fbx"), (
            f"Mesh file must have .fbx extension, got: {mesh_path!r}"
        )

    # -----------------------------------------------------------------------
    # 3. Textures: pos + rot entries exist on disk
    # -----------------------------------------------------------------------

    def test_textures_key_present(self):
        """Result contains a 'textures' key."""
        assert "textures" in self._result, (
            f"export_vat result must contain 'textures' key; "
            f"keys present: {sorted(self._result.keys())}"
        )

    def test_textures_is_list(self):
        """result['textures'] must be a list."""
        textures = self._result.get("textures")
        assert isinstance(textures, list), (
            f"result['textures'] must be a list, got {type(textures).__name__!r}"
        )

    def test_textures_has_pos_entry(self):
        """Textures list must contain at least one path with 'pos' in the filename."""
        textures = self._result.get("textures", [])
        pos_entries = [t for t in textures if "pos" in os.path.basename(t).lower()]
        assert pos_entries, (
            f"Textures list must contain a position texture ('pos' in filename); "
            f"textures: {textures}"
        )

    def test_textures_has_rot_entry(self):
        """Textures list must contain at least one path with 'rot' in the filename."""
        textures = self._result.get("textures", [])
        rot_entries = [t for t in textures if "rot" in os.path.basename(t).lower()]
        assert rot_entries, (
            f"Textures list must contain a rotation texture ('rot' in filename); "
            f"textures: {textures}"
        )

    def test_textures_pos_file_exists_on_disk(self):
        """The position texture file must exist on disk."""
        textures = self._result.get("textures", [])
        pos_entries = [t for t in textures if "pos" in os.path.basename(t).lower()]
        assert pos_entries, "No pos texture in result"
        for p in pos_entries:
            assert os.path.isfile(p), (
                f"Position texture does not exist on disk: {p!r}"
            )

    def test_textures_rot_file_exists_on_disk(self):
        """The rotation texture file must exist on disk."""
        textures = self._result.get("textures", [])
        rot_entries = [t for t in textures if "rot" in os.path.basename(t).lower()]
        assert rot_entries, "No rot texture in result"
        for p in rot_entries:
            assert os.path.isfile(p), (
                f"Rotation texture does not exist on disk: {p!r}"
            )

    # -----------------------------------------------------------------------
    # 4. Sidecar (.export.json) exists and parses correctly
    # -----------------------------------------------------------------------

    def test_sidecar_key_present(self):
        """Result contains a 'sidecar' key with the .export.json path."""
        assert "sidecar" in self._result, (
            f"export_vat result must contain 'sidecar' key; "
            f"keys present: {sorted(self._result.keys())}"
        )

    def test_sidecar_file_exists(self):
        """The .export.json sidecar file exists on disk."""
        sidecar_path = self._result.get("sidecar", "")
        assert sidecar_path, "result['sidecar'] must be a non-empty path string"
        assert os.path.isfile(sidecar_path), (
            f"Sidecar file does not exist on disk: {sidecar_path!r}\n"
            f"out_dir contents: {os.listdir(self._out_dir)}"
        )

    def test_sidecar_is_valid_json(self):
        """The sidecar file parses as valid JSON."""
        sidecar_path = self._result.get("sidecar", "")
        assert sidecar_path and os.path.isfile(sidecar_path), "sidecar file missing"
        with open(sidecar_path, encoding="utf-8") as fh:
            data = json.load(fh)  # raises json.JSONDecodeError on malformed JSON
        assert isinstance(data, dict), (
            f"Sidecar JSON must be a dict, got {type(data).__name__!r}"
        )

    def test_sidecar_tool_is_houdini_export_vat(self):
        """Sidecar JSON must have tool == 'houdini_export_vat' (FR-8 sidecar contract)."""
        sidecar_path = self._result.get("sidecar", "")
        assert sidecar_path and os.path.isfile(sidecar_path), "sidecar file missing"
        with open(sidecar_path, encoding="utf-8") as fh:
            data = json.load(fh)
        tool = data.get("tool")
        assert tool == "houdini_export_vat", (
            f"sidecar['tool'] must be 'houdini_export_vat', got {tool!r}"
        )

    def test_sidecar_version_triple_present(self):
        """Sidecar JSON must contain a 'version_triple' dict."""
        sidecar_path = self._result.get("sidecar", "")
        assert sidecar_path and os.path.isfile(sidecar_path), "sidecar file missing"
        with open(sidecar_path, encoding="utf-8") as fh:
            data = json.load(fh)
        vt = data.get("version_triple")
        assert isinstance(vt, dict), (
            f"sidecar['version_triple'] must be a dict, got {type(vt).__name__!r}: {vt!r}"
        )

    def test_sidecar_version_triple_houdini_non_null(self):
        """sidecar['version_triple']['houdini'] must be a non-empty string."""
        sidecar_path = self._result.get("sidecar", "")
        assert sidecar_path and os.path.isfile(sidecar_path), "sidecar file missing"
        with open(sidecar_path, encoding="utf-8") as fh:
            data = json.load(fh)
        vt = data.get("version_triple", {})
        houdini_ver = vt.get("houdini")
        assert isinstance(houdini_ver, str) and houdini_ver, (
            f"sidecar['version_triple']['houdini'] must be a non-empty string, "
            f"got {houdini_ver!r}"
        )


# ===========================================================================
# Standalone runner (hython tests/export_vat_hython_smoke.py)
# ===========================================================================

def main() -> int:
    if not HOU_AVAILABLE:
        print("ERROR: hou not importable -- run this script with hython, not python.")
        return 1

    failures: list[str] = []

    print("\n--- export_vat smoke (PP12-111d, PR-4) ---")
    print("Using: dispatch() -> handler(**params) [real dispatcher path]")
    print("RED-phase expected failure: 'unknown command export_vat'")

    try:
        dispatch = _get_dispatcher()
    except ImportError as exc:
        print(f"  FAIL  dispatcher/export_handlers import: {exc}")
        return 1

    # ---- Build animated fixture ------------------------------------------
    print("\n  [setup] Build animated box fixture (box + P.y += @Frame * 0.1)")
    try:
        hou.hipFile.clear(suppress_save_prompt=True)
        geo, sop_path = _build_animated_fixture("geo_vat_smoke")
        out_dir = tempfile.mkdtemp(prefix="vat_smoke_")
        print(f"    fixture: {sop_path}")
        print(f"    out_dir: {out_dir}")
    except Exception as exc:
        print(f"    FAIL  fixture build raised: {exc}")
        return 1

    # ---- [1] export_vat dispatch ------------------------------------------
    print("\n  [1] dispatch('export_vat', {node, out_dir, export_type='soft', frame_range=[1,5]})")
    try:
        raw = dispatch(
            "export_vat",
            {
                "node": sop_path,
                "out_dir": out_dir,
                "export_type": "soft",
                "frame_range": [1, 5],
            },
        )
        result = _data(raw)
        ok = True

        # envelope
        if not isinstance(result, dict):
            failures.append(f"[1] result is not a dict: {type(result).__name__!r}")
            print(f"    FAIL  result is {type(result).__name__!r}")
            ok = False

        if ok and result.get("ok") is not True:
            failures.append(f"[1] result['ok'] is not True: {result.get('ok')!r}")
            print(f"    FAIL  ok={result.get('ok')!r}")
            ok = False

        # mesh
        mesh_path = result.get("mesh", "") if ok else ""
        if ok:
            if not mesh_path:
                failures.append("[1] result missing 'mesh' path")
                print("    FAIL  no 'mesh' in result")
                ok = False
            elif not os.path.isfile(mesh_path):
                failures.append(f"[1] mesh file not on disk: {mesh_path!r}")
                print(f"    FAIL  mesh not found: {mesh_path!r}")
                ok = False
            else:
                print(f"    OK    mesh={mesh_path!r}")

        # textures
        textures = result.get("textures", []) if ok else []
        if ok:
            if not isinstance(textures, list):
                failures.append(f"[1] result['textures'] is not a list: {type(textures).__name__!r}")
                print(f"    FAIL  textures not a list")
                ok = False
            else:
                pos_entries = [t for t in textures if "pos" in os.path.basename(t).lower()]
                rot_entries = [t for t in textures if "rot" in os.path.basename(t).lower()]
                if not pos_entries:
                    failures.append("[1] no pos texture in result['textures']")
                    print(f"    FAIL  no pos texture; textures: {textures}")
                    ok = False
                if not rot_entries:
                    failures.append("[1] no rot texture in result['textures']")
                    print(f"    FAIL  no rot texture; textures: {textures}")
                    ok = False
                if ok:
                    for p in pos_entries + rot_entries:
                        if not os.path.isfile(p):
                            failures.append(f"[1] texture not on disk: {p!r}")
                            print(f"    FAIL  texture not on disk: {p!r}")
                            ok = False
                            break
                if ok:
                    print(
                        f"    OK    textures: {len(textures)} files; "
                        f"pos={[os.path.basename(t) for t in pos_entries]}  "
                        f"rot={[os.path.basename(t) for t in rot_entries]}"
                    )

        # sidecar
        sidecar_path = result.get("sidecar", "") if ok else ""
        if ok:
            if not sidecar_path or not os.path.isfile(sidecar_path):
                failures.append(f"[1] sidecar not on disk: {sidecar_path!r}")
                print(f"    FAIL  sidecar not found: {sidecar_path!r}")
                ok = False
            else:
                with open(sidecar_path, encoding="utf-8") as fh:
                    sidecar_data = json.load(fh)
                tool = sidecar_data.get("tool")
                vt = sidecar_data.get("version_triple", {})
                hou_ver = vt.get("houdini") if isinstance(vt, dict) else None
                if tool != "houdini_export_vat":
                    failures.append(f"[1] sidecar tool={tool!r} (expected 'houdini_export_vat')")
                    print(f"    FAIL  sidecar tool={tool!r}")
                    ok = False
                if not (isinstance(hou_ver, str) and hou_ver):
                    failures.append(f"[1] sidecar version_triple.houdini={hou_ver!r} (expected non-empty str)")
                    print(f"    FAIL  version_triple.houdini={hou_ver!r}")
                    ok = False
                if ok:
                    print(
                        f"    OK    sidecar tool={tool!r}  houdini={hou_ver!r}"
                    )
    except Exception as exc:
        failures.append(f"[1] export_vat raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    # ---- Teardown --------------------------------------------------------
    try:
        geo.destroy()
    except Exception:
        pass

    print()
    total = 1
    group_failures = set()
    for f in failures:
        if f.startswith("["):
            group_failures.add(f[1])
    groups_passed = total - len(group_failures)
    print(f"Result: {groups_passed}/{total} passed")
    if failures:
        print("SMOKE: FAIL")
        for f in failures:
            print(f"  {f}")
        return 1
    print("SMOKE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
