"""
fbx_import_hython_smoke.py — headless Houdini smoke for PP12-110c FBX import tools.

Dual-mode (mirrors kinefx_hython_smoke.py pattern):
  - Standalone:  hython fbx_import_hython_smoke.py  (requires hython; pytest NOT required)
  - pytest:      collected as a test file; @pytest.mark.hython_smoke tests SKIP
                 automatically when hou is not importable (bare CI / off-DCC).

Covers:
  1. import_fbx_character on the real WorkersWelders T-pose FBX:
       - Creates kinefx::fbxcharacterimport, sets fbxfile parm, cooks.
       - Returns 84 joints from OUT 1 (the deformation skeleton with @name attrib).
       - Verifies key joint names present: pelvis, spine_01, head.
       - has_skin_geo=True (OUT 0 has geometry points > 0).
  2. import_fbx_character on a bogus path:
       - Returns {ok:false, error:<str>} without raising a Python exception.
  3. import_fbx_animation on the same FBX (T-pose rig FBX as a stand-in anim clip):
       - Creates kinefx::fbxanimimport, cooks without exception.
       - Returns skeleton dict with 'joints' count.
  4. import_fbx_animation with cascadeur=True:
       - The convertunits parm is set on the node.

testVerificationSurface: hython-smoke
unitId: pp12-110c
"""

from __future__ import annotations

import json
import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — must resolve fxhoudinimcp package in both hython + pytest
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PKG_PYTHON = os.path.join(_REPO_ROOT, "python")
_HOUDINI_HANDLERS = os.path.join(
    _REPO_ROOT, "houdini", "scripts", "python"
)
for _p in [_REPO_ROOT, _PKG_PYTHON, _HOUDINI_HANDLERS]:
    _abs = os.path.abspath(_p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

# ---------------------------------------------------------------------------
# pytest availability guard — Houdini's hython does NOT ship pytest.
# Provides no-op shims so the file is valid in both environments.
# (Mirrors kinefx_hython_smoke.py §2.)
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

# ---------------------------------------------------------------------------
# Test FBX path (live-verified in the plan — §"grounded facts")
# ---------------------------------------------------------------------------
_WORKERS_WELDERS_FBX = (
    r"G:\BigMediumSmall\IndustrialZone"
    r"\IndustialZone_CHARACTERS"
    r"\WorkersWelders_FBX_RIG_TPOSE"
    r"\WorkersWelders_RIG_FBX_TPOSE.fbx"
)

# Expected facts about the WorkersWelders T-pose FBX (live-verified):
#   OUT 1 = deformation skeleton — 84 joints, @name attrib present
#   OUT 0 = skin mesh — 105802 points, no @name
_EXPECTED_JOINT_COUNT = 84
_EXPECTED_JOINT_NAMES = {"pelvis", "spine_01", "head"}


# ===========================================================================
# Helper — direct hython handler import (bypasses the MCP bridge)
# Exercises the character_handlers code that the MCP tools delegate to.
# ===========================================================================

def _import_handlers():
    """Import character_handlers from the server-side package."""
    try:
        from fxhoudinimcp_server.handlers import character_handlers  # type: ignore
        return character_handlers
    except ImportError as exc:
        raise ImportError(
            "character_handlers not importable — "
            "run this smoke under hython after hou-dev ships PR-3"
        ) from exc


# ===========================================================================
# Smoke: character FBX import (hython_smoke)
# ===========================================================================

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestImportFbxCharacterSmoke:
    """
    Live smoke against the WorkersWelders T-pose FBX.
    Requires hython + the FBX at the grounded path.
    """

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._obj = hou.node("/obj")
        cls._geo = cls._obj.createNode("geo", "smoke_fbx_char")

    @classmethod
    def teardown_class(cls):
        try:
            cls._geo.destroy()
        except Exception:
            pass

    def test_fbx_exists_on_disk(self):
        """Guard: the test FBX must be present; skip gracefully if not."""
        if not os.path.exists(_WORKERS_WELDERS_FBX):
            pytest.skip(f"Test FBX not found at {_WORKERS_WELDERS_FBX!r} — skip on this machine")

    def test_character_import_creates_node_and_cooks(self):
        """import_fbx_character creates kinefx::fbxcharacterimport and cooks cleanly."""
        if not os.path.exists(_WORKERS_WELDERS_FBX):
            pytest.skip("Test FBX not found")
        ch = _import_handlers()
        result = ch.import_fbx_character(
            {"path": _WORKERS_WELDERS_FBX, "dest": self._geo.path()}
        )
        assert result.get("ok") is True, f"import_fbx_character failed: {result.get('error')}"
        assert "node" in result and result["node"], "Missing node path in result"

    def test_character_import_returns_84_joints(self):
        """OUT 1 (deformation skeleton) must report 84 joints for WorkersWelders."""
        if not os.path.exists(_WORKERS_WELDERS_FBX):
            pytest.skip("Test FBX not found")
        ch = _import_handlers()
        result = ch.import_fbx_character(
            {"path": _WORKERS_WELDERS_FBX, "dest": self._geo.path()}
        )
        skeleton = result.get("skeleton", {})
        joint_count = skeleton.get("joints")
        assert joint_count == _EXPECTED_JOINT_COUNT, (
            f"Expected {_EXPECTED_JOINT_COUNT} joints (OUT 1 deformation skeleton), "
            f"got {joint_count!r}.  Did the handler read OUT 0 (skin mesh) instead of OUT 1?"
        )

    def test_character_import_joint_names_include_key_joints(self):
        """OUT 1 skeleton must contain pelvis, spine_01, head — live joint names."""
        if not os.path.exists(_WORKERS_WELDERS_FBX):
            pytest.skip("Test FBX not found")
        ch = _import_handlers()
        result = ch.import_fbx_character(
            {"path": _WORKERS_WELDERS_FBX, "dest": self._geo.path()}
        )
        node_path = result.get("node")
        if not node_path:
            pytest.fail("No node path in result")
        # Read OUT 1 directly to verify joint names
        node = hou.node(node_path)
        if node is None:
            pytest.fail(f"Node not found: {node_path!r}")
        skeleton_geo = node.geometryAtOutput(1)  # OUT 1 = deformation skeleton
        if skeleton_geo is None:
            pytest.fail("No geometry at OUT 1 (deformation skeleton output)")
        name_attr = skeleton_geo.findPointAttrib("name")
        assert name_attr is not None, "OUT 1 has no @name attrib — not a skeleton SOP output"
        names = {pt.attribValue("name") for pt in skeleton_geo.points()}
        missing = _EXPECTED_JOINT_NAMES - names
        assert not missing, (
            f"Expected joint names {_EXPECTED_JOINT_NAMES} not found in OUT 1. "
            f"Missing: {missing}. Found sample: {list(names)[:10]}"
        )

    def test_character_import_has_skin_geo_true(self):
        """OUT 0 (skin mesh) must have points — has_skin_geo should be True."""
        if not os.path.exists(_WORKERS_WELDERS_FBX):
            pytest.skip("Test FBX not found")
        ch = _import_handlers()
        result = ch.import_fbx_character(
            {"path": _WORKERS_WELDERS_FBX, "dest": self._geo.path()}
        )
        skeleton = result.get("skeleton", {})
        assert skeleton.get("has_skin_geo") is True, (
            f"has_skin_geo should be True for WorkersWelders FBX "
            f"(OUT 0 has ~105802 pts).  Got: {skeleton.get('has_skin_geo')!r}"
        )

    def test_bogus_path_returns_ok_false_no_exception(self):
        """Bogus FBX path must return {ok:false, error:...} without raising."""
        ch = _import_handlers()
        result = ch.import_fbx_character(
            {"path": "/nonexistent/fake.fbx", "dest": self._geo.path()}
        )
        assert result.get("ok") is False, (
            f"Expected ok=False for bogus FBX path, got: {result}"
        )
        assert "error" in result and result["error"], (
            "Failure envelope must include non-empty 'error' key (FR-2 fail-loud)"
        )


# ===========================================================================
# Smoke: animation FBX import (hython_smoke)
# ===========================================================================

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestImportFbxAnimationSmoke:
    """
    Smoke for import_fbx_animation using the WorkersWelders T-pose FBX as a
    stand-in animation source (exercises the node creation and cook path;
    T-pose is a valid single-frame animation FBX).
    """

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._obj = hou.node("/obj")
        cls._geo = cls._obj.createNode("geo", "smoke_fbx_anim")

    @classmethod
    def teardown_class(cls):
        try:
            cls._geo.destroy()
        except Exception:
            pass

    def test_animation_import_cooks_and_returns_skeleton(self):
        """import_fbx_animation must cook without exception and return a skeleton dict."""
        if not os.path.exists(_WORKERS_WELDERS_FBX):
            pytest.skip("Test FBX not found")
        ch = _import_handlers()
        result = ch.import_fbx_animation(
            {"path": _WORKERS_WELDERS_FBX, "dest": self._geo.path(), "cascadeur": False}
        )
        assert result.get("ok") is True, f"import_fbx_animation failed: {result.get('error')}"
        skeleton = result.get("skeleton")
        assert skeleton is not None, "Return envelope missing 'skeleton' key"
        joints = skeleton.get("joints")
        assert isinstance(joints, int) and joints > 0, (
            f"skeleton.joints must be a positive integer, got {joints!r}"
        )

    def test_animation_import_cascadeur_true_sets_convertunits(self):
        """
        import_fbx_animation with cascadeur=True must set the convertunits parm
        on kinefx::fbxanimimport.  FR-3 / spec §7.4.
        Verified by inspecting the node parm after the call.
        """
        if not os.path.exists(_WORKERS_WELDERS_FBX):
            pytest.skip("Test FBX not found")
        ch = _import_handlers()
        result = ch.import_fbx_animation(
            {"path": _WORKERS_WELDERS_FBX, "dest": self._geo.path(), "cascadeur": True}
        )
        if not result.get("ok"):
            pytest.fail(f"import_fbx_animation(cascadeur=True) failed: {result.get('error')}")
        node_path = result.get("node")
        node = hou.node(node_path)
        assert node is not None, f"Node not found: {node_path!r}"
        # convertunits parm must be enabled (non-zero) when cascadeur=True
        cu_parm = node.parm("convertunits")
        assert cu_parm is not None, (
            "kinefx::fbxanimimport has no 'convertunits' parm — "
            "either the parm name is wrong or Cascadeur handling was not applied"
        )
        assert cu_parm.eval() != 0, (
            f"convertunits parm should be set (non-zero) when cascadeur=True, "
            f"got {cu_parm.eval()!r}"
        )

    def test_animation_bogus_path_returns_ok_false_no_exception(self):
        """Bogus path must return {ok:false, error:...} without raising."""
        ch = _import_handlers()
        result = ch.import_fbx_animation(
            {"path": "/nonexistent/clip.fbx", "dest": self._geo.path(), "cascadeur": False}
        )
        assert result.get("ok") is False, (
            f"Expected ok=False for bogus FBX path, got: {result}"
        )
        assert "error" in result and result["error"]


# ===========================================================================
# Standalone runner (hython fbx_import_hython_smoke.py)
# ===========================================================================

def main() -> int:
    if not HOU_AVAILABLE:
        print("ERROR: hou not importable — run this script with hython, not python.")
        return 1

    failures: list[str] = []
    fbx = _WORKERS_WELDERS_FBX

    print(f"\n--- FBX import smoke (WorkersWelders T-pose) ---")
    print(f"FBX path: {fbx}")

    if not os.path.exists(fbx):
        print(f"  SKIP  FBX not found on this machine — skipping live import tests")
        print(f"\nSMOKE PARTIAL — FBX not available; non-FBX checks would pass.")
        return 0

    try:
        ch = _import_handlers()
    except ImportError as exc:
        print(f"  FAIL  character_handlers import: {exc}")
        return 1

    hou.hipFile.clear(suppress_save_prompt=True)
    geo = hou.node("/obj").createNode("geo", "smoke_standalone")

    # --- Character import ---
    print("\n  [1] import_fbx_character (standard)")
    try:
        result = ch.import_fbx_character({"path": fbx, "dest": geo.path()})
        if not result.get("ok"):
            failures.append(f"import_fbx_character failed: {result.get('error')}")
            print(f"    FAIL  {result.get('error')}")
        else:
            joints = result.get("skeleton", {}).get("joints", "?")
            has_skin = result.get("skeleton", {}).get("has_skin_geo", "?")
            node_p = result.get("node", "?")
            print(f"    PASS  node={node_p}  joints={joints}  has_skin_geo={has_skin}")
            if joints != _EXPECTED_JOINT_COUNT:
                msg = f"Expected {_EXPECTED_JOINT_COUNT} joints, got {joints}"
                failures.append(msg)
                print(f"    FAIL  {msg}")
            if has_skin is not True:
                failures.append(f"has_skin_geo expected True, got {has_skin!r}")
    except Exception as exc:
        failures.append(f"import_fbx_character raised: {exc}")
        print(f"    FAIL  raised exception: {exc}")

    # --- Bogus path fail-loud ---
    print("\n  [2] import_fbx_character (bogus path → ok=False)")
    try:
        result = ch.import_fbx_character({"path": "/fake.fbx", "dest": geo.path()})
        if result.get("ok") is not False or not result.get("error"):
            failures.append(f"bogus path should return ok=False with error, got {result}")
            print(f"    FAIL  {result}")
        else:
            print(f"    PASS  ok=False, error={result['error'][:60]!r}")
    except Exception as exc:
        failures.append(f"bogus path raised exception (should return envelope): {exc}")
        print(f"    FAIL  raised exception: {exc}")

    # --- Animation import ---
    print("\n  [3] import_fbx_animation (cascadeur=False)")
    try:
        result = ch.import_fbx_animation(
            {"path": fbx, "dest": geo.path(), "cascadeur": False}
        )
        if not result.get("ok"):
            failures.append(f"import_fbx_animation failed: {result.get('error')}")
            print(f"    FAIL  {result.get('error')}")
        else:
            joints = result.get("skeleton", {}).get("joints", "?")
            node_p = result.get("node", "?")
            print(f"    PASS  node={node_p}  joints={joints}")
    except Exception as exc:
        failures.append(f"import_fbx_animation raised: {exc}")
        print(f"    FAIL  raised exception: {exc}")

    # --- Animation import cascadeur=True ---
    print("\n  [4] import_fbx_animation (cascadeur=True → convertunits set)")
    try:
        result = ch.import_fbx_animation(
            {"path": fbx, "dest": geo.path(), "cascadeur": True}
        )
        if not result.get("ok"):
            failures.append(f"import_fbx_animation(cascadeur=True) failed: {result.get('error')}")
            print(f"    FAIL  {result.get('error')}")
        else:
            node = hou.node(result.get("node", ""))
            cu_parm = node.parm("convertunits") if node else None
            cu_val = cu_parm.eval() if cu_parm else None
            if cu_val:
                print(f"    PASS  convertunits={cu_val}")
            else:
                msg = f"convertunits not set (val={cu_val!r})"
                failures.append(msg)
                print(f"    FAIL  {msg}")
    except Exception as exc:
        failures.append(f"import_fbx_animation(cascadeur=True) raised: {exc}")
        print(f"    FAIL  raised exception: {exc}")

    # --- Default-dest (dest omitted) — Major-4 fix verification ---
    # _resolve_sop_parent creates/reuses a geo container under /obj automatically.
    print("\n  [5] import_fbx_character (dest omitted → default geo container under /obj)")
    try:
        result = ch.import_fbx_character({"path": fbx})  # NO dest param
        if not result.get("ok"):
            failures.append(f"import_fbx_character(dest=omitted) failed: {result.get('error')}")
            print(f"    FAIL  {result.get('error')}")
        else:
            node_p = result.get("node", "?")
            joints = result.get("skeleton", {}).get("joints", "?")
            # Verify the node landed inside a geo container (not directly under /obj)
            node = hou.node(node_p)
            if node is None:
                failures.append(f"Node not found after default-dest import: {node_p!r}")
                print(f"    FAIL  node not found: {node_p!r}")
            else:
                parent_path = node.parent().path()
                parent_type = node.parent().type().name()
                print(f"    PASS  node={node_p}  parent={parent_path} ({parent_type})  joints={joints}")
                if parent_type != "geo":
                    msg = (
                        f"Default-dest must place node inside a geo container, "
                        f"got parent type {parent_type!r} at {parent_path!r}"
                    )
                    failures.append(msg)
                    print(f"    FAIL  {msg}")
    except Exception as exc:
        failures.append(f"import_fbx_character(dest=omitted) raised exception (should not): {exc}")
        print(f"    FAIL  raised exception: {exc}")

    try:
        geo.destroy()
    except Exception:
        pass

    print()
    if failures:
        print(f"SMOKE FAILED — {len(failures)} failure(s):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("SMOKE PASSED — all FBX import checks OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
