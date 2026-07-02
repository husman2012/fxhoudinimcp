"""
bonedeform_hython_smoke.py — headless Houdini smoke for PP12-110d bonedeform tool.

Dual-mode (mirrors fbx_import_hython_smoke.py pattern):
  - Standalone:  hython bonedeform_hython_smoke.py  (requires hython; pytest NOT required)
  - pytest:      collected as a test file; @pytest.mark.hython_smoke tests SKIP
                 automatically when hou is not importable (bare CI / off-DCC).

Covers (dispatcher-based — fix#007 real production path):
  1. setup_bonedeform positive: WorkersWelders FBX character + anim + geo wired
       - dispatch("setup_bonedeform", {rest, anim, geo, dest}) → ok=True.
       - validator.deformed_points > 0.
       - validator.has_capture_weight mirrors capture state of INPUT geo.
  2. FR-5 negative: geo node lacks boneCapture_* attribs → handler still returns
       ok=True but has_capture_weight=False (no explicit pre-check needed when
       bonedeform can deform uncaptured geo; if bonedeform errors, the cook-error
       path catches it via the outer FR-2 envelope).
  3. Bogus-path negative: invalid rest/anim/geo paths → ok=False without exception.

NOTE (fix#007): All calls go through the REAL dispatcher path:
    dispatch("setup_bonedeform", {"rest": ..., "anim": ..., "geo": ..., "dest": ...})
This calls handler(**params), which is the authoritative production path.

Fixture note: The WorkersWelders test depends on the live FBX at the grounded path.
If the FBX is absent on the current machine, the live-fixture tests skip gracefully
and only the bogus-path negative test runs.

testVerificationSurface: hython-smoke
unitId: pp12-110d
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — resolves fxhoudinimcp package in both hython + pytest
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PKG_PYTHON = os.path.join(_REPO_ROOT, "python")
_HOUDINI_HANDLERS = os.path.join(_REPO_ROOT, "houdini", "scripts", "python")
for _p in [_REPO_ROOT, _PKG_PYTHON, _HOUDINI_HANDLERS]:
    _abs = os.path.abspath(_p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

# ---------------------------------------------------------------------------
# pytest availability guard — Houdini's hython does NOT ship pytest.
# Provides no-op shims so the file is valid in both environments.
# (Mirrors fbx_import_hython_smoke.py §2.)
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
# Test FBX path (live-verified in the plan — grounded facts)
# ---------------------------------------------------------------------------
_WORKERS_WELDERS_FBX = (
    r"G:\BigMediumSmall\IndustrialZone"
    r"\IndustialZone_CHARACTERS"
    r"\WorkersWelders_FBX_RIG_TPOSE"
    r"\WorkersWelders_RIG_FBX_TPOSE.fbx"
)

# Expected facts about WorkersWelders (live-verified by PR-3 smoke):
#   OUT 1 deformation skeleton = 84 joints
_EXPECTED_JOINT_COUNT = 84


# ===========================================================================
# Dispatcher helper — the REAL production path (fix#007)
# ===========================================================================

def _get_dispatcher():
    """Import and return the dispatcher, registering all handlers.

    Importing character_handlers fires register_handler() for all five handlers
    (kinefx_probe, query_skeleton, inspect_apex, import_fbx_character,
    import_fbx_animation) PLUS the new setup_bonedeform (PR-4), exactly as the
    real Houdini server does at startup.
    """
    try:
        from fxhoudinimcp_server.handlers import character_handlers  # noqa: F401
        from fxhoudinimcp_server.dispatcher import dispatch
        return dispatch
    except ImportError as exc:
        raise ImportError(
            "dispatcher or character_handlers not importable — "
            "run this smoke under hython after PR-4 ships"
        ) from exc


# ===========================================================================
# Smoke: bonedeform positive (hython_smoke)
# ===========================================================================

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestSetupBonedeformSmoke:
    """
    Live smoke for setup_bonedeform using WorkersWelders character FBX.
    Strategy: import the FBX as both character (OUT 1 skeleton) and anim
    (OUT 0 animated skeleton), then wire them into a bonedeform.
    All calls go through dispatch("setup_bonedeform", {...}) — real dispatcher.

    Fixture: WorkersWelders T-pose FBX at the grounded path.
    If absent, tests skip gracefully.
    """

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._obj = hou.node("/obj")
        cls._geo = cls._obj.createNode("geo", "smoke_bonedeform")
        cls._dispatch = _get_dispatcher()

        # Build skeleton + captured geo nodes using the FBX importer tools
        # (already smoke-tested in PR-3). These are used as inputs to bonedeform.
        cls._rest_node = None
        cls._anim_node = None
        cls._geo_node = None  # captured skin geo

    @classmethod
    def teardown_class(cls):
        try:
            cls._geo.destroy()
        except Exception:
            pass

    def _ensure_fbx_imported(self):
        """Import WorkersWelders once into the smoke geo container.

        Returns True if nodes are ready; False if the FBX is absent.
        """
        if self.__class__._rest_node is not None:
            return True
        if not os.path.exists(_WORKERS_WELDERS_FBX):
            return False

        # Import character rig — OUT 0 = skin geo, OUT 1 = deformation skeleton
        result = self._dispatch(
            "import_fbx_character",
            {"path": _WORKERS_WELDERS_FBX, "dest": self._geo.path()},
        )
        data = result.get("data", result)
        if not data.get("ok"):
            pytest.fail(f"FBX import failed during smoke setup: {data.get('error')}")

        char_node = hou.node(data["node"])
        if char_node is None:
            pytest.fail(f"character import node not found: {data['node']!r}")

        # Import anim FBX (T-pose as single-frame stand-in) for the animated skeleton
        anim_result = self._dispatch(
            "import_fbx_animation",
            {"path": _WORKERS_WELDERS_FBX, "dest": self._geo.path(), "cascadeur": False},
        )
        anim_data = anim_result.get("data", anim_result)
        if not anim_data.get("ok"):
            pytest.fail(f"FBX anim import failed during smoke setup: {anim_data.get('error')}")

        anim_node = hou.node(anim_data["node"])
        if anim_node is None:
            pytest.fail(f"anim import node not found: {anim_data['node']!r}")

        # OUT 1 of the character import is the deformation skeleton (rest pose).
        # We use an attribute wrangle to expose output 1 as a separate node.
        # For bonedeform smoke: rest=OUT1 of char import, anim=OUT0 of anim import,
        # geo=OUT0 of char import (skin mesh).
        # Use hou.SopNode.geometry(idx) to verify outputs exist.
        skel_geo = char_node.geometry(1)
        if skel_geo is None or skel_geo.intrinsicValue("pointcount") == 0:
            pytest.fail("Character import OUT 1 (skeleton) has no points — can't setup bonedeform")

        # Null pass-throughs to pin each output as an accessible scene path.
        skel_null = self._geo.createNode("null", "smoke_rest_skel")
        skel_null.setInput(0, char_node, 1)  # OUT 1 = deformation skeleton

        skin_null = self._geo.createNode("null", "smoke_skin_geo")
        skin_null.setInput(0, char_node, 0)  # OUT 0 = skin mesh

        anim_null = self._geo.createNode("null", "smoke_anim_skel")
        anim_null.setInput(0, anim_node, 0)  # OUT 0 = animated skeleton

        # Cook the nulls so geometry is accessible.
        for n in (skel_null, skin_null, anim_null):
            n.cook(force=True)

        self.__class__._rest_node = skel_null
        self.__class__._anim_node = anim_null
        self.__class__._geo_node = skin_null
        return True

    def test_fbx_exists_on_disk(self):
        """Guard: the test FBX must be present; skip gracefully if not."""
        if not os.path.exists(_WORKERS_WELDERS_FBX):
            pytest.skip(f"Test FBX not found at {_WORKERS_WELDERS_FBX!r} — skip on this machine")

    def test_setup_bonedeform_creates_node_and_cooks(self):
        """setup_bonedeform must create a bonedeform node and cook without error."""
        if not self._ensure_fbx_imported():
            pytest.skip("Test FBX not found — skipping live bonedeform smoke")

        result = self._dispatch(
            "setup_bonedeform",
            {
                "rest": self._rest_node.path(),
                "anim": self._anim_node.path(),
                "geo": self._geo_node.path(),
                "dest": self._geo.path(),
            },
        )
        data = result.get("data", result)
        assert data.get("ok") is True, f"setup_bonedeform failed: {data.get('error')}"
        assert "node" in data and data["node"], "Missing node path in result"

    def test_setup_bonedeform_returns_validator_dict(self):
        """Return envelope must contain validator sub-dict with expected keys (FR-12)."""
        if not self._ensure_fbx_imported():
            pytest.skip("Test FBX not found")

        result = self._dispatch(
            "setup_bonedeform",
            {
                "rest": self._rest_node.path(),
                "anim": self._anim_node.path(),
                "geo": self._geo_node.path(),
                "dest": self._geo.path(),
            },
        )
        data = result.get("data", result)
        assert data.get("ok") is True, f"setup_bonedeform failed: {data.get('error')}"

        validator = data.get("validator")
        assert validator is not None, "Return envelope missing 'validator' key"
        assert "cook_errors" in validator, "validator missing 'cook_errors'"
        assert "deformed_points" in validator, "validator missing 'deformed_points'"
        assert "has_capture_weight" in validator, "validator missing 'has_capture_weight'"
        assert isinstance(validator["deformed_points"], int) and validator["deformed_points"] > 0, (
            f"deformed_points should be > 0 for a wired bonedeform, got {validator['deformed_points']!r}"
        )
        # FR-12 / pp12-110d fix#1: WorkersWelders skin has boneCapture — must be True.
        assert validator.get("has_capture_weight") is True, (
            f"has_capture_weight must be True for a KineFX-captured skin; got "
            f"{validator.get('has_capture_weight')!r}. "
            "KineFX stores capture as 'boneCapture' (no _weight/_index suffix). "
            "Handler must use startswith('boneCapture') prefix check on INPUT geo_node."
        )

    def test_setup_bonedeform_skeleton_joints_from_anim(self):
        """skeleton.joints must be read from anim node @name point attrib."""
        if not self._ensure_fbx_imported():
            pytest.skip("Test FBX not found")

        result = self._dispatch(
            "setup_bonedeform",
            {
                "rest": self._rest_node.path(),
                "anim": self._anim_node.path(),
                "geo": self._geo_node.path(),
                "dest": self._geo.path(),
            },
        )
        data = result.get("data", result)
        assert data.get("ok") is True, f"setup_bonedeform failed: {data.get('error')}"

        skeleton = data.get("skeleton", {})
        joints = skeleton.get("joints")
        assert isinstance(joints, int) and joints > 0, (
            f"skeleton.joints must be a positive int from anim @name attr, got {joints!r}"
        )


# ===========================================================================
# Smoke: FR-5 negative — geo lacks capture attributes
# ===========================================================================

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestSetupBonedeformNoCapture:
    """
    FR-5 smoke: when geo lacks boneCapture_* attribs, has_capture_weight=False.

    Uses synthetic geometry (no FBX dependency) — creates a simple box as geo,
    a null as rest skeleton, a null as anim skeleton, and confirms the validator
    correctly reports has_capture_weight=False because the INPUT geo has no
    boneCapture_* point attribs.

    NOTE: This test intentionally passes uncaptured geo to bonedeform.
    bonedeform may or may not error in this case — if it errors, the FR-2
    outer envelope returns ok=False (acceptable); if it succeeds, the validator
    MUST report has_capture_weight=False (the critical FR-5 invariant).
    The test only asserts the has_capture_weight value when ok=True.
    """

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._obj = hou.node("/obj")
        cls._geo = cls._obj.createNode("geo", "smoke_nocap")
        cls._dispatch = _get_dispatcher()

        # Synthetic geo: a box with no boneCapture_* attribs
        cls._box = cls._geo.createNode("box", "smoke_uncaptured_box")
        cls._box.cook(force=True)

        # Minimal rest + anim skeleton stubs: nulls (no geometry)
        cls._rest_null = cls._geo.createNode("null", "smoke_nocap_rest")
        cls._anim_null = cls._geo.createNode("null", "smoke_nocap_anim")
        for n in (cls._rest_null, cls._anim_null):
            n.cook(force=True)

    @classmethod
    def teardown_class(cls):
        try:
            cls._geo.destroy()
        except Exception:
            pass

    def test_no_capture_attrib_reports_false(self):
        """When INPUT geo lacks boneCapture_*, validator.has_capture_weight must be False."""
        # Verify the box indeed has no boneCapture_* attribs (guard assertion)
        box_geom = self._box.geometry()
        assert box_geom.findPointAttrib("boneCapture_weight") is None, (
            "Test precondition violated: smoke_uncaptured_box has boneCapture_weight — "
            "a prior test may have contaminated the scene"
        )

        result = self._dispatch(
            "setup_bonedeform",
            {
                "rest": self._rest_null.path(),
                "anim": self._anim_null.path(),
                "geo": self._box.path(),
                "dest": self._geo.path(),
            },
        )
        data = result.get("data", result)

        if data.get("ok") is True:
            # Bonedeform succeeded (possibly deforming with empty skeleton).
            # The critical invariant: has_capture_weight must be False.
            validator = data.get("validator", {})
            assert validator.get("has_capture_weight") is False, (
                "has_capture_weight must be False when INPUT geo lacks boneCapture_* attribs. "
                f"Got: {validator.get('has_capture_weight')!r}. "
                "Check that handler reads from geo_node.geometry(), NOT bd.geometry()."
            )
        else:
            # Bonedeform errored on uncaptured geo (also valid, FR-2 envelope returned).
            # The error path is acceptable — the FR-5 read-from-input invariant only
            # matters when the cook succeeds.
            assert "error" in data and data["error"], (
                "ok=False envelope must contain non-empty 'error' (FR-2 fail-loud)"
            )


# ===========================================================================
# Smoke: empty-input guard (FR-2 fix#2 — pp12-110d)
# ===========================================================================

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestSetupBonedeformEmptyInput:
    """FR-2 envelope: empty/None rest/anim/geo must return {ok:False} — no exception.

    This case targets the specific BLOCKER from the tier-2 review (pp12-110d fix#2):
    the pre-try: guard previously used raise ValueError(...) which escaped the FR-2
    envelope.  After the fix it must return {ok: False, error: ...} with no exception.
    """

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._geo = hou.node("/obj").createNode("geo", "smoke_empty_input")
        cls._dispatch = _get_dispatcher()

    @classmethod
    def teardown_class(cls):
        try:
            cls._geo.destroy()
        except Exception:
            pass

    def test_empty_rest_returns_ok_false_no_exception(self):
        """Empty rest param must return {ok:False, error:...} — no ValueError raised."""
        result = self._dispatch(
            "setup_bonedeform",
            {
                "rest": "",
                "anim": "/obj/valid_anim",
                "geo": "/obj/valid_geo",
            },
        )
        data = result.get("data", result)
        assert data.get("ok") is False, (
            f"Expected ok=False for empty 'rest', got: {data}\n"
            "FR-2 fix#2 (pp12-110d): the pre-try: guard must RETURN, not raise."
        )
        assert "error" in data and data["error"], (
            "Failure envelope must contain non-empty 'error' (FR-2 fail-loud)"
        )


# ===========================================================================
# Smoke: bogus path negative
# ===========================================================================

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestSetupBonedeformBogusPath:
    """Bogus scene paths must return {ok:False, error:...} without raising."""

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._geo = hou.node("/obj").createNode("geo", "smoke_bogus_bd")
        cls._dispatch = _get_dispatcher()

    @classmethod
    def teardown_class(cls):
        try:
            cls._geo.destroy()
        except Exception:
            pass

    def test_bogus_rest_path_returns_ok_false_no_exception(self):
        """Invalid rest node path must return {ok:False, error:...} without raising."""
        result = self._dispatch(
            "setup_bonedeform",
            {
                "rest": "/obj/does_not_exist_rest",
                "anim": "/obj/does_not_exist_anim",
                "geo": "/obj/does_not_exist_geo",
                "dest": self._geo.path(),
            },
        )
        data = result.get("data", result)
        assert data.get("ok") is False, (
            f"Expected ok=False for bogus node paths, got: {data}"
        )
        assert "error" in data and data["error"], (
            "Failure envelope must include non-empty 'error' key (FR-2 fail-loud)"
        )


# ===========================================================================
# Standalone runner (hython bonedeform_hython_smoke.py)
# ===========================================================================

def main() -> int:
    if not HOU_AVAILABLE:
        print("ERROR: hou not importable — run this script with hython, not python.")
        return 1

    failures: list[str] = []
    fbx = _WORKERS_WELDERS_FBX

    print("\n--- bonedeform smoke (PP12-110d, PR-4) ---")
    print(f"FBX path: {fbx}")
    print("Using: dispatch() -> handler(**params) [fix#007 real dispatcher path]")

    try:
        dispatch = _get_dispatcher()
    except ImportError as exc:
        print(f"  FAIL  dispatcher/character_handlers import: {exc}")
        return 1

    def _data(result):
        """Extract handler result from dispatcher envelope."""
        return result.get("data", result)

    # ── [1] Bogus path — always runnable (no FBX dependency) ─────────────────
    print("\n  [1] setup_bonedeform (bogus node paths → ok=False)")
    hou.hipFile.clear(suppress_save_prompt=True)
    test_geo = hou.node("/obj").createNode("geo", "smoke_standalone")
    try:
        result = dispatch(
            "setup_bonedeform",
            {
                "rest": "/obj/nonexistent_rest",
                "anim": "/obj/nonexistent_anim",
                "geo": "/obj/nonexistent_geo",
                "dest": test_geo.path(),
            },
        )
        data = _data(result)
        if data.get("ok") is not False or not data.get("error"):
            failures.append(f"bogus path should return ok=False with error, got {data}")
            print(f"    FAIL  {data}")
        else:
            print(f"    PASS  ok=False, error={data['error'][:80]!r}")
    except Exception as exc:
        failures.append(f"bogus path raised exception (should return envelope): {exc}")
        print(f"    FAIL  raised: {exc}")

    # ── [2] FR-2 fix#2 empty-input guard — no FBX dependency ────────────────
    print("\n  [2] setup_bonedeform (empty rest → ok=False, no exception — FR-2 fix#2)")
    try:
        result = dispatch(
            "setup_bonedeform",
            {
                "rest": "",
                "anim": "/obj/valid_anim",
                "geo": "/obj/valid_geo",
            },
        )
        data = _data(result)
        if data.get("ok") is not False or not data.get("error"):
            failures.append(f"empty-input guard should return ok=False with error, got {data}")
            print(f"    FAIL  {data}")
        else:
            print(f"    PASS  ok=False, error={data['error'][:80]!r}")
    except Exception as exc:
        failures.append(
            f"empty-input guard raised exception (FR-2 fix#2 violated — must return envelope): {exc}"
        )
        print(f"    FAIL  raised: {exc}")

    # ── [3] FR-5 no-capture — no FBX dependency ──────────────────────────────
    print("\n  [3] FR-5 has_capture_weight=False when INPUT geo lacks boneCapture_*")
    try:
        box = test_geo.createNode("box", "uncaptured_box")
        box.cook(force=True)
        rest_null = test_geo.createNode("null", "smoke_rest")
        anim_null = test_geo.createNode("null", "smoke_anim")
        for n in (rest_null, anim_null):
            n.cook(force=True)

        result = dispatch(
            "setup_bonedeform",
            {
                "rest": rest_null.path(),
                "anim": anim_null.path(),
                "geo": box.path(),
                "dest": test_geo.path(),
            },
        )
        data = _data(result)
        if data.get("ok") is True:
            has_cap = data.get("validator", {}).get("has_capture_weight")
            if has_cap is False:
                print(f"    PASS  ok=True, has_capture_weight=False (correct — reads from INPUT)")
            else:
                msg = (
                    f"has_capture_weight should be False for uncaptured geo, got {has_cap!r}. "
                    f"Handler may be reading from bd.geometry() instead of geo_node.geometry()."
                )
                failures.append(msg)
                print(f"    FAIL  {msg}")
        else:
            # Bonedeform errored on uncaptured geo — also valid (FR-2 envelope).
            print(f"    PASS  ok=False (bonedeform errored on uncaptured geo — acceptable), "
                  f"error={data.get('error','')[:60]!r}")
    except Exception as exc:
        failures.append(f"FR-5 test raised exception: {exc}")
        print(f"    FAIL  raised: {exc}")

    # ── [4] Live WorkersWelders test (FBX-dependent) ──────────────────────────
    if not os.path.exists(fbx):
        print(f"\n  [4-6] SKIP  FBX not found at {fbx!r}")
        print("\nSMOKE PARTIAL — bogus-path + empty-input + no-capture checks ran; FBX-dependent skipped.")
        if failures:
            print(f"SMOKE FAILED — {len(failures)} failure(s):")
            for f in failures:
                print(f"  {f}")
            return 1
        print("SMOKE PASSED (partial — FBX not available)")
        return 0

    print(f"\n  [4] setup_bonedeform (WorkersWelders, via dispatcher)")
    try:
        # Import character: OUT 0 = skin geo, OUT 1 = rest skeleton
        char_result = dispatch(
            "import_fbx_character",
            {"path": fbx, "dest": test_geo.path()},
        )
        char_data = _data(char_result)
        if not char_data.get("ok"):
            failures.append(f"FBX char import failed: {char_data.get('error')}")
            print(f"    FAIL  FBX char import: {char_data.get('error')}")
        else:
            char_node = hou.node(char_data["node"])

            # Import anim FBX (T-pose as single-frame stand-in)
            anim_result = dispatch(
                "import_fbx_animation",
                {"path": fbx, "dest": test_geo.path(), "cascadeur": False},
            )
            anim_data = _data(anim_result)
            if not anim_data.get("ok"):
                failures.append(f"FBX anim import failed: {anim_data.get('error')}")
                print(f"    FAIL  FBX anim import: {anim_data.get('error')}")
            else:
                anim_node = hou.node(anim_data["node"])

                # Wire null pass-throughs to pin outputs as scene paths
                rest_null2 = test_geo.createNode("null", "smoke_rest_live")
                skin_null2 = test_geo.createNode("null", "smoke_skin_live")
                anim_null2 = test_geo.createNode("null", "smoke_anim_live")
                rest_null2.setInput(0, char_node, 1)  # OUT 1 = deformation skeleton
                skin_null2.setInput(0, char_node, 0)  # OUT 0 = skin mesh
                anim_null2.setInput(0, anim_node, 0)  # OUT 0 = animated skeleton
                for n in (rest_null2, skin_null2, anim_null2):
                    n.cook(force=True)

                result = dispatch(
                    "setup_bonedeform",
                    {
                        "rest": rest_null2.path(),
                        "anim": anim_null2.path(),
                        "geo": skin_null2.path(),
                        "dest": test_geo.path(),
                    },
                )
                data = _data(result)
                if not data.get("ok"):
                    failures.append(f"setup_bonedeform failed: {data.get('error')}")
                    print(f"    FAIL  {data.get('error')}")
                else:
                    node_p = data.get("node", "?")
                    validator = data.get("validator", {})
                    skel = data.get("skeleton", {})
                    print(
                        f"    PASS  node={node_p}  joints={skel.get('joints')}  "
                        f"deformed_pts={validator.get('deformed_points')}  "
                        f"has_capture={validator.get('has_capture_weight')}"
                    )
                    if not (validator.get("deformed_points", 0) > 0):
                        msg = f"deformed_points should be > 0, got {validator.get('deformed_points')!r}"
                        failures.append(msg)
                        print(f"    FAIL  {msg}")
                    # pp12-110d fix#1: KineFX stores capture as 'boneCapture' (prefix, not
                    # _weight/_index). Must be True for a captured WorkersWelders skin.
                    if validator.get("has_capture_weight") is not True:
                        msg = (
                            f"has_capture_weight must be True for a KineFX-captured skin; "
                            f"got {validator.get('has_capture_weight')!r}. "
                            "Handler must use startswith('boneCapture') prefix check on "
                            "INPUT geo_node (not bd.geometry())."
                        )
                        failures.append(msg)
                        print(f"    FAIL  {msg}")

    except Exception as exc:
        failures.append(f"WorkersWelders bonedeform smoke raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    try:
        test_geo.destroy()
    except Exception:
        pass

    print()
    if failures:
        print(f"SMOKE FAILED — {len(failures)} failure(s):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("SMOKE PASSED — all bonedeform checks OK (dispatcher path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
