"""
secondarymotion_hython_smoke.py — headless Houdini smoke for PP12-110f.

Dual-mode (mirrors retarget_hython_smoke.py pattern):
  - Standalone:  hython secondarymotion_hython_smoke.py  (requires hython; pytest NOT required)
  - pytest:      collected as a test file; @pytest.mark.hython_smoke tests SKIP
                 automatically when hou is not importable (bare CI / off-DCC).

Covers (dispatcher-based — same real production path as fix#007 / mcp-subprocess-delegation):
  1. apply_secondarymotion (lagovershoot) positive:
       WorkersWelders skeleton, effect="lagovershoot", explicit joints list.
       dispatch("apply_secondarymotion", {...}) → ok=True, affected_joints > 0.
  2. apply_secondarymotion (jiggle) with stiffness parm:
       Same skeleton, params={"effect": "jiggle", "stiffness": 0.5}.
       dispatch(...) → ok=True.
  3. joints=None (all-joints path):
       No jointgroup specified → applies to ALL joints.
       dispatch(...) → ok=True, affected_joints > 0.
  4. Bogus node path → ok=False without exception.
  5. Unknown params key → ok=True (non-fatal), ignored_params contains the key.

NOTE: All calls go through the REAL dispatcher path:
    dispatch("apply_secondarymotion", {"node": ..., "joints": ..., "params": ..., "dest": ...})
This calls handler(**params), which is the authoritative production path.

Fixture note: WorkersWelders FBX provides the skeleton SOP (OUT 1 of
kinefx::fbxcharacterimport).  If the FBX is absent on the current machine,
live-fixture tests skip gracefully.

testVerificationSurface: hython-smoke
unitId: pp12-110f
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
# Test FBX path (live-verified in the plan — WorkersWelders T-pose rig)
# ---------------------------------------------------------------------------
_WORKERS_WELDERS_FBX = (
    r"G:\BigMediumSmall\IndustrialZone"
    r"\IndustialZone_CHARACTERS"
    r"\WorkersWelders_FBX_RIG_TPOSE"
    r"\WorkersWelders_RIG_FBX_TPOSE.fbx"
)


# ===========================================================================
# Dispatcher helper — the REAL production path
# ===========================================================================

def _get_dispatcher():
    """Import and return the dispatcher, registering all handlers.

    Importing character_handlers fires register_handler() for all handlers
    including the new apply_secondarymotion (PR-6), exactly as the real
    Houdini server does at startup.
    """
    try:
        from fxhoudinimcp_server.handlers import character_handlers  # noqa: F401
        from fxhoudinimcp_server.dispatcher import dispatch
        return dispatch
    except ImportError as exc:
        raise ImportError(
            "dispatcher or character_handlers not importable — "
            "run this smoke under hython after PR-6 ships"
        ) from exc


def _data(result):
    """Extract handler result from dispatcher envelope."""
    return result.get("data", result)


# ===========================================================================
# Smoke: bogus node path (FR-2) — no FBX dependency
# ===========================================================================

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestApplySecondarymotionBogusPath:
    """Bogus scene node path must return {ok:False, error:...} without raising."""

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._geo = hou.node("/obj").createNode("geo", "smoke_sm_bogus")
        cls._dispatch = _get_dispatcher()

    @classmethod
    def teardown_class(cls):
        try:
            cls._geo.destroy()
        except Exception:
            pass

    def test_bogus_node_path_returns_ok_false_no_exception(self):
        """Invalid node path must return {ok:False, error:...} without raising."""
        result = self._dispatch(
            "apply_secondarymotion",
            {
                "node": "/obj/does_not_exist_skeleton",
                "dest": self._geo.path(),
            },
        )
        data = _data(result)
        assert data.get("ok") is False, (
            f"Expected ok=False for bogus node path, got: {data}\n"
            "FR-2: the handler must RETURN an envelope, not raise."
        )
        assert "error" in data and data["error"], (
            "Failure envelope must contain non-empty 'error' (FR-2 fail-loud)"
        )

    def test_empty_node_param_returns_ok_false_no_exception(self):
        """Empty node param must return {ok:False, error:...} — pre-try guard."""
        result = self._dispatch(
            "apply_secondarymotion",
            {
                "node": "",
                "dest": self._geo.path(),
            },
        )
        data = _data(result)
        assert data.get("ok") is False, (
            f"Expected ok=False for empty 'node', got: {data}"
        )
        assert "error" in data and data["error"], (
            "Failure envelope must contain non-empty 'error' (FR-2 fail-loud)"
        )


# ===========================================================================
# Smoke: live WorkersWelders tests (FBX-dependent)
# ===========================================================================

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestApplySecondarymotionSmoke:
    """
    Live smoke for apply_secondarymotion using WorkersWelders T-pose skeleton.

    Strategy: import the FBX character to get a skeleton SOP (OUT 1 of the
    fbxcharacterimport node), wire a null through it for a stable scene path,
    then call dispatch("apply_secondarymotion", {...}) for each test scenario.

    All calls go through dispatch("apply_secondarymotion", {...}) — real
    dispatcher path (handler(**params) calling convention).

    If FBX absent on current machine, tests skip gracefully.
    """

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._obj = hou.node("/obj")
        cls._geo = cls._obj.createNode("geo", "smoke_sm")
        cls._dispatch = _get_dispatcher()
        cls._skel_null = None

    @classmethod
    def teardown_class(cls):
        try:
            cls._geo.destroy()
        except Exception:
            pass

    def _ensure_fbx_imported(self):
        """Import WorkersWelders character once and expose OUT 1 (deformation skeleton).

        Returns True if the skeleton null node is ready; False if FBX absent.
        """
        if self.__class__._skel_null is not None:
            return True
        if not os.path.exists(_WORKERS_WELDERS_FBX):
            return False

        result = self._dispatch(
            "import_fbx_character",
            {"path": _WORKERS_WELDERS_FBX, "dest": self._geo.path()},
        )
        data = _data(result)
        if not data.get("ok"):
            pytest.fail(f"FBX import failed during secondarymotion smoke setup: {data.get('error')}")

        char_node = hou.node(data["node"])
        if char_node is None:
            pytest.fail(f"character import node not found: {data['node']!r}")

        # OUT 1 = deformation skeleton
        skel_geo = char_node.geometry(1)
        if skel_geo is None or skel_geo.intrinsicValue("pointcount") == 0:
            pytest.fail("Character import OUT 1 (skeleton) has no points — smoke invalid")

        skel_null = self._geo.createNode("null", "smoke_sm_skel")
        skel_null.setInput(0, char_node, 1)
        skel_null.cook(force=True)

        self.__class__._skel_null = skel_null
        return True

    def _two_joint_names(self):
        """Return first two joint names from the cooked skeleton (for joints list fixture)."""
        if self._skel_null is None:
            return []
        try:
            geo = self._skel_null.geometry()
            if geo is None:
                return []
            all_names = list(geo.pointStringAttribValues("name"))
            return all_names[:2] if len(all_names) >= 2 else all_names
        except Exception:
            return []

    def test_fbx_exists_on_disk(self):
        """Guard: the test FBX must be present; skip gracefully if not."""
        if not os.path.exists(_WORKERS_WELDERS_FBX):
            pytest.skip(f"Test FBX not found at {_WORKERS_WELDERS_FBX!r} — skip on this machine")

    def test_lagovershoot_with_explicit_joints_creates_node(self):
        """apply_secondarymotion (lagovershoot, explicit joints) must return ok=True."""
        if not self._ensure_fbx_imported():
            pytest.skip("Test FBX not found — skipping live secondarymotion smoke")

        joints = self._two_joint_names()
        if not joints:
            pytest.skip("Skeleton has no @name joints — skipping")

        result = self._dispatch(
            "apply_secondarymotion",
            {
                "node": self._skel_null.path(),
                "joints": joints,
                "params": {"effect": "lagovershoot", "lag": 0.3, "overshoot": 0.6},
                "dest": self._geo.path(),
            },
        )
        data = _data(result)
        assert data.get("ok") is True, (
            f"apply_secondarymotion (lagovershoot) failed: {data.get('error')}"
        )
        assert "node" in data and data["node"], (
            f"Missing 'node' path in result: {data}"
        )
        sm_node = hou.node(data["node"])
        assert sm_node is not None, (
            f"secondarymotion node path {data['node']!r} not found in scene"
        )
        assert "affected_joints" in data, (
            f"Result missing 'affected_joints' key: {data}"
        )
        assert isinstance(data["affected_joints"], int) and data["affected_joints"] >= 0, (
            f"affected_joints must be a non-negative int, got {data['affected_joints']!r}"
        )
        # Readback: verify string-token path + scalar-broadcast for 2-value parmTuples
        assert sm_node.parm("effect").evalAsString() == "lagovershoot", (
            f"effect parm readback mismatch: expected 'lagovershoot', "
            f"got {sm_node.parm('effect').evalAsString()!r}"
        )
        assert sm_node.parmTuple("lag").eval() == (0.3, 0.3), (
            f"lag scalar-broadcast mismatch: expected (0.3, 0.3), "
            f"got {sm_node.parmTuple('lag').eval()!r}"
        )
        assert sm_node.parmTuple("overshoot").eval() == (0.6, 0.6), (
            f"overshoot scalar-broadcast mismatch: expected (0.6, 0.6), "
            f"got {sm_node.parmTuple('overshoot').eval()!r}"
        )

    def test_jiggle_with_stiffness_param_returns_ok(self):
        """apply_secondarymotion (jiggle, stiffness parm) must return ok=True."""
        if not self._ensure_fbx_imported():
            pytest.skip("Test FBX not found — skipping")

        result = self._dispatch(
            "apply_secondarymotion",
            {
                "node": self._skel_null.path(),
                "params": {"effect": 1, "stiffness": 0.5},  # 1 = jiggle
                "dest": self._geo.path(),
            },
        )
        data = _data(result)
        assert data.get("ok") is True, (
            f"apply_secondarymotion (jiggle, stiffness=0.5) failed: {data.get('error')}"
        )
        sm_node = hou.node(data.get("node", ""))
        assert sm_node is not None, (
            f"secondarymotion node path {data.get('node')!r} not found in scene"
        )

    def test_all_joints_path_joints_none_returns_ok(self):
        """apply_secondarymotion with joints=None must apply to ALL joints (ok=True)."""
        if not self._ensure_fbx_imported():
            pytest.skip("Test FBX not found — skipping")

        result = self._dispatch(
            "apply_secondarymotion",
            {
                "node": self._skel_null.path(),
                "joints": None,
                "params": {"effect": 2},  # 2 = spring
                "dest": self._geo.path(),
            },
        )
        data = _data(result)
        assert data.get("ok") is True, (
            f"apply_secondarymotion (joints=None, spring) failed: {data.get('error')}"
        )
        assert isinstance(data.get("affected_joints"), int) and data["affected_joints"] > 0, (
            f"affected_joints must be a positive int for all-joints path, got: {data.get('affected_joints')!r}"
        )

    def test_unknown_params_key_non_fatal_ignored_params(self):
        """Unknown parm key in params must be non-fatal — returned in 'ignored_params'."""
        if not self._ensure_fbx_imported():
            pytest.skip("Test FBX not found — skipping")

        result = self._dispatch(
            "apply_secondarymotion",
            {
                "node": self._skel_null.path(),
                "params": {"effect": 0, "totally_made_up_parm": 999.0},
                "dest": self._geo.path(),
            },
        )
        data = _data(result)
        # Unknown parms are non-fatal — tool must still return ok=True
        assert data.get("ok") is True, (
            f"Unknown parm key 'totally_made_up_parm' caused failure (must be non-fatal): {data.get('error')}"
        )
        ignored = data.get("ignored_params", [])
        assert "totally_made_up_parm" in ignored, (
            f"'totally_made_up_parm' must appear in ignored_params, got: {ignored!r}"
        )

    def test_return_envelope_has_required_keys(self):
        """Successful result must contain ok, node, affected_joints, frame_range."""
        if not self._ensure_fbx_imported():
            pytest.skip("Test FBX not found — skipping")

        result = self._dispatch(
            "apply_secondarymotion",
            {
                "node": self._skel_null.path(),
                "dest": self._geo.path(),
            },
        )
        data = _data(result)
        assert data.get("ok") is True, (
            f"apply_secondarymotion failed: {data.get('error')}"
        )
        for key in ("node", "affected_joints", "frame_range"):
            assert key in data, (
                f"Result missing required key '{key}': {data}"
            )


# ===========================================================================
# Standalone runner (hython secondarymotion_hython_smoke.py)
# ===========================================================================

def main() -> int:
    if not HOU_AVAILABLE:
        print("ERROR: hou not importable — run this script with hython, not python.")
        return 1

    failures: list[str] = []
    fbx = _WORKERS_WELDERS_FBX

    print("\n--- secondarymotion smoke (PP12-110f, PR-6) ---")
    print(f"FBX path: {fbx}")
    print("Using: dispatch() -> handler(**params) [real dispatcher path]")

    try:
        dispatch = _get_dispatcher()
    except ImportError as exc:
        print(f"  FAIL  dispatcher/character_handlers import: {exc}")
        return 1

    hou.hipFile.clear(suppress_save_prompt=True)
    test_geo = hou.node("/obj").createNode("geo", "smoke_standalone_sm")

    # ── [1] Bogus path — always runnable (no FBX dependency) ─────────────────
    print("\n  [1] apply_secondarymotion (bogus node path → ok=False, no exception — FR-2)")
    try:
        result = dispatch(
            "apply_secondarymotion",
            {
                "node": "/obj/nonexistent_skeleton",
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
        failures.append(f"bogus path raised exception (FR-2 violated — must return envelope): {exc}")
        print(f"    FAIL  raised: {exc}")

    # ── FBX-dependent tests ───────────────────────────────────────────────────
    if not os.path.exists(fbx):
        print(f"\n  [2-5] SKIP  FBX not found at {fbx!r}")
        print("\nSMOKE PARTIAL — bogus-path check ran; FBX-dependent tests skipped.")
        if failures:
            print(f"SMOKE FAILED — {len(failures)} failure(s):")
            for f in failures:
                print(f"  {f}")
            return 1
        print("SMOKE PASSED (partial — FBX not available)")
        return 0

    # Import character once (OUT 1 = deformation skeleton)
    char_result = dispatch(
        "import_fbx_character",
        {"path": fbx, "dest": test_geo.path()},
    )
    char_data = _data(char_result)
    if not char_data.get("ok"):
        failures.append(f"FBX char import failed: {char_data.get('error')}")
        print(f"\n  FAIL  FBX char import: {char_data.get('error')}")
        test_geo.destroy()
        print(f"\nSMOKE FAILED — {len(failures)} failure(s):")
        for f in failures:
            print(f"  {f}")
        return 1

    char_node = hou.node(char_data["node"])
    skel_null = test_geo.createNode("null", "smoke_sm_skel")
    skel_null.setInput(0, char_node, 1)
    skel_null.cook(force=True)

    # Read joint names for explicit-joints test
    skel_geo = skel_null.geometry()
    all_joints = list(skel_geo.pointStringAttribValues("name")) if skel_geo else []
    two_joints = all_joints[:2] if len(all_joints) >= 2 else all_joints

    # ── [2] lagovershoot + explicit joints + lag/overshoot readback ──────────
    print(f"\n  [2] apply_secondarymotion (lagovershoot, lag=0.3, overshoot=0.6, joints={two_joints!r})")
    try:
        result = dispatch(
            "apply_secondarymotion",
            {
                "node": skel_null.path(),
                "joints": two_joints or None,
                "params": {"effect": "lagovershoot", "lag": 0.3, "overshoot": 0.6},
                "dest": test_geo.path(),
            },
        )
        data = _data(result)
        if not data.get("ok"):
            failures.append(f"[2] lagovershoot positive failed: {data.get('error')}")
            print(f"    FAIL  {data.get('error')}")
        else:
            sm_path = data.get("node", "?")
            aff = data.get("affected_joints", "?")
            fr = data.get("frame_range", "?")
            print(f"    PASS  node={sm_path}  affected_joints={aff}  frame_range={fr}")
            # Readback: effect string-token + scalar-broadcast for 2-value parmTuples
            sm_node = hou.node(sm_path)
            if sm_node is not None:
                effect_token = sm_node.parm("effect").evalAsString()
                lag_val = sm_node.parmTuple("lag").eval()
                overshoot_val = sm_node.parmTuple("overshoot").eval()
                print(f"    READBACK  effect={effect_token!r}  lag={lag_val!r}  overshoot={overshoot_val!r}")
                if effect_token != "lagovershoot":
                    failures.append(f"[2] effect readback mismatch: expected 'lagovershoot', got {effect_token!r}")
                    print(f"    FAIL  effect readback mismatch")
                if lag_val != (0.3, 0.3):
                    failures.append(f"[2] lag readback mismatch: expected (0.3, 0.3), got {lag_val!r}")
                    print(f"    FAIL  lag readback mismatch")
                if overshoot_val != (0.6, 0.6):
                    failures.append(f"[2] overshoot readback mismatch: expected (0.6, 0.6), got {overshoot_val!r}")
                    print(f"    FAIL  overshoot readback mismatch")
    except Exception as exc:
        failures.append(f"[2] lagovershoot raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    # ── [3] jiggle + stiffness parm ───────────────────────────────────────────
    print("\n  [3] apply_secondarymotion (jiggle, stiffness=0.5)")
    try:
        result = dispatch(
            "apply_secondarymotion",
            {
                "node": skel_null.path(),
                "params": {"effect": 1, "stiffness": 0.5},
                "dest": test_geo.path(),
            },
        )
        data = _data(result)
        if not data.get("ok"):
            failures.append(f"[3] jiggle/stiffness failed: {data.get('error')}")
            print(f"    FAIL  {data.get('error')}")
        else:
            print(f"    PASS  node={data.get('node')}  stiffness parm accepted")
    except Exception as exc:
        failures.append(f"[3] jiggle/stiffness raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    # ── [4] joints=None (all-joints path) ─────────────────────────────────────
    print("\n  [4] apply_secondarymotion (joints=None → all-joints, spring)")
    try:
        result = dispatch(
            "apply_secondarymotion",
            {
                "node": skel_null.path(),
                "joints": None,
                "params": {"effect": 2},
                "dest": test_geo.path(),
            },
        )
        data = _data(result)
        if not data.get("ok"):
            failures.append(f"[4] all-joints spring failed: {data.get('error')}")
            print(f"    FAIL  {data.get('error')}")
        else:
            aff = data.get("affected_joints", 0)
            if not (isinstance(aff, int) and aff > 0):
                failures.append(f"[4] all-joints affected_joints should be > 0, got {aff!r}")
                print(f"    FAIL  affected_joints={aff!r} (expected > 0)")
            else:
                print(f"    PASS  affected_joints={aff} (all-joints path)")
    except Exception as exc:
        failures.append(f"[4] all-joints path raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    # ── [5] Unknown params key → ignored_params ───────────────────────────────
    print("\n  [5] apply_secondarymotion (unknown parm key → ignored_params, non-fatal)")
    try:
        result = dispatch(
            "apply_secondarymotion",
            {
                "node": skel_null.path(),
                "params": {"effect": 0, "totally_made_up_parm": 999.0},
                "dest": test_geo.path(),
            },
        )
        data = _data(result)
        if not data.get("ok"):
            failures.append(f"[5] unknown parm key caused failure (should be non-fatal): {data.get('error')}")
            print(f"    FAIL  {data.get('error')}")
        else:
            ignored = data.get("ignored_params", [])
            if "totally_made_up_parm" not in ignored:
                failures.append(
                    f"[5] 'totally_made_up_parm' not in ignored_params: {ignored!r}"
                )
                print(f"    FAIL  ignored_params={ignored!r}, expected 'totally_made_up_parm'")
            else:
                print(f"    PASS  ok=True, ignored_params={ignored!r}")
    except Exception as exc:
        failures.append(f"[5] unknown-parm smoke raised: {exc}")
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
    print("SMOKE PASSED — all secondarymotion checks OK (dispatcher path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
