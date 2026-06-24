"""
retarget_hython_smoke.py — headless Houdini smoke for PP12-110e retarget tool.

Dual-mode (mirrors bonedeform_hython_smoke.py pattern):
  - Standalone:  hython retarget_hython_smoke.py  (requires hython; pytest NOT required)
  - pytest:      collected as a test file; @pytest.mark.hython_smoke tests SKIP
                 automatically when hou is not importable (bare CI / off-DCC).

Covers (dispatcher-based — same fix#007 real production path):
  1. setup_retarget positive (by-name): WorkersWelders source + target (same FBX)
       - dispatch("setup_retarget", {source, target, dest}) → ok=True.
       - validator.unmapped_target_joints is a list (may be non-empty — allowed for T-pose).
       - retarget_node path is valid in the scene.
  2. setup_retarget bogus paths → ok=False without exception.
  3. setup_retarget empty-input guard → ok=False without exception.

NOTE: All calls go through the REAL dispatcher path:
    dispatch("setup_retarget", {"source": ..., "target": ..., ...})
This calls handler(**params), which is the authoritative production path.

Fixture note: WorkersWelders FBX used as both source and target (T-pose self-retarget)
to test the by-name path without needing two distinct character FBX files.
If the FBX is absent on the current machine, live-fixture tests skip gracefully.

testVerificationSurface: hython-smoke
unitId: pp12-110e
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
# Test FBX path (same as bonedeform smoke — live-verified in the plan)
# ---------------------------------------------------------------------------
_WORKERS_WELDERS_FBX = (
    r"G:\BigMediumSmall\IndustrialZone"
    r"\IndustialZone_CHARACTERS"
    r"\WorkersWelders_FBX_RIG_TPOSE"
    r"\WorkersWelders_RIG_FBX_TPOSE.fbx"
)


# ===========================================================================
# Dispatcher helper — the REAL production path (fix#007)
# ===========================================================================

def _get_dispatcher():
    """Import and return the dispatcher, registering all handlers.

    Importing character_handlers fires register_handler() for all handlers
    including the new setup_retarget (PR-5), exactly as the real Houdini server
    does at startup.
    """
    try:
        from fxhoudinimcp_server.handlers import character_handlers  # noqa: F401
        from fxhoudinimcp_server.dispatcher import dispatch
        return dispatch
    except ImportError as exc:
        raise ImportError(
            "dispatcher or character_handlers not importable — "
            "run this smoke under hython after PR-5 ships"
        ) from exc


def _data(result):
    """Extract handler result from dispatcher envelope."""
    return result.get("data", result)


# ===========================================================================
# Smoke: empty-input guard (FR-2)
# ===========================================================================

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestSetupRetargetEmptyInput:
    """FR-2 envelope: empty/None source/target must return {ok:False} — no exception."""

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._geo = hou.node("/obj").createNode("geo", "smoke_retarget_empty")
        cls._dispatch = _get_dispatcher()

    @classmethod
    def teardown_class(cls):
        try:
            cls._geo.destroy()
        except Exception:
            pass

    def test_empty_source_returns_ok_false_no_exception(self):
        """Empty source param must return {ok:False, error:...} — no ValueError raised."""
        result = self._dispatch(
            "setup_retarget",
            {
                "source": "",
                "target": "/obj/valid_target",
            },
        )
        data = _data(result)
        assert data.get("ok") is False, (
            f"Expected ok=False for empty 'source', got: {data}\n"
            "FR-2: the pre-try: guard must RETURN, not raise."
        )
        assert "error" in data and data["error"], (
            "Failure envelope must contain non-empty 'error' (FR-2 fail-loud)"
        )

    def test_empty_target_returns_ok_false_no_exception(self):
        """Empty target param must return {ok:False, error:...} — no ValueError raised."""
        result = self._dispatch(
            "setup_retarget",
            {
                "source": "/obj/valid_source",
                "target": "",
            },
        )
        data = _data(result)
        assert data.get("ok") is False, (
            f"Expected ok=False for empty 'target', got: {data}"
        )
        assert "error" in data and data["error"], (
            "Failure envelope must contain non-empty 'error' (FR-2 fail-loud)"
        )


# ===========================================================================
# Smoke: bogus path negative
# ===========================================================================

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestSetupRetargetBogusPath:
    """Bogus scene paths must return {ok:False, error:...} without raising."""

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._geo = hou.node("/obj").createNode("geo", "smoke_retarget_bogus")
        cls._dispatch = _get_dispatcher()

    @classmethod
    def teardown_class(cls):
        try:
            cls._geo.destroy()
        except Exception:
            pass

    def test_bogus_source_path_returns_ok_false_no_exception(self):
        """Invalid source/target paths must return {ok:False, error:...} without raising."""
        result = self._dispatch(
            "setup_retarget",
            {
                "source": "/obj/does_not_exist_source",
                "target": "/obj/does_not_exist_target",
                "dest": self._geo.path(),
            },
        )
        data = _data(result)
        assert data.get("ok") is False, (
            f"Expected ok=False for bogus node paths, got: {data}"
        )
        assert "error" in data and data["error"], (
            "Failure envelope must include non-empty 'error' key (FR-2 fail-loud)"
        )


# ===========================================================================
# Smoke: retarget positive (hython_smoke) — WorkersWelders self-retarget
# ===========================================================================

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestSetupRetargetSmoke:
    """
    Live smoke for setup_retarget using WorkersWelders T-pose FBX.

    Strategy: import the FBX once as a character skeleton (OUT 1), use the
    same skeleton as both source and target (T-pose self-retarget) to exercise
    the by-name path without needing two distinct character FBX files.

    The by-name path (mapping=None) configures:
        rigmatchpose → fullbodyik (mapusing=1="matchattrib", attribtomatch="name")

    All calls go through dispatch("setup_retarget", {...}) — real dispatcher path.
    Fixture: WorkersWelders T-pose FBX at the grounded path.
    If absent, tests skip gracefully.
    """

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._obj = hou.node("/obj")
        cls._geo = cls._obj.createNode("geo", "smoke_retarget")
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
            pytest.fail(f"FBX import failed during retarget smoke setup: {data.get('error')}")

        char_node = hou.node(data["node"])
        if char_node is None:
            pytest.fail(f"character import node not found: {data['node']!r}")

        # OUT 1 = deformation skeleton
        skel_geo = char_node.geometry(1)
        if skel_geo is None or skel_geo.intrinsicValue("pointcount") == 0:
            pytest.fail("Character import OUT 1 (skeleton) has no points — smoke invalid")

        skel_null = self._geo.createNode("null", "smoke_retarget_skel")
        skel_null.setInput(0, char_node, 1)
        skel_null.cook(force=True)

        self.__class__._skel_null = skel_null
        return True

    def test_fbx_exists_on_disk(self):
        """Guard: the test FBX must be present; skip gracefully if not."""
        if not os.path.exists(_WORKERS_WELDERS_FBX):
            pytest.skip(f"Test FBX not found at {_WORKERS_WELDERS_FBX!r} — skip on this machine")

    def test_setup_retarget_by_name_creates_node_and_cooks(self):
        """setup_retarget (by-name) must create fullbodyik node and cook without error."""
        if not self._ensure_fbx_imported():
            pytest.skip("Test FBX not found — skipping live retarget smoke")

        result = self._dispatch(
            "setup_retarget",
            {
                "source": self._skel_null.path(),
                "target": self._skel_null.path(),  # self-retarget (same skeleton)
                "dest": self._geo.path(),
            },
        )
        data = _data(result)
        assert data.get("ok") is True, (
            f"setup_retarget failed: {data.get('error')}"
        )
        assert "retarget_node" in data and data["retarget_node"], (
            f"Missing 'retarget_node' path in result: {data}"
        )
        retarget_node = hou.node(data["retarget_node"])
        assert retarget_node is not None, (
            f"retarget_node path {data['retarget_node']!r} not found in scene"
        )

    def test_setup_retarget_returns_validator_dict_with_required_keys(self):
        """Return envelope must contain validator sub-dict with expected keys (FR-12)."""
        if not self._ensure_fbx_imported():
            pytest.skip("Test FBX not found")

        result = self._dispatch(
            "setup_retarget",
            {
                "source": self._skel_null.path(),
                "target": self._skel_null.path(),
                "dest": self._geo.path(),
            },
        )
        data = _data(result)
        assert data.get("ok") is True, f"setup_retarget failed: {data.get('error')}"

        validator = data.get("validator")
        assert validator is not None, "Return envelope missing 'validator' key"
        assert "unmapped_target_joints" in validator, (
            "validator missing 'unmapped_target_joints' key (FR-12)"
        )
        assert "cook_errors" in validator, "validator missing 'cook_errors'"
        assert isinstance(validator["unmapped_target_joints"], list), (
            f"unmapped_target_joints must be a list, got {type(validator['unmapped_target_joints'])!r}"
        )

    def test_setup_retarget_returns_target_skeleton_dict(self):
        """target_skeleton sub-dict must contain joints count (FR-12)."""
        if not self._ensure_fbx_imported():
            pytest.skip("Test FBX not found")

        result = self._dispatch(
            "setup_retarget",
            {
                "source": self._skel_null.path(),
                "target": self._skel_null.path(),
                "dest": self._geo.path(),
            },
        )
        data = _data(result)
        assert data.get("ok") is True, f"setup_retarget failed: {data.get('error')}"

        tgt_skel = data.get("target_skeleton")
        assert tgt_skel is not None, "Return envelope missing 'target_skeleton' key"
        joints = tgt_skel.get("joints")
        assert isinstance(joints, int) and joints > 0, (
            f"target_skeleton.joints must be a positive int, got {joints!r}"
        )

    def test_setup_retarget_by_name_unmapped_is_empty(self):
        """In the by-name path (mapping=None) unmapped_target_joints must be empty.

        By-name mode uses fullbodyik matchattrib — FBIK matches every joint by @name
        automatically.  Nothing is "unmapped" in the advisory sense, so the list must
        be [] (DF-2 fix: passing [] pairs to unmapped_target_joints returned ALL targets
        as unmapped, which was semantically wrong for the by-name contract).
        """
        if not self._ensure_fbx_imported():
            pytest.skip("Test FBX not found")

        result = self._dispatch(
            "setup_retarget",
            {
                "source": self._skel_null.path(),
                "target": self._skel_null.path(),
                "dest": self._geo.path(),
            },
        )
        data = _data(result)
        assert data.get("ok") is True, f"setup_retarget failed: {data.get('error')}"

        unmapped = data.get("validator", {}).get("unmapped_target_joints", "MISSING")
        assert unmapped != "MISSING", "validator.unmapped_target_joints key absent"
        assert isinstance(unmapped, list), (
            f"unmapped_target_joints must be a list, got {type(unmapped)!r}"
        )
        assert len(unmapped) == 0, (
            f"By-name path: unmapped_target_joints must be [] (FBIK matches by @name), "
            f"got {len(unmapped)} entries: {unmapped[:5]!r}{'...' if len(unmapped) > 5 else ''}"
        )

    def test_setup_retarget_explicit_mapping_cooks_and_inserts_mappoints(self):
        """setup_retarget (explicit mapping) must insert kinefx::mappoints, cook ok=True,
        and report unmapped_target_joints containing ONLY joints not in the mapping subset.

        FIX 1 regression guard: before the 0-indexed fix, this test triggered
        hou.OperationFailed because from{N} (out-of-range) was set and from0 was never
        set, causing the cook to fail with ok=False.

        Strategy:
          - Read the target skeleton's @name joint list at runtime.
          - Take the first half as the mapping subset (identity pairs: [[n,n] ...]).
          - Pass mapping=[[n, n] for n in subset] to exercise the explicit-mapping path.
          - Assert ok=True, retarget_node valid, kinefx::mappoints present in scene,
            and unmapped_target_joints == the OMITTED joints (second half).
        """
        if not self._ensure_fbx_imported():
            pytest.skip("Test FBX not found")

        # Read the skeleton's @name attrib list from the cooked null node
        skel_geo = self._skel_null.geometry()
        if skel_geo is None:
            pytest.skip("Skeleton null has no cooked geometry — skip")
        all_joints = list(skel_geo.pointStringAttribValues("name"))
        if not all_joints:
            pytest.skip("Skeleton has no @name attrib points — skip")

        # Use first half as the mapping subset; remainder should appear in unmapped
        half = max(1, len(all_joints) // 2)
        subset = all_joints[:half]
        omitted = set(all_joints[half:])
        mapping = [[n, n] for n in subset]

        result = self._dispatch(
            "setup_retarget",
            {
                "source": self._skel_null.path(),
                "target": self._skel_null.path(),
                "dest": self._geo.path(),
                "mapping": mapping,
            },
        )
        data = _data(result)
        assert data.get("ok") is True, (
            f"setup_retarget (explicit mapping) failed: {data.get('error')}\n"
            f"FIX 1 regression: if this was 'parm not found' the 0-indexed fix did not land."
        )

        # retarget_node must be a valid scene path
        retarget_node_path = data.get("retarget_node", "")
        assert retarget_node_path, f"Missing 'retarget_node' in result: {data}"
        retarget_node = hou.node(retarget_node_path)
        assert retarget_node is not None, (
            f"retarget_node path {retarget_node_path!r} not found in scene"
        )

        # kinefx::mappoints must exist in the same geo parent (explicit path taken)
        sop_parent = retarget_node.parent()
        mp_nodes = [
            ch for ch in sop_parent.children()
            if ch.type().name() == "kinefx::mappoints"
        ]
        assert mp_nodes, (
            "kinefx::mappoints node not found in the retarget geo network — "
            "explicit-mapping path may not have been taken"
        )

        # unmapped_target_joints must contain exactly the joints NOT in the subset
        validator = data.get("validator", {})
        unmapped = validator.get("unmapped_target_joints", "MISSING")
        assert unmapped != "MISSING", "validator.unmapped_target_joints key absent"
        assert isinstance(unmapped, list), (
            f"unmapped_target_joints must be a list, got {type(unmapped)!r}"
        )
        unmapped_set = set(unmapped)
        # Every omitted joint must appear in unmapped
        missing_from_unmapped = omitted - unmapped_set
        assert not missing_from_unmapped, (
            f"Joints omitted from mapping but absent from unmapped_target_joints: "
            f"{sorted(missing_from_unmapped)}"
        )
        # No mapped joint should appear in unmapped
        mapped_in_unmapped = set(subset) & unmapped_set
        assert not mapped_in_unmapped, (
            f"Joints that WERE explicitly mapped appear in unmapped_target_joints "
            f"(should NOT): {sorted(mapped_in_unmapped)}"
        )


# ===========================================================================
# Standalone runner (hython retarget_hython_smoke.py)
# ===========================================================================

def main() -> int:
    if not HOU_AVAILABLE:
        print("ERROR: hou not importable — run this script with hython, not python.")
        return 1

    failures: list[str] = []
    fbx = _WORKERS_WELDERS_FBX

    print("\n--- retarget smoke (PP12-110e, PR-5) ---")
    print(f"FBX path: {fbx}")
    print("Using: dispatch() -> handler(**params) [real dispatcher path]")

    try:
        dispatch = _get_dispatcher()
    except ImportError as exc:
        print(f"  FAIL  dispatcher/character_handlers import: {exc}")
        return 1

    hou.hipFile.clear(suppress_save_prompt=True)
    test_geo = hou.node("/obj").createNode("geo", "smoke_standalone_retarget")

    # ── [1] Empty-input guard — always runnable (no FBX dependency) ──────────
    print("\n  [1] setup_retarget (empty source → ok=False, no exception — FR-2)")
    try:
        result = dispatch(
            "setup_retarget",
            {
                "source": "",
                "target": "/obj/valid_target",
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
            f"empty-input guard raised exception (FR-2 violated — must return envelope): {exc}"
        )
        print(f"    FAIL  raised: {exc}")

    # ── [2] Bogus path — always runnable ─────────────────────────────────────
    print("\n  [2] setup_retarget (bogus node paths → ok=False)")
    try:
        result = dispatch(
            "setup_retarget",
            {
                "source": "/obj/nonexistent_source",
                "target": "/obj/nonexistent_target",
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

    # ── [3-5] Live WorkersWelders tests (FBX-dependent) ──────────────────────
    if not os.path.exists(fbx):
        print(f"\n  [3-5] SKIP  FBX not found at {fbx!r}")
        print("\nSMOKE PARTIAL — empty-input + bogus-path checks ran; FBX-dependent skipped.")
        if failures:
            print(f"SMOKE FAILED — {len(failures)} failure(s):")
            for f in failures:
                print(f"  {f}")
            return 1
        print("SMOKE PASSED (partial — FBX not available)")
        return 0

    print(f"\n  [3] setup_retarget (by-name, WorkersWelders self-retarget, via dispatcher)")
    try:
        # Import character: OUT 1 = deformation skeleton
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

            # OUT 1 = deformation skeleton — wire through a null to pin it as a scene path
            skel_null = test_geo.createNode("null", "smoke_retarget_skel")
            skel_null.setInput(0, char_node, 1)
            skel_null.cook(force=True)

            # self-retarget: same node as both source and target
            result = dispatch(
                "setup_retarget",
                {
                    "source": skel_null.path(),
                    "target": skel_null.path(),
                    "dest": test_geo.path(),
                },
            )
            data = _data(result)
            if not data.get("ok"):
                failures.append(f"setup_retarget failed: {data.get('error')}")
                print(f"    FAIL  {data.get('error')}")
            else:
                retarget_node_path = data.get("retarget_node", "?")
                tgt_skel = data.get("target_skeleton", {})
                validator = data.get("validator", {})
                unmapped = validator.get("unmapped_target_joints", "MISSING")
                print(
                    f"    PASS  retarget_node={retarget_node_path}  "
                    f"joints={tgt_skel.get('joints')}  "
                    f"unmapped={unmapped}"
                )

                # [4] Check validator keys
                print("\n  [4] validator sub-dict keys (FR-12)")
                if "unmapped_target_joints" not in validator:
                    msg = "validator missing 'unmapped_target_joints'"
                    failures.append(msg)
                    print(f"    FAIL  {msg}")
                elif "cook_errors" not in validator:
                    msg = "validator missing 'cook_errors'"
                    failures.append(msg)
                    print(f"    FAIL  {msg}")
                else:
                    print(f"    PASS  validator has unmapped_target_joints + cook_errors")

                # [5] by-name unmapped check: must be empty (DF-2 fix)
                print("\n  [5] by-name: unmapped_target_joints == [] (FBIK matches by @name)")
                if unmapped == "MISSING":
                    msg = "validator.unmapped_target_joints key missing from return envelope"
                    failures.append(msg)
                    print(f"    FAIL  {msg}")
                elif not isinstance(unmapped, list):
                    msg = f"unmapped_target_joints must be a list, got {type(unmapped)!r}"
                    failures.append(msg)
                    print(f"    FAIL  {msg}")
                elif len(unmapped) != 0:
                    msg = (
                        f"By-name path: unmapped_target_joints must be [] "
                        f"(FBIK matches by @name), got {len(unmapped)} entries"
                    )
                    failures.append(msg)
                    print(f"    FAIL  {msg}")
                else:
                    print(
                        f"    PASS  unmapped_target_joints == [] (correct for by-name path)"
                    )

    except Exception as exc:
        failures.append(f"WorkersWelders retarget smoke raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    # ── [6] Explicit-mapping path — FIX 1 regression guard ───────────────────
    print("\n  [6] setup_retarget (explicit mapping, partial subset → ok=True, mappoints present)")
    try:
        # Reuse the already-cooked char_node / skel_null from [3] when available;
        # re-import if needed (FBX already checked above).
        if "skel_null" not in dir():
            char_result = dispatch(
                "import_fbx_character",
                {"path": fbx, "dest": test_geo.path()},
            )
            char_data = _data(char_result)
            if not char_data.get("ok"):
                failures.append(f"FBX re-import for [6] failed: {char_data.get('error')}")
                print(f"    FAIL  FBX re-import: {char_data.get('error')}")
                raise RuntimeError("FBX re-import failed")
            char_node = hou.node(char_data["node"])
            skel_null = test_geo.createNode("null", "smoke_retarget_skel6")
            skel_null.setInput(0, char_node, 1)
            skel_null.cook(force=True)

        skel_geo = skel_null.geometry()
        all_joints = list(skel_geo.pointStringAttribValues("name")) if skel_geo else []
        if not all_joints:
            failures.append("[6] skeleton has no @name attrib — cannot build mapping")
            print("    FAIL  skeleton @name attrib missing")
        else:
            half = max(1, len(all_joints) // 2)
            subset = all_joints[:half]
            omitted = set(all_joints[half:])
            mapping_pairs = [[n, n] for n in subset]

            result6 = dispatch(
                "setup_retarget",
                {
                    "source": skel_null.path(),
                    "target": skel_null.path(),
                    "dest": test_geo.path(),
                    "mapping": mapping_pairs,
                },
            )
            d6 = _data(result6)
            if not d6.get("ok"):
                failures.append(
                    f"[6] explicit-mapping returned ok=False: {d6.get('error')}\n"
                    "  FIX 1 regression: if 'parm not found' → 0-indexed fix did not land."
                )
                print(f"    FAIL  {d6.get('error')}")
            else:
                retarget_path6 = d6.get("retarget_node", "")
                rn6 = hou.node(retarget_path6)
                mp_nodes6 = [
                    ch for ch in (rn6.parent().children() if rn6 else [])
                    if ch.type().name() == "kinefx::mappoints"
                ]
                validator6 = d6.get("validator", {})
                unmapped6 = validator6.get("unmapped_target_joints", [])
                unmapped_set6 = set(unmapped6)
                missing_from_unmapped6 = omitted - unmapped_set6
                mapped_in_unmapped6 = set(subset) & unmapped_set6

                if not mp_nodes6:
                    failures.append("[6] kinefx::mappoints node not found — explicit path not taken")
                    print("    FAIL  kinefx::mappoints absent in network")
                elif missing_from_unmapped6:
                    failures.append(
                        f"[6] omitted joints missing from unmapped_target_joints: "
                        f"{sorted(missing_from_unmapped6)}"
                    )
                    print(f"    FAIL  omitted joints not in unmapped: {sorted(missing_from_unmapped6)}")
                elif mapped_in_unmapped6:
                    failures.append(
                        f"[6] mapped joints appear in unmapped_target_joints (should not): "
                        f"{sorted(mapped_in_unmapped6)}"
                    )
                    print(f"    FAIL  mapped joints in unmapped: {sorted(mapped_in_unmapped6)}")
                else:
                    print(
                        f"    PASS  ok=True  retarget_node={retarget_path6}  "
                        f"mappoints=1  "
                        f"subset={len(subset)}/{len(all_joints)} joints mapped  "
                        f"unmapped={len(unmapped6)} (omitted joints={len(omitted)})"
                    )

    except RuntimeError:
        pass  # FBX re-import failed — already recorded above
    except Exception as exc:
        failures.append(f"[6] explicit-mapping smoke raised: {exc}")
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
    print("SMOKE PASSED — all retarget checks OK (dispatcher path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
