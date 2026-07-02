"""
export_hython_smoke.py -- dispatcher-based headless smoke for PP12-111 PR-3 export handlers.

Dual-mode (mirrors retarget_hython_smoke.py pattern):
  - Standalone:  hython tests/export_hython_smoke.py  (requires hython; pytest NOT required)
  - pytest:      collected as a test file; tests SKIP automatically when hou is not
                 importable (bare CI / off-DCC).

Covers (dispatcher-based -- same real production path as all other PP12 smokes):
  1. probe_versions (no target_ue): returns {houdini, labs_vat_rop, rop_alembic, rop_fbx},
     no 'skew' key.
  2. probe_versions (target_ue='5.4'): adds {skew: {verdict, notes}}.
  3. validate_budget (fbx, box fixture): verdict in pass/warn/fail, wrote_files=False.
  4. validate_budget (chaos_gc, non-sequential gc_piece fixture):
     gc_piece_sequential check status='fail', wrote_files=False.

NOTE: All calls go through the REAL dispatcher path:
    dispatch("cmd", {"param": value, ...})
This calls handler(**params), which is the authoritative production path.

Handler registration: importing export_handlers fires register_handler() for
probe_versions + validate_budget -- exactly as the real server does at startup.

checks is a LIST of dicts (each with an 'id' field), NOT a dict keyed by id.
Find a check by: next(c for c in checks if c['id'] == 'gc_piece_sequential').

testVerificationSurface: hython-smoke
unitId: pp12-111c
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap -- resolves fxhoudinimcp packages in both hython + pytest.
# Mirrors retarget_hython_smoke.py exactly.
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
# Importing export_handlers fires register_handler() for probe_versions +
# validate_budget, exactly as the real Houdini server does at startup.
# Without this import, available_commands is EMPTY and every dispatch()
# call returns "command not found".
# ===========================================================================

def _get_dispatcher():
    """Import and return the dispatcher, registering export handlers.

    Importing export_handlers fires register_handler() for probe_versions
    and validate_budget -- exactly as the real server does at startup.
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
# Smoke tests (pytest class form -- also called from standalone main())
# ===========================================================================

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available -- requires hython")
class TestProbeVersionsNoTarget:
    """probe_versions() with no target_ue: base keys present, no skew block."""

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._dispatch = _get_dispatcher()

    def test_probe_versions_returns_dict(self):
        result = self._dispatch("probe_versions", {})
        data = _data(result)
        assert isinstance(data, dict), (
            f"probe_versions result must be a dict, got {type(data).__name__!r}"
        )

    def test_probe_versions_has_core_keys(self):
        result = self._dispatch("probe_versions", {})
        data = _data(result)
        for key in ("houdini", "labs_vat_rop", "rop_alembic", "rop_fbx"):
            assert key in data, (
                f"probe_versions result missing key {key!r}; "
                f"keys present: {sorted(data.keys())}"
            )

    def test_probe_versions_rop_fields_are_bool(self):
        result = self._dispatch("probe_versions", {})
        data = _data(result)
        assert isinstance(data.get("rop_alembic"), bool), (
            f"rop_alembic must be bool, got {type(data.get('rop_alembic')).__name__!r}"
        )
        assert isinstance(data.get("rop_fbx"), bool), (
            f"rop_fbx must be bool, got {type(data.get('rop_fbx')).__name__!r}"
        )

    def test_probe_versions_labs_vat_rop_is_none_or_str(self):
        result = self._dispatch("probe_versions", {})
        data = _data(result)
        labs = data.get("labs_vat_rop")
        assert labs is None or isinstance(labs, str), (
            f"labs_vat_rop must be None or str, got {type(labs).__name__!r}: {labs!r}"
        )

    def test_probe_versions_no_skew_without_target_ue(self):
        result = self._dispatch("probe_versions", {})
        data = _data(result)
        assert "skew" not in data, (
            f"'skew' must not be present when target_ue is not supplied; "
            f"keys: {sorted(data.keys())}"
        )


@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available -- requires hython")
class TestProbeVersionsWithTargetUe:
    """probe_versions() with target_ue='5.4': adds skew block."""

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._dispatch = _get_dispatcher()

    def test_probe_versions_with_target_ue_has_skew(self):
        result = self._dispatch("probe_versions", {"target_ue": "5.4"})
        data = _data(result)
        assert "skew" in data, (
            f"'skew' key must be present when target_ue='5.4'; "
            f"keys: {sorted(data.keys())}"
        )

    def test_probe_versions_with_target_ue_skew_is_dict(self):
        result = self._dispatch("probe_versions", {"target_ue": "5.4"})
        data = _data(result)
        skew = data.get("skew", {})
        assert isinstance(skew, dict), (
            f"skew block must be a dict, got {type(skew).__name__!r}"
        )

    def test_probe_versions_with_target_ue_skew_has_verdict_and_notes(self):
        result = self._dispatch("probe_versions", {"target_ue": "5.4"})
        data = _data(result)
        skew = data.get("skew", {})
        for key in ("verdict", "notes"):
            assert key in skew, (
                f"skew block missing key {key!r}; skew keys: {sorted(skew.keys())}"
            )

    def test_probe_versions_with_target_ue_still_has_core_keys(self):
        result = self._dispatch("probe_versions", {"target_ue": "5.4"})
        data = _data(result)
        for key in ("houdini", "labs_vat_rop", "rop_alembic", "rop_fbx"):
            assert key in data, (
                f"probe_versions result missing core key {key!r} when target_ue supplied; "
                f"keys: {sorted(data.keys())}"
            )


@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available -- requires hython")
class TestValidateBudgetFbx:
    """validate_budget (fbx, box fixture): verdict in pass/warn/fail, dry-run."""

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._dispatch = _get_dispatcher()
        geo = hou.node("/obj").createNode("geo", "geo_smoke_export")
        box = geo.createNode("box", "box1")
        box.cook(force=True)
        cls._box_path = box.path()
        cls._geo = geo

    @classmethod
    def teardown_class(cls):
        try:
            cls._geo.destroy()
        except Exception:
            pass

    def test_validate_budget_fbx_returns_dict(self):
        result = self._dispatch("validate_budget", {"node": self._box_path, "target": "fbx"})
        data = _data(result)
        assert isinstance(data, dict), (
            f"validate_budget must return a dict, got {type(data).__name__!r}"
        )

    def test_validate_budget_fbx_has_required_keys(self):
        result = self._dispatch("validate_budget", {"node": self._box_path, "target": "fbx"})
        data = _data(result)
        for key in ("verdict", "checks", "wrote_files"):
            assert key in data, (
                f"validate_budget result missing key {key!r}; "
                f"keys: {sorted(data.keys())}"
            )

    def test_validate_budget_fbx_is_dry_run(self):
        result = self._dispatch("validate_budget", {"node": self._box_path, "target": "fbx"})
        data = _data(result)
        assert data.get("wrote_files") is False, (
            f"validate_budget must be DRY-RUN (wrote_files=False), "
            f"got wrote_files={data.get('wrote_files')!r}"
        )

    def test_validate_budget_fbx_verdict_is_valid(self):
        result = self._dispatch("validate_budget", {"node": self._box_path, "target": "fbx"})
        data = _data(result)
        verdict = data.get("verdict")
        assert isinstance(verdict, str) and verdict in ("pass", "warn", "fail"), (
            f"verdict must be one of pass/warn/fail, got {verdict!r}"
        )

    def test_validate_budget_fbx_checks_is_list(self):
        result = self._dispatch("validate_budget", {"node": self._box_path, "target": "fbx"})
        data = _data(result)
        checks = data.get("checks")
        assert isinstance(checks, list), (
            f"checks must be a list of dicts, got {type(checks).__name__!r}"
        )


@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available -- requires hython")
class TestValidateBudgetChaosGcFail:
    """validate_budget (chaos_gc, non-sequential gc_piece): gc_piece_sequential fails."""

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._dispatch = _get_dispatcher()
        # Build fixture: box -> attribwrangle stamps unreal_gc_piece = 0,1,3,3 (gap at 2)
        geo = hou.node("/obj").createNode("geo", "geo_smoke_chaos")
        box = geo.createNode("box", "box1")
        wr = geo.createNode("attribwrangle", "gc_piece_wrangle")
        wr.setFirstInput(box)
        wr.parm("class").set(1)  # prim class
        wr.parm("snippet").set(
            "i@unreal_gc_piece = (@primnum==2) ? 3 : @primnum;"
        )
        wr.cook(force=True)
        cls._fixture_path = wr.path()
        cls._geo = geo

    @classmethod
    def teardown_class(cls):
        try:
            cls._geo.destroy()
        except Exception:
            pass

    def test_validate_budget_chaos_gc_returns_dict(self):
        result = self._dispatch(
            "validate_budget", {"node": self._fixture_path, "target": "chaos_gc"}
        )
        data = _data(result)
        assert isinstance(data, dict), (
            f"validate_budget must return a dict, got {type(data).__name__!r}"
        )

    def test_validate_budget_chaos_gc_is_dry_run(self):
        result = self._dispatch(
            "validate_budget", {"node": self._fixture_path, "target": "chaos_gc"}
        )
        data = _data(result)
        assert data.get("wrote_files") is False, (
            f"validate_budget must be DRY-RUN (wrote_files=False), "
            f"got wrote_files={data.get('wrote_files')!r}"
        )

    def test_validate_budget_chaos_gc_checks_is_list(self):
        result = self._dispatch(
            "validate_budget", {"node": self._fixture_path, "target": "chaos_gc"}
        )
        data = _data(result)
        checks = data.get("checks")
        assert isinstance(checks, list), (
            f"checks must be a list of dicts (each with 'id' field), "
            f"got {type(checks).__name__!r}"
        )

    def test_validate_budget_chaos_gc_has_gc_piece_sequential_check(self):
        result = self._dispatch(
            "validate_budget", {"node": self._fixture_path, "target": "chaos_gc"}
        )
        data = _data(result)
        checks = data.get("checks", [])
        # checks is a LIST -- find by id field, not dict key
        gc_check = next((c for c in checks if c.get("id") == "gc_piece_sequential"), None)
        assert gc_check is not None, (
            f"gc_piece_sequential check not found in checks list; "
            f"check ids: {[c.get('id') for c in checks]}"
        )

    def test_validate_budget_chaos_gc_gc_piece_sequential_fails(self):
        result = self._dispatch(
            "validate_budget", {"node": self._fixture_path, "target": "chaos_gc"}
        )
        data = _data(result)
        checks = data.get("checks", [])
        # checks is a LIST -- find by id field
        gc_check = next((c for c in checks if c.get("id") == "gc_piece_sequential"), None)
        assert gc_check is not None, (
            f"gc_piece_sequential check not found; "
            f"check ids: {[c.get('id') for c in checks]}"
        )
        gc_status = gc_check.get("status")
        assert gc_status == "fail", (
            f"gc_piece_sequential check must have status='fail' for non-sequential "
            f"gc_piece values [0,1,3] (gap at 2); got status={gc_status!r}\n"
            f"full check: {gc_check}"
        )


# ===========================================================================
# Standalone runner (hython tests/export_hython_smoke.py)
# ===========================================================================

def main() -> int:
    if not HOU_AVAILABLE:
        print("ERROR: hou not importable -- run this script with hython, not python.")
        return 1

    failures: list[str] = []

    print("\n--- export smoke (PP12-111c, PR-3) ---")
    print("Using: dispatch() -> handler(**params) [real dispatcher path]")

    try:
        dispatch = _get_dispatcher()
    except ImportError as exc:
        print(f"  FAIL  dispatcher/export_handlers import: {exc}")
        return 1

    # ---- [1] probe_versions -- no target_ue --------------------------------
    print("\n  [1] probe_versions ({}) -- base keys, no skew")
    try:
        result = dispatch("probe_versions", {})
        data = _data(result)
        ok = True
        for key in ("houdini", "labs_vat_rop", "rop_alembic", "rop_fbx"):
            if key not in data:
                failures.append(f"[1] probe_versions missing key {key!r}")
                print(f"    FAIL  missing key {key!r}")
                ok = False
        if "skew" in data:
            failures.append("[1] probe_versions has unexpected 'skew' key (no target_ue)")
            print("    FAIL  unexpected 'skew' key")
            ok = False
        if ok:
            print(
                f"    PASS  houdini={data.get('houdini')!r}  "
                f"labs_vat_rop={data.get('labs_vat_rop')!r}  "
                f"rop_alembic={data.get('rop_alembic')}  "
                f"rop_fbx={data.get('rop_fbx')}"
            )
    except Exception as exc:
        failures.append(f"[1] probe_versions raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    # ---- [2] probe_versions -- with target_ue ------------------------------
    print("\n  [2] probe_versions ({target_ue='5.4'}) -- adds skew block")
    try:
        result = dispatch("probe_versions", {"target_ue": "5.4"})
        data = _data(result)
        ok = True
        if "skew" not in data:
            failures.append("[2] probe_versions missing 'skew' key when target_ue='5.4'")
            print("    FAIL  missing 'skew' key")
            ok = False
        else:
            skew = data.get("skew", {})
            for key in ("verdict", "notes"):
                if key not in skew:
                    failures.append(f"[2] skew block missing key {key!r}")
                    print(f"    FAIL  skew missing key {key!r}")
                    ok = False
        if ok:
            skew = data.get("skew", {})
            print(
                f"    PASS  skew.verdict={skew.get('verdict')!r}  "
                f"skew.notes={skew.get('notes')!r}"
            )
    except Exception as exc:
        failures.append(f"[2] probe_versions with target_ue raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    # ---- [3] validate_budget -- fbx ----------------------------------------
    print("\n  [3] validate_budget (fbx, box fixture) -- verdict + dry-run")
    try:
        hou.hipFile.clear(suppress_save_prompt=True)
        geo = hou.node("/obj").createNode("geo", "geo_smoke_export")
        box = geo.createNode("box", "box1")
        box.cook(force=True)
        box_path = box.path()

        result = dispatch("validate_budget", {"node": box_path, "target": "fbx"})
        data = _data(result)
        ok = True
        for key in ("verdict", "checks", "wrote_files"):
            if key not in data:
                failures.append(f"[3] validate_budget (fbx) missing key {key!r}")
                print(f"    FAIL  missing key {key!r}")
                ok = False
        if data.get("wrote_files") is not False:
            failures.append(
                f"[3] validate_budget is NOT dry-run: wrote_files={data.get('wrote_files')!r}"
            )
            print(f"    FAIL  wrote_files={data.get('wrote_files')!r} (expected False)")
            ok = False
        checks = data.get("checks", [])
        if not isinstance(checks, list):
            failures.append(f"[3] checks is not a list: {type(checks).__name__!r}")
            print(f"    FAIL  checks is {type(checks).__name__!r}, expected list")
            ok = False
        if ok:
            print(
                f"    PASS  verdict={data.get('verdict')!r}  "
                f"checks={len(checks)} items  wrote_files=False"
            )
    except Exception as exc:
        failures.append(f"[3] validate_budget (fbx) raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    # ---- [4] validate_budget -- chaos_gc FAIL path -------------------------
    print("\n  [4] validate_budget (chaos_gc, non-sequential gc_piece) -- gc_piece_sequential FAIL")
    try:
        hou.hipFile.clear(suppress_save_prompt=True)
        geo2 = hou.node("/obj").createNode("geo", "geo_smoke_chaos")
        box2 = geo2.createNode("box", "box1")
        wr = geo2.createNode("attribwrangle", "gc_piece_wrangle")
        wr.setFirstInput(box2)
        wr.parm("class").set(1)  # prim class
        # Box has 6 faces; assign 0,1,3,3,... (gap at 2 starting from prim 2)
        wr.parm("snippet").set(
            "i@unreal_gc_piece = (@primnum==2) ? 3 : @primnum;"
        )
        wr.cook(force=True)
        fixture_path = wr.path()

        result = dispatch("validate_budget", {"node": fixture_path, "target": "chaos_gc"})
        data = _data(result)
        ok = True

        if data.get("wrote_files") is not False:
            failures.append(
                f"[4] validate_budget (chaos_gc) not dry-run: wrote_files={data.get('wrote_files')!r}"
            )
            print(f"    FAIL  wrote_files={data.get('wrote_files')!r}")
            ok = False

        checks = data.get("checks", [])
        if not isinstance(checks, list):
            failures.append(f"[4] checks is not a list: {type(checks).__name__!r}")
            print(f"    FAIL  checks is {type(checks).__name__!r}, expected list")
            ok = False
        else:
            # checks is a LIST -- find by 'id' field
            gc_check = next((c for c in checks if c.get("id") == "gc_piece_sequential"), None)
            if gc_check is None:
                failures.append(
                    f"[4] gc_piece_sequential check not found in checks list; "
                    f"check ids: {[c.get('id') for c in checks]}"
                )
                print(
                    f"    FAIL  gc_piece_sequential not in checks; "
                    f"ids: {[c.get('id') for c in checks]}"
                )
                ok = False
            else:
                gc_status = gc_check.get("status")
                if gc_status != "fail":
                    failures.append(
                        f"[4] gc_piece_sequential status={gc_status!r} (expected 'fail')"
                    )
                    print(
                        f"    FAIL  gc_piece_sequential status={gc_status!r} (expected 'fail')"
                    )
                    ok = False
                if ok:
                    print(
                        f"    PASS  verdict={data.get('verdict')!r}  "
                        f"gc_piece_sequential status='fail'  wrote_files=False"
                    )
    except Exception as exc:
        failures.append(f"[4] validate_budget (chaos_gc) raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    print()
    total = 4
    passed = total - len([f for f in failures if f.startswith("[")])
    # count by group
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
