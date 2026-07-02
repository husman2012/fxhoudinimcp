"""Dispatcher-based headless smoke for render_lint_settings — PP12-114c.

Dual-mode (mirrors export_hython_smoke.py pattern):
  - Standalone:  hython tests/render_lint_hython_smoke.py  (requires hython)
  - pytest:      collected and SKIPPED when hou is not importable (bare CI / off-DCC).

What this verifies (plan pp12-114c §4.2 return shape):
  1. render_lint_settings responds with the §4.2 JSON shape:
       {render_node, preset, results:[RuleResult.to_dict], summary:{ok,warn,error},
        ready_to_render}
  2. results is a list of dicts each with keys: rule_id, title, severity, message,
       fix_id (str or null), target_paths (list[str]).
  3. The broken Karma/UsdRender stage fires at minimum:
       - crypto_name rule (CryptoPrimitives var, crypto_kind='primitive')
       - legacy_exr rule (beauty product: multipart=True + cryptomatte var present)
  4. ready_to_render is False (errors present on the stage).
  5. summary['error'] > 0.
  6. FR-2 fail-loud: invalid render_node returns {ok: false, error: '...'}.

The BROKEN stage (reused from E4 stage_reader_smoke.py p11-05-e4 fixture):
  /Render/Products/beauty     RenderProduct (raster, compression=zip, multipart=True)
    /Vars/Cf                  RenderVar sourceName='C'           beauty AOV
    /Vars/CryptoObject        RenderVar sourceName='CryptoObject' Nuke-friendly crypto
    /Vars/CryptoPrimitives    RenderVar sourceName='CryptoPrimitives' ← fires crypto_name
    /Vars/diffuse_clr         RenderVar sourceName='diffuse', channel_prefix='diffuse'
  /Render/Products/depth_only RenderProduct (raster, no compression)
    /Vars/Pz                  RenderVar sourceName='Pz'          missing required AOVs

Expected rule firings:
  - crypto_name: CryptoPrimitives uses crypto_kind='primitive' (Nuke degrades)
  - legacy_exr:  beauty product is multipart + contains cryptomatte vars

NOTE: all dispatch calls go through the REAL dispatcher path:
    dispatch("render_lint_settings", {"render_node": ..., "preset": ...})
This calls handler(**params), which is the authoritative production path.
Handler registration: importing render_readback_handlers fires register_handler()
for render_lint_settings — exactly as the real server does at startup.

testVerificationSurface: hython-smoke
unitId: pp12-114c
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — resolves fxhoudinimcp packages in both hython + pytest.
# Mirrors export_hython_smoke.py exactly (3-path bootstrap: repo root, python/,
# houdini/scripts/python/).
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PKG_PYTHON = os.path.join(_REPO_ROOT, "python")
_HOUDINI_HANDLERS = os.path.join(_REPO_ROOT, "houdini", "scripts", "python")
for _p in [_REPO_ROOT, _PKG_PYTHON, _HOUDINI_HANDLERS]:
    _abs = os.path.abspath(_p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

# ---------------------------------------------------------------------------
# homedini package root (for stage_reader, rules, handoff_linter)
# ---------------------------------------------------------------------------
_HOMEDINI_PKG = os.path.abspath(
    os.path.join(_REPO_ROOT, "..", "HoudiniUtilTools", "scripts", "python")
)
if os.path.isdir(_HOMEDINI_PKG) and _HOMEDINI_PKG not in sys.path:
    sys.path.insert(0, _HOMEDINI_PKG)

# ---------------------------------------------------------------------------
# pytest availability guard — hython does NOT ship pytest.
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
# hou / pxr availability guards
# ---------------------------------------------------------------------------
try:
    import hou  # type: ignore[import-not-found]
    HOU_AVAILABLE = True
except ImportError:
    HOU_AVAILABLE = False

try:
    import pxr  # type: ignore[import-not-found]  # noqa: F401
    PXR_AVAILABLE = True
except ImportError:
    PXR_AVAILABLE = False

# pytestmark only meaningful when collected by pytest.
if _PYTEST_AVAILABLE:
    pytestmark = pytest.mark.hython_smoke


# ---------------------------------------------------------------------------
# Broken Karma/UsdRender stage builder (LOP node — real scene path)
#
# Bug fix (pp12-114c): the previous _build_test_stage() created an in-memory
# pxr.Usd.Stage that was never connected to a live LOP node.  The handler
# calls stage_reader.read(node) -> node.stage(), so it requires a REAL hou.Node
# whose composed stage carries the UsdRender prims.
#
# Fix: author the identical broken-Karma content into a pythonscript LOP node
# via hou.pwd().editableStage() so that node.stage() returns the prims.
# Probe-confirmed (hython): node.stage().Traverse() yields 2 RenderProducts +
# 5 RenderVars before the fix was committed.
# ---------------------------------------------------------------------------

# The Python code that runs INSIDE the pythonscript LOP's execution context.
# hou.pwd() resolves to the pythonscript node itself; editableStage() returns
# its writable USD layer.  pxr imports are available in the LOP context.
_BROKEN_KARMA_SCRIPT = """\
from pxr import UsdRender, Sdf
node = hou.pwd()
stage = node.editableStage()

UsdRender.Settings.Define(stage, "/Render/settings")

# Product 1: beauty (multipart + CryptoPrimitives = broken)
prod1_path = "/Render/Products/beauty"
prod1 = UsdRender.Product.Define(stage, prod1_path)
prod1.GetProductTypeAttr().Set("raster")
prod1.GetProductNameAttr().Set("/tmp/beauty.####.exr")

p1 = stage.GetPrimAtPath(prod1_path)
p1.CreateAttribute("driver:parameters:OpenEXR:compression", Sdf.ValueTypeNames.String).Set("zip")
p1.CreateAttribute("driver:parameters:OpenEXR:multipart", Sdf.ValueTypeNames.Bool).Set(True)

# Beauty AOV
var_cf = UsdRender.Var.Define(stage, "/Render/Products/beauty/Vars/Cf")
var_cf.GetSourceNameAttr().Set("C")
var_cf.GetDataTypeAttr().Set("color3f")

# Nuke-friendly cryptomatte
var_co = UsdRender.Var.Define(stage, "/Render/Products/beauty/Vars/CryptoObject")
var_co.GetSourceNameAttr().Set("CryptoObject")
var_co.GetDataTypeAttr().Set("color4f")

# BROKEN: primitive cryptomatte -- fires crypto_name rule
var_cp = UsdRender.Var.Define(stage, "/Render/Products/beauty/Vars/CryptoPrimitives")
var_cp.GetSourceNameAttr().Set("CryptoPrimitives")
var_cp.GetDataTypeAttr().Set("color4f")

# AOV with channel_prefix -- fires lowercase_channel if rule is present
var_diff_path = "/Render/Products/beauty/Vars/diffuse_clr"
var_diff = UsdRender.Var.Define(stage, var_diff_path)
var_diff.GetSourceNameAttr().Set("diffuse")
var_diff.GetDataTypeAttr().Set("color3f")
stage.GetPrimAtPath(var_diff_path).CreateAttribute(
    "driver:parameters:aov:channel_prefix", Sdf.ValueTypeNames.String
).Set("diffuse")

prod1.GetOrderedVarsRel().SetTargets([
    Sdf.Path("/Render/Products/beauty/Vars/Cf"),
    Sdf.Path("/Render/Products/beauty/Vars/CryptoObject"),
    Sdf.Path("/Render/Products/beauty/Vars/CryptoPrimitives"),
    Sdf.Path(var_diff_path),
])

# Product 2: depth_only (no compression, no crypto -- missing required AOVs)
prod2_path = "/Render/Products/depth_only"
prod2 = UsdRender.Product.Define(stage, prod2_path)
prod2.GetProductTypeAttr().Set("raster")
prod2.GetProductNameAttr().Set("/tmp/depth.####.exr")

var_pz = UsdRender.Var.Define(stage, "/Render/Products/depth_only/Vars/Pz")
var_pz.GetSourceNameAttr().Set("Pz")
var_pz.GetDataTypeAttr().Set("float")
prod2.GetOrderedVarsRel().SetTargets([Sdf.Path("/Render/Products/depth_only/Vars/Pz")])
"""


def _build_lop_stage(node_name: str = "broken_karma_lint_stage") -> str:
    """Create a pythonscript LOP in /stage that authors the broken-Karma fixture.

    Authors the same UsdRender prims as E4's stage_reader_smoke.py fixture into
    a real Houdini LOP node so that node.stage() returns a stage with the prims.
    The handler calls stage_reader.read(node) -> node.stage(), so the fixture
    must live on a live hou.Node (not an in-memory pxr.Usd.Stage).

    The pythonscript LOP executes _BROKEN_KARMA_SCRIPT on cook, which authors
    2 RenderProducts + 5 RenderVars into the node's writable USD layer via
    hou.pwd().editableStage().

    Args:
        node_name: Name for the pythonscript LOP node under /stage.

    Returns:
        The scene path of the created node (e.g. '/stage/broken_karma_lint_stage').
        Pass this path to dispatch('render_lint_settings', {'render_node': path}).

    Raises:
        RuntimeError: if hou is not importable or /stage context does not exist.
    """
    if not HOU_AVAILABLE:
        raise RuntimeError("_build_lop_stage: hou not available -- run under hython")

    stage_ctx = hou.node("/stage")
    if stage_ctx is None:
        raise RuntimeError("_build_lop_stage: /stage context not found in scene")

    ps = stage_ctx.createNode("pythonscript", node_name)
    ps.parm("python").set(_BROKEN_KARMA_SCRIPT)
    # Trigger a cook by accessing the stage; validates prims were authored.
    node_stage = ps.stage()
    if node_stage is None:
        raise RuntimeError(
            f"_build_lop_stage: node.stage() returned None for {ps.path()!r}; "
            "pythonscript cook may have failed."
        )
    return ps.path()


# ---------------------------------------------------------------------------
# Dispatcher helper
# ---------------------------------------------------------------------------

def _get_dispatcher():
    """Import and return dispatch(), registering render_readback_handlers.

    Importing render_readback_handlers fires register_handler() for
    render_lint_settings — exactly as the real server does at startup.
    Without this import, dispatch() returns 'command not found'.
    """
    try:
        from fxhoudinimcp_server.handlers import render_readback_handlers  # noqa: F401
        from fxhoudinimcp_server.dispatcher import dispatch
        return dispatch
    except ImportError as exc:
        raise ImportError(
            "render_readback_handlers not importable — "
            "hou-dev must create "
            "houdini/scripts/python/fxhoudinimcp_server/handlers/render_readback_handlers.py"
        ) from exc


def _data(result):
    """Unwrap dispatcher envelope.

    dispatch() returns {'status': 'success', 'data': {...}, 'timing_ms': N}.
    Assertions must target the unwrapped data, not the envelope.
    """
    return result.get("data", result)


# ---------------------------------------------------------------------------
# Assertion helper
# ---------------------------------------------------------------------------

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Smoke: §4.2 shape + expected rule firings
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestRenderLintSettingsShape:
    """RL-S1: §4.2 JSON shape contract."""

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._dispatch = staticmethod(_get_dispatcher())
        # Build a real LOP node whose stage() carries the broken-Karma prims.
        # The handler calls stage_reader.read(node) -> node.stage(); the node
        # path is the required render_node argument (not a bare lopnet or Stage).
        cls._render_node_path = _build_lop_stage("smoke_lint_shape")

    def test_result_is_dict(self):
        result = self._dispatch(
            "render_lint_settings",
            {"render_node": self._render_node_path, "preset": "nuke_safe"},
        )
        data = _data(result)
        assert isinstance(data, dict), (
            f"render_lint_settings must return a dict; got {type(data).__name__!r}"
        )

    def test_result_has_render_node_key(self):
        result = self._dispatch(
            "render_lint_settings",
            {"render_node": self._render_node_path, "preset": "nuke_safe"},
        )
        data = _data(result)
        assert "render_node" in data, (
            f"§4.2: 'render_node' key missing; keys present: {sorted(data.keys())}"
        )

    def test_result_has_preset_key(self):
        result = self._dispatch(
            "render_lint_settings",
            {"render_node": self._render_node_path, "preset": "nuke_safe"},
        )
        data = _data(result)
        assert "preset" in data, (
            f"§4.2: 'preset' key missing; keys present: {sorted(data.keys())}"
        )

    def test_result_has_results_list(self):
        result = self._dispatch(
            "render_lint_settings",
            {"render_node": self._render_node_path, "preset": "nuke_safe"},
        )
        data = _data(result)
        assert "results" in data, (
            f"§4.2: 'results' key missing; keys present: {sorted(data.keys())}"
        )
        assert isinstance(data["results"], list), (
            f"§4.2: 'results' must be a list; got {type(data['results']).__name__!r}"
        )

    def test_result_has_summary_dict(self):
        result = self._dispatch(
            "render_lint_settings",
            {"render_node": self._render_node_path, "preset": "nuke_safe"},
        )
        data = _data(result)
        assert "summary" in data, (
            f"§4.2: 'summary' key missing; keys present: {sorted(data.keys())}"
        )
        summary = data["summary"]
        assert isinstance(summary, dict), (
            f"§4.2: 'summary' must be a dict; got {type(summary).__name__!r}"
        )
        for sub_key in ("ok", "warn", "error"):
            assert sub_key in summary, (
                f"§4.2: summary['{sub_key}'] missing; summary keys: {sorted(summary.keys())}"
            )

    def test_result_has_ready_to_render(self):
        result = self._dispatch(
            "render_lint_settings",
            {"render_node": self._render_node_path, "preset": "nuke_safe"},
        )
        data = _data(result)
        assert "ready_to_render" in data, (
            f"§4.2: 'ready_to_render' key missing; keys present: {sorted(data.keys())}"
        )
        assert isinstance(data["ready_to_render"], bool), (
            f"§4.2: 'ready_to_render' must be bool; got {type(data['ready_to_render']).__name__!r}"
        )


@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestRenderLintSettingsRuleResults:
    """RL-S2: RuleResult shape and expected rule firings on the broken stage."""

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._dispatch = staticmethod(_get_dispatcher())
        # Build a real LOP node with the broken-Karma stage content.
        cls._render_node_path = _build_lop_stage("smoke_lint_rules")
        cls._result_data = _data(
            cls._dispatch(
                "render_lint_settings",
                {"render_node": cls._render_node_path, "preset": "nuke_safe"},
            )
        )

    def test_each_result_has_required_keys(self):
        results = self._result_data.get("results", [])
        required = {"rule_id", "title", "severity", "message", "fix_id", "target_paths"}
        for i, r in enumerate(results):
            missing = required - set(r.keys())
            assert not missing, (
                f"Result[{i}] missing keys {missing}; got: {sorted(r.keys())}"
            )

    def test_fix_id_is_str_or_null(self):
        results = self._result_data.get("results", [])
        for i, r in enumerate(results):
            fix_id = r.get("fix_id")
            assert fix_id is None or isinstance(fix_id, str), (
                f"Result[{i}].fix_id must be str or null; got {type(fix_id).__name__!r}"
            )

    def test_target_paths_is_list(self):
        results = self._result_data.get("results", [])
        for i, r in enumerate(results):
            tp = r.get("target_paths")
            assert isinstance(tp, list), (
                f"Result[{i}].target_paths must be a list; got {type(tp).__name__!r}"
            )

    def test_crypto_name_rule_fires(self):
        """CryptoPrimitives var (crypto_kind='primitive') must fire crypto_name."""
        results = self._result_data.get("results", [])
        fix_ids = [r.get("fix_id") for r in results]
        assert "crypto_name" in fix_ids, (
            f"Expected fix_id='crypto_name' in results (CryptoPrimitives var), "
            f"but fix_ids seen: {fix_ids}"
        )

    def test_legacy_exr_rule_fires(self):
        """beauty product (multipart + cryptomatte) must fire legacy_exr."""
        results = self._result_data.get("results", [])
        fix_ids = [r.get("fix_id") for r in results]
        assert "legacy_exr" in fix_ids, (
            f"Expected fix_id='legacy_exr' in results (beauty: multipart + crypto), "
            f"but fix_ids seen: {fix_ids}"
        )

    def test_ready_to_render_is_false(self):
        """Broken stage has errors — ready_to_render must be False."""
        assert self._result_data.get("ready_to_render") is False, (
            f"ready_to_render must be False when errors are present; "
            f"got: {self._result_data.get('ready_to_render')!r}"
        )

    def test_summary_error_count_gt_zero(self):
        """Broken stage must have summary['error'] > 0."""
        summary = self._result_data.get("summary", {})
        error_count = summary.get("error", 0)
        assert error_count > 0, (
            f"summary['error'] must be > 0 for the broken stage; got {error_count}"
        )


# ---------------------------------------------------------------------------
# RL-S3: FR-2 fail-loud — invalid render_node
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestRenderLintSettingsFailLoud:
    """RL-S3: FR-2 — invalid render_node returns {ok: False, error: '...'} not silence."""

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._dispatch = staticmethod(_get_dispatcher())

    def test_invalid_render_node_returns_error_dict(self):
        """FR-2 fail-loud: non-existent path must return a dict with ok=False and error str."""
        result = self._dispatch(
            "render_lint_settings",
            {"render_node": "/stage/__nonexistent__", "preset": "nuke_safe"},
        )
        data = _data(result)
        assert isinstance(data, dict), (
            f"Invalid render_node must return a dict; got {type(data).__name__!r}"
        )
        assert data.get("ok") is False, (
            f"FR-2: invalid render_node must return ok=False; got: {data!r}"
        )
        assert isinstance(data.get("error"), str) and data["error"], (
            f"FR-2: invalid render_node must return a non-empty 'error' string; got: {data!r}"
        )


# ---------------------------------------------------------------------------
# Standalone main() — mirrors export_hython_smoke.py entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all smoke classes in sequence under hython; print SMOKE PASS/FAIL."""
    import traceback

    if not HOU_AVAILABLE:
        print("SMOKE SKIP: hou not available — run under hython", file=sys.stderr)
        return 0
    if not PXR_AVAILABLE:
        print("SMOKE SKIP: pxr not available — run under hython", file=sys.stderr)
        return 0

    failed: list[str] = []

    def _run_class(cls_name, cls):
        try:
            cls.setup_class()
        except ImportError as exc:
            failed.append(
                f"{cls_name}.setup_class IMPORT ERROR (expected RED): {exc}"
            )
            return
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{cls_name}.setup_class ERROR: {exc}")
            traceback.print_exc(file=sys.stderr)
            return

        instance = cls()
        for attr in dir(cls):
            if not attr.startswith("test_"):
                continue
            fn = getattr(instance, attr)
            try:
                fn()
                print(f"  PASS  {cls_name}.{attr}")
            except AssertionError as exc:
                print(f"  FAIL  {cls_name}.{attr}: {exc}", file=sys.stderr)
                failed.append(f"{cls_name}.{attr}: {exc}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR {cls_name}.{attr}: {exc}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                failed.append(f"{cls_name}.{attr}: {exc}")

    _run_class("TestRenderLintSettingsShape", TestRenderLintSettingsShape)
    _run_class("TestRenderLintSettingsRuleResults", TestRenderLintSettingsRuleResults)
    _run_class("TestRenderLintSettingsFailLoud", TestRenderLintSettingsFailLoud)

    if failed:
        print(f"\nSMOKE FAIL — {len(failed)} failure(s):", file=sys.stderr)
        for f in failed:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("\nSMOKE PASS — all render_lint_settings assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
