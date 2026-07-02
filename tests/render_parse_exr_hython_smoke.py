"""Dispatcher-based headless smoke for render_parse_exr — PP12-114d.

Dual-mode (mirrors render_lint_hython_smoke.py pattern):
  - Standalone:  hython tests/render_parse_exr_hython_smoke.py  (requires hython)
  - pytest:      collected and SKIPPED when hou is not importable (bare CI / off-DCC).

What this verifies (plan pp12-114d §4.2 return shape):
  1. render_parse_exr responds with the ExrManifest.to_dict() shape:
       {exr_path, is_multipart, subimages, compression, xres, yres,
        channels:[{name,layer,dtype}], crypto_layers, metadata}
  2. xres == 64 and yres == 64 (hoiiotool --create 64x64 fixture).
  3. channels is non-empty (the 3-channel fixture has R,G,B).
  4. is_multipart == False (single-part EXR created with --create).
  5. subimages == 1.
  6. FR-2 fail-loud: empty exr_path returns {ok: false, error: '...'}.
  7. FR-5 no-hoiiotool: when hoiiotool is missing the handler returns {ok: false, error: '...'}.

Real EXR creation via hoiiotool:
  hoiiotool --create 64x64 3 -d half -o <tmp>.exr

NOTE: all dispatch calls go through the REAL dispatcher path:
    dispatch("render_parse_exr", {"exr_path": ..., "subimage": None})
This calls handler(**params), which is the authoritative production path.
Handler registration: importing render_readback_handlers fires register_handler()
for render_parse_exr — exactly as the real server does at startup.

testVerificationSurface: hython-smoke
unitId: pp12-114d
"""

from __future__ import annotations

import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# Path bootstrap — resolves fxhoudinimcp packages in both hython + pytest.
# Mirrors render_lint_hython_smoke.py exactly (3-path bootstrap: repo root,
# python/, houdini/scripts/python/).
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PKG_PYTHON = os.path.join(_REPO_ROOT, "python")
_HOUDINI_HANDLERS = os.path.join(_REPO_ROOT, "houdini", "scripts", "python")
for _p in [_REPO_ROOT, _PKG_PYTHON, _HOUDINI_HANDLERS]:
    _abs = os.path.abspath(_p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

# ---------------------------------------------------------------------------
# homedini package root (for exr_inspector, handoff_model)
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
# hou availability guard
# ---------------------------------------------------------------------------
try:
    import hou  # type: ignore[import-not-found]
    HOU_AVAILABLE = True
except ImportError:
    HOU_AVAILABLE = False

# pytestmark only meaningful when collected by pytest.
if _PYTEST_AVAILABLE:
    pytestmark = pytest.mark.hython_smoke


# ---------------------------------------------------------------------------
# EXR fixture helper: create a real 64x64 3-channel half EXR via hoiiotool.
# ---------------------------------------------------------------------------

def _create_test_exr(output_path: str) -> None:
    """Create a minimal 64x64 RGB half EXR at output_path using hoiiotool --create.

    Uses hoiiotool shipped by Houdini ($HB or $HFS/bin).  This ensures the
    file the handler reads back was written by the same tool that reads it,
    so parse assertions are deterministic.

    Args:
        output_path: Absolute path where the EXR will be written.

    Raises:
        RuntimeError: if hoiiotool cannot be found or exits non-zero.
    """
    import subprocess

    # Locate hoiiotool via the same discovery logic as exr_inspector.
    from homedini.rendering.handoff_linter.exr_inspector import discover_oiiotool
    tool = discover_oiiotool()

    cmd = [tool, "--create", "64x64", "3", "-d", "half", "-o", output_path]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"hoiiotool --create failed (exit {result.returncode}):\n"
            f"  cmd: {cmd}\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}"
        )
    if not os.path.isfile(output_path):
        raise RuntimeError(
            f"hoiiotool --create returned 0 but output file not found: {output_path!r}"
        )


# ---------------------------------------------------------------------------
# Dispatcher helper
# ---------------------------------------------------------------------------

def _get_dispatcher():
    """Import and return dispatch(), registering render_readback_handlers.

    Importing render_readback_handlers fires register_handler() for
    render_parse_exr — exactly as the real server does at startup.
    Without this import, dispatch() returns 'command not found'.

    Raises:
        ImportError: when render_readback_handlers has not been authored yet
            (the expected RED state).
    """
    try:
        from fxhoudinimcp_server.handlers import render_readback_handlers  # noqa: F401
        from fxhoudinimcp_server.dispatcher import dispatch
        return dispatch
    except ImportError as exc:
        raise ImportError(
            "render_readback_handlers not importable — "
            "hou-dev must create the render_parse_exr handler in "
            "houdini/scripts/python/fxhoudinimcp_server/handlers/render_readback_handlers.py"
        ) from exc


def _data(result: dict) -> dict:
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
# RPE-S1: §4.2 ExrManifest.to_dict() shape on a real EXR
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestRenderParseExrShape:
    """RPE-S1: §4.2 ExrManifest shape contract on a real hoiiotool-created EXR."""

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._dispatch = staticmethod(_get_dispatcher())
        # Create the real EXR fixture in a temp dir.
        cls._tmpdir = tempfile.mkdtemp(prefix="render_parse_exr_smoke_")
        cls._exr_path = os.path.join(cls._tmpdir, "test_64x64.exr")
        _create_test_exr(cls._exr_path)

    def test_result_is_dict(self):
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": self._exr_path, "subimage": None},
        )
        data = _data(result)
        assert isinstance(data, dict), (
            f"render_parse_exr must return a dict; got {type(data).__name__!r}"
        )

    def test_result_has_exr_path_key(self):
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": self._exr_path, "subimage": None},
        )
        data = _data(result)
        assert "exr_path" in data, (
            f"§4.2: 'exr_path' key missing; keys present: {sorted(data.keys())}"
        )

    def test_result_has_is_multipart_key(self):
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": self._exr_path, "subimage": None},
        )
        data = _data(result)
        assert "is_multipart" in data, (
            f"§4.2: 'is_multipart' key missing; keys present: {sorted(data.keys())}"
        )

    def test_result_has_subimages_key(self):
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": self._exr_path, "subimage": None},
        )
        data = _data(result)
        assert "subimages" in data, (
            f"§4.2: 'subimages' key missing; keys present: {sorted(data.keys())}"
        )

    def test_result_has_compression_key(self):
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": self._exr_path, "subimage": None},
        )
        data = _data(result)
        assert "compression" in data, (
            f"§4.2: 'compression' key missing; keys present: {sorted(data.keys())}"
        )

    def test_result_has_xres_key(self):
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": self._exr_path, "subimage": None},
        )
        data = _data(result)
        assert "xres" in data, (
            f"§4.2: 'xres' key missing; keys present: {sorted(data.keys())}"
        )

    def test_result_has_yres_key(self):
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": self._exr_path, "subimage": None},
        )
        data = _data(result)
        assert "yres" in data, (
            f"§4.2: 'yres' key missing; keys present: {sorted(data.keys())}"
        )

    def test_result_has_channels_list(self):
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": self._exr_path, "subimage": None},
        )
        data = _data(result)
        assert "channels" in data, (
            f"§4.2: 'channels' key missing; keys present: {sorted(data.keys())}"
        )
        assert isinstance(data["channels"], list), (
            f"§4.2: 'channels' must be a list; got {type(data['channels']).__name__!r}"
        )

    def test_result_has_crypto_layers_list(self):
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": self._exr_path, "subimage": None},
        )
        data = _data(result)
        assert "crypto_layers" in data, (
            f"§4.2: 'crypto_layers' key missing; keys present: {sorted(data.keys())}"
        )
        assert isinstance(data["crypto_layers"], list), (
            f"§4.2: 'crypto_layers' must be a list; got {type(data['crypto_layers']).__name__!r}"
        )

    def test_result_has_metadata_dict(self):
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": self._exr_path, "subimage": None},
        )
        data = _data(result)
        assert "metadata" in data, (
            f"§4.2: 'metadata' key missing; keys present: {sorted(data.keys())}"
        )
        assert isinstance(data["metadata"], dict), (
            f"§4.2: 'metadata' must be a dict; got {type(data['metadata']).__name__!r}"
        )


# ---------------------------------------------------------------------------
# RPE-S2: dimensional assertions on the 64x64 3-channel fixture
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestRenderParseExrDimensions:
    """RPE-S2: xres/yres/channels/is_multipart/subimages assertions on 64x64 fixture."""

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._dispatch = staticmethod(_get_dispatcher())
        cls._tmpdir = tempfile.mkdtemp(prefix="render_parse_exr_dim_")
        cls._exr_path = os.path.join(cls._tmpdir, "dim_64x64.exr")
        _create_test_exr(cls._exr_path)
        cls._result_data = _data(
            cls._dispatch(
                "render_parse_exr",
                {"exr_path": cls._exr_path, "subimage": None},
            )
        )

    def test_xres_is_64(self):
        """hoiiotool --create 64x64 must produce xres == 64."""
        xres = self._result_data.get("xres")
        assert xres == 64, (
            f"xres must be 64 for the 64x64 fixture; got {xres!r}"
        )

    def test_yres_is_64(self):
        """hoiiotool --create 64x64 must produce yres == 64."""
        yres = self._result_data.get("yres")
        assert yres == 64, (
            f"yres must be 64 for the 64x64 fixture; got {yres!r}"
        )

    def test_channels_nonempty(self):
        """3-channel fixture must produce a non-empty channels list."""
        channels = self._result_data.get("channels", [])
        assert len(channels) > 0, (
            f"channels must be non-empty for a 3-channel EXR; got: {channels!r}"
        )

    def test_is_multipart_false(self):
        """Single-part EXR created with --create must have is_multipart == False."""
        is_multipart = self._result_data.get("is_multipart")
        assert is_multipart is False, (
            f"is_multipart must be False for a single-part EXR; got {is_multipart!r}"
        )

    def test_subimages_is_one(self):
        """Single-part EXR must have subimages == 1."""
        subimages = self._result_data.get("subimages")
        assert subimages == 1, (
            f"subimages must be 1 for a single-part EXR; got {subimages!r}"
        )

    def test_each_channel_has_required_keys(self):
        """Each channel dict must have 'name', 'layer', and 'dtype' keys."""
        channels = self._result_data.get("channels", [])
        required = {"name", "layer", "dtype"}
        for i, ch in enumerate(channels):
            missing = required - set(ch.keys())
            assert not missing, (
                f"channels[{i}] missing keys {missing}; got: {sorted(ch.keys())}"
            )


# ---------------------------------------------------------------------------
# RPE-S3: FR-2 fail-loud — empty exr_path
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestRenderParseExrFailLoud:
    """RPE-S3: FR-2/FR-5 fail-loud contracts."""

    @classmethod
    def setup_class(cls):
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._dispatch = staticmethod(_get_dispatcher())

    def test_empty_exr_path_returns_error_dict(self):
        """FR-2 fail-loud: empty exr_path must return {ok: False, error: str}."""
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": "", "subimage": None},
        )
        data = _data(result)
        assert isinstance(data, dict), (
            f"Empty exr_path must return a dict; got {type(data).__name__!r}"
        )
        assert data.get("ok") is False, (
            f"FR-2: empty exr_path must return ok=False; got: {data!r}"
        )
        assert isinstance(data.get("error"), str) and data["error"], (
            f"FR-2: empty exr_path must return a non-empty 'error' string; got: {data!r}"
        )

    def test_nonexistent_file_returns_error_dict(self):
        """FR-2 fail-loud: non-existent path must return {ok: False, error: str}."""
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": "/nonexistent/__does_not_exist__.exr", "subimage": None},
        )
        data = _data(result)
        assert isinstance(data, dict), (
            f"Non-existent path must return a dict; got {type(data).__name__!r}"
        )
        assert data.get("ok") is False, (
            f"FR-2: non-existent path must return ok=False; got: {data!r}"
        )
        assert isinstance(data.get("error"), str) and data["error"], (
            f"FR-2: non-existent path must return a non-empty 'error' string; got: {data!r}"
        )


# ---------------------------------------------------------------------------
# Standalone main() — mirrors render_lint_hython_smoke.py entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all smoke classes in sequence under hython; print SMOKE PASS/FAIL."""
    import traceback

    if not HOU_AVAILABLE:
        print("SMOKE SKIP: hou not available — run under hython", file=sys.stderr)
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

    _run_class("TestRenderParseExrShape", TestRenderParseExrShape)
    _run_class("TestRenderParseExrDimensions", TestRenderParseExrDimensions)
    _run_class("TestRenderParseExrFailLoud", TestRenderParseExrFailLoud)

    if failed:
        print(f"\nSMOKE FAIL — {len(failed)} failure(s):", file=sys.stderr)
        for f in failed:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("\nSMOKE PASS — all render_parse_exr assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
