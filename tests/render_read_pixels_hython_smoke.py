"""hython-smoke tests for render_read_pixels handler.

Run from hython (headless Houdini 21):
    "C:/Program Files/Side Effects Software/Houdini 21.0.729/bin/hython.exe" \
        C:/Users/husma/development/fxhoudinimcp/tests/render_read_pixels_hython_smoke.py

Expected result: RED — handler absent (UNKNOWN_COMMAND) until hou-dev authors
fxhoudinimcp_server/handlers/render_readback_handlers.py and registers
'render_read_pixels' with the dispatcher.

Test surface: hython-smoke
unitId: pp12-114e
Plan contract tested (pp12-114e, §4.2):
  - summary mode: returns plane, xres, yres, channels, dtype, mode, stats, histogram,
                  pixels (empty), page, page_size, total_pages, truncated
  - sample mode:  pixels is a non-empty list when max_pixels is large enough
  - roi mode:     honours [x0, y0, x1, y1] HALF-OPEN crop rectangle
                  roi=[0,0,8,8] on a 64x64 image -> 8*8 pixels max in sample submodes
  - truncated flag set when pixel list is capped by max_pixels

EXR fixture: 64x64 3-channel float16 created by oiiotool (discovered via
homedini.rendering.handoff_linter.exr_inspector.discover_oiiotool).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# 3-path bootstrap: make fxhoudinimcp_server importable in hython
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PKG_PYTHON = os.path.join(_REPO_ROOT, "python")
_HOUDINI_HANDLERS = os.path.join(_REPO_ROOT, "houdini", "scripts", "python")
for _p in [_REPO_ROOT, _PKG_PYTHON, _HOUDINI_HANDLERS]:
    _abs = os.path.abspath(_p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

# Optional: homedini package for discover_oiiotool
_HOMEDINI_PKG = os.path.abspath(
    os.path.join(_REPO_ROOT, "..", "HoudiniUtilTools", "scripts", "python")
)
if os.path.isdir(_HOMEDINI_PKG) and _HOMEDINI_PKG not in sys.path:
    sys.path.insert(0, _HOMEDINI_PKG)

# ---------------------------------------------------------------------------
# pytest shim: defines assert / PASS / FAIL symbols for non-pytest environments
# ---------------------------------------------------------------------------
try:
    import pytest  # noqa: F401
    _PYTEST_AVAILABLE = True
except ImportError:
    _PYTEST_AVAILABLE = False
    import types

    class _PytestShim:
        """Minimal pytest shim for hython environments without pytest."""

        @staticmethod
        def raises(exc_type):
            import contextlib

            @contextlib.contextmanager
            def _ctx():
                raised = []
                try:
                    yield raised
                except exc_type as e:
                    raised.append(e)
                except Exception as e:
                    raise AssertionError(
                        f"Expected {exc_type.__name__} but got {type(e).__name__}: {e}"
                    ) from e
                else:
                    raise AssertionError(
                        f"Expected {exc_type.__name__} was not raised"
                    )

            return _ctx()

    pytest = _PytestShim()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# EXR creation helper
# ---------------------------------------------------------------------------

def _create_test_exr(path: str, width: int = 64, height: int = 64) -> None:
    """Write a width×height 3-channel float16 EXR to *path* via oiiotool.

    Requires homedini.rendering.handoff_linter.exr_inspector.discover_oiiotool
    (present in HoudiniUtilTools on-disk at _HOMEDINI_PKG).
    """
    try:
        from homedini.rendering.handoff_linter.exr_inspector import discover_oiiotool

        oiiotool = discover_oiiotool()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            f"Cannot discover oiiotool: {exc}. "
            "Ensure HoudiniUtilTools is beside the fxhoudinimcp repo "
            "and homedini.rendering.handoff_linter.exr_inspector is importable."
        ) from exc

    # Generate a 3-channel float16 EXR with hoiiotool --create.
    # discover_oiiotool() returns the hoiiotool binary (Houdini's own OIIO variant).
    # Syntax: hoiiotool --create <WxH> <nchannels> -d half -o <path>
    size_str = f"{width}x{height}"
    cmd = [oiiotool, "--create", size_str, "3", "-d", "half", "-o", path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"oiiotool failed to create EXR at {path!r}.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    if not os.path.isfile(path):
        raise RuntimeError(f"oiiotool did not write EXR at {path!r}")


# ---------------------------------------------------------------------------
# Dispatcher resolver
# ---------------------------------------------------------------------------

def _get_dispatcher():
    """Import render_readback_handlers and return the dispatch callable.

    RED until hou-dev authors fxhoudinimcp_server/handlers/render_readback_handlers.py
    and registers 'render_read_pixels' with the dispatcher.
    Raises ImportError if the module does not yet exist.
    """
    try:
        from fxhoudinimcp_server.handlers import render_readback_handlers  # noqa: F401
        from fxhoudinimcp_server.dispatcher import dispatch
        return dispatch
    except ImportError as exc:
        raise ImportError(
            "render_readback_handlers not importable — this is expected RED. "
            f"Detail: {exc}"
        ) from exc


def _data(result: dict) -> dict:
    """Unwrap the dispatcher envelope {status, data} -> data dict."""
    if not isinstance(result, dict):
        raise AssertionError(f"dispatch() result must be a dict; got {type(result)!r}")
    if result.get("status") != "success":
        raise AssertionError(
            f"dispatch() returned non-success: {result.get('status')!r}; "
            f"full: {result!r}"
        )
    return result.get("data", result)


# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------

_PASS_COUNT = 0
_FAIL_COUNT = 0
_ERRORS: list[tuple[str, str]] = []


def _run_class(cls):
    """Run all test_* methods on a class instance; print per-test PASS/FAIL."""
    global _PASS_COUNT, _FAIL_COUNT
    inst = cls()
    # Setup
    if hasattr(inst, "setup_class"):
        # class-level setup
        try:
            cls.setup_class()
        except Exception as exc:
            print(f"  [SETUP-ERROR] {cls.__name__}.setup_class: {exc}")
            _FAIL_COUNT += 1
            _ERRORS.append((f"{cls.__name__}.setup_class", str(exc)))
            return
    for name in dir(cls):
        if not name.startswith("test_"):
            continue
        method = getattr(inst, name)
        try:
            method()
            print(f"  [PASS] {cls.__name__}.{name}")
            _PASS_COUNT += 1
        except AssertionError as exc:
            print(f"  [FAIL] {cls.__name__}.{name}: {exc}")
            _FAIL_COUNT += 1
            _ERRORS.append((f"{cls.__name__}.{name}", str(exc)))
        except Exception as exc:
            print(f"  [ERROR] {cls.__name__}.{name}: {type(exc).__name__}: {exc}")
            _FAIL_COUNT += 1
            _ERRORS.append((f"{cls.__name__}.{name}", f"{type(exc).__name__}: {exc}"))
    # Teardown
    if hasattr(inst, "teardown_class"):
        try:
            cls.teardown_class()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# RRP-S1: summary-mode response shape
# ---------------------------------------------------------------------------

class TestRenderReadPixelsSummaryShape:
    """RRP-S1: render_read_pixels summary mode returns §4.2 keys.

    RED until handler registers 'render_read_pixels' with the dispatcher.
    """

    @classmethod
    def setup_class(cls):
        try:
            import hou
            hou.hipFile.clear(suppress_save_prompt=True)
        except ImportError:
            pass  # not in hython — tests will still exercise the dispatcher path

        cls._dispatch = staticmethod(_get_dispatcher())
        cls._tmpdir = tempfile.mkdtemp(prefix="render_read_pixels_smoke_")
        cls._exr_path = os.path.join(cls._tmpdir, "test_64x64.exr")
        _create_test_exr(cls._exr_path)

    def test_dispatch_returns_dict(self):
        result = self._dispatch("render_read_pixels", {
            "source": self._exr_path,
            "plane": "C",
            "mode": "summary",
            "roi": None,
            "max_pixels": 4096,
            "downsample": 1,
            "page": 0,
            "page_size": 1024,
        })
        assert isinstance(result, dict), (
            f"dispatch must return a dict; got {type(result)!r}"
        )

    def test_summary_returns_plane_key(self):
        result = self._dispatch("render_read_pixels", {
            "source": self._exr_path,
            "plane": "C",
            "mode": "summary",
            "roi": None,
            "max_pixels": 4096,
            "downsample": 1,
            "page": 0,
            "page_size": 1024,
        })
        data = _data(result)
        assert "plane" in data, f"§4.2: 'plane' missing; keys: {sorted(data.keys())}"

    def test_summary_xres_equals_64(self):
        result = self._dispatch("render_read_pixels", {
            "source": self._exr_path,
            "plane": "C",
            "mode": "summary",
            "roi": None,
            "max_pixels": 4096,
            "downsample": 1,
            "page": 0,
            "page_size": 1024,
        })
        data = _data(result)
        assert data.get("xres") == 64, (
            f"§4.2: xres must be 64 for 64x64 EXR; got {data.get('xres')!r}"
        )

    def test_summary_yres_equals_64(self):
        result = self._dispatch("render_read_pixels", {
            "source": self._exr_path,
            "plane": "C",
            "mode": "summary",
            "roi": None,
            "max_pixels": 4096,
            "downsample": 1,
            "page": 0,
            "page_size": 1024,
        })
        data = _data(result)
        assert data.get("yres") == 64, (
            f"§4.2: yres must be 64 for 64x64 EXR; got {data.get('yres')!r}"
        )

    def test_summary_channels_equals_3(self):
        result = self._dispatch("render_read_pixels", {
            "source": self._exr_path,
            "plane": "C",
            "mode": "summary",
            "roi": None,
            "max_pixels": 4096,
            "downsample": 1,
            "page": 0,
            "page_size": 1024,
        })
        data = _data(result)
        assert data.get("channels") == 3, (
            f"§4.2: channels must be 3 for 3-channel EXR; got {data.get('channels')!r}"
        )

    def test_summary_pixels_is_empty_list(self):
        result = self._dispatch("render_read_pixels", {
            "source": self._exr_path,
            "plane": "C",
            "mode": "summary",
            "roi": None,
            "max_pixels": 4096,
            "downsample": 1,
            "page": 0,
            "page_size": 1024,
        })
        data = _data(result)
        assert "pixels" in data, f"§4.2: 'pixels' key missing; keys: {sorted(data.keys())}"
        assert data["pixels"] == [], (
            f"§4.2: summary mode must return pixels=[]; got {type(data['pixels'])!r} "
            f"with {len(data['pixels'])} entries"
        )

    def test_summary_stats_has_required_keys(self):
        result = self._dispatch("render_read_pixels", {
            "source": self._exr_path,
            "plane": "C",
            "mode": "summary",
            "roi": None,
            "max_pixels": 4096,
            "downsample": 1,
            "page": 0,
            "page_size": 1024,
        })
        data = _data(result)
        stats = data.get("stats", {})
        assert isinstance(stats, dict), (
            f"§4.2: stats must be a dict; got {type(stats)!r}"
        )
        required = {"min", "max", "mean", "nan_count", "inf_count"}
        missing = required - set(stats.keys())
        assert not missing, (
            f"§4.2: stats missing keys: {missing}; got: {sorted(stats.keys())}"
        )

    def test_summary_has_histogram_key(self):
        result = self._dispatch("render_read_pixels", {
            "source": self._exr_path,
            "plane": "C",
            "mode": "summary",
            "roi": None,
            "max_pixels": 4096,
            "downsample": 1,
            "page": 0,
            "page_size": 1024,
        })
        data = _data(result)
        assert "histogram" in data, (
            f"§4.2: 'histogram' missing; keys: {sorted(data.keys())}"
        )

    def test_summary_has_truncated_key(self):
        result = self._dispatch("render_read_pixels", {
            "source": self._exr_path,
            "plane": "C",
            "mode": "summary",
            "roi": None,
            "max_pixels": 4096,
            "downsample": 1,
            "page": 0,
            "page_size": 1024,
        })
        data = _data(result)
        assert "truncated" in data, (
            f"§4.2: 'truncated' key missing; keys: {sorted(data.keys())}"
        )


# ---------------------------------------------------------------------------
# RRP-S2: sample mode — pixels populated, truncation flag, roi
# ---------------------------------------------------------------------------

class TestRenderReadPixelsSampleAndRoi:
    """RRP-S2: sample + roi mode pixel content tests."""

    @classmethod
    def setup_class(cls):
        try:
            import hou
            hou.hipFile.clear(suppress_save_prompt=True)
        except ImportError:
            pass

        cls._dispatch = staticmethod(_get_dispatcher())
        cls._tmpdir = tempfile.mkdtemp(prefix="render_read_pixels_smoke2_")
        cls._exr_path = os.path.join(cls._tmpdir, "test_64x64.exr")
        _create_test_exr(cls._exr_path)

    def test_sample_mode_pixels_non_empty(self):
        """sample mode with large max_pixels must return a non-empty pixels list."""
        result = self._dispatch("render_read_pixels", {
            "source": self._exr_path,
            "plane": "C",
            "mode": "sample",
            "roi": None,
            "max_pixels": 65536,  # larger than 64*64=4096, so no truncation
            "downsample": 1,
            "page": 0,
            "page_size": 65536,
        })
        data = _data(result)
        pixels = data.get("pixels", [])
        assert isinstance(pixels, list), (
            f"pixels must be a list; got {type(pixels)!r}"
        )
        assert len(pixels) > 0, (
            "sample mode with max_pixels=65536 on 64x64 image must return non-empty pixels"
        )

    def test_sample_mode_truncated_when_max_pixels_small(self):
        """Requesting fewer pixels than the image size must set truncated=True."""
        result = self._dispatch("render_read_pixels", {
            "source": self._exr_path,
            "plane": "C",
            "mode": "sample",
            "roi": None,
            "max_pixels": 10,   # far below 64*64=4096
            "downsample": 1,
            "page": 0,
            "page_size": 10,
        })
        data = _data(result)
        assert data.get("truncated") is True, (
            f"truncated must be True when max_pixels=10 on 64x64 image; "
            f"got truncated={data.get('truncated')!r}"
        )
        pixels = data.get("pixels", [])
        assert len(pixels) <= 10, (
            f"pixels list must not exceed max_pixels=10; got {len(pixels)} entries"
        )

    def test_roi_mode_half_open_crop(self):
        """roi=[0,0,8,8] HALF-OPEN must crop to 8*8=64 pixels max in sample sub-mode.

        The roi is [x0, y0, x1, y1] where x1/y1 are EXCLUSIVE — so
        [0,0,8,8] yields 8 columns x 8 rows = 64 pixels.
        """
        result = self._dispatch("render_read_pixels", {
            "source": self._exr_path,
            "plane": "C",
            "mode": "roi",
            "roi": [0, 0, 8, 8],
            "max_pixels": 4096,  # large enough not to truncate within the crop
            "downsample": 1,
            "page": 0,
            "page_size": 4096,
        })
        data = _data(result)
        pixels = data.get("pixels", [])
        assert isinstance(pixels, list), (
            f"roi mode pixels must be a list; got {type(pixels)!r}"
        )
        # The crop is 8x8=64 pixels; with 3 channels each pixel is a list of 3,
        # OR pixels is a flat list of floats (64*3=192). Either way len must be <=192.
        assert len(pixels) <= 192, (
            f"roi=[0,0,8,8] should yield at most 8*8*3=192 values; "
            f"got {len(pixels)} entries"
        )
        assert len(pixels) > 0, (
            "roi=[0,0,8,8] on a 64x64 image must return at least 1 pixel"
        )


# ---------------------------------------------------------------------------
# RRP-S3: fail-loud tests (missing file, unknown command before handler exists)
# ---------------------------------------------------------------------------

class TestRenderReadPixelsFailLoud:
    """RRP-S3: handler must fail loudly for invalid inputs — even when handler exists."""

    @classmethod
    def setup_class(cls):
        cls._dispatch = staticmethod(_get_dispatcher())

    def test_missing_file_returns_error(self):
        """Missing file -> SUCCESS envelope with ok=False.

        Locked contract (pp12-114e lockedFieldContract): ALL handler-level failures
        return {status:'success', data:{ok:False, error:'<msg>'}} — NEVER a raised
        exception that the dispatcher wraps as {status:'error'}.
        """
        result = self._dispatch("render_read_pixels", {
            "source": "/definitely/does/not/exist.exr",
            "plane": "C",
            "mode": "summary",
            "roi": None,
            "max_pixels": 4096,
            "downsample": 1,
            "page": 0,
            "page_size": 1024,
        })
        assert isinstance(result, dict), (
            f"result must be a dict; got {type(result)!r}"
        )
        assert result.get("status") == "success", (
            f"Missing file must return status='success' (ok=False envelope); "
            f"got status={result.get('status')!r}; full: {result!r}"
        )
        data = result["data"]
        assert data.get("ok") is False, (
            f"Missing file: data['ok'] must be False; got {data.get('ok')!r}"
        )
        assert data.get("error"), (
            f"Missing file: data['error'] must be a non-empty string; got {data.get('error')!r}"
        )

    def test_in_scene_source_returns_error(self):
        """In-scene node path -> SUCCESS envelope with ok=False and 'in-scene' in error.

        Locked contract (pp12-114e lockedFieldContract §SOURCE-DETECT): when
        hou.node(expanded) is not None (i.e. the source resolves to a live Houdini
        node, not a file), the handler must return
        {ok:False, error:'in-scene plane source not yet supported (EXR-source v1)'}
        rather than raising or crashing.  '/obj' is always present in a fresh scene.
        """
        result = self._dispatch("render_read_pixels", {
            "source": "/obj",
            "plane": "C",
            "mode": "summary",
            "roi": None,
            "max_pixels": 4096,
            "downsample": 1,
            "page": 0,
            "page_size": 1024,
        })
        assert isinstance(result, dict), (
            f"result must be a dict; got {type(result)!r}"
        )
        assert result.get("status") == "success", (
            f"In-scene source must return status='success' (ok=False envelope); "
            f"got status={result.get('status')!r}; full: {result!r}"
        )
        data = result["data"]
        assert data.get("ok") is False, (
            f"In-scene source: data['ok'] must be False; got {data.get('ok')!r}"
        )
        assert data.get("error"), (
            f"In-scene source: data['error'] must be non-empty; got {data.get('error')!r}"
        )
        assert "in-scene" in str(data.get("error", "")).lower(), (
            f"In-scene source: error message must contain 'in-scene'; "
            f"got {data.get('error')!r}"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 64)
    print("render_read_pixels hython-smoke  (expected RED)")
    print("=" * 64)

    test_classes = [
        TestRenderReadPixelsSummaryShape,
        TestRenderReadPixelsSampleAndRoi,
        TestRenderReadPixelsFailLoud,
    ]

    for cls in test_classes:
        print(f"\n--- {cls.__name__} ---")
        try:
            _run_class(cls)
        except ImportError as exc:
            # _get_dispatcher() raised — expected RED state
            print(f"  [IMPORT ERROR (expected RED)] {cls.__name__}: {exc}")
            global _FAIL_COUNT
            _FAIL_COUNT += 1
            _ERRORS.append((cls.__name__, f"ImportError: {exc}"))
        except Exception as exc:
            print(f"  [SETUP ERROR] {cls.__name__}: {type(exc).__name__}: {exc}")
            _FAIL_COUNT += 1
            _ERRORS.append((cls.__name__, f"{type(exc).__name__}: {exc}"))

    print("\n" + "=" * 64)
    print(f"PASS: {_PASS_COUNT}   FAIL: {_FAIL_COUNT}")
    if _ERRORS:
        print("\nFailing tests:")
        for name, msg in _ERRORS:
            print(f"  FAIL {name}")
            if msg:
                first_line = msg.split("\n")[0]
                print(f"       {first_line}")
    print("=" * 64)

    # Return 1 (failure) so hython exit code is non-zero when RED.
    return 1 if _FAIL_COUNT > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
