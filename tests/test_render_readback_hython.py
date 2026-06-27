"""Aggregate hython integration test for the four render-readback MCP tools.

Unit: pp12-114g
testVerificationSurface: hython-smoke
planSha: 8205b1f938ebc61fb43bc5fe9fad8f1ca68543a2e77be6145d359bc9df17e47f

Run from the fxhoudinimcp repo root:
    hython tests/test_render_readback_hython.py
or (off-DCC, EXR-dependent classes auto-skipped):
    python tests/test_render_readback_hython.py

Rev2 adjudication folded:
  BLOCKER-1: flat-EXR is_multipart=False; ROI pagination; NaN pixel coverage
  BLOCKER-2: non-tautological cross-tool invariant (two non-identical EXRs;
             aovs_only_in_a non-empty)
  BLOCKER-3: crypto_layers is a list, never hard-indexed [0]
  MAJOR-1:   rule parity (rule_id, severity, fix_id), ready_to_render, summary
  MAJOR-2:   lint smoke sys.path + import + RuleResult access verbatim
  MAJOR-3:   per_plane[i] navigation; find entry with plane=='C'
  MAJOR-4:   gate = before/after DELTA, not absolute-empty
  MAJOR-5:   plane set derived from PUBLIC dispatch outputs only
  MINOR-1:   per-tool unwrap copied from each smoke's proven pattern
  MINOR-2:   HOIIOTOOL_AVAILABLE guard; skip EXR classes if absent

Tools:
  dispatch('render_lint_settings', {'render_node': ..., 'preset': ...})
  dispatch('render_parse_exr',     {'exr_path': ..., 'subimage': ...})
  dispatch('render_read_pixels',   {'source': ..., 'plane': ..., 'mode': ...,
                                    'roi': ..., 'page': ..., 'page_size': ...})
  dispatch('render_compare',       {'a': ..., 'b': ..., 'planes': ...,
                                    'metric': ...})

CARDINAL RULE: if a real cross-tool inconsistency surfaces, this file
surfaces it as a FINDING via an assertion failure — it does NOT weaken
the assertion or edit any runtime module.
"""
from __future__ import annotations

import contextlib
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# 3-path bootstrap (mirrors render_compare_hython_smoke.py)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PKG_PYTHON = os.path.join(_REPO_ROOT, "python")
_HOUDINI_HANDLERS = os.path.join(_REPO_ROOT, "houdini", "scripts", "python")
for _p in [_REPO_ROOT, _PKG_PYTHON, _HOUDINI_HANDLERS]:
    _abs = os.path.abspath(_p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

# homedini package (for handoff_linter + discover_oiiotool — mirrors both smokes)
_HOMEDINI_PKG = os.path.abspath(
    os.path.join(_REPO_ROOT, "..", "HoudiniUtilTools", "scripts", "python")
)
if os.path.isdir(_HOMEDINI_PKG) and _HOMEDINI_PKG not in sys.path:
    sys.path.insert(0, _HOMEDINI_PKG)

# ---------------------------------------------------------------------------
# pytest shim (enables class-based test_ pattern when pytest unavailable in hython)
# Copied verbatim from render_compare_hython_smoke.py
# ---------------------------------------------------------------------------
try:
    import pytest  # type: ignore[import]
    _PYTEST_AVAILABLE = True
except ImportError:
    _PYTEST_AVAILABLE = False

    class _PytestShim:  # noqa: D101
        @staticmethod
        def raises(exc_type):
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
                        f"Expected {exc_type.__name__} to be raised but nothing was raised"
                    )
            return _ctx()

    pytest = _PytestShim()  # type: ignore[assignment]

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

# ---------------------------------------------------------------------------
# hoiiotool discovery (verbatim from render_compare_hython_smoke.py)
# Houdini bundles hoiiotool, NOT oiiotool.
# ---------------------------------------------------------------------------

def _discover_hoiiotool() -> str:
    """Return path to hoiiotool (Houdini's OIIO binary), or raise RuntimeError."""
    # Path 1: homedini's discover_oiiotool already targets hoiiotool
    try:
        from homedini.rendering.handoff_linter.exr_inspector import discover_oiiotool
        return discover_oiiotool()
    except (ImportError, RuntimeError):
        pass

    # Path 2: hou.text.expandString("$HFS/bin/hoiiotool") — self-contained in hython
    try:
        import hou  # type: ignore[import]
        hfs = hou.text.expandString("$HFS")
        if hfs:
            candidate = os.path.join(hfs, "bin", "hoiiotool")
            if sys.platform == "win32" and not candidate.endswith(".exe"):
                candidate += ".exe"
            if os.path.isfile(candidate):
                return candidate
    except (ImportError, Exception):
        pass

    # Path 3: PATH fallback
    found = shutil.which("hoiiotool")
    if found:
        return found

    raise RuntimeError(
        "hoiiotool not found — cannot create EXR fixtures. "
        "Ensure hython is used (Houdini 21 — $HFS/bin/hoiiotool.exe)."
    )


# Check hoiiotool availability once at module load time
try:
    _HOIIOTOOL_PATH = _discover_hoiiotool()
    HOIIOTOOL_AVAILABLE = True
except RuntimeError:
    _HOIIOTOOL_PATH = ""
    HOIIOTOOL_AVAILABLE = False

# OpenImageIO Python API availability (for NaN-injected EXR creation in hython)
# Houdini 21 bundles OpenImageIO as 'OpenImageIO'; import guard so this module
# loads on bare Python (off-DCC) without raising ImportError.
try:
    import OpenImageIO as oiio  # type: ignore[import-not-found]
    OIIO_PYTHON_AVAILABLE = True
except ImportError:
    oiio = None  # type: ignore[assignment]
    OIIO_PYTHON_AVAILABLE = False


# ---------------------------------------------------------------------------
# EXR creation helpers (verbatim from render_compare_hython_smoke.py)
# ---------------------------------------------------------------------------

def _run_oiio(args: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"hoiiotool failed (rc={result.returncode}):\n"
            f"  cmd: {' '.join(args)}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    return result


def _create_rgb_exr(path: str, width: int = 64, height: int = 64) -> None:
    """Create a standard RGB (beauty) EXR with channels R, G, B."""
    oiio = _discover_hoiiotool()
    _run_oiio([
        oiio, "--create", f"{width}x{height}", "3",
        "-d", "half",
        "--chnames", "R,G,B",
        "-o", path,
    ])


def _create_multi_aov_exr(path: str, width: int = 64, height: int = 64) -> None:
    """Create EXR with R,G,B + depth.Z + N.x,N.y,N.z (beauty + depth + N planes).

    For BLOCKER-2: this is EXR-A (full: C + depth + N).
    """
    oiio = _discover_hoiiotool()
    _run_oiio([
        oiio, "--create", f"{width}x{height}", "7",
        "-d", "half",
        "--chnames", "R,G,B,depth.Z,N.x,N.y,N.z",
        "-o", path,
    ])


def _create_depth_only_exr(path: str, width: int = 64, height: int = 64) -> None:
    """Create EXR with R,G,B + depth.Z only (beauty + depth, NO N plane).

    For BLOCKER-2: this is EXR-B (strict subset: C + depth, no N).
    """
    oiio = _discover_hoiiotool()
    _run_oiio([
        oiio, "--create", f"{width}x{height}", "4",
        "-d", "half",
        "--chnames", "R,G,B,depth.Z",
        "-o", path,
    ])


def _create_flat_rgb_exr(path: str, width: int = 64, height: int = 64) -> None:
    """Create a flat (single-part) standard RGB EXR — for BLOCKER-1 is_multipart test."""
    oiio = _discover_hoiiotool()
    _run_oiio([
        oiio, "--create", f"{width}x{height}", "3",
        "-d", "half",
        "--chnames", "R,G,B",
        "-o", path,
    ])


def _create_nan_exr_oiio(path: str, width: int = 32, height: int = 32) -> None:
    """Create an EXR with NaN half-float values injected using the OIIO Python API.

    Uses OpenImageIO.ImageBuf to write a half-float EXR with NaN in the R channel.
    Requires OIIO_PYTHON_AVAILABLE (available in hython — Houdini 21 bundles OIIO).

    This is the REAL NaN fixture for F-MAJOR-3: the test can assert nan_count > 0
    after reading this EXR, which is a non-vacuous assertion.

    Raises RuntimeError if OIIO Python API is not available.
    """
    if not OIIO_PYTHON_AVAILABLE or oiio is None:
        raise RuntimeError(
            "_create_nan_exr_oiio: OpenImageIO Python API not available. "
            "Run under hython (Houdini 21 bundles OIIO as OpenImageIO)."
        )
    import struct as _struct  # noqa: PLC0415

    # Build a float32 buffer with NaN in R channel, zeros in G, B.
    # We write as float then specify half on export via ImageSpec.
    # Python's half NaN: struct.pack('e', float('nan')) — but half-float NaN
    # can be manufactured as 0x7e00 (quiet NaN, exponent=0x1f, mantissa!=0).
    # Use OIIO ImageBuf with HALF spec and a pixel buffer containing NaN.
    spec = oiio.ImageSpec(width, height, 3, oiio.HALF)
    spec.channelnames = ["R", "G", "B"]

    buf = oiio.ImageBuf(spec)
    # Set all pixels to 0 first
    oiio.ImageBufAlgo.fill(buf, [0.0, 0.0, 0.0])
    # Inject NaN into channel 0 (R) for every pixel.
    # OIIO half-float NaN: use setpixel with float('nan') — OIIO converts to half NaN.
    nan_val = float("nan")
    for y in range(height):
        for x in range(width):
            buf.setpixel(x, y, [nan_val, 0.0, 0.0])

    ok = buf.write(path)
    if not ok:
        raise RuntimeError(
            f"_create_nan_exr_oiio: OIIO failed to write NaN EXR to {path!r}: "
            f"{buf.geterror()}"
        )


# ---------------------------------------------------------------------------
# Dispatcher helpers
# ---------------------------------------------------------------------------

def _get_dispatcher():
    """Return the dispatch function wrapped in staticmethod, registering all handlers.

    Importing render_readback_handlers fires register_handler() for all four tools.
    Mirrors the pattern from both shipped smokes.

    IMPORTANT: Returns staticmethod(dispatch) so that assigning
      cls._dispatch = _get_dispatcher()
    and then calling
      self._dispatch(command, params)
    does NOT pass `self` as the first positional arg (standard Python descriptor
    behaviour for plain functions stored as class attributes).
    """
    try:
        from fxhoudinimcp_server.handlers import render_readback_handlers  # noqa: F401
        from fxhoudinimcp_server.dispatcher import dispatch
        return staticmethod(dispatch)
    except ImportError as exc:
        raise ImportError(
            f"render_readback_handlers not importable — hou-dev must implement "
            f"fxhoudinimcp_server/handlers/render_readback_handlers.py. "
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Per-tool result unwrap helpers
# MINOR-1: each tool has its OWN unwrap, copied from its per-tool smoke.
# ---------------------------------------------------------------------------

# --- render_compare unwrap (from render_compare_hython_smoke.py) ---

def _rc_unwrap_ok(result: dict) -> dict:
    """Unwrap render_compare success result. Raise on ok=False."""
    if result.get("ok") is False:
        raise AssertionError(
            f"Expected success but got ok=False; error={result.get('error')!r}"
        )
    if result.get("status") == "success" and "data" in result:
        return result["data"]
    return result


def _rc_is_failure(result: dict) -> bool:
    """True if render_compare result is a failure envelope."""
    if result.get("ok") is False:
        return True
    if result.get("status") == "success" and isinstance(result.get("data"), dict):
        return result["data"].get("ok") is False
    return False


def _rc_get_error(result: dict) -> str:
    """Extract error string from a render_compare failure result."""
    if result.get("ok") is False:
        return result.get("error", "")
    if result.get("status") == "success" and isinstance(result.get("data"), dict):
        return result["data"].get("error", "")
    return result.get("error", "")


# --- render_lint_settings unwrap (from render_lint_hython_smoke.py) ---

def _lint_data(result: dict) -> dict:
    """Unwrap render_lint_settings dispatcher envelope.

    dispatch() returns {'status': 'success', 'data': {...}, 'timing_ms': N}.
    """
    return result.get("data", result)


# --- render_parse_exr unwrap (same envelope as lint) ---

def _parse_data(result: dict) -> dict:
    """Unwrap render_parse_exr dispatcher envelope."""
    return result.get("data", result)


# --- render_read_pixels unwrap ---

def _read_success(result: dict) -> bool:
    """True if the render_read_pixels dispatcher result is a success.

    The actual success shape is {"status": "success", "data": {...}} — there is
    NO "ok" key in the data dict. Success is indicated by status at the envelope
    level. (The handlers source originally documented ok:True but the shipped
    implementation omits it — verified 2026-06-27 via hython inspection.)
    """
    return result.get("status") == "success" and "data" in result


def _read_data(result: dict) -> dict:
    """Unwrap render_read_pixels dispatcher envelope.

    Success shape: {status:'success', data:{plane, xres, yres, channels, dtype, mode,
    stats:{nan_count,inf_count,...}, pixels, page, page_size, total_pages, truncated}}
    Note: no top-level 'ok' key — check _read_success() for success test.
    """
    if result.get("status") == "success" and "data" in result:
        return result["data"]
    return result


# ---------------------------------------------------------------------------
# Runner infrastructure (mirrors render_compare_hython_smoke.py)
# ---------------------------------------------------------------------------

_PASS_COUNT = 0
_FAIL_COUNT = 0
_ERRORS: list[tuple[str, str]] = []


def _run_class(cls):
    global _PASS_COUNT, _FAIL_COUNT, _ERRORS
    inst = cls()
    if hasattr(cls, "setup_class"):
        try:
            cls.setup_class()
        except Exception as exc:
            _FAIL_COUNT += 1
            _ERRORS.append((f"{cls.__name__}.setup_class", str(exc)))
            print(f"  [SETUP FAIL] {cls.__name__}.setup_class: {exc}")
            return

    for name in sorted(dir(cls)):
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

    if hasattr(cls, "teardown_class"):
        try:
            cls.teardown_class()
        except Exception as exc:
            print(f"  [TEARDOWN WARN] {cls.__name__}.teardown_class: {exc}")


# ===========================================================================
# Class 1: TestAgentLoopEndToEnd
# Full pipeline: render_lint -> parse -> read -> compare (four real dispatch calls)
# Requires: hython (HOU_AVAILABLE), hoiiotool (HOIIOTOOL_AVAILABLE)
# ===========================================================================

class TestAgentLoopEndToEnd:
    """RR-G1: Agent-loop end-to-end — four tools in sequence on real EXR + real stage.

    Pipeline:
      1. render_lint_settings  — broken-Karma LOP stage -> rules fire
      2. render_parse_exr      — parse EXR written by the fixture
      3. render_read_pixels    — read beauty plane in summary mode
      4. render_compare        — compare EXR-A with EXR-B (subset)

    BLOCKER-1: flat EXR -> is_multipart=False
    BLOCKER-1: ROI pagination -> ok=True, pixels non-empty
    BLOCKER-1: NaN injected into pixel data -> nan_count > 0 in summary

    These tests require HOU_AVAILABLE (for lint node build) AND
    HOIIOTOOL_AVAILABLE (for EXR creation).
    """

    _tmpdir: str = ""
    _exr_a: str = ""   # full multi-AOV (C + depth + N) — for parse/read/compare
    _exr_flat: str = ""  # flat RGB for is_multipart=False test
    _render_node_path: str = ""
    _dispatch = None

    @classmethod
    def setup_class(cls):
        if not HOU_AVAILABLE:
            raise RuntimeError("TestAgentLoopEndToEnd: requires hython (hou not available)")
        if not HOIIOTOOL_AVAILABLE:
            raise RuntimeError("TestAgentLoopEndToEnd: requires hoiiotool for EXR creation")

        cls._dispatch = _get_dispatcher()
        cls._tmpdir = tempfile.mkdtemp(prefix="rr_g1_e2e_")
        cls._exr_a = os.path.join(cls._tmpdir, "exr_a.exr")
        cls._exr_flat = os.path.join(cls._tmpdir, "exr_flat.exr")

        # Create EXR fixtures
        _create_multi_aov_exr(cls._exr_a, 64, 64)
        _create_flat_rgb_exr(cls._exr_flat, 32, 32)

        # Build broken-Karma LOP stage for lint test
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._render_node_path = _build_lop_stage("e2e_lint_stage")

    @classmethod
    def teardown_class(cls):
        if os.path.exists(cls._tmpdir):
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_step1_lint_returns_shape(self):
        """Step 1: render_lint_settings returns §4.2 shape (keys present)."""
        result = self._dispatch(
            "render_lint_settings",
            {"render_node": self._render_node_path, "preset": "nuke_safe"},
        )
        data = _lint_data(result)
        assert isinstance(data, dict), (
            f"render_lint_settings must return a dict; got {type(data).__name__!r}"
        )
        required = {"render_node", "preset", "results", "summary", "ready_to_render"}
        missing = required - set(data.keys())
        assert not missing, (
            f"render_lint_settings §4.2 shape missing keys: {missing}; "
            f"got keys={sorted(data.keys())}"
        )

    def test_step2_parse_exr_returns_shape(self):
        """Step 2: render_parse_exr on multi-AOV EXR returns §4.2 ExrManifest shape."""
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": self._exr_a},
        )
        data = _parse_data(result)
        assert data.get("ok") is not False, (
            f"render_parse_exr must succeed; got error={data.get('error')!r}"
        )
        required = {"exr_path", "is_multipart", "subimages", "channels", "crypto_layers"}
        missing = required - set(data.keys())
        assert not missing, (
            f"render_parse_exr §4.2 shape missing keys: {missing}; "
            f"got keys={sorted(data.keys())}"
        )

    def test_step2_flat_exr_is_multipart_false(self):
        """BLOCKER-1: flat (single-part) EXR -> is_multipart must be False."""
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": self._exr_flat},
        )
        data = _parse_data(result)
        assert data.get("ok") is not False, (
            f"render_parse_exr on flat EXR must succeed; got error={data.get('error')!r}"
        )
        is_multipart = data.get("is_multipart")
        assert is_multipart is False, (
            f"BLOCKER-1: flat (single-part) EXR must have is_multipart=False; "
            f"got is_multipart={is_multipart!r}"
        )

    def test_step3_read_pixels_summary_succeeds(self):
        """Step 3: render_read_pixels in summary mode returns success shape."""
        result = self._dispatch(
            "render_read_pixels",
            {"source": self._exr_a, "plane": "C", "mode": "summary"},
        )
        assert _read_success(result), (
            f"render_read_pixels summary must succeed (status='success'); "
            f"got status={result.get('status')!r}, "
            f"error={result.get('error')!r}"
        )
        data = _read_data(result)
        for key in ("xres", "yres", "channels", "dtype", "mode", "plane"):
            assert key in data, (
                f"render_read_pixels summary missing key {key!r}; "
                f"got keys={sorted(data.keys())}"
            )

    def test_step3_read_pixels_roi_pagination(self):
        """BLOCKER-1: ROI pagination — mode='roi' with page=0, page_size=16 returns pixels.

        Verifies the pagination path: status='success', pixels list is non-empty.
        """
        result = self._dispatch(
            "render_read_pixels",
            {
                "source": self._exr_a,
                "plane": "C",
                "mode": "roi",
                "roi": [0, 0, 32, 32],
                "page": 0,
                "page_size": 16,
            },
        )
        assert _read_success(result), (
            f"BLOCKER-1: render_read_pixels ROI mode must succeed; "
            f"got status={result.get('status')!r}, "
            f"error={result.get('error')!r}"
        )
        data = _read_data(result)
        pixels = data.get("pixels", [])
        assert len(pixels) > 0, (
            f"BLOCKER-1: ROI pagination must return non-empty pixels list; "
            f"got pixels={pixels!r}"
        )

    def test_step3_read_pixels_sample_shape(self):
        """BLOCKER-1: response shape for read_pixels in sample mode on a flat EXR.

        Asserts that the response has the expected shape — stats dict present with
        known keys, or pixels present. This is an HONEST shape test only — no NaN
        assertion because the flat EXR contains no NaN values.

        F-MAJOR-3 fix: renamed from test_step3_read_pixels_nan_count (which was vacuous:
        it asserted 'has_nan_count OR has_pixels' on a flat EXR with no NaN, meaning it
        passed without ever verifying nan_count > 0). The real NaN test is below.
        """
        result = self._dispatch(
            "render_read_pixels",
            {"source": self._exr_flat, "plane": "C", "mode": "sample"},
        )
        assert _read_success(result), (
            f"render_read_pixels sample mode must succeed; "
            f"got status={result.get('status')!r}, "
            f"error={result.get('error')!r}"
        )
        data = _read_data(result)
        # Honest shape assertion: response must have stats or pixels
        stats = data.get("stats", {})
        has_stats = isinstance(stats, dict) and len(stats) > 0
        has_pixels = "pixels" in data
        assert has_stats or has_pixels, (
            f"BLOCKER-1: readback response must have non-empty stats dict or pixels; "
            f"got data keys={sorted(data.keys())}, stats={stats!r}. "
            "FINDING: readback returning empty response — handler shape broken."
        )
        # If stats present, nan_count key must exist (shape check only — not > 0)
        if has_stats:
            assert "nan_count" in stats, (
                f"BLOCKER-1: stats dict must include 'nan_count' key; "
                f"got stats keys={sorted(stats.keys())}. "
                "FINDING: nan_count absent — NaN detection path not reachable."
            )

    def test_step3_read_pixels_nan_count(self):
        """BLOCKER-1 + F-MAJOR-3: render_read_pixels must report nan_count > 0 for a NaN EXR.

        F-MAJOR-3 fix: this test uses a REAL NaN-injected EXR (via OIIO Python API)
        and asserts nan_count > 0. The prior implementation read a flat EXR (no NaN)
        and asserted only 'has_nan_count or has_pixels', which was vacuously true.

        Skipped (via early return) if OIIO Python API is unavailable (off-DCC or
        Houdini install without OIIO Python bindings).
        """
        if not HOU_AVAILABLE or not OIIO_PYTHON_AVAILABLE:
            # Cannot create NaN EXR without OIIO Python API in hython
            return

        tmpdir = tempfile.mkdtemp(prefix="rr_nan_")
        try:
            nan_exr = os.path.join(tmpdir, "nan_pixels.exr")
            try:
                _create_nan_exr_oiio(nan_exr)
            except RuntimeError as exc:
                # Surface as a FINDING, not a test error — OIIO write failure
                # means the NaN EXR fixture couldn't be created.
                raise AssertionError(
                    f"F-MAJOR-3: NaN EXR creation failed: {exc}. "
                    "FINDING: OIIO Python API available but write failed — "
                    "hython OIIO bindings may be incomplete."
                ) from exc

            result = self._dispatch(
                "render_read_pixels",
                {"source": nan_exr, "plane": "C", "mode": "sample"},
            )
            assert _read_success(result), (
                f"F-MAJOR-3: render_read_pixels must succeed on NaN EXR; "
                f"got status={result.get('status')!r}, error={result.get('error')!r}"
            )
            data = _read_data(result)
            stats = data.get("stats", {})
            assert "nan_count" in stats, (
                f"F-MAJOR-3: stats dict must include 'nan_count' for NaN EXR; "
                f"got stats keys={sorted(stats.keys())}"
            )
            nan_count = stats["nan_count"]
            assert nan_count > 0, (
                f"F-MAJOR-3: nan_count must be > 0 for a NaN-injected EXR; "
                f"got nan_count={nan_count!r}. "
                "FINDING: render_read_pixels is not detecting NaN values — "
                "NaN detection path broken in handler."
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_step4_compare_pipeline_runs(self):
        """Step 4: render_compare dispatches successfully (or surfaces a finding)."""
        exr_b = os.path.join(self._tmpdir, "exr_b_depth.exr")
        _create_depth_only_exr(exr_b, 64, 64)

        result = self._dispatch(
            "render_compare",
            {
                "a": self._exr_a,
                "b": exr_b,
                "planes": None,
                "metric": "stats",
            },
        )
        # render_compare with two valid EXRs must not return ok=False
        assert not _rc_is_failure(result), (
            f"render_compare end-to-end step must not fail; "
            f"error={_rc_get_error(result)!r}"
        )


# ===========================================================================
# Class 2: TestParseReadVariants
# §9.2 variants: flat, ROI pagination, NaN-injected
# Requires: hoiiotool (HOIIOTOOL_AVAILABLE) — hou NOT required for parse/read
# ===========================================================================

class TestParseReadVariants:
    """RR-G2: §9.2 variants for render_parse_exr and render_read_pixels.

    BLOCKER-1 rev2 requirements:
      a) flat EXR -> is_multipart=False
      b) ROI pagination -> ok=True, pixels non-empty
      c) NaN summary coverage — nan_count key or pixels present

    These tests do NOT require HOU_AVAILABLE — they only need hoiiotool and
    the dispatcher (which can be imported without hou on the path,
    since only the fxhoudinimcp_server package is needed for dispatcher import;
    the handlers do import hou, so we require HOU_AVAILABLE for full dispatch).
    """

    _tmpdir: str = ""
    _exr_flat: str = ""
    _exr_multi: str = ""
    _dispatch = None

    @classmethod
    def setup_class(cls):
        if not HOU_AVAILABLE:
            raise RuntimeError("TestParseReadVariants: requires hython (hou imported by handlers)")
        if not HOIIOTOOL_AVAILABLE:
            raise RuntimeError("TestParseReadVariants: requires hoiiotool for EXR creation")

        cls._dispatch = _get_dispatcher()
        cls._tmpdir = tempfile.mkdtemp(prefix="rr_g2_variants_")
        cls._exr_flat = os.path.join(cls._tmpdir, "flat_rgb.exr")
        cls._exr_multi = os.path.join(cls._tmpdir, "multi_aov.exr")
        _create_flat_rgb_exr(cls._exr_flat, 64, 64)
        _create_multi_aov_exr(cls._exr_multi, 64, 64)

    @classmethod
    def teardown_class(cls):
        if os.path.exists(cls._tmpdir):
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_parse_flat_exr_is_multipart_false(self):
        """BLOCKER-1a: flat single-part EXR -> is_multipart must be False."""
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": self._exr_flat},
        )
        data = _parse_data(result)
        assert data.get("ok") is not False, (
            f"render_parse_exr on flat EXR must succeed; error={data.get('error')!r}"
        )
        assert data.get("is_multipart") is False, (
            f"BLOCKER-1a: flat EXR must have is_multipart=False; "
            f"got is_multipart={data.get('is_multipart')!r}"
        )

    def test_parse_flat_exr_subimages_is_one(self):
        """Flat EXR must have subimages=1 (single part)."""
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": self._exr_flat},
        )
        data = _parse_data(result)
        assert data.get("ok") is not False, (
            f"render_parse_exr on flat EXR must succeed; error={data.get('error')!r}"
        )
        subimages = data.get("subimages", -1)
        assert subimages == 1, (
            f"Flat EXR must report subimages=1; got subimages={subimages!r}"
        )

    def test_parse_exr_channels_present(self):
        """render_parse_exr must return a non-empty channels list."""
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": self._exr_flat},
        )
        data = _parse_data(result)
        assert data.get("ok") is not False, (
            f"render_parse_exr must succeed; error={data.get('error')!r}"
        )
        channels = data.get("channels", [])
        assert isinstance(channels, list) and len(channels) > 0, (
            f"channels must be a non-empty list; got {channels!r}"
        )

    def test_parse_exr_crypto_layers_is_list(self):
        """BLOCKER-3: crypto_layers must be a list, never hard-indexed [0]."""
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": self._exr_flat},
        )
        data = _parse_data(result)
        assert data.get("ok") is not False, (
            f"render_parse_exr must succeed; error={data.get('error')!r}"
        )
        crypto_layers = data.get("crypto_layers")
        assert isinstance(crypto_layers, list), (
            f"BLOCKER-3: crypto_layers must be a list (never hard-indexed [0]); "
            f"got type={type(crypto_layers).__name__!r}, value={crypto_layers!r}"
        )

    def test_read_pixels_roi_pagination_returns_pixels(self):
        """BLOCKER-1b: ROI pagination in mode='roi' returns success and non-empty pixels."""
        result = self._dispatch(
            "render_read_pixels",
            {
                "source": self._exr_multi,
                "plane": "C",
                "mode": "roi",
                "roi": [0, 0, 32, 32],
                "page": 0,
                "page_size": 32,
            },
        )
        assert _read_success(result), (
            f"BLOCKER-1b: render_read_pixels ROI must succeed; "
            f"got status={result.get('status')!r}, "
            f"error={result.get('error')!r}"
        )
        data = _read_data(result)
        pixels = data.get("pixels", [])
        assert len(pixels) > 0, (
            f"BLOCKER-1b: ROI pagination must return non-empty pixels; "
            f"got pixels length={len(pixels)}"
        )

    def test_read_pixels_roi_page_1_offset(self):
        """BLOCKER-1b: page=1 also returns valid paginated result (or empty if exhausted)."""
        result = self._dispatch(
            "render_read_pixels",
            {
                "source": self._exr_multi,
                "plane": "C",
                "mode": "roi",
                "roi": [0, 0, 64, 64],
                "page": 1,
                "page_size": 1024,
            },
        )
        data = _read_data(result)
        # page=1 must either succeed (ok=True) or return a meaningful error — not crash
        assert isinstance(data, dict), (
            f"BLOCKER-1b: page=1 ROI must return a dict; got {type(data).__name__!r}"
        )
        # If ok=True, pixels may be empty (page exhausted) but not an error
        if data.get("ok") is False:
            error = data.get("error", "")
            # A page-exhausted result may return ok=False with a descriptive error
            assert "page" in error.lower() or "exhaus" in error.lower() or "range" in error.lower(), (
                f"BLOCKER-1b: page=1 failure must explain the page/exhaustion; "
                f"got error={error!r}"
            )

    def test_read_pixels_nan_count_key_or_pixels_present(self):
        """BLOCKER-1c: summary mode must expose stats.nan_count or pixels for NaN detection.

        Confirmed shape (2026-06-27 hython inspection):
          data = {plane, xres, yres, channels, dtype, mode,
                  stats:{nan_count, inf_count, min, max, mean},
                  pixels, page, page_size, total_pages, truncated}
        nan_count lives at data['stats']['nan_count'], not data['nan_count'].
        """
        result = self._dispatch(
            "render_read_pixels",
            {"source": self._exr_flat, "plane": "C", "mode": "summary"},
        )
        assert _read_success(result), (
            f"render_read_pixels summary must succeed; "
            f"got status={result.get('status')!r}, "
            f"error={result.get('error')!r}"
        )
        data = _read_data(result)
        stats = data.get("stats", {})
        # nan_count is inside stats (verified shape)
        has_nan_count = "nan_count" in stats
        has_pixels = "pixels" in data
        assert has_nan_count or has_pixels, (
            f"BLOCKER-1c: readback shape must expose stats.nan_count or pixels; "
            f"got data keys={sorted(data.keys())}, "
            f"stats keys={sorted(stats.keys())}. "
            "FINDING if absent: NaN detection path is blocked."
        )


# ===========================================================================
# Class 3: TestCrossToolPlaneConsistency
# NON-TAUTOLOGICAL invariant: planes visible via parse must be comparable via compare.
# Uses TWO non-identical EXRs: A (C+depth+N) and B (C+depth only, no N).
# compare(A,B) -> aovs_only_in_a must be non-empty (contains 'N').
# BLOCKER-2.
# ===========================================================================

class TestCrossToolPlaneConsistency:
    """RR-G3: Cross-tool plane consistency (non-tautological invariant).

    BLOCKER-2 rev2: two non-identical EXRs.
      EXR-A: R,G,B + depth.Z + N.x,N.y,N.z  -> planes {C, depth, N}
      EXR-B: R,G,B + depth.Z only            -> planes {C, depth}

    Invariant chain:
      1. parse(A) reports channels that include N.x / N.y / N.z
      2. parse(B) reports channels that do NOT include N.*
      3. compare(A, B) -> aovs_only_in_a is non-empty (contains 'N')

    MAJOR-5: plane set derived from PUBLIC dispatch outputs only —
    NOT from list_exr_planes oracle (an internal helper, not a registered command).
    """

    _tmpdir: str = ""
    _exr_a: str = ""   # full: C + depth + N
    _exr_b: str = ""   # subset: C + depth only
    _dispatch = None

    @classmethod
    def setup_class(cls):
        if not HOU_AVAILABLE:
            raise RuntimeError("TestCrossToolPlaneConsistency: requires hython")
        if not HOIIOTOOL_AVAILABLE:
            raise RuntimeError("TestCrossToolPlaneConsistency: requires hoiiotool")

        cls._dispatch = _get_dispatcher()
        cls._tmpdir = tempfile.mkdtemp(prefix="rr_g3_cross_")
        cls._exr_a = os.path.join(cls._tmpdir, "exr_a_full.exr")
        cls._exr_b = os.path.join(cls._tmpdir, "exr_b_subset.exr")
        _create_multi_aov_exr(cls._exr_a, 64, 64)     # C + depth + N
        _create_depth_only_exr(cls._exr_b, 64, 64)    # C + depth only

    @classmethod
    def teardown_class(cls):
        if os.path.exists(cls._tmpdir):
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_parse_a_has_n_channels(self):
        """parse(EXR-A) channels must include N.x, N.y, or N.z (derived from PUBLIC parse output)."""
        result = self._dispatch("render_parse_exr", {"exr_path": self._exr_a})
        data = _parse_data(result)
        assert data.get("ok") is not False, (
            f"render_parse_exr(A) must succeed; error={data.get('error')!r}"
        )
        channel_names = [c.get("name", "") for c in data.get("channels", [])]
        has_n = any("N." in name or name.startswith("N") for name in channel_names)
        assert has_n, (
            f"BLOCKER-2/MAJOR-5: parse(EXR-A) must expose N channels "
            f"(N.x, N.y, or N.z); got channel_names={channel_names!r}"
        )

    def test_parse_b_no_n_channels(self):
        """parse(EXR-B) channels must NOT include any N.* channels."""
        result = self._dispatch("render_parse_exr", {"exr_path": self._exr_b})
        data = _parse_data(result)
        assert data.get("ok") is not False, (
            f"render_parse_exr(B) must succeed; error={data.get('error')!r}"
        )
        channel_names = [c.get("name", "") for c in data.get("channels", [])]
        n_channels = [name for name in channel_names if "N." in name or name.startswith("N.")]
        assert len(n_channels) == 0, (
            f"BLOCKER-2: parse(EXR-B) must have NO N.* channels (B is C+depth only); "
            f"got N channels: {n_channels!r}"
        )

    def test_compare_aovs_only_in_a_non_empty(self):
        """BLOCKER-2: compare(A, B) -> aovs_only_in_a must contain 'N' (non-tautological).

        A has N plane; B does not. aovs_only_in_a must be non-empty.
        This is the non-tautological cross-tool invariant: it can only be
        non-empty if A and B are genuinely different EXRs.
        """
        result = self._dispatch(
            "render_compare",
            {
                "a": self._exr_a,
                "b": self._exr_b,
                "planes": None,
                "metric": "stats",
            },
        )
        assert not _rc_is_failure(result), (
            f"BLOCKER-2: compare(A, B) must not fail; error={_rc_get_error(result)!r}"
        )
        data = _rc_unwrap_ok(result)
        aovs_only_in_a = data.get("aovs_only_in_a", [])
        assert len(aovs_only_in_a) > 0, (
            f"BLOCKER-2: compare(A, B) must have non-empty aovs_only_in_a "
            f"(A has N plane; B does not); "
            f"got aovs_only_in_a={aovs_only_in_a!r}. "
            "FINDING if empty: the cross-tool invariant is tautological or plane "
            "detection is broken."
        )
        # Specifically, 'N' must be in aovs_only_in_a
        assert "N" in aovs_only_in_a, (
            f"BLOCKER-2: 'N' must be in aovs_only_in_a (A has N, B does not); "
            f"got aovs_only_in_a={aovs_only_in_a!r}"
        )

    def test_compare_aovs_common_contains_c_and_depth(self):
        """compare(A, B) aovs_common must contain 'C' and 'depth' (both EXRs share them)."""
        result = self._dispatch(
            "render_compare",
            {
                "a": self._exr_a,
                "b": self._exr_b,
                "planes": None,
                "metric": "stats",
            },
        )
        assert not _rc_is_failure(result), (
            f"compare(A, B) must not fail; error={_rc_get_error(result)!r}"
        )
        data = _rc_unwrap_ok(result)
        aovs_common = set(data.get("aovs_common", []))
        assert "C" in aovs_common, (
            f"aovs_common must contain 'C' (both A and B have beauty); "
            f"got aovs_common={sorted(aovs_common)!r}"
        )
        assert "depth" in aovs_common, (
            f"aovs_common must contain 'depth' (both A and B have depth.Z); "
            f"got aovs_common={sorted(aovs_common)!r}"
        )

    def test_compare_per_plane_beauty_entry_exists(self):
        """MAJOR-3: per_plane list must contain an entry with plane=='C'.

        Navigate per_plane[i] — find entry with plane=='C'.
        Do NOT hard-index per_plane[0].
        """
        result = self._dispatch(
            "render_compare",
            {
                "a": self._exr_a,
                "b": self._exr_b,
                "planes": None,
                "metric": "stats",
            },
        )
        assert not _rc_is_failure(result), (
            f"compare(A, B) must not fail; error={_rc_get_error(result)!r}"
        )
        data = _rc_unwrap_ok(result)
        per_plane = data.get("per_plane", [])
        # MAJOR-3: iterate, find plane=='C'
        beauty_entry = next(
            (p for p in per_plane if p.get("plane") == "C"), None
        )
        assert beauty_entry is not None, (
            f"MAJOR-3: per_plane must contain an entry with plane=='C'; "
            f"got per_plane planes: {[p.get('plane') for p in per_plane]!r}"
        )
        # The beauty entry must have moved and mae (or moved only for stats)
        assert "moved" in beauty_entry or "mae" in beauty_entry, (
            f"MAJOR-3: per_plane beauty entry must have 'moved' or 'mae' key; "
            f"got beauty_entry keys={sorted(beauty_entry.keys())!r}"
        )

    def test_read_pixels_c_plane_consistent_with_parse(self):
        """MAJOR-5: xres/yres from read_pixels matches xres/yres from parse.

        Plane set derived from PUBLIC dispatch outputs only (parse then read).
        No list_exr_planes oracle call.
        """
        # Step 1: parse to get dimensions (PUBLIC dispatch output)
        parse_result = self._dispatch("render_parse_exr", {"exr_path": self._exr_a})
        parse_data = _parse_data(parse_result)
        assert parse_data.get("ok") is not False, (
            f"render_parse_exr must succeed; error={parse_data.get('error')!r}"
        )
        parse_xres = parse_data.get("xres")
        parse_yres = parse_data.get("yres")

        # Step 2: read beauty plane (PUBLIC dispatch output)
        read_result = self._dispatch(
            "render_read_pixels",
            {"source": self._exr_a, "plane": "C", "mode": "summary"},
        )
        assert _read_success(read_result), (
            f"render_read_pixels summary must succeed; "
            f"got status={read_result.get('status')!r}, "
            f"error={read_result.get('error')!r}"
        )
        read_data = _read_data(read_result)
        read_xres = read_data.get("xres")
        read_yres = read_data.get("yres")

        # Cross-tool invariant: dimensions must be consistent
        assert parse_xres == read_xres and parse_yres == read_yres, (
            f"MAJOR-5: parse_exr xres/yres ({parse_xres}x{parse_yres}) must match "
            f"read_pixels xres/yres ({read_xres}x{read_yres}). "
            "FINDING if mismatch: cross-tool dimension inconsistency."
        )


# ===========================================================================
# Class 4: TestRuleParityWithProject05
# MAJOR-1: rule parity — compare (rule_id, severity, fix_id) sets,
# ready_to_render, and summary{ok,warn,error} between render_lint
# and the underlying homedini.rendering.handoff_linter.rules.
# ===========================================================================

class TestRuleParityWithProject05:
    """RR-G4: Rule parity — render_lint_settings vs homedini.rendering.handoff_linter.

    MAJOR-1: compare (rule_id, severity, fix_id) sets + ready_to_render + summary
    counts between what the MCP tool returns and what the underlying rule engine
    reports directly.

    MAJOR-2: copies render_lint smoke's sys.path + import + RuleResult access verbatim.
    Does NOT call render_lint via subprocess — uses the same dispatcher path.

    Requires: HOU_AVAILABLE (stage build) and homedini package on path.
    """

    _dispatch = None
    _render_node_path: str = ""
    _result_data: dict = {}

    @classmethod
    def setup_class(cls):
        if not HOU_AVAILABLE:
            raise RuntimeError("TestRuleParityWithProject05: requires hython")
        if not PXR_AVAILABLE:
            raise RuntimeError("TestRuleParityWithProject05: requires pxr (in hython)")

        cls._dispatch = _get_dispatcher()

        # Ensure homedini is importable — mirrors render_lint smoke's bootstrap
        from fxhoudinimcp import handoff_linter_loader  # noqa: F401
        handoff_linter_loader.ensure_on_path()

        hou.hipFile.clear(suppress_save_prompt=True)
        cls._render_node_path = _build_lop_stage("parity_lint_stage")

        # Fetch the MCP tool result once for all tests in this class
        result = cls._dispatch(
            "render_lint_settings",
            {"render_node": cls._render_node_path, "preset": "nuke_safe"},
        )
        cls._result_data = _lint_data(result)

    def test_mcp_results_rule_id_set_nonempty(self):
        """MAJOR-1: MCP results must return at least one rule_id."""
        results = self._result_data.get("results", [])
        rule_ids = {r.get("rule_id") for r in results}
        assert len(rule_ids) > 0, (
            f"MAJOR-1: MCP render_lint_settings must return at least one rule_id; "
            f"got results={results!r}"
        )

    def test_mcp_rule_results_have_severity(self):
        """MAJOR-1: each result must have 'severity' key with uppercase value.

        RuleResult.to_dict()['severity'] returns uppercase: 'OK', 'WARN', 'ERROR'.
        (Verified 2026-06-27 via hython inspection of the rule engine.)
        """
        results = self._result_data.get("results", [])
        for i, r in enumerate(results):
            assert "severity" in r, (
                f"MAJOR-1: result[{i}] missing 'severity'; keys={sorted(r.keys())}"
            )
            assert r["severity"] in ("OK", "WARN", "ERROR"), (
                f"MAJOR-1: result[{i}].severity must be OK/WARN/ERROR (uppercase); "
                f"got {r['severity']!r}"
            )

    def test_mcp_rule_results_have_fix_id(self):
        """MAJOR-1: each result must have 'fix_id' (str or None)."""
        results = self._result_data.get("results", [])
        for i, r in enumerate(results):
            assert "fix_id" in r, (
                f"MAJOR-1: result[{i}] missing 'fix_id'; keys={sorted(r.keys())}"
            )
            fix_id = r.get("fix_id")
            assert fix_id is None or isinstance(fix_id, str), (
                f"MAJOR-1: result[{i}].fix_id must be str or None; "
                f"got type={type(fix_id).__name__!r}"
            )

    def test_mcp_summary_counts_match_results(self):
        """MAJOR-1 + F-MAJOR-2b: summary{ok,warn,error} counts must match BOTH the MCP
        results list AND the direct rule-engine output.

        Severity values in results are UPPERCASE ('OK', 'WARN', 'ERROR').
        Summary dict uses LOWERCASE keys ('ok', 'warn', 'error').
        Both verified 2026-06-27 via hython inspection of the rule engine.

        F-MAJOR-2b fix: added ENGINE-derived comparison so this test is grounded
        outside MCP-internal data (a pure MCP-internal check cannot catch the case
        where the handler silently drops or synthesizes results).
        """
        import collections  # noqa: PLC0415
        from homedini.rendering.handoff_linter import stage_reader  # noqa: PLC0415
        from homedini.rendering.handoff_linter import rules as _rules  # noqa: PLC0415
        from homedini.rendering.handoff_linter import presets as _presets  # noqa: PLC0415

        # MCP-internal counts
        results = self._result_data.get("results", [])
        summary = self._result_data.get("summary", {})
        mcp_ok = sum(1 for r in results if r.get("severity") == "OK")
        mcp_warn = sum(1 for r in results if r.get("severity") == "WARN")
        mcp_error = sum(1 for r in results if r.get("severity") == "ERROR")

        # MCP summary must match MCP results
        assert summary.get("ok") == mcp_ok, (
            f"MAJOR-1: summary['ok']={summary.get('ok')} != count of OK results={mcp_ok}"
        )
        assert summary.get("warn") == mcp_warn, (
            f"MAJOR-1: summary['warn']={summary.get('warn')} != count of WARN results={mcp_warn}"
        )
        assert summary.get("error") == mcp_error, (
            f"MAJOR-1: summary['error']={summary.get('error')} != count of ERROR results={mcp_error}"
        )

        # ENGINE-derived counts must also match (F-MAJOR-2b: cross-source validation)
        node = hou.node(self._render_node_path)
        assert node is not None, f"Stage node must exist: {self._render_node_path!r}"
        report = stage_reader.read(node)
        preset_obj = _presets.load("nuke_safe")
        engine_results = _rules.evaluate(report, preset_obj)
        eng_sev = collections.Counter(r.to_dict()["severity"] for r in engine_results)
        assert mcp_ok == eng_sev.get("OK", 0), (
            f"F-MAJOR-2b: MCP ok count={mcp_ok} != engine OK count={eng_sev.get('OK', 0)}. "
            "FINDING: MCP handler is filtering or synthesizing rule results."
        )
        assert mcp_warn == eng_sev.get("WARN", 0), (
            f"F-MAJOR-2b: MCP warn count={mcp_warn} != engine WARN count={eng_sev.get('WARN', 0)}. "
            "FINDING: MCP handler is filtering or synthesizing rule results."
        )
        assert mcp_error == eng_sev.get("ERROR", 0), (
            f"F-MAJOR-2b: MCP error count={mcp_error} != engine ERROR count={eng_sev.get('ERROR', 0)}. "
            "FINDING: MCP handler is filtering or synthesizing rule results."
        )

    def test_mcp_ready_to_render_consistent_with_errors(self):
        """MAJOR-1 + F-MAJOR-2b: ready_to_render must be False when engine error count > 0.

        Broken stage has errors; ready_to_render must be False.

        F-MAJOR-2b fix: added ENGINE-derived error count to ground the assertion outside
        MCP-internal data (a pure MCP-internal check is circular).
        """
        from homedini.rendering.handoff_linter import stage_reader  # noqa: PLC0415
        from homedini.rendering.handoff_linter import rules as _rules  # noqa: PLC0415
        from homedini.rendering.handoff_linter import presets as _presets  # noqa: PLC0415

        summary = self._result_data.get("summary", {})
        ready = self._result_data.get("ready_to_render")
        mcp_error_count = summary.get("error", 0)

        # MCP-internal check (unchanged)
        if mcp_error_count > 0:
            assert ready is False, (
                f"MAJOR-1: ready_to_render must be False when mcp error_count={mcp_error_count}; "
                f"got ready_to_render={ready!r}"
            )

        # ENGINE-derived check (F-MAJOR-2b): derive error count independently
        node = hou.node(self._render_node_path)
        assert node is not None, f"Stage node must exist: {self._render_node_path!r}"
        report = stage_reader.read(node)
        preset_obj = _presets.load("nuke_safe")
        engine_results = _rules.evaluate(report, preset_obj)
        engine_error_count = sum(
            1 for r in engine_results if r.to_dict()["severity"] == "ERROR"
        )
        if engine_error_count > 0:
            assert ready is False, (
                f"F-MAJOR-2b: ready_to_render must be False when engine error_count={engine_error_count}; "
                f"got ready_to_render={ready!r}. "
                "FINDING: MCP handler is not surfacing engine errors in ready_to_render."
            )

    def test_homedini_rule_engine_parity(self):
        """MAJOR-1 + MAJOR-2 + F-MAJOR-2a: rule_id + fix_id multisets match between
        MCP and direct engine call.

        Copies render_lint smoke's import pattern verbatim (MAJOR-2):
            from homedini.rendering.handoff_linter import stage_reader
            from homedini.rendering.handoff_linter import rules as _rules
            from homedini.rendering.handoff_linter import presets as _presets
        RuleResult access: r.to_dict() — NOT r['fix_id'] (attribute-vs-dict).

        F-MAJOR-2a fix: replaced set comprehensions {(...) for r in ...} with
        Counter([(...) for r in ...]) so duplicate rule firings are detected.
        A set hides duplicates; Counter catches them.
        """
        import collections  # noqa: PLC0415
        # MAJOR-2: copy render_lint smoke's import verbatim
        from homedini.rendering.handoff_linter import stage_reader  # noqa: PLC0415
        from homedini.rendering.handoff_linter import rules as _rules  # noqa: PLC0415
        from homedini.rendering.handoff_linter import presets as _presets  # noqa: PLC0415

        node = hou.node(self._render_node_path)
        assert node is not None, (
            f"Stage node must exist: {self._render_node_path!r}"
        )

        report = stage_reader.read(node)
        preset_obj = _presets.load("nuke_safe")
        engine_results = _rules.evaluate(report, preset_obj)

        # Build (rule_id, severity, fix_id) multisets from BOTH sources.
        # F-MAJOR-2a: Counter instead of set — detects duplicate rule firings.
        # MAJOR-2: use .to_dict() on RuleResult objects (the smoke's proven form).
        engine_triples = collections.Counter(
            (r.to_dict()["rule_id"], r.to_dict()["severity"], r.to_dict()["fix_id"])
            for r in engine_results
        )
        mcp_triples = collections.Counter(
            (r.get("rule_id"), r.get("severity"), r.get("fix_id"))
            for r in self._result_data.get("results", [])
        )

        assert engine_triples == mcp_triples, (
            f"MAJOR-1: (rule_id, severity, fix_id) multisets must match between MCP tool "
            f"and direct rule engine call. "
            f"Only in MCP: {dict(mcp_triples - engine_triples)!r}. "
            f"Only in engine: {dict(engine_triples - mcp_triples)!r}. "
            "FINDING if mismatch: the MCP handler is filtering, transforming, or "
            "deduplicating rule results."
        )

    def test_crypto_name_fires_via_mcp(self):
        """Crypto_name rule (CryptoPrimitives) must fire in MCP results."""
        results = self._result_data.get("results", [])
        fix_ids = [r.get("fix_id") for r in results]
        assert "crypto_name" in fix_ids, (
            f"crypto_name rule must fire (CryptoPrimitives var present in stage); "
            f"got fix_ids={fix_ids!r}"
        )

    def test_legacy_exr_fires_via_mcp(self):
        """legacy_exr rule (multipart beauty + crypto) must fire in MCP results."""
        results = self._result_data.get("results", [])
        fix_ids = [r.get("fix_id") for r in results]
        assert "legacy_exr" in fix_ids, (
            f"legacy_exr rule must fire (beauty: multipart + crypto); "
            f"got fix_ids={fix_ids!r}"
        )

    def test_crypto_layers_is_list_not_indexed(self):
        """BLOCKER-3: when a parse includes crypto channels, crypto_layers must be a list.

        This tests via the lint-path — the broken stage has CryptoPrimitives.
        Parse an EXR to confirm crypto_layers type is list (never hard-indexed).
        (No EXR fixture needed — we just verify the type contract when crypto
        channels are present or absent.)
        """
        # We use a parse call on a hoiiotool-available EXR if possible,
        # or skip if hoiiotool not available (lint tests still run).
        if not HOIIOTOOL_AVAILABLE:
            return  # Cannot create EXR — skip the crypto_layers list check here

        tmpdir = tempfile.mkdtemp(prefix="rr_crypto_")
        try:
            exr_path = os.path.join(tmpdir, "plain_rgb.exr")
            _create_rgb_exr(exr_path, 32, 32)
            result = self._dispatch("render_parse_exr", {"exr_path": exr_path})
            data = _parse_data(result)
            assert data.get("ok") is not False, (
                f"render_parse_exr must succeed; error={data.get('error')!r}"
            )
            crypto_layers = data.get("crypto_layers")
            assert isinstance(crypto_layers, list), (
                f"BLOCKER-3: crypto_layers must always be a list; "
                f"got type={type(crypto_layers).__name__!r}, value={crypto_layers!r}"
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# Class 5: TestNegativePathsThroughDispatcher
# FR-2 fail-loud: missing/invalid args return {ok: False, error: '...'}.
# All four tools. Dispatched ONLY — no direct handler calls.
# ===========================================================================

class TestNegativePathsThroughDispatcher:
    """RR-G5: Negative paths — FR-2 fail-loud through REAL dispatcher.

    All four tools must return {ok: False, error: '...'} for invalid/missing args.
    Dispatched only — no direct handler calls.

    Does NOT require HOU_AVAILABLE for arg-validation tests (metric, empty paths)
    since those return before any hou.* call. BUT since importing the handler
    does import hou, HOU_AVAILABLE is required for the import itself.
    """

    _dispatch = None

    @classmethod
    def setup_class(cls):
        if not HOU_AVAILABLE:
            raise RuntimeError(
                "TestNegativePathsThroughDispatcher: handlers import hou; "
                "requires hython"
            )
        cls._dispatch = _get_dispatcher()

    def test_lint_invalid_render_node_returns_error(self):
        """FR-2: render_lint_settings with non-existent node returns {ok:False}."""
        result = self._dispatch(
            "render_lint_settings",
            {"render_node": "/stage/__nonexistent_rr_g5__", "preset": "nuke_safe"},
        )
        data = _lint_data(result)
        assert data.get("ok") is False, (
            f"FR-2: invalid render_node must return ok=False; got data={data!r}"
        )
        assert isinstance(data.get("error"), str) and data["error"], (
            f"FR-2: must return non-empty error string; got {data!r}"
        )

    def test_lint_empty_render_node_returns_error(self):
        """FR-2: render_lint_settings with empty string returns {ok:False}."""
        result = self._dispatch(
            "render_lint_settings",
            {"render_node": "", "preset": "nuke_safe"},
        )
        data = _lint_data(result)
        assert data.get("ok") is False, (
            f"FR-2: empty render_node must return ok=False; got data={data!r}"
        )

    def test_parse_nonexistent_exr_returns_error(self):
        """FR-2: render_parse_exr with non-existent file returns {ok:False}."""
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": "/nonexistent/__rr_g5__.exr"},
        )
        data = _parse_data(result)
        assert data.get("ok") is False, (
            f"FR-2: non-existent exr_path must return ok=False; got data={data!r}"
        )
        assert isinstance(data.get("error"), str) and data["error"], (
            f"FR-2: must return non-empty error string; got {data!r}"
        )

    def test_parse_empty_exr_path_returns_error(self):
        """FR-2: render_parse_exr with empty exr_path returns {ok:False}."""
        result = self._dispatch(
            "render_parse_exr",
            {"exr_path": ""},
        )
        data = _parse_data(result)
        assert data.get("ok") is False, (
            f"FR-2: empty exr_path must return ok=False; got data={data!r}"
        )

    def test_read_pixels_nonexistent_source_returns_error(self):
        """FR-2: render_read_pixels with non-existent file returns {ok:False}."""
        result = self._dispatch(
            "render_read_pixels",
            {"source": "/nonexistent/__rr_g5__.exr", "plane": "C", "mode": "summary"},
        )
        data = _read_data(result)
        assert data.get("ok") is False, (
            f"FR-2: non-existent source must return ok=False; got data={data!r}"
        )
        assert isinstance(data.get("error"), str) and data["error"], (
            f"FR-2: must return non-empty error string; got {data!r}"
        )

    def test_read_pixels_invalid_mode_returns_error(self):
        """FR-2: render_read_pixels with mode='bogus' returns {ok:False}."""
        result = self._dispatch(
            "render_read_pixels",
            {"source": "/nonexistent/__rr_g5__.exr", "plane": "C", "mode": "bogus"},
        )
        data = _read_data(result)
        assert data.get("ok") is False, (
            f"FR-2: invalid mode must return ok=False; got data={data!r}"
        )
        error = data.get("error", "").lower()
        assert "mode" in error or "bogus" in error, (
            f"FR-2: error must mention the invalid mode; got error={data.get('error')!r}"
        )

    def test_compare_invalid_metric_returns_error(self):
        """MAJOR-3: render_compare with metric='bogus' must return error before file I/O."""
        result = self._dispatch(
            "render_compare",
            {
                "a": "/nonexistent/__rr_g5_a__.exr",
                "b": "/nonexistent/__rr_g5_b__.exr",
                "planes": None,
                "metric": "bogus",
            },
        )
        assert _rc_is_failure(result), (
            f"MAJOR-3: invalid metric must produce ok=False; got result={result!r}"
        )
        error = _rc_get_error(result)
        assert "bogus" in error or "metric" in error.lower(), (
            f"MAJOR-3: error must name the invalid metric; got error={error!r}"
        )
        # Must NOT be a file-not-found error (metric validates FIRST)
        assert "not found" not in error.lower() and "no such file" not in error.lower(), (
            f"MAJOR-3: metric error must fire BEFORE file I/O; got error={error!r}"
        )

    def test_compare_empty_paths_returns_error(self):
        """FR-2: render_compare with empty a/b paths returns {ok:False}."""
        result = self._dispatch(
            "render_compare",
            {"a": "", "b": "", "planes": None, "metric": "stats"},
        )
        assert _rc_is_failure(result), (
            f"FR-2: empty paths must return ok=False; got result={result!r}"
        )

    def test_compare_nonexistent_files_returns_error(self):
        """FR-2: render_compare with non-existent files returns {ok:False}."""
        result = self._dispatch(
            "render_compare",
            {
                "a": "/nonexistent/__rr_g5_a__.exr",
                "b": "/nonexistent/__rr_g5_b__.exr",
                "planes": None,
                "metric": "stats",
            },
        )
        assert _rc_is_failure(result), (
            f"FR-2: non-existent files must return ok=False; got result={result!r}"
        )


# ===========================================================================
# Class 6: TestGateUngated
# FR-10: all four tools are registered as Capability.READONLY — verified via
# capability_of() from the dispatcher registry (authoritative, non-vacuous).
# ===========================================================================

class TestGateUngated:
    """RR-G6: Gate / ungated verification — all four read-only tools bypass the 109 gate.

    FR-10 contract: a READONLY capability means the handler is registered with
    Capability.READONLY in the dispatcher registry. This is what the 109 gate
    reads when deciding whether to queue a pending call — so asserting
    capability_of(cmd) == Capability.READONLY is the direct, non-vacuous test
    of the gate contract.

    Fix applied (F-MAJOR-1): replaced the vacuous _get_pending_count() delta
    pattern — which always returned -1 because list_pending_calls is NOT a
    registered dispatcher command — with direct capability_of() assertions.
    The if-before != -1 guard is gone: these tests ALWAYS assert.

    Requires: HOU_AVAILABLE (handlers import hou; registration fires on import).
    """

    _dispatch = None

    @classmethod
    def setup_class(cls):
        if not HOU_AVAILABLE:
            raise RuntimeError("TestGateUngated: requires hython")
        # Importing render_readback_handlers fires register_handler() for all four commands.
        cls._dispatch = _get_dispatcher()

    def _assert_readonly(self, command: str) -> None:
        """Assert command is registered as Capability.READONLY in the dispatcher.

        CARDINAL RULE: if this assertion fails, it reveals a real registration bug.
        STOP and surface it — do NOT weaken the assertion.
        """
        from fxhoudinimcp_server.dispatcher import Capability, capability_of  # noqa: PLC0415
        cap = capability_of(command)
        assert cap is not None, (
            f"FR-10: '{command}' is NOT registered in the dispatcher. "
            f"FINDING: register_handler('{command}', ...) was never called. "
            "hou-dev must add the registration call in render_readback_handlers.py."
        )
        assert cap == Capability.READONLY, (
            f"FR-10: '{command}' is registered with capability={cap!r}, "
            f"expected Capability.READONLY. "
            "FINDING: wrong capability declared — handler will be gated (queued) instead "
            "of running inline. hou-dev must pass Capability.READONLY to register_handler."
        )

    def test_lint_settings_is_readonly(self):
        """render_lint_settings must be registered as Capability.READONLY (FR-10)."""
        self._assert_readonly("render_lint_settings")

    def test_parse_exr_is_readonly(self):
        """render_parse_exr must be registered as Capability.READONLY (FR-10)."""
        self._assert_readonly("render_parse_exr")

    def test_read_pixels_is_readonly(self):
        """render_read_pixels must be registered as Capability.READONLY (FR-10)."""
        self._assert_readonly("render_read_pixels")

    def test_compare_is_readonly(self):
        """render_compare must be registered as Capability.READONLY (FR-10)."""
        self._assert_readonly("render_compare")

    def test_all_four_tools_are_readonly(self):
        """FR-10 comprehensive: all four render-readback tools are Capability.READONLY.

        Single gate that fails with a clear FINDING if any tool's capability is wrong.
        """
        from fxhoudinimcp_server.dispatcher import Capability, capability_of  # noqa: PLC0415
        commands = [
            "render_lint_settings",
            "render_parse_exr",
            "render_read_pixels",
            "render_compare",
        ]
        wrong = []
        for cmd in commands:
            cap = capability_of(cmd)
            if cap is None:
                wrong.append(f"{cmd}: NOT REGISTERED")
            elif cap != Capability.READONLY:
                wrong.append(f"{cmd}: capability={cap!r} (expected READONLY)")
        assert not wrong, (
            "FR-10: the following commands do not have Capability.READONLY — "
            "they will be gated instead of running inline:\n"
            + "\n".join(f"  {w}" for w in wrong)
        )


# ===========================================================================
# Broken-Karma LOP stage builder (mirrors render_lint_hython_smoke.py verbatim)
# MAJOR-2: copy the lint smoke's _build_lop_stage verbatim — do NOT reinvent.
# ===========================================================================

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

# AOV with channel_prefix
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


def _build_lop_stage(node_name: str = "rr_broken_karma_stage") -> str:
    """Create a pythonscript LOP in /stage with the broken-Karma fixture.

    Verbatim from render_lint_hython_smoke.py (MAJOR-2 compliance).
    Returns the scene path of the created node.
    """
    if not HOU_AVAILABLE:
        raise RuntimeError("_build_lop_stage: hou not available — run under hython")

    stage_ctx = hou.node("/stage")
    if stage_ctx is None:
        raise RuntimeError("_build_lop_stage: /stage context not found in scene")

    ps = stage_ctx.createNode("pythonscript", node_name)
    ps.parm("python").set(_BROKEN_KARMA_SCRIPT)
    node_stage = ps.stage()
    if node_stage is None:
        raise RuntimeError(
            f"_build_lop_stage: node.stage() returned None for {ps.path()!r}; "
            "pythonscript cook may have failed."
        )
    return ps.path()


# ===========================================================================
# Entry point
# ===========================================================================

_TEST_CLASSES = [
    TestAgentLoopEndToEnd,
    TestParseReadVariants,
    TestCrossToolPlaneConsistency,
    TestRuleParityWithProject05,
    TestNegativePathsThroughDispatcher,
    TestGateUngated,
]


def main() -> int:
    print("=" * 70)
    print("render_readback aggregate hython integration test (pp12-114g)")
    print(f"  HOU_AVAILABLE={HOU_AVAILABLE}  HOIIOTOOL_AVAILABLE={HOIIOTOOL_AVAILABLE}")
    print("=" * 70)
    for cls in _TEST_CLASSES:
        print(f"\n{cls.__name__}:")
        _run_class(cls)
    print("\n" + "=" * 70)
    print(f"Results: {_PASS_COUNT} PASS, {_FAIL_COUNT} FAIL")
    if _ERRORS:
        print("\nFailed tests:")
        for name, msg in _ERRORS:
            print(f"  FAIL: {name}")
            print(f"        {msg[:300]}")
    print("=" * 70)
    return 1 if _FAIL_COUNT > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
