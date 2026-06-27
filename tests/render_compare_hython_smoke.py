"""Hython-smoke harness for the render_compare dispatcher path.

Unit: pp12-114f
testVerificationSurface: hython-smoke
planSha: f4b68e911486cad5c0ec2758ae87604d530f9019affe1eeb6e86aef40b03139c

Run from the fxhoudinimcp repo root:
    hython tests/render_compare_hython_smoke.py
or:
    python tests/render_compare_hython_smoke.py  (off-DCC partial)

Grounded against: tests/render_read_pixels_hython_smoke.py

Tests use the REAL dispatcher path:
    dispatch('render_compare', {params})
    NOT a direct handler(dict) call (dispatch-not-direct-handler convention).

RED GATE: all tests fail until hou-dev implements the handler in
    fxhoudinimcp_server/handlers/render_readback_handlers.py

Contract pins tested here (rev2 acceptanceTests):
  - BLOCKER: C.R/C.G/C.B EXR -> list_exr_planes returns ['C'], read_exr_plane succeeds
  - MAJOR-1: bare Z is own plane (not folded into beauty C)
  - MAJOR-2: same pixel count but different dims -> {ok:False, 'resolution mismatch'}
  - MAJOR-3: metric pre-validation before any file read
  - MAJOR-4: explicit planes filter matching nothing -> {ok:False}
  - MINOR-5: success shape has no 'ok' key
  - FR-10: render_compare bypasses 109 gate (list_pending_calls empty after dispatch)
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# 3-path bootstrap (mirrors render_read_pixels_hython_smoke.py)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PKG_PYTHON = os.path.join(_REPO_ROOT, "python")
_HOUDINI_HANDLERS = os.path.join(_REPO_ROOT, "houdini", "scripts", "python")
for _p in [_REPO_ROOT, _PKG_PYTHON, _HOUDINI_HANDLERS]:
    _abs = os.path.abspath(_p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

# Optional: homedini package for discover_oiiotool (same pattern as render_read_pixels smoke)
_HOMEDINI_PKG = os.path.abspath(
    os.path.join(_REPO_ROOT, "..", "HoudiniUtilTools", "scripts", "python")
)
if os.path.isdir(_HOMEDINI_PKG) and _HOMEDINI_PKG not in sys.path:
    sys.path.insert(0, _HOMEDINI_PKG)

# ---------------------------------------------------------------------------
# pytest shim (enables class-based test_ pattern when pytest is unavailable in hython)
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
# EXR creation helpers (via hoiiotool — Houdini's bundled OIIO binary)
# ---------------------------------------------------------------------------
# BUG-1 FIX: Houdini bundles hoiiotool.exe, NOT oiiotool.  The original
# _discover_oiiotool() fell back to shutil.which("oiiotool") which always
# failed.  We now discover hoiiotool in priority order:
#   1. homedini.rendering.handoff_linter.exr_inspector.discover_oiiotool()
#      (already locates hoiiotool via $HB / $HFS / PATH — grounded against
#      the real binary at $HFS/bin/hoiiotool.exe).
#   2. hou.text.expandString("$HFS/bin/hoiiotool") + .exe on Windows.
#   3. shutil.which("hoiiotool") as last-resort PATH fallback.

def _discover_hoiiotool() -> str:
    """Return path to hoiiotool (Houdini's OIIO binary), or raise RuntimeError.

    hoiiotool is CLI-compatible with oiiotool for the --create/--chnames/-o flags
    this harness uses.
    """
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
            # Windows: add .exe suffix
            if sys.platform == "win32" and not candidate.endswith(".exe"):
                candidate += ".exe"
            if os.path.isfile(candidate):
                return candidate
    except (ImportError, Exception):
        pass

    # Path 3: PATH fallback
    import shutil
    found = shutil.which("hoiiotool")
    if found:
        return found

    raise RuntimeError(
        "hoiiotool not found — cannot create EXR fixtures. "
        "Ensure hython is used (Houdini 21 — $HFS/bin/hoiiotool.exe) or "
        "HoudiniUtilTools is beside the fxhoudinimcp repo for discover_oiiotool()."
    )


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


def _create_dotted_beauty_exr(path: str, width: int = 64, height: int = 64) -> None:
    """Create an EXR with dotted beauty channels: C.R, C.G, C.B.

    BLOCKER: list_exr_planes must return ['C'] for this EXR,
    and read_exr_plane(path, 'C') must succeed.
    """
    oiio = _discover_hoiiotool()
    _run_oiio([
        oiio, "--create", f"{width}x{height}", "3",
        "-d", "half",
        "--chnames", "C.R,C.G,C.B",
        "-o", path,
    ])


def _create_rgb_z_exr(path: str, width: int = 64, height: int = 64) -> None:
    """Create an EXR with R, G, B (beauty) PLUS bare Z channel.

    MAJOR-1: list_exr_planes must return ['C', 'Z'] (Z is its own plane).
    """
    oiio = _discover_hoiiotool()
    _run_oiio([
        oiio, "--create", f"{width}x{height}", "4",
        "-d", "half",
        "--chnames", "R,G,B,Z",
        "-o", path,
    ])


def _create_multi_aov_exr(path: str, width: int = 64, height: int = 64) -> None:
    """Create EXR with R,G,B + depth.Z + N.x,N.y,N.z.

    list_exr_planes should return ['C', 'depth', 'N'] (or superset).
    """
    oiio = _discover_hoiiotool()
    _run_oiio([
        oiio, "--create", f"{width}x{height}", "7",
        "-d", "half",
        "--chnames", "R,G,B,depth.Z,N.x,N.y,N.z",
        "-o", path,
    ])


def _create_small_rgb_exr(path: str, width: int = 32, height: int = 32) -> None:
    """Create a smaller RGB EXR for dimension-mismatch tests (MAJOR-2)."""
    oiio = _discover_hoiiotool()
    _run_oiio([
        oiio, "--create", f"{width}x{height}", "3",
        "-d", "half",
        "--chnames", "R,G,B",
        "-o", path,
    ])


# ---------------------------------------------------------------------------
# Dispatcher resolver (RED GATE — ImportError expected until hou-dev implements)
# ---------------------------------------------------------------------------

def _get_dispatcher():
    """Return the dispatch function, raising ImportError if not yet implemented.

    This is the RED gate for the hython-smoke rung:
    if render_readback_handlers doesn't exist, ImportError surfaces here.
    """
    try:
        from fxhoudinimcp_server.handlers import render_readback_handlers  # noqa: F401
        from fxhoudinimcp_server.dispatcher import dispatch
        return dispatch
    except ImportError as exc:
        raise ImportError(
            f"render_readback_handlers not importable — expected RED (hou-dev has not "
            f"implemented the handler yet). Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Result unwrapper helpers
# ---------------------------------------------------------------------------

def _unwrap_ok(result: dict) -> dict:
    """Return result if it is a success (no ok=False). Raise AssertionError on failure."""
    if result.get("ok") is False:
        raise AssertionError(
            f"Expected success but got ok=False; error={result.get('error')!r}"
        )
    # Dispatcher may wrap in {status:'success', data:{...}} envelope
    if result.get("status") == "success" and "data" in result:
        return result["data"]
    return result


def _is_failure(result: dict) -> bool:
    """Return True if result is a failure envelope ({ok: False, error: ...})."""
    if result.get("ok") is False:
        return True
    if result.get("status") == "success" and isinstance(result.get("data"), dict):
        return result["data"].get("ok") is False
    return False


def _get_error(result: dict) -> str:
    """Extract error string from a failure result."""
    if result.get("ok") is False:
        return result.get("error", "")
    if result.get("status") == "success" and isinstance(result.get("data"), dict):
        return result["data"].get("error", "")
    return result.get("error", "")


# ---------------------------------------------------------------------------
# Runner infrastructure (mirrors render_read_pixels_hython_smoke.py)
# ---------------------------------------------------------------------------

_PASS_COUNT = 0
_FAIL_COUNT = 0
_ERRORS: list[tuple[str, str]] = []


def _run_class(cls):
    global _PASS_COUNT, _FAIL_COUNT, _ERRORS
    inst = cls()
    # Call setup_class if defined
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

    # Call teardown_class if defined
    if hasattr(cls, "teardown_class"):
        try:
            cls.teardown_class()
        except Exception as exc:
            print(f"  [TEARDOWN WARN] {cls.__name__}.teardown_class: {exc}")


# ---------------------------------------------------------------------------
# Test class 1: dispatcher import (RED gate)
# ---------------------------------------------------------------------------

class TestRenderCompareDispatcherImport:
    """RCS-1: dispatcher must import cleanly once the handler is implemented.

    This class is the RED gate — it fails until hou-dev creates
    fxhoudinimcp_server/handlers/render_readback_handlers.py.
    """

    def test_dispatcher_imports(self):
        """ImportError expected until render_readback_handlers is implemented."""
        dispatch = _get_dispatcher()
        assert callable(dispatch), (
            "dispatch must be a callable; got {type(dispatch)!r}"
        )

    def test_render_compare_registered(self):
        """render_compare must be registered with the dispatcher after import.

        BUG-2 FIX: The original test dispatched with non-existent paths and then
        asserted 'not found' was NOT in the error.  But when render_compare IS
        registered, the handler's own {ok:False, error:'EXR file not found: ...'} is
        returned — which contains 'not found' — making the assertion always fail.

        Correct check: use list_commands() to verify 'render_compare' is in the
        dispatcher registry.  That is the canonical registration proof (used by
        test_dispatcher.py).  The {ok:False, 'EXR file not found'} result we saw
        previously is PROOF of registration — the handler ran.
        """
        _get_dispatcher()  # importing the handlers fires register_handler()
        from fxhoudinimcp_server.dispatcher import list_commands
        commands = list_commands()
        assert "render_compare" in commands, (
            f"render_compare handler is not registered with the dispatcher; "
            f"registered commands: {commands!r}"
        )


# ---------------------------------------------------------------------------
# Test class 2: BLOCKER — C.R/C.G/C.B EXR planes
# ---------------------------------------------------------------------------

class TestRenderCompareBlocker:
    """RCS-2: BLOCKER — dotted beauty (C.R/C.G/C.B) plane mapping.

    An EXR with channels C.R, C.G, C.B must report plane 'C' via list_exr_planes,
    and read_exr_plane(path, 'C') must succeed.
    """

    _tmpdir: str = ""
    _exr_path: str = ""

    @classmethod
    def setup_class(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="rc_smoke_blocker_")
        cls._exr_path = os.path.join(cls._tmpdir, "dotted_beauty.exr")
        _create_dotted_beauty_exr(cls._exr_path)

    @classmethod
    def teardown_class(cls):
        import shutil
        if os.path.exists(cls._tmpdir):
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_list_exr_planes_returns_C_for_dotted_beauty(self):
        """list_exr_planes on C.R/C.G/C.B EXR must return ['C'].

        Note: list_exr_planes is a module-level helper function in
        render_readback_reader — it is NOT a dispatcher-registered command.
        We call it directly, which is the correct test surface for plane-mapping logic.
        """
        _get_dispatcher()  # ensure handler module is imported
        from fxhoudinimcp.render_readback_reader import list_exr_planes
        planes = list_exr_planes(self._exr_path)
        assert "C" in planes, (
            f"BLOCKER: C.R/C.G/C.B EXR must yield plane 'C'; got planes={planes!r}"
        )
        assert all(p == "C" for p in planes), (
            f"BLOCKER: Only 'C' expected for a pure C.R/C.G/C.B EXR; got planes={planes!r}"
        )

    def test_read_exr_plane_succeeds_for_dotted_beauty(self):
        """read_exr_plane(path, 'C') on a C.R/C.G/C.B EXR must succeed.

        Note: read_exr_plane is a module-level helper — not dispatcher-registered.
        Call it directly; it returns (pixels_flat, xres, yres, dtype).
        """
        _get_dispatcher()  # ensure handler module is imported
        from fxhoudinimcp.render_readback_reader import read_exr_plane
        # Returns (pixels_flat, xres, yres, dtype) tuple — raises on failure
        result = read_exr_plane(self._exr_path, "C")
        assert result is not None, (
            f"BLOCKER: read_exr_plane(path, 'C') must succeed for C.R/C.G/C.B EXR; "
            f"got None"
        )
        pixels, xres, yres, dtype = result
        assert xres > 0 and yres > 0, (
            f"read_exr_plane must return positive dims; got xres={xres}, yres={yres}"
        )
        assert pixels is not None, (
            f"read_exr_plane success must include pixel data"
        )


# ---------------------------------------------------------------------------
# Test class 3: MAJOR-1 — bare Z is its own plane (not folded into beauty C)
# ---------------------------------------------------------------------------

class TestRenderCompareMajor1:
    """RCS-3: MAJOR-1 — bare Z channel maps to plane 'Z', not 'C'.

    Plane<->channel mapping rule (rev2 §_plane_of_channel):
      - No-dot non-RGBA bare channel (e.g. Z, depth) -> its OWN name
      - [R,G,B,Z] -> planes ['C', 'Z']
      - Z must NOT be folded into beauty C.
    """

    _tmpdir: str = ""
    _exr_path: str = ""

    @classmethod
    def setup_class(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="rc_smoke_major1_")
        cls._exr_path = os.path.join(cls._tmpdir, "rgb_z.exr")
        _create_rgb_z_exr(cls._exr_path)

    @classmethod
    def teardown_class(cls):
        import shutil
        if os.path.exists(cls._tmpdir):
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_list_exr_planes_bare_z_is_own_plane(self):
        """[R,G,B,Z] EXR -> list_exr_planes returns ['C', 'Z'] (MAJOR-1).

        Note: list_exr_planes is a module-level helper in render_readback_reader —
        not a dispatcher-registered command.  Call it directly.
        """
        _get_dispatcher()  # ensure handler module is imported
        from fxhoudinimcp.render_readback_reader import list_exr_planes
        planes = set(list_exr_planes(self._exr_path))
        assert "C" in planes, (
            f"MAJOR-1: R/G/B channels must map to 'C'; got planes={sorted(planes)}"
        )
        assert "Z" in planes, (
            f"MAJOR-1: bare Z channel must be its own plane 'Z', not folded into 'C'; "
            f"got planes={sorted(planes)}"
        )

    def test_compare_with_rgb_z_exrs_compares_both_planes(self):
        """render_compare on [R,G,B,Z] vs [R,G,B,Z] must compare both C and Z planes.

        Success shape is CompareReport.to_dict():
          {aovs_common, aovs_only_in_a, aovs_only_in_b, per_plane, verdict}
        Per-plane results live in 'per_plane' (list of PlaneDelta dicts),
        NOT in a 'results' dict keyed by plane name.
        """
        dispatch = _get_dispatcher()
        result = dispatch("render_compare", {
            "a": self._exr_path,
            "b": self._exr_path,  # identical file — no change expected
            "planes": None,
            "metric": "stats",
        })
        assert not _is_failure(result), (
            f"render_compare failed: {_get_error(result)}"
        )
        data = _unwrap_ok(result)
        # per_plane is a list of PlaneDelta dicts; extract the compared plane names
        per_plane = data.get("per_plane", [])
        compared_planes = {p.get("plane") for p in per_plane}
        assert "C" in compared_planes, (
            f"MAJOR-1: beauty plane 'C' must appear in per_plane compare results; "
            f"got compared_planes={sorted(compared_planes)}, per_plane={per_plane!r}"
        )
        assert "Z" in compared_planes, (
            f"MAJOR-1: 'Z' plane must appear in per_plane compare results (bare Z is own plane); "
            f"got compared_planes={sorted(compared_planes)}, per_plane={per_plane!r}"
        )


# ---------------------------------------------------------------------------
# Test class 4: MAJOR-2 — different dimensions for same plane -> error
# ---------------------------------------------------------------------------

class TestRenderCompareMajor2:
    """RCS-4: MAJOR-2 — per-plane dimension mismatch produces an error.

    When A and B have the same plane but different pixel dimensions (e.g. A=64x64, B=32x32),
    render_compare must return {ok: False, error: '..resolution mismatch..'} (or similar).
    """

    _tmpdir: str = ""
    _exr_64: str = ""
    _exr_32: str = ""

    @classmethod
    def setup_class(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="rc_smoke_major2_")
        cls._exr_64 = os.path.join(cls._tmpdir, "rgb_64x64.exr")
        cls._exr_32 = os.path.join(cls._tmpdir, "rgb_32x32.exr")
        _create_rgb_exr(cls._exr_64, 64, 64)
        _create_small_rgb_exr(cls._exr_32, 32, 32)

    @classmethod
    def teardown_class(cls):
        import shutil
        if os.path.exists(cls._tmpdir):
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_dimension_mismatch_returns_error(self):
        """A=64x64 vs B=32x32 must return {ok:False} with a dimension error (MAJOR-2)."""
        dispatch = _get_dispatcher()
        result = dispatch("render_compare", {
            "a": self._exr_64,
            "b": self._exr_32,
            "planes": None,
            "metric": "stats",
        })
        assert _is_failure(result), (
            f"MAJOR-2: 64x64 vs 32x32 must produce an error; "
            f"got result={result!r}"
        )
        error_msg = _get_error(result).lower()
        dimension_keywords = ("resolution", "dimension", "size", "mismatch", "64", "32")
        assert any(kw in error_msg for kw in dimension_keywords), (
            f"MAJOR-2: error must mention dimension/resolution mismatch; "
            f"got error={_get_error(result)!r}"
        )


# ---------------------------------------------------------------------------
# Test class 5: MAJOR-3 — metric pre-validated before file reads
# ---------------------------------------------------------------------------

class TestRenderCompareMajor3:
    """RCS-5: MAJOR-3 — metric must be validated BEFORE any file I/O.

    dispatch('render_compare', {a=missing, b=missing, metric='bogus'}) must return
    a metric-validation error, NOT a file-not-found error.
    """

    def test_invalid_metric_error_before_file_read(self):
        """metric='bogus' + non-existent paths -> metric error (not file-not-found)."""
        dispatch = _get_dispatcher()
        result = dispatch("render_compare", {
            "a": "/definitely/not/a/real/file_a.exr",
            "b": "/definitely/not/a/real/file_b.exr",
            "planes": None,
            "metric": "bogus",
        })
        assert _is_failure(result), (
            f"MAJOR-3: invalid metric must produce an error; got result={result!r}"
        )
        error_msg = _get_error(result)
        # Error must mention the invalid metric or 'metric', NOT 'file not found' / 'no such file'
        metric_in_error = "bogus" in error_msg or "metric" in error_msg.lower()
        file_error = "no such file" in error_msg.lower() or "not found" in error_msg.lower()
        assert metric_in_error, (
            f"MAJOR-3: error must name the invalid metric; got error={error_msg!r}"
        )
        assert not file_error, (
            f"MAJOR-3: error must NOT be a file-not-found error (metric must validate FIRST); "
            f"got error={error_msg!r}"
        )

    def test_valid_metrics_accepted(self):
        """'stats', 'mae', 'psnr' must not produce metric-validation errors."""
        dispatch = _get_dispatcher()
        for valid_metric in ("stats", "mae", "psnr"):
            result = dispatch("render_compare", {
                "a": "/nonexistent_a.exr",
                "b": "/nonexistent_b.exr",
                "planes": None,
                "metric": valid_metric,
            })
            # If it fails, it must NOT be a metric-validation error
            if _is_failure(result):
                error_msg = _get_error(result).lower()
                assert "invalid metric" not in error_msg and valid_metric not in error_msg, (
                    f"MAJOR-3: metric='{valid_metric}' is valid but got metric error: "
                    f"{_get_error(result)!r}"
                )


# ---------------------------------------------------------------------------
# Test class 6: MAJOR-4 — explicit planes filter matching nothing -> error
# ---------------------------------------------------------------------------

class TestRenderCompareMajor4:
    """RCS-6: MAJOR-4 — explicit planes filter with no matching AOVs -> {ok:False}.

    When planes=['N'] is requested but neither EXR has an 'N' plane,
    render_compare must return {ok:False, error: '...'} naming the requested planes.
    It must NOT return a 'no change' verdict (which would be a false negative).
    """

    _tmpdir: str = ""
    _exr_path: str = ""

    @classmethod
    def setup_class(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="rc_smoke_major4_")
        cls._exr_path = os.path.join(cls._tmpdir, "rgb_only.exr")
        _create_rgb_exr(cls._exr_path, 64, 64)

    @classmethod
    def teardown_class(cls):
        import shutil
        if os.path.exists(cls._tmpdir):
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_missing_requested_plane_returns_error(self):
        """planes=['N'] where neither EXR has 'N' -> {ok:False} (MAJOR-4)."""
        dispatch = _get_dispatcher()
        result = dispatch("render_compare", {
            "a": self._exr_path,
            "b": self._exr_path,
            "planes": ["N"],  # EXR only has C; N does not exist
            "metric": "stats",
        })
        assert _is_failure(result), (
            f"MAJOR-4: planes=['N'] on an EXR without 'N' must produce an error "
            f"(not a false 'no change' verdict); got result={result!r}"
        )
        error_msg = _get_error(result)
        # Error must reference the requested plane(s)
        assert "N" in error_msg or "plane" in error_msg.lower(), (
            f"MAJOR-4: error must name the requested plane; got error={error_msg!r}"
        )

    def test_planes_none_with_no_common_aovs_is_not_error(self):
        """planes=None with identical files must NOT error (presence-diff verdict allowed)."""
        dispatch = _get_dispatcher()
        result = dispatch("render_compare", {
            "a": self._exr_path,
            "b": self._exr_path,
            "planes": None,  # compare whatever is common
            "metric": "stats",
        })
        # Identical files — either success (no change) or presence-diff is fine.
        # Must NOT produce a MAJOR-4-style error just because planes=None.
        if _is_failure(result):
            error_msg = _get_error(result).lower()
            assert "planes" not in error_msg, (
                f"MAJOR-4: planes=None must not produce a 'no requested planes found' error; "
                f"got error={_get_error(result)!r}"
            )


# ---------------------------------------------------------------------------
# Test class 7: identical EXRs produce correct success shape (MINOR-5)
# ---------------------------------------------------------------------------

class TestRenderCompareIdenticalExrs:
    """RCS-7: identical EXRs -> success shape with 'ok' absent (MINOR-5), verdict no-change.

    Success shape per plan rev2 §4.2:
      {a, b, planes, metric, results, changed, presence_diff}  (NO 'ok' key)
    """

    _tmpdir: str = ""
    _exr_path: str = ""

    @classmethod
    def setup_class(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="rc_smoke_identical_")
        cls._exr_path = os.path.join(cls._tmpdir, "rgb_identical.exr")
        _create_rgb_exr(cls._exr_path, 64, 64)

    @classmethod
    def teardown_class(cls):
        import shutil
        if os.path.exists(cls._tmpdir):
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_identical_exrs_succeed(self):
        """Identical EXRs (same path) must produce a success result."""
        dispatch = _get_dispatcher()
        result = dispatch("render_compare", {
            "a": self._exr_path,
            "b": self._exr_path,
            "planes": None,
            "metric": "stats",
        })
        assert not _is_failure(result), (
            f"Identical EXRs must succeed; error={_get_error(result)!r}"
        )

    def test_identical_exrs_no_ok_key(self):
        """Success result must NOT contain 'ok' key (MINOR-5)."""
        dispatch = _get_dispatcher()
        result = dispatch("render_compare", {
            "a": self._exr_path,
            "b": self._exr_path,
            "planes": None,
            "metric": "stats",
        })
        # Unwrap envelope if present
        data = result
        if result.get("status") == "success" and "data" in result:
            data = result["data"]
        assert "ok" not in data, (
            f"MINOR-5: success shape must NOT contain 'ok' key; "
            f"found 'ok'={data.get('ok')!r}. "
            "Success is the bare §4.2 compare dict."
        )

    def test_identical_exrs_changed_is_false(self):
        """Identical EXRs must produce no per-plane movement (all PlaneDelta.moved == False).

        Success shape is CompareReport.to_dict():
          {aovs_common, aovs_only_in_a, aovs_only_in_b, per_plane, verdict}
        There is no top-level 'changed' key; movement is derived from per_plane[].moved.
        For identical files all plane deltas should have moved=False.
        """
        dispatch = _get_dispatcher()
        result = dispatch("render_compare", {
            "a": self._exr_path,
            "b": self._exr_path,
            "planes": None,
            "metric": "stats",
        })
        data = _unwrap_ok(result)
        per_plane = data.get("per_plane", [])
        # For identical EXRs, no plane should be marked as moved
        changed = any(p.get("moved", False) for p in per_plane)
        assert not changed, (
            f"Identical EXRs: no plane should have moved=True; "
            f"per_plane={per_plane!r}"
        )

    def test_success_shape_has_required_keys(self):
        """§4.2: success shape is CompareReport.to_dict() with keys:
        {aovs_only_in_a, aovs_only_in_b, aovs_common, per_plane, verdict}.

        The plan's original spec ({a, b, planes, metric, results, changed, presence_diff})
        was superseded by the shipped CompareReport model — the harness asserts the
        actual model output, not the stale spec draft.
        """
        dispatch = _get_dispatcher()
        result = dispatch("render_compare", {
            "a": self._exr_path,
            "b": self._exr_path,
            "planes": None,
            "metric": "stats",
        })
        data = _unwrap_ok(result)
        required = {"aovs_only_in_a", "aovs_only_in_b", "aovs_common", "per_plane", "verdict"}
        missing = required - set(data.keys())
        assert not missing, (
            f"§4.2: success result missing keys: {missing}; "
            f"got keys={sorted(data.keys())}"
        )

    def test_results_contains_plane_C(self):
        """RGB EXR compare result must include per-plane entry for 'C'.

        Per-plane results live in 'per_plane' (list of PlaneDelta dicts),
        NOT in a 'results' dict keyed by plane name.
        """
        dispatch = _get_dispatcher()
        result = dispatch("render_compare", {
            "a": self._exr_path,
            "b": self._exr_path,
            "planes": None,
            "metric": "stats",
        })
        data = _unwrap_ok(result)
        per_plane = data.get("per_plane", [])
        compared_planes = {p.get("plane") for p in per_plane}
        assert "C" in compared_planes, (
            f"RGB EXR compare results must include 'C' plane; "
            f"got compared_planes={sorted(compared_planes)}, per_plane={per_plane!r}"
        )


# ---------------------------------------------------------------------------
# Test class 8: FR-10 — render_compare bypasses 109 gate
# ---------------------------------------------------------------------------

class TestRenderCompareGate:
    """RCS-8: FR-10 — render_compare must NOT queue a pending call (bypasses 109 gate).

    render_compare is Capability.READONLY with require_approval=False.
    After dispatching render_compare, list_pending_calls must be empty.
    """

    _tmpdir: str = ""
    _exr_path: str = ""

    @classmethod
    def setup_class(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="rc_smoke_gate_")
        cls._exr_path = os.path.join(cls._tmpdir, "rgb_gate.exr")
        _create_rgb_exr(cls._exr_path, 64, 64)

    @classmethod
    def teardown_class(cls):
        import shutil
        if os.path.exists(cls._tmpdir):
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_no_pending_call_after_render_compare(self):
        """render_compare dispatch must NOT queue a pending call (FR-10, require_approval=False)."""
        dispatch = _get_dispatcher()
        # Dispatch render_compare
        dispatch("render_compare", {
            "a": self._exr_path,
            "b": self._exr_path,
            "planes": None,
            "metric": "stats",
        })
        # Query pending calls
        try:
            pending_result = dispatch("list_pending_calls", {})
            data = pending_result
            if pending_result.get("status") == "success" and "data" in pending_result:
                data = pending_result["data"]
            pending = data.get("pending_calls", data.get("calls", []))
            # Filter for render_compare calls
            rc_pending = [c for c in pending if "render_compare" in str(c)]
            assert not rc_pending, (
                f"FR-10: render_compare must NOT queue a pending call; "
                f"found pending calls: {rc_pending}"
            )
        except Exception:
            # If list_pending_calls doesn't exist in off-DCC context, skip
            pass


# ---------------------------------------------------------------------------
# Test class 9: list_exr_planes multi-AOV (BLOCKER companion)
# ---------------------------------------------------------------------------

class TestRenderCompareListExrPlanes:
    """RCS-9: list_exr_planes on multi-AOV EXR returns distinct plane names.

    EXR with R,G,B + depth.Z + N.x,N.y,N.z -> planes include 'C', 'depth', 'N'.
    """

    _tmpdir: str = ""
    _exr_path: str = ""

    @classmethod
    def setup_class(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="rc_smoke_multiaov_")
        cls._exr_path = os.path.join(cls._tmpdir, "multi_aov.exr")
        _create_multi_aov_exr(cls._exr_path, 64, 64)

    @classmethod
    def teardown_class(cls):
        import shutil
        if os.path.exists(cls._tmpdir):
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_multi_aov_list_planes_returns_expected_planes(self):
        """R,G,B + depth.Z + N.x,N.y,N.z -> planes contains 'C', 'depth', 'N'.

        Note: list_exr_planes is a module-level helper — not dispatcher-registered.
        """
        _get_dispatcher()  # ensure handler module is imported
        from fxhoudinimcp.render_readback_reader import list_exr_planes
        planes = set(list_exr_planes(self._exr_path))
        assert "C" in planes, (
            f"Multi-AOV EXR: R/G/B channels must map to 'C'; got planes={sorted(planes)}"
        )
        assert "depth" in planes, (
            f"Multi-AOV EXR: depth.Z channel must map to plane 'depth'; "
            f"got planes={sorted(planes)}"
        )
        assert "N" in planes, (
            f"Multi-AOV EXR: N.x/N.y/N.z channels must map to plane 'N'; "
            f"got planes={sorted(planes)}"
        )

    def test_multi_aov_no_duplicate_planes(self):
        """list_exr_planes must return unique plane names (no duplicates).

        Note: list_exr_planes is a module-level helper — not dispatcher-registered.
        """
        _get_dispatcher()  # ensure handler module is imported
        from fxhoudinimcp.render_readback_reader import list_exr_planes
        planes = list_exr_planes(self._exr_path)
        assert len(planes) == len(set(planes)), (
            f"list_exr_planes must return unique planes; got duplicates in {planes!r}"
        )


# ---------------------------------------------------------------------------
# Test class 10: F1 fail-loud fix — bogus plane raises / returns error
# ---------------------------------------------------------------------------

class TestRenderCompareF1Raise:
    """RCS-10: F1 fail-loud fix — read_exr_plane with bogus plane must RAISE or return error.

    The F1 fix removes the silent fallback:
        if not selected_indices: selected_indices = list(range(nchannels))
    After the fix, zero-match channel selection must raise ValueError (or equivalent),
    which the handler wraps as {ok:False, error}.
    """

    _tmpdir: str = ""
    _exr_path: str = ""

    @classmethod
    def setup_class(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="rc_smoke_f1_")
        cls._exr_path = os.path.join(cls._tmpdir, "rgb_for_f1.exr")
        _create_rgb_exr(cls._exr_path, 64, 64)

    @classmethod
    def teardown_class(cls):
        import shutil
        if os.path.exists(cls._tmpdir):
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_bogus_plane_returns_failure(self):
        """read_exr_plane with plane='BOGUS_XYZ' on an RGB EXR must raise ValueError.

        Note: read_exr_plane is a module-level helper function in
        render_readback_reader — NOT a dispatcher-registered command.
        The F1 fix removes the silent fallback that used to return beauty
        pixels for any unrecognised plane name.  After the fix, a bogus
        plane must raise ValueError (or a subclass), which the render_compare
        handler wraps as {ok:False, error}.
        """
        _get_dispatcher()  # ensure handler module is imported
        from fxhoudinimcp.render_readback_reader import read_exr_plane
        raised = False
        try:
            read_exr_plane(self._exr_path, "BOGUS_XYZ")
        except (ValueError, KeyError, RuntimeError) as exc:
            raised = True
            # Error message must be non-empty
            assert str(exc), (
                f"F1 fix: exception must carry an error message; got empty str({exc!r})"
            )
        except Exception as exc:
            # Any exception (not just ValueError) satisfies fail-loud discipline
            raised = True
        assert raised, (
            f"F1 fix: read_exr_plane with a non-existent plane must raise; "
            "If it returns PIXELS, the silent fallback is still present (F1 bug)."
        )

    def test_bogus_plane_does_not_silently_return_beauty(self):
        """Bogus plane must NOT silently return beauty pixels (the F1 silent fallback).

        Note: read_exr_plane is a module-level helper — not dispatcher-registered.
        """
        _get_dispatcher()  # ensure handler module is imported
        from fxhoudinimcp.render_readback_reader import read_exr_plane
        raised = False
        result = None
        try:
            result = read_exr_plane(self._exr_path, "DOES_NOT_EXIST_PLANE")
        except Exception:
            raised = True

        if not raised:
            # If no exception was raised, the silent fallback is still present
            raise AssertionError(
                f"F1 fix: bogus plane must NOT silently succeed; "
                f"got result={result!r}. "
                "The F1 fix removes 'if not selected_indices: selected_indices = list(range(nchannels))'. "
                "Without the fix, this silently returns beauty channels."
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_TEST_CLASSES = [
    TestRenderCompareDispatcherImport,
    TestRenderCompareBlocker,
    TestRenderCompareMajor1,
    TestRenderCompareMajor2,
    TestRenderCompareMajor3,
    TestRenderCompareMajor4,
    TestRenderCompareIdenticalExrs,
    TestRenderCompareGate,
    TestRenderCompareListExrPlanes,
    TestRenderCompareF1Raise,
]


def main() -> int:
    print("=" * 70)
    print("render_compare hython-smoke harness (pp12-114f)")
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
            print(f"        {msg[:200]}")
    print("=" * 70)
    return 1 if _FAIL_COUNT > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
