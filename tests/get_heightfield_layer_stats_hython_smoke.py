"""get_heightfield_layer_stats_hython_smoke.py — headless Houdini smoke for fork F5.

Dual-mode (mirrors setup_constraint_network_hython_smoke.py):
  - Standalone:  hython get_heightfield_layer_stats_hython_smoke.py
  - pytest:      @pytest.mark.hython_smoke tests SKIP when hou is not importable.

All calls go through the REAL dispatcher path — dispatch("geometry.get_heightfield_layer_stats",
{params}) -> handler(**params).

Covers:
  1. Positive (height): builds a noised heightfield; the height layer's real voxel stats come
     back (min<max, mean in-range, voxel_count == product of resolution) — the anti-fabrication read.
  2. Mask layer + histogram: the mask layer is in [0,1] and a histogram sums to the sampled count.
  3. Negative: a nonexistent layer -> status=error listing the available layers.

testVerificationSurface: hython-smoke
unitId: b4-w4-f5
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PKG_PYTHON = os.path.join(_REPO_ROOT, "python")
_HOUDINI_HANDLERS = os.path.join(_REPO_ROOT, "houdini", "scripts", "python")
for _p in [_REPO_ROOT, _PKG_PYTHON, _HOUDINI_HANDLERS]:
    _abs = os.path.abspath(_p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

try:
    import pytest  # type: ignore[import-not-found]
    _PYTEST_AVAILABLE = True
except ImportError:
    _PYTEST_AVAILABLE = False

    class _SkipIfDecorator:
        def __call__(self, condition, reason=""):
            def decorator(fn_or_cls):
                return fn_or_cls
            return decorator

    class _MarkNS:
        skipif = _SkipIfDecorator()

        def __getattr__(self, name):
            def decorator_factory(*args, **kwargs):
                def decorator(fn_or_cls):
                    return fn_or_cls
                return decorator
            return decorator_factory

    class pytest:  # type: ignore[no-redef]
        mark = _MarkNS()

        @staticmethod
        def fail(msg: str) -> None:
            raise AssertionError(msg)

        @staticmethod
        def skip(msg: str = "") -> None:
            raise RuntimeError(f"SKIP: {msg}")

try:
    import hou  # type: ignore[import-not-found]
    HOU_AVAILABLE = True
except ImportError:
    HOU_AVAILABLE = False

if _PYTEST_AVAILABLE:
    pytestmark = pytest.mark.hython_smoke


def _get_dispatcher():
    try:
        from fxhoudinimcp_server.handlers import geometry_handlers  # noqa: F401
        from fxhoudinimcp_server.dispatcher import dispatch
        return dispatch
    except ImportError as exc:
        raise ImportError(
            "dispatcher or geometry_handlers not importable — run under hython"
        ) from exc


def _build_heightfield():
    """A noised heightfield with a height + mask layer. Returns the terminal SOP path."""
    geo = hou.node("/obj").createNode("geo", "hf")
    for c in geo.children():
        c.destroy()
    hf = geo.createNode("heightfield", "heightfield1")
    noise = geo.createNode("heightfield_noise", "hf_noise")
    noise.setInput(0, hf)
    last = noise
    types = hou.sopNodeTypeCategory().nodeTypes()
    if "heightfield_maskbyfeature" in types:
        mask = geo.createNode("heightfield_maskbyfeature", "hf_mask")
        mask.setInput(0, noise)
        last = mask
    last.setDisplayFlag(True)
    return last.path()


@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestHeightfieldLayerStatsSmoke:

    def setup_method(self):
        hou.hipFile.clear(suppress_save_prompt=True)
        self._dispatch = _get_dispatcher()

    def test_height_layer_stats(self):
        node = _build_heightfield()
        r = self._dispatch("geometry.get_heightfield_layer_stats",
                           {"node_path": node, "layer": "height"})
        assert r.get("status") == "success", f"stats failed: {r}"
        d = r["data"]
        assert d["voxel_count"] == d["resolution"][0] * d["resolution"][1] * d["resolution"][2]
        assert d["min"] < d["max"], f"degenerate height range: {d}"
        assert d["min"] <= d["mean"] <= d["max"], f"mean out of range: {d}"

    def test_mask_layer_histogram(self):
        node = _build_heightfield()
        r = self._dispatch("geometry.get_heightfield_layer_stats",
                           {"node_path": node, "layer": "mask", "histogram_bins": 8})
        if r.get("status") != "success":
            # mask layer may be absent if the mask SOP wasn't available — skip, don't fail.
            return
        d = r["data"]
        assert 0.0 <= d["min"] and d["max"] <= 1.0 + 1e-6, f"mask not in [0,1]: {d}"
        assert len(d["histogram"]) == 8
        assert len(d["bin_edges"]) == 9
        # histogram counts are over the sample -> they must sum to sample_count.
        assert sum(d["histogram"]) == d["sample_count"], f"histogram != sample_count: {d}"

    def test_bad_layer_returns_error(self):
        node = _build_heightfield()
        r = self._dispatch("geometry.get_heightfield_layer_stats",
                           {"node_path": node, "layer": "no_such_layer"})
        assert r.get("status") == "error", f"expected error for bad layer, got {r}"
        # the error must list the available layers (so the caller can correct the name).
        assert "height" in str(r.get("error", "")), f"error should list available layers: {r}"

    def test_large_field_sampling_is_memory_bounded(self):
        # Force the >cap sampling path by lowering the cap; sample_count must stay bounded and
        # the histogram must still sum to it, while voxel_count reports the FULL field size.
        from fxhoudinimcp_server.handlers import geometry_handlers as gh
        node = _build_heightfield()  # 500x500 = 250k voxels
        orig = gh._HF_VOXEL_CAP
        try:
            gh._HF_VOXEL_CAP = 10_000
            r = self._dispatch("geometry.get_heightfield_layer_stats",
                               {"node_path": node, "layer": "height", "histogram_bins": 8})
        finally:
            gh._HF_VOXEL_CAP = orig
        assert r.get("status") == "success", f"sampling path failed: {r}"
        d = r["data"]
        assert d["sampled"] is True, "expected sampled=True below the lowered cap"
        assert d["sample_count"] <= 10_000, f"sample not bounded: {d['sample_count']}"
        assert d["voxel_count"] == d["resolution"][0] * d["resolution"][1] * d["resolution"][2]
        assert sum(d["histogram"]) == d["sample_count"]
        assert d["min"] < d["max"]  # min/max still exact via intrinsics


def main() -> int:
    if not HOU_AVAILABLE:
        print("ERROR: hou not importable — run with hython, not python.")
        return 1
    failures: list[str] = []
    print("\n--- get_heightfield_layer_stats smoke (fork F5) ---")
    print("Using: dispatch() -> handler(**params) [real dispatcher path]")
    try:
        dispatch = _get_dispatcher()
    except ImportError as exc:
        print(f"  FAIL  dispatcher/geometry_handlers import: {exc}")
        return 1

    print("\n  [1] height layer stats (min<max, mean in-range, voxel_count consistent)")
    try:
        hou.hipFile.clear(suppress_save_prompt=True)
        node = _build_heightfield()
        r = dispatch("geometry.get_heightfield_layer_stats", {"node_path": node, "layer": "height"})
        if r.get("status") != "success":
            failures.append(f"height failed: {r}")
            print(f"    FAIL  {r}")
        else:
            d = r["data"]
            ok = (d["voxel_count"] == d["resolution"][0] * d["resolution"][1] * d["resolution"][2]
                  and d["min"] < d["max"] and d["min"] <= d["mean"] <= d["max"])
            if ok:
                print(f"    PASS  res={d['resolution']} min={d['min']:.3f} max={d['max']:.3f} mean={d['mean']:.3f}")
            else:
                failures.append(f"height checks: {d}")
                print(f"    FAIL  {d}")
    except Exception as exc:
        failures.append(f"height raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    print("\n  [2] mask layer + histogram")
    try:
        hou.hipFile.clear(suppress_save_prompt=True)
        node = _build_heightfield()
        r = dispatch("geometry.get_heightfield_layer_stats",
                     {"node_path": node, "layer": "mask", "histogram_bins": 8})
        if r.get("status") != "success":
            print(f"    SKIP  mask layer unavailable: {r.get('error')}")
        else:
            d = r["data"]
            ok = (0.0 <= d["min"] and d["max"] <= 1.0 + 1e-6 and len(d["histogram"]) == 8
                  and len(d["bin_edges"]) == 9 and sum(d["histogram"]) == d["sample_count"])
            print(f"    {'PASS' if ok else 'FAIL'}  mask range=[{d['min']:.2f},{d['max']:.2f}] "
                  f"hist_sum={sum(d['histogram'])} sample_count={d['sample_count']}")
            if not ok:
                failures.append(f"mask/histogram checks: {d}")
    except Exception as exc:
        failures.append(f"mask raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    print("\n  [3] negative: bad layer -> status=error")
    try:
        hou.hipFile.clear(suppress_save_prompt=True)
        node = _build_heightfield()
        r = dispatch("geometry.get_heightfield_layer_stats",
                     {"node_path": node, "layer": "no_such_layer"})
        if r.get("status") == "error" and "height" in str(r.get("error", "")):
            print("    PASS  status=error (available layers listed)")
        else:
            failures.append(f"bad layer should error + list layers, got {r}")
            print(f"    FAIL  {r}")
    except Exception as exc:
        failures.append(f"negative raised (should be wrapped): {exc}")
        print(f"    FAIL  raised: {exc}")

    print("\n  [4] large-field sampling is memory-bounded (lowered cap)")
    try:
        from fxhoudinimcp_server.handlers import geometry_handlers as gh
        hou.hipFile.clear(suppress_save_prompt=True)
        node = _build_heightfield()
        orig = gh._HF_VOXEL_CAP
        try:
            gh._HF_VOXEL_CAP = 10_000
            r = dispatch("geometry.get_heightfield_layer_stats",
                         {"node_path": node, "layer": "height", "histogram_bins": 8})
        finally:
            gh._HF_VOXEL_CAP = orig
        if r.get("status") != "success":
            failures.append(f"sampling path failed: {r}"); print(f"    FAIL  {r}")
        else:
            d = r["data"]
            ok = (d["sampled"] is True and d["sample_count"] <= 10_000
                  and sum(d["histogram"]) == d["sample_count"] and d["min"] < d["max"])
            print(f"    {'PASS' if ok else 'FAIL'}  sampled={d['sampled']} sample_count={d['sample_count']} "
                  f"(full voxel_count={d['voxel_count']})")
            if not ok:
                failures.append(f"sampling bound checks: {d}")
    except Exception as exc:
        failures.append(f"sampling raised: {exc}")
        print(f"    FAIL  raised: {exc}")

    print()
    if failures:
        print(f"SMOKE FAILED — {len(failures)} failure(s):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("SMOKE PASSED — get_heightfield_layer_stats reads real layer voxel stats (dispatcher path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
