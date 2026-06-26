"""render_readback_model.py — pure-logic model for render readback and comparison.

No hou, no numpy, no Qt, no pxr imports. Plain Python only (math, dataclasses,
typing, standard library). This module is pytest-able off-DCC.

Public API
----------
Dataclasses
    PlaneDelta       — per-plane comparison result
    CompareReport    — full AOV comparison report
    PixelSummary     — statistical summary of a pixel buffer
    ReadbackPage     — one page of pixel readback

Functions
    compute_plane_delta    — compare two channel buffers for one AOV plane
    aov_presence_diff      — find AOVs only in A, only in B, and common to both
    build_verdict          — human-readable verdict string from a CompareReport
    summarize_pixels       — compute statistics and histogram for a pixel buffer
    clamp_and_paginate     — clamp and paginate a list of pixel rows
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PlaneDelta:
    """Per-plane comparison result between two renders.

    Parameters
    ----------
    plane : str
        Name of the AOV plane (e.g. "C", "N", "depth").
    mean_delta : List[float]
        Signed mean difference per channel.
    max_abs_delta : List[float]
        Maximum absolute difference per channel.
    mae : float
        Mean absolute error over all channels and pixels.
    psnr : float
        Peak signal-to-noise ratio (float('inf') when mae == 0).
    moved : bool
        True when the plane has non-zero differences (mae > 0).
    """

    plane: str
    mean_delta: List[float]
    max_abs_delta: List[float]
    mae: float
    psnr: float
    moved: bool

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict.

        ``psnr`` is emitted as ``None`` when its value is non-finite
        (``float('inf')`` or ``float('nan')``) because JSON does not support
        either; ``null`` is the unambiguous sentinel.
        """
        psnr_val = None if (math.isinf(self.psnr) or math.isnan(self.psnr)) else self.psnr
        return {
            "plane": self.plane,
            "mean_delta": list(self.mean_delta),
            "max_abs_delta": list(self.max_abs_delta),
            "mae": self.mae,
            "psnr": psnr_val,
            "moved": self.moved,
        }

    @staticmethod
    def from_dict(d: dict) -> "PlaneDelta":
        psnr = d.get("psnr")
        if psnr is None:
            psnr = math.inf
        return PlaneDelta(
            plane=d["plane"],
            mean_delta=list(d["mean_delta"]),
            max_abs_delta=list(d["max_abs_delta"]),
            mae=float(d["mae"]),
            psnr=float(psnr),
            moved=bool(d["moved"]),
        )


@dataclass
class CompareReport:
    """Full AOV comparison report.

    Parameters
    ----------
    aovs_only_in_a : List[str]
        AOVs present in render A but not in render B.
    aovs_only_in_b : List[str]
        AOVs present in render B but not in render A.
    aovs_common : List[str]
        AOVs present in both renders.
    per_plane : List[PlaneDelta]
        Per-plane deltas for each common AOV.
    verdict : str
        Human-readable summary of changes.
    """

    aovs_only_in_a: List[str] = field(default_factory=list)
    aovs_only_in_b: List[str] = field(default_factory=list)
    aovs_common: List[str] = field(default_factory=list)
    per_plane: List[PlaneDelta] = field(default_factory=list)
    verdict: str = ""

    def to_dict(self) -> dict:
        return {
            "aovs_only_in_a": list(self.aovs_only_in_a),
            "aovs_only_in_b": list(self.aovs_only_in_b),
            "aovs_common": list(self.aovs_common),
            "per_plane": [p.to_dict() for p in self.per_plane],
            "verdict": self.verdict,
        }

    @staticmethod
    def from_dict(d: dict) -> "CompareReport":
        return CompareReport(
            aovs_only_in_a=list(d.get("aovs_only_in_a", [])),
            aovs_only_in_b=list(d.get("aovs_only_in_b", [])),
            aovs_common=list(d.get("aovs_common", [])),
            per_plane=[PlaneDelta.from_dict(p) for p in d.get("per_plane", [])],
            verdict=str(d.get("verdict", "")),
        )


@dataclass
class PixelSummary:
    """Statistical summary of a pixel buffer.

    Parameters
    ----------
    plane : str
        Name of the AOV plane.
    xres : int
        Horizontal resolution.
    yres : int
        Vertical resolution.
    channels : int
        Number of channels per pixel.
    dtype : str
        Data type string (e.g. "float32").
    stats : Dict[str, float]
        Per-buffer statistics: min, max, mean, nan_count, inf_count.
    histogram : Dict[str, list]
        Histogram with keys "bins" (edge values) and "counts" (per-bin counts).
        One entry per channel.
    """

    plane: str
    xres: int
    yres: int
    channels: int
    dtype: str
    stats: Dict[str, float] = field(default_factory=dict)
    histogram: Dict[str, list] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "plane": self.plane,
            "xres": self.xres,
            "yres": self.yres,
            "channels": self.channels,
            "dtype": self.dtype,
            "stats": dict(self.stats),
            "histogram": {
                "bins": self.histogram["bins"],
                "counts": [list(c) for c in self.histogram["counts"]],
            },
        }


@dataclass
class ReadbackPage:
    """One page of pixel readback.

    Parameters
    ----------
    pixels : List[List[float]]
        Pixel rows in this page; each row is a list of channel values.
    page : int
        Zero-based page index.
    page_size : int
        Number of pixels per page.
    total_pages : int
        Total number of pages after clamping.
    truncated : bool
        True when the original pixel list was longer than max_pixels.
    """

    pixels: List[List[float]] = field(default_factory=list)
    page: int = 0
    page_size: int = 100
    total_pages: int = 1
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "pixels": [list(row) for row in self.pixels],
            "page": self.page,
            "page_size": self.page_size,
            "total_pages": self.total_pages,
            "truncated": self.truncated,
        }


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_plane_delta(
    a: List[List[float]],
    b: List[List[float]],
    plane: str,
    peak_value: float = 1.0,
) -> PlaneDelta:
    """Compare two channel buffers for one AOV plane.

    Both ``a`` and ``b`` are lists indexed by channel, where each element is a
    flat list of pixel values: ``a[channel_index][pixel_index]``.

    Parameters
    ----------
    a : List[List[float]]
        Channel buffers from render A.
    b : List[List[float]]
        Channel buffers from render B.
    plane : str
        Name of the AOV plane being compared.
    peak_value : float, optional
        Signal peak used in the PSNR formula: ``10 * log10(peak_value**2 / mse)``.
        Default 1.0 preserves behaviour for [0, 1]-range AOVs (beauty, normals).
        Use the actual peak for HDR, depth, or position AOVs (e.g. 100.0).

    Returns
    -------
    PlaneDelta
        Comparison result for this plane.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have a different number of channels, or if any
        corresponding channel has a different number of pixels.
    """
    num_channels = len(a)

    # F5: validate shape before any computation
    if len(b) != num_channels:
        raise ValueError(
            f"compute_plane_delta: channel count mismatch for plane '{plane}': "
            f"len(a)={num_channels} != len(b)={len(b)}"
        )
    for ch in range(num_channels):
        if len(a[ch]) != len(b[ch]):
            raise ValueError(
                f"compute_plane_delta: pixel count mismatch for plane '{plane}' "
                f"channel {ch}: len(a[{ch}])={len(a[ch])} != len(b[{ch}])={len(b[ch])}"
            )

    if num_channels == 0:
        return PlaneDelta(
            plane=plane,
            mean_delta=[],
            max_abs_delta=[],
            mae=0.0,
            psnr=math.inf,
            moved=False,
        )

    # Per-channel metrics
    mean_delta: List[float] = []
    max_abs_delta: List[float] = []

    # Accumulate absolute errors across all channels and pixels for MAE and MSE.
    # F3/F1: non-finite pixels (NaN or Inf) are excluded from mae/psnr but their
    # presence is tracked to set moved=True.
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_count = 0
    has_nonfinite = False  # F3: flag set when any pixel is NaN or Inf

    for ch in range(num_channels):
        ch_a = a[ch]
        ch_b = b[ch]
        n = len(ch_a)

        ch_sum_signed = 0.0
        ch_max_abs = 0.0
        ch_sum_abs = 0.0
        ch_sum_sq = 0.0

        for px in range(n):
            va = ch_a[px]
            vb = ch_b[px]
            # F3/F1: exclude non-finite samples from numeric aggregation
            if not math.isfinite(va) or not math.isfinite(vb):
                has_nonfinite = True
                continue
            diff = va - vb
            abs_diff = abs(diff)
            ch_sum_signed += diff
            ch_sum_abs += abs_diff
            ch_sum_sq += diff * diff
            if abs_diff > ch_max_abs:
                ch_max_abs = abs_diff

        total_abs_error += ch_sum_abs
        total_sq_error += ch_sum_sq
        total_count += n

        mean_delta.append(ch_sum_signed / n if n > 0 else 0.0)
        max_abs_delta.append(ch_max_abs)

    # MAE over all channels and pixels (finite samples only)
    mae = total_abs_error / total_count if total_count > 0 else 0.0

    # PSNR: 10 * log10(peak_value**2 / mse) — F4 adds peak_value param
    mse = total_sq_error / total_count if total_count > 0 else 0.0
    if mse == 0.0:
        psnr = math.inf
    elif not math.isfinite(mse):
        # F1: mse=inf when squared diffs overflow; PSNR is undefined
        psnr = 0.0
    else:
        psnr = 10.0 * math.log10(peak_value ** 2 / mse)

    # F3: moved=True when any non-finite sample is present OR finite mae > 0
    moved = has_nonfinite or mae > 0.0

    return PlaneDelta(
        plane=plane,
        mean_delta=mean_delta,
        max_abs_delta=max_abs_delta,
        mae=mae,
        psnr=psnr,
        moved=moved,
    )


def aov_presence_diff(
    aovs_a: List[str],
    aovs_b: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Find AOVs only in A, only in B, and common to both.

    Order is preserved: results respect the original list order of ``aovs_a``
    and ``aovs_b`` respectively.

    Parameters
    ----------
    aovs_a : List[str]
        AOV names from render A.
    aovs_b : List[str]
        AOV names from render B.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        ``(only_in_a, only_in_b, common)`` — each is order-stable from the
        original list.
    """
    set_b = set(aovs_b)
    set_a = set(aovs_a)

    only_in_a = [aov for aov in aovs_a if aov not in set_b]
    only_in_b = [aov for aov in aovs_b if aov not in set_a]
    common = [aov for aov in aovs_a if aov in set_b]

    return only_in_a, only_in_b, common


def build_verdict(report: CompareReport) -> str:
    """Build a human-readable verdict string from a CompareReport.

    Parameters
    ----------
    report : CompareReport
        The comparison report to summarise.

    Returns
    -------
    str
        A plain-English description of what changed between the two renders.
    """
    parts: List[str] = []

    # Per-plane changes
    for delta in report.per_plane:
        if delta.moved:
            parts.append(
                f"{delta.plane} changed (mae {delta.mae:.3f})"
            )
        else:
            parts.append(f"{delta.plane} unchanged")

    # Lost AOVs (only in A)
    for aov in report.aovs_only_in_a:
        parts.append(f"lost AOV {aov}")

    # Gained AOVs (only in B)
    for aov in report.aovs_only_in_b:
        parts.append(f"gained AOV {aov}")

    if not parts:
        return "no change"

    # Check if anything actually changed
    any_change = (
        any(d.moved for d in report.per_plane)
        or bool(report.aovs_only_in_a)
        or bool(report.aovs_only_in_b)
    )

    if not any_change and report.per_plane:
        return "no change"

    return "; ".join(parts)


def summarize_pixels(
    channels: List[List[float]],
    plane: str,
    xres: int,
    yres: int,
    dtype: str = "float32",
    num_bins: int = 16,
) -> PixelSummary:
    """Compute statistics and histogram for a pixel buffer.

    Parameters
    ----------
    channels : List[List[float]]
        Per-channel flat pixel buffers: ``channels[ch][pixel_index]``.
    plane : str
        Name of the AOV plane.
    xres : int
        Horizontal resolution.
    yres : int
        Vertical resolution.
    dtype : str
        Data type string.
    num_bins : int
        Number of histogram bins per channel.

    Returns
    -------
    PixelSummary
        Summary including min, max, mean, nan_count, inf_count, and per-channel
        histograms.
    """
    if not channels:
        return PixelSummary(
            plane=plane,
            xres=xres,
            yres=yres,
            channels=0,
            dtype=dtype,
            stats={
                "min": [],
                "max": [],
                "mean": [],
                "nan_count": 0,
                "inf_count": 0,
            },
            histogram={"bins": num_bins, "counts": []},
        )

    # Per-channel statistics
    ch_mins: List[float] = []
    ch_maxs: List[float] = []
    ch_means: List[float] = []
    nan_count = 0
    inf_count = 0

    for ch_buf in channels:
        ch_min = math.inf
        ch_max = -math.inf
        ch_sum = 0.0
        ch_clean = 0

        for v in ch_buf:
            if math.isnan(v):
                nan_count += 1
            elif math.isinf(v):
                inf_count += 1
            else:
                if v < ch_min:
                    ch_min = v
                if v > ch_max:
                    ch_max = v
                ch_sum += v
                ch_clean += 1

        if ch_clean == 0:
            ch_min = 0.0
            ch_max = 0.0
            ch_mean = 0.0
        else:
            ch_mean = ch_sum / ch_clean

        ch_mins.append(ch_min if not math.isinf(ch_min) else 0.0)
        ch_maxs.append(ch_max if not math.isinf(ch_max) else 0.0)
        ch_means.append(ch_mean)

    stats: Dict[str, object] = {
        "min": ch_mins,
        "max": ch_maxs,
        "mean": ch_means,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }

    # Per-channel histograms
    # Use a fixed range [0, 1] for histogram binning (standard for render AOVs)
    hist_min = 0.0
    hist_max = 1.0
    bin_width = (hist_max - hist_min) / num_bins

    # Per-channel counts — histogram["bins"] is an INTEGER (the count)
    all_counts: List[List[int]] = []
    for ch_buf in channels:
        counts = [0] * num_bins
        for v in ch_buf:
            if math.isnan(v) or math.isinf(v):
                continue
            # Clamp to bin range
            idx = int((v - hist_min) / bin_width)
            if idx < 0:
                idx = 0
            elif idx >= num_bins:
                idx = num_bins - 1
            counts[idx] += 1
        all_counts.append(counts)

    histogram: Dict[str, object] = {
        "bins": num_bins,   # INTEGER — the bin count, not bin edges
        "counts": all_counts,
    }

    return PixelSummary(
        plane=plane,
        xres=xres,
        yres=yres,
        channels=len(channels),
        dtype=dtype,
        stats=stats,
        histogram=histogram,
    )


def clamp_and_paginate(
    pixels: List[List[float]],
    page: int,
    page_size: int,
    max_pixels: int,
) -> ReadbackPage:
    """Clamp and paginate a list of pixel rows.

    Parameters
    ----------
    pixels : List[List[float]]
        Full list of pixel rows; each row is a list of channel values.
    page : int
        Zero-based page index to return.
    page_size : int
        Maximum number of pixels per page.
    max_pixels : int
        Maximum total pixels to consider (clamping limit).

    Returns
    -------
    ReadbackPage
        The requested page of pixel data, including pagination metadata.
    """
    truncated = len(pixels) > max_pixels
    effective = pixels[:max_pixels] if truncated else pixels
    effective_len = len(effective)

    if page_size <= 0:
        page_size = 1

    total_pages = math.ceil(effective_len / page_size) if effective_len > 0 else 1

    start = page * page_size
    end = start + page_size
    page_pixels = effective[start:end]

    return ReadbackPage(
        pixels=page_pixels,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        truncated=truncated,
    )
