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
        If ``peak_value <= 0``, or if ``a`` and ``b`` have a different number of
        channels, or if any corresponding channel has a different number of pixels.
    """
    # RR-2: peak_value must be positive; guard before any computation.
    if peak_value <= 0:
        raise ValueError(
            f"compute_plane_delta: peak_value must be > 0, got {peak_value}"
        )

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
    # RR-1: total_finite_count tracks only finite-sample pairs so the denominator
    # of MAE and MSE is the finite-sample count, not the full buffer length.
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_finite_count = 0  # RR-1: count only finite-sample pairs
    has_nonfinite = False  # F3: flag set when any pixel is NaN or Inf

    for ch in range(num_channels):
        ch_a = a[ch]
        ch_b = b[ch]

        ch_sum_signed = 0.0
        ch_max_abs = 0.0
        ch_sum_abs = 0.0
        ch_sum_sq = 0.0
        ch_finite_count = 0  # RR-1: per-channel finite-sample count

        for px in range(len(ch_a)):
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
            ch_finite_count += 1

        total_abs_error += ch_sum_abs
        total_sq_error += ch_sum_sq
        total_finite_count += ch_finite_count  # RR-1: accumulate finite-only count

        # RR-1: divide signed mean by finite-sample count per channel, not full length
        mean_delta.append(ch_sum_signed / ch_finite_count if ch_finite_count > 0 else 0.0)
        max_abs_delta.append(ch_max_abs)

    # MAE over all channels and pixels (finite samples only — RR-1 fix)
    mae = total_abs_error / total_finite_count if total_finite_count > 0 else 0.0

    # PSNR: 10 * log10(peak_value**2 / mse) — F4 adds peak_value param
    mse = total_sq_error / total_finite_count if total_finite_count > 0 else 0.0
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


def build_compare(
    aovs_a: List[str],
    aovs_b: List[str],
    channels_a: Dict[str, List[List[float]]],
    channels_b: Dict[str, List[List[float]]],
    planes: Optional[List[str]] = None,
    metric: str = "stats",
    peak_value: float = 1.0,
) -> dict:
    """Orchestrate a full render comparison and return the §4.2 envelope.

    Reuses ``aov_presence_diff``, ``compute_plane_delta``, ``build_verdict``,
    and ``CompareReport.to_dict()`` — does NOT re-implement their logic.

    Parameters
    ----------
    aovs_a : List[str]
        AOV names present in render A (order-stable).
    aovs_b : List[str]
        AOV names present in render B (order-stable).
    channels_a : Dict[str, List[List[float]]]
        Per-AOV channel buffers for render A.
        ``channels_a[aov][channel_index][pixel_index]``.
    channels_b : Dict[str, List[List[float]]]
        Per-AOV channel buffers for render B.
    planes : Optional[List[str]]
        Which planes to include in ``per_plane``.
        ``None`` (default) -> all common AOVs.
        A list -> intersection with common AOVs, in requested order.
        A plane in the list that is not in the common set is silently skipped.
        A SELECTED plane that is present in both AOV lists (i.e. in ``common``)
        but absent from ``channels_a`` or ``channels_b`` raises ``ValueError``
        (rev2 contract — callers must supply channel data for every selected
        plane; silent skipping would yield a false "no-change" verdict).
    metric : str
        One of ``"stats"``, ``"mae"``, or ``"psnr"``.

        **v1 client-projection hint.** This parameter does NOT change the
        returned shape: all three values always return the full ``per_plane``
        list with every field (``plane``, ``mean_delta``, ``max_abs_delta``,
        ``mae``, ``psnr``, ``moved``).  ``metric`` is validated and reserved
        for future server-side field filtering in a later API version.
    peak_value : float
        Signal peak forwarded to every ``compute_plane_delta`` call.
        Must be > 0 (enforced by ``compute_plane_delta``).

        **PSNR meaningfulness caveat.** PSNR is well-defined only for planes
        whose pixel values are normalised to ``[0, peak_value]``.  The default
        ``peak_value=1.0`` gives sensible results for beauty (C), normals (N),
        alpha (A), and other [0, 1]-range AOVs.  For unbounded planes — depth
        (``Pz``), world-space position (``P``), motion vectors, object/material
        IDs — the global ``peak_value=1.0`` produces *misleadingly low* dB
        values because real pixel magnitudes far exceed 1.0.  Supply a
        per-plane peak map at the call site (e.g. ``peak_value=100.0`` for a
        depth plane clipped at 100 scene units) or treat ``psnr`` as
        informational-only for those planes.  A future v2 API will accept a
        per-plane peak map directly.

    Returns
    -------
    dict
        ``CompareReport.to_dict()`` shape::

            {
              "aovs_only_in_a": list[str],
              "aovs_only_in_b": list[str],
              "aovs_common":    list[str],
              "per_plane": [
                {
                  "plane":         str,
                  "mean_delta":    list[float],
                  "max_abs_delta": list[float],
                  "mae":           float,
                  "psnr":          float | None,
                  "moved":         bool,
                },
                ...
              ],
              "verdict": str,
            }

        **All-non-finite sentinel.** When a selected plane has NO finite pixel
        samples in both renders, ``compute_plane_delta`` returns the triple
        ``moved=True``, ``mae=0.0``, ``psnr=None``.  This sentinel means
        *"no numeric comparison was possible"* — it MUST NOT be read as
        "renders are identical/unchanged".  It is distinct from an unchanged
        plane (``moved=False``, ``mae=0.0``, ``psnr=None``).

        **Finite-count weighting.** ``mae`` and the MSE used to compute
        ``psnr`` are averaged over the *global* finite-sample count across all
        channels and pixels; non-finite samples (NaN, Inf) are excluded from
        both numerator and denominator (RR-1 fix).  ``mean_delta`` per channel
        uses that channel's *own* finite count — not the global total.

    Raises
    ------
    ValueError
        If ``metric`` is not one of ``"stats"``, ``"mae"``, ``"psnr"``, or if
        a SELECTED plane is absent from ``channels_a`` or ``channels_b``.
    """
    _VALID_METRICS = {"stats", "mae", "psnr"}
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"build_compare: unknown metric {metric!r}; "
            f"must be one of {sorted(_VALID_METRICS)}"
        )

    # Step 1: determine AOV presence sets via the existing primitive.
    only_in_a, only_in_b, common = aov_presence_diff(aovs_a, aovs_b)

    # Step 2: determine which planes to delta.
    if planes is None:
        selected = common
    else:
        common_set = set(common)
        selected = [p for p in planes if p in common_set]

    # Step 3: compute per-plane deltas.
    # rev2: a SELECTED plane absent from either channels dict raises ValueError
    # (silent skip would yield a false "no-change" verdict with no signal).
    per_plane: List[PlaneDelta] = []
    for plane_name in selected:
        if plane_name not in channels_a or plane_name not in channels_b:
            raise ValueError(
                f"build_compare: plane '{plane_name}' is selected but absent "
                f"from the channel dict(s) — ensure channel data is provided "
                f"for every selected plane"
            )
        delta = compute_plane_delta(
            channels_a[plane_name],
            channels_b[plane_name],
            plane_name,
            peak_value=peak_value,
        )
        per_plane.append(delta)

    # Step 4: assemble the CompareReport and populate the verdict.
    report = CompareReport(
        aovs_only_in_a=only_in_a,
        aovs_only_in_b=only_in_b,
        aovs_common=common,
        per_plane=per_plane,
        verdict="",
    )
    report.verdict = build_verdict(report)

    # Step 5: return the §4.2 locked shape via CompareReport.to_dict().
    return report.to_dict()


def build_readback(
    channels: List[List[float]],
    plane: str,
    xres: int,
    yres: int,
    dtype: str = "float32",
    mode: str = "summary",
    roi: Optional[List[int]] = None,
    max_pixels: int = 4096,
    downsample: int = 1,
    page: int = 0,
    page_size: int = 1024,
    num_bins: int = 16,
) -> dict:
    """Assemble the §4.2 render_read_pixels response dict from per-channel pixel buffers.

    Pure Python — no hou, no numpy, no OIIO imports.  Off-DCC, pytest-able.

    Parameters
    ----------
    channels : List[List[float]]
        Per-channel flat pixel buffers: ``channels[ch][pixel_index]``.
        Pixel index = ``y * xres + x`` (row-major).
    plane : str
        Name of the AOV plane (e.g. "C", "N", "depth").
    xres : int
        Horizontal resolution of the SOURCE frame (ALWAYS reflected verbatim in output).
    yres : int
        Vertical resolution of the SOURCE frame (ALWAYS reflected verbatim in output).
    dtype : str
        Data type string (e.g. "float32", "float16").
    mode : str
        One of "summary", "roi", or "sample".
        - "summary": return stats + histogram; pixels=[].
        - "roi": crop to ``roi`` rect, then paginate.
        - "sample": full frame strided by ``downsample``, then paginate.
    roi : Optional[List[int]]
        ``[x0, y0, x1, y1]`` — half-open rect (x1/y1 are exclusive).
        Bounds are clamped to ``[0, xres)`` / ``[0, yres)``.
        ``None`` in "roi" mode defaults to the full frame ``[0, 0, xres, yres]``.
        Ignored in "summary" and "sample" modes.
    max_pixels : int
        Maximum total pixels to paginate (clamping limit for clamp_and_paginate).
    downsample : int
        Stride for "sample" mode.  Values < 1 are treated as 1.
    page : int
        Zero-based page index (for "roi" and "sample" modes).
    page_size : int
        Maximum pixels per page (for "roi" and "sample" modes).
    num_bins : int
        Number of histogram bins for summarize_pixels.

    Returns
    -------
    dict
        §4.2 locked shape::

            {
              'plane':       str,
              'xres':        int,   # SOURCE dims, never roi dims
              'yres':        int,
              'channels':    int,
              'dtype':       str,
              'mode':        str,
              'stats':       {'min':[..], 'max':[..], 'mean':[..],
                              'nan_count':int, 'inf_count':int},
              'histogram':   {'bins':int, 'counts':[[..],..] },
              'pixels':      [[..], ..],   # [] in 'summary' mode
              'page':        int,
              'page_size':   int,
              'total_pages': int,
              'truncated':   bool,
            }
    """
    # ------------------------------------------------------------------
    # Empty-channels fast path (AC-4) — always a valid zero response.
    # ------------------------------------------------------------------
    if not channels:
        return {
            "plane": plane,
            "xres": xres,
            "yres": yres,
            "channels": 0,
            "dtype": dtype,
            "mode": mode,
            "stats": {
                "min": [],
                "max": [],
                "mean": [],
                "nan_count": 0,
                "inf_count": 0,
            },
            "histogram": {"bins": num_bins, "counts": []},
            "pixels": [],
            "page": page,
            "page_size": page_size,
            "total_pages": 1,
            "truncated": False,
        }

    # ------------------------------------------------------------------
    # Mode dispatch.
    # ------------------------------------------------------------------

    if mode == "summary":
        # Stats over all pixels; no pixel rows returned.
        summary = summarize_pixels(channels, plane, xres, yres, dtype, num_bins)
        stats_dict = dict(summary.stats)
        hist_dict = {
            "bins": summary.histogram["bins"],
            "counts": [list(c) for c in summary.histogram["counts"]],
        }
        return {
            "plane": plane,
            "xres": xres,
            "yres": yres,
            "channels": len(channels),
            "dtype": dtype,
            "mode": mode,
            "stats": stats_dict,
            "histogram": hist_dict,
            "pixels": [],
            "page": page,
            "page_size": page_size,
            "total_pages": 1,
            "truncated": False,
        }

    elif mode == "roi":
        # Crop to roi rect (bounds clamped), then paginate.
        if roi is None:
            x0, y0, x1, y1 = 0, 0, xres, yres
        else:
            x0 = max(0, roi[0])
            y0 = max(0, roi[1])
            x1 = min(xres, roi[2])
            y1 = min(yres, roi[3])

        # M-02: guard against undersized channel buffers before indexing.
        if any(len(ch) < xres * yres for ch in channels):
            raise ValueError(
                f"build_readback: channel buffer too short for {xres}x{yres} frame "
                f"(mode={mode!r})"
            )

        # Build pixel rows for the cropped region (row-major traversal).
        pixel_rows: List[List[float]] = []
        for row_y in range(y0, y1):
            for col_x in range(x0, x1):
                flat_idx = row_y * xres + col_x
                pixel_rows.append([ch[flat_idx] for ch in channels])

        # Rebuild per-channel buffers for the roi pixels for stats.
        roi_channels: List[List[float]] = [[] for _ in range(len(channels))]
        for row in pixel_rows:
            for ch_i, val in enumerate(row):
                roi_channels[ch_i].append(val)

        summary = summarize_pixels(roi_channels, plane, xres, yres, dtype, num_bins)
        stats_dict = dict(summary.stats)
        hist_dict = {
            "bins": summary.histogram["bins"],
            "counts": [list(c) for c in summary.histogram["counts"]],
        }

        rb_page = clamp_and_paginate(pixel_rows, page, page_size, max_pixels)

        return {
            "plane": plane,
            "xres": xres,     # AC-5: source dims, never roi dims
            "yres": yres,
            "channels": len(channels),
            "dtype": dtype,
            "mode": mode,
            "stats": stats_dict,
            "histogram": hist_dict,
            "pixels": rb_page.pixels,
            "page": rb_page.page,
            "page_size": rb_page.page_size,
            "total_pages": rb_page.total_pages,
            "truncated": rb_page.truncated,
        }

    elif mode == "sample":
        # Full frame with stride = max(downsample, 1).
        stride = max(downsample, 1)

        # M-02: guard against undersized channel buffers before indexing.
        if any(len(ch) < xres * yres for ch in channels):
            raise ValueError(
                f"build_readback: channel buffer too short for {xres}x{yres} frame "
                f"(mode={mode!r})"
            )

        # Walk the source frame in stride steps on both axes.
        pixel_rows_s: List[List[float]] = []
        for row_y in range(0, yres, stride):
            for col_x in range(0, xres, stride):
                flat_idx = row_y * xres + col_x
                pixel_rows_s.append([ch[flat_idx] for ch in channels])

        # Rebuild per-channel buffers for the strided pixels for stats.
        sample_channels: List[List[float]] = [[] for _ in range(len(channels))]
        for row in pixel_rows_s:
            for ch_i, val in enumerate(row):
                sample_channels[ch_i].append(val)

        summary = summarize_pixels(sample_channels, plane, xres, yres, dtype, num_bins)
        stats_dict = dict(summary.stats)
        hist_dict = {
            "bins": summary.histogram["bins"],
            "counts": [list(c) for c in summary.histogram["counts"]],
        }

        rb_page = clamp_and_paginate(pixel_rows_s, page, page_size, max_pixels)

        return {
            "plane": plane,
            "xres": xres,     # AC-5: source dims always
            "yres": yres,
            "channels": len(channels),
            "dtype": dtype,
            "mode": mode,
            "stats": stats_dict,
            "histogram": hist_dict,
            "pixels": rb_page.pixels,
            "page": rb_page.page,
            "page_size": rb_page.page_size,
            "total_pages": rb_page.total_pages,
            "truncated": rb_page.truncated,
        }

    else:
        raise ValueError(f"build_readback: unknown mode {mode!r}")
