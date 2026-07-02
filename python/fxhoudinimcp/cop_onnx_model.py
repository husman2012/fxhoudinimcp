"""
cop_onnx_model.py — pure-logic core for the Copernicus-ONNX MCP surface
(PP12-113 PR-1).

Contract: imports NO hou, NO Qt/PySide6, NO pxr, NO numpy, NO MaterialX.
Plain Python stdlib only (dataclasses, math, typing). Pytest-able off-DCC
(CL-015) — this module never touches Houdini, PySide6, USD, or numpy, so
it runs on bare CI with plain pytest.

This is the H22-STABLE surface (API-agnostic layout-guessing + pagination
math per the 2026-07-01 research): no Houdini coupling, so it is the safe
unblocked first move for the Copernicus-ONNX member (PP12-113). Handlers,
@mcp.tool wrappers, and any hou-layer code are out of scope for this PR.

Classes
-------
TensorSpec    — a single ONNX tensor's name/shape/dtype/layout-guess
OnnxContract  — the full loaded-model contract (inputs/outputs/opset/...)
PixelSummary  — per-plane pixel statistics + histogram (nan/inf-honest)
ReadbackPage  — one paginated page of pixel readback

Functions
---------
guess_layout(shape) -> str
    Heuristic NCHW/NHWC/unknown layout guess from a tensor shape.
clamp_readback(xres, yres, channels, max_pixels) -> int
    Server-side stride computation keeping a readback within a pixel budget.
paginate(total_items, page, page_size) -> dict
    Page-boundary math for slicing a flat pixel/item list.
count_nan_inf(values) -> dict
    NaN/Inf-honest stats helper: counts NaN/Inf without dropping them,
    computes min/max/mean over the finite values only.

Cross-references
-----------------
Plan pp12-113a lockedFieldContract (BINDING)
spec.md §7.2 (data model), §9.1 (unit tests)
CL-015: pure-logic module, no hou/Qt/pxr/numpy/MaterialX
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Union


# ---------------------------------------------------------------------------
# TensorSpec
# ---------------------------------------------------------------------------

@dataclass
class TensorSpec:
    """A single ONNX tensor's name/shape/dtype/layout-guess.

    Fields
    ------
    name         : str   — tensor name as declared in the ONNX graph
    shape        : list  — dims, each an int OR the literal string "dynamic"
                            for a symbolic dim
    dtype        : str   — declared dtype string (e.g. "float32")
    layout_guess : str   — "NCHW" | "NHWC" | "unknown"; meaningful for INPUT
                            tensors only. Defaults to "unknown".
    """

    name: str
    shape: list
    dtype: str
    layout_guess: str = "unknown"

    def to_dict(self) -> dict:
        """Serialize to a plain dict with exactly four keys.

        `shape` is copied (not aliased) and 'dynamic' dims are preserved
        verbatim — never coerced to None/0.
        """
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "layout_guess": self.layout_guess,
        }


# ---------------------------------------------------------------------------
# OnnxContract
# ---------------------------------------------------------------------------

@dataclass
class OnnxContract:
    """The full loaded-model contract for an ONNX file.

    Fields
    ------
    model_path : str            — path to the .onnx file
    inputs     : list[TensorSpec] — input tensor specs
    outputs    : list[TensorSpec] — output tensor specs
    opset      : int | None     — ONNX opset version (None if unknown)
    producer   : str | None     — producer_name from the model metadata
    loadable   : bool           — True if the model loaded successfully
    error      : str | None     — load error message (None if loadable)
    """

    model_path: str
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    opset: Optional[int] = None
    producer: Optional[str] = None
    loadable: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to a plain dict; nested TensorSpecs via their to_dict().

        Dynamic dims in any nested TensorSpec round-trip verbatim through
        this call (list copy at TensorSpec.to_dict, not by reference).
        """
        return {
            "model_path": self.model_path,
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "opset": self.opset,
            "producer": self.producer,
            "loadable": self.loadable,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# PixelSummary
# ---------------------------------------------------------------------------

@dataclass
class PixelSummary:
    """Per-plane pixel statistics + histogram (nan/inf-honest).

    Fields
    ------
    plane            : str        — AOV/plane name
    xres, yres       : int        — plane resolution
    channels         : int        — channel count
    dtype            : str        — declared dtype string
    min, max, mean   : list       — per-channel stats (len == channels);
                                     an all-nan/all-inf channel reports
                                     None for min/max/mean (sentinel)
    nan_count        : int        — total NaN samples across the plane
    inf_count        : int        — total +/-Inf samples across the plane
    histogram_bins   : int        — number of histogram bins per channel
    histogram_counts : list       — per-channel list of per-bin counts

    to_dict() nests min/max/mean/nan_count/inf_count under 'stats' and
    histogram_bins/histogram_counts under 'histogram' (spec §4.2 shape).
    """

    plane: str
    xres: int
    yres: int
    channels: int
    dtype: str
    min: list = field(default_factory=list)
    max: list = field(default_factory=list)
    mean: list = field(default_factory=list)
    nan_count: int = 0
    inf_count: int = 0
    histogram_bins: int = 0
    histogram_counts: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to the spec §4.2 read_pixels 'summary' nested shape."""
        return {
            "plane": self.plane,
            "xres": self.xres,
            "yres": self.yres,
            "channels": self.channels,
            "dtype": self.dtype,
            "stats": {
                "min": list(self.min),
                "max": list(self.max),
                "mean": list(self.mean),
                "nan_count": self.nan_count,
                "inf_count": self.inf_count,
            },
            "histogram": {
                "bins": self.histogram_bins,
                "counts": [list(c) for c in self.histogram_counts],
            },
        }


# ---------------------------------------------------------------------------
# ReadbackPage
# ---------------------------------------------------------------------------

_VALID_READBACK_MODES = frozenset({"summary", "roi", "sample"})


@dataclass
class ReadbackPage:
    """One paginated page of pixel readback.

    Fields
    ------
    plane       : str   — AOV/plane name
    mode        : str   — one of 'summary', 'roi', 'sample'
    pixels      : list  — paginated pixel rows (empty for 'summary' mode)
    page        : int   — zero-based page index
    page_size   : int   — requested page size
    total_pages : int   — total page count after clamping
    truncated   : bool  — True when more pages remain / data was clamped

    Raises
    ------
    ValueError (M-4)
        Raised at construction time if `mode` is not one of
        'summary', 'roi', 'sample'.
    """

    plane: str
    mode: str
    pixels: list = field(default_factory=list)
    page: int = 0
    page_size: int = 0
    total_pages: int = 1
    truncated: bool = False

    def __post_init__(self) -> None:
        """M-4: validate mode at construction time."""
        if self.mode not in _VALID_READBACK_MODES:
            raise ValueError(
                f"ReadbackPage.mode must be one of {sorted(_VALID_READBACK_MODES)!r}, "
                f"got {self.mode!r}"
            )

    def to_dict(self) -> dict:
        """Serialize to a plain dict with exactly seven keys."""
        return {
            "plane": self.plane,
            "mode": self.mode,
            "pixels": [list(row) if isinstance(row, list) else row for row in self.pixels],
            "page": self.page,
            "page_size": self.page_size,
            "total_pages": self.total_pages,
            "truncated": self.truncated,
        }


# ---------------------------------------------------------------------------
# guess_layout
# ---------------------------------------------------------------------------

_CHANNEL_CANDIDATES = (1, 3, 4)


def _is_large_or_dynamic(dim: Union[int, str]) -> bool:
    """A dim counts as a plausible image dim if it's 'dynamic' or a large int.

    'dynamic' always counts as large/image (per lockedFieldContract).
    A non-negative int is treated as large when it is NOT itself a
    plausible channel-count value (1, 3, 4) — this keeps a shape like
    [1, 3, 3, 3] from spuriously matching both layouts, and degrades to
    'unknown' via the caller's ambiguity check instead.
    """
    if dim == "dynamic":
        return True
    if not isinstance(dim, int):
        return False
    if dim <= 0:
        return False
    return dim not in _CHANNEL_CANDIDATES


def guess_layout(shape: list) -> str:
    """Heuristic NCHW/NHWC/unknown layout guess from a tensor shape.

    Pure. Never raises — always degrades to 'unknown' on odd/ambiguous
    shapes (empty shape, wrong rank, non-4D, all-dynamic, zero dims).

    Rule (spec.md §7.2):
      A 4D shape is 'NCHW' if shape[1] is a plausible channel count
      (1, 3, or 4) AND shape[2], shape[3] are large-or-dynamic image dims.
      It is 'NHWC' if shape[-1] is a plausible channel count AND
      shape[1], shape[2] are large-or-dynamic image dims.
      When both match, neither match, rank != 4, or the channel-count
      doesn't sit cleanly at index 1 or -1 -> 'unknown'.

    Parameters
    ----------
    shape : list
        Tensor shape: ints and/or the literal string 'dynamic'.

    Returns
    -------
    str
        'NCHW' | 'NHWC' | 'unknown'.
    """
    if not isinstance(shape, list) or len(shape) != 4:
        return "unknown"

    dim0, dim1, dim2, dim3 = shape

    is_nchw = (
        isinstance(dim1, int)
        and dim1 in _CHANNEL_CANDIDATES
        and _is_large_or_dynamic(dim2)
        and _is_large_or_dynamic(dim3)
    )
    is_nhwc = (
        isinstance(dim3, int)
        and dim3 in _CHANNEL_CANDIDATES
        and _is_large_or_dynamic(dim1)
        and _is_large_or_dynamic(dim2)
    )

    if is_nchw and not is_nhwc:
        return "NCHW"
    if is_nhwc and not is_nchw:
        return "NHWC"
    return "unknown"


# ---------------------------------------------------------------------------
# clamp_readback
# ---------------------------------------------------------------------------

def clamp_readback(xres: int, yres: int, channels: int, max_pixels: int) -> int:
    """Compute a positive integer stride keeping a readback within budget.

    Pure. Returns the smallest stride s (s >= 1) such that
    ceil(xres/s) * ceil(yres/s) <= max_pixels. Monotonic: a bigger plane
    or a smaller max_pixels budget never yields a smaller stride.

    Parameters
    ----------
    xres, yres : int
        Plane resolution. Must be positive.
    channels : int
        Channel count (accepted for API symmetry with the caller's pixel
        contract; does not affect the stride formula — the pixel-count
        budget is defined over spatial samples, not channel-scaled
        samples, per the pinned spec case).
    max_pixels : int
        Maximum allowed strided pixel count. Must be positive.

    Returns
    -------
    int
        The stride (>= 1).

    Raises
    ------
    ValueError
        If max_pixels <= 0, or xres <= 0, or yres <= 0 (fail-loud, never
        a silent 0/negative stride).
    """
    if max_pixels <= 0:
        raise ValueError(f"clamp_readback: max_pixels must be > 0, got {max_pixels}")
    if xres <= 0:
        raise ValueError(f"clamp_readback: xres must be > 0, got {xres}")
    if yres <= 0:
        raise ValueError(f"clamp_readback: yres must be > 0, got {yres}")

    stride = 1
    while math.ceil(xres / stride) * math.ceil(yres / stride) > max_pixels:
        stride += 1
    return stride


# ---------------------------------------------------------------------------
# paginate
# ---------------------------------------------------------------------------

def paginate(total_items: int, page: int, page_size: int) -> dict:
    """Compute page-boundary math for slicing a flat pixel/item list.

    Pure. total_pages is always >= 1 (1 when total_items == 0, so a
    single empty page is always valid).

    Parameters
    ----------
    total_items : int
        Total number of items to paginate.
    page : int
        Zero-based page index requested.
    page_size : int
        Maximum items per page. Must be > 0.

    Returns
    -------
    dict
        {page, page_size, total_pages, start, end, truncated}.

    Raises
    ------
    ValueError
        If page < 0, or page >= total_pages (when total_items > 0).
    """
    if page_size <= 0:
        raise ValueError(f"paginate: page_size must be > 0, got {page_size}")

    total_pages = math.ceil(total_items / page_size) if total_items > 0 else 1

    if page < 0:
        raise ValueError(f"paginate: page must be >= 0, got {page}")
    if page >= total_pages:
        raise ValueError(
            f"paginate: page {page} out of range (total_pages={total_pages})"
        )

    start = page * page_size
    end = min(start + page_size, total_items)
    truncated = (end < total_items) or (page < total_pages - 1)

    return {
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "start": start,
        "end": end,
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# count_nan_inf
# ---------------------------------------------------------------------------

def count_nan_inf(values: List[float]) -> dict:
    """NaN/Inf-honest stats helper for a flat list of floats.

    Pure. Counts NaN and +/-Inf occurrences WITHOUT dropping them from
    the reported counts (FR-6 — never silently dropped). min/max/mean
    are computed over the FINITE values only. When there are no finite
    values (all-nan, all-inf, or an empty list), min/max/mean are None
    (sentinel) rather than raising or returning a bogus 0.0.

    Parameters
    ----------
    values : list[float]
        A plain Python list of floats (no numpy array required).

    Returns
    -------
    dict
        {min, max, mean, nan_count, inf_count} — JSON-serialisable
        (None sentinels, not NaN/Inf floats, keep the result JSON-safe).
    """
    nan_count = 0
    inf_count = 0
    finite_sum = 0.0
    finite_count = 0
    finite_min: Optional[float] = None
    finite_max: Optional[float] = None

    for v in values:
        if math.isnan(v):
            nan_count += 1
            continue
        if math.isinf(v):
            inf_count += 1
            continue
        finite_sum += v
        finite_count += 1
        if finite_min is None or v < finite_min:
            finite_min = v
        if finite_max is None or v > finite_max:
            finite_max = v

    mean = (finite_sum / finite_count) if finite_count > 0 else None

    return {
        "min": finite_min,
        "max": finite_max,
        "mean": mean,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


# ---------------------------------------------------------------------------
# contract_from_setup_shapes (PP12-113 PR-2)
# ---------------------------------------------------------------------------

def contract_from_setup_shapes(
    model_path: str,
    raw_inputs: list,
    raw_outputs: list,
    opset: Optional[int] = None,
    producer: Optional[str] = None,
) -> OnnxContract:
    """Build an OnnxContract from raw input/output dicts assembled by the
    cop_onnx_inspect_model handler (PP12-113 PR-2).

    Pure. Never touches hou/Qt/pxr/onnx — the HANDLER is responsible for
    reading the cop/onnx node's parms (via the temp-node-Setup-Shapes
    mechanism) and normalizing them into plain dicts BEFORE calling this
    function. This function only maps those dicts onto TensorSpec/
    OnnxContract; it never reads a hou.Parm itself.

    Parameters
    ----------
    model_path : str
        Path to the .onnx file (forwarded verbatim into the result).
    raw_inputs : list[dict]
        One dict per input tensor: {"name": str, "dtype": str | None,
        "shape": list[int | "dynamic"]}. Dynamic dims must already be
        normalized to the literal string "dynamic" by the caller (the
        handler) — this function does not interpret any other sentinel
        (e.g. -1) as dynamic; it passes shape values through verbatim.
    raw_outputs : list[dict]
        Same shape as raw_inputs, for output tensors.
    opset : int | None
        ONNX opset version, forwarded verbatim (default None).
    producer : str | None
        Model producer_name, forwarded verbatim (default None).

    Returns
    -------
    OnnxContract
        loadable=True, error=None on every call (build failure diagnosis
        — e.g. the temp node itself erroring — is the handler's FR-5
        try/except concern, not this pure mapper's).

    Layout guessing
    ----------------
    Input TensorSpecs get layout_guess = guess_layout(shape) (the same
    pure heuristic used everywhere else in this module). Output
    TensorSpecs always get layout_guess = "unknown" — cop/onnx exposes no
    automatic layout inference for outputs, and guessing on an output
    tensor is out of scope (Section 4 of the pp12-113b research memo).
    """
    inputs = [
        TensorSpec(
            name=raw["name"],
            shape=list(raw["shape"]),
            dtype=raw.get("dtype"),
            layout_guess=guess_layout(raw["shape"]),
        )
        for raw in raw_inputs
    ]
    outputs = [
        TensorSpec(
            name=raw["name"],
            shape=list(raw["shape"]),
            dtype=raw.get("dtype"),
            layout_guess="unknown",
        )
        for raw in raw_outputs
    ]
    return OnnxContract(
        model_path=model_path,
        inputs=inputs,
        outputs=outputs,
        opset=opset,
        producer=producer,
        loadable=True,
        error=None,
    )


# ---------------------------------------------------------------------------
# choose_provider — PP12-113 PR-3
# ---------------------------------------------------------------------------

def choose_provider(requested: str, available: List[str]) -> tuple:
    """Pure requested/available Execution Provider -> (will_bind, warning) mapping.

    APPENDED for PP12-113 PR-3 (cop_onnx_set_provider). Pure — never touches
    hou/Qt/pxr. The HANDLER is responsible for reading the onnx node's
    ``provider`` parm's ``menuItems()`` at RUNTIME (platform-filtered,
    lowercase) and passing that list in as ``available``; this function only
    maps a requested token onto the runtime-available set.

    Locked contract (plan pp12-113c lockedFieldContract, PLAN-REVIEW FOLD
    m2-provider-edge):

    - ``requested`` (case-INSENSITIVE) IS in ``available``
        -> ``(requested.lower(), None)`` — no warning.
    - ``requested`` NOT in ``available`` AND ``'automatic'`` IS in ``available``
        -> ``('automatic', <non-empty warning str>)`` — the safe fallback.
    - ``requested`` NOT in ``available`` AND ``'automatic'`` NOT in ``available``
        -> ``(available[0], <non-empty warning str>)`` — first-available fallback.
    - ``available == []`` (the onnx node exposes NO Execution Provider
      options at all) -> raises ``ValueError``. This is the ONLY raise case.

    This function NEVER raises merely because ``requested`` is unavailable
    (FR-4) — raising is reserved exclusively for the ``available == []``
    case (an onnx node with zero provider menu items to choose from).

    Parameters
    ----------
    requested : str
        The caller-requested provider token (any case; matched
        case-insensitively against ``available``).
    available : list[str]
        The RUNTIME, platform-filtered ``menuItems()`` list read off the
        onnx node's ``provider`` parm (e.g.
        ``['automatic', 'cpu', 'cuda', 'directml']`` on Windows — no
        ``coreml``). Lowercase tokens, per the live Houdini 21 probe.

    Returns
    -------
    tuple[str, str | None]
        ``(will_bind, warning)`` — ``will_bind`` is always a member of
        ``available`` (never a fabricated token); ``warning`` is ``None``
        exactly when ``requested`` matched directly, else a non-empty
        human-readable string.

    Raises
    ------
    ValueError
        When ``available`` is empty — there is nothing to bind to.
    """
    if not available:
        raise ValueError(
            "choose_provider: `available` is empty — the onnx node exposes "
            "no Execution Provider options"
        )

    requested_lower = requested.strip().lower() if requested else ""
    available_lower = [a.lower() for a in available]

    if requested_lower in available_lower:
        # Bind the lowercase match exactly as it appears in `available`.
        matched = available_lower[available_lower.index(requested_lower)]
        return (matched, None)

    if "automatic" in available_lower:
        warning = (
            f"provider {requested!r} not available on this platform "
            f"(available: {available!r}); fell back to automatic"
        )
        return ("automatic", warning)

    fallback = available[0]
    warning = (
        f"provider {requested!r} not available on this platform "
        f"(available: {available!r}); fell back to {fallback!r}"
    )
    return (fallback, warning)


# ---------------------------------------------------------------------------
# cooked_from_errors + normalize_plane_dtype — PP12-113 PR-4
# ---------------------------------------------------------------------------

def cooked_from_errors(errors: list) -> bool:
    """Pure FR-5 no-silent-success predicate: cooked iff zero cook errors.

    APPENDED for PP12-113 PR-4 (cop_onnx_run_inference). Pure — never
    touches hou/Qt/pxr. The raised-but-empty-errors edge (a cook that
    RAISES hou.OperationFailed but leaves node.errors() empty) is folded
    into a non-empty `errors` list by the HANDLER before this predicate is
    called — this function only ever sees the already-folded list.

    Parameters
    ----------
    errors : list
        The cook's errors list (already folded by the handler when the
        cook raised but node.errors() was empty).

    Returns
    -------
    bool
        True iff `errors` is empty (a clean cook); False otherwise.
    """
    return not errors


def normalize_plane_dtype(storage_type) -> str:
    """Map a hou.imageLayerStorageType enum (or its str form) to a plain
    lowercase dtype token.

    APPENDED for PP12-113 PR-4 (cop_onnx_run_inference's output-plane
    manifest). Pure — never touches hou/Qt/pxr. Accepts either a plain
    string already in the enum-qualified form (e.g.
    ``"imageLayerStorageType.Float32"``), an unqualified plain string
    (e.g. ``"Float32"``), or any object whose ``str()`` produces one of
    those forms (e.g. the real ``hou.imageLayerStorageType`` enum member)
    — ``str()`` is applied first, per the locked contract.

    Parameters
    ----------
    storage_type
        A ``hou.imageLayerStorageType`` enum member, or a str already in
        either the qualified (``"imageLayerStorageType.Float32"``) or
        unqualified (``"Float32"``) form.

    Returns
    -------
    str
        The plain lowercase dtype token, e.g. ``"float32"``.
    """
    return str(storage_type).rsplit(".", 1)[-1].lower()
