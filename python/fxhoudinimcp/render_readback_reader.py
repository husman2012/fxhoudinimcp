"""render_readback_reader.py — OIIO EXR pixel reader for render_read_pixels.

Reads an EXR file from disk via OpenImageIO (OIIO) and returns per-channel
flat float lists suitable for consumption by build_readback() / build_compare().

NOT pure-logic (imports OpenImageIO/numpy) — tested via hython-smoke only.

Public API
----------
_plane_of_channel(name) -> str
    Map a single OIIO channel name to its AOV plane name.
    (Shared by list_exr_planes and read_exr_plane — ONE consistent definition.)

read_exr_plane(path, plane=None, subimage=None)
    -> (channels, xres, yres, dtype)

    channels  : list[list[float]]  — per-channel flat pixel lists (row-major)
    xres      : int
    yres      : int
    dtype     : str  — OIIO type string (e.g. "float16", "float32")

list_exr_planes(path, subimage=None) -> list[str]
    Return the ordered list of distinct AOV plane names present in the EXR.

PP12-114 / pp12-114e — unitId: pp12-114e
PP12-114 / pp12-114f — rev2: _plane_of_channel helper (BLOCKER+MAJOR-1 fix);
    F1 fail-loud (remove silent fallback); list_exr_planes (new).
"""
from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# Shared plane<->channel mapping helper (rev2 — pp12-114f)
# ---------------------------------------------------------------------------

def _plane_of_channel(name: str) -> str:
    """Map an OIIO channel name to its AOV plane name.

    Rules (binding — must match in BOTH list_exr_planes and read_exr_plane):

    1. BEAUTY plane 'C':
       - A no-dot channel whose name is exactly one of R/G/B/A
         (case-insensitive):  R, G, B, A -> 'C'
       - A dotted channel whose prefix-before-first-dot is exactly 'C'
         (i.e. C.R / C.G / C.B / C.A) -> 'C'   [BLOCKER fix]

    2. Any OTHER dotted channel -> its prefix-before-first-dot.
       e.g. depth.Z -> 'depth', N.x -> 'N', diffuse.R -> 'diffuse'

    3. Any OTHER no-dot channel (a bare non-RGBA channel like 'Z' or 'depth')
       -> ITS OWN NAME.  NOT folded into beauty.   [MAJOR-1 fix]

    Worked examples:
        [R, G, B, depth.Z, N.x, N.y, N.z] -> planes ['C', 'depth', 'N']
        [C.R, C.G, C.B]                   -> ['C']  (beauty stored dotted)
        [diffuse.R, diffuse.G, diffuse.B]  -> ['diffuse']
        [R, G, B, Z]                       -> ['C', 'Z']  (bare Z is own plane)
    """
    if "." in name:
        prefix = name.split(".", 1)[0]
        # C.* -> beauty 'C'
        if prefix == "C":
            return "C"
        return prefix
    else:
        # No dot: R/G/B/A (case-insensitive) -> beauty 'C'
        if name.upper() in ("R", "G", "B", "A"):
            return "C"
        # Any other bare channel (Z, depth, custom) -> its own name
        return name


# ---------------------------------------------------------------------------
# list_exr_planes — new in pp12-114f
# ---------------------------------------------------------------------------

def list_exr_planes(
    path: str,
    subimage: "int | None" = None,
) -> "list[str]":
    """Return the ordered list of distinct AOV plane names in an EXR file.

    Uses the same _plane_of_channel mapping as read_exr_plane, so every plane
    this function reports is a plane read_exr_plane can successfully read.

    Args:
        path:     Absolute path to the EXR file (already Houdini-expanded).
        subimage: Sub-image index for multi-part EXR.  None reads subimage 0.

    Returns:
        Ordered list of distinct plane names in first-seen channel order.
        e.g. for channels [R, G, B, depth.Z, N.x] -> ['C', 'depth', 'N']

    Raises:
        FileNotFoundError: When path does not exist.
        RuntimeError:      When OIIO cannot open the EXR.
        ImportError:       When OpenImageIO is not available.
    """
    try:
        import OpenImageIO as _oiio
    except ImportError as exc:
        raise ImportError(
            "OpenImageIO is required to read EXR files; "
            "it is bundled with Houdini's hython but not available in plain Python."
        ) from exc

    if not os.path.isfile(path):
        raise FileNotFoundError(f"EXR file not found: {path!r}")

    sub = subimage if subimage is not None else 0
    buf = _oiio.ImageBuf(path, sub, 0)

    err = buf.geterror()
    if err:
        raise RuntimeError(f"OIIO could not open {path!r}: {err}")

    spec = buf.spec()
    channel_names: list[str] = list(spec.channelnames)

    # Collect distinct plane names in first-seen order
    seen: set[str] = set()
    planes: list[str] = []
    for ch_name in channel_names:
        p = _plane_of_channel(ch_name)
        if p not in seen:
            seen.add(p)
            planes.append(p)

    return planes


# ---------------------------------------------------------------------------
# read_exr_plane
# ---------------------------------------------------------------------------

def read_exr_plane(
    path: str,
    plane: "str | None" = None,
    subimage: "int | None" = None,
) -> "tuple[list[list[float]], int, int, str]":
    """Read an EXR file and return flat per-channel float buffers.

    Plane selection uses _plane_of_channel so it is consistent with
    list_exr_planes (BLOCKER fix: C.R/C.G/C.B/C.A are now beauty 'C').

    Args:
        path:     Absolute path to the EXR file (already Houdini-expanded by
                  the handler before this call).
        plane:    AOV plane name.  ``None``, ``"C"``, or ``"beauty"`` select
                  the top-level beauty plane (all channels whose
                  _plane_of_channel() returns 'C').  Any other value matches
                  channels whose _plane_of_channel() returns that plane name.
        subimage: When set, reads only this sub-image index from a multi-part
                  EXR.  ``None`` reads subimage 0 (the default first image).

    Returns:
        ``(channels, xres, yres, dtype)`` where:

        - ``channels`` is a list of ``xres * yres`` flat float values per
          channel, indexed as ``channels[ch][y * xres + x]``.
        - ``xres`` and ``yres`` are the source image dimensions.
        - ``dtype`` is the OIIO base type string (e.g. ``"float16"``).

    Raises:
        FileNotFoundError: When *path* does not exist.
        RuntimeError:      When OIIO cannot open or read the EXR.
        ValueError:        When the requested plane matches ZERO channels in the
                           file.  A zero-match is always a real error — the
                           handler's except turns this into {ok:False,error}.
                           (F1 fail-loud: the old silent fallback is removed.)
        ImportError:       When OpenImageIO or numpy is not available.
    """
    import numpy as _np

    try:
        import OpenImageIO as _oiio
    except ImportError as exc:
        raise ImportError(
            "OpenImageIO is required to read EXR files; "
            "it is bundled with Houdini's hython but not available in plain Python."
        ) from exc

    if not os.path.isfile(path):
        raise FileNotFoundError(f"EXR file not found: {path!r}")

    # ------------------------------------------------------------------
    # Open the image
    # ------------------------------------------------------------------
    sub = subimage if subimage is not None else 0
    buf = _oiio.ImageBuf(path, sub, 0)  # (path, subimage, miplevel)

    # Check for open errors
    err = buf.geterror()
    if err:
        raise RuntimeError(f"OIIO could not open {path!r}: {err}")

    spec = buf.spec()
    xres: int = spec.width
    yres: int = spec.height
    nchannels: int = spec.nchannels
    channel_names: list[str] = list(spec.channelnames)

    # Determine the OIIO channel format string for the first data channel.
    # spec.format is the base type for all channels; use its string form.
    dtype_str: str = str(spec.format)

    # ------------------------------------------------------------------
    # Plane selection using shared _plane_of_channel mapping (rev2)
    # ------------------------------------------------------------------
    # Normalise request: None / "C" / "beauty" all mean the 'C' beauty plane.
    requested_plane = "C" if (plane is None or plane == "beauty") else plane

    selected_indices = [
        i for i, name in enumerate(channel_names)
        if _plane_of_channel(name) == requested_plane
    ]

    # F1 fail-loud: zero-match is always an error — no silent fallback.
    if not selected_indices:
        available = list(dict.fromkeys(
            _plane_of_channel(n) for n in channel_names
        ))
        raise ValueError(
            f"read_exr_plane: plane {requested_plane!r} matched no channels in "
            f"{path!r} (available: {available!r})"
        )

    # ------------------------------------------------------------------
    # Read pixels as numpy float32 (OIIO converts on the fly)
    # ------------------------------------------------------------------
    # buf.get_pixels returns (H, W, C) ndarray in the requested type.
    pixels_np = buf.get_pixels(_oiio.FLOAT)  # always read as float32 for safety

    if pixels_np is None:
        err = buf.geterror()
        raise RuntimeError(f"OIIO get_pixels returned None for {path!r}: {err}")

    # pixels_np shape: (yres, xres, nchannels) — guaranteed by OIIO for 2D images.
    # Reshape to (n_pixels, nchannels) for per-channel slicing.
    flat_pixels = pixels_np.reshape(-1, nchannels)  # (H*W, C)

    # Per-channel flat lists, only for selected indices.
    channels: list[list[float]] = [
        flat_pixels[:, ch_i].tolist()
        for ch_i in selected_indices
    ]

    return channels, xres, yres, dtype_str
