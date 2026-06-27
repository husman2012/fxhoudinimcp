"""render_readback_reader.py — OIIO EXR pixel reader for render_read_pixels.

Reads an EXR file from disk via OpenImageIO (OIIO) and returns per-channel
flat float lists suitable for consumption by build_readback().

NOT pure-logic (imports OpenImageIO/numpy) — tested via hython-smoke only.

Public API
----------
read_exr_plane(path, plane=None, subimage=None)
    -> (channels, xres, yres, dtype)

    channels  : list[list[float]]  — per-channel flat pixel lists (row-major)
    xres      : int
    yres      : int
    dtype     : str  — OIIO type string (e.g. "float16", "float32")

PP12-114 / pp12-114e — unitId: pp12-114e
"""
from __future__ import annotations

import os


def read_exr_plane(
    path: str,
    plane: "str | None" = None,
    subimage: "int | None" = None,
) -> "tuple[list[list[float]], int, int, str]":
    """Read an EXR file and return flat per-channel float buffers.

    The plane is selected by layer-prefix (the part of the channel name before
    the first dot).  Special-cased plane names for the top-level RGB beauty:
    ``None``, ``"C"``, and ``"beauty"`` all select channels whose names contain
    no dot (the raw R/G/B channels).

    Args:
        path:     Absolute path to the EXR file (already Houdini-expanded by
                  the handler before this call).
        plane:    AOV plane name.  ``None``, ``"C"``, or ``"beauty"`` select
                  the top-level beauty (channels with no dot).  Any other value
                  matches channels whose name starts with ``"<plane>."``.
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
    # Plane selection by layer-prefix
    # ------------------------------------------------------------------
    # "Beauty" plane: None / "C" / "beauty" -> channels with no dot (top-level R/G/B/A)
    is_beauty = (plane is None) or (plane in ("C", "beauty"))

    if is_beauty:
        selected_indices = [
            i for i, name in enumerate(channel_names)
            if "." not in name
        ]
    else:
        # Match channels whose name starts with "<plane>."
        prefix = plane + "."
        selected_indices = [
            i for i, name in enumerate(channel_names)
            if name.startswith(prefix)
        ]

    # Fallback: if selection is empty, include ALL channels.
    if not selected_indices:
        selected_indices = list(range(nchannels))

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
