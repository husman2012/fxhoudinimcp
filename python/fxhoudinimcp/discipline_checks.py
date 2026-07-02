"""
discipline_checks.py — pure-logic USD discipline validation functions
for the fxhoudinimcp USD/MaterialX export pipeline (PP12-112 PR-1).

Contract: imports NO hou, NO Qt/PySide6, NO pxr, NO numpy, NO MaterialX.
Plain Python stdlib only.  Pytest-able off-DCC (CL-015).

Public API
----------
format_for_extension(path)                   -> str
format_from_magic_bytes(header)              -> str
rule_no_world_wrapper(root_prims)            -> DisciplineCheck
rule_default_prim_set(default_prim)          -> DisciplineCheck
rule_format_extension_known(out_path)        -> DisciplineCheck
rule_format_matches_ext(actual_format, out_path) -> DisciplineCheck
rule_abs_texture_paths(texture_paths)        -> DisciplineCheck
aggregate_verdict(checks)                    -> str
run_discipline_checks(summary, *, out_path, actual_format, texture_paths) -> ValidationReport

Key fixes (from lockedFieldContract rev2)
------------------------------------------
M-1: rule_format_extension_known is present (preflight check, not silently omitted).
M-2: rule_format_matches_ext receives actual_format from magic-bytes, not from extension
     — the check is non-tautological.
M-3: Cross-platform absolute path detection (Windows drive, UNC, POSIX, forward-slash).
mn-1: whitespace-only defaultPrim treated as missing (FAIL).
mn-3: bare '.usd' basename (os.path.splitext gives ('', '.usd') on '.usd') -> ValueError.
mn-4: run_discipline_checks always produces >= 2 checks.
"""
from __future__ import annotations

import ntpath
import os
import posixpath
import re

from fxhoudinimcp.usd_export_model import DisciplineCheck, ValidationReport

# ---------------------------------------------------------------------------
# Extension-to-format lookup
# ---------------------------------------------------------------------------

_EXT_TO_FORMAT: dict = {
    ".usda": "usda",
    ".usdc": "usdc",
    ".usd":  "usdc",   # bare .usd is crate by convention
    ".usdz": "usdz",
}


def format_for_extension(path: str) -> str:
    """Return the USD format string for the file extension of *path*.

    Mapping (case-insensitive):
        .usda -> 'usda'
        .usdc -> 'usdc'
        .usd  -> 'usdc'   (crate default)
        .usdz -> 'usdz'

    The last extension wins for multi-extension paths (e.g. 'a.usd.usda' -> 'usda').

    Args:
        path: a file path string; only the extension is inspected.

    Returns:
        One of 'usda', 'usdc', 'usdz'.

    Raises:
        ValueError: if the extension is absent, empty, or not a recognized USD format.
                    mn-3: a file named literally '.usd' has an empty stem — raises.
    """
    # Extract only the final component's extension (handles directory paths cleanly)
    basename = os.path.basename(path)
    _, ext = os.path.splitext(basename)

    # mn-3: os.path.splitext treats a leading-dot name as the stem, so
    # splitext('.usd') -> ('.usd', '') (ext is EMPTY). A bare '.usd' basename
    # therefore raises below via the "no extension" branch. The stem=='' guard
    # here defends the rarer case where splitext does report a non-empty ext
    # with an empty stem.
    stem = basename[: len(basename) - len(ext)]
    if stem == "" and ext:
        # The entire basename IS the extension
        raise ValueError(
            f"format_for_extension: bare dotfile basename {basename!r} has no stem; "
            "cannot resolve format"
        )

    if not ext:
        raise ValueError(
            f"format_for_extension: path {path!r} has no file extension"
        )

    fmt = _EXT_TO_FORMAT.get(ext.lower())
    if fmt is None:
        raise ValueError(
            f"format_for_extension: extension {ext!r} is not a recognized USD format; "
            f"expected one of {sorted(_EXT_TO_FORMAT.keys())}"
        )
    return fmt


# ---------------------------------------------------------------------------
# Magic-byte format detection
# ---------------------------------------------------------------------------

def format_from_magic_bytes(header: bytes) -> str:
    """Detect the USD format from the first bytes of a file (M-2).

    Byte signatures:
        b'#usda'     -> 'usda'  (ASCII USD text)
        b'PXR-USDC'  -> 'usdc'  (USD crate binary)
        b'PK\\x03\\x04' -> 'usdz'  (ZIP local-file header)

    Note: the ZIP signature validates ZIP magic only, NOT USDZ package
    structure — any ZIP-magic file is reported 'usdz'. rule_format_matches_ext
    catches an extension mismatch downstream.

    The function inspects ONLY the bytes supplied — it has no knowledge of any
    file path or extension.  This independence is the M-2 non-tautology guarantee:
    a caller that supplies USDC magic bytes with a .usda out_path will get 'usdc'
    back, not 'usda'.

    Args:
        header: the first N bytes of a file (any length ≥ 0; shorter than the
                magic prefix is treated as unrecognized).

    Returns:
        One of 'usda', 'usdc', 'usdz'.

    Raises:
        ValueError: if the byte sequence does not match any known USD magic.
    """
    if header[:5] == b"#usda":
        return "usda"
    if header[:8] == b"PXR-USDC":
        return "usdc"
    if header[:4] == b"PK\x03\x04":
        return "usdz"
    raise ValueError(
        f"format_from_magic_bytes: unrecognized USD magic bytes {header[:8]!r}"
    )


# ---------------------------------------------------------------------------
# Individual rule functions
# ---------------------------------------------------------------------------

def rule_no_world_wrapper(root_prims: list) -> DisciplineCheck:
    """Check that the layer does not use UE-style wrapper prims at root level.

    The Homedini USD discipline forbids '/World' and '/root' as top-level prims
    (per usd-publish-discipline.md).  The check is case-sensitive (only the
    exact strings '/World' and '/root' fail; '/world', '/Root', '/asset/World'
    all pass).

    Args:
        root_prims: list of prim path strings at the top level of the layer.

    Returns:
        DisciplineCheck(id='no_world_wrapper', status='pass'|'fail', msg=...).
    """
    forbidden = {"/World", "/root"}
    offenders = [p for p in root_prims if p in forbidden]
    if offenders:
        return DisciplineCheck(
            id="no_world_wrapper",
            status="fail",
            msg=f"forbidden root prim(s): {', '.join(offenders)}",
        )
    return DisciplineCheck(id="no_world_wrapper", status="pass")


def rule_default_prim_set(default_prim: "str | None") -> DisciplineCheck:
    """Check that a defaultPrim opinion is set and non-empty.

    mn-1: A whitespace-only string (e.g. '   ', '\\t') is treated as missing.

    Args:
        default_prim: the layer's defaultPrim string, or None when unset.

    Returns:
        DisciplineCheck(id='default_prim_set', status='pass'|'fail', msg=...).
    """
    if default_prim is None or not default_prim.strip():
        return DisciplineCheck(
            id="default_prim_set",
            status="fail",
            msg="defaultPrim is not set or is empty",
        )
    return DisciplineCheck(id="default_prim_set", status="pass")


def rule_format_extension_known(out_path: str) -> DisciplineCheck:
    """Preflight check: ensure the out_path extension maps to a known USD format (M-1).

    A PASS result carries a msg naming the resolved format so the operator can
    confirm the intended output type.

    Args:
        out_path: the intended output file path string.

    Returns:
        DisciplineCheck(id='format_extension_known', status='pass'|'fail', msg=...).
    """
    try:
        fmt = format_for_extension(out_path)
        return DisciplineCheck(
            id="format_extension_known",
            status="pass",
            msg=f"resolves to {fmt}",
        )
    except ValueError as exc:
        return DisciplineCheck(
            id="format_extension_known",
            status="fail",
            msg=str(exc),
        )


def rule_format_matches_ext(actual_format: str, out_path: str) -> DisciplineCheck:
    """Postwrite check: confirm that the bytes-detected format matches the extension (M-2).

    *actual_format* MUST be the value returned by format_from_magic_bytes(), NOT
    re-derived from the extension.  That independence is what makes this check
    non-tautological: a file written as USDC binary but named .usda will FAIL.

    Args:
        actual_format: format string as read from the file's magic bytes.
        out_path:      the output file path (extension is inspected for the expected format).

    Returns:
        DisciplineCheck(id='format_matches_ext', status='pass'|'fail', msg=...).
    """
    try:
        expected = format_for_extension(out_path)
    except ValueError:
        return DisciplineCheck(
            id="format_matches_ext",
            status="fail",
            msg=f"cannot determine expected format from extension: {out_path!r}",
        )
    if actual_format == expected:
        return DisciplineCheck(id="format_matches_ext", status="pass")
    return DisciplineCheck(
        id="format_matches_ext",
        status="fail",
        msg=(
            f"actual format from magic bytes is '{actual_format}' but "
            f"extension implies '{expected}'"
        ),
    )


# ---------------------------------------------------------------------------
# Absolute-path detection helpers (M-3)
# ---------------------------------------------------------------------------

_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")


def _is_absolute(path: str) -> bool:
    r"""Return True if *path* is absolute by any cross-platform convention.

    Detects:
        - Windows drive-letter paths:  'C:/...' or 'C:\...' (lowercase too)
        - UNC paths:                   '\\\\server\\share\\...'
        - POSIX absolute:              '/home/...'
        - ntpath absolute:             via ntpath.isabs
        - posixpath absolute:          via posixpath.isabs
        - leading forward-slash:       '/something'
    """
    if ntpath.isabs(path):
        return True
    if posixpath.isabs(path):
        return True
    if _WIN_DRIVE_RE.match(path):
        return True
    if path.startswith("\\\\"):  # UNC
        return True
    if path.startswith("/"):     # POSIX
        return True
    return False


def rule_abs_texture_paths(texture_paths: list) -> DisciplineCheck:
    """Check for absolute texture file paths (M-3).

    Absolute paths are a portability hazard — they bind a USD file to one
    machine.  This is a WARN (not FAIL) because absolute paths sometimes
    arise legitimately in studio pipelines where the artist is on their own
    machine and has not yet token-ised the path.

    Args:
        texture_paths: list of texture path strings extracted from a MaterialX
                       or USD document.

    Returns:
        DisciplineCheck(id='abs_texture_paths', status='pass'|'warn', msg=...).
    """
    abs_paths = [p for p in texture_paths if _is_absolute(p)]
    if abs_paths:
        count = len(abs_paths)
        return DisciplineCheck(
            id="abs_texture_paths",
            status="warn",
            msg=f"{count} absolute texture path(s) found; consider using relative or tokenised paths",
        )
    return DisciplineCheck(id="abs_texture_paths", status="pass")


# ---------------------------------------------------------------------------
# Verdict aggregation
# ---------------------------------------------------------------------------

def aggregate_verdict(checks: list) -> str:
    """Aggregate a list of DisciplineCheck results into a single verdict string.

    Precedence (highest wins):
        'fail'  — any check has status 'fail'
        'warn'  — any check has status 'warn' (and none are 'fail')
        'pass'  — all checks are 'pass', or the list is empty

    Args:
        checks: list of DisciplineCheck instances.

    Returns:
        One of 'fail', 'warn', 'pass'.
    """
    verdict = "pass"
    for check in checks:
        if check.status == "fail":
            return "fail"
        if check.status == "warn":
            verdict = "warn"
    return verdict


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_discipline_checks(
    summary: dict,
    *,
    out_path: "str | None" = None,
    actual_format: "str | None" = None,
    texture_paths: "list | None" = None,
) -> ValidationReport:
    """Run the applicable set of USD discipline checks and return a ValidationReport.

    Modes
    -----
    MINIMAL (no out_path, no actual_format):
        Runs: no_world_wrapper + default_prim_set
        (always >= 2 checks — mn-4)

    PREFLIGHT (out_path, no actual_format):
        Runs: no_world_wrapper + default_prim_set + format_extension_known

    POSTWRITE (out_path + actual_format):
        Runs: no_world_wrapper + default_prim_set + format_matches_ext + abs_texture_paths
        (texture_paths defaults to [] when absent)

    Error cases
    -----------
    actual_format given without out_path -> ValueError

    Args:
        summary:        dict with at minimum 'default_prim' and 'root_prims' keys.
        out_path:       intended output file path (None for minimal mode).
        actual_format:  format string from format_from_magic_bytes() (postwrite only).
        texture_paths:  list of texture path strings (postwrite only; defaults to []).

    Returns:
        ValidationReport with verdict, checks list, and wrote_files=False.

    Raises:
        ValueError: if actual_format is supplied without out_path.
    """
    if actual_format is not None and out_path is None:
        raise ValueError(
            "run_discipline_checks: actual_format requires out_path to be set; "
            "cannot run format_matches_ext without knowing the intended extension"
        )

    default_prim = summary.get("default_prim")
    root_prims = summary.get("root_prims", [])

    # Always-present checks (satisfies mn-4 >= 2 even in minimal mode)
    checks: list = [
        rule_no_world_wrapper(root_prims),
        rule_default_prim_set(default_prim),
    ]

    if out_path is not None and actual_format is None:
        # PREFLIGHT mode — add extension-known check
        checks.append(rule_format_extension_known(out_path))

    elif out_path is not None and actual_format is not None:
        # POSTWRITE mode — replace preflight extension check with:
        # format_matches_ext + abs_texture_paths
        checks.append(rule_format_matches_ext(actual_format, out_path))
        checks.append(rule_abs_texture_paths(texture_paths or []))

    return ValidationReport.from_checks(checks, wrote_files=False)
