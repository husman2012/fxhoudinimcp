"""
Tests for discipline_checks.py — pure-logic USD discipline validator layer (PP12-112 PR-1).

TDD phase: RED — discipline_checks.py does NOT exist yet.
Expected failure: ModuleNotFoundError on import.

Contract: imports NO hou, NO Qt/PySide6, NO pxr, NO numpy, NO MaterialX.
Plain Python stdlib only.  Runs under the fork .venv with plain pytest
headless (off-DCC, no Houdini install required).

Covers the public contract of:
  - format_for_extension(path: str) -> str
      '.usda' -> 'usda', '.usdc' -> 'usdc', '.usd' -> 'usdc', '.usdz' -> 'usdz'
      Case-insensitive. ValueError on unknown/missing ext.
      Edge: bare '.usd' basename -> ValueError.
      Edge: 'a.usd.usda' -> 'usda' (last extension wins).

  - format_from_magic_bytes(header: bytes) -> str  [M-2]
      b'#usda' -> 'usda'
      b'PXR-USDC' -> 'usdc'
      b'PK\\x03\\x04' -> 'usdz'
      else -> ValueError('unrecognized USD magic bytes')

  - rule_no_world_wrapper(root_prims: list[str]) -> DisciplineCheck
      id='no_world_wrapper'; FAIL when any == '/World' or '/root' (case-sensitive).

  - rule_default_prim_set(default_prim: str | None) -> DisciplineCheck
      id='default_prim_set'; FAIL when None OR whitespace-only [mn-1].

  - rule_format_extension_known(out_path: str) -> DisciplineCheck  [M-1]
      id='format_extension_known'; PASS with msg; FAIL when unknown ext.

  - rule_format_matches_ext(actual_format: str, out_path: str) -> DisciplineCheck  [M-2]
      id='format_matches_ext'; FAIL when actual_format != format_for_extension(out_path).
      actual_format MUST come from format_from_magic_bytes, NOT from extension.

  - rule_abs_texture_paths(texture_paths: list[str]) -> DisciplineCheck  [M-3]
      id='abs_texture_paths'; WARN (not fail) when any path is absolute.
      Cross-platform: Windows drive, UNC, POSIX absolute all count.

  - aggregate_verdict(checks: list[DisciplineCheck]) -> str
      'fail' > 'warn' > 'pass'. [] -> 'pass'.

  - run_discipline_checks(summary: dict, *, out_path=None, actual_format=None,
                           texture_paths=None) -> ValidationReport
      PREFLIGHT (out_path, no actual_format):
          ids == {'no_world_wrapper', 'default_prim_set', 'format_extension_known'}
      POSTWRITE (out_path + actual_format):
          ids == {'no_world_wrapper', 'default_prim_set', 'format_matches_ext', 'abs_texture_paths'}
      actual_format WITHOUT out_path -> ValueError.
      Always >= 2 checks [mn-4].

Key fixes under test:
  M-1: rule_format_extension_known is present (preflight check, not silently omitted).
  M-2: rule_format_matches_ext receives actual_format from magic-bytes, not extension
       — the test supplies a deliberately mismatched combination to prove non-tautology.
  M-3: Cross-platform absolute path detection (Windows drive, UNC, POSIX, forward-slash).
  mn-1: whitespace-only defaultPrim string treated as missing (FAIL).
  mn-3: bare '.usd' basename (splitext gives '' not '.usd') -> ValueError.
  mn-4: run_discipline_checks always produces >= 2 checks.

Cross-references:
  - Plan pp12-112a lockedFieldContract rev2 (BINDING)
  - tdd-with-agents.md §4: hou-test writes red; hou-dev turns green
  - CL-015: pure-logic module, no hou/Qt/pxr
"""

from __future__ import annotations

import json
import sys
import os

# ---------------------------------------------------------------------------
# Path bootstrap — allow running standalone and via pytest.
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_PKG_ROOT))

import pytest

# ---------------------------------------------------------------------------
# The module under test — does NOT exist yet; all tests below are RED.
# ---------------------------------------------------------------------------
from fxhoudinimcp.discipline_checks import (
    format_for_extension,
    format_from_magic_bytes,
    rule_no_world_wrapper,
    rule_default_prim_set,
    rule_format_extension_known,
    rule_format_matches_ext,
    rule_abs_texture_paths,
    aggregate_verdict,
    run_discipline_checks,
)
from fxhoudinimcp.usd_export_model import DisciplineCheck, ValidationReport


# ===========================================================================
# Section 1 — format_for_extension(path)
# ===========================================================================

class TestFormatForExtension:
    """format_for_extension — extension-to-format lookup table."""

    # Core mapping
    def test_usda_extension(self):
        assert format_for_extension("scene.usda") == "usda"

    def test_usdc_extension(self):
        assert format_for_extension("asset.usdc") == "usdc"

    def test_usd_extension_resolves_to_usdc(self):
        """.usd without magic-byte read resolves to 'usdc' by convention."""
        assert format_for_extension("asset.usd") == "usdc"

    def test_usdz_extension(self):
        assert format_for_extension("asset.usdz") == "usdz"

    # Case-insensitivity
    def test_usda_uppercase(self):
        assert format_for_extension("scene.USDA") == "usda"

    def test_usdc_uppercase(self):
        assert format_for_extension("asset.USDC") == "usdc"

    def test_usd_mixed_case(self):
        assert format_for_extension("asset.Usd") == "usdc"

    # Last extension wins
    def test_double_extension_last_wins(self):
        """'a.usd.usda' -> last extension '.usda' -> 'usda'."""
        assert format_for_extension("a.usd.usda") == "usda"

    def test_double_extension_usdc_wins(self):
        assert format_for_extension("a.usda.usdc") == "usdc"

    # Unknown extension
    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError):
            format_for_extension("asset.abc")

    def test_no_extension_raises(self):
        with pytest.raises(ValueError):
            format_for_extension("asset")

    # mn-3: bare '.usd' basename
    def test_bare_dot_usd_basename_raises(self):
        """mn-3: a file named literally '.usd' has empty stem — splitext gives (''|'.usd').
        The contract says ValueError for this edge case.
        """
        with pytest.raises(ValueError):
            format_for_extension(".usd")

    def test_path_with_dirs(self):
        """Path with directory components is handled by extracting last extension."""
        assert format_for_extension("/some/path/to/file.usda") == "usda"

    def test_windows_path_usdc(self):
        """Windows-style path works correctly."""
        assert format_for_extension("C:/renders/asset.usdc") == "usdc"


# ===========================================================================
# Section 2 — format_from_magic_bytes(header: bytes)  [M-2]
# ===========================================================================

class TestFormatFromMagicBytes:
    """format_from_magic_bytes — byte-sniff format detection."""

    def test_usda_magic(self):
        """b'#usda 1.0\\n...' header -> 'usda'."""
        assert format_from_magic_bytes(b"#usda 1.0\n") == "usda"

    def test_usda_magic_exact_prefix(self):
        """Prefix b'#usda' alone is sufficient."""
        assert format_from_magic_bytes(b"#usda") == "usda"

    def test_usdc_magic(self):
        """b'PXR-USDC' header -> 'usdc'."""
        assert format_from_magic_bytes(b"PXR-USDC\x00\x00") == "usdc"

    def test_usdc_magic_exact(self):
        """Exact b'PXR-USDC' prefix alone is sufficient."""
        assert format_from_magic_bytes(b"PXR-USDC") == "usdc"

    def test_usdz_magic(self):
        """b'PK\\x03\\x04' ZIP header -> 'usdz'."""
        assert format_from_magic_bytes(b"PK\x03\x04" + b"\x00" * 20) == "usdz"

    def test_usdz_magic_exact(self):
        """Exact b'PK\\x03\\x04' prefix alone is sufficient."""
        assert format_from_magic_bytes(b"PK\x03\x04") == "usdz"

    def test_unknown_bytes_raises(self):
        """Unrecognized magic bytes raise ValueError."""
        with pytest.raises(ValueError):
            format_from_magic_bytes(b"\x00\x00\x00\x00")

    def test_empty_bytes_raises(self):
        """Empty bytes raise ValueError."""
        with pytest.raises(ValueError):
            format_from_magic_bytes(b"")

    def test_random_text_raises(self):
        """Random text that is not a USD magic raises ValueError."""
        with pytest.raises(ValueError):
            format_from_magic_bytes(b"Hello, world!")

    # M-2: non-tautology check
    # If format_from_magic_bytes simply re-reads the extension from the filename,
    # it is a tautological mirror.  Prove independence by supplying USDC magic
    # with a .usda extension — the magic wins, not the extension.
    def test_magic_beats_extension_usdc_magic_usda_name(self):
        """M-2 non-tautology: supplying USDC magic with a notional .usda extension
        must return 'usdc' — the function operates only on the bytes, not on the
        filename the caller might have.  The caller is responsible for the mismatch;
        format_from_magic_bytes returns what the bytes say.
        """
        # Pure bytes — no filename involved. USDC magic → 'usdc'.
        result = format_from_magic_bytes(b"PXR-USDC\x00\x00")
        assert result == "usdc", (
            "M-2: format_from_magic_bytes must return 'usdc' for PXR-USDC bytes "
            "regardless of any extension context"
        )


# ===========================================================================
# Section 3 — rule_no_world_wrapper
# ===========================================================================

class TestRuleNoWorldWrapper:
    """rule_no_world_wrapper — '/World' and '/root' at root level."""

    def test_returns_discipline_check(self):
        """Return type is DisciplineCheck."""
        result = rule_no_world_wrapper(["/asset"])
        assert isinstance(result, DisciplineCheck)

    def test_id_is_no_world_wrapper(self):
        result = rule_no_world_wrapper(["/asset"])
        assert result.id == "no_world_wrapper"

    def test_clean_prims_pass(self):
        """Non-forbidden root prims give status='pass'."""
        result = rule_no_world_wrapper(["/asset", "/mtl"])
        assert result.status == "pass"

    def test_world_prim_fails(self):
        """'/World' at root gives status='fail'."""
        result = rule_no_world_wrapper(["/World"])
        assert result.status == "fail"

    def test_root_prim_fails(self):
        """'/root' at root gives status='fail'."""
        result = rule_no_world_wrapper(["/root"])
        assert result.status == "fail"

    def test_world_among_others_fails(self):
        """Any forbidden prim in the list causes fail."""
        result = rule_no_world_wrapper(["/asset", "/World"])
        assert result.status == "fail"

    def test_case_sensitive_world_nested_passes(self):
        """'/asset/World' is not a root-level '/World' — must PASS."""
        result = rule_no_world_wrapper(["/asset/World"])
        assert result.status == "pass", (
            "'/asset/World' is a nested prim, not '/World' at root — must not fail"
        )

    def test_empty_list_passes(self):
        """Empty root_prims list returns status='pass'."""
        result = rule_no_world_wrapper([])
        assert result.status == "pass"

    def test_world_lowercase_passes(self):
        """The check is case-sensitive: '/world' (lowercase) must NOT trigger fail."""
        result = rule_no_world_wrapper(["/world"])
        assert result.status == "pass", (
            "rule_no_world_wrapper is case-sensitive: '/world' != '/World'"
        )

    def test_root_nested_passes(self):
        """'/asset/root' is a nested prim — must PASS."""
        result = rule_no_world_wrapper(["/asset/root"])
        assert result.status == "pass"


# ===========================================================================
# Section 4 — rule_default_prim_set  [mn-1]
# ===========================================================================

class TestRuleDefaultPrimSet:
    """rule_default_prim_set — None and whitespace-only treated as missing."""

    def test_returns_discipline_check(self):
        result = rule_default_prim_set("/asset")
        assert isinstance(result, DisciplineCheck)

    def test_id_is_default_prim_set(self):
        result = rule_default_prim_set("/asset")
        assert result.id == "default_prim_set"

    def test_valid_prim_passes(self):
        result = rule_default_prim_set("/asset")
        assert result.status == "pass"

    def test_none_fails(self):
        """None default_prim -> status='fail'."""
        result = rule_default_prim_set(None)
        assert result.status == "fail"

    def test_empty_string_fails(self):
        """Empty string -> status='fail'."""
        result = rule_default_prim_set("")
        assert result.status == "fail"

    def test_whitespace_only_fails(self):
        """mn-1: whitespace-only string is treated as missing -> status='fail'."""
        result = rule_default_prim_set("   ")
        assert result.status == "fail", (
            "mn-1: whitespace-only defaultPrim must be treated as missing (FAIL)"
        )

    def test_whitespace_tab_fails(self):
        """mn-1: tab-only string is treated as missing."""
        result = rule_default_prim_set("\t")
        assert result.status == "fail"

    def test_non_prim_string_passes(self):
        """Any non-empty, non-whitespace string passes (contract doesn't validate USD syntax)."""
        result = rule_default_prim_set("asset")
        assert result.status == "pass"


# ===========================================================================
# Section 5 — rule_format_extension_known  [M-1]
# ===========================================================================

class TestRuleFormatExtensionKnown:
    """rule_format_extension_known — preflight check that out_path extension is recognized."""

    def test_returns_discipline_check(self):
        result = rule_format_extension_known("asset.usdc")
        assert isinstance(result, DisciplineCheck)

    def test_id_is_format_extension_known(self):
        result = rule_format_extension_known("asset.usdc")
        assert result.id == "format_extension_known"

    def test_usdc_passes(self):
        result = rule_format_extension_known("asset.usdc")
        assert result.status == "pass"

    def test_usda_passes(self):
        result = rule_format_extension_known("scene.usda")
        assert result.status == "pass"

    def test_usd_passes(self):
        result = rule_format_extension_known("asset.usd")
        assert result.status == "pass"

    def test_usdz_passes(self):
        result = rule_format_extension_known("archive.usdz")
        assert result.status == "pass"

    def test_pass_has_msg(self):
        """M-1: passing check carries a msg naming the resolved format."""
        result = rule_format_extension_known("asset.usdc")
        assert result.status == "pass"
        assert result.msg, "rule_format_extension_known PASS must carry a msg naming the format"

    def test_pass_msg_contains_format(self):
        """The msg for a PASS mentions the resolved format string."""
        result = rule_format_extension_known("asset.usdc")
        assert "usdc" in result.msg.lower(), (
            f"Pass msg should mention the resolved format 'usdc', got msg={result.msg!r}"
        )

    def test_unknown_extension_fails(self):
        """Unknown extension produces status='fail'."""
        result = rule_format_extension_known("asset.abc")
        assert result.status == "fail"

    def test_no_extension_fails(self):
        """No extension at all produces status='fail'."""
        result = rule_format_extension_known("asset")
        assert result.status == "fail"

    def test_unknown_extension_fail_has_msg(self):
        """FAIL result must carry a msg naming the bad extension."""
        result = rule_format_extension_known("asset.obj")
        assert result.status == "fail"
        assert result.msg, "FAIL result must carry a descriptive msg"


# ===========================================================================
# Section 6 — rule_format_matches_ext  [M-2]
# ===========================================================================

class TestRuleFormatMatchesExt:
    """rule_format_matches_ext — actual_format from magic-bytes vs. extension."""

    def test_returns_discipline_check(self):
        result = rule_format_matches_ext("usdc", "asset.usdc")
        assert isinstance(result, DisciplineCheck)

    def test_id_is_format_matches_ext(self):
        result = rule_format_matches_ext("usdc", "asset.usdc")
        assert result.id == "format_matches_ext"

    def test_matching_usdc_passes(self):
        result = rule_format_matches_ext("usdc", "asset.usdc")
        assert result.status == "pass"

    def test_matching_usda_passes(self):
        result = rule_format_matches_ext("usda", "scene.usda")
        assert result.status == "pass"

    def test_matching_usd_extension_passes(self):
        """.usd resolves to 'usdc'; actual_format='usdc' passes."""
        result = rule_format_matches_ext("usdc", "asset.usd")
        assert result.status == "pass"

    def test_mismatch_fails(self):
        """actual_format='usda' with out_path '.usdc' -> FAIL."""
        result = rule_format_matches_ext("usda", "asset.usdc")
        assert result.status == "fail"

    # M-2: non-tautology — actual_format is INDEPENDENT of out_path extension.
    # If actual_format comes from magic-bytes and contradicts the extension, FAIL.
    def test_m2_magic_bytes_mismatch_is_detectable(self):
        """M-2: A .usda file that was actually written as USDC binary must FAIL.

        Caller scenario:
          header = read_first_bytes(out_path)    # b'PXR-USDC...'
          actual_format = format_from_magic_bytes(header)  # -> 'usdc'
          check = rule_format_matches_ext(actual_format, out_path)  # out_path ends '.usda'
          # Result: FAIL — bytes say 'usdc', extension says 'usda'

        This proves the check is non-tautological: if actual_format were derived
        from the extension (as the tautological impl would do), both sides would
        agree and the check would silently pass when the file is actually broken.
        """
        result = rule_format_matches_ext("usdc", "asset.usda")
        assert result.status == "fail", (
            "M-2: actual_format='usdc' with extension '.usda' must FAIL — "
            "magic-bytes say usdc but extension says usda"
        )

    def test_usdz_match_passes(self):
        result = rule_format_matches_ext("usdz", "archive.usdz")
        assert result.status == "pass"

    def test_mismatch_has_msg(self):
        """FAIL result carries a descriptive msg."""
        result = rule_format_matches_ext("usda", "asset.usdc")
        assert result.status == "fail"
        assert result.msg, "FAIL result must carry a msg"


# ===========================================================================
# Section 7 — rule_abs_texture_paths  [M-3]
# ===========================================================================

class TestRuleAbsTexturePaths:
    """rule_abs_texture_paths — cross-platform absolute path detection."""

    def test_returns_discipline_check(self):
        result = rule_abs_texture_paths([])
        assert isinstance(result, DisciplineCheck)

    def test_id_is_abs_texture_paths(self):
        result = rule_abs_texture_paths([])
        assert result.id == "abs_texture_paths"

    def test_empty_list_passes(self):
        """No texture paths -> status='pass'."""
        result = rule_abs_texture_paths([])
        assert result.status == "pass"

    def test_relative_paths_pass(self):
        """All-relative paths -> status='pass'."""
        result = rule_abs_texture_paths(["./tex/albedo.png", "../mtl/normal.png"])
        assert result.status == "pass"

    def test_posix_absolute_warns(self):
        """M-3: POSIX absolute path '/home/user/tex.png' -> status='warn' (not fail)."""
        result = rule_abs_texture_paths(["/home/user/tex.png"])
        assert result.status == "warn", (
            "M-3: absolute paths produce WARN not FAIL"
        )

    def test_windows_drive_absolute_warns(self):
        """M-3: Windows drive 'C:/Users/...' -> status='warn'."""
        result = rule_abs_texture_paths(["C:/Users/user/tex.png"])
        assert result.status == "warn"

    def test_windows_drive_backslash_warns(self):
        """M-3: 'C:\\Users\\user\\tex.png' -> status='warn'."""
        result = rule_abs_texture_paths(["C:\\Users\\user\\tex.png"])
        assert result.status == "warn"

    def test_unc_path_warns(self):
        """M-3: UNC path '\\\\server\\share\\tex.png' -> status='warn'."""
        result = rule_abs_texture_paths(["\\\\server\\share\\tex.png"])
        assert result.status == "warn"

    def test_forward_slash_absolute_warns(self):
        """M-3: Path starting with '/' -> status='warn'."""
        result = rule_abs_texture_paths(["/tex/albedo.png"])
        assert result.status == "warn"

    def test_mixed_relative_and_absolute_warns(self):
        """Any absolute in a mixed list -> status='warn'."""
        result = rule_abs_texture_paths(["./rel.png", "C:/abs/tex.png"])
        assert result.status == "warn"

    def test_absolute_is_warn_not_fail(self):
        """M-3: The status for absolute paths is explicitly 'warn', not 'fail'."""
        result = rule_abs_texture_paths(["C:/tex.png"])
        assert result.status != "fail", (
            "M-3: absolute texture paths produce 'warn', never 'fail'"
        )

    def test_warn_has_msg(self):
        """WARN result carries a msg."""
        result = rule_abs_texture_paths(["C:/tex.png"])
        assert result.status == "warn"
        assert result.msg, "WARN for absolute paths must carry a msg"

    def test_windows_drive_lowercase(self):
        """M-3: 'd:/renders/tex.png' (lowercase drive letter) -> status='warn'."""
        result = rule_abs_texture_paths(["d:/renders/tex.png"])
        assert result.status == "warn"


# ===========================================================================
# Section 8 — aggregate_verdict
# ===========================================================================

class TestAggregateVerdict:
    """aggregate_verdict — severity ordering and empty-list base case."""

    def test_empty_list_returns_pass(self):
        assert aggregate_verdict([]) == "pass"

    def test_all_pass_returns_pass(self):
        checks = [
            DisciplineCheck(id="a", status="pass"),
            DisciplineCheck(id="b", status="pass"),
        ]
        assert aggregate_verdict(checks) == "pass"

    def test_one_warn_returns_warn(self):
        checks = [
            DisciplineCheck(id="a", status="pass"),
            DisciplineCheck(id="b", status="warn"),
        ]
        assert aggregate_verdict(checks) == "warn"

    def test_one_fail_returns_fail(self):
        checks = [
            DisciplineCheck(id="a", status="pass"),
            DisciplineCheck(id="b", status="fail"),
        ]
        assert aggregate_verdict(checks) == "fail"

    def test_fail_dominates_warn(self):
        checks = [
            DisciplineCheck(id="a", status="warn"),
            DisciplineCheck(id="b", status="fail"),
        ]
        assert aggregate_verdict(checks) == "fail"

    def test_fail_dominates_all(self):
        checks = [
            DisciplineCheck(id="a", status="pass"),
            DisciplineCheck(id="b", status="warn"),
            DisciplineCheck(id="c", status="fail"),
        ]
        assert aggregate_verdict(checks) == "fail"

    def test_warn_dominates_pass(self):
        checks = [
            DisciplineCheck(id="a", status="pass"),
            DisciplineCheck(id="b", status="pass"),
            DisciplineCheck(id="c", status="warn"),
        ]
        assert aggregate_verdict(checks) == "warn"

    def test_returns_string(self):
        result = aggregate_verdict([])
        assert isinstance(result, str)

    def test_single_fail(self):
        assert aggregate_verdict([DisciplineCheck(id="x", status="fail")]) == "fail"

    def test_single_warn(self):
        assert aggregate_verdict([DisciplineCheck(id="x", status="warn")]) == "warn"


# ===========================================================================
# Section 9 — run_discipline_checks
# ===========================================================================

# Minimal summary dicts used throughout
_SUMMARY_WITH_PRIM = {
    "default_prim": "/asset",
    "root_prims": ["/asset"],
}
_SUMMARY_NO_PRIM = {
    "default_prim": None,
    "root_prims": ["/asset"],
}
_SUMMARY_WORLD = {
    "default_prim": "/World",
    "root_prims": ["/World"],
}


class TestRunDisciplineChecksPreflight:
    """run_discipline_checks — PREFLIGHT mode (out_path given, no actual_format)."""

    def test_returns_validation_report(self):
        vr = run_discipline_checks(_SUMMARY_WITH_PRIM, out_path="asset.usdc")
        assert isinstance(vr, ValidationReport)

    def test_preflight_check_ids(self):
        """Preflight produces exactly {no_world_wrapper, default_prim_set, format_extension_known}."""
        vr = run_discipline_checks(_SUMMARY_WITH_PRIM, out_path="asset.usdc")
        ids = {c.id for c in vr.checks}
        assert ids == {"no_world_wrapper", "default_prim_set", "format_extension_known"}, (
            f"Preflight check IDs must be exactly the 3-check set, got {ids!r}"
        )

    def test_preflight_has_at_least_two_checks(self):
        """mn-4: run_discipline_checks always produces >= 2 checks."""
        vr = run_discipline_checks(_SUMMARY_WITH_PRIM, out_path="asset.usdc")
        assert len(vr.checks) >= 2, (
            f"mn-4: must produce >= 2 checks, got {len(vr.checks)}"
        )

    def test_preflight_clean_summary_passes(self):
        """Clean summary with valid prim and known extension -> verdict='pass'."""
        vr = run_discipline_checks(_SUMMARY_WITH_PRIM, out_path="asset.usdc")
        assert vr.verdict == "pass"

    def test_preflight_unknown_extension_fails(self):
        """Unknown extension in preflight produces verdict='fail'."""
        vr = run_discipline_checks(_SUMMARY_WITH_PRIM, out_path="asset.abc")
        assert vr.verdict == "fail"

    def test_preflight_world_prim_fails(self):
        """/World root prim produces verdict='fail'."""
        vr = run_discipline_checks(_SUMMARY_WORLD, out_path="asset.usdc")
        assert vr.verdict == "fail"

    def test_preflight_missing_prim_fails(self):
        """Missing default_prim produces verdict='fail'."""
        vr = run_discipline_checks(_SUMMARY_NO_PRIM, out_path="asset.usdc")
        assert vr.verdict == "fail"

    def test_preflight_does_not_include_postwrite_ids(self):
        """Preflight must NOT include 'format_matches_ext' or 'abs_texture_paths'."""
        vr = run_discipline_checks(_SUMMARY_WITH_PRIM, out_path="asset.usdc")
        ids = {c.id for c in vr.checks}
        assert "format_matches_ext" not in ids
        assert "abs_texture_paths" not in ids


class TestRunDisciplineChecksPostwrite:
    """run_discipline_checks — POSTWRITE mode (out_path + actual_format given)."""

    def test_returns_validation_report(self):
        vr = run_discipline_checks(
            _SUMMARY_WITH_PRIM,
            out_path="asset.usdc",
            actual_format="usdc",
        )
        assert isinstance(vr, ValidationReport)

    def test_postwrite_check_ids(self):
        """Postwrite produces exactly {no_world_wrapper, default_prim_set, format_matches_ext, abs_texture_paths}."""
        vr = run_discipline_checks(
            _SUMMARY_WITH_PRIM,
            out_path="asset.usdc",
            actual_format="usdc",
        )
        ids = {c.id for c in vr.checks}
        assert ids == {
            "no_world_wrapper",
            "default_prim_set",
            "format_matches_ext",
            "abs_texture_paths",
        }, f"Postwrite check IDs must be exactly the 4-check set, got {ids!r}"

    def test_postwrite_has_at_least_two_checks(self):
        """mn-4: postwrite also produces >= 2 checks."""
        vr = run_discipline_checks(
            _SUMMARY_WITH_PRIM,
            out_path="asset.usdc",
            actual_format="usdc",
        )
        assert len(vr.checks) >= 2

    def test_postwrite_clean_passes(self):
        """Clean postwrite scenario -> verdict='pass'."""
        vr = run_discipline_checks(
            _SUMMARY_WITH_PRIM,
            out_path="asset.usdc",
            actual_format="usdc",
        )
        assert vr.verdict == "pass"

    def test_postwrite_format_mismatch_fails(self):
        """actual_format contradicts extension -> verdict='fail'."""
        vr = run_discipline_checks(
            _SUMMARY_WITH_PRIM,
            out_path="asset.usda",
            actual_format="usdc",  # bytes said usdc but extension is usda
        )
        assert vr.verdict == "fail"

    def test_postwrite_abs_texture_warns(self):
        """Absolute texture paths produce at least 'warn'."""
        vr = run_discipline_checks(
            _SUMMARY_WITH_PRIM,
            out_path="asset.usdc",
            actual_format="usdc",
            texture_paths=["C:/tex/albedo.png"],
        )
        assert vr.verdict in ("warn", "fail")

    def test_postwrite_does_not_include_preflight_extension_known(self):
        """Postwrite must NOT include 'format_extension_known'."""
        vr = run_discipline_checks(
            _SUMMARY_WITH_PRIM,
            out_path="asset.usdc",
            actual_format="usdc",
        )
        ids = {c.id for c in vr.checks}
        assert "format_extension_known" not in ids


class TestRunDisciplineChecksInvalidCalls:
    """run_discipline_checks — error cases."""

    def test_actual_format_without_out_path_raises(self):
        """Supplying actual_format without out_path raises ValueError."""
        with pytest.raises(ValueError):
            run_discipline_checks(_SUMMARY_WITH_PRIM, actual_format="usdc")

    def test_no_out_path_no_actual_format_raises_or_returns_minimal(self):
        """Without out_path or actual_format, behaviour must be deterministic.

        The contract says actual_format without out_path raises ValueError.
        Without EITHER, the spec does not define a valid call mode — the impl
        may raise or return a minimal report.  We assert it does not crash silently
        with an empty check list (mn-4 still applies if it returns at all).
        """
        try:
            vr = run_discipline_checks(_SUMMARY_WITH_PRIM)
            # If it returns, it must still have >= 2 checks
            assert len(vr.checks) >= 2, "mn-4: even the minimal mode must produce >= 2 checks"
        except (ValueError, TypeError):
            pass  # Raising is also an acceptable outcome for this ambiguous call


class TestRunDisciplineChecksOutputContract:
    """run_discipline_checks — ValidationReport attributes on the returned object."""

    def test_wrote_files_is_false_in_preflight(self):
        """Preflight mode never sets wrote_files=True."""
        vr = run_discipline_checks(_SUMMARY_WITH_PRIM, out_path="asset.usdc")
        assert vr.wrote_files is False

    def test_wrote_files_is_false_in_postwrite_default(self):
        """Postwrite without wrote_files kwarg gives wrote_files=False."""
        vr = run_discipline_checks(
            _SUMMARY_WITH_PRIM,
            out_path="asset.usdc",
            actual_format="usdc",
        )
        assert vr.wrote_files is False

    def test_result_is_json_serialisable(self):
        """ValidationReport returned by run_discipline_checks is JSON-serialisable."""
        vr = run_discipline_checks(_SUMMARY_WITH_PRIM, out_path="asset.usdc")
        json.dumps(vr.to_dict())

    def test_verdict_consistent_with_checks(self):
        """verdict on the returned report is consistent with aggregate_verdict(checks)."""
        vr = run_discipline_checks(_SUMMARY_WITH_PRIM, out_path="asset.usdc")
        expected = aggregate_verdict(vr.checks)
        assert vr.verdict == expected, (
            f"ValidationReport.verdict={vr.verdict!r} does not match "
            f"aggregate_verdict(checks)={expected!r}"
        )
