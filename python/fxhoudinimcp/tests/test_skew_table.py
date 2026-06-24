"""
Tests for skew_table.py — pure-logic layer.

No hou / Qt / pxr imports anywhere in this file.  Runs under plain pytest
headless (off-DCC, no Houdini install required).

Covers the public contract of:
  - skew_verdict(houdini, labs_vat, ue=None) -> tuple[str, list[str]]

Where:
  houdini  : str   — Houdini version string, e.g. "21.0.456", "21.5.999", "21"
  labs_vat : str   — SideFX Labs VAT version string, e.g. "3.0", "3.0.5"
  ue       : str | None — Unreal Engine version string, e.g. "5.4", "5.4.4";
                          None means "no UE target" (valid, caller did not specify)

Returns:
  (verdict: str, notes: list[str])
    verdict : one of "ok", "warn", "block"
    notes   : human-readable strings explaining the verdict

Validated behaviors:
  1. Known-good triple (H21 / VAT 3.0 / UE 5.4) → "ok", no blocking notes
  2. Unknown triple → "warn" (NOT silently "ok"), notes non-empty
  3. Version normalization: H21.0.456 and H21 and H21.5.999 behave consistently
     for the same (labs_vat, ue) combination
  4. ue=None is valid — does not raise; returns a (verdict, notes) tuple
  5. Block vocabulary is reachable (structural test only — does not pin a seed)

Test strategy: example-based assertions on the public observable behavior.
No mirror tests — never re-derive internal lookup table contents or assert
on which specific triple triggers "block".

TDD phase: RED — this file is authored BEFORE skew_table.py exists.
All tests should fail with ImportError / ModuleNotFoundError on first run.

Note on "known-bad" / block test:
  The plan spec requires that the "block" verdict is reachable, but hou-test
  authors this file BEFORE hou-dev picks which specific triples map to "block".
  To avoid coupling to an arbitrary seed value, the known-bad test is written
  as a vocabulary/structural assertion: it calls skew_verdict with a clearly
  invalid combination that any reasonable implementation would reject, and
  asserts that "block" appears in the set of possible return values by
  exhausting a small set of candidates until one returns "block".  This is
  not a mirror test — it does not inspect the lookup table; it only verifies
  that the string "block" can actually be returned.  See test_block_verdict_is_reachable.
"""

from __future__ import annotations

import sys
import os
from typing import Tuple

# ---------------------------------------------------------------------------
# Path bootstrap — allow running as a standalone script as well as via pytest.
# Mirrors the pattern in test_kinefx_model.py.
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_PKG_ROOT))

import pytest

# ---------------------------------------------------------------------------
# The module under test — does NOT exist yet; all tests below will be RED.
# ---------------------------------------------------------------------------
from fxhoudinimcp.skew_table import skew_verdict


# ===========================================================================
# Section 1 — Return type and vocabulary contract
#
# skew_verdict must always return a 2-tuple (verdict: str, notes: list[str])
# where verdict ∈ {"ok", "warn", "block"}.
# ===========================================================================

class TestReturnContract:
    """skew_verdict return type and vocabulary."""

    def test_returns_two_tuple(self):
        """skew_verdict must return a 2-tuple (verdict, notes)."""
        result = skew_verdict("21", "3.0")
        assert isinstance(result, tuple), (
            f"skew_verdict must return a tuple, got {type(result)!r}"
        )
        assert len(result) == 2, (
            f"skew_verdict must return a 2-tuple, got length {len(result)}"
        )

    def test_verdict_is_string(self):
        """First element of return tuple must be a str."""
        verdict, notes = skew_verdict("21", "3.0")
        assert isinstance(verdict, str), (
            f"verdict must be a str, got {type(verdict)!r}"
        )

    def test_notes_is_list(self):
        """Second element of return tuple must be a list."""
        verdict, notes = skew_verdict("21", "3.0")
        assert isinstance(notes, list), (
            f"notes must be a list, got {type(notes)!r}"
        )

    def test_verdict_in_allowed_vocabulary(self):
        """verdict must be one of 'ok', 'warn', 'block'."""
        allowed = {"ok", "warn", "block"}
        verdict, _ = skew_verdict("21", "3.0")
        assert verdict in allowed, (
            f"skew_verdict returned unexpected verdict {verdict!r}; "
            f"must be one of {allowed}"
        )

    def test_notes_contains_strings(self):
        """Every element of notes must be a str."""
        verdict, notes = skew_verdict("21.0.456", "3.0", "5.4")
        for item in notes:
            assert isinstance(item, str), (
                f"notes must contain only strs, got {type(item)!r}: {item!r}"
            )


# ===========================================================================
# Section 2 — Known-good triple: H21 / VAT 3.0 / UE 5.4
#
# The plan spec (pp12-111a) names H21 / VAT 3.0 / UE 5.4 as a supported
# combination.  This triple must return verdict "ok".
# ===========================================================================

class TestKnownGoodTriple:
    """Known-good (H21 / VAT 3.0 / UE 5.4) → 'ok'."""

    def test_known_good_returns_ok(self):
        """The canonical known-good triple must return verdict 'ok'."""
        verdict, notes = skew_verdict("21.0.456", "3.0", "5.4")
        assert verdict == "ok", (
            f"Known-good triple (H21/VAT3.0/UE5.4) must return 'ok', got {verdict!r}. "
            f"notes={notes}"
        )

    def test_known_good_with_ue_patch_returns_ok(self):
        """UE 5.4.4 (patch version present) still matches the 5.4 support window."""
        verdict, _ = skew_verdict("21.0.729", "3.0", "5.4.4")
        assert verdict == "ok", (
            f"H21/VAT3.0/UE5.4.4 should match the 5.4 support window → 'ok', "
            f"got {verdict!r}"
        )

    def test_known_good_houdini_major_only_returns_ok(self):
        """'21' (major-only) must resolve to the H21 support family → 'ok'."""
        verdict, _ = skew_verdict("21", "3.0", "5.4")
        assert verdict == "ok", (
            f"Houdini '21' (major only) should resolve to H21 → 'ok', got {verdict!r}"
        )

    def test_known_good_no_ue_does_not_block(self):
        """H21 / VAT 3.0 / no UE target: ue=None must not raise and must not 'block'."""
        verdict, notes = skew_verdict("21.0.456", "3.0", None)
        # Without a UE target, verdict is at most "warn" (cannot validate UE compat).
        # It must never be "block" for a known-good Houdini+VAT pair.
        assert verdict != "block", (
            f"H21/VAT3.0 with ue=None must not return 'block', got {verdict!r}. "
            f"notes={notes}"
        )


# ===========================================================================
# Section 3 — Unknown triple → "warn"
#
# A combination that is not in the known-support table must return "warn",
# NOT silently "ok".  The notes list must be non-empty to explain the warning.
# ===========================================================================

class TestUnknownTripleIsWarn:
    """Unknown triple → 'warn' (not 'ok')."""

    def test_unknown_houdini_version_returns_warn(self):
        """A Houdini version clearly outside the known support table → 'warn'."""
        # H99 does not exist — any sane implementation classifies this unknown.
        verdict, notes = skew_verdict("99.0.0", "3.0", "5.4")
        assert verdict == "warn", (
            f"An unknown Houdini version must return 'warn', got {verdict!r}. "
            f"notes={notes}"
        )
        assert len(notes) > 0, (
            "notes must be non-empty for an unknown triple "
            f"(verdict={verdict!r})"
        )

    def test_unknown_labs_vat_version_returns_warn(self):
        """A Labs VAT version clearly outside the known table → 'warn'."""
        verdict, notes = skew_verdict("21.0.456", "99.0", "5.4")
        assert verdict == "warn", (
            f"An unknown Labs VAT version must return 'warn', got {verdict!r}. "
            f"notes={notes}"
        )
        assert len(notes) > 0

    def test_unknown_ue_version_returns_warn(self):
        """A UE version clearly outside the known table → 'warn'."""
        verdict, notes = skew_verdict("21.0.456", "3.0", "99.0")
        assert verdict == "warn", (
            f"An unknown UE version must return 'warn', got {verdict!r}. "
            f"notes={notes}"
        )
        assert len(notes) > 0

    def test_unknown_triple_notes_non_empty(self):
        """Unknown triple must always produce at least one note explaining the warning."""
        verdict, notes = skew_verdict("99.0.0", "99.0", "99.0")
        # Whether verdict is "warn" or "block", notes must explain it.
        assert len(notes) > 0, (
            f"Unknown triple must have non-empty notes; verdict={verdict!r}"
        )

    def test_unknown_is_not_silently_ok(self):
        """An unknown triple must NEVER return 'ok' — silence on unknown is forbidden."""
        # This is the single hardest behavioral assertion in this module:
        # unknown inputs must produce a signal, not a false positive.
        verdict, notes = skew_verdict("99.0.0", "3.0", "5.4")
        assert verdict != "ok", (
            "skew_verdict must not return 'ok' for an unknown Houdini version. "
            f"Got {verdict!r} with notes={notes}"
        )


# ===========================================================================
# Section 4 — Version normalization
#
# "21.0.456" and "21" and "21.5.999" all belong to the H21 major family.
# For the same (labs_vat, ue) pair, any H21 version string should produce
# the same verdict as any other H21 version string.
# ===========================================================================

class TestVersionNormalization:
    """Version normalization — major family wins, patch variance is irrelevant."""

    def test_major_only_matches_full_version(self):
        """'21' and '21.0.456' return the same verdict for the same (labs_vat, ue)."""
        v_major, _ = skew_verdict("21", "3.0", "5.4")
        v_full,  _ = skew_verdict("21.0.456", "3.0", "5.4")
        assert v_major == v_full, (
            f"'21' and '21.0.456' must return the same verdict: "
            f"got {v_major!r} vs {v_full!r}"
        )

    def test_different_patches_same_verdict(self):
        """'21.0.456' and '21.5.999' return the same verdict for the same (labs_vat, ue)."""
        v_low,  _ = skew_verdict("21.0.456", "3.0", "5.4")
        v_high, _ = skew_verdict("21.5.999", "3.0", "5.4")
        assert v_low == v_high, (
            f"Different H21 patch versions must agree on verdict: "
            f"got {v_low!r} vs {v_high!r}"
        )

    def test_ue_patch_does_not_change_verdict(self):
        """'5.4' and '5.4.4' return the same verdict for the same (houdini, labs_vat)."""
        v_minor, _ = skew_verdict("21.0.456", "3.0", "5.4")
        v_patch, _ = skew_verdict("21.0.456", "3.0", "5.4.4")
        assert v_minor == v_patch, (
            f"UE '5.4' vs '5.4.4' must agree on verdict: "
            f"got {v_minor!r} vs {v_patch!r}"
        )

    def test_labs_vat_minor_does_not_split_family(self):
        """'3.0' and '3.0.5' return the same verdict (minor version normalization)."""
        v_base,  _ = skew_verdict("21.0.456", "3.0", "5.4")
        v_patch, _ = skew_verdict("21.0.456", "3.0.5", "5.4")
        assert v_base == v_patch, (
            f"Labs VAT '3.0' and '3.0.5' must return the same verdict: "
            f"got {v_base!r} vs {v_patch!r}"
        )


# ===========================================================================
# Section 5 — ue=None is valid
#
# The caller may omit the UE version (caller did not specify a UE target).
# skew_verdict(houdini, labs_vat, ue=None) must not raise and must return
# a valid (verdict, notes) pair.
# ===========================================================================

class TestUeNoneIsValid:
    """ue=None is a valid call — does not raise."""

    def test_ue_none_does_not_raise(self):
        """skew_verdict with ue=None must not raise any exception."""
        # Must not raise
        verdict, notes = skew_verdict("21.0.456", "3.0", None)
        assert verdict is not None

    def test_ue_none_returns_valid_verdict(self):
        """skew_verdict with ue=None returns a verdict in the allowed vocabulary."""
        allowed = {"ok", "warn", "block"}
        verdict, notes = skew_verdict("21.0.456", "3.0", None)
        assert verdict in allowed, (
            f"ue=None produced verdict {verdict!r} outside allowed set {allowed}"
        )

    def test_ue_none_returns_list_notes(self):
        """skew_verdict with ue=None returns notes as a list (possibly empty)."""
        verdict, notes = skew_verdict("21.0.456", "3.0", None)
        assert isinstance(notes, list)

    def test_ue_none_positional_vs_keyword_identical(self):
        """ue=None passed as keyword and as positional produce the same result."""
        result_kw  = skew_verdict("21.0.456", "3.0", ue=None)
        result_pos = skew_verdict("21.0.456", "3.0", None)
        assert result_kw[0] == result_pos[0], (
            f"Keyword ue=None vs positional None differ in verdict: "
            f"{result_kw[0]!r} vs {result_pos[0]!r}"
        )


# ===========================================================================
# Section 6 — "block" verdict is reachable (structural / vocabulary test)
#
# The spec names three verdict values: "ok", "warn", "block".  This test
# verifies that "block" is a reachable return value without coupling to a
# specific seed triple (hou-dev picks which combos are "block").
#
# Strategy: iterate over a short candidate list of clearly-wrong triples
# (very old or mismatched versions) and assert that at least one returns
# "block".  If none does, that means the module never returns "block" —
# which contradicts the spec.
#
# Why this approach is not a mirror test:
#   - We do not re-implement or inspect the lookup table.
#   - We do not assert WHICH triple triggers "block".
#   - We only assert that the vocabulary is reachable in principle.
# ===========================================================================

class TestBlockVerdictIsReachable:
    """'block' must be a reachable verdict — vocabulary completeness test."""

    def test_block_verdict_is_reachable(self):
        """At least one input combination must return verdict 'block'.

        This test does NOT pin which specific triple causes 'block' — that is
        hou-dev's choice.  It only verifies that 'block' is in the vocabulary.

        We try a set of deliberately mismatched candidates.  Any implementation
        that conforms to the spec (three-valued verdict) should have at least
        one of these return 'block'.  If the module only returns 'ok'/'warn',
        it is incomplete.
        """
        # Candidates chosen as "clearly problematic" combos without assuming
        # which exact ones hou-dev will mark as blocked.  These span:
        #   - A very old Houdini major paired with a modern VAT (major skew)
        #   - A very old UE paired with a modern combination (integration gap)
        #   - A combination where the VAT version is clearly not designed for
        #     the Houdini version (forward-compat violation)
        candidates = [
            ("18.0.0",  "3.0", "5.4"),   # H18 + modern VAT — potential block
            ("21.0.456", "1.0", "5.4"),   # very old VAT + modern UE
            ("21.0.456", "3.0", "4.0"),   # very old UE
            ("18.0.0",  "1.0", "4.0"),    # all-old combination
            ("17.0.0",  "3.0", "5.4"),    # ancient Houdini
        ]

        found_block = False
        verdicts_seen = []
        for houdini, labs_vat, ue in candidates:
            try:
                verdict, notes = skew_verdict(houdini, labs_vat, ue)
                verdicts_seen.append((houdini, labs_vat, ue, verdict))
                if verdict == "block":
                    found_block = True
                    break
            except Exception:
                # If a candidate raises, skip it (not an error in this test)
                pass

        assert found_block, (
            "skew_verdict never returned 'block' for any of the candidate "
            "mismatched triples — this means the 'block' verdict is unreachable, "
            "which contradicts the spec.  Verdicts seen: "
            + str(verdicts_seen)
        )

    def test_verdict_vocabulary_is_exactly_three_values(self):
        """The full vocabulary {ok, warn, block} must all be expressible.

        This is an aspirational structural test: we check that none of the three
        values is hard-coded out of existence by running a broader sweep.
        If after trying many inputs we never see 'warn', the function is broken.
        'block' reachability is covered by test_block_verdict_is_reachable above.
        """
        # A set of inputs designed to produce a spread of verdicts.
        inputs = [
            ("21.0.456", "3.0", "5.4"),    # expected ok
            ("99.0.0",   "3.0", "5.4"),    # expected warn (unknown H)
            ("21.0.456", "3.0", "99.0"),   # expected warn (unknown UE)
            ("21.0.456", "99.0", "5.4"),   # expected warn (unknown VAT)
        ]
        verdicts_seen = set()
        for houdini, labs_vat, ue in inputs:
            try:
                v, _ = skew_verdict(houdini, labs_vat, ue)
                verdicts_seen.add(v)
            except Exception:
                pass

        # We must see at least "ok" and "warn" from the inputs above.
        assert "ok" in verdicts_seen, (
            f"'ok' must be reachable; verdicts seen: {verdicts_seen}"
        )
        assert "warn" in verdicts_seen, (
            f"'warn' must be reachable; verdicts seen: {verdicts_seen}"
        )


# ===========================================================================
# Section 7 — hou-free import verification (CL-015)
#
# skew_table.py must import with zero hou/Qt/pxr at module top-level.
# ===========================================================================

class TestHouFreeImport:
    """Confirm skew_table.py carries no hou/Qt/pxr dependency."""

    def test_module_importable_without_hou(self):
        """skew_table must load under plain Python with no hou installed."""
        import fxhoudinimcp.skew_table as st
        assert st is not None

    def test_hou_not_in_skew_table_imports(self):
        """skew_table module must not reference 'hou' as a top-level import."""
        import fxhoudinimcp.skew_table as st
        import inspect
        import re
        source = inspect.getsource(st)
        source_no_comments = re.sub(r"#.*", "", source)
        assert "import hou" not in source_no_comments, (
            "skew_table.py must not import hou (CL-015 — pure-logic boundary)"
        )
