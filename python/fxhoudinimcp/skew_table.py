"""Pure-logic version-skew compatibility table. No hou / Qt / pxr imports."""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Version normalization helpers
# ---------------------------------------------------------------------------

def _normalize_houdini(version: str) -> str:
    """Normalize a Houdini version string to its major component.

    Examples:
        "21.0.456" -> "21"
        "21.5.999" -> "21"
        "21"       -> "21"
    """
    return version.split(".")[0]


def _normalize_ue(version: str) -> str:
    """Normalize an Unreal Engine version string to major.minor.

    Examples:
        "5.4.4" -> "5.4"
        "5.4"   -> "5.4"
        "5"     -> "5"
    """
    parts = version.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


def _normalize_vat(version: str) -> str:
    """Normalize a Labs VAT version string to major.minor.

    Examples:
        "3.0.5" -> "3.0"
        "3.0"   -> "3.0"
        "1.0"   -> "1.0"
    """
    parts = version.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


# ---------------------------------------------------------------------------
# Compatibility lookup table
#
# Keys: (houdini_major: str, vat_major_minor: str, ue_major_minor: str | None)
# Values: (verdict: str, notes: list[str])
#
# Verdict vocabulary: "ok", "warn", "block"
#
# "block" is for combinations that are definitively incompatible and must not
# be used together.  "warn" is for combinations that may work but require
# verification.  "ok" is for explicitly validated support windows.
# ---------------------------------------------------------------------------

_NOTE_VALIDATED = (
    "Validated support window: Houdini 21 / SideFX Labs VAT 3.0 / UE 5.4. "
    "This combination is production-ready."
)
_NOTE_OLD_VAT = (
    "SideFX Labs VAT 1.x is not designed for use with modern Houdini 21 and UE 5.x. "
    "This combination is blocked — upgrade Labs VAT to 3.0 or later."
)
_NOTE_OLD_UE = (
    "Unreal Engine 4.x is not compatible with the SideFX Labs VAT 3.x shader "
    "interface.  This combination is blocked — upgrade to UE 5.x."
)
_NOTE_OLD_HOUDINI = (
    "Houdini 18 and older are not supported by the current Labs VAT / UE pipeline. "
    "This combination is blocked — upgrade to Houdini 21."
)
_NOTE_ANCIENT_HOUDINI = (
    "Houdini 17 and older are not supported by any current Labs VAT / UE build. "
    "This combination is blocked — upgrade to Houdini 21."
)

# Explicit table — (h_major, vat_maj_min, ue_maj_min | None) -> (verdict, notes)
_TABLE: dict[tuple, tuple] = {
    # -----------------------------------------------------------------------
    # Known-good support window
    # -----------------------------------------------------------------------
    ("21", "3.0", "5.4"): ("ok", [_NOTE_VALIDATED]),

    # -----------------------------------------------------------------------
    # Known-bad / blocked combinations
    # -----------------------------------------------------------------------
    # Very old VAT with modern UE — forward-compat violation
    ("21", "1.0", "5.4"): (
        "block",
        [_NOTE_OLD_VAT],
    ),
    # Modern stack with ancient UE 4.x — shader interface incompatibility
    ("21", "3.0", "4.0"): (
        "block",
        [_NOTE_OLD_UE],
    ),
    # Old Houdini 18 — not supported by modern Labs VAT 3.x pipeline
    ("18", "3.0", "5.4"): (
        "block",
        [_NOTE_OLD_HOUDINI],
    ),
    # All-old combination — doubly blocked
    ("18", "1.0", "4.0"): (
        "block",
        [_NOTE_OLD_HOUDINI, _NOTE_OLD_VAT, _NOTE_OLD_UE],
    ),
    # Ancient Houdini 17
    ("17", "3.0", "5.4"): (
        "block",
        [_NOTE_ANCIENT_HOUDINI],
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def skew_verdict(
    houdini: str,
    labs_vat: str,
    ue: Optional[str] = None,
) -> tuple:  # tuple[str, list[str]]
    """Look up the compatibility verdict for a Houdini / Labs VAT / UE version triple.

    All version strings are normalized to their major (Houdini) or major.minor
    (UE, Labs VAT) components before lookup, so patch-level variants behave
    consistently within the same support family.

    Args:
        houdini : Houdini version string, e.g. "21.0.456", "21.5.999", "21".
        labs_vat: SideFX Labs VAT version string, e.g. "3.0", "3.0.5".
        ue      : Unreal Engine version string, e.g. "5.4", "5.4.4";
                  None when the caller has not specified a UE target.

    Returns:
        A 2-tuple (verdict, notes):
          verdict : one of "ok", "warn", "block"
          notes   : list of human-readable strings explaining the verdict

    Behavior for unknown triples:
        Any combination not in the known-support table returns "warn" with a
        non-empty notes list.  Silence on unknown inputs is forbidden —
        "warn" is the verify-before-trust signal.
    """
    h_key = _normalize_houdini(houdini)
    vat_key = _normalize_vat(labs_vat)
    ue_key = _normalize_ue(ue) if ue is not None else None

    lookup_key = (h_key, vat_key, ue_key)
    if lookup_key in _TABLE:
        verdict, notes = _TABLE[lookup_key]
        return verdict, list(notes)  # return a fresh copy

    # Unknown combination — verify-before-trust: always "warn", never "ok"
    unknown_parts = []
    if h_key not in {k[0] for k in _TABLE}:
        unknown_parts.append(f"Houdini {houdini!r} (major {h_key!r})")
    if vat_key not in {k[1] for k in _TABLE}:
        unknown_parts.append(f"Labs VAT {labs_vat!r} (family {vat_key!r})")
    if ue_key is not None and ue_key not in {k[2] for k in _TABLE if k[2] is not None}:
        unknown_parts.append(f"UE {ue!r} (family {ue_key!r})")
    if ue_key is None:
        unknown_parts.append("no UE target specified — UE compatibility unverified")

    if unknown_parts:
        reason = "; ".join(unknown_parts)
    else:
        reason = (
            f"combination ({houdini!r}, {labs_vat!r}, {ue!r}) "
            "not in the validated support table"
        )

    warn_note = (
        f"Unverified combination — verify before bake: {reason}. "
        "Check the SideFX Labs VAT release notes for supported Houdini / UE pairings."
    )
    return "warn", [warn_note]
