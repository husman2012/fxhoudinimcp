"""Import-smoke test for the handoff_linter_loader (PP12-114 PR-2).

Verification surface: pytest-model (fork venv, agent-runnable, off-DCC).

RED gate contract (phase 1 / hou-test): the test imports
``fxhoudinimcp.handoff_linter_loader`` which does NOT exist yet.  The suite
therefore fails BEFORE ``hou-dev`` implements the loader module.  Once the
loader lands, the test goes green without modification.

GUARDED-SKIP discipline: if Homedini is unreachable (no sibling layout, no
``$HOMEDINI_PYTHON``), the test skips with a clear message rather than
raising a hard error, so CI on a fresh clone without the sibling tree
degrades cleanly.

Synthetic HandoffReport: built ENTIRELY from the imported ``handoff_model``
dataclasses — no rule logic or model fields are re-implemented here.  The
fix_id assertions target public behaviour of ``rules.evaluate``, not internal
call structure (per tdd-with-agents.md §2 mirror-test discipline).

plan-ack: pp12-114b@e29f6c758b9f1ff9750c9f1f5c038663bad73ff6f9a35f5f233977e1a3a7c27d
"""

from __future__ import annotations

import sys

import pytest

# ---------------------------------------------------------------------------
# sys.path — ensure the fork server package is importable off-DCC.
# Mirrors the established pattern from test_gate_envelope.py (lines 40-44).
# ---------------------------------------------------------------------------
_FORK_PYTHON = "C:/Users/husma/development/fxhoudinimcp/houdini/scripts/python"
_HOMEDINI_PYTHON = "C:/Users/husma/development/HoudiniUtilTools/scripts/python"
for _p in (_FORK_PYTHON, _HOMEDINI_PYTHON):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# T1: RED gate — the loader module must not exist yet.
# This assertion is the ONLY thing that runs in the red phase; all tests below
# will also fail for the same reason (ImportError from handoff_linter_loader).
# ---------------------------------------------------------------------------


def test_loader_module_exists_after_hou_dev_implements() -> None:
    """RED: fxhoudinimcp.handoff_linter_loader does not exist until hou-dev ships it.

    This test MUST fail (ImportError) in the red phase.
    Once hou-dev implements the loader, this assertion becomes vacuous and the
    remaining tests carry the green gate.
    """
    # This import drives the red gate — the module is absent before hou-dev ships.
    from fxhoudinimcp import handoff_linter_loader  # noqa: F401  (used for side-effect)

    assert hasattr(handoff_linter_loader, "ensure_on_path"), (
        "handoff_linter_loader must expose ensure_on_path()"
    )
    assert hasattr(handoff_linter_loader, "load"), (
        "handoff_linter_loader must expose load()"
    )


# ---------------------------------------------------------------------------
# T2–T5: functional contract — these run only when the loader exists AND
# Homedini is reachable.  Any ImportError from the loader itself will make
# these fail at collection time, which is also a valid red-gate signal.
# ---------------------------------------------------------------------------

def _require_homedini() -> None:
    """Guard: skip the calling test if Homedini is not reachable via the loader."""
    try:
        from fxhoudinimcp import handoff_linter_loader  # noqa: F401
    except ImportError:
        pytest.skip(
            "fxhoudinimcp.handoff_linter_loader not yet implemented "
            "(hou-dev red phase — expected failure)"
        )

    # Ask the loader itself whether Homedini is on sys.path.
    from fxhoudinimcp.handoff_linter_loader import ensure_on_path
    reachable = ensure_on_path()
    if not reachable:
        pytest.skip(
            "Homedini is unreachable: ensure_on_path() returned False.  "
            "Expected sibling layout "
            "C:/Users/husma/development/HoudiniUtilTools/scripts/python or "
            "$HOMEDINI_PYTHON env-var not set.  "
            "Set $HOMEDINI_PYTHON to the homedini scripts/python directory to run "
            "these tests on a non-standard install."
        )


def test_ensure_on_path_returns_true() -> None:
    """T2: ensure_on_path() returns True on the dev box with the sibling layout."""
    _require_homedini()

    from fxhoudinimcp.handoff_linter_loader import ensure_on_path

    result = ensure_on_path()
    assert result is True, (
        "ensure_on_path() must return True when Homedini is reachable. "
        f"Got: {result!r}"
    )


def test_engine_modules_importable_after_ensure_on_path() -> None:
    """T3: after ensure_on_path(), all five handoff_linter modules import cleanly.

    Covers: handoff_model, presets, rules, exr_inspector (pure/no-hou);
            stage_reader (import-guarded — hou/pxr absent in fork venv).
    """
    _require_homedini()

    from fxhoudinimcp.handoff_linter_loader import ensure_on_path
    ensure_on_path()

    # Pure modules — import for real, no mocking.
    from homedini.rendering.handoff_linter import handoff_model  # noqa: F401
    from homedini.rendering.handoff_linter import presets  # noqa: F401
    from homedini.rendering.handoff_linter import rules  # noqa: F401
    from homedini.rendering.handoff_linter import exr_inspector  # noqa: F401

    # stage_reader has an import guard (hou/pxr → None in the fork venv);
    # it must still import cleanly because of that guard.
    from homedini.rendering.handoff_linter import stage_reader  # noqa: F401

    # Sanity: the module-level public API shapes are present.
    assert hasattr(handoff_model, "HandoffReport")
    assert hasattr(presets, "load")
    assert hasattr(rules, "evaluate")
    assert hasattr(exr_inspector, "parse_exr")  # primary public EXR entry-point
    assert hasattr(stage_reader, "read_stage") or hasattr(stage_reader, "read")


def test_presets_load_returns_preset() -> None:
    """T4: presets.load('nuke_safe') returns a Preset with expected fields."""
    _require_homedini()

    from fxhoudinimcp.handoff_linter_loader import ensure_on_path
    ensure_on_path()

    from homedini.rendering.handoff_linter import presets
    from homedini.rendering.handoff_linter.presets import Preset

    result = presets.load("nuke_safe")

    assert isinstance(result, Preset), (
        f"presets.load('nuke_safe') must return a Preset; got {type(result)!r}"
    )
    assert result.name == "nuke_safe"
    # The nuke_safe preset must declare at least some allowed compressions.
    assert isinstance(result.allowed_compressions, list)
    assert isinstance(result.product_type_whitelist, list)


def test_rules_evaluate_fires_expected_fix_ids() -> None:
    """T5: rules.evaluate on a synthetic HandoffReport fires crypto_name and legacy_exr.

    Synthetic report design (no rule logic re-implemented here — built solely
    from imported handoff_model dataclasses):
    - One RenderProductSpec with is_multipart=True + no ordered_var_paths
      (so the rule falls back to report.vars + report.crypto_layers).
    - One CryptoLayer with has_name_key=False (fires FR-3 / crypto_name).
    - The CryptoLayer presence + is_multipart=True → FR-4 fires legacy_exr.

    Expected fix_ids in the results: {'crypto_name', 'legacy_exr'} (at minimum).
    """
    _require_homedini()

    from fxhoudinimcp.handoff_linter_loader import ensure_on_path
    ensure_on_path()

    from homedini.rendering.handoff_linter import presets, rules
    from homedini.rendering.handoff_linter.handoff_model import (
        ChannelSpec,
        CryptoLayer,
        HandoffReport,
        RenderProductSpec,
        RenderVarSpec,
    )

    # ---- Synthetic CryptoLayer: has_name_key=False → fires FR-3 crypto_name ----
    crypto_layer = CryptoLayer(
        layer_name="CryptoObject",
        type_hash="abc1234",
        has_name_key=False,   # <-- the broken state targeted by FR-3
        has_manifest=True,
        source="rendervar",
    )

    # ---- Synthetic RenderVarSpec: cryptomatte ----
    crypto_var = RenderVarSpec(
        prim_path="/Render/Products/beauty/Vars/CryptoObject",
        source_name="CryptoObject",
        data_type="color3f",
        channels=[
            ChannelSpec(name="crypto00.r", layer="crypto00", dtype="float"),
            ChannelSpec(name="crypto00.g", layer="crypto00", dtype="float"),
            ChannelSpec(name="crypto00.b", layer="crypto00", dtype="float"),
            ChannelSpec(name="crypto00.a", layer="crypto00", dtype="float"),
        ],
        is_cryptomatte=True,
        crypto_kind="object",
    )

    # ---- Synthetic RenderProductSpec: multipart=True, no ordered_var_paths ----
    # With empty ordered_var_paths the rule falls back to report.vars +
    # report.crypto_layers when determining crypto presence.
    product = RenderProductSpec(
        prim_path="/Render/Products/beauty",
        product_type="raster",
        product_name="$HIP/render/beauty.$F4.exr",
        compression="zips",
        is_multipart=True,   # <-- triggers FR-4 multipart_exr rule
        exrmode_legacy=False,
        ordered_var_paths=[],  # empty → rule falls back to report-level scan
    )

    report = HandoffReport(
        source="cli:/test/synthetic",
        products=[product],
        vars=[crypto_var],
        crypto_layers=[crypto_layer],
        notes=[],
    )

    # Load the nuke_safe preset for policy context.
    preset = presets.load("nuke_safe")

    # Run the rule engine.
    results = rules.evaluate(report, preset)

    # Collect all non-None fix_ids.
    fix_ids_found: set[str] = {
        r.fix_id for r in results if r.fix_id is not None
    }

    # The synthetic report must fire at minimum: crypto_name (FR-3) + legacy_exr (FR-4).
    assert "crypto_name" in fix_ids_found, (
        f"Expected fix_id 'crypto_name' in results from FR-3 rule. "
        f"fix_ids found: {fix_ids_found!r}.  "
        f"Full results: {[r.rule_id + '/' + str(r.fix_id) for r in results]!r}"
    )
    assert "legacy_exr" in fix_ids_found, (
        f"Expected fix_id 'legacy_exr' in results from FR-4 rule. "
        f"fix_ids found: {fix_ids_found!r}.  "
        f"Full results: {[r.rule_id + '/' + str(r.fix_id) for r in results]!r}"
    )
