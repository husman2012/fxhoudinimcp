"""FR-1's promise, enforced: the KineFX probe list must name what the handler USES.

Unit: fork-kinefx-fr1 (follow-on to the H22 migration's Phase 6)

FR-1 (110 spec): "Node probe (read-only). `houdini_kinefx_probe` returns the Houdini
build and confirms every KineFX/APEX node type **used** resolves."

It does not. Measured 2026-07-15 against the shipping build (H21.0.729 + H22.0.368) and
the handler's own AST:

  * `motiontransform` is LISTED but exists in NO node-type category on EITHER build, and
    the handler never creates it. The live probe dutifully reports `false` forever — the
    probe is honest; the FR-1 spec named a node Houdini never shipped. No node carries
    the label "Motion Transform"; the nearest are `kinefx::fktransfer` ("FK Transfer")
    and `kinefx::biped_retarget` ("Biped Retarget"), and picking one would be a guess.
    The retarget chain the handler really builds is
    `kinefx::rigmatchpose -> [kinefx::mappoints] -> kinefx::fullbodyik`.
  * `rigmatchpose` is listed BARE while the handler creates `kinefx::rigmatchpose`. The
    bare form only resolves through Houdini's un-namespaced fallback, so the probe
    reports on a *different string* than the code runs.
  * `kinefx::mappoints` and `kinefx::fullbodyik` are USED but NOT listed — the probe is
    blind to two of the three nodes in its own retarget chain.

The list was duplicated in the handler and in tests/kinefx_hython_smoke.py, which is how
they drifted apart. There is now ONE source of truth and this test pins it to the AST of
the handler, so the list cannot silently diverge from the code again.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

FORK = pathlib.Path(__file__).resolve().parents[3]
HANDLER = (
    FORK / "houdini" / "scripts" / "python" / "fxhoudinimcp_server"
    / "handlers" / "character_handlers.py"
)
SMOKE = FORK / "tests" / "kinefx_hython_smoke.py"

# Exists in NO category on H21.0.729 or H22.0.368 (hython-probed).
NONEXISTENT = {"motiontransform"}

# Probed present + non-deprecated on BOTH builds.
REAL = {
    "kinefx::fbxcharacterimport",
    "kinefx::fbxanimimport",
    "kinefx::secondarymotion",
    "kinefx::rigmatchpose",
    "kinefx::mappoints",
    "kinefx::fullbodyik",
    "bonedeform",
    "apex::autorigcomponent",
}


@pytest.fixture(scope="module")
def handler_src() -> str:
    assert HANDLER.is_file(), HANDLER
    return HANDLER.read_text(encoding="utf-8")


def _probe_list(src: str) -> list[str]:
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Assign) and any(
            getattr(t, "id", "") == "_KINEFX_NODE_TYPES" for t in n.targets
        ):
            return [e.value for e in n.value.elts if isinstance(e, ast.Constant)]
    raise AssertionError("_KINEFX_NODE_TYPES not found in the handler")


def _created(src: str) -> set[str]:
    """Node types the handler REALLY creates (AST, so comments cannot fake it)."""
    out = set()
    for n in ast.walk(ast.parse(src)):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "createNode"
            and n.args
            and isinstance(n.args[0], ast.Constant)
            and isinstance(n.args[0].value, str)
        ):
            out.add(n.args[0].value)
    out.discard("geo")  # the container, not a KineFX type
    return out


class TestProbeListIsReal:
    def test_no_nonexistent_type_is_probed(self, handler_src):
        bad = set(_probe_list(handler_src)) & NONEXISTENT
        assert not bad, (
            f"the probe lists {sorted(bad)}, which exists in NO node-type category on "
            "H21 or H22. The probe reports `false` for it forever — honestly, but the "
            "FR-1 spec named a node Houdini never shipped. The real retarget chain is "
            "kinefx::rigmatchpose -> [kinefx::mappoints] -> kinefx::fullbodyik."
        )

    def test_every_probed_type_is_a_verified_real_type(self, handler_src):
        unknown = set(_probe_list(handler_src)) - REAL
        assert not unknown, (
            f"probed types not verified to exist on H21+H22: {sorted(unknown)}. "
            "Every entry must be probe-confirmed against the shipping build."
        )


class TestProbeListMatchesWhatTheHandlerUses:
    """FR-1's actual promise: 'confirms every KineFX/APEX node type USED resolves'."""

    def test_no_used_type_is_unprobed(self, handler_src):
        missing = _created(handler_src) - set(_probe_list(handler_src))
        assert not missing, (
            f"the handler creates {sorted(missing)} but the probe does not check them — "
            "FR-1 promises to confirm every type USED resolves, so the probe is blind "
            "to part of its own retarget chain."
        )

    def test_probed_names_are_the_names_the_handler_actually_creates(self, handler_src):
        """Bare `rigmatchpose` resolves only via Houdini's un-namespaced fallback, so
        probing it reports on a different string than the code runs."""
        listed = set(_probe_list(handler_src))
        assert "rigmatchpose" not in listed or "kinefx::rigmatchpose" in listed, (
            "the probe lists bare `rigmatchpose` while the handler creates "
            "`kinefx::rigmatchpose` — probe the string the code uses"
        )
        assert "kinefx::rigmatchpose" in listed


class TestSingleSourceOfTruth:
    """The list was duplicated in the handler and the hython smoke — which is exactly
    how they drifted. One definition, imported."""

    def test_smoke_does_not_redefine_the_list(self):
        if not SMOKE.is_file():
            pytest.skip("hython smoke not present")
        src = SMOKE.read_text(encoding="utf-8")
        redefined = [
            n for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", "") == "_KINEFX_NODE_TYPES" for t in n.targets)
            and isinstance(n.value, (ast.List, ast.Tuple))
        ]
        assert not redefined, (
            "tests/kinefx_hython_smoke.py redefines _KINEFX_NODE_TYPES as a literal — "
            "import it from the handler instead; the duplicate is how the list drifted "
            "from the code."
        )
