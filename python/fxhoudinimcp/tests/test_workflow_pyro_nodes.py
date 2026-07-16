"""Pin the Pyro builder's node set against the shipping Houdini surface.

Unit: fork-pyro-sparse (Homedini _plans/fork-pyro-sparse.plan.json)

WHY THESE EXIST
---------------
`workflow_handlers._setup_pyro_sim_dop` builds the classic DOP Pyro chain. Probed on
the live H22 build (22.0.368) it requests three node types the build itself reports as
DEPRECATED, and the deprecation is SILENT — `createNode("smokeobject")` still succeeds,
so the handler's `except hou.OperationFailed` fallback never fires and nothing warns:

    smokeobject          -> smokeobject_sparse   (deprecated 22.0)
    smokeconfigureobject -> smokeobject_sparse   (deprecated 22.0)
    pyrosolver::2.0      -> pyrosolver_sparse    (deprecated 22.0)
    sourcevolume         -> volumesource         (deprecated 17.0 (!))

AND a real bug rides along: `sourcevolume`'s source-path parm is **`source_path`**, but
the handler tries `("sop_path", "soppath", "geometry")` via `_set_parm_safe`, which
returns False silently on a miss (its warning only fires when a parm EXISTS but set()
throws). So the DOP path has always built a source volume that is **never pointed at
the source geometry**. `volumesource` carries `soppath` — the swap makes the existing
loop start working.

The handler had ZERO test coverage. These are source-level assertions (the
`pytest-model` surface — no `hou` at import), so they run on bare CI. The live wiring
is covered by the hython-smoke in the same unit.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

HANDLER = (
    pathlib.Path(__file__).resolve().parents[3]
    / "houdini" / "scripts" / "python" / "fxhoudinimcp_server" / "handlers"
    / "workflow_handlers.py"
)

# Verified on H21.0.729 AND H22.0.368 via hython (nodeType.deprecationInfo()).
# H22 deprecated the first three; sourcevolume has been deprecated since H17.
DEPRECATED_ON_H22 = {
    "smokeobject": "smokeobject_sparse",
    "smokeconfigureobject": "smokeobject_sparse",
    "pyrosolver::2.0": "pyrosolver_sparse",
    "sourcevolume": "volumesource",
}

# Exists in NO node-type category on either build -> its createNode raises every call.
NONEXISTENT = {"pyrosolver::3.0"}

# The parm names the handler tries for the source-volume's SOP path. volumesource
# carries `soppath`; sourcevolume does NOT carry any of them (its parm is `source_path`).
SOURCE_PATH_CANDIDATES = ("sop_path", "soppath", "geometry")


@pytest.fixture(scope="module")
def source() -> str:
    assert HANDLER.is_file(), f"handler not found: {HANDLER}"
    return HANDLER.read_text(encoding="utf-8")


def _fn(src: str, name: str) -> ast.FunctionDef:
    """The AST of ONE function. Codex review finding #4: an unscoped module-wide walk
    lets the SOP builder satisfy a DOP-path assertion (its `pyrosolver.setInput(0,
    pyrosource, 0)` is CORRECT — different node, different connector). Scope or you are
    asserting about the wrong function."""
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(f"function {name!r} not found")


def _created_types(src: str) -> set[str]:
    """Every node type the module REALLY passes to createNode().

    AST-based, not regex — the house precedent is check_pure_logic_boundary.py:
    "AST-based (not regex) so commented-out / string-literal mentions ... never trip
    it". Learned here the hard way: a regex over raw text matched the explanatory
    comments *documenting* the deprecated names and reported the fix as unfixed.
    """
    types: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "createNode"):
            continue
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(
            node.args[0].value, str
        ):
            types.add(node.args[0].value)
    return types


class TestNoDeprecatedNodeTypes:
    def test_builder_requests_no_deprecated_type(self, source):
        used = _created_types(source) & set(DEPRECATED_ON_H22)
        assert not used, (
            "the Pyro builder creates node types the shipping Houdini reports as "
            "DEPRECATED: "
            + ", ".join(f"{t} -> use {DEPRECATED_ON_H22[t]}" for t in sorted(used))
            + ". Note createNode() still SUCCEEDS for these, so the existing "
            "`except hou.OperationFailed` fallback never fires (CL-023)."
        )

    def test_sparse_replacements_are_the_ones_used(self, source):
        used = _created_types(source)
        for replacement in ("smokeobject_sparse", "pyrosolver_sparse", "volumesource"):
            assert replacement in used, f"expected the builder to create {replacement}"


class TestNoNonexistentNodeTypes:
    def test_builder_requests_no_type_that_exists_nowhere(self, source):
        used = _created_types(source) & NONEXISTENT
        assert not used, (
            f"the builder requests {sorted(used)}, which exists in NO node-type "
            "category on H21 or H22 — the createNode raises on every call and the "
            "except-branch always fires. Remove the dead attempt."
        )


class TestSourceGeometryIsActuallyWired:
    """The bug the swap fixes: the source volume must carry a parm the handler sets."""

    def test_source_volume_node_carries_a_parm_the_handler_sets(self, source):
        used = _created_types(source)
        assert "volumesource" in used, (
            "the builder must create `volumesource` — `sourcevolume`'s source-path parm "
            "is `source_path`, which is not among the "
            f"{SOURCE_PATH_CANDIDATES} the handler tries, so the source geometry is "
            "NEVER wired and _set_parm_safe swallows the miss silently."
        )
        assert "sourcevolume" not in used

    def test_handler_still_sets_the_source_path_parm(self, source):
        """Guard the other direction: the swap must not 'simplify' the parm loop away —
        wiring the source is the point."""
        assert "soppath" in source, (
            "the builder must still set a source-path parm on the source-volume node; "
            "volumesource carries `soppath`"
        )


class TestSolverInputsAreSemantic:
    """SUPERSEDED PREMISE — this class originally asserted "input 0 only", which
    ENSHRINED the bug. The cross-vendor (Codex) review caught it and a live probe
    confirmed: the solver's inputs are SEMANTIC, not positional —

        pyrosolver_sparse : ('Objects', 'Advection', 'Sourcing', 'Forces')
        pyrosolver::2.0   : ('Object', 'Pre-solve', 'Velocity Update',
                             'Advection', 'Sourcing (post-solve)')

    The old code merged the smoke object AND the source volume and fed the merge to
    input 0 — handing a SOURCE to the OBJECTS connector. Wrong on the old solver too.
    Live-verified on 22.0.368: source on input 2 -> dopnet cooks, 1 sim object;
    merged into input 0 -> zero sim objects. Counting inputs (4-vs-5) was the wrong
    question entirely; reading the labels was the right one.
    """

    def _solver_inputs(self, source) -> dict[str, str]:
        """{arg-repr: node-var} for every pyrosolver.setInput(...) in the module."""
        out = {}
        for n in ast.walk(_fn(source, "_setup_pyro_sim_dop")):   # scoped, see _fn
            if not (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "setInput"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "pyrosolver"
                and len(n.args) >= 2
            ):
                continue
            key = n.args[0].id if isinstance(n.args[0], ast.Name) else ast.dump(n.args[0])
            val = n.args[1].id if isinstance(n.args[1], ast.Name) else "?"
            out[key] = val
        return out

    def test_source_goes_to_the_sourcing_connector_not_objects(self, source):
        wiring = self._solver_inputs(source)
        assert wiring.get("_SOLVER_IN_SOURCING") == "source_vol", (
            "the volume source must be wired to the solver's SOURCING connector "
            f"(_SOLVER_IN_SOURCING), got {wiring!r}. Feeding it to Objects is the bug "
            "that made the sim cook with zero sourcing."
        )

    def test_smoke_object_goes_to_the_objects_connector(self, source):
        wiring = self._solver_inputs(source)
        assert wiring.get("_SOLVER_IN_OBJECTS") == "smokeobj", (
            f"the smoke object must be wired to the OBJECTS connector, got {wiring!r}"
        )

    def test_connectors_are_named_constants_not_bare_indices(self, source):
        """An index is not self-documenting; `setInput(0, merge)` is how the bug hid."""
        wiring = self._solver_inputs(source)
        bare = [k for k in wiring if k.startswith("Constant")]
        assert not bare, f"solver inputs must use the named constants, found {bare}"
        assert "_SOLVER_IN_SOURCING = 2" in source
        assert "_SOLVER_IN_OBJECTS = 0" in source


class TestNoDeadResizeWiring:
    """gasresizefluiddynamic has ZERO inputs (inputLabels == ()), so the old
    `resize.setInput(0, pyrosolver, 0)` raised hou.InvalidInput on EVERY call and a
    broad `except Exception` swallowed it. The sparse solver resizes internally."""

    def test_no_external_resize_node_is_created(self, source):
        dop_types = {
            n.args[0].value
            for n in ast.walk(_fn(source, "_setup_pyro_sim_dop"))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "createNode"
            and n.args
            and isinstance(n.args[0], ast.Constant)
        }
        assert "gasresizefluiddynamic" not in dop_types, (
            "gasresizefluiddynamic takes no inputs — wiring the solver into it always "
            "raises; the sparse solver resizes internally"
        )


class TestSolverIsTheDisplayNode:
    """Codex Blocker #1 (second half). The dopnet wires no output to the solver, so
    without the display flag the network cooks to NOTHING while the handler still
    returns success. Isolated on 22.0.368: flag unset -> 0 sim objects; set -> 1."""

    def test_solver_carries_the_display_flag(self, source):
        calls = [
            n for n in ast.walk(_fn(source, "_setup_pyro_sim_dop"))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "setDisplayFlag"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "pyrosolver"
        ]
        assert calls, (
            "pyrosolver must carry the dopnet display flag — without it the built "
            "network cooks to zero sim objects and the handler reports success anyway"
        )
        assert any(
            a.value is True for c in calls for a in c.args if isinstance(a, ast.Constant)
        ), "setDisplayFlag must be called with True"
