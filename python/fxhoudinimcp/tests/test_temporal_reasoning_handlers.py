"""Handler-level (mocked-hou) pytest tests for describe_sim_events and
assert_simulation (PP12-117 PR-2, READONLY) -- the REVISION-2
lockedFieldContract red suite.

Unit: pp12-117b
testVerificationSurface: pytest-model (rung 1) + hython-smoke (rung 2, the
    skip-guarded TestAssertSimulationHythonSmoke class at the bottom of
    this file)
planSha: 596ba5d0454501373c5b30de5841e1da66f96e2342aa058fddfef5bb1b26eb2e

These tests import the HANDLER module (temporal_reasoning_handlers.py)
with a mocked `hou` module installed into sys.modules BEFORE import, per
test-fixture-conventions.md Sec.2.3 (`monkeypatch.setitem(sys.modules,
'hou', fake)`). This lets the handler's Python control flow -- geometry
resolution for BOTH hou.ObjNode and hou.SopNode paths, the per-metric
scalar reads, the per-assertion node routing, the expect-normalization,
the frame_range validation, the frame-save/step/restore discipline, and
the {ok:false,error} vs ValueError-propagates boundary -- be exercised
off-DCC without a real Houdini session.

This is a genuinely mocked-hou rung (test-fixture-conventions.md Sec.2 --
the handler cannot be fully split into *_model.py because it calls
hou.node()/hou.setFrame()/geo.boundingBox() as its core job), NOT a
substitute for the real hython-smoke rung (the skip-guarded class at the
bottom) or the MANDATORY live-MCP subprocess rung named in the plan's
lockedFieldContract "verification ladder" clause (orchestrator-run, out of
scope for this red-test unit).

Tests exercise the REAL dispatcher (`fxhoudinimcp_server.dispatcher.dispatch`)
-- NOT a direct `handler({dict})` call -- per the plan's HANDLER-test
directive: `dispatch(cmd, {params})` calls `handler(**params)`, the exact
calling convention the shipped MCP fork uses end-to-end (the PP12-110
4-bug convention class: a direct call bypasses the dispatcher's own
calling convention and would miss a signature mismatch). The ONE exception
is the ValueError-propagation class below, which mirrors the shipped
spatial_reasoning exemplar's dual check (a direct handler call via
pytest.raises PLUS the dispatcher's error envelope).

Coverage this file pins (plan pp12-117b lockedFieldContract, REVISION 2):
  - describe_sim_events is a PURE delegate (no hou) to
    temporal_reasoning_model.describe_sim_events() -- the exact SPEC 4.1
    vocab dict, verbatim.
  - assert_simulation, cook_job non-null -> the documented {ok:false,
    error:...} BEFORE frame_range is even validated (ordering pin).
  - assert_simulation, frame_range validation (Major-7): inverted
    [10,1]/non-int/bool -> ValueError; a wrong-length list -> ValueError;
    single-frame [f,f] is allowed.
  - assert_simulation, assertions=[] -> NO frame stepping (no
    hou.setFrame call at all), returns evaluate_assertions([]) verbatim
    (pass:true, results:[]).
  - assert_simulation, expect-normalization (Blocker-6): a top-level
    predicate (the spec's own {"metric":"velocity_bounds","max":250}
    example) is normalized into expect={"max":250}; BOTH expect and a
    top-level predicate present -> ValueError naming the conflict.
  - assert_simulation, per-assertion `node` routing (Blocker-3): an
    assertion's own `node` field (when present) is the read source instead
    of `network`; TWO same-metric assertions on DIFFERENT nodes build TWO
    independent series in INPUT order.
  - assert_simulation, per-metric SCALAR reads (the METRIC->SOURCE table):
    piece_count (DETERMINISTIC precedence: unique non-empty prim `name` ->
    unique non-empty point `name` -> primitivecount), point_count
    (pointcount intrinsic), velocity_bounds (max |v| over points),
    bbox_over_time (the world-AABB max extent), mass_conservation (sum of
    the `mass` point attribute) -- each pinned via an outcome-based
    assertion on the returned series, not a call-shape assertion on which
    hou.* methods fired (tdd-with-agents.md Sec.2 mirror-test ban).
  - assert_simulation, a missing `v` attribute -> the WHOLE call degrades
    to one {ok:false,error} (Major-10 -- raised AS hou.OperationFailed
    inside the read phase, never a partial result).
  - assert_simulation, a missing `mass` attribute -> the WHOLE call
    degrades to one {ok:false,error} (Major-10, same boundary).
  - assert_simulation, the deferred field_stats/constraint_count metrics
    (Blocker-4/5) -> the EXACT structured "unsupported in PR-2" {ok:false}
    strings named in the locked contract.
  - assert_simulation, an unresolvable node/network -> {ok:false,error}
    WITHOUT raising.
  - assert_simulation, the saved frame is RESTORED in a finally -- both on
    the success path AND on an error path raised mid-stepping (proving the
    restore is NOT merely the happy-path tail).
  - assert_simulation, an unknown-metric ValueError (a metric string the
    model itself does not recognize) PROPAGATES -- is NOT folded into
    {ok:false} -- verified both via a direct handler call
    (pytest.raises(ValueError)) and via the dispatcher's standard error
    envelope (status:"error", error.code:"ValueError").
  - Both commands are registered Capability.READONLY.

Assertions target the RETURNED DICT (the public contract) -- never which
hou.* methods were called or in what order (tdd-with-agents.md Sec.2
mirror-test ban; test-fixture-conventions.md Sec.2.3 discipline). The
per-metric read tests are deliberately OUTCOME-based (a wrong read formula
produces an observably wrong series value) so they cannot be gamed by an
implementation that merely LOOKS like it follows the contract.
"""

from __future__ import annotations

import math
import os
import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Minimal fake hou primitives -- local to THIS file (test-fixture-
# conventions.md Sec.2.3 discipline: a test file defines its own fixtures
# rather than importing a sibling test file's private helpers, avoiding
# cross-module class-identity coupling on isinstance() checks).
# ---------------------------------------------------------------------------

class _FakeOperationFailed(Exception):
    """Stand-in for hou.OperationFailed -- a plain Exception subclass is
    sufficient since the handler only does `except hou.OperationFailed`."""


class _FakeSopNode(MagicMock):
    """Stand-in for hou.SopNode -- a distinct type so isinstance(n,
    hou.SopNode) is True only for these instances (not hou.ObjNode
    instances), matching Houdini's real class hierarchy discrimination."""


class _FakeObjNode(MagicMock):
    """Stand-in for hou.ObjNode -- see _FakeSopNode."""


class _FakeVector3(tuple):
    """A tuple-based hou.Vector3 stand-in (indexed access only -- no
    matrix multiply needed here since the bbox_over_time fixture below
    uses an IDENTITY-ROTATION world transform, so world extents equal
    local extents regardless of translate)."""

    def __new__(cls, x, y=None, z=None):
        if y is None and z is None:
            x, y, z = x[0], x[1], x[2]
        return super().__new__(cls, (float(x), float(y), float(z)))


class _FakeMatrix4:
    """A translate-only affine-transform stand-in for hou.Matrix4 (zero
    rotation -- bbox_over_time's fixture below deliberately uses an
    identity rotation so the expected world-AABB extents are computable by
    hand as exactly the LOCAL extents, independent of translate)."""

    def __init__(self, translate=(0.0, 0.0, 0.0), rotate_deg=(0.0, 0.0, 0.0)):
        self._translate = tuple(translate)
        self._rotate_deg = tuple(rotate_deg)

    def at(self, r, c):
        rows = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        rows[3][0], rows[3][1], rows[3][2] = self._translate
        return rows[r][c]

    def extractTranslates(self):
        return _FakeVector3(*self._translate)

    def extractRotates(self):
        return _FakeVector3(*self._rotate_deg)


class _FakeBoundingBox:
    """hou.BoundingBox stand-in: .minvec() / .maxvec() only."""

    def __init__(self, minv, maxv):
        self._minv = _FakeVector3(*minv)
        self._maxv = _FakeVector3(*maxv)

    def minvec(self):
        return self._minv

    def maxvec(self):
        return self._maxv


def _node_lookup(mapping):
    """A hou.node() side_effect robust to either a positional or keyword
    `path` call convention."""

    def _fn(*args, **kwargs):
        path = args[0] if args else kwargs.get("path")
        return mapping.get(path)

    return _fn


class _FrameState:
    """Shared mutable frame-tracking state for the mock_hou fixture below.
    Starts at a value FAR from any tested frame_range (42.0) so a
    frame-restore assertion is meaningful (a handler that forgot to
    restore would leave hou.frame() sitting at the LAST stepped frame,
    never back at 42.0)."""

    def __init__(self, start: float = 42.0):
        self.current = start


# ---------------------------------------------------------------------------
# mock_hou fixture -- installs a fake `hou` module into sys.modules BEFORE
# the handler module is imported (test-fixture-conventions.md Sec.2.3).
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_hou(monkeypatch):
    """Install a mock `hou` module so temporal_reasoning_handlers.py loads
    off-DCC. Returns the mock so tests can configure hou.node.side_effect,
    etc. hou.frame()/hou.setFrame(f) are wired to a shared _FrameState so
    tests can assert genuine save/step/restore behavior (not merely that
    SOME frame value is returned)."""
    fake = MagicMock(name="hou")
    fake.OperationFailed = _FakeOperationFailed
    fake.SopNode = _FakeSopNode
    fake.ObjNode = _FakeObjNode

    state = _FrameState()
    fake._frame_state = state
    fake.frame.side_effect = lambda: state.current

    def _set_frame(f):
        state.current = float(f)

    fake.setFrame.side_effect = _set_frame
    monkeypatch.setitem(sys.modules, "hou", fake)

    # Ensure the fork's non-standard package roots are importable (mirrors
    # test_spatial_reasoning_handlers.py's sys.path bootstrap): pytest's
    # rootdir discovery does not put houdini/scripts/python (fxhoudinimcp_
    # server's home) on sys.path by default.
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    _pkg_python = os.path.join(_repo_root, "python")
    _houdini_handlers = os.path.join(_repo_root, "houdini", "scripts", "python")
    for _p in (_pkg_python, _houdini_handlers):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    return fake


def _import_handler_module():
    """Import temporal_reasoning_handlers fresh (relies on mock_hou already
    being installed in sys.modules AND sys.path already patched by the
    caller's mock_hou fixture)."""
    import importlib
    if "fxhoudinimcp_server.handlers.temporal_reasoning_handlers" in sys.modules:
        importlib.reload(sys.modules["fxhoudinimcp_server.handlers.temporal_reasoning_handlers"])
        return sys.modules["fxhoudinimcp_server.handlers.temporal_reasoning_handlers"]
    import fxhoudinimcp_server.handlers.temporal_reasoning_handlers as _handlers
    return _handlers


def _dispatch(command: str, params: dict):
    """Import the REAL dispatcher fresh each call (cheap -- it's already
    cached in sys.modules after the first import) and route through it,
    per the plan's HANDLER-test directive (NOT a direct handler({dict})
    call)."""
    from fxhoudinimcp_server.dispatcher import dispatch as _real_dispatch
    return _real_dispatch(command, params)


# ---------------------------------------------------------------------------
# Geometry fixture builders
# ---------------------------------------------------------------------------

def _make_geo(
    *,
    prim_names=None,
    point_names=None,
    v=None,
    mass=None,
    n_points=None,
    n_prims=None,
    bbox=None,
):
    """Build a plain (unrestricted) MagicMock geometry object exposing
    exactly the surface the METRIC->SOURCE table needs:
      - intrinsicValue('pointcount'/'primitivecount')
      - findPrimAttrib('name') / findPointAttrib('name'/'v'/'mass') --
        existence checks (None when the attribute is absent, matching the
        real hou.Geometry.findPointAttrib/findPrimAttrib contract)
      - prims() / points() iteration with .attribValue(name) per item
        (mirrors the shipped geometry_handlers.py:177-183 prim-name idiom
        and character_handlers.py:214-218's point-name idiom)
      - boundingBox() (for bbox_over_time via an ObjNode/displayNode path)

    Geo is intentionally NOT spec-bound (unlike the HoudiniBridge mock in
    the wrapper tests) -- the PP12-110 spec-bound-mock guard is specific to
    the bridge; a real hou.Geometry has a much larger surface this fixture
    does not need to fully replicate.
    """
    geo = MagicMock(name="fake_geo")

    resolved_n_points = n_points if n_points is not None else (
        len(v) if v is not None else (len(mass) if mass is not None else (
            len(point_names) if point_names is not None else 0
        ))
    )
    resolved_n_prims = n_prims if n_prims is not None else (
        len(prim_names) if prim_names is not None else 0
    )

    def _intrinsic(key):
        return {"pointcount": resolved_n_points, "primitivecount": resolved_n_prims}.get(key, 0)

    geo.intrinsicValue.side_effect = _intrinsic

    def _find_prim_attrib(name):
        return object() if (name == "name" and prim_names is not None) else None

    def _find_point_attrib(name):
        if name == "name" and point_names is not None:
            return object()
        if name == "v" and v is not None:
            return object()
        if name == "mass" and mass is not None:
            return object()
        return None

    geo.findPrimAttrib.side_effect = _find_prim_attrib
    geo.findPointAttrib.side_effect = _find_point_attrib

    prim_mocks = []
    for nm in (prim_names or []):
        p = MagicMock()

        def _prim_attrib_value(attr, _nm=nm):
            return _nm if attr == "name" else None

        p.attribValue.side_effect = _prim_attrib_value
        prim_mocks.append(p)
    geo.prims.return_value = prim_mocks

    point_mocks = []
    for i in range(resolved_n_points):
        pt = MagicMock()
        _pname = point_names[i] if point_names else None
        _v = tuple(v[i]) if v else None
        _mass = mass[i] if mass else None

        def _point_attrib_value(attr, _pname=_pname, _v=_v, _mass=_mass):
            if attr == "name":
                return _pname
            if attr == "v":
                return _v
            if attr == "mass":
                return _mass
            return None

        pt.attribValue.side_effect = _point_attrib_value
        point_mocks.append(pt)
    geo.points.return_value = point_mocks

    if bbox is not None:
        geo.boundingBox.return_value = _FakeBoundingBox(*bbox)

    return geo


def _make_sop_node(mock_hou, path, geo):
    """A SopNode whose .geometry() returns *geo* (static across all
    frames -- sufficient for most per-metric read tests, which only need
    to prove the SCALAR FORMULA is right, not that per-frame variance is
    read)."""
    node = mock_hou.SopNode(name=path.replace("/", "_"))
    node.path.return_value = path
    node.geometry.return_value = geo
    return node


def _make_obj_node(mock_hou, path, geo, translate=(0.0, 0.0, 0.0)):
    """An ObjNode whose displayNode().geometry() returns *geo*, with an
    IDENTITY-ROTATION worldTransform (translate arbitrary) -- per the
    METRIC->SOURCE table: 'if ObjNode use displayNode().geometry()'."""
    node = mock_hou.ObjNode(name=path.replace("/", "_"))
    node.path.return_value = path
    node.worldTransform.return_value = _FakeMatrix4(translate=translate, rotate_deg=(0.0, 0.0, 0.0))
    display = MagicMock(name=f"fake_display_{path}")
    display.geometry.return_value = geo
    node.displayNode.return_value = display
    return node


def _make_frame_reactive_pointcount_node(mock_hou, path):
    """A SopNode whose point count TRACKS the currently-set frame exactly
    (pointcount == int(hou.frame())) -- used to prove the handler performs
    a GENUINE per-frame read (hou.setFrame(f) before each read), not a
    single snapshot repeated across the series."""
    node = mock_hou.SopNode(name=path.replace("/", "_"))
    node.path.return_value = path

    def _geo():
        n = int(mock_hou._frame_state.current)
        return _make_geo(n_points=n, n_prims=0)

    node.geometry.side_effect = _geo
    return node


# ---------------------------------------------------------------------------
# PRIMARY RED GATE -- both handler functions must exist
# ---------------------------------------------------------------------------

class TestHandlerImport:
    def test_describe_sim_events_handler_importable(self, mock_hou):
        """temporal_reasoning_handlers.py must expose describe_sim_events.
        FAILS RED (ImportError/ModuleNotFoundError) until hou-dev implements
        it."""
        handlers = _import_handler_module()
        assert hasattr(handlers, "describe_sim_events"), (
            "temporal_reasoning_handlers.py must expose describe_sim_events."
        )
        assert callable(handlers.describe_sim_events)

    def test_assert_simulation_handler_importable(self, mock_hou):
        """temporal_reasoning_handlers.py must expose assert_simulation.
        FAILS RED until hou-dev implements it."""
        handlers = _import_handler_module()
        assert hasattr(handlers, "assert_simulation"), (
            "temporal_reasoning_handlers.py must expose assert_simulation."
        )
        assert callable(handlers.assert_simulation)


# ---------------------------------------------------------------------------
# Capability.READONLY registration
# ---------------------------------------------------------------------------

class TestCapabilityReadonly:
    """Both commands must be registered Capability.READONLY -- no 109
    gate (Blocker-2's explicit reversible-frame-evaluation exception)."""

    def test_describe_sim_events_capability_readonly(self, mock_hou):
        _import_handler_module()
        from fxhoudinimcp_server import dispatcher
        assert dispatcher.capability_of("describe_sim_events") == dispatcher.Capability.READONLY, (
            f"describe_sim_events must be registered Capability.READONLY, "
            f"got {dispatcher.capability_of('describe_sim_events')!r}."
        )

    def test_assert_simulation_capability_readonly(self, mock_hou):
        _import_handler_module()
        from fxhoudinimcp_server import dispatcher
        assert dispatcher.capability_of("assert_simulation") == dispatcher.Capability.READONLY, (
            f"assert_simulation must be registered Capability.READONLY, "
            f"got {dispatcher.capability_of('assert_simulation')!r}."
        )


# ---------------------------------------------------------------------------
# describe_sim_events -- a PURE delegate to the model (no hou touched)
# ---------------------------------------------------------------------------

class TestDescribeSimEventsDelegatesToModel:
    """describe_sim_events() is a pure delegate: `return
    temporal_reasoning_model.describe_sim_events()` -- no hou read at all.
    The dispatched result must equal the model's own return value exactly,
    AND match the SPEC 4.1 shape directly."""

    def test_dispatch_returns_the_exact_model_vocab_dict(self, mock_hou):
        _import_handler_module()
        from fxhoudinimcp import temporal_reasoning_model as model

        result = _dispatch("describe_sim_events", {})

        assert result["status"] == "success", f"describe_sim_events must not raise, got {result!r}"
        expected = model.describe_sim_events()
        assert result["data"] == expected, (
            f"describe_sim_events handler must return the model's exact SPEC 4.1 "
            f"vocab dict verbatim. Expected {expected!r}, got {result['data']!r}."
        )

    def test_vocab_shape_is_spec_4_1(self, mock_hou):
        _import_handler_module()
        result = _dispatch("describe_sim_events", {})
        data = result["data"]
        assert len(data["events"]) == 5, f"expected 5 event types, got {len(data['events'])}"
        assert data["triggers"] == ["stress_gt", "collision_with", "frame_eq", "field_gt"]
        assert data["assertions"] == [
            "piece_count", "constraint_count", "point_count", "velocity_bounds",
            "bbox_over_time", "field_stats", "mass_conservation",
        ]


# ---------------------------------------------------------------------------
# assert_simulation -- cook_job non-null (checked BEFORE frame_range, per
# the locked contract's step ordering "(0) cook_job ... (1) VALIDATE
# frame_range")
# ---------------------------------------------------------------------------

class TestAssertSimulationCookJobDocumentedUnavailable:
    """A non-null cook_job returns the documented {ok:false,error} and
    happens BEFORE frame_range validation -- pinned by pairing a non-null
    cook_job with an otherwise-INVALID frame_range and expecting the
    cook_job message, not a ValueError."""

    def test_non_null_cook_job_returns_documented_unavailable_message(self, mock_hou):
        handlers = _import_handler_module()
        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim", "frame_range": [1, 10],
            "assertions": [], "cook_job": "cook-9a",
        })
        assert result["status"] == "success", f"expected a normal (non-raising) result, got {result!r}"
        data = result["data"]
        assert data == {
            "ok": False,
            "error": (
                "cook_job reuse unavailable (115 cook registry not built); "
                "omit cook_job to read the current synchronous sim state"
            ),
        }, f"unexpected cook_job-unavailable payload: {data!r}"

    def test_cook_job_check_precedes_frame_range_validation(self, mock_hou):
        handlers = _import_handler_module()
        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim",
            "frame_range": [10, 1],  # INVALID (inverted) -- would raise ValueError at step (1)
            "assertions": [],
            "cook_job": "cook-9a",
        })
        assert result["status"] == "success", (
            f"cook_job must be checked BEFORE frame_range validation -- expected the "
            f"documented cook_job message (a normal return), not a propagated ValueError. "
            f"Got {result!r}."
        )
        assert result["data"]["ok"] is False
        assert "cook_job" in result["data"]["error"]


# ---------------------------------------------------------------------------
# assert_simulation -- frame_range validation (Major-7)
# ---------------------------------------------------------------------------

class TestAssertSimulationFrameRangeValidation:
    """frame_range must be exactly two ints, start <= end (single-frame
    [f,f] allowed), bool REJECTED. Validated BEFORE any hou read."""

    @pytest.mark.parametrize("bad_frame_range", [
        [10, 1],
        [1, 2, 3],
        [1],
        [1.5, 10],
        [True, 10],
        [1, False],
        "not-a-list",
    ])
    def test_malformed_frame_range_raises_value_error(self, mock_hou, bad_frame_range):
        handlers = _import_handler_module()
        with pytest.raises(ValueError):
            handlers.assert_simulation(
                network="/obj/rbd_sim",
                frame_range=bad_frame_range,
                assertions=[],
                cook_job=None,
            )

    def test_single_frame_range_is_allowed(self, mock_hou):
        handlers = _import_handler_module()
        node = _make_frame_reactive_pointcount_node(mock_hou, "/obj/rbd_sim")
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim": node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim",
            "frame_range": [5, 5],
            "assertions": [{"metric": "point_count", "expect": {"eq": 5, "at_frame": 5}}],
            "cook_job": None,
        })
        assert result["status"] == "success", f"a single-frame range must be valid, got {result!r}"
        assert result["data"]["pass"] is True, f"expected pass:true, got {result['data']!r}"


# ---------------------------------------------------------------------------
# assert_simulation -- empty assertions -> NO frame stepping
# ---------------------------------------------------------------------------

class TestAssertSimulationEmptyAssertions:
    """assertions=[] must NOT step any frame and must return
    evaluate_assertions([]) verbatim (pass:true, results:[])."""

    def test_empty_assertions_no_stepping_pass_true(self, mock_hou):
        handlers = _import_handler_module()
        # No node registered at all -- if the handler tried to resolve
        # ANY node it would fail; empty assertions must short-circuit
        # before any node resolution or frame read.
        mock_hou.node.side_effect = _node_lookup({})

        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim", "frame_range": [1, 80],
            "assertions": [], "cook_job": None,
        })
        assert result["status"] == "success", f"expected a normal result, got {result!r}"
        data = result["data"]
        assert data == {"results": [], "pass": True}, f"expected the model's empty-assertions shape, got {data!r}"
        assert mock_hou.setFrame.call_count == 0, (
            f"assertions=[] must perform NO frame stepping at all, but hou.setFrame was "
            f"called {mock_hou.setFrame.call_count} time(s)."
        )


class TestAssertSimulationAssertionsNoneDoesNotShortCircuit:
    """FIX-PASS (codex-reviewer Major-3): ONLY assertions == [] (the
    genuine empty-list case) may short-circuit to
    evaluate_assertions([])/pass:true. A non-list `assertions` (e.g. None)
    is a caller-contract error -- the model's OWN ValueError for a
    non-list `assertions` must PROPAGATE (via the dispatcher's error
    envelope), never be silently treated as "falsy therefore empty"."""

    def test_assertions_none_raises_value_error_directly(self, mock_hou):
        handlers = _import_handler_module()
        with pytest.raises(ValueError):
            handlers.assert_simulation(
                network="/obj/rbd_sim",
                frame_range=[1, 1],
                assertions=None,
                cook_job=None,
            )

    def test_assertions_none_propagates_via_dispatch_not_pass_true(self, mock_hou):
        handlers = _import_handler_module()
        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim",
            "frame_range": [1, 1],
            "assertions": None,
            "cook_job": None,
        })
        assert result["status"] == "error", (
            f"assertions=None must propagate the model's caller-contract ValueError as the "
            f"dispatcher's error envelope, NOT silently return {{results:[],pass:true}} as if "
            f"it were an empty list. Got {result!r}."
        )
        assert result["error"]["code"] == "ValueError", (
            f"expected error.code == 'ValueError', got {result['error']!r}"
        )


# ---------------------------------------------------------------------------
# assert_simulation -- expect-normalization (Blocker-6)
# ---------------------------------------------------------------------------

class TestAssertSimulationExpectNormalization:
    """A top-level predicate (no `expect` key) is normalized into
    expect={...}; both expect AND a top-level predicate present -> raise
    ValueError naming the conflict."""

    def test_top_level_predicate_normalized_into_expect(self, mock_hou):
        """The spec's OWN example: {"metric":"velocity_bounds","max":250}
        (no expect key) -- normalized to expect={"max":250}."""
        handlers = _import_handler_module()
        node = _make_sop_node(mock_hou, "/obj/rbd_sim", _make_geo(v=[(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]))
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim": node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim",
            "frame_range": [1, 1],
            "assertions": [{"metric": "velocity_bounds", "max": 250}],
            "cook_job": None,
        })
        assert result["status"] == "success", (
            f"a top-level predicate must be normalized into expect, not raise. Got {result!r}"
        )
        data = result["data"]
        assert data["pass"] is True, (
            f"max |v|=2.0 must pass expect.max=250 once normalized correctly, got {data!r}"
        )

    def test_expect_and_top_level_predicate_conflict_raises_value_error(self, mock_hou):
        handlers = _import_handler_module()
        with pytest.raises(ValueError):
            handlers.assert_simulation(
                network="/obj/rbd_sim",
                frame_range=[1, 1],
                assertions=[{"metric": "velocity_bounds", "expect": {"max": 300}, "max": 250}],
                cook_job=None,
            )

    def test_expect_and_top_level_conflict_propagates_via_dispatch(self, mock_hou):
        handlers = _import_handler_module()
        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim",
            "frame_range": [1, 1],
            "assertions": [{"metric": "velocity_bounds", "expect": {"max": 300}, "max": 250}],
            "cook_job": None,
        })
        assert result["status"] == "error", (
            f"an expect/top-level conflict must propagate as the dispatcher's error "
            f"envelope, NOT be folded into {{ok:false}}. Got {result!r}."
        )
        assert result["error"]["code"] == "ValueError"


# ---------------------------------------------------------------------------
# assert_simulation -- per-assertion `node` routing (Blocker-3)
# ---------------------------------------------------------------------------

class TestAssertSimulationPerAssertionNodeRouting:
    """An assertion's own `node` (when present) is the read source instead
    of `network`; an assertion with NO `node` key falls back to `network`.
    TWO same-metric assertions on DIFFERENT nodes build TWO independent
    series in INPUT order."""

    def test_assertion_without_node_key_uses_network(self, mock_hou):
        handlers = _import_handler_module()
        node = _make_sop_node(mock_hou, "/obj/net/DEFAULT", _make_geo(n_points=5, n_prims=0))
        mock_hou.node.side_effect = _node_lookup({"/obj/net/DEFAULT": node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/net/DEFAULT",
            "frame_range": [1, 1],
            "assertions": [{"metric": "point_count", "expect": {"eq": 5, "at_frame": 1}}],
            "cook_job": None,
        })
        assert result["status"] == "success"
        assert result["data"]["pass"] is True, (
            f"an assertion with no 'node' key must resolve against `network`, got {result['data']!r}"
        )

    def test_two_assertions_same_metric_different_nodes_build_independent_series(self, mock_hou):
        handlers = _import_handler_module()
        node_a = _make_sop_node(mock_hou, "/obj/net/A", _make_geo(n_points=3, n_prims=0))
        node_b = _make_sop_node(mock_hou, "/obj/net/B", _make_geo(n_points=7, n_prims=0))
        mock_hou.node.side_effect = _node_lookup({
            "/obj/net/A": node_a, "/obj/net/B": node_b,
        })

        result = _dispatch("assert_simulation", {
            "network": "/obj/net/A",
            "frame_range": [1, 1],
            "assertions": [
                {"metric": "point_count", "node": "/obj/net/A", "expect": {"eq": 3, "at_frame": 1}},
                {"metric": "point_count", "node": "/obj/net/B", "expect": {"eq": 7, "at_frame": 1}},
            ],
            "cook_job": None,
        })
        assert result["status"] == "success", f"expected a normal result, got {result!r}"
        results = result["data"]["results"]
        assert len(results) == 2, f"expected 2 result entries (input order), got {len(results)}"
        assert results[0]["series"] == [[1, 3.0]], (
            f"first assertion (node A, 3 points) must build its OWN series, got {results[0]!r}"
        )
        assert results[1]["series"] == [[1, 7.0]], (
            f"second assertion (node B, 7 points) must build its OWN independent series, "
            f"got {results[1]!r}"
        )
        assert results[0]["pass"] is True and results[1]["pass"] is True
        assert result["data"]["pass"] is True


# ---------------------------------------------------------------------------
# assert_simulation -- per-metric SCALAR reads (the METRIC->SOURCE table)
# ---------------------------------------------------------------------------

class TestAssertSimulationPieceCountPrecedence:
    """piece_count precedence (Major-8, DETERMINISTIC):
    1. unique non-empty PRIMITIVE `name` attribute values, if present
    2. else unique non-empty POINT `name` attribute values
    3. else geo.intrinsicValue('primitivecount')
    """

    def test_unique_nonempty_prim_names_take_precedence(self, mock_hou):
        handlers = _import_handler_module()
        # 4 prims, 3 distinct non-empty names + 1 empty-string name (must
        # be excluded from the distinct count) -- 6 total prims via
        # n_prims to prove COUNT-BY-DISTINCT-NAME, not COUNT-BY-PRIM.
        geo = _make_geo(prim_names=["wall_a", "wall_b", "wall_a", "", "wall_c", ""], n_prims=6)
        node = _make_sop_node(mock_hou, "/obj/rbd_sim/wall", geo)
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim/wall": node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim/wall",
            "frame_range": [1, 1],
            "assertions": [{"metric": "piece_count", "expect": {"eq": 3, "at_frame": 1}}],
            "cook_job": None,
        })
        assert result["status"] == "success", f"expected a normal result, got {result!r}"
        assert result["data"]["pass"] is True, (
            f"expected piece_count == 3 distinct non-empty prim names (wall_a/wall_b/wall_c), "
            f"got {result['data']!r}"
        )

    def test_falls_back_to_point_names_when_no_prim_name_attrib(self, mock_hou):
        handlers = _import_handler_module()
        geo = _make_geo(point_names=["p1", "p2", "p1", ""], n_points=4, n_prims=0)
        node = _make_sop_node(mock_hou, "/obj/pop_sim", geo)
        mock_hou.node.side_effect = _node_lookup({"/obj/pop_sim": node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/pop_sim",
            "frame_range": [1, 1],
            "assertions": [{"metric": "piece_count", "expect": {"eq": 2, "at_frame": 1}}],
            "cook_job": None,
        })
        assert result["status"] == "success"
        assert result["data"]["pass"] is True, (
            f"expected piece_count == 2 distinct non-empty point names (p1/p2) when no prim "
            f"`name` attribute exists, got {result['data']!r}"
        )

    def test_falls_back_to_primitivecount_when_no_name_attrib_at_all(self, mock_hou):
        handlers = _import_handler_module()
        geo = _make_geo(n_prims=9, n_points=0)
        node = _make_sop_node(mock_hou, "/obj/rbd_sim/no_names", geo)
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim/no_names": node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim/no_names",
            "frame_range": [1, 1],
            "assertions": [{"metric": "piece_count", "expect": {"eq": 9, "at_frame": 1}}],
            "cook_job": None,
        })
        assert result["status"] == "success"
        assert result["data"]["pass"] is True, (
            f"expected piece_count == primitivecount (9) when no name attribute exists at all, "
            f"got {result['data']!r}"
        )

    def test_prim_name_attrib_exists_but_all_empty_falls_back_to_point_names(self, mock_hou):
        """FIX-PASS (codex-reviewer Major-1): the precedence must be keyed on
        NON-EMPTY values present, not on attribute EXISTENCE. A primitive
        `name` attribute that EXISTS but whose every value is "" must NOT
        win the precedence -- it must fall through to the non-empty POINT
        `name` values (here 2 distinct: piece_a/piece_b), NOT return 0."""
        handlers = _import_handler_module()
        geo = _make_geo(
            prim_names=["", "", ""],  # attribute EXISTS, but every value is empty
            n_prims=3,
            point_names=["piece_a", "piece_b", "piece_a"],
            n_points=3,
        )
        node = _make_sop_node(mock_hou, "/obj/rbd_sim/empty_prim_names", geo)
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim/empty_prim_names": node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim/empty_prim_names",
            "frame_range": [1, 1],
            "assertions": [{"metric": "piece_count", "expect": {"eq": 2, "at_frame": 1}}],
            "cook_job": None,
        })
        assert result["status"] == "success", f"expected a normal result, got {result!r}"
        assert result["data"]["pass"] is True, (
            f"expected piece_count == 2 (the non-empty POINT names piece_a/piece_b), because "
            f"the primitive `name` attribute EXISTS but every value is empty and must NOT "
            f"win the precedence on existence alone. Got {result['data']!r}"
        )

    def test_prim_name_attrib_all_empty_and_no_point_names_falls_back_to_primitivecount(self, mock_hou):
        """FIX-PASS (codex-reviewer Major-1): when BOTH the primitive `name`
        attribute (all-empty) AND no point `name` attribute produce zero
        usable non-empty names, the precedence must fall all the way
        through to geo.intrinsicValue('primitivecount') -- NOT return 0."""
        handlers = _import_handler_module()
        geo = _make_geo(
            prim_names=["", "", ""],  # attribute EXISTS, all empty
            n_prims=3,
            point_names=None,  # no point `name` attribute at all
            n_points=0,
        )
        node = _make_sop_node(mock_hou, "/obj/rbd_sim/all_empty_names", geo)
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim/all_empty_names": node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim/all_empty_names",
            "frame_range": [1, 1],
            "assertions": [{"metric": "piece_count", "expect": {"eq": 3, "at_frame": 1}}],
            "cook_job": None,
        })
        assert result["status"] == "success", f"expected a normal result, got {result!r}"
        assert result["data"]["pass"] is True, (
            f"expected piece_count == primitivecount (3) when neither the prim `name` nor "
            f"the point `name` attribute yields any non-empty value, got {result['data']!r}"
        )


class TestAssertSimulationPointCount:
    def test_point_count_reads_pointcount_intrinsic(self, mock_hou):
        handlers = _import_handler_module()
        geo = _make_geo(n_points=42, n_prims=0)
        node = _make_sop_node(mock_hou, "/obj/flip_sim", geo)
        mock_hou.node.side_effect = _node_lookup({"/obj/flip_sim": node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/flip_sim",
            "frame_range": [1, 1],
            "assertions": [{"metric": "point_count", "expect": {"eq": 42, "at_frame": 1}}],
            "cook_job": None,
        })
        assert result["status"] == "success"
        assert result["data"]["pass"] is True, f"expected point_count == 42, got {result['data']!r}"

    def test_point_count_via_objnode_resolves_via_display_node(self, mock_hou):
        """An ObjNode wire source resolves via displayNode().geometry()
        (per the METRIC->SOURCE table: 'if ObjNode use displayNode().
        geometry()'), pinned via a DELIBERATELY WRONG geometry directly on
        the ObjNode itself so a handler that mistakenly reads
        node.geometry() would compute a visibly different (wrong)
        point_count."""
        handlers = _import_handler_module()
        obj_node = _make_obj_node(mock_hou, "/obj/pyro_sim", _make_geo(n_points=11, n_prims=0))
        wrong_geo = _make_geo(n_points=999, n_prims=0)
        obj_node.geometry.return_value = wrong_geo
        mock_hou.node.side_effect = _node_lookup({"/obj/pyro_sim": obj_node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/pyro_sim",
            "frame_range": [1, 1],
            "assertions": [{"metric": "point_count", "expect": {"eq": 11, "at_frame": 1}}],
            "cook_job": None,
        })
        assert result["status"] == "success"
        assert result["data"]["pass"] is True, (
            f"expected point_count == 11 (via displayNode().geometry()), NOT 999 (via "
            f"node.geometry() directly). Got {result['data']!r}"
        )


class TestAssertSimulationVelocityBounds:
    def test_velocity_bounds_reads_max_magnitude_over_points(self, mock_hou):
        handlers = _import_handler_module()
        # |v| magnitudes: 1.0, 5.0 (3-4-0 triangle), 2.0 -- max is 5.0
        geo = _make_geo(v=[(1.0, 0.0, 0.0), (3.0, 4.0, 0.0), (0.0, 2.0, 0.0)])
        node = _make_sop_node(mock_hou, "/obj/rbd_sim", geo)
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim": node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim",
            "frame_range": [1, 1],
            "assertions": [{"metric": "velocity_bounds", "expect": {"eq": 5.0, "at_frame": 1}}],
            "cook_job": None,
        })
        assert result["status"] == "success"
        assert result["data"]["pass"] is True, (
            f"expected max |v| == 5.0 (from the 3-4-0 point), got {result['data']!r}"
        )

    def test_velocity_bounds_zero_when_no_points(self, mock_hou):
        handlers = _import_handler_module()
        geo = _make_geo(n_points=0, n_prims=0)
        geo.findPointAttrib.side_effect = lambda name: object() if name == "v" else None
        node = _make_sop_node(mock_hou, "/obj/rbd_sim_empty", geo)
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim_empty": node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim_empty",
            "frame_range": [1, 1],
            "assertions": [{"metric": "velocity_bounds", "expect": {"eq": 0.0, "at_frame": 1}}],
            "cook_job": None,
        })
        assert result["status"] == "success"
        assert result["data"]["pass"] is True, f"expected velocity_bounds == 0.0 with no points, got {result['data']!r}"

    def test_missing_v_attribute_degrades_whole_call(self, mock_hou):
        """A missing `v` attribute must raise hou.OperationFailed inside
        the read phase (Major-10), degrading the WHOLE call to one
        {ok:false,error}, never a partial result."""
        handlers = _import_handler_module()
        geo = _make_geo(n_points=3, n_prims=0)  # no v= given -> findPointAttrib('v') is None
        node = _make_sop_node(mock_hou, "/obj/rbd_sim_no_v", geo)
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim_no_v": node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim_no_v",
            "frame_range": [1, 1],
            "assertions": [{"metric": "velocity_bounds", "expect": {"max": 250}}],
            "cook_job": None,
        })
        assert result["status"] == "success", f"a metric-unavailable failure must be a normal return, got {result!r}"
        data = result["data"]
        assert data.get("ok") is False, f"expected ok:false when 'v' attribute is missing, got {data!r}"
        assert "error" in data and data["error"]

    def test_empty_source_zero_before_missing_v_check(self, mock_hou):
        """FIX-PASS (codex-reviewer Major-2): the contract's "0.0 if no
        points" case must be checked BEFORE the missing-`v`-attribute
        error -- an EMPTY source (zero points, no `v` attribute defined AT
        ALL) must return series [[frame, 0.0]] and PASS the spec's own
        example {"metric":"velocity_bounds","max":250}, NOT degrade to
        {ok:false} merely because there is no `v` attribute to check on an
        already-empty geometry."""
        handlers = _import_handler_module()
        geo = _make_geo(n_points=0, n_prims=0)  # no v= given AND zero points
        node = _make_sop_node(mock_hou, "/obj/rbd_sim_truly_empty", geo)
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim_truly_empty": node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim_truly_empty",
            "frame_range": [1, 1],
            "assertions": [{"metric": "velocity_bounds", "max": 250}],
            "cook_job": None,
        })
        assert result["status"] == "success", (
            f"an EMPTY (no-points) source must short-circuit to 0.0 BEFORE the missing-'v' "
            f"check -- expected a normal result, got {result!r}"
        )
        data = result["data"]
        assert data.get("ok") is not False, (
            f"expected pass:true (0.0 <= 250) for an empty source, NOT {{ok:false}} -- the "
            f"no-points case must be checked before requiring the 'v' attribute to exist. "
            f"Got {data!r}"
        )
        assert data["pass"] is True, f"expected pass:true, got {data!r}"
        assert data["results"][0]["series"] == [[1, 0.0]], (
            f"expected series [[1, 0.0]] for an empty (no-points, no-'v') source, got "
            f"{data['results'][0]['series']!r}"
        )


class TestAssertSimulationBboxOverTime:
    def test_bbox_over_time_reads_world_aabb_max_extent(self, mock_hou):
        """Local bbox (0,0,0)-(4,2,6) -> extents x=4,y=2,z=6 -> w=4,d=6,h=2
        -> max extent = 6. An IDENTITY-ROTATION world transform (translate
        arbitrary, per _make_obj_node) leaves world extents == local
        extents, so this pins the extent-mapping/max-reduction without
        needing rotation math."""
        handlers = _import_handler_module()
        geo = _make_geo(n_points=0, n_prims=0, bbox=((0.0, 0.0, 0.0), (4.0, 2.0, 6.0)))
        obj_node = _make_obj_node(mock_hou, "/obj/table1", geo, translate=(10.0, 20.0, 30.0))
        mock_hou.node.side_effect = _node_lookup({"/obj/table1": obj_node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/table1",
            "frame_range": [1, 1],
            "assertions": [{"metric": "bbox_over_time", "expect": {"eq": 6.0, "at_frame": 1}}],
            "cook_job": None,
        })
        assert result["status"] == "success", f"expected a normal result, got {result!r}"
        assert result["data"]["pass"] is True, (
            f"expected bbox_over_time max extent == 6.0 (the z-extent, largest of 4/2/6), "
            f"got {result['data']!r}"
        )


class TestAssertSimulationMassConservation:
    def test_mass_conservation_reads_sum_of_mass_attribute(self, mock_hou):
        handlers = _import_handler_module()
        geo = _make_geo(mass=[1.0, 2.5, 0.5])
        node = _make_sop_node(mock_hou, "/obj/rbd_sim", geo)
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim": node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim",
            "frame_range": [1, 1],
            "assertions": [{"metric": "mass_conservation", "expect": {"eq": 4.0, "at_frame": 1}}],
            "cook_job": None,
        })
        assert result["status"] == "success"
        assert result["data"]["pass"] is True, (
            f"expected mass_conservation == sum(mass) == 4.0, got {result['data']!r}"
        )

    def test_missing_mass_attribute_degrades_whole_call(self, mock_hou):
        handlers = _import_handler_module()
        geo = _make_geo(n_points=3, n_prims=0)  # no mass= given -> findPointAttrib('mass') is None
        node = _make_sop_node(mock_hou, "/obj/rbd_sim_no_mass", geo)
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim_no_mass": node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim_no_mass",
            "frame_range": [1, 1],
            "assertions": [{"metric": "mass_conservation", "expect": {"tolerance": 0.02}}],
            "cook_job": None,
        })
        assert result["status"] == "success", f"a metric-unavailable failure must be a normal return, got {result!r}"
        data = result["data"]
        assert data.get("ok") is False, f"expected ok:false when 'mass' attribute is missing, got {data!r}"
        assert "error" in data and data["error"]


# ---------------------------------------------------------------------------
# assert_simulation -- deferred metrics (Blocker-4/5)
# ---------------------------------------------------------------------------

class TestAssertSimulationDeferredMetricsUnsupported:
    """field_stats and constraint_count have no shipped aggregate reader
    in PR-2 and must return the EXACT structured 'unsupported in PR-2'
    {ok:false} strings named in the locked contract -- degrading the whole
    call, never a raised exception through the dispatcher."""

    def test_field_stats_unsupported(self, mock_hou):
        handlers = _import_handler_module()
        node = _make_sop_node(mock_hou, "/obj/pyro_sim", _make_geo(n_points=0, n_prims=0))
        mock_hou.node.side_effect = _node_lookup({"/obj/pyro_sim": node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/pyro_sim",
            "frame_range": [1, 1],
            "assertions": [{"metric": "field_stats", "field": "density", "expect": {"max_gt": 0.1}}],
            "cook_job": None,
        })
        assert result["status"] == "success", f"expected a normal result, got {result!r}"
        data = result["data"]
        assert data == {
            "ok": False,
            "error": "field_stats unsupported in PR-2: no shipped DOP aggregate field-stat reader",
        }, f"unexpected field_stats-unsupported payload: {data!r}"

    def test_constraint_count_unsupported(self, mock_hou):
        handlers = _import_handler_module()
        node = _make_sop_node(mock_hou, "/obj/rbd_sim", _make_geo(n_points=0, n_prims=0))
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim": node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim",
            "frame_range": [1, 1],
            "assertions": [{"metric": "constraint_count", "expect": {"max": 100}}],
            "cook_job": None,
        })
        assert result["status"] == "success", f"expected a normal result, got {result!r}"
        data = result["data"]
        assert data == {
            "ok": False,
            "error": "constraint_count unsupported in PR-2: no shipped active-constraint reader",
        }, f"unexpected constraint_count-unsupported payload: {data!r}"


# ---------------------------------------------------------------------------
# assert_simulation -- unresolvable node/network -> {ok:false} WITHOUT raising
# ---------------------------------------------------------------------------

class TestAssertSimulationUnresolvableSourceDegradesGracefully:
    def test_unresolvable_network_is_ok_false(self, mock_hou):
        handlers = _import_handler_module()
        mock_hou.node.side_effect = _node_lookup({})

        result = _dispatch("assert_simulation", {
            "network": "/obj/does_not_exist",
            "frame_range": [1, 1],
            "assertions": [{"metric": "point_count", "expect": {"max": 100}}],
            "cook_job": None,
        })
        assert result["status"] == "success", (
            f"a scene-resolution failure must be a NORMAL return, never a dispatcher-level "
            f"exception. Got {result!r}."
        )
        data = result["data"]
        assert data.get("ok") is False
        assert "error" in data and isinstance(data["error"], str) and data["error"]
        assert "does_not_exist" in data["error"]

    def test_unresolvable_assertion_node_is_ok_false(self, mock_hou):
        handlers = _import_handler_module()
        real_network = _make_sop_node(mock_hou, "/obj/rbd_sim", _make_geo(n_points=5, n_prims=0))
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim": real_network})

        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim",
            "frame_range": [1, 1],
            "assertions": [
                {"metric": "point_count", "node": "/obj/rbd_sim/ghost_wall", "expect": {"max": 100}},
            ],
            "cook_job": None,
        })
        assert result["status"] == "success"
        data = result["data"]
        assert data.get("ok") is False
        assert "ghost_wall" in data["error"] or "/obj/rbd_sim/ghost_wall" in data["error"]


# ---------------------------------------------------------------------------
# assert_simulation -- the saved frame is RESTORED in a finally
# ---------------------------------------------------------------------------

class TestAssertSimulationFrameSaveAndRestore:
    """The handler must SAVE hou.frame() BEFORE stepping and RESTORE it in
    a finally -- on BOTH the success path AND an error path raised
    mid-stepping (proving the restore is not merely the happy-path tail).
    The _FrameState fixture starts at 42.0 (far from any tested
    frame_range), so a genuine restore is unambiguous."""

    def test_frame_restored_after_successful_multi_frame_read(self, mock_hou):
        handlers = _import_handler_module()
        node = _make_frame_reactive_pointcount_node(mock_hou, "/obj/rbd_sim")
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim": node})

        assert mock_hou._frame_state.current == 42.0  # sanity: the pre-call frame

        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim",
            "frame_range": [1, 3],
            "assertions": [{"metric": "point_count", "expect": {"max": 3}}],
            "cook_job": None,
        })
        assert result["status"] == "success", f"expected a normal result, got {result!r}"
        # The frame-reactive node makes pointcount == the current frame --
        # a genuine per-frame read produces the increasing series [1,2,3].
        series = result["data"]["results"][0]["series"]
        assert series == [[1, 1.0], [2, 2.0], [3, 3.0]], (
            f"expected a genuinely per-frame-varying series (proving hou.setFrame(f) was "
            f"called before each read, not a single snapshot repeated), got {series!r}"
        )
        assert mock_hou._frame_state.current == 42.0, (
            f"the saved pre-call frame (42.0) must be restored after a successful read, "
            f"but hou.frame() now reports {mock_hou._frame_state.current!r}."
        )
        assert mock_hou.setFrame.call_args_list[-1][0][0] == 42.0, (
            "the LAST hou.setFrame call must restore the original frame."
        )

    def test_frame_restored_even_when_a_read_raises_mid_stepping(self, mock_hou):
        """A missing `v` attribute raises hou.OperationFailed mid-loop
        (degrading the whole call to {ok:false}) -- the frame MUST still
        be restored (the finally fires on the error path too)."""
        handlers = _import_handler_module()
        geo_no_v = _make_geo(n_points=2, n_prims=0)  # no v -> triggers hou.OperationFailed
        node = _make_sop_node(mock_hou, "/obj/rbd_sim_no_v", geo_no_v)
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim_no_v": node})

        assert mock_hou._frame_state.current == 42.0

        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim_no_v",
            "frame_range": [1, 5],
            "assertions": [{"metric": "velocity_bounds", "expect": {"max": 250}}],
            "cook_job": None,
        })
        assert result["status"] == "success"
        assert result["data"].get("ok") is False, f"expected the missing-v failure shape, got {result['data']!r}"
        assert mock_hou._frame_state.current == 42.0, (
            f"the saved pre-call frame (42.0) must be restored even when a read RAISES "
            f"mid-stepping, but hou.frame() now reports {mock_hou._frame_state.current!r}."
        )


# ---------------------------------------------------------------------------
# assert_simulation -- an unknown-metric ValueError PROPAGATES
# ---------------------------------------------------------------------------

class TestAssertSimulationUnknownMetricPropagates:
    """A metric string the model itself does not recognize (NOT one of
    the 7 _ASSERT_ORDER names) is a caller-contract error -- it PROPAGATES
    as a raised ValueError, which the DISPATCHER (not the handler) turns
    into its standard error envelope. It must NEVER be folded into
    {"ok": False}."""

    def test_direct_handler_call_raises_value_error(self, mock_hou):
        handlers = _import_handler_module()
        geo = _make_geo(n_points=2, n_prims=2, v=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], mass=[1.0, 1.0])
        node = _make_sop_node(mock_hou, "/obj/rbd_sim", geo)
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim": node})

        with pytest.raises(ValueError):
            handlers.assert_simulation(
                network="/obj/rbd_sim",
                frame_range=[1, 1],
                assertions=[{"metric": "totally_bogus_metric_xyz", "expect": {"max": 1}}],
                cook_job=None,
            )

    def test_dispatch_propagates_as_error_envelope_not_ok_false(self, mock_hou):
        handlers = _import_handler_module()
        geo = _make_geo(n_points=2, n_prims=2, v=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], mass=[1.0, 1.0])
        node = _make_sop_node(mock_hou, "/obj/rbd_sim", geo)
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim": node})

        result = _dispatch("assert_simulation", {
            "network": "/obj/rbd_sim",
            "frame_range": [1, 1],
            "assertions": [{"metric": "totally_bogus_metric_xyz", "expect": {"max": 1}}],
            "cook_job": None,
        })
        assert result["status"] == "error", (
            f"a caller-contract ValueError (unknown metric) must propagate as the "
            f"DISPATCHER's error envelope (status:error), NOT be folded into a normal "
            f"ok:false success return. Got {result!r}."
        )
        assert result["error"]["code"] == "ValueError", (
            f"expected error.code == 'ValueError', got {result['error']!r}"
        )


# =============================================================================
# hython-smoke rung (rung 2 of the plan's verification ladder) -- SKIPPED
# under bare pytest (no real Houdini). This class is the scaffold named in
# the plan's decomposition: "an hython-smoke scaffold driving
# dispatch('assert_simulation', {...}) through the REAL dispatcher on a
# small RBD fixture (piece_count jump + frame-restore)." It is deliberately
# distinct from ALL tests above -- those use a monkeypatched `hou` and
# NEVER touch a real Houdini session; this class only executes when a real
# (non-mock) `hou` module is importable, i.e. under `hython` or inside an
# already-running Houdini/hython interpreter.
# =============================================================================

def _real_hou_available() -> bool:
    try:
        import hou as _hou  # noqa: F401
    except ImportError:
        return False
    return not isinstance(_hou, MagicMock)


@pytest.mark.skipif(
    not _real_hou_available(),
    reason=(
        "requires a real Houdini/hython interpreter -- this is the hython-smoke rung "
        "(rung 2 of the verification ladder), not the mocked-hou pytest rung above"
    ),
)
class TestAssertSimulationHythonSmoke:
    """Drives dispatch('assert_simulation', {...}) through the REAL
    dispatcher against a small SOP fixture built inline with real hou
    calls (a switch driven by an $F expression standing in for a
    fracture-style piece-count jump, since building a genuine RBD
    fracture setup is out of scope for a red-test scaffold) -- asserting
    a piece_count jump reads correctly across the cooked frame range AND
    the frame is restored to its pre-call value afterward.

    This class is entirely SKIPPED (collected but not run) under bare
    `pytest` with no real Houdini on the path; it only executes under
    `hython` (mirrors cop_onnx_hython_smoke.py's two-mode RED/GREEN guard,
    ported here as a pytest skip rather than a standalone script since
    this file already carries the mock_hou-based rung-1 suite).
    """

    def test_piece_count_jump_reads_across_cooked_range_and_frame_restored(self):
        import hou  # real hou -- only reached when _real_hou_available() is True

        from fxhoudinimcp_server.dispatcher import dispatch

        saved_frame = hou.frame()
        geo_obj = hou.node("/obj").createNode("geo", "pp12_117b_smoke_fixture")
        try:
            box = geo_obj.createNode("box")
            switch = geo_obj.createNode("switch")
            switch.setInput(0, box)
            # Before frame 40: pass box through untouched (1 piece).
            # At/after frame 40: fan the box out via copytopoints onto a
            # small grid of points (a stand-in "fracture" -- a real RBD
            # jump is out of scope for this scaffold).
            grid = geo_obj.createNode("grid")
            grid.parm("rows").set(9)
            grid.parm("cols").set(9)
            copy_to_points = geo_obj.createNode("copytopoints")
            copy_to_points.setInput(0, box)
            copy_to_points.setInput(1, grid)
            switch.setInput(1, copy_to_points)
            switch.parm("index").setExpression("$F >= 40 ? 1 : 0")
            out_null = geo_obj.createNode("null", "OUT")
            out_null.setInput(0, switch)

            result = dispatch("assert_simulation", {
                "network": geo_obj.path(),
                "frame_range": [38, 41],
                "assertions": [
                    {
                        "metric": "piece_count",
                        "node": out_null.path(),
                        "expect": {"at_frame": 40, "jump_gt": 5},
                    },
                ],
                "cook_job": None,
            })
            assert result["status"] == "success", f"expected a normal result, got {result!r}"
            data = result["data"]
            assert data["pass"] is True, (
                f"expected the piece_count jump at frame 40 (1 piece -> 81 pieces) to satisfy "
                f"jump_gt=5, got {data!r}"
            )
        finally:
            geo_obj.destroy()
            hou.setFrame(saved_frame)
            assert hou.frame() == saved_frame, "the pre-test frame must be restored"


# =============================================================================
# compile_timeline / _preview_compile_timeline — PP12-117 PR-3, the FIRST
# MUTATING/109-gated handler of the Temporal/Sim-Reasoning MCP member
# (NARROWED SCOPE per plan pp12-117c lockedFieldContract, REVISION 2).
#
# TDD phase: RED — compile_timeline and _preview_compile_timeline do NOT
# exist yet on temporal_reasoning_handlers.py. Every test below calls
# _import_handler_module() (already defined above), which raises
# ModuleNotFoundError/AttributeError until hou-dev lands the green impl —
# scoped to THESE new tests only; the PR-2 suite above stays green.
#
# Grounded LAYER-FOR-LAYER on test_spatial_reasoning_handlers.py's
# TestSolveLayout* classes (THE shipped MUTATING/109-gated exemplar):
# Capability.MUTATING + preview_fn + preview_required registration
# (asserted via dispatcher._preview_registry_fallback, the mock-safe
# pattern — dispatcher.preview_of() auto-vivifies a MagicMock attribute
# under mock_hou and is NOT mock-safe); apply=false -> zero mutation;
# preflight-validate-then-atomic-apply; a raise inside the preview ->
# DENY; a model ValueError PROPAGATES (never folded into ok:false/
# unresolved).
#
# Locked field contract (plan pp12-117c lockedFieldContract, REVISION 2):
#     compile_timeline(*, network, events, frame_range, apply=True) -> dict
#         (1) resolve network via hou.node(network) -> unresolvable ->
#             {ok:false,error}
#         (2) temporal_reasoning_model.compile_plan(events, network) ->
#             the plan (a model ValueError PROPAGATES)
#         (3) PREFLIGHT-VALIDATE every compiled.keyframes entry: target
#             exists AND is `network` OR under `network + '/'` AND the
#             parm exists on it AND the frames payload is well-formed --
#             a failing entry moves to unresolved (its source event id)
#             and is DROPPED, BEFORE any write
#         (4) apply=True -> ATOMICALLY apply ONLY the fully-validated set
#             via the SHIPPED animation_handlers._set_keyframes, frames
#             [[f,v]...] CONVERTED to [{frame,value}...] (Blocker-1)
#         (5) return {compiled, event_graph, applied, unresolved}
#             (unresolved = compile_plan's unresolved UNION any
#             preflight-dropped ids)
#     apply=False returns the SAME plan with ZERO _set_keyframes calls.
#     GATED: register_handler('compile_timeline', compile_timeline,
#         Capability.MUTATING, preview_fn=_preview_compile_timeline,
#         preview_required=True)
#
#     _preview_compile_timeline(params: dict) -> dict (POSITIONAL) runs
#     the SAME read-only preflight validation apply does (Major-6), so
#     its would-apply/unresolved split matches EXACTLY what apply would
#     do -- NO writes. A raise inside the preview -> the 109 gate DENIES.
# =============================================================================

def _make_node_with_parm(mock_hou, path, parm_names):
    """A node exposing .path() and .parm(name) -> a truthy stand-in when
    name is in parm_names, else None (mirrors hou.Node.parm's real
    contract: returns None for a non-existent parm name). Sufficient for
    the preflight-validation surface compile_timeline needs (node
    existence + parm existence + path-based network-scope check) --
    deliberately NOT a full SopNode/ObjNode stand-in (the write path is
    exercised entirely through the mocked _set_keyframes fixture below,
    never through this node's own attributes)."""
    node = MagicMock(name=f"fake_node_{path}")
    node.path.return_value = path

    def _parm(name):
        return MagicMock(name=f"parm_{name}") if name in parm_names else None

    node.parm.side_effect = _parm
    return node


@pytest.fixture()
def mock_set_keyframes(monkeypatch, mock_hou):
    """Patch the SHIPPED `_set_keyframes` (animation_handlers.py,
    Major-8 direct-import) so compile_timeline's apply-time calls can be
    asserted without a real Houdini parm write.

    Imports animation_handlers FIRST (mock_hou is already installed, so
    its own top-level `import hou` succeeds off-DCC), patches ITS
    `_set_keyframes` attribute, then POPS any already-imported
    temporal_reasoning_handlers module out of sys.modules so the NEXT
    _import_handler_module() call performs a genuinely FRESH import that
    re-executes temporal_reasoning_handlers' own top-level direct-import
    of _set_keyframes against the now-patched attribute -- correct
    regardless of whether hou-dev writes
    `from fxhoudinimcp_server.handlers.animation_handlers import
    _set_keyframes` (name copied at import time) or
    `from fxhoudinimcp_server.handlers import animation_handlers` +
    `animation_handlers._set_keyframes(...)` (attribute looked up at call
    time) -- both styles resolve through THIS patched object.

    Returns the MagicMock so tests can assert call_count / call_args.
    """
    import fxhoudinimcp_server.handlers.animation_handlers as _anim

    fake_set_keyframes = MagicMock(name="_set_keyframes", return_value={
        "node_path": "unused", "parm_name": "unused",
        "keyframes_set": 0, "status": "keyframes_set",
    })
    monkeypatch.setattr(_anim, "_set_keyframes", fake_set_keyframes)
    sys.modules.pop("fxhoudinimcp_server.handlers.temporal_reasoning_handlers", None)
    return fake_set_keyframes


# ---------------------------------------------------------------------------
# PRIMARY RED GATE — compile_timeline / _preview_compile_timeline must exist
# ---------------------------------------------------------------------------

class TestCompileTimelineHandlerImport:
    def test_compile_timeline_handler_importable(self, mock_hou):
        handlers = _import_handler_module()
        assert hasattr(handlers, "compile_timeline"), (
            "temporal_reasoning_handlers.py must expose compile_timeline (PP12-117 PR-3)."
        )
        assert callable(handlers.compile_timeline)

    def test_preview_compile_timeline_importable(self, mock_hou):
        handlers = _import_handler_module()
        assert hasattr(handlers, "_preview_compile_timeline"), (
            "temporal_reasoning_handlers.py must expose _preview_compile_timeline "
            "(the positional gate preview_fn, PP12-117 PR-3)."
        )
        assert callable(handlers._preview_compile_timeline)


# ---------------------------------------------------------------------------
# Capability.MUTATING + preview registration
# ---------------------------------------------------------------------------

class TestCompileTimelineCapabilityAndPreviewRegistration:
    """Mirrors test_spatial_reasoning_handlers.py's
    TestSolveLayoutCapabilityAndPreviewRegistration -- the proven
    registration-assertion pattern for a MUTATING+preview_fn+
    preview_required tool, via the mock-safe
    dispatcher._preview_registry_fallback store (dispatcher.preview_of()
    reads hou.session._fxhoudinimcp_preview_registry, which
    auto-vivifies into a MagicMock attribute under mock_hou and is NOT
    mock-safe)."""

    def test_compile_timeline_capability_mutating(self, mock_hou):
        _import_handler_module()
        from fxhoudinimcp_server import dispatcher
        assert dispatcher.capability_of("compile_timeline") == dispatcher.Capability.MUTATING, (
            f"compile_timeline must be registered Capability.MUTATING (GATED), "
            f"got {dispatcher.capability_of('compile_timeline')!r}."
        )

    def test_compile_timeline_preview_registered_and_required(self, mock_hou):
        handlers = _import_handler_module()
        from fxhoudinimcp_server import dispatcher
        record = dispatcher._preview_registry_fallback["compile_timeline"]
        assert record["preview_fn"] is handlers._preview_compile_timeline, (
            f"compile_timeline must register preview_fn=_preview_compile_timeline, "
            f"got {record!r}."
        )
        assert record["preview_required"] is True, (
            f"compile_timeline must register preview_required=True (fail-safe "
            f"DENY-on-raise), got {record!r}."
        )


# ---------------------------------------------------------------------------
# apply=false -> ZERO _set_keyframes calls, plan returned unmutated
# ---------------------------------------------------------------------------

class TestCompileTimelineApplyFalseNoMutation:
    """apply=false must return the SAME compiled plan (post-preflight-
    validation) but make ZERO _set_keyframes calls."""

    def test_apply_false_returns_plan_and_calls_nothing(self, mock_hou, mock_set_keyframes):
        node = _make_node_with_parm(mock_hou, "/obj/rbd_sim", {"tx"})
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim": node})

        events = [{
            "id": "kf1", "type": "keyframe", "target": "/obj/rbd_sim",
            "params": {"parm": "tx", "frames": [[1, 0], [10, 5]]},
        }]

        result = _dispatch("compile_timeline", {
            "network": "/obj/rbd_sim", "events": events,
            "frame_range": [1, 10], "apply": False,
        })
        assert result["status"] == "success", f"compile_timeline must not raise, got {result!r}"
        data = result["data"]
        assert data["applied"] is False
        assert data["compiled"]["keyframes"] == [
            {"node": "/obj/rbd_sim", "parm": "tx", "frames": [[1, 0], [10, 5]]},
        ]
        assert data["unresolved"] == []
        assert mock_set_keyframes.call_count == 0, (
            f"apply=false must make ZERO _set_keyframes calls, got "
            f"{mock_set_keyframes.call_count}"
        )


# ---------------------------------------------------------------------------
# apply=true -> _set_keyframes called with CONVERTED [{frame,value}]
# ---------------------------------------------------------------------------

class TestCompileTimelineApplyTrueSetsKeyframes:
    """apply=true must call the SHIPPED _set_keyframes ONCE per validated
    compiled.keyframes entry, converting frames [[f,v]...] ->
    [{frame,value}...] (Blocker-1)."""

    def test_apply_true_calls_set_keyframes_with_converted_frames(self, mock_hou, mock_set_keyframes):
        node = _make_node_with_parm(mock_hou, "/obj/rbd_sim", {"tx"})
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim": node})

        events = [{
            "id": "kf1", "type": "keyframe", "target": "/obj/rbd_sim",
            "params": {"parm": "tx", "frames": [[1, 0], [10, 5]]},
        }]

        result = _dispatch("compile_timeline", {
            "network": "/obj/rbd_sim", "events": events,
            "frame_range": [1, 10], "apply": True,
        })
        assert result["status"] == "success", f"compile_timeline must not raise, got {result!r}"
        data = result["data"]
        assert data["applied"] is True
        assert data["unresolved"] == []

        assert mock_set_keyframes.call_count == 1, (
            f"expected exactly ONE _set_keyframes call for the single validated "
            f"keyframe entry, got {mock_set_keyframes.call_count}"
        )
        call = mock_set_keyframes.call_args
        args = call.args
        kwargs = call.kwargs
        node_arg = args[0] if len(args) > 0 else kwargs.get("node_path")
        parm_arg = args[1] if len(args) > 1 else kwargs.get("parm_name")
        keyframes_arg = args[2] if len(args) > 2 else kwargs.get("keyframes")

        assert node_arg == "/obj/rbd_sim", (
            f"expected _set_keyframes to be called with node_path='/obj/rbd_sim', "
            f"got {node_arg!r}"
        )
        assert parm_arg == "tx", f"expected parm_name='tx', got {parm_arg!r}"
        assert keyframes_arg == [{"frame": 1, "value": 0}, {"frame": 10, "value": 5}], (
            f"expected frames [[1,0],[10,5]] CONVERTED to [{{frame,value}}...] "
            f"(Blocker-1), got {keyframes_arg!r}"
        )

    def test_apply_true_calls_set_keyframes_once_per_validated_entry(self, mock_hou, mock_set_keyframes):
        # ROUND-3 FIX-PASS note (M3): `network` itself must now resolve as a
        # real node -- the round-3 fix checks network resolution
        # UNCONDITIONALLY, up front. Registering it here isolates this
        # test's intended scenario (two validated in-scope entries) from
        # the separate unresolvable-network case.
        network_node = _make_node_with_parm(mock_hou, "/obj/rbd_sim", set())
        node_a = _make_node_with_parm(mock_hou, "/obj/rbd_sim/a", {"tx"})
        node_b = _make_node_with_parm(mock_hou, "/obj/rbd_sim/b", {"ty"})
        mock_hou.node.side_effect = _node_lookup({
            "/obj/rbd_sim": network_node,
            "/obj/rbd_sim/a": node_a, "/obj/rbd_sim/b": node_b,
        })

        events = [
            {"id": "kf1", "type": "keyframe", "target": "/obj/rbd_sim/a",
             "params": {"parm": "tx", "frames": [[1, 0]]}},
            {"id": "kf2", "type": "keyframe", "target": "/obj/rbd_sim/b",
             "params": {"parm": "ty", "frames": [[5, 2]]}},
        ]

        result = _dispatch("compile_timeline", {
            "network": "/obj/rbd_sim", "events": events,
            "frame_range": [1, 5], "apply": True,
        })
        assert result["status"] == "success"
        assert result["data"]["unresolved"] == []
        assert mock_set_keyframes.call_count == 2, (
            f"expected ONE _set_keyframes call per validated compiled.keyframes entry, "
            f"got {mock_set_keyframes.call_count}"
        )


# ---------------------------------------------------------------------------
# Blocker-4 — network-scope enforcement: out-of-network target -> unresolved
# ---------------------------------------------------------------------------

class TestCompileTimelineNetworkScopeEnforcement:
    """Blocker-4: a compiled.keyframes entry whose target is NEITHER
    `network` itself NOR under `network + '/'` is dropped to unresolved
    BEFORE any write -- never applied."""

    def test_out_of_network_target_is_unresolved_not_applied(self, mock_hou, mock_set_keyframes):
        # ROUND-3 FIX-PASS note (M3): `network` itself must now resolve as a
        # real node -- the round-3 fix checks network resolution
        # UNCONDITIONALLY, up front, not only when a compiled entry's own
        # target also fails to resolve. Registering it here isolates THIS
        # test's intended scenario (an out-of-scope but perfectly
        # resolvable target) from the separate unresolvable-network case
        # pinned by TestCompileTimelineNetworkResolvedUpFrontUnconditionally.
        network_node = _make_node_with_parm(mock_hou, "/obj/rbd_sim", set())
        in_scope = _make_node_with_parm(mock_hou, "/obj/rbd_sim/inside", {"tx"})
        out_of_scope = _make_node_with_parm(mock_hou, "/obj/other_network/outside", {"tx"})
        mock_hou.node.side_effect = _node_lookup({
            "/obj/rbd_sim": network_node,
            "/obj/rbd_sim/inside": in_scope,
            "/obj/other_network/outside": out_of_scope,
        })

        events = [
            {"id": "kf_in", "type": "keyframe", "target": "/obj/rbd_sim/inside",
             "params": {"parm": "tx", "frames": [[1, 0]]}},
            {"id": "kf_out", "type": "keyframe", "target": "/obj/other_network/outside",
             "params": {"parm": "tx", "frames": [[1, 0]]}},
        ]

        result = _dispatch("compile_timeline", {
            "network": "/obj/rbd_sim", "events": events,
            "frame_range": [1, 1], "apply": True,
        })
        assert result["status"] == "success", f"an out-of-network target must NOT raise, got {result!r}"
        data = result["data"]
        assert "kf_out" in data["unresolved"], (
            f"an out-of-network target must be routed to unresolved, got {data['unresolved']!r}"
        )
        assert "kf_in" not in data["unresolved"]
        assert mock_set_keyframes.call_count == 1, (
            f"only the IN-SCOPE entry may be applied -- expected exactly 1 "
            f"_set_keyframes call, got {mock_set_keyframes.call_count}"
        )

    def test_target_equal_to_network_itself_is_in_scope(self, mock_hou, mock_set_keyframes):
        """The target being EQUAL to `network` (not merely a descendant)
        is also in-scope."""
        node = _make_node_with_parm(mock_hou, "/obj/rbd_sim", {"tx"})
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim": node})

        events = [{"id": "kf1", "type": "keyframe", "target": "/obj/rbd_sim",
                   "params": {"parm": "tx", "frames": [[1, 0]]}}]

        result = _dispatch("compile_timeline", {
            "network": "/obj/rbd_sim", "events": events,
            "frame_range": [1, 1], "apply": True,
        })
        assert result["status"] == "success"
        assert result["data"]["unresolved"] == []
        assert mock_set_keyframes.call_count == 1

    def test_sibling_path_prefix_is_not_falsely_in_scope(self, mock_hou, mock_set_keyframes):
        """A path that merely STARTS WITH the network string (without a
        '/' boundary) must NOT be treated as in-scope -- e.g.
        '/obj/rbd_sim_other' must not match network '/obj/rbd_sim'."""
        # ROUND-3 FIX-PASS note (M3): `network` must resolve unconditionally.
        network_node = _make_node_with_parm(mock_hou, "/obj/rbd_sim", set())
        node = _make_node_with_parm(mock_hou, "/obj/rbd_sim_other", {"tx"})
        mock_hou.node.side_effect = _node_lookup({
            "/obj/rbd_sim": network_node,
            "/obj/rbd_sim_other": node,
        })

        events = [{"id": "kf1", "type": "keyframe", "target": "/obj/rbd_sim_other",
                   "params": {"parm": "tx", "frames": [[1, 0]]}}]

        result = _dispatch("compile_timeline", {
            "network": "/obj/rbd_sim", "events": events,
            "frame_range": [1, 1], "apply": True,
        })
        assert result["status"] == "success"
        assert result["data"]["unresolved"] == ["kf1"], (
            f"a path-prefix match WITHOUT a '/' boundary must NOT be treated as "
            f"in-scope, got unresolved={result['data']['unresolved']!r}"
        )
        assert mock_set_keyframes.call_count == 0


# ---------------------------------------------------------------------------
# FIX-PASS (codex-reviewer Blocker-1, SECURITY): network-scope enforcement
# must use the CANONICAL resolved node path, not the raw target string.
# ---------------------------------------------------------------------------

class TestCompileTimelineNetworkScopeUsesCanonicalPath:
    """Blocker-1 (SECURITY, path-traversal escape): the raw target string
    '/obj/rbd_sim/../cam_escape' STRING-PASSES the scope check (it starts
    with '/obj/rbd_sim/'), but hou.node() resolves it to the CANONICAL
    '/obj/cam_escape' -- OUTSIDE the gated network. The scope check must
    be performed against the node's own CANONICAL path (node.path()),
    resolved via hou.node(), never the raw caller-supplied string. A
    traversal target must be dropped to unresolved and NEVER written."""

    def test_path_traversal_target_resolving_outside_network_is_unresolved_not_applied(
        self, mock_hou, mock_set_keyframes
    ):
        # ROUND-3 FIX-PASS note (M3): `network` must resolve unconditionally.
        network_node = _make_node_with_parm(mock_hou, "/obj/rbd_sim", set())
        escape_node = _make_node_with_parm(mock_hou, "/obj/cam_escape", {"tx"})
        # hou.node() is invoked with the RAW (unresolved) target string --
        # the mock returns the escape node for that exact raw lookup, while
        # the node's OWN .path() reports its CANONICAL (already-escaped)
        # path, exactly as a real hou.node()/Node.path() pair would after
        # Houdini normalizes a '..' path component.
        mock_hou.node.side_effect = _node_lookup({
            "/obj/rbd_sim": network_node,
            "/obj/rbd_sim/../cam_escape": escape_node,
        })

        events = [{
            "id": "kf_escape", "type": "keyframe",
            "target": "/obj/rbd_sim/../cam_escape",
            "params": {"parm": "tx", "frames": [[1, 0]]},
        }]

        result = _dispatch("compile_timeline", {
            "network": "/obj/rbd_sim", "events": events,
            "frame_range": [1, 1], "apply": True,
        })
        assert result["status"] == "success", (
            f"a path-traversal target must NOT crash, got {result!r}"
        )
        data = result["data"]
        assert "kf_escape" in data["unresolved"], (
            f"a target that CANONICALLY resolves outside the gated network via "
            f"a '..' path-traversal component must be routed to unresolved "
            f"(the scope check must use the resolved node's CANONICAL path, "
            f"not the raw target string) -- got unresolved={data['unresolved']!r}"
        )
        assert mock_set_keyframes.call_count == 0, (
            f"a path-traversal target that canonically resolves OUTSIDE the "
            f"network must NEVER be written -- expected 0 _set_keyframes calls, "
            f"got {mock_set_keyframes.call_count}"
        )


# ---------------------------------------------------------------------------
# Blocker-3 — missing-parm preflight: unresolved, pre-write, no crash
# ---------------------------------------------------------------------------

class TestCompileTimelineMissingParmPreflight:
    """Blocker-3: a compiled.keyframes entry whose target node exists (and
    is in-scope) but does NOT have the named parm is dropped to unresolved
    BEFORE any write -- never a crash."""

    def test_missing_parm_is_unresolved_not_applied(self, mock_hou, mock_set_keyframes):
        node = _make_node_with_parm(mock_hou, "/obj/rbd_sim", {"ty"})  # no 'tx'
        mock_hou.node.side_effect = _node_lookup({"/obj/rbd_sim": node})

        events = [{"id": "kf1", "type": "keyframe", "target": "/obj/rbd_sim",
                   "params": {"parm": "tx", "frames": [[1, 0]]}}]

        result = _dispatch("compile_timeline", {
            "network": "/obj/rbd_sim", "events": events,
            "frame_range": [1, 1], "apply": True,
        })
        assert result["status"] == "success", f"a missing parm must NOT crash, got {result!r}"
        data = result["data"]
        assert data["unresolved"] == ["kf1"], (
            f"a missing parm must route the event to unresolved, got {data['unresolved']!r}"
        )
        assert mock_set_keyframes.call_count == 0

    def test_unresolvable_target_node_is_unresolved_not_applied(self, mock_hou, mock_set_keyframes):
        """M3 FIX-PASS note: `network` itself is made resolvable in this
        fixture (a real node) so this test continues to isolate its
        INTENDED scenario -- an unresolvable TARGET node -- distinct from
        the stricter "network itself is genuinely unresolvable" {ok:false}
        case pinned separately below (TestCompileTimelineUnresolvableNetworkContractEquivalence).
        Before this adjustment, `network` was ALSO left unresolvable by
        the shared empty node_lookup, which the corrected M3 network-
        resolution check would otherwise (incorrectly, for THIS test's
        purpose) treat as a network-resolution failure."""
        network_node = _make_node_with_parm(mock_hou, "/obj/rbd_sim", set())
        mock_hou.node.side_effect = _node_lookup({
            "/obj/rbd_sim": network_node,
        })  # network resolves; the TARGET "/obj/does_not_exist" does not

        events = [{"id": "kf1", "type": "keyframe", "target": "/obj/does_not_exist",
                   "params": {"parm": "tx", "frames": [[1, 0]]}}]

        result = _dispatch("compile_timeline", {
            "network": "/obj/rbd_sim", "events": events,
            "frame_range": [1, 1], "apply": True,
        })
        assert result["status"] == "success", f"an unresolvable target must NOT crash, got {result!r}"
        assert result["data"]["unresolved"] == ["kf1"]
        assert mock_set_keyframes.call_count == 0

    def test_mixed_validated_and_missing_parm_entries_only_applies_the_validated_one(
        self, mock_hou, mock_set_keyframes
    ):
        # ROUND-3 FIX-PASS note (M3): `network` must resolve unconditionally
        # -- previously this test relied on the OLD conditional check never
        # even asking whether `network` resolves (since a validated entry
        # short-circuited the old guard).
        network_node = _make_node_with_parm(mock_hou, "/obj/rbd_sim", set())
        ok_node = _make_node_with_parm(mock_hou, "/obj/rbd_sim/ok", {"tx"})
        bad_node = _make_node_with_parm(mock_hou, "/obj/rbd_sim/bad", set())
        mock_hou.node.side_effect = _node_lookup({
            "/obj/rbd_sim": network_node,
            "/obj/rbd_sim/ok": ok_node, "/obj/rbd_sim/bad": bad_node,
        })

        events = [
            {"id": "kf_ok", "type": "keyframe", "target": "/obj/rbd_sim/ok",
             "params": {"parm": "tx", "frames": [[1, 0]]}},
            {"id": "kf_bad", "type": "keyframe", "target": "/obj/rbd_sim/bad",
             "params": {"parm": "tx", "frames": [[1, 0]]}},
        ]

        result = _dispatch("compile_timeline", {
            "network": "/obj/rbd_sim", "events": events,
            "frame_range": [1, 1], "apply": True,
        })
        assert result["status"] == "success"
        assert "kf_bad" in result["data"]["unresolved"]
        assert "kf_ok" not in result["data"]["unresolved"]
        assert mock_set_keyframes.call_count == 1


# ---------------------------------------------------------------------------
# FIX-PASS (codex-reviewer Blocker-2, NON-ATOMIC apply): preflight must
# reject a NON-WRITABLE (locked) parm -- checking `parm is not None` alone
# is insufficient.
# ---------------------------------------------------------------------------

def _make_locked_node(mock_hou, path, parm_name):
    """A node whose named parm EXISTS (passes a bare `parm is not None`
    check) but reports itself LOCKED via parm.isLocked() -- a STABLE parm
    object returned on every .parm(name) call (unlike
    _make_node_with_parm's fresh-MagicMock-per-call side_effect), so a
    test can pre-configure isLocked() and have it observed by the
    handler's own later .parm(name) call."""
    node = MagicMock(name=f"fake_locked_node_{path}")
    node.path.return_value = path
    parm_mock = MagicMock(name=f"parm_{parm_name}")
    parm_mock.isLocked.return_value = True
    node.parm.side_effect = lambda name: parm_mock if name == parm_name else None
    return node


class TestCompileTimelineLockedParmPreflight:
    """Blocker-2: a compiled.keyframes entry whose target node + parm both
    EXIST but the parm is LOCKED (non-writable) must be preflight-rejected
    to unresolved BEFORE any write -- `parm is not None` alone is not a
    writability check. The unrelated, unlocked entry must still apply
    cleanly, and the whole call must complete WITHOUT an uncaught
    exception (no partial mutation left by an exception escaping mid-loop
    in `_apply_validated_keyframes`)."""

    def test_locked_parm_is_preflight_rejected_and_never_applied(
        self, mock_hou, mock_set_keyframes
    ):
        # ROUND-3 FIX-PASS note (M3): `network` must resolve unconditionally
        # -- previously this test relied on the OLD conditional check never
        # even asking whether `network` resolves (since a validated entry
        # short-circuited the old guard).
        network_node = _make_node_with_parm(mock_hou, "/obj/rbd_sim", set())
        ok_node = _make_node_with_parm(mock_hou, "/obj/rbd_sim/ok", {"tx"})
        locked_node = _make_locked_node(mock_hou, "/obj/rbd_sim/locked", "tx")
        mock_hou.node.side_effect = _node_lookup({
            "/obj/rbd_sim": network_node,
            "/obj/rbd_sim/ok": ok_node, "/obj/rbd_sim/locked": locked_node,
        })

        events = [
            {"id": "kf_ok", "type": "keyframe", "target": "/obj/rbd_sim/ok",
             "params": {"parm": "tx", "frames": [[1, 0]]}},
            {"id": "kf_locked", "type": "keyframe", "target": "/obj/rbd_sim/locked",
             "params": {"parm": "tx", "frames": [[1, 0]]}},
        ]

        result = _dispatch("compile_timeline", {
            "network": "/obj/rbd_sim", "events": events,
            "frame_range": [1, 1], "apply": True,
        })
        assert result["status"] == "success", (
            f"a locked target parm must be caught by preflight -- NOT an uncaught "
            f"exception propagating through the dispatcher. Got {result!r}"
        )
        data = result["data"]
        assert "kf_locked" in data["unresolved"], (
            f"a LOCKED target parm must be preflight-rejected to unresolved "
            f"(parm.isLocked() must be checked, not merely 'parm is not None'), "
            f"got unresolved={data['unresolved']!r}"
        )
        assert "kf_ok" not in data["unresolved"], (
            f"the unrelated, unlocked entry must NOT be affected, got "
            f"unresolved={data['unresolved']!r}"
        )
        assert mock_set_keyframes.call_count == 1, (
            f"exactly the ONE unlocked, valid entry must be applied -- the LOCKED "
            f"entry must NEVER reach _set_keyframes. Got "
            f"{mock_set_keyframes.call_count} call(s)."
        )
        call = mock_set_keyframes.call_args
        applied_node = call.args[0] if call.args else call.kwargs.get("node_path")
        assert applied_node == "/obj/rbd_sim/ok", (
            f"the applied entry must be the UNLOCKED one, got node={applied_node!r}"
        )


# ---------------------------------------------------------------------------
# network resolution failure -> {ok:false} WITHOUT raising
# ---------------------------------------------------------------------------

class TestCompileTimelineUnresolvableNetwork:
    def test_unresolvable_network_is_ok_false_not_raise(self, mock_hou, mock_set_keyframes):
        mock_hou.node.side_effect = _node_lookup({})  # network itself doesn't resolve

        result = _dispatch("compile_timeline", {
            "network": "/obj/does_not_exist",
            "events": [{"id": "kf1", "type": "keyframe", "target": "/obj/does_not_exist",
                        "params": {"parm": "tx", "frames": [[1, 0]]}}],
            "frame_range": [1, 1], "apply": True,
        })
        assert result["status"] == "success", (
            f"an unresolvable network must be a NORMAL dispatcher return, never an "
            f"exception. Got {result!r}"
        )
        data = result["data"]
        assert data.get("ok") is False, f"expected ok:false for an unresolvable network, got {data!r}"
        assert mock_set_keyframes.call_count == 0


# ---------------------------------------------------------------------------
# FIX-PASS (codex-reviewer Major-3): unresolvable network -> {ok:false} in
# BOTH apply AND preview, regardless of whether any compiled entry's
# target equals `network` verbatim (per the literal contract, not the
# narrowed impl) -- and preview must mirror apply's terminal shape exactly
# (Major-6).
# ---------------------------------------------------------------------------

class TestCompileTimelineUnresolvableNetworkContractEquivalence:
    """Major-3: `hou.node(network)` unresolvable must degrade the WHOLE
    call to {"ok": False, "error": ...} -- ALWAYS -- not only when a
    compiled entry's target happens to equal `network` exactly. AND the
    preview must return the SAME terminal shape for the SAME input
    (Major-6 -- the preview must never diverge from what apply would do)."""

    def test_unresolvable_network_is_ok_false_even_when_no_entry_targets_network_directly(
        self, mock_hou, mock_set_keyframes
    ):
        mock_hou.node.side_effect = _node_lookup({})  # nothing resolves, incl. network itself

        events = [{
            "id": "kf1", "type": "keyframe", "target": "/obj/missing/child",
            "params": {"parm": "tx", "frames": [[1, 0]]},
        }]

        result = _dispatch("compile_timeline", {
            "network": "/obj/missing", "events": events,
            "frame_range": [1, 1], "apply": True,
        })
        assert result["status"] == "success", f"an unresolvable network must not raise, got {result!r}"
        data = result["data"]
        assert data.get("ok") is False, (
            f"an unresolvable `network` must degrade the WHOLE call to {{ok:false}} "
            f"per the contract -- regardless of whether any compiled entry's "
            f"target equals `network` verbatim. Got {data!r}"
        )
        assert mock_set_keyframes.call_count == 0

    def test_preview_matches_apply_on_unresolvable_network(self, mock_hou, mock_set_keyframes):
        mock_hou.node.side_effect = _node_lookup({})  # nothing resolves, incl. network itself

        handlers = _import_handler_module()
        events = [{
            "id": "kf1", "type": "keyframe", "target": "/obj/missing/child",
            "params": {"parm": "tx", "frames": [[1, 0]]},
        }]
        params = {"network": "/obj/missing", "events": events, "frame_range": [1, 1], "apply": True}

        preview = handlers._preview_compile_timeline(params)
        result = _dispatch("compile_timeline", params)
        apply_data = result["data"]

        assert apply_data.get("ok") is False, f"apply must degrade to ok:false, got {apply_data!r}"
        assert preview.get("ok") is False, (
            f"the PREVIEW must surface the SAME terminal unresolvable-network "
            f"condition apply does (Major-6 -- preview must never diverge from "
            f"apply). Got preview={preview!r}"
        )
        assert mock_set_keyframes.call_count == 0, (
            "the preview must perform NO writes at all."
        )


# ---------------------------------------------------------------------------
# Major-6 — _preview_compile_timeline mirrors the apply validation
# ---------------------------------------------------------------------------

class TestPreviewCompileTimelineMirrorsApplyValidation:
    """Major-6: _preview_compile_timeline runs the SAME read-only
    preflight validation apply does, so its would-apply/unresolved split
    matches exactly what apply would do -- WITHOUT any write."""

    def test_preview_reports_would_set_keyframes_and_unresolved_matching_apply(
        self, mock_hou, mock_set_keyframes
    ):
        # ROUND-3 FIX-PASS note (M3): `network` must resolve unconditionally.
        network_node = _make_node_with_parm(mock_hou, "/obj/rbd_sim", set())
        in_scope = _make_node_with_parm(mock_hou, "/obj/rbd_sim/inside", {"tx"})
        no_parm = _make_node_with_parm(mock_hou, "/obj/rbd_sim/no_parm", set())
        mock_hou.node.side_effect = _node_lookup({
            "/obj/rbd_sim": network_node,
            "/obj/rbd_sim/inside": in_scope,
            "/obj/rbd_sim/no_parm": no_parm,
        })

        handlers = _import_handler_module()
        events = [
            {"id": "kf_ok", "type": "keyframe", "target": "/obj/rbd_sim/inside",
             "params": {"parm": "tx", "frames": [[1, 0]]}},
            {"id": "kf_no_parm", "type": "keyframe", "target": "/obj/rbd_sim/no_parm",
             "params": {"parm": "tx", "frames": [[1, 0]]}},
        ]

        preview = handlers._preview_compile_timeline({
            "network": "/obj/rbd_sim", "events": events,
            "frame_range": [1, 1], "apply": True,
        })

        assert mock_set_keyframes.call_count == 0, (
            "the preview must perform NO writes at all."
        )
        would_nodes = {e["node"] for e in preview["would_set_keyframes"]}
        assert would_nodes == {"/obj/rbd_sim/inside"}, (
            f"expected only the validated in-scope+has-parm entry in "
            f"would_set_keyframes, got {preview['would_set_keyframes']!r}"
        )
        assert "kf_no_parm" in preview["unresolved"], (
            f"the missing-parm entry must appear in the preview's unresolved list "
            f"too, got {preview['unresolved']!r}"
        )
        assert preview["event_graph"] == {"nodes": 2, "edges": 0}

    def test_preview_split_matches_a_real_apply_on_the_same_input(self, mock_hou, mock_set_keyframes):
        """The preview's would_set_keyframes/unresolved split must be
        computable from the SAME validation apply performs -- proven here
        by running apply=true on the IDENTICAL events afterward and
        checking the SAME event ended up applied/unresolved."""
        # ROUND-3 FIX-PASS note (M3): `network` must resolve unconditionally.
        network_node = _make_node_with_parm(mock_hou, "/obj/rbd_sim", set())
        in_scope = _make_node_with_parm(mock_hou, "/obj/rbd_sim/inside", {"tx"})
        out_of_scope = _make_node_with_parm(mock_hou, "/obj/elsewhere", {"tx"})
        mock_hou.node.side_effect = _node_lookup({
            "/obj/rbd_sim": network_node,
            "/obj/rbd_sim/inside": in_scope, "/obj/elsewhere": out_of_scope,
        })
        handlers = _import_handler_module()

        events = [
            {"id": "kf_in", "type": "keyframe", "target": "/obj/rbd_sim/inside",
             "params": {"parm": "tx", "frames": [[1, 0]]}},
            {"id": "kf_out", "type": "keyframe", "target": "/obj/elsewhere",
             "params": {"parm": "tx", "frames": [[1, 0]]}},
        ]
        params = {"network": "/obj/rbd_sim", "events": events, "frame_range": [1, 1], "apply": True}

        preview = handlers._preview_compile_timeline(params)
        assert "kf_out" in preview["unresolved"]
        assert {e["node"] for e in preview["would_set_keyframes"]} == {"/obj/rbd_sim/inside"}

        result = _dispatch("compile_timeline", params)
        assert set(result["data"]["unresolved"]) == set(preview["unresolved"]), (
            "the preview's unresolved split must match a real apply's unresolved split "
            f"on identical input, got preview={preview['unresolved']!r} vs "
            f"apply={result['data']['unresolved']!r}"
        )
        assert mock_set_keyframes.call_count == 1, (
            "the real apply run must have applied exactly the ONE in-scope entry"
        )

    def test_preview_raise_denies(self, mock_hou):
        """A raise inside the preview -> the 109 gate DENIES (mirrors
        _preview_solve_layout's raise->DENY contract). A malformed event
        (unknown type) makes the pure compile_plan raise ValueError -- the
        preview must PROPAGATE it (never swallow it into a benign dict),
        so the gate denies the call."""
        handlers = _import_handler_module()
        with pytest.raises(ValueError):
            handlers._preview_compile_timeline({
                "network": "/obj/rbd_sim",
                "events": [{"id": "e1", "type": "not_a_real_event"}],
                "frame_range": [1, 1], "apply": True,
            })


# ---------------------------------------------------------------------------
# A model ValueError on a malformed event PROPAGATES (never folded into
# {ok:false}/unresolved)
# ---------------------------------------------------------------------------

class TestCompileTimelineMalformedEventPropagates:
    """A model-layer ValueError (a malformed event, e.g. an unknown type,
    or a cyclic causes graph) PROPAGATES through compile_timeline as the
    dispatcher's standard error envelope -- it must NEVER be folded into
    {ok:false} or a benign unresolved entry."""

    def test_direct_handler_call_raises_value_error(self, mock_hou):
        handlers = _import_handler_module()
        with pytest.raises(ValueError):
            handlers.compile_timeline(
                network="/obj/rbd_sim",
                events=[{"id": "e1", "type": "not_a_real_event"}],
                frame_range=[1, 1],
                apply=True,
            )

    def test_dispatch_propagates_as_error_envelope_not_ok_false(self, mock_hou):
        handlers = _import_handler_module()
        result = _dispatch("compile_timeline", {
            "network": "/obj/rbd_sim",
            "events": [{"id": "e1", "type": "not_a_real_event"}],
            "frame_range": [1, 1], "apply": True,
        })
        assert result["status"] == "error", (
            f"a caller-contract ValueError (malformed event) must propagate as the "
            f"DISPATCHER's error envelope (status:error), NOT be folded into a normal "
            f"ok:false/unresolved success return. Got {result!r}."
        )
        assert result["error"]["code"] == "ValueError", (
            f"expected error.code == 'ValueError', got {result['error']!r}"
        )

    def test_cyclic_causes_graph_propagates(self, mock_hou):
        handlers = _import_handler_module()
        events = [
            {"id": "e1", "type": "fracture", "causes": ["e2"]},
            {"id": "e2", "type": "emit", "causes": ["e1"]},
        ]
        result = _dispatch("compile_timeline", {
            "network": "/obj/rbd_sim", "events": events,
            "frame_range": [1, 1], "apply": True,
        })
        assert result["status"] == "error", f"a cyclic causes graph must propagate, got {result!r}"
        assert result["error"]["code"] == "ValueError"


# =============================================================================
# ROUND-3 FIX-PASS (unitId pp12-117c, planSha UNCHANGED -- 379e4799ad34...) --
# Sol's tier-3 diff-scoped RE-REVIEW (round 2 re-review) found 1 Blocker +
# 2 Majors STILL-OPEN after a PRIOR reviewer wrongly marked them CLOSED.
# See docs/homedini/plans/_agentic/_artifacts/codex-reviewer/pp12-117c/
# review-verdict-rereview.json. NO contract change -- these tests pin the
# ALREADY-LOCKED contract more precisely; they are RED against the CURRENT
# (round-2) impl and must go GREEN under a round-3 fix that does not
# regress anything above.
# =============================================================================

def _make_set_keyframes_with_mid_apply_failure(mock_hou, store: dict, *, fail_node: str, fail_parm: str):
    """A `_set_keyframes` side_effect that RECORDS a genuine write into
    *store* (keyed by (node_path, parm_name) -> set of written frames) for
    every call EXCEPT the one matching (fail_node, fail_parm), which
    RAISES hou.OperationFailed instead -- simulating a real Houdini-level
    write failure (e.g. a callback that locked the parm strictly AFTER
    preflight already observed isLocked() -> False; a genuine TOCTOU race
    no preflight check can close in advance). The raise happens ONLY at
    the write itself, never at construction, matching a real
    parm.setKeyframe() failing mid-apply."""

    def _fn(node_path, parm_name, keyframes):
        if node_path == fail_node and parm_name == fail_parm:
            raise mock_hou.OperationFailed(
                f"simulated mid-apply write failure on {parm_name} of {node_path} "
                f"(parm locked by a callback after preflight)"
            )
        store.setdefault((node_path, parm_name), set()).update(
            kf["frame"] for kf in keyframes
        )
        return {
            "node_path": node_path, "parm_name": parm_name,
            "keyframes_set": len(keyframes), "status": "keyframes_set",
        }

    return _fn


def _make_set_keyframes_with_mid_apply_runtime_error(store: dict, *, fail_node: str, fail_parm: str):
    """SIBLING of `_make_set_keyframes_with_mid_apply_failure` above, for
    Sol's re-review of REVISION 3 (docs/homedini/plans/_agentic/_artifacts/
    codex-reviewer/pp12-117c/review-verdict-rereview.json): the locked
    contract's B2 clause says the handler must catch "ANY mid-apply raise"
    and degrade to the documented {ok:false} partial-apply/Undo error --
    not merely the 3 narrow types `_apply_exception_types()` currently
    enumerates (hou.OperationFailed, AttributeError, hou.PermissionError
    when genuine). This variant raises a plain `RuntimeError` instead --
    an exception type OUTSIDE that narrow tuple -- from the SAME injection
    point (the `_set_keyframes` call for the SECOND validated entry), to
    pin that "ANY raise" is honored for a type the current except clause
    does NOT catch. Mirrors `_make_set_keyframes_with_mid_apply_failure`
    exactly except for the raised exception type (no `mock_hou` needed --
    RuntimeError is a real builtin, not a `hou.*` stand-in)."""

    def _fn(node_path, parm_name, keyframes):
        if node_path == fail_node and parm_name == fail_parm:
            raise RuntimeError(
                f"simulated mid-apply write failure on {parm_name} of {node_path} "
                f"(a non-hou.OperationFailed/AttributeError/PermissionError raise -- "
                f"e.g. an unexpected internal error inside setKeyframe)"
            )
        store.setdefault((node_path, parm_name), set()).update(
            kf["frame"] for kf in keyframes
        )
        return {
            "node_path": node_path, "parm_name": parm_name,
            "keyframes_set": len(keyframes), "status": "keyframes_set",
        }

    return _fn


# ---------------------------------------------------------------------------
# B2 (ATOMICITY, REVISION 3 -- native-undo based, NOT a bespoke rollback):
# on a mid-apply failure the handler STOPS immediately and returns the
# documented partial-apply/use-Undo {ok:false} error; it must NEVER call a
# bespoke `_delete_keyframe` compensating rollback (REVISION 3 REMOVED it --
# it was the source of a data-loss Blocker: deleting a keyframe the call had
# merely OVERWRITTEN instead of restoring its PRIOR value). Recovery is the
# NATIVE `hou.undos.group` undo block the operator reverts with a single
# Undo, plus the 109 operator preview/approve -- appropriate for a gated,
# undo-wrapped tool.
# ---------------------------------------------------------------------------

class TestCompileTimelineMidApplyFailureStopsAndReportsUndo:
    """REVISION 3 (operator decision 2026-07-11, after Sol's round-3
    confirmation found the round-2 bespoke-rollback fix introduced a
    DATA-LOSS Blocker -- deleting a keyframe the apply call OVERWROTE
    rather than restoring its PRIOR value -- and made the class this
    replaces (TestCompileTimelineAtomicApplyRollback) VACUOUS: its fixture
    never registered the network node itself ('/obj/rbd_sim'), so the M3
    unconditional up-front network-resolution check short-circuited to
    {ok:false} BEFORE the apply path -- and therefore the mid-apply-
    failure path -- ever ran; both the OLD (bespoke-rollback) and the NEW
    (native-undo) impl degrade to {ok:false} on an unresolvable network,
    so the old test passed for the WRONG reason).

    The rev-3 apply-time atomicity guarantee for this GATED tool is NOT a
    custom delete-based rollback. It is: (1) the ENTIRE validated set is
    applied inside exactly ONE `hou.undos.group(...)` (a single native
    undo step the operator reverts with one Undo); (2) on ANY mid-apply
    raise -- a rare TOCTOU (a parm that passed preflight isLocked()==False
    but a callback locks it, or any other genuine setKeyframe failure,
    strictly AFTER preflight) -- the handler STOPS and degrades the WHOLE
    call to {ok:false, error:'partial apply -- the gated call was
    interrupted mid-write; use Undo to revert'}. It does NOT call the
    SHIPPED compensating `_delete_keyframe` API -- there is NO bespoke
    rollback to invoke. The residual (a rare TOCTOU can leave a partial
    write) is ACCEPTED + documented as recoverable via the native undo
    block, not papered over with a custom delete that risks destroying a
    pre-existing keyframe value.

    This test REGISTERS the network node ('/obj/rbd_sim') in the node
    lookup -- unlike the vacuous class it replaces -- so the M3
    unconditional network-resolution check passes and this test genuinely
    REACHES `_apply_validated_keyframes` (not merely the {ok:false}
    network-unresolvable short-circuit).

    Assertions target the PUBLIC RETURNED CONTRACT (the {ok,error} shape)
    plus two narrow checks that ARE the rev-3 contract itself (not an
    accidental implementation-internals peek, tdd-with-agents.md Sec.2):
    `_delete_keyframe` must NEVER be called (there is no bespoke rollback
    to call), and `hou.undos.group` MUST have been entered (the native-
    undo recovery mechanism the contract names). This test deliberately
    does NOT assert whether entry-1's write is present or absent in any
    tracking store after the failure -- REVISION 3 explicitly leaves that
    outcome to Houdini's OWN native undo stack, which this mock cannot
    faithfully observe either way."""

    def test_mid_apply_failure_stops_and_returns_partial_apply_error(
        self, mock_hou, monkeypatch
    ):
        network_node = _make_node_with_parm(mock_hou, "/obj/rbd_sim", set())
        node_a = _make_node_with_parm(mock_hou, "/obj/rbd_sim/a", {"tx"})
        node_b = _make_node_with_parm(mock_hou, "/obj/rbd_sim/b", {"ty"})
        mock_hou.node.side_effect = _node_lookup({
            "/obj/rbd_sim": network_node,
            "/obj/rbd_sim/a": node_a, "/obj/rbd_sim/b": node_b,
        })

        import fxhoudinimcp_server.handlers.animation_handlers as _anim

        store: dict = {}
        fake_set_keyframes = MagicMock(
            name="_set_keyframes",
            side_effect=_make_set_keyframes_with_mid_apply_failure(
                mock_hou, store, fail_node="/obj/rbd_sim/b", fail_parm="ty",
            ),
        )
        # No side_effect: `_delete_keyframe` must never be reached under
        # the rev-3 contract -- a bare spy is sufficient to prove that.
        fake_delete_keyframe = MagicMock(name="_delete_keyframe")
        monkeypatch.setattr(_anim, "_set_keyframes", fake_set_keyframes)
        monkeypatch.setattr(_anim, "_delete_keyframe", fake_delete_keyframe)
        sys.modules.pop("fxhoudinimcp_server.handlers.temporal_reasoning_handlers", None)

        events = [
            {"id": "kf_a", "type": "keyframe", "target": "/obj/rbd_sim/a",
             "params": {"parm": "tx", "frames": [[1, 0], [10, 5]]}},
            {"id": "kf_b", "type": "keyframe", "target": "/obj/rbd_sim/b",
             "params": {"parm": "ty", "frames": [[1, 1], [10, 9]]}},
        ]

        result = _dispatch("compile_timeline", {
            "network": "/obj/rbd_sim", "events": events,
            "frame_range": [1, 10], "apply": True,
        })
        assert result["status"] == "success", (
            f"a mid-apply write failure must degrade to a normal dispatcher "
            f"return -- never an uncaught exception. Got {result!r}"
        )
        data = result["data"]
        assert data.get("ok") is False, (
            f"a mid-apply write failure must degrade the WHOLE call to "
            f"{{ok:false}}, got {data!r}"
        )
        assert data.get("error") == (
            "partial apply — the gated call was interrupted mid-write; "
            "use Undo to revert"
        ), (
            f"REVISION 3 pins the EXACT documented partial-apply error "
            f"string (lockedFieldContract 'REVISION 3 -- B2 ATOMICITY "
            f"GUARANTEE, CLARIFIED') -- got error={data.get('error')!r}"
        )
        assert fake_delete_keyframe.call_count == 0, (
            f"REVISION 3 REMOVED the bespoke rollback -- compile_timeline "
            f"must NEVER call the compensating _delete_keyframe API on a "
            f"mid-apply failure (it was the source of a data-loss Blocker: "
            f"deleting a keyframe the call OVERWROTE instead of restoring "
            f"its prior value). Got {fake_delete_keyframe.call_count} "
            f"call(s): {fake_delete_keyframe.call_args_list!r}"
        )
        assert mock_hou.undos.group.called, (
            "the ENTIRE apply must run inside exactly ONE hou.undos.group(...) "
            "block -- the native undo step the operator recovers a partial "
            "apply through -- but hou.undos.group was never entered."
        )

    def test_mid_apply_runtime_error_ALSO_stops_and_returns_partial_apply_error(
        self, mock_hou, monkeypatch
    ):
        """Sol's re-review (round 3): the locked B2 clause says the handler
        catches "ANY mid-apply raise ... (broadened except incl
        hou.PermissionError)" -- not merely the 3 narrow types
        `_apply_exception_types()` currently enumerates. This test injects
        a plain `RuntimeError` (a type OUTSIDE that tuple) from the SAME
        injection point as the sibling OperationFailed test above --
        entry-2's `_set_keyframes` call, on the APPLY path, never from the
        MODEL -- and pins the IDENTICAL rev-3 outcome: {ok:false} with the
        exact documented partial-apply/Undo error string, zero
        `_delete_keyframe` calls (there is no bespoke rollback to invoke),
        and `hou.undos.group` entered. This is RED against the CURRENT impl
        (whose except clause does not catch RuntimeError, so it escapes
        uncaught and the dispatcher degrades the call to {status:"error"}
        instead of the documented {status:"success", data:{ok:false,...}}
        -- proving the "ANY mid-apply raise" contract is not yet fully
        honored for an exception type outside the narrow tuple."""
        network_node = _make_node_with_parm(mock_hou, "/obj/rbd_sim", set())
        node_a = _make_node_with_parm(mock_hou, "/obj/rbd_sim/a", {"tx"})
        node_b = _make_node_with_parm(mock_hou, "/obj/rbd_sim/b", {"ty"})
        mock_hou.node.side_effect = _node_lookup({
            "/obj/rbd_sim": network_node,
            "/obj/rbd_sim/a": node_a, "/obj/rbd_sim/b": node_b,
        })

        import fxhoudinimcp_server.handlers.animation_handlers as _anim

        store: dict = {}
        fake_set_keyframes = MagicMock(
            name="_set_keyframes",
            side_effect=_make_set_keyframes_with_mid_apply_runtime_error(
                store, fail_node="/obj/rbd_sim/b", fail_parm="ty",
            ),
        )
        # No side_effect: `_delete_keyframe` must never be reached under
        # the rev-3 contract -- a bare spy is sufficient to prove that.
        fake_delete_keyframe = MagicMock(name="_delete_keyframe")
        monkeypatch.setattr(_anim, "_set_keyframes", fake_set_keyframes)
        monkeypatch.setattr(_anim, "_delete_keyframe", fake_delete_keyframe)
        sys.modules.pop("fxhoudinimcp_server.handlers.temporal_reasoning_handlers", None)

        events = [
            {"id": "kf_a", "type": "keyframe", "target": "/obj/rbd_sim/a",
             "params": {"parm": "tx", "frames": [[1, 0], [10, 5]]}},
            {"id": "kf_b", "type": "keyframe", "target": "/obj/rbd_sim/b",
             "params": {"parm": "ty", "frames": [[1, 1], [10, 9]]}},
        ]

        result = _dispatch("compile_timeline", {
            "network": "/obj/rbd_sim", "events": events,
            "frame_range": [1, 10], "apply": True,
        })
        assert result["status"] == "success", (
            f"a mid-apply write failure -- of ANY exception type, per the "
            f"locked B2 clause's 'ANY mid-apply raise' -- must degrade to a "
            f"normal dispatcher return, never an uncaught exception escaping "
            f"as {{status:'error'}}. Got {result!r}"
        )
        data = result["data"]
        assert data.get("ok") is False, (
            f"a mid-apply RuntimeError must degrade the WHOLE call to "
            f"{{ok:false}} exactly like a hou.OperationFailed does, got {data!r}"
        )
        assert data.get("error") == (
            "partial apply — the gated call was interrupted mid-write; "
            "use Undo to revert"
        ), (
            f"REVISION 3 pins the EXACT documented partial-apply error "
            f"string for ANY mid-apply raise (lockedFieldContract "
            f"'REVISION 3 -- B2 ATOMICITY GUARANTEE, CLARIFIED') -- got "
            f"error={data.get('error')!r}"
        )
        assert fake_delete_keyframe.call_count == 0, (
            f"REVISION 3 REMOVED the bespoke rollback -- compile_timeline "
            f"must NEVER call the compensating _delete_keyframe API on a "
            f"mid-apply failure of ANY type. Got "
            f"{fake_delete_keyframe.call_count} call(s): "
            f"{fake_delete_keyframe.call_args_list!r}"
        )
        assert mock_hou.undos.group.called, (
            "the ENTIRE apply must run inside exactly ONE hou.undos.group(...) "
            "block -- the native undo step the operator recovers a partial "
            "apply through -- but hou.undos.group was never entered."
        )


# ---------------------------------------------------------------------------
# M3 (MAJOR, round-3 re-review STILL-OPEN) -- network resolution must be
# checked UNCONDITIONALLY, up front -- not only when a compiled entry's
# own target ALSO fails to resolve as a node.
# ---------------------------------------------------------------------------

class TestCompileTimelineNetworkResolvedUpFrontUnconditionally:
    """M3 (MAJOR): per the ORIGINAL locked contract, step (1) is
    UNCONDITIONALLY 'resolve network via hou.node(network) ->
    unresolvable -> {ok:false,error}' -- BEFORE anything else. The
    round-2 impl's `_network_genuinely_unresolvable` only fires when
    `validated` is empty AND at least one dropped entry ALSO failed
    because ITS OWN target node could not be resolved
    (`node_unresolved_ids` non-empty). An `events=[]` call (or any call
    whose compiled set is empty -- e.g. every event routes to
    unresolved via compile_plan itself, producing zero
    compiled_by_id entries) makes `node_unresolved_ids` empty by
    construction (there is no target to even ATTEMPT resolving), so the
    OLD conditional check never asks whether `network` itself resolves
    at all -- silently returning an ordinary (non-error) result for a
    `network` that does not exist as a scene node. Both `compile_timeline`
    (apply=true AND apply=false) and `_preview_compile_timeline` must
    surface the SAME unconditional {ok:false,error}."""

    def test_apply_true_unresolvable_network_with_empty_events_is_ok_false(
        self, mock_hou, mock_set_keyframes
    ):
        mock_hou.node.side_effect = _node_lookup({})  # network genuinely does not resolve

        result = _dispatch("compile_timeline", {
            "network": "/obj/does_not_exist", "events": [],
            "frame_range": [1, 1], "apply": True,
        })
        assert result["status"] == "success", (
            f"an unresolvable network must not raise, got {result!r}"
        )
        data = result["data"]
        assert data.get("ok") is False, (
            f"network resolution must be checked UNCONDITIONALLY (contract "
            f"step 1), even with events=[] (zero compiled entries, so there "
            f"is no per-entry node-resolution failure to trigger the OLD "
            f"conditional check) -- expected {{ok:false}}, got a non-error "
            f"result: {data!r}"
        )
        assert mock_set_keyframes.call_count == 0

    def test_apply_false_unresolvable_network_with_empty_events_is_ok_false(
        self, mock_hou, mock_set_keyframes
    ):
        mock_hou.node.side_effect = _node_lookup({})

        result = _dispatch("compile_timeline", {
            "network": "/obj/does_not_exist", "events": [],
            "frame_range": [1, 1], "apply": False,
        })
        assert result["status"] == "success", (
            f"an unresolvable network must not raise, got {result!r}"
        )
        data = result["data"]
        assert data.get("ok") is False, (
            f"apply=false must ALSO surface the unresolvable-network "
            f"{{ok:false}} unconditionally (the gate is per-COMMAND, the "
            f"scene-resolution check is not conditioned on apply), got "
            f"{data!r}"
        )
        assert mock_set_keyframes.call_count == 0

    def test_preview_unresolvable_network_with_empty_events_is_ok_false(self, mock_hou):
        mock_hou.node.side_effect = _node_lookup({})
        handlers = _import_handler_module()

        preview = handlers._preview_compile_timeline({
            "network": "/obj/does_not_exist", "events": [],
            "frame_range": [1, 1], "apply": True,
        })
        assert preview.get("ok") is False, (
            f"_preview_compile_timeline must ALSO surface the SAME "
            f"unconditional unresolvable-network {{ok:false}} for "
            f"events=[] (Major-6 -- preview must never diverge from "
            f"apply), got {preview!r}"
        )


# ---------------------------------------------------------------------------
# B1-residual (MAJOR, round-3 re-review STILL-OPEN) -- compare targets
# against the CANONICAL resolved network path, not the RAW `network` arg.
# ---------------------------------------------------------------------------

class TestCompileTimelineNetworkCanonicalizedOnce:
    """B1-residual (MAJOR): the round-2 Blocker-1 fix already compares a
    TARGET's resolved CANONICAL path against `network` -- but `network`
    ITSELF is still compared as the RAW caller-supplied string, never
    independently resolved/canonicalized. Two consequences:

    (a) a non-canonical but RESOLVABLE `network` arg (e.g. a trailing
        slash) false-REJECTS an otherwise perfectly in-scope target --
        `_target_in_network_scope(canonical_target, "/obj/rbd_sim/")`
        computes `canonical_target.startswith("/obj/rbd_sim//")` (double
        slash), which never matches a real canonical child path.

    (b) an EMPTY/unresolvable `network` string BYPASSES scope entirely --
        `_target_in_network_scope(target, "")` returns True for virtually
        ANY absolute Houdini path (`target.startswith("" + "/")` is True
        for any path beginning with '/'), an unscoped-write security hole.

    The fix resolves `network` to its OWN canonical
    `hou.node(network).path()` ONCE, up front (folding into the SAME
    unconditional resolution M3 requires), and compares every target's
    canonical path against THAT resolved canonical network path -- never
    the raw `network` string."""

    def test_non_canonical_trailing_slash_network_still_accepts_in_scope_target(
        self, mock_hou, mock_set_keyframes
    ):
        network_node = _make_node_with_parm(mock_hou, "/obj/rbd_sim", set())
        target_node = _make_node_with_parm(mock_hou, "/obj/rbd_sim/box", {"tx"})
        # hou.node() is invoked with the RAW `network` string the caller
        # supplied ("/obj/rbd_sim/" -- a trailing slash); Houdini itself
        # resolves that to the SAME canonical node as "/obj/rbd_sim" would
        # -- the node's OWN .path() reports the CANONICAL form (no
        # trailing slash), exactly as a real hou.Node.path() would.
        mock_hou.node.side_effect = _node_lookup({
            "/obj/rbd_sim/": network_node,
            "/obj/rbd_sim/box": target_node,
        })

        events = [{"id": "kf1", "type": "keyframe", "target": "/obj/rbd_sim/box",
                   "params": {"parm": "tx", "frames": [[1, 0]]}}]

        result = _dispatch("compile_timeline", {
            "network": "/obj/rbd_sim/",  # NON-canonical (trailing slash)
            "events": events, "frame_range": [1, 1], "apply": True,
        })
        assert result["status"] == "success", (
            f"a resolvable non-canonical `network` string must not crash, "
            f"got {result!r}"
        )
        data = result["data"]
        assert data["unresolved"] == [], (
            f"a perfectly valid, in-scope target must NOT be false-rejected "
            f"to unresolved merely because `network` was supplied in a "
            f"non-canonical form (the raw-string scope comparison bug) -- "
            f"got unresolved={data['unresolved']!r}"
        )
        assert mock_set_keyframes.call_count == 1, (
            f"the in-scope target must actually be APPLIED once `network` "
            f"is resolved to its own canonical path for the scope "
            f"comparison, got {mock_set_keyframes.call_count} call(s)"
        )

    def test_empty_network_string_does_not_bypass_scope_and_degrades_ok_false(
        self, mock_hou, mock_set_keyframes
    ):
        """An empty `network` does not resolve to any real scene node -- it
        must degrade the WHOLE call to {ok:false} (the same up-front,
        unconditional network-resolution requirement M3 pins), NOT be
        silently treated as 'no scope boundary'. The raw-string comparison
        `target.startswith('' + '/')` is True for virtually every absolute
        Houdini path -- an unscoped-write security hole this test closes."""
        target_node = _make_node_with_parm(mock_hou, "/obj/anything/at/all", {"tx"})
        mock_hou.node.side_effect = _node_lookup({"/obj/anything/at/all": target_node})
        # "" is deliberately NOT registered -- hou.node("") does not
        # resolve to any real scene node.

        events = [{"id": "kf1", "type": "keyframe", "target": "/obj/anything/at/all",
                   "params": {"parm": "tx", "frames": [[1, 0]]}}]

        result = _dispatch("compile_timeline", {
            "network": "", "events": events,
            "frame_range": [1, 1], "apply": True,
        })
        assert result["status"] == "success", f"an empty network must not raise, got {result!r}"
        data = result["data"]
        assert data.get("ok") is False, (
            f"an empty/unresolvable `network` must degrade the WHOLE call "
            f"to {{ok:false}} -- it must NEVER be treated as an unscoped "
            f"wildcard that accepts an arbitrary target. Got {data!r}"
        )
        assert mock_set_keyframes.call_count == 0, (
            f"an empty network must NEVER result in a write -- expected 0 "
            f"_set_keyframes calls, got {mock_set_keyframes.call_count}"
        )
