"""Handler-level (mocked-hou) pytest tests for describe_relations and
assert_scene (PP12-116 PR-2, READONLY) — the REVISION-2 lockedFieldContract
red suite.

Unit: pp12-116b
testVerificationSurface: pytest-model
planSha: 1e391cf7bd9173ea2219a4a186bedf7c2bc5127809f72a8df25aaa1ad3eb33a5

These tests import the HANDLER module (spatial_reasoning_handlers.py) with
a mocked `hou` module installed into sys.modules BEFORE import, per
test-fixture-conventions.md §2.3 (`monkeypatch.setitem(sys.modules, 'hou',
fake)`). This lets the handler's Python control flow — geometry resolution
for BOTH hou.ObjNode and hou.SopNode paths, the world-AABB-center-not-origin
math, the (w,d,h) Y-up extent mapping, the per-object resolution table, and
the {ok:false,error} vs ValueError-propagates try/except boundary — be
exercised off-DCC without a real Houdini session.

This is a genuinely mocked-hou rung (test-fixture-conventions.md §2 — the
handler cannot be fully split into *_model.py because it calls
hou.node()/node.geometry()/node.worldTransform() as its core job), NOT a
substitute for the real hython-smoke (hou-dev's authority on real geometry
reads) or the MANDATORY live-MCP subprocess rung named in the plan's
lockedFieldContract "verification" clause (orchestrator-run, out of scope
for this red-test unit).

Tests exercise the REAL dispatcher (`fxhoudinimcp_server.dispatcher.dispatch`)
— NOT a direct `handler({dict})` call — per the plan's HANDLER-test
directive: `dispatch(cmd, {params})` calls `handler(**params)`, the exact
calling convention the shipped MCP fork uses end-to-end (the PP12-110 4-bug
convention class: a direct call bypasses the dispatcher's own calling
convention and would miss a signature mismatch).

Coverage this file pins (plan pp12-116b lockedFieldContract, REVISION 2):
  - describe_relations is a PURE delegate (no hou) to
    spatial_reasoning_model.describe_relations() — the exact SPEC 4.1 vocab
    dict, verbatim.
  - assert_scene, caller-supplies-everything path: zero hou read needed
    (node may be absent) -- delegates to the byte-frozen model and returns
    its 4.1 dict verbatim.
  - assert_scene, THE LOAD-BEARING off-origin-center pin (Blocker-3),
    RESTRUCTURED (fix-cycle 2) to the spec's PRIMARY case -- an hou.ObjNode
    path: an omitted bbox+transform resolves via a TRUE world-AABB
    (8-corner) read whose CENTER (not worldTransform().extractTranslates(),
    the origin) becomes `t`, and whose Y-up (w=x,d=z,h=y) extent mapping is
    correct -- pinned via a single collision-count outcome assertion (no
    internals inspected) that a center==origin OR a w/d/h-swapped
    implementation both fail.
  - assert_scene, NEW (fix-cycle 2) SopNode-branch pin
    (TestAssertSceneSopNodeUsesObjAncestorWorldTransform): a SopNode's
    worldTransform MUST be resolved via its OBJ ancestor (node.parent()),
    NEVER by calling worldTransform() directly on the SopNode (a real
    hou.SopNode exposes no such method -- the mock_hou SopNode fixture is
    now spec-restricted to {geometry, parent, path} to match reality, per
    the codex-adversarial-reviewer Blocker+Major fold). THIS TEST FAILS
    against the pre-fix-cycle-2 implementation, which calls
    node.worldTransform() unconditionally regardless of node type.
  - assert_scene, ObjNode path: resolves geometry via
    displayNode().geometry() -- NOT node.geometry() directly (Blocker-2).
  - assert_scene, missing/unresolvable node (a null hou.node() result, an
    ObjNode whose displayNode() is None, or a node that is neither
    hou.SopNode nor hou.ObjNode) -- degrades to {"ok": false, "error": ...}
    WITHOUT raising (Blocker-4 scene-resolution try/except boundary).
  - assert_scene, a bad relation type -- a caller-contract ValueError
    PROPAGATES (is NOT folded into {ok:false}) -- verified both via a
    direct handler call (pytest.raises(ValueError)) and via the dispatcher's
    standard error envelope (status:"error", error.code:"ValueError").
  - Both commands are registered Capability.READONLY.

Assertions target the RETURNED DICT (the public contract) -- never which
hou.* methods were called or in what order (tdd-with-agents.md §2 mirror-
test ban; test-fixture-conventions.md §2.3 discipline). The ObjNode-vs-
node.geometry() and off-origin-center tests are deliberately OUTCOME-based
(a wrong resolution path/formula produces an observably wrong collision
result) rather than call-count assertions, so they cannot be gamed by an
implementation that merely LOOKS like it follows the contract.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Minimal REAL (non-MagicMock) geometry stand-ins — hou.BoundingBox /
# hou.Vector3 / hou.Matrix4 — so the off-origin/ObjNode fixtures below can
# perform a genuine affine point-transform (not a special-cased stub),
# robust to whatever correct 8-corner-transform code shape hou-dev writes
# (Vector3.__mul__(Matrix4) OR manual per-component `at(r, c)` access).
# ---------------------------------------------------------------------------

class _FakeVector3(tuple):
    """A tuple-based hou.Vector3 stand-in supporting indexed access (v[0],
    v[1], v[2]) AND the `v * matrix` point-transform idiom."""

    def __new__(cls, x, y=None, z=None):
        if y is None and z is None:
            x, y, z = x[0], x[1], x[2]
        return super().__new__(cls, (float(x), float(y), float(z)))

    def __mul__(self, other):
        if isinstance(other, _FakeMatrix4):
            x, y, z = self
            m = other
            nx = x * m.at(0, 0) + y * m.at(1, 0) + z * m.at(2, 0) + m.at(3, 0)
            ny = x * m.at(0, 1) + y * m.at(1, 1) + z * m.at(2, 1) + m.at(3, 1)
            nz = x * m.at(0, 2) + y * m.at(1, 2) + z * m.at(2, 2) + m.at(3, 2)
            return _FakeVector3(nx, ny, nz)
        return NotImplemented


class _FakeMatrix4:
    """A genuine row-major 4x4 affine-transform stand-in for hou.Matrix4 —
    translate-only (no rotation baked into the multiply rows, since these
    fixtures only need identity/translate), with a real `.at(r, c)` so a
    manual per-component transform implementation works too."""

    def __init__(self, translate=(0.0, 0.0, 0.0), rotate_deg=(0.0, 0.0, 0.0)):
        self._rows = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        self._rows[3][0], self._rows[3][1], self._rows[3][2] = translate
        self._translate = tuple(translate)
        self._rotate_deg = tuple(rotate_deg)

    def at(self, r, c):
        return self._rows[r][c]

    def extractTranslates(self):
        return _FakeVector3(*self._translate)

    def extractRotates(self):
        return _FakeVector3(*self._rotate_deg)


class _FakeBoundingBox:
    """hou.BoundingBox stand-in: .minvec() / .maxvec() only (the two calls
    the locked contract names)."""

    def __init__(self, minv, maxv):
        self._minv = _FakeVector3(*minv)
        self._maxv = _FakeVector3(*maxv)

    def minvec(self):
        return self._minv

    def maxvec(self):
        return self._maxv


class _FakeOperationFailed(Exception):
    """Stand-in for hou.OperationFailed — a plain Exception subclass is
    sufficient since the handler only does `except hou.OperationFailed`."""


class _FakeSopNode(MagicMock):
    """Stand-in for hou.SopNode — a distinct type so isinstance(n, hou.SopNode)
    is True only for these instances (not hou.ObjNode instances), matching
    Houdini's real class hierarchy discrimination without modeling the
    actual hou.Node base classes.

    SPEC-RESTRICTED to {geometry, parent, path} ONLY — matching reality: a
    real hou.SopNode has NO `worldTransform` method (only hou.ObjNode does;
    a SOP has no world transform of its own, its world placement is its
    CONTAINING OBJECT's worldTransform(), found by walking node.parent() up
    to an hou.ObjNode). Accessing `.worldTransform` on an instance now
    raises AttributeError exactly as it would on a real hou.SopNode.

    This restriction is the fix for the codex-adversarial-reviewer Major
    finding (fix-cycle 2, docs/homedini/plans/_agentic/_artifacts/
    codex-adversarial-reviewer/pp12-116b/plan-review.md): the prior fixture
    mocked a SopNode WITH worldTransform (more permissive than reality), so
    the exact Blocker this fixture is meant to pin — the handler calling
    `node.worldTransform()` directly on a SopNode instead of walking to its
    OBJ ancestor — could not surface as a test failure."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("spec", ["geometry", "parent", "path"])
        super().__init__(*args, **kwargs)


class _FakeObjNode(MagicMock):
    """Stand-in for hou.ObjNode — see _FakeSopNode."""


class _FakeOtherNode(MagicMock):
    """Stand-in for a hou.Node subtype that is NEITHER Sop NOR Obj (e.g. a
    LOP node) — exercises the handler's `else -> scene-resolution error`
    branch."""


def _node_lookup(mapping):
    """A hou.node() side_effect robust to either a positional or keyword
    `path` call convention."""

    def _fn(*args, **kwargs):
        path = args[0] if args else kwargs.get("path")
        return mapping.get(path)

    return _fn


# ---------------------------------------------------------------------------
# mock_hou fixture — installs a fake `hou` module into sys.modules BEFORE
# the handler module is imported (test-fixture-conventions.md §2.3).
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_hou(monkeypatch):
    """Install a mock `hou` module so spatial_reasoning_handlers.py loads
    off-DCC. Returns the mock so tests can configure hou.node.side_effect,
    etc."""
    fake = MagicMock(name="hou")
    fake.OperationFailed = _FakeOperationFailed
    fake.SopNode = _FakeSopNode
    fake.ObjNode = _FakeObjNode
    monkeypatch.setitem(sys.modules, "hou", fake)

    # Ensure the fork's non-standard package roots are importable (mirrors
    # the shipped exemplar test_cop_onnx_run_inference_handler.py's
    # sys.path bootstrap): pytest's rootdir discovery does not put
    # houdini/scripts/python (fxhoudinimcp_server's home) on sys.path by
    # default.
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
    """Import spatial_reasoning_handlers fresh (relies on mock_hou already
    being installed in sys.modules AND sys.path already patched by the
    caller's mock_hou fixture)."""
    import importlib
    if "fxhoudinimcp_server.handlers.spatial_reasoning_handlers" in sys.modules:
        importlib.reload(sys.modules["fxhoudinimcp_server.handlers.spatial_reasoning_handlers"])
        return sys.modules["fxhoudinimcp_server.handlers.spatial_reasoning_handlers"]
    import fxhoudinimcp_server.handlers.spatial_reasoning_handlers as _handlers
    return _handlers


def _dispatch(command: str, params: dict):
    """Import the REAL dispatcher fresh each call (cheap — it's already
    cached in sys.modules after the first import) and route through it,
    per the plan's HANDLER-test directive (NOT a direct handler({dict})
    call)."""
    from fxhoudinimcp_server.dispatcher import dispatch as _real_dispatch
    return _real_dispatch(command, params)


# ---------------------------------------------------------------------------
# PRIMARY RED GATE — both handler functions must exist
# ---------------------------------------------------------------------------

class TestHandlerImport:
    def test_describe_relations_handler_importable(self, mock_hou):
        """spatial_reasoning_handlers.py must expose describe_relations.
        FAILS RED (ImportError/ModuleNotFoundError) until hou-dev implements
        it."""
        handlers = _import_handler_module()
        assert hasattr(handlers, "describe_relations"), (
            "spatial_reasoning_handlers.py must expose describe_relations."
        )
        assert callable(handlers.describe_relations)

    def test_assert_scene_handler_importable(self, mock_hou):
        """spatial_reasoning_handlers.py must expose assert_scene. FAILS RED
        until hou-dev implements it."""
        handlers = _import_handler_module()
        assert hasattr(handlers, "assert_scene"), (
            "spatial_reasoning_handlers.py must expose assert_scene."
        )
        assert callable(handlers.assert_scene)


# ---------------------------------------------------------------------------
# Capability.READONLY registration
# ---------------------------------------------------------------------------

class TestCapabilityReadonly:
    """Both commands must be registered Capability.READONLY — no 109 gate."""

    def test_describe_relations_capability_readonly(self, mock_hou):
        _import_handler_module()
        from fxhoudinimcp_server import dispatcher
        assert dispatcher.capability_of("describe_relations") == dispatcher.Capability.READONLY, (
            f"describe_relations must be registered Capability.READONLY, "
            f"got {dispatcher.capability_of('describe_relations')!r}."
        )

    def test_assert_scene_capability_readonly(self, mock_hou):
        _import_handler_module()
        from fxhoudinimcp_server import dispatcher
        assert dispatcher.capability_of("assert_scene") == dispatcher.Capability.READONLY, (
            f"assert_scene must be registered Capability.READONLY, "
            f"got {dispatcher.capability_of('assert_scene')!r}."
        )


# ---------------------------------------------------------------------------
# describe_relations — a PURE delegate to the model (no hou touched)
# ---------------------------------------------------------------------------

class TestDescribeRelationsDelegatesToModel:
    """describe_relations() is a pure delegate: `return
    spatial_reasoning_model.describe_relations()` — no hou read at all. The
    dispatched result must equal the model's own return value exactly, AND
    match the SPEC 4.1 shape directly (7 relation types, stable order,
    name/params/desc keys each)."""

    def test_dispatch_returns_the_exact_model_vocab_dict(self, mock_hou):
        _import_handler_module()
        from fxhoudinimcp import spatial_reasoning_model as model

        result = _dispatch("describe_relations", {})

        assert result["status"] == "success", f"describe_relations must not raise, got {result!r}"
        expected = model.describe_relations()
        assert result["data"] == expected, (
            f"describe_relations handler must return the model's exact SPEC 4.1 "
            f"vocab dict verbatim. Expected {expected!r}, got {result['data']!r}."
        )

    def test_vocab_shape_is_spec_4_1(self, mock_hou):
        _import_handler_module()
        result = _dispatch("describe_relations", {})
        relations = result["data"]["relations"]
        assert len(relations) == 7, f"expected 7 relation types, got {len(relations)}"
        names = [r["name"] for r in relations]
        assert names == [
            "on_top_of", "under", "adjacent", "non_overlap",
            "aligned", "oriented_toward", "clearance",
        ], f"unexpected relation name order: {names!r}"
        for r in relations:
            assert set(r.keys()) == {"name", "params", "desc"}, (
                f"each relation vocab entry must have exactly {{name,params,desc}} keys, "
                f"got {set(r.keys())!r} for {r!r}"
            )


# ---------------------------------------------------------------------------
# assert_scene — caller supplies objects + full transforms (zero hou read)
# ---------------------------------------------------------------------------

class TestAssertSceneCallerSuppliedFull:
    """When BOTH bbox and transform are caller-supplied for every object,
    the per-object resolution table requires ZERO hou reads (node may be
    absent) — the handler must delegate ALL geometry math to the byte-
    frozen model and return its 4.1 dict verbatim."""

    def test_two_overlapping_boxes_delegates_to_model_verbatim(self, mock_hou):
        handlers = _import_handler_module()
        from fxhoudinimcp import spatial_reasoning_model as model

        objects_wire = [
            {"id": "box_a", "bbox": [2.0, 2.0, 2.0], "fixed": False, "node": ""},
            {"id": "box_b", "bbox": [2.0, 2.0, 2.0], "fixed": False, "node": ""},
        ]
        transforms_wire = {
            "box_a": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "box_b": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
        }

        result = _dispatch("assert_scene", {
            "objects": objects_wire,
            "transforms": transforms_wire,
            "relations": None,
            "checks": None,
        })

        assert result["status"] == "success", (
            f"handler must not raise for a fully caller-supplied scene, got {result!r}"
        )
        data = result["data"]

        # Oracle: the SAME resolved inputs, called directly against the
        # byte-frozen model (both objects fully caller-supplied -> zero hou
        # read needed, so the handler's output must equal this exactly).
        objects_spec = [model._object_from_dict(d) for d in objects_wire]
        expected = model.assert_scene(objects_spec, transforms_wire, [], None)

        assert data == expected, (
            f"assert_scene handler must delegate ALL geometry math to the byte-frozen "
            f"model verbatim when objects+transforms are fully caller-supplied. "
            f"Expected {expected!r}, got {data!r}."
        )
        assert data["collision"]["count"] == 1, "two identical fully-overlapping boxes must collide"
        assert data["pass"] is False

    def test_non_overlapping_boxes_pass_true(self, mock_hou):
        handlers = _import_handler_module()

        objects_wire = [
            {"id": "box_a", "bbox": [1.0, 1.0, 1.0], "fixed": False, "node": ""},
            {"id": "box_b", "bbox": [1.0, 1.0, 1.0], "fixed": False, "node": ""},
        ]
        transforms_wire = {
            "box_a": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
            "box_b": {"t": [10.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
        }

        result = _dispatch("assert_scene", {
            "objects": objects_wire,
            "transforms": transforms_wire,
            "relations": None,
            "checks": ["collision"],
        })

        assert result["status"] == "success"
        data = result["data"]
        assert data["collision"]["count"] == 0
        assert data["pass"] is True


# ---------------------------------------------------------------------------
# THE LOAD-BEARING TEST (Blocker-3): omitted bbox+transform -> world-AABB
# CENTER (not extractTranslates -- the origin), correct (w,d,h) mapping.
# ---------------------------------------------------------------------------

class TestAssertSceneOffOriginCenterPin:
    """The handler's hou-read `t` MUST be the WORLD-SPACE AABB CENTER, never
    worldTransform().extractTranslates() (the node origin/pivot) -- and the
    (w,d,h) extent mapping (w=x-extent, d=z-extent, h=y-extent) must be
    correct.

    RESTRUCTURED (fix-cycle 2) to the SPEC'S PRIMARY CASE: an hou.ObjNode
    path (the spec's example paths, e.g. /obj/table, are OBJ paths).
    Resolved via displayNode().geometry() (Blocker-2) with
    node.worldTransform() called on the ObjNode ITSELF -- correct for an
    ObjNode (an ObjNode DOES have worldTransform in reality; contrast the
    sibling TestAssertSceneSopNodeUsesObjAncestorWorldTransform test below,
    where worldTransform() must NOT be called on the node itself since a
    SopNode has no such method).

    A geometry that sits OFF-CENTER in the node's local space (local bbox
    min=(1,10,100), max=(3,14,108) -> extents x=2,y=4,z=8 -> local/world
    CENTER=(2,12,104)), combined with an IDENTITY world transform
    (translate=(0,0,0)), makes the world-AABB center provably DIFFERENT from
    worldTransform().extractTranslates() == (0,0,0) (the origin) -- a
    center==origin implementation bug fails this test outright (the anchor,
    placed relative to the TRUE center, would show no collision at all if
    t resolved to the origin instead).

    The companion 'anchor' object is placed exactly 3 units along +Z from
    box_off's CORRECT center (104 -> 107). box_off's correct z-extent (d) is
    8 (half=4); a WRONG w/d/h mapping (e.g. swapping d<->h so the effective
    z-half-extent becomes h/2=2 instead of d/2=4) flips the AABB-overlap
    outcome on that single axis: (4 + 0.05) - 3 = 1.05 > 0 (overlap,
    CORRECT mapping) vs (2 + 0.05) - 3 = -0.95 <= 0 (no overlap, WRONG
    mapping). So this single collision-count assertion pins BOTH the
    center-not-origin fact AND the w=x/d=z/h=y axis mapping in one
    observable outcome (no internals inspected).
    """

    def test_omitted_bbox_and_transform_use_world_aabb_center_not_origin(self, mock_hou):
        handlers = _import_handler_module()

        obj_node = mock_hou.ObjNode(name="fake_obj_box_off")
        obj_node.path.return_value = "/obj/box_off"
        obj_node.worldTransform.return_value = _FakeMatrix4(
            translate=(0.0, 0.0, 0.0), rotate_deg=(0.0, 0.0, 0.0)
        )

        display_sop = MagicMock(name="fake_display_sop_box_off")
        geo = MagicMock(name="fake_geo_box_off")
        geo.boundingBox.return_value = _FakeBoundingBox((1.0, 10.0, 100.0), (3.0, 14.0, 108.0))
        display_sop.geometry.return_value = geo
        obj_node.displayNode.return_value = display_sop

        mock_hou.node.side_effect = _node_lookup({"/obj/box_off": obj_node})

        objects_wire = [
            {"id": "anchor", "bbox": [0.1, 0.1, 0.1], "fixed": False, "node": ""},
            {"id": "box_off", "bbox": None, "fixed": False, "node": "/obj/box_off"},
        ]
        transforms_wire = {
            "anchor": {"t": [2.0, 12.0, 107.0], "r": [0.0, 0.0, 0.0]},
        }

        result = _dispatch("assert_scene", {
            "objects": objects_wire,
            "transforms": transforms_wire,
            "relations": None,
            "checks": ["collision"],
        })

        assert result["status"] == "success", (
            f"expected a normal (non-raising) result, got {result!r}"
        )
        data = result["data"]
        assert data["collision"]["count"] == 1, (
            "anchor (placed 3 units +Z from box_off's TRUE world-AABB center "
            "(2,12,104)) must collide with box_off -- this only happens if the "
            "handler resolved t=(2,12,104) [the AABB CENTER] with the correct "
            "d=z-extent=8 (half=4), NOT t=(0,0,0) [extractTranslates -- the "
            "origin] and NOT a w/d/h-swapped mapping. "
            f"Got collision={data['collision']!r}."
        )
        assert data["collision"]["pairs"] == [["anchor", "box_off"]]


# ---------------------------------------------------------------------------
# NEW (fix-cycle 2): SopNode branch -- a SopNode's world transform MUST be
# resolved via its OBJ ancestor (node.parent()), NEVER by calling
# worldTransform() directly on the SopNode (which has no such method on a
# real hou.SopNode). THIS TEST IS EXPECTED TO FAIL against the pre-fix
# implementation (see docstring below).
# ---------------------------------------------------------------------------

class TestAssertSceneSopNodeUsesObjAncestorWorldTransform:
    """A hou.SopNode has NO worldTransform of its own -- its world placement
    is its CONTAINING OBJECT's worldTransform(), found by walking
    node.parent() up to an hou.ObjNode. The handler MUST NOT call
    node.worldTransform() directly on a SopNode; it must walk to the OBJ
    ancestor via node.parent() and use THAT node's worldTransform().

    Uses the SAME off-center local bbox as the ObjNode pin above (local
    bbox min=(1,10,100), max=(3,14,108) -> center (2,12,104)) so a correct
    implementation produces the IDENTICAL collision outcome via the SopNode
    + OBJ-ancestor path as the ObjNode-direct path does above -- proving the
    two branches are equivalent once correctly resolved.

    THIS TEST IS EXPECTED TO FAIL against the pre-fix-cycle-2
    implementation: that implementation calls `wt = node.worldTransform()`
    unconditionally inside `_world_aabb_center_and_extents` (regardless of
    SopNode vs ObjNode), which on this spec-restricted SopNode mock (no
    `worldTransform` attribute -- matching a real hou.SopNode) raises
    AttributeError. That AttributeError is caught by assert_scene's
    scene-resolution try/except (Blocker-4's boundary) and folded into
    {"ok": False, "error": ...} instead of the expected collision result --
    exactly the codex-adversarial-reviewer Blocker this fixture pins.
    """

    def test_sopnode_worldtransform_resolved_via_obj_ancestor_parent(self, mock_hou):
        handlers = _import_handler_module()

        sop_node = mock_hou.SopNode(name="fake_sop_box_off_child")
        sop_node.path.return_value = "/obj/geo1/box_off"
        geo = MagicMock(name="fake_geo_box_off_sop")
        geo.boundingBox.return_value = _FakeBoundingBox((1.0, 10.0, 100.0), (3.0, 14.0, 108.0))
        sop_node.geometry.return_value = geo

        obj_ancestor = mock_hou.ObjNode(name="fake_obj_ancestor")
        obj_ancestor.path.return_value = "/obj/geo1"
        obj_ancestor.worldTransform.return_value = _FakeMatrix4(
            translate=(0.0, 0.0, 0.0), rotate_deg=(0.0, 0.0, 0.0)
        )
        sop_node.parent.return_value = obj_ancestor

        mock_hou.node.side_effect = _node_lookup({"/obj/geo1/box_off": sop_node})

        objects_wire = [
            {"id": "anchor", "bbox": [0.1, 0.1, 0.1], "fixed": False, "node": ""},
            {"id": "box_off", "bbox": None, "fixed": False, "node": "/obj/geo1/box_off"},
        ]
        transforms_wire = {
            "anchor": {"t": [2.0, 12.0, 107.0], "r": [0.0, 0.0, 0.0]},
        }

        result = _dispatch("assert_scene", {
            "objects": objects_wire,
            "transforms": transforms_wire,
            "relations": None,
            "checks": ["collision"],
        })

        assert result["status"] == "success", (
            f"expected a normal (non-raising) result, got {result!r}"
        )
        data = result["data"]
        assert data.get("ok") is not False, (
            "assert_scene must resolve a SopNode's world transform via its "
            "OBJ ancestor (node.parent()), NOT by calling worldTransform() "
            "directly on the SopNode (a real hou.SopNode has no such "
            f"method). Got a scene-resolution failure instead: {data!r}"
        )
        assert "collision" in data, (
            f"expected a resolved collision result (not a degraded "
            f"scene-resolution error), got {data!r}"
        )
        assert data["collision"]["count"] == 1, (
            "anchor (placed 3 units +Z from box_off's TRUE world-AABB "
            "center (2,12,104), resolved via the SopNode's OBJ ANCESTOR "
            f"worldTransform) must collide with box_off. Got "
            f"collision={data['collision']!r}."
        )
        assert data["collision"]["pairs"] == [["anchor", "box_off"]]


# ---------------------------------------------------------------------------
# ObjNode path: resolves geometry via displayNode().geometry() (Blocker-2)
# ---------------------------------------------------------------------------

class TestAssertSceneObjNodePath:
    """An ObjNode wire node path resolves geometry via
    displayNode().geometry() -- NOT node.geometry() directly (node.geometry()
    does not exist on a real hou.ObjNode; here it is deliberately configured
    to return a DIFFERENT, clearly-wrong geometry so a handler that
    mistakenly calls node.geometry() directly produces an observably wrong
    result)."""

    def test_obj_node_resolves_via_display_node_geometry(self, mock_hou):
        handlers = _import_handler_module()

        obj_node = mock_hou.ObjNode(name="fake_obj_table")
        obj_node.path.return_value = "/obj/table1"
        obj_node.worldTransform.return_value = _FakeMatrix4(
            translate=(0.0, 0.0, 0.0), rotate_deg=(0.0, 0.0, 0.0)
        )

        # The CORRECT geometry -- reachable ONLY via displayNode().geometry().
        # w=4 (x-ext), d=6 (z-ext), h=2 (y-ext); center = (2, 1, 3).
        display_sop = MagicMock(name="fake_display_sop")
        correct_geo = MagicMock(name="fake_correct_geo")
        correct_geo.boundingBox.return_value = _FakeBoundingBox((0.0, 0.0, 0.0), (4.0, 2.0, 6.0))
        display_sop.geometry.return_value = correct_geo
        obj_node.displayNode.return_value = display_sop

        # A DELIBERATELY WRONG geometry directly on the ObjNode -- a handler
        # that (incorrectly) calls node.geometry() instead of
        # displayNode().geometry() would pick this up instead.
        wrong_geo = MagicMock(name="fake_wrong_geo")
        wrong_geo.boundingBox.return_value = _FakeBoundingBox((100.0, 100.0, 100.0), (102.0, 102.0, 102.0))
        obj_node.geometry.return_value = wrong_geo

        mock_hou.node.side_effect = _node_lookup({"/obj/table1": obj_node})

        objects_wire = [
            {"id": "anchor", "bbox": [0.1, 0.1, 0.1], "fixed": False, "node": ""},
            {"id": "table", "bbox": None, "fixed": False, "node": "/obj/table1"},
        ]
        transforms_wire = {"anchor": {"t": [2.0, 1.0, 3.0], "r": [0.0, 0.0, 0.0]}}

        result = _dispatch("assert_scene", {
            "objects": objects_wire,
            "transforms": transforms_wire,
            "relations": None,
            "checks": ["collision"],
        })

        assert result["status"] == "success", f"expected a normal result, got {result!r}"
        data = result["data"]
        assert data["collision"]["count"] == 1, (
            "anchor at (2,1,3) must collide with 'table' ONLY if the handler "
            "resolved table's geometry via displayNode().geometry() (the "
            "CORRECT small box near the origin) rather than node.geometry() "
            "directly (the WRONG box near 100,100,100, which would show NO "
            f"collision with an anchor at 2,1,3). Got {data['collision']!r}."
        )
        assert data["collision"]["pairs"] == [["anchor", "table"]]


# ---------------------------------------------------------------------------
# missing/unresolvable node -> {"ok": False, "error": ...} WITHOUT raising
# ---------------------------------------------------------------------------

class TestAssertSceneMissingNodeDegradesGracefully:
    """A needed hou read against an unresolvable node degrades to
    {"ok": False, "error": "<reason>"} -- it must NOT raise (Blocker-4
    scene-resolution try/except boundary). Covers all three hard-error
    branches named in the locked contract."""

    def test_unresolvable_node_path_is_ok_false(self, mock_hou):
        handlers = _import_handler_module()
        mock_hou.node.side_effect = _node_lookup({})

        objects_wire = [{"id": "ghost", "bbox": None, "fixed": False, "node": "/obj/does_not_exist"}]
        result = _dispatch("assert_scene", {
            "objects": objects_wire, "transforms": None, "relations": None, "checks": ["collision"],
        })

        assert result["status"] == "success", (
            f"a scene-resolution failure must be a NORMAL return, never a "
            f"dispatcher-level exception. Got {result!r}."
        )
        data = result["data"]
        assert data.get("ok") is False, f"expected ok:false for an unresolvable node, got {data!r}"
        assert "error" in data and isinstance(data["error"], str) and data["error"], (
            f"expected a non-empty error string, got {data!r}"
        )
        assert "ghost" in data["error"] or "/obj/does_not_exist" in data["error"], (
            f"the error must name the offending id/path, got error={data['error']!r}"
        )

    def test_obj_node_with_no_display_node_is_ok_false(self, mock_hou):
        handlers = _import_handler_module()
        obj_node = mock_hou.ObjNode(name="fake_empty_obj")
        obj_node.path.return_value = "/obj/empty_subnet"
        obj_node.displayNode.return_value = None
        mock_hou.node.side_effect = _node_lookup({"/obj/empty_subnet": obj_node})

        objects_wire = [{"id": "empty", "bbox": None, "fixed": False, "node": "/obj/empty_subnet"}]
        result = _dispatch("assert_scene", {
            "objects": objects_wire, "transforms": None, "relations": None, "checks": ["collision"],
        })

        assert result["status"] == "success", f"expected a normal result, got {result!r}"
        data = result["data"]
        assert data.get("ok") is False, (
            f"expected ok:false when an ObjNode's displayNode() is None, got {data!r}"
        )
        assert "error" in data and data["error"]

    def test_non_sop_non_obj_node_is_ok_false(self, mock_hou):
        handlers = _import_handler_module()

        other_node = _FakeOtherNode(name="fake_lop_node")
        other_node.path.return_value = "/stage/lop1"
        mock_hou.node.side_effect = _node_lookup({"/stage/lop1": other_node})

        objects_wire = [{"id": "weird", "bbox": None, "fixed": False, "node": "/stage/lop1"}]
        result = _dispatch("assert_scene", {
            "objects": objects_wire, "transforms": None, "relations": None, "checks": ["collision"],
        })

        assert result["status"] == "success", f"expected a normal result, got {result!r}"
        data = result["data"]
        assert data.get("ok") is False, (
            f"a node that is neither hou.SopNode nor hou.ObjNode must degrade to "
            f"ok:false (the 'else -> scene-resolution error' branch), got {data!r}"
        )
        assert "error" in data and data["error"]


# ---------------------------------------------------------------------------
# bad relation type -> ValueError PROPAGATES (never folded into {ok:false})
# ---------------------------------------------------------------------------

class TestAssertSceneBadRelationPropagates:
    """A caller-contract error (an unknown relation type) is NOT a scene-
    resolution error -- it PROPAGATES as a raised ValueError, which the
    DISPATCHER (not the handler) turns into its standard error envelope. It
    must NEVER be folded into {"ok": False} (Blocker-4's try/except
    boundary explicitly does not wrap the model-call phase)."""

    _VALID_OBJECTS = [
        {"id": "box_a", "bbox": [2.0, 2.0, 2.0], "fixed": False, "node": ""},
        {"id": "box_b", "bbox": [2.0, 2.0, 2.0], "fixed": False, "node": ""},
    ]
    _VALID_TRANSFORMS = {
        "box_a": {"t": [0.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
        "box_b": {"t": [5.0, 0.0, 0.0], "r": [0.0, 0.0, 0.0]},
    }

    def test_direct_handler_call_raises_value_error(self, mock_hou):
        handlers = _import_handler_module()
        bad_relations = [{"type": "bogus_relation_type", "a": "box_a", "b": "box_b"}]

        with pytest.raises(ValueError):
            handlers.assert_scene(
                objects=self._VALID_OBJECTS,
                transforms=self._VALID_TRANSFORMS,
                relations=bad_relations,
                checks=["collision"],
            )

    def test_dispatch_propagates_as_error_envelope_not_ok_false(self, mock_hou):
        handlers = _import_handler_module()
        bad_relations = [{"type": "bogus_relation_type", "a": "box_a", "b": "box_b"}]

        result = _dispatch("assert_scene", {
            "objects": self._VALID_OBJECTS,
            "transforms": self._VALID_TRANSFORMS,
            "relations": bad_relations,
            "checks": ["collision"],
        })

        assert result["status"] == "error", (
            f"a caller-contract ValueError (bad relation type) must propagate "
            f"as the DISPATCHER's error envelope (status:error), NOT be folded "
            f"into a normal ok:false success return. Got {result!r}."
        )
        assert result["error"]["code"] == "ValueError", (
            f"expected error.code == 'ValueError', got {result['error']!r}"
        )
