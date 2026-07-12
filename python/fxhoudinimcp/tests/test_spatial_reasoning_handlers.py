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


# =============================================================================
# PR-3 (pp12-116c) — solve_layout: RED tests for the MUTATING, 109-GATED
# relation-solve + apply tool. Extends this file's EXISTING fixtures
# (mock_hou, _FakeObjNode, _FakeSopNode, _FakeMatrix4, _FakeVector3,
# _FakeBoundingBox, _node_lookup, _import_handler_module, _dispatch) --
# all defined ABOVE, UNMODIFIED.
#
# Unit: pp12-116c
# testVerificationSurface: pytest-model
# planSha: 8f36b134044347a08c2025aafbc41c8140f98a9f8871b6e663e07e823203b366
#
# Coverage (plan pp12-116c lockedFieldContract REVISION 2):
#   (i)    apply=false -> the model's result is returned verbatim, and ZERO
#          parmTuple.set calls occur (fail-safe: solve_layout is GATED
#          regardless of apply).
#   (ii)   THE LOAD-BEARING APPLY-ORDER PIN (B1/B2/B3): apply=true on an
#          OFF-CENTER-geometry, identity-parent, ROTATING (oriented_toward)
#          object -- the apply MUST rotate FIRST, RE-READ the
#          POST-ROTATION world-AABB center, THEN translate by
#          cur_objt + (solved_t - post_rotation_center). A translate-
#          before-rotate impl AND a naive "OBJ.t = solved_t" impl BOTH
#          write a DIFFERENT (wrong) parmTuple('t').set value than the one
#          this test asserts -- pinned via a shared per-node call recorder
#          that captures BOTH the call ORDER and the exact VALUE written.
#   (iii)  a movable OBJ with a NON-IDENTITY parentAndSubnetTransform() ->
#          _preview_solve_layout RAISES (gate DENY); the SAME scenario
#          dispatched with apply=true degrades to {ok:false, error:...}
#          (a scene-resolution failure, never a raised exception through
#          the dispatcher).
#   (iv)   two movable objects resolving to the SAME OBJ ancestor (two SOPs
#          under one geo), and a movable object sharing a FIXED object's
#          OBJ -- BOTH scenarios DENY at _preview_solve_layout.
#   (v)    a FIXED object's parmTuple is NEVER .set(), even when a sibling
#          movable object in the SAME apply IS moved.
#   (vi)   _preview_solve_layout is CHEAP: it returns a no-mutation preview
#          dict ({would_move, apply, note} per the locked contract) and
#          does NOT invoke spatial_reasoning_model.solve_layout.
#   (vii)  taxonomy: a caller-contract ValueError (bad relation type)
#          PROPAGATES as the dispatcher's error envelope; an unsatisfiable
#          (over-constrained, both-fixed) spec degrades to solved:false +
#          a populated unsatisfied[] WITHOUT raising.
#   (viii) atomicity: the whole multi-object apply runs inside exactly ONE
#          hou.undos.group(...) context (entered once for the WHOLE apply,
#          never per-object); a mid-apply parm-write exception surfaces as
#          an OVERALL {ok:false} failure, never a partial "success".
#
# SCOPE NOTE: the REAL 109 gate queue -> operator-approve -> live-move loop
# is the MANDATORY orchestrator-run live-MCP subprocess rung named in the
# plan's lockedFieldContract "verification" clause -- OUT OF SCOPE here.
# This file pins the pure handler-level contract (mock-hou, off-DCC,
# pytest-model); it does NOT assert the real 109 gate's queue/approve
# machinery, which cannot be exercised off-DCC.
# =============================================================================

import math as _math


class _FakeParmTuple:
    """A minimal hou.ParmTuple stand-in: .eval() / .set(value), recording
    every .set() call (name, value-tuple) into a SHARED per-node recorder
    list IN CALL ORDER. This is what lets the tests below pin BOTH the
    call ORDER (rotate before translate) and the exact VALUE written in a
    single assertion, without inspecting any internal call graph beyond
    the public parmTuple('t'/'r').set() surface the locked contract
    names."""

    def __init__(self, name, initial, recorder):
        self._name = name
        self._value = tuple(float(v) for v in initial)
        self._recorder = recorder

    def eval(self):
        return tuple(self._value)

    def set(self, value):
        self._value = tuple(float(v) for v in value)
        self._recorder.append((self._name, tuple(self._value)))


def _make_parm_tuple_lookup(t0, r0, recorder):
    """Build a node.parmTuple(name) side_effect dict-lookup closure,
    sharing ONE recorder list across both the 't' and 'r' FakeParmTuple
    instances (so .set() calls on EITHER parm append to the SAME ordered
    list). Returns (side_effect_fn, {"t": ..., "r": ...})."""
    parms = {
        "t": _FakeParmTuple("t", t0, recorder),
        "r": _FakeParmTuple("r", r0, recorder),
    }

    def _side_effect(name):
        return parms[name]

    return _side_effect, parms


class _FakeRotatingMatrix4:
    """A row-major 4x4 affine matrix supporting a REAL Y-axis rotation (not
    just translate) -- needed so a re-read of _world_aabb_center_and_extents
    AFTER a parmTuple('r').set() call observably changes the computed
    world-AABB center for an off-center local geometry (the ORDER pin,
    (ii) above). Uses the SAME .at(row, col) row-vector convention already
    proven against spatial_reasoning_handlers._transform_point (see the
    shared _FakeMatrix4 above): row i<3 is the linear (rotation) part, row
    3 is translate; result[j] = sum_i v[i] * M[i][j].

    Only a Y-axis rotation is modeled (sufficient for these tests -- the
    single 'oriented_toward' scenario below only ever touches ry). This is
    an INTERNALLY CONSISTENT, fully self-contained rotation transform used
    ONLY to compute a mocked node's world-AABB center from its local
    geometry -- it need not (and does not claim to) match Houdini's own
    internal rotation-matrix bit layout; it only needs to agree with
    itself across the 'before r.set' and 'after r.set' reads, which it
    does by reading the LIVE parm state on every call (see
    _make_rotating_world_transform_side_effect below).
    """

    def __init__(self, translate, ry_deg):
        theta = _math.radians(ry_deg)
        c, s = _math.cos(theta), _math.sin(theta)
        self._rows = [
            [c, 0.0, -s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [s, 0.0, c, 0.0],
            [float(translate[0]), float(translate[1]), float(translate[2]), 1.0],
        ]
        self._ry_deg = float(ry_deg)

    def at(self, r, c):
        return self._rows[r][c]

    def extractTranslates(self):
        return _FakeVector3(self._rows[3][0], self._rows[3][1], self._rows[3][2])

    def extractRotates(self):
        return _FakeVector3(0.0, self._ry_deg, 0.0)


def _make_rotating_world_transform_side_effect(t_parm, r_parm):
    """node.worldTransform side_effect reading the CURRENT live t/r
    FakeParmTuple state on EVERY call -- so calling it again AFTER
    parmTuple('r').set(...) has been invoked reflects the NEW rotation.
    Valid for an IDENTITY-PARENT OBJ (world transform == the node's own
    parm transform, since parentAndSubnetTransform() is identity --
    exactly the B1 scoping this apply-mapping relies on)."""

    def _side_effect():
        tx, ty, tz = t_parm.eval()
        _rx, ry, _rz = r_parm.eval()
        return _FakeRotatingMatrix4(translate=(tx, ty, tz), ry_deg=ry)

    return _side_effect


class _FakeComparableMatrix4:
    """A 4x4 matrix stand-in exposing BOTH the canonical Houdini
    identity-check idiom (``m == hou.Matrix4(1)`` -- hou.Matrix4(1)
    constructs the 4x4 IDENTITY matrix per the HOM reference; hou.Matrix4()
    with no args is the ZERO matrix) AND raw .at(row, col) element access
    (so a manual per-component diagonal/off-diagonal identity check also
    works), plus extractTranslates()/extractRotates() (so a decomposed
    translate/rotate identity check also works) -- covering the reasonable
    implementation space for a parentAndSubnetTransform() identity test
    without over-constraining hou-dev to one specific check mechanism.
    mock_hou.Matrix4 is wired to _fake_matrix4_ctor (see the node-factory
    helpers below) so that ``hou.Matrix4(1) == this`` is a REAL, working
    comparison rather than a MagicMock-vs-real-object mismatch."""

    def __init__(self, rows, translate=None, rotate_deg=None):
        self._rows = [list(r) for r in rows]
        self._translate = (
            tuple(translate) if translate is not None
            else (self._rows[3][0], self._rows[3][1], self._rows[3][2])
        )
        self._rotate_deg = tuple(rotate_deg) if rotate_deg is not None else (0.0, 0.0, 0.0)

    def at(self, r, c):
        return self._rows[r][c]

    def extractTranslates(self):
        return _FakeVector3(*self._translate)

    def extractRotates(self):
        return _FakeVector3(*self._rotate_deg)

    def __eq__(self, other):
        if not hasattr(other, "at"):
            return NotImplemented
        try:
            return all(
                abs(self.at(r, c) - other.at(r, c)) < 1e-9
                for r in range(4) for c in range(4)
            )
        except TypeError:
            return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result

    def __hash__(self):
        return id(self)


def _identity_matrix4():
    """An IDENTITY parentAndSubnetTransform() -- the standard /obj-level
    case where an OBJ node's LOCAL t/r parms equal its WORLD t/r."""
    rows = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    return _FakeComparableMatrix4(rows, translate=(0.0, 0.0, 0.0), rotate_deg=(0.0, 0.0, 0.0))


def _non_identity_matrix4():
    """A NON-identity parentAndSubnetTransform() -- e.g. the OBJ sits
    under a parent subnet with its own non-zero placement, so the node's
    LOCAL t/r parms are NOT the same as its WORLD t/r (the exact B1 hazard
    the identity-parent scoping rule exists to avoid)."""
    rows = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    rows[3][0] = 25.0  # a non-zero parent-subnet X offset
    return _FakeComparableMatrix4(rows, translate=(25.0, 0.0, 0.0), rotate_deg=(0.0, 0.0, 0.0))


def _fake_matrix4_ctor(scale=None):
    """Stand-in for hou.Matrix4(...): hou.Matrix4() (no args) is the ZERO
    matrix; hou.Matrix4(1) is the IDENTITY matrix -- the canonical Houdini
    HOM idiom for constructing/comparing-against an identity transform.
    Wired to mock_hou.Matrix4 by the node-factory helpers below so that,
    IF the (not-yet-written) implementation checks identity via
    ``node.parentAndSubnetTransform() == hou.Matrix4(1)``, the comparison
    is a real, working one rather than a MagicMock-vs-real-object
    mismatch."""
    if scale is None:
        rows = [[0.0] * 4 for _ in range(4)]
        return _FakeComparableMatrix4(rows, translate=(0.0, 0.0, 0.0), rotate_deg=(0.0, 0.0, 0.0))
    rows = [[float(scale) if i == j else 0.0 for j in range(4)] for i in range(4)]
    return _FakeComparableMatrix4(rows, translate=(0.0, 0.0, 0.0), rotate_deg=(0.0, 0.0, 0.0))


def _make_static_objnode(mock_hou, path, local_min, local_max, translate,
                          rotate_deg=(0.0, 0.0, 0.0), parent_transform=None):
    """A non-rotating-apply ObjNode fixture (used for FIXED / anchor
    objects that never move, and for simple caller-supplied-translate-only
    scenarios): identity displayNode/geometry + a STATIC worldTransform
    (the shared _FakeMatrix4 above -- translate/rotate fixed at
    construction, matching the existing PR-2 fixtures). Also wires
    parmTuple('t'/'r') (recording into its OWN `._test_recorder` list) and
    parentAndSubnetTransform() (default: identity) so it satisfies the
    identity-parent / _resolve_obj_node contract even when unused (a FIXED
    object is still resolved for its initial t/r, just never applied)."""
    mock_hou.Matrix4 = _fake_matrix4_ctor
    safe_name = path.replace("/", "_")
    node = mock_hou.ObjNode(name=f"fake_{safe_name}")
    node.path.return_value = path
    node.worldTransform.return_value = _FakeMatrix4(translate=translate, rotate_deg=rotate_deg)
    node.parentAndSubnetTransform.return_value = (
        parent_transform if parent_transform is not None else _identity_matrix4()
    )

    display_sop = MagicMock(name=f"fake_display_{safe_name}")
    geo = MagicMock(name=f"fake_geo_{safe_name}")
    geo.boundingBox.return_value = _FakeBoundingBox(local_min, local_max)
    display_sop.geometry.return_value = geo
    node.displayNode.return_value = display_sop

    recorder = []
    parm_side_effect, _parms = _make_parm_tuple_lookup(translate, rotate_deg, recorder)
    node.parmTuple.side_effect = parm_side_effect
    node._test_recorder = recorder
    return node


def _make_rotating_objnode(mock_hou, path, local_min, local_max, t0,
                            r0=(0.0, 0.0, 0.0), parent_transform=None):
    """A movable ObjNode fixture whose worldTransform() DYNAMICALLY tracks
    the current parmTuple('t'/'r') state (see
    _make_rotating_world_transform_side_effect) -- required for the ORDER
    pin (ii), where the world-AABB center must observably CHANGE after
    parmTuple('r').set() is called and BEFORE parmTuple('t').set() is
    called."""
    mock_hou.Matrix4 = _fake_matrix4_ctor
    safe_name = path.replace("/", "_")
    node = mock_hou.ObjNode(name=f"fake_{safe_name}")
    node.path.return_value = path
    node.parentAndSubnetTransform.return_value = (
        parent_transform if parent_transform is not None else _identity_matrix4()
    )

    display_sop = MagicMock(name=f"fake_display_{safe_name}")
    geo = MagicMock(name=f"fake_geo_{safe_name}")
    geo.boundingBox.return_value = _FakeBoundingBox(local_min, local_max)
    display_sop.geometry.return_value = geo
    node.displayNode.return_value = display_sop

    recorder = []
    parm_side_effect, parms = _make_parm_tuple_lookup(t0, r0, recorder)
    node.parmTuple.side_effect = parm_side_effect
    node.worldTransform.side_effect = _make_rotating_world_transform_side_effect(
        parms["t"], parms["r"]
    )
    node._test_recorder = recorder
    return node


def _obj_wire(oid, node_path, fixed=False):
    """A minimal solve_layout wire object: {id, node, fixed} -- bbox is
    omitted (resolved from the live scene, per the locked contract's
    'bbox = caller's if given else the hou world-AABB extents'); solve_
    layout's wire schema has NO caller-transform override (unlike
    assert_scene's separate `transforms` param) -- t/r are ALWAYS resolved
    via the node, so `node` is REQUIRED for every object."""
    return {"id": oid, "node": node_path, "fixed": fixed}


# ---------------------------------------------------------------------------
# PRIMARY RED GATE — solve_layout / _preview_solve_layout / _resolve_obj_node
# ---------------------------------------------------------------------------

class TestSolveLayoutHandlerImport:
    def test_solve_layout_handler_importable(self, mock_hou):
        handlers = _import_handler_module()
        assert hasattr(handlers, "solve_layout"), (
            "spatial_reasoning_handlers.py must expose solve_layout (PP12-116 PR-3)."
        )
        assert callable(handlers.solve_layout)

    def test_preview_solve_layout_importable(self, mock_hou):
        handlers = _import_handler_module()
        assert hasattr(handlers, "_preview_solve_layout"), (
            "spatial_reasoning_handlers.py must expose _preview_solve_layout "
            "(the positional gate preview_fn, PP12-116 PR-3)."
        )
        assert callable(handlers._preview_solve_layout)

    def test_resolve_obj_node_importable(self, mock_hou):
        handlers = _import_handler_module()
        assert hasattr(handlers, "_resolve_obj_node"), (
            "spatial_reasoning_handlers.py must expose _resolve_obj_node "
            "(the new ObjNode-resolution helper, PP12-116 PR-3)."
        )
        assert callable(handlers._resolve_obj_node)


# ---------------------------------------------------------------------------
# Capability.MUTATING + preview registration
# ---------------------------------------------------------------------------

class TestSolveLayoutCapabilityAndPreviewRegistration:
    def test_solve_layout_capability_mutating(self, mock_hou):
        _import_handler_module()
        from fxhoudinimcp_server import dispatcher
        assert dispatcher.capability_of("solve_layout") == dispatcher.Capability.MUTATING, (
            f"solve_layout must be registered Capability.MUTATING (GATED), "
            f"got {dispatcher.capability_of('solve_layout')!r}."
        )

    def test_solve_layout_preview_registered_and_required(self, mock_hou):
        # NOTE: dispatcher.preview_of() reads hou.session._fxhoudinimcp_preview_registry,
        # which auto-vivifies into a MagicMock attribute under mock_hou (hou is a
        # MagicMock, not None) — preview_of() is mock-incompatible (it only has a
        # `hou is None` off-DCC fallback, not a `hou is MagicMock` one). Assert via
        # the mock-safe _preview_registry_fallback store instead, which
        # register_handler() writes unconditionally and which _preview_registry()
        # itself falls back to when `hou` is genuinely absent — the same mock-safe
        # pattern the sibling test_solve_layout_capability_mutating test uses via
        # capability_of().
        handlers = _import_handler_module()
        from fxhoudinimcp_server import dispatcher
        record = dispatcher._preview_registry_fallback["solve_layout"]
        assert record["preview_fn"] is handlers._preview_solve_layout, (
            f"solve_layout must register preview_fn=_preview_solve_layout, got {record!r}."
        )
        assert record["preview_required"] is True, (
            f"solve_layout must register preview_required=True (fail-safe DENY-on-raise), "
            f"got {record!r}."
        )


# ---------------------------------------------------------------------------
# _resolve_obj_node — the new ObjNode->self / SopNode->OBJ-ancestor helper
# ---------------------------------------------------------------------------

class TestResolveObjNodeHelper:
    """_resolve_obj_node: hou.ObjNode -> itself; hou.SopNode -> its OBJ
    ancestor (walking node.parent()); raises hou.OperationFailed when no
    OBJ ancestor is found. Does NOT modify the byte-frozen
    _world_aabb_center_and_extents (which returns (t, r, bbox), never the
    node)."""

    def test_objnode_resolves_to_itself(self, mock_hou):
        handlers = _import_handler_module()
        obj_node = mock_hou.ObjNode(name="fake_obj")
        obj_node.path.return_value = "/obj/thing"
        assert handlers._resolve_obj_node(obj_node) is obj_node

    def test_sopnode_resolves_to_its_obj_ancestor(self, mock_hou):
        handlers = _import_handler_module()
        obj_ancestor = mock_hou.ObjNode(name="fake_obj_ancestor")
        obj_ancestor.path.return_value = "/obj/geo1"
        sop_node = mock_hou.SopNode(name="fake_sop")
        sop_node.path.return_value = "/obj/geo1/box"
        sop_node.parent.return_value = obj_ancestor

        assert handlers._resolve_obj_node(sop_node) is obj_ancestor

    def test_sopnode_with_no_obj_ancestor_raises(self, mock_hou):
        handlers = _import_handler_module()
        sop_node = mock_hou.SopNode(name="fake_sop_orphan")
        sop_node.path.return_value = "/orphan/box"
        sop_node.parent.return_value = None

        with pytest.raises(mock_hou.OperationFailed):
            handlers._resolve_obj_node(sop_node)


# ---------------------------------------------------------------------------
# (i) apply=false -> verbatim model result, ZERO mutation
# ---------------------------------------------------------------------------

class TestSolveLayoutApplyFalseNoMutation:
    """apply=false must still resolve + solve (so the caller gets the
    proposed transforms), but must NEVER call parmTuple(...).set on any
    node -- solve_layout is GATED regardless of apply (fail-safe; a
    separate ungated dry-run tool is out-of-scope, M3)."""

    def test_apply_false_returns_model_result_and_sets_nothing(self, mock_hou):
        handlers = _import_handler_module()
        from fxhoudinimcp import spatial_reasoning_model as model

        box_a = _make_static_objnode(
            mock_hou, "/obj/box_a", (0.0, 0.0, 0.0), (2.0, 2.0, 2.0), translate=(0.0, 0.0, 0.0)
        )
        box_b = _make_static_objnode(
            mock_hou, "/obj/box_b", (0.0, 0.0, 0.0), (2.0, 2.0, 2.0), translate=(10.0, 0.0, 0.0)
        )
        mock_hou.node.side_effect = _node_lookup({"/obj/box_a": box_a, "/obj/box_b": box_b})

        objects_wire = [_obj_wire("box_a", "/obj/box_a"), _obj_wire("box_b", "/obj/box_b")]

        result = _dispatch("solve_layout", {
            "objects": objects_wire, "relations": [], "bounds": None,
            "apply": False, "max_iters": 200,
        })

        assert result["status"] == "success", f"solve_layout must not raise, got {result!r}"
        data = result["data"]

        # Oracle: the SAME resolved specs (world-AABB centers (1,1,1) /
        # (11,1,1) for local bbox (0,0,0)-(2,2,2) at translate (0,0,0) /
        # (10,0,0)), called directly against the byte-frozen model. No
        # relations -> transforms == the objects' own resolved t/r,
        # unchanged.
        box_a_spec = model._object_from_dict({
            "id": "box_a", "bbox": (2.0, 2.0, 2.0), "fixed": False,
            "t": (1.0, 1.0, 1.0), "r": (0.0, 0.0, 0.0),
        })
        box_b_spec = model._object_from_dict({
            "id": "box_b", "bbox": (2.0, 2.0, 2.0), "fixed": False,
            "t": (11.0, 1.0, 1.0), "r": (0.0, 0.0, 0.0),
        })
        expected = model.solve_layout(
            model.LayoutSpec(objects=[box_a_spec, box_b_spec], relations=[], bounds=None, max_iters=200)
        )
        assert data == expected, f"Expected {expected!r}, got {data!r}"

        assert box_a._test_recorder == [], (
            f"apply=false must set NOTHING on box_a, got {box_a._test_recorder!r}"
        )
        assert box_b._test_recorder == [], (
            f"apply=false must set NOTHING on box_b, got {box_b._test_recorder!r}"
        )


# ---------------------------------------------------------------------------
# (ii) THE LOAD-BEARING APPLY-ORDER PIN (B1/B2/B3)
# ---------------------------------------------------------------------------

class TestSolveLayoutApplyOrderPin:
    """THE LOAD-BEARING PIN (B1/B2/B3): apply=true must ROTATE FIRST,
    RE-READ the POST-ROTATION world-AABB center, THEN translate -- verified
    via an off-center-geometry, identity-parent, rotating (oriented_toward)
    object, so a translate-before-rotate OR a naive "OBJ.t = solved_t"
    implementation both write a DIFFERENT (wrong) parmTuple('t').set value
    than the one this test asserts.

    Scenario (hand-derived; the math is affine-exact, not approximate --
    see the module comment above and _FakeRotatingMatrix4's docstring):
    box_off's local bbox is min=(0,0,0), max=(4,2,2) -- OFF-CENTER, local
    center (2,1,1). Its OBJ starts at parm t=(98,4,49) [world center
    (100,5,50), since the initial rotation is 0], parm r=(0,0,0). A single
    `oriented_toward` relation toward a static target at world (200,5,50)
    makes the model solve box_off's ry to 90deg WITHOUT moving its
    position (oriented_toward never touches position) -- solved
    t=(100,5,50), solved r=(0,90,0).

    The CORRECT apply:
      1. r.set((0.0, 90.0, 0.0))
      2. re-read the world-AABB center via _world_aabb_center_and_extents
         -- box_off's LOCAL center (2,1,1) rotated 90deg about Y is
         (1,1,-2) (an affine map preserves a symmetric point-set's AABB
         centroid regardless of rotation angle); + the UNCHANGED translate
         (98,4,49) = POST-ROTATION center (99,5,47).
      3. t.set(cur_objt + (solved_t - post_rotation_center))
         = (98,4,49) + ((100,5,50) - (99,5,47)) = (99.0, 4.0, 52.0)

    A translate-BEFORE-rotate implementation would instead compute the
    delta from the PRE-rotation center -- which in this scenario equals
    solved_t EXACTLY, (100,5,50), since box_off's position never changes
    -- giving a delta of (0,0,0) and writing t=(98.0,4.0,49.0) [literally
    unchanged] BEFORE rotating. A naive "OBJ.t = solved_t" implementation
    would instead write t=(100.0,5.0,50.0) directly, ignoring the
    rotation-induced pivot offset entirely. Both wrong values differ from
    the correct (99.0, 4.0, 52.0) -- and the correct sequence is ALSO
    order-pinned (r before t) via the shared per-node call recorder.
    """

    def test_rotate_then_reread_then_translate(self, mock_hou):
        handlers = _import_handler_module()

        box_off = _make_rotating_objnode(
            mock_hou, "/obj/box_off",
            local_min=(0.0, 0.0, 0.0), local_max=(4.0, 2.0, 2.0),
            t0=(98.0, 4.0, 49.0), r0=(0.0, 0.0, 0.0),
        )
        target = _make_static_objnode(
            mock_hou, "/obj/target_marker",
            local_min=(-0.05, -0.05, -0.05), local_max=(0.05, 0.05, 0.05),
            translate=(200.0, 5.0, 50.0),
        )
        mock_hou.node.side_effect = _node_lookup({
            "/obj/box_off": box_off, "/obj/target_marker": target,
        })

        objects_wire = [
            _obj_wire("box_off", "/obj/box_off", fixed=False),
            _obj_wire("target_marker", "/obj/target_marker", fixed=True),
        ]
        relations_wire = [
            {"type": "oriented_toward", "a": "box_off", "b": "", "target": "target_marker", "params": {}},
        ]

        result = _dispatch("solve_layout", {
            "objects": objects_wire, "relations": relations_wire, "bounds": None,
            "apply": True, "max_iters": 200,
        })

        assert result["status"] == "success", f"solve_layout must not raise, got {result!r}"
        data = result["data"]
        assert data["solved"] is True, f"expected solved:true, got {data!r}"

        solved_t = data["transforms"]["box_off"]["t"]
        solved_r = data["transforms"]["box_off"]["r"]
        assert solved_r[1] == pytest.approx(90.0, abs=1e-3), (
            f"expected the model to solve box_off's ry to 90deg (oriented_toward "
            f"target_marker at world (200,5,50)), got r={solved_r!r}"
        )
        assert solved_t[0] == pytest.approx(100.0, abs=1e-6), (
            f"expected box_off's position UNCHANGED by oriented_toward, got t={solved_t!r}"
        )

        recorder = box_off._test_recorder
        assert len(recorder) == 2, (
            f"expected exactly 2 parmTuple.set calls on box_off (one 'r', one 't'), "
            f"got {recorder!r}"
        )
        names = [name for name, _ in recorder]
        assert names == ["r", "t"], (
            f"apply MUST rotate BEFORE translating (rotate-then-reread-then-translate, "
            f"B2) -- expected call order ['r', 't'], got {names!r}. A translate-before-"
            f"rotate implementation would produce ['t', 'r'] here."
        )

        r_name, r_value = recorder[0]
        assert r_value == pytest.approx((0.0, 90.0, 0.0), abs=1e-3), (
            f"expected parmTuple('r').set((0.0, 90.0, 0.0)), got {r_value!r}"
        )

        t_name, t_value = recorder[1]
        assert t_value == pytest.approx((99.0, 4.0, 52.0), abs=1e-3), (
            f"expected parmTuple('t').set((99.0, 4.0, 52.0)) -- computed as "
            f"cur_objt(98,4,49) + (solved_t(100,5,50) - POST-ROTATION center(99,5,47)). "
            f"A naive 'OBJ.t = solved_t' impl would instead write (100.0, 5.0, 50.0); "
            f"a translate-before-rotate impl would write (98.0, 4.0, 49.0) [literally "
            f"unchanged, since solved_t happens to equal the PRE-rotation center in "
            f"this scenario]. Got {t_value!r}."
        )

        assert target._test_recorder == [], (
            f"target_marker is fixed=True and must NEVER have parmTuple.set called, "
            f"got {target._test_recorder!r}"
        )


# ---------------------------------------------------------------------------
# (iii) NON-IDENTITY parent -> DENY (preview raises; apply degrades ok:false)
# ---------------------------------------------------------------------------

class TestSolveLayoutNonIdentityParentDenied:
    def test_preview_raises_for_non_identity_parent(self, mock_hou):
        handlers = _import_handler_module()

        movable = _make_static_objnode(
            mock_hou, "/obj/tilted_box", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
            translate=(0.0, 0.0, 0.0), parent_transform=_non_identity_matrix4(),
        )
        mock_hou.node.side_effect = _node_lookup({"/obj/tilted_box": movable})

        objects_wire = [_obj_wire("tilted_box", "/obj/tilted_box", fixed=False)]

        with pytest.raises(mock_hou.OperationFailed):
            handlers._preview_solve_layout({
                "objects": objects_wire, "relations": [], "bounds": None,
                "apply": True, "max_iters": 200,
            })

    def test_apply_degrades_to_ok_false_for_non_identity_parent(self, mock_hou):
        handlers = _import_handler_module()

        movable = _make_static_objnode(
            mock_hou, "/obj/tilted_box", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
            translate=(0.0, 0.0, 0.0), parent_transform=_non_identity_matrix4(),
        )
        mock_hou.node.side_effect = _node_lookup({"/obj/tilted_box": movable})

        objects_wire = [_obj_wire("tilted_box", "/obj/tilted_box", fixed=False)]

        result = _dispatch("solve_layout", {
            "objects": objects_wire, "relations": [], "bounds": None,
            "apply": True, "max_iters": 200,
        })

        assert result["status"] == "success", (
            f"a scene-resolution failure must be a NORMAL dispatcher return, "
            f"never an exception. Got {result!r}"
        )
        data = result["data"]
        assert data.get("ok") is False, (
            f"expected ok:false when a movable object's OBJ has a NON-IDENTITY "
            f"parentAndSubnetTransform() (B1 -- LOCAL t/r parms would silently "
            f"diverge from WORLD t/r), got {data!r}"
        )
        assert "error" in data and data["error"], f"expected a non-empty error string, got {data!r}"
        assert movable._test_recorder == [], (
            f"a denied apply must NEVER write any parm, got {movable._test_recorder!r}"
        )


# ---------------------------------------------------------------------------
# (iv) shared OBJ ancestor -> DENY
# ---------------------------------------------------------------------------

class TestSolveLayoutSharedObjAncestorDenied:
    """Two objects that resolve to the SAME mutable OBJ node (their t/r
    parms are the SAME single pair) cannot be independently placed --
    _preview_solve_layout must DENY (raise)."""

    def test_two_movable_sops_sharing_one_obj_ancestor_denied(self, mock_hou):
        handlers = _import_handler_module()

        shared_obj = _make_static_objnode(
            mock_hou, "/obj/geo1", (0.0, 0.0, 0.0), (5.0, 5.0, 5.0), translate=(0.0, 0.0, 0.0)
        )
        sop_a = mock_hou.SopNode(name="fake_sop_a")
        sop_a.path.return_value = "/obj/geo1/part_a"
        geo_a = MagicMock(name="fake_geo_a")
        geo_a.boundingBox.return_value = _FakeBoundingBox((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        sop_a.geometry.return_value = geo_a
        sop_a.parent.return_value = shared_obj

        sop_b = mock_hou.SopNode(name="fake_sop_b")
        sop_b.path.return_value = "/obj/geo1/part_b"
        geo_b = MagicMock(name="fake_geo_b")
        geo_b.boundingBox.return_value = _FakeBoundingBox((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        sop_b.geometry.return_value = geo_b
        sop_b.parent.return_value = shared_obj

        mock_hou.node.side_effect = _node_lookup({
            "/obj/geo1/part_a": sop_a, "/obj/geo1/part_b": sop_b,
        })

        objects_wire = [
            _obj_wire("part_a", "/obj/geo1/part_a", fixed=False),
            _obj_wire("part_b", "/obj/geo1/part_b", fixed=False),
        ]

        with pytest.raises(mock_hou.OperationFailed):
            handlers._preview_solve_layout({
                "objects": objects_wire, "relations": [], "bounds": None,
                "apply": True, "max_iters": 200,
            })

    def test_movable_sharing_fixed_objects_obj_denied(self, mock_hou):
        handlers = _import_handler_module()

        shared_obj = _make_static_objnode(
            mock_hou, "/obj/shared", (0.0, 0.0, 0.0), (3.0, 3.0, 3.0), translate=(0.0, 0.0, 0.0)
        )
        sop_movable = mock_hou.SopNode(name="fake_sop_movable")
        sop_movable.path.return_value = "/obj/shared/child_geo"
        geo_m = MagicMock(name="fake_geo_movable")
        geo_m.boundingBox.return_value = _FakeBoundingBox((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        sop_movable.geometry.return_value = geo_m
        sop_movable.parent.return_value = shared_obj

        mock_hou.node.side_effect = _node_lookup({
            "/obj/shared": shared_obj, "/obj/shared/child_geo": sop_movable,
        })

        objects_wire = [
            _obj_wire("fixed_anchor", "/obj/shared", fixed=True),
            _obj_wire("movable_child", "/obj/shared/child_geo", fixed=False),
        ]

        with pytest.raises(mock_hou.OperationFailed):
            handlers._preview_solve_layout({
                "objects": objects_wire, "relations": [], "bounds": None,
                "apply": True, "max_iters": 200,
            })


# ---------------------------------------------------------------------------
# (v) FIXED objects are NEVER moved, even alongside a moving sibling
# ---------------------------------------------------------------------------

class TestSolveLayoutFixedNeverMoved:
    def test_fixed_object_parms_never_set_while_movable_sibling_moves(self, mock_hou):
        handlers = _import_handler_module()

        fixed_lamp = _make_rotating_objnode(
            mock_hou, "/obj/fixed_lamp",
            local_min=(-0.5, -0.5, -0.5), local_max=(0.5, 0.5, 0.5),
            t0=(0.0, 0.0, 0.0), r0=(0.0, 0.0, 0.0),
        )
        movable_box = _make_rotating_objnode(
            mock_hou, "/obj/movable_box",
            local_min=(-0.5, -0.5, -0.5), local_max=(0.5, 0.5, 0.5),
            t0=(20.0, 0.0, 0.0), r0=(0.0, 0.0, 0.0),
        )
        mock_hou.node.side_effect = _node_lookup({
            "/obj/fixed_lamp": fixed_lamp, "/obj/movable_box": movable_box,
        })

        objects_wire = [
            _obj_wire("fixed_lamp", "/obj/fixed_lamp", fixed=True),
            _obj_wire("movable_box", "/obj/movable_box", fixed=False),
        ]
        relations_wire = [
            {"type": "adjacent", "a": "movable_box", "b": "fixed_lamp",
             "target": "", "params": {"gap": 1.0, "side": "+x"}},
        ]

        result = _dispatch("solve_layout", {
            "objects": objects_wire, "relations": relations_wire, "bounds": None,
            "apply": True, "max_iters": 200,
        })

        assert result["status"] == "success", f"solve_layout must not raise, got {result!r}"
        data = result["data"]
        assert data["solved"] is True, f"expected the adjacent relation to be satisfiable, got {data!r}"

        assert fixed_lamp._test_recorder == [], (
            f"a fixed=True object must NEVER have parmTuple.set called, even when "
            f"a sibling movable object in the SAME apply IS moved. Got "
            f"{fixed_lamp._test_recorder!r}"
        )
        assert movable_box._test_recorder != [], (
            f"expected movable_box to actually move (the adjacent relation requires "
            f"a position change), got {movable_box._test_recorder!r}"
        )


# ---------------------------------------------------------------------------
# (vi) preview is cheap -- no solve, no mutation
# ---------------------------------------------------------------------------

class TestPreviewSolveLayoutIsCheap:
    def test_preview_does_not_invoke_model_solve_and_returns_no_mutation_dict(self, mock_hou, monkeypatch):
        handlers = _import_handler_module()
        from fxhoudinimcp import spatial_reasoning_model as model

        box_a = _make_static_objnode(
            mock_hou, "/obj/box_a", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), translate=(0.0, 0.0, 0.0)
        )
        box_b = _make_static_objnode(
            mock_hou, "/obj/box_b", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), translate=(5.0, 0.0, 0.0)
        )
        mock_hou.node.side_effect = _node_lookup({"/obj/box_a": box_a, "/obj/box_b": box_b})

        solve_calls = []

        def _tracking_solve(spec):
            solve_calls.append(spec)
            return {"transforms": {}, "solved": True, "iterations": 0, "unsatisfied": []}

        monkeypatch.setattr(model, "solve_layout", _tracking_solve)

        objects_wire = [
            _obj_wire("box_a", "/obj/box_a", fixed=False),
            _obj_wire("box_b", "/obj/box_b", fixed=False),
        ]

        preview = handlers._preview_solve_layout({
            "objects": objects_wire, "relations": [], "bounds": None,
            "apply": True, "max_iters": 200,
        })

        assert solve_calls == [], (
            f"_preview_solve_layout must NOT invoke spatial_reasoning_model.solve_layout "
            f"-- it performs ONLY the cheap resolvable/distinct-OBJ/identity-parent "
            f"validation (M4: stay well under the ~30s preview timeout). Got "
            f"{len(solve_calls)} call(s): {solve_calls!r}"
        )
        assert isinstance(preview, dict), f"expected a dict, got {preview!r}"
        assert "would_move" in preview, f"expected a 'would_move' key, got {preview!r}"
        assert set(preview["would_move"]) == {"box_a", "box_b"}, (
            f"expected would_move to list both movable object ids, got {preview!r}"
        )
        assert preview.get("apply") is True, f"expected apply echoed back, got {preview!r}"
        assert "note" in preview, f"expected a 'note' key, got {preview!r}"

        assert box_a._test_recorder == [] and box_b._test_recorder == [], (
            "the preview must not mutate any node."
        )


# ---------------------------------------------------------------------------
# (vii) error taxonomy: ValueError propagates; unsatisfiable degrades
# ---------------------------------------------------------------------------

class TestSolveLayoutErrorTaxonomy:
    def test_bad_relation_type_propagates_as_error_envelope(self, mock_hou):
        handlers = _import_handler_module()

        box_a = _make_static_objnode(
            mock_hou, "/obj/box_a", (0.0, 0.0, 0.0), (2.0, 2.0, 2.0), translate=(0.0, 0.0, 0.0)
        )
        box_b = _make_static_objnode(
            mock_hou, "/obj/box_b", (0.0, 0.0, 0.0), (2.0, 2.0, 2.0), translate=(10.0, 0.0, 0.0)
        )
        mock_hou.node.side_effect = _node_lookup({"/obj/box_a": box_a, "/obj/box_b": box_b})

        objects_wire = [_obj_wire("box_a", "/obj/box_a"), _obj_wire("box_b", "/obj/box_b")]
        bad_relations = [{"type": "bogus_relation_type", "a": "box_a", "b": "box_b", "target": "", "params": {}}]

        result = _dispatch("solve_layout", {
            "objects": objects_wire, "relations": bad_relations, "bounds": None,
            "apply": False, "max_iters": 200,
        })

        assert result["status"] == "error", (
            f"a caller-contract ValueError (bad relation type) must propagate as the "
            f"DISPATCHER's error envelope, NOT be folded into ok:false. Got {result!r}"
        )
        assert result["error"]["code"] == "ValueError", (
            f"expected error.code == 'ValueError', got {result['error']!r}"
        )

    def test_direct_handler_call_raises_value_error(self, mock_hou):
        handlers = _import_handler_module()

        box_a = _make_static_objnode(
            mock_hou, "/obj/box_a", (0.0, 0.0, 0.0), (2.0, 2.0, 2.0), translate=(0.0, 0.0, 0.0)
        )
        box_b = _make_static_objnode(
            mock_hou, "/obj/box_b", (0.0, 0.0, 0.0), (2.0, 2.0, 2.0), translate=(10.0, 0.0, 0.0)
        )
        mock_hou.node.side_effect = _node_lookup({"/obj/box_a": box_a, "/obj/box_b": box_b})

        objects_wire = [_obj_wire("box_a", "/obj/box_a"), _obj_wire("box_b", "/obj/box_b")]
        bad_relations = [{"type": "bogus_relation_type", "a": "box_a", "b": "box_b", "target": "", "params": {}}]

        with pytest.raises(ValueError):
            handlers.solve_layout(
                objects=objects_wire, relations=bad_relations, bounds=None,
                apply=False, max_iters=200,
            )

    def test_unsatisfiable_overconstrained_returns_solved_false_no_raise(self, mock_hou):
        handlers = _import_handler_module()

        box_a = _make_static_objnode(
            mock_hou, "/obj/box_a", (0.0, 0.0, 0.0), (2.0, 2.0, 2.0), translate=(0.0, 0.0, 0.0)
        )
        box_b = _make_static_objnode(
            mock_hou, "/obj/box_b", (0.0, 0.0, 0.0), (2.0, 2.0, 2.0), translate=(0.5, 0.0, 0.5)
        )
        mock_hou.node.side_effect = _node_lookup({"/obj/box_a": box_a, "/obj/box_b": box_b})

        objects_wire = [
            _obj_wire("box_a", "/obj/box_a", fixed=True),
            _obj_wire("box_b", "/obj/box_b", fixed=True),
        ]
        relations_wire = [{"type": "non_overlap", "a": "box_a", "b": "box_b", "target": "", "params": {}}]

        result = _dispatch("solve_layout", {
            "objects": objects_wire, "relations": relations_wire, "bounds": None,
            "apply": False, "max_iters": 200,
        })

        assert result["status"] == "success", f"an unsatisfiable spec must NOT raise, got {result!r}"
        data = result["data"]
        assert data["solved"] is False, f"expected solved:false for two FIXED overlapping boxes, got {data!r}"
        assert len(data["unsatisfied"]) > 0, f"expected a non-empty unsatisfied list, got {data!r}"


# ---------------------------------------------------------------------------
# (viii) atomicity: ONE undo group for the whole apply; mid-apply failure ->
# overall {ok:false}
# ---------------------------------------------------------------------------

class TestSolveLayoutApplyAtomicity:
    """The whole multi-object apply runs inside ONE hou.undos.group(...)
    context -- entered ONCE for the whole apply, not per-object -- and a
    mid-apply parm-set exception surfaces as an OVERALL failure (never a
    partial 'success'). The literal value-rollback Houdini's own undo
    mechanism performs is NOT testable via a mocked hou -- these tests pin
    the STRUCTURAL contract (single undo-group scope + failure
    propagation) that is what MAKES that rollback possible."""

    def test_whole_apply_wrapped_in_a_single_undo_group(self, mock_hou):
        handlers = _import_handler_module()

        box_a = _make_rotating_objnode(
            mock_hou, "/obj/box_a", (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), t0=(0.0, 0.0, 0.0)
        )
        box_b = _make_rotating_objnode(
            mock_hou, "/obj/box_b", (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), t0=(20.0, 0.0, 0.0)
        )
        mock_hou.node.side_effect = _node_lookup({"/obj/box_a": box_a, "/obj/box_b": box_b})

        objects_wire = [
            _obj_wire("box_a", "/obj/box_a", fixed=False),
            _obj_wire("box_b", "/obj/box_b", fixed=False),
        ]
        relations_wire = [
            {"type": "adjacent", "a": "box_b", "b": "box_a", "target": "", "params": {"gap": 1.0, "side": "+x"}},
        ]

        result = _dispatch("solve_layout", {
            "objects": objects_wire, "relations": relations_wire, "bounds": None,
            "apply": True, "max_iters": 200,
        })

        assert result["status"] == "success", f"solve_layout must not raise, got {result!r}"
        assert mock_hou.undos.group.called, (
            "the apply must wrap its parm writes in hou.undos.group(...) -- "
            "no call to hou.undos.group() was recorded at all."
        )
        assert mock_hou.undos.group.call_count == 1, (
            f"expected hou.undos.group(...) to be entered EXACTLY ONCE for the "
            f"whole multi-object apply, got {mock_hou.undos.group.call_count} call(s) "
            f"-- a per-object undo group would let a mid-apply failure leave a "
            f"PARTIALLY-moved scene (the exact atomicity hazard this test guards)."
        )

    def test_mid_apply_exception_surfaces_as_overall_failure(self, mock_hou):
        handlers = _import_handler_module()

        box_a = _make_rotating_objnode(
            mock_hou, "/obj/box_a", (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), t0=(0.0, 0.0, 0.0)
        )
        box_b = _make_rotating_objnode(
            mock_hou, "/obj/box_b", (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), t0=(20.0, 0.0, 0.0)
        )
        mock_hou.node.side_effect = _node_lookup({"/obj/box_a": box_a, "/obj/box_b": box_b})

        # Simulate an unexpected runtime failure writing box_b's parms (a
        # genuine hou.OperationFailed AFTER pre-validation already passed
        # -- e.g. the node was deleted out from under the apply). Both
        # 'r' and 't' are sabotaged so the test does not depend on whether
        # a correct implementation skips a .set() call when the value is
        # unchanged -- box_b's position DEFINITELY changes here (the
        # adjacent relation), so its t.set() WILL be invoked regardless.
        def _boom(value):
            raise mock_hou.OperationFailed("simulated mid-apply failure")

        box_b.parmTuple("r").set = _boom
        box_b.parmTuple("t").set = _boom

        objects_wire = [
            _obj_wire("box_a", "/obj/box_a", fixed=False),
            _obj_wire("box_b", "/obj/box_b", fixed=False),
        ]
        relations_wire = [
            {"type": "adjacent", "a": "box_b", "b": "box_a", "target": "", "params": {"gap": 1.0, "side": "+x"}},
        ]

        result = _dispatch("solve_layout", {
            "objects": objects_wire, "relations": relations_wire, "bounds": None,
            "apply": True, "max_iters": 200,
        })

        assert result["status"] == "success", (
            f"a mid-apply failure must degrade to a normal {{ok:false}} return "
            f"(the scene-resolution try/except boundary), NEVER a raw exception "
            f"through the dispatcher. Got {result!r}"
        )
        data = result["data"]
        assert data.get("ok") is False, (
            f"a mid-apply parm-write exception must surface as an OVERALL failure "
            f"-- never a partial 'success' with only some objects moved. Got {data!r}"
        )


# ---------------------------------------------------------------------------
# FIX-CYCLE 3 -- B1: apply must re-read the WIRE node's geometry, not the
# resolved OBJ node's displayNode() geometry
# ---------------------------------------------------------------------------

class TestSolveLayoutApplyRereadsWireNodeGeometry:
    """B1 (fix-cycle 3, codex-adversarial-reviewer verified Blocker):
    solve_layout's INITIAL resolution reads a movable object's world-AABB
    center/bbox via the WIRE node
    (_world_aabb_center_and_extents(node) -- solve_layout line ~612,
    `node = hou.node(node_path)`), but the current
    _apply_solved_transforms re-reads the POST-ROTATION world-AABB center
    via the RESOLVED OBJ NODE instead
    (_world_aabb_center_and_extents(obj_node) -- line ~515, where
    obj_node = _resolve_obj_node(node)). When the movable object's wire
    `node` is a SopNode that is NOT its OBJ's displayNode(), these two
    reads consult DIFFERENT geometry (different local bbox/center), so
    the post-rotation re-read is WRONG and corrupts the translate delta
    even when the object needs zero net movement.

    Fixture: an OBJ (/obj/thing) whose displayNode() geometry has local
    bbox (4,0,0)-(6,2,2) [local center (5,1,1)] -- DIFFERENT from the
    WIRE object's own node (/obj/thing/wire_child, a SopNode child of the
    OBJ) whose geometry has local bbox (0,0,0)-(2,2,2) [local center
    (1,1,1)]. The OBJ's t0=(10,0,0), r0=(0,0,0), identity parent. With
    relations=[], the model must leave the object's ALREADY-CORRECT
    resolved position (computed from the WIRE node: world center
    (11,1,1)) unchanged -- so a CORRECT apply writes parmTuple('t') back
    to the ORIGINAL (10,0,0) (net delta zero). The CURRENT (buggy) apply
    re-reads via the OBJ's displayNode() geometry instead (world center
    (15,1,1) at the same t0/transform -- a (4,0,0) local-center offset
    from the wire node's geometry), producing delta = solved_t(11,1,1) -
    post_rotation_center(15,1,1) = (-4,0,0), so it INCORRECTLY writes
    t=(6.0, 0.0, 0.0) -- silently dragging an already-correctly-placed
    object away from its solved position. This test FAILS RED against
    that bug and pins the CORRECT (10.0, 0.0, 0.0) value.
    """

    def test_apply_rereads_via_the_wire_node_not_the_display_node(self, mock_hou):
        handlers = _import_handler_module()

        # MUST be set before constructing an _identity_matrix4()-bearing
        # fixture -- without it, `hou.Matrix4(1)` in _check_movable_targets's
        # identity-parent check resolves to a bare, unconfigured MagicMock
        # rather than _fake_matrix4_ctor's real comparable identity matrix,
        # which would make the identity-parent check itself misfire (a
        # broken-mock false positive, not the B1 geometry-source bug this
        # test targets) -- matches the _make_static_objnode /
        # _make_rotating_objnode convention used elsewhere in this file.
        mock_hou.Matrix4 = _fake_matrix4_ctor

        obj_node = mock_hou.ObjNode(name="fake_obj_thing")
        obj_node.path.return_value = "/obj/thing"
        obj_node.parentAndSubnetTransform.return_value = _identity_matrix4()

        # The OBJ's OWN display SOP has a DIFFERENT local bbox/center
        # than the wire SOP below -- local center (5,1,1), vs the wire
        # node's own (1,1,1).
        display_sop = MagicMock(name="fake_display_thing")
        display_geo = MagicMock(name="fake_display_geo")
        display_geo.boundingBox.return_value = _FakeBoundingBox((4.0, 0.0, 0.0), (6.0, 2.0, 2.0))
        display_sop.geometry.return_value = display_geo
        obj_node.displayNode.return_value = display_sop

        t0 = (10.0, 0.0, 0.0)
        r0 = (0.0, 0.0, 0.0)
        obj_node.worldTransform.return_value = _FakeMatrix4(translate=t0, rotate_deg=r0)

        recorder = []
        parm_side_effect, _parms = _make_parm_tuple_lookup(t0, r0, recorder)
        obj_node.parmTuple.side_effect = parm_side_effect
        obj_node._test_recorder = recorder

        # The WIRE object's node is a DIFFERENT SopNode (a child of
        # obj_node) with its OWN geometry -- local bbox (0,0,0)-(2,2,2),
        # local center (1,1,1). solve_layout's INITIAL resolution ALWAYS
        # reads t/r via the wire `node` (there is no caller-transform
        # override, unlike assert_scene).
        wire_sop = mock_hou.SopNode(name="fake_wire_sop")
        wire_sop.path.return_value = "/obj/thing/wire_child"
        wire_sop.parent.return_value = obj_node
        wire_geo = MagicMock(name="fake_wire_geo")
        wire_geo.boundingBox.return_value = _FakeBoundingBox((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
        wire_sop.geometry.return_value = wire_geo

        mock_hou.node.side_effect = _node_lookup({"/obj/thing/wire_child": wire_sop})

        objects_wire = [_obj_wire("thing", "/obj/thing/wire_child", fixed=False)]

        result = _dispatch("solve_layout", {
            "objects": objects_wire, "relations": [], "bounds": None,
            "apply": True, "max_iters": 200,
        })

        assert result["status"] == "success", f"solve_layout must not raise, got {result!r}"
        data = result["data"]

        # Oracle: the object's resolved (and, with relations=[],
        # unchanged) world-AABB center is computed from the WIRE node's
        # OWN geometry -- local center (1,1,1) + world translate
        # (10,0,0) = world center (11,1,1).
        solved_t = data["transforms"]["thing"]["t"]
        assert solved_t == pytest.approx((11.0, 1.0, 1.0), abs=1e-6), (
            f"expected the resolved world-AABB center computed from the WIRE "
            f"node's own geometry, got {solved_t!r}"
        )

        t_sets = [v for name, v in obj_node._test_recorder if name == "t"]
        assert t_sets, (
            f"expected at least one parmTuple('t').set call on the OBJ node, "
            f"got {obj_node._test_recorder!r}"
        )
        final_t = t_sets[-1]

        assert final_t == pytest.approx(t0, abs=1e-6), (
            f"apply's post-rotation re-read MUST use the WIRE node's OWN "
            f"geometry (the SAME source used for the object's INITIAL "
            f"resolution) -- with relations=[] the object needs ZERO net "
            f"movement, so the OBJ's t parm must be left at its original "
            f"value {t0!r}. Got {final_t!r} instead. A re-read that uses the "
            f"resolved OBJ node's displayNode() geometry (a DIFFERENT local "
            f"bbox/center than the wire node's own geometry) computes a WRONG "
            f"post-rotation center and corrupts the translate delta -- the "
            f"exact B1 bug (fix-cycle 3)."
        )


# ---------------------------------------------------------------------------
# FIX-CYCLE 3 -- B2: the OBJ-distinctness check must be ANCESTRY-AWARE, not
# exact-path-equality-only
# ---------------------------------------------------------------------------

class TestSolveLayoutAncestryAwareDistinctness:
    """B2 (fix-cycle 3, codex-adversarial-reviewer verified Blocker):
    _check_movable_targets's OBJ-distinctness check currently rejects ONLY
    an EXACT OBJ-path match (`other_path == path`) -- it does NOT detect
    OBJ NESTING. A MOVABLE object whose OBJ is an ANCESTOR of another
    supplied object's OBJ (e.g. movable `/obj/subnet` and another object
    at `/obj/subnet/geo_b`, both identity-parent) silently drags the
    descendant along when the ancestor is moved, without being flagged --
    two objects sharing an ancestor/descendant OBJ relationship cannot be
    independently placed any more than two objects sharing the SAME OBJ
    node can (the case the current exact-equality check already catches).

    The check MUST reject when a movable OBJ path is an ANCESTOR OR
    DESCENDANT of any OTHER supplied object's OBJ path -- both orderings
    (ancestor listed first, ancestor listed second) are covered below.
    Both tests FAIL RED against the current exact-path-equality-only
    check (no exception is raised -- `pytest.raises` reports
    DID NOT RAISE), since "/obj/subnet" != "/obj/subnet/geo_b".
    """

    @staticmethod
    def _make_two_nested_objnodes(mock_hou, ancestor_path, descendant_path):
        # MUST be set before constructing _identity_matrix4()-bearing fixtures
        # below -- without it, `hou.Matrix4(1)` in _check_movable_targets's
        # identity-parent check resolves to a bare, unconfigured MagicMock
        # rather than _fake_matrix4_ctor's real comparable identity matrix,
        # which makes the identity-parent check itself raise for the WRONG
        # reason (a broken-mock false positive, not the ancestry bug this
        # test targets) -- matches the _make_static_objnode /
        # _make_rotating_objnode convention above.
        mock_hou.Matrix4 = _fake_matrix4_ctor

        ancestor = mock_hou.ObjNode(name="fake_ancestor")
        ancestor.path.return_value = ancestor_path
        ancestor.parentAndSubnetTransform.return_value = _identity_matrix4()

        descendant = mock_hou.ObjNode(name="fake_descendant")
        descendant.path.return_value = descendant_path
        descendant.parentAndSubnetTransform.return_value = _identity_matrix4()

        return ancestor, descendant

    def test_ancestor_object_listed_first_then_descendant_denied(self, mock_hou):
        handlers = _import_handler_module()
        ancestor, descendant = self._make_two_nested_objnodes(
            mock_hou, "/obj/subnet", "/obj/subnet/geo_b"
        )
        mock_hou.node.side_effect = _node_lookup({
            "/obj/subnet": ancestor, "/obj/subnet/geo_b": descendant,
        })

        objects_wire = [
            _obj_wire("parent_obj", "/obj/subnet", fixed=False),
            _obj_wire("child_obj", "/obj/subnet/geo_b", fixed=False),
        ]

        with pytest.raises(mock_hou.OperationFailed):
            handlers._preview_solve_layout({
                "objects": objects_wire, "relations": [], "bounds": None,
                "apply": True, "max_iters": 200,
            })

    def test_descendant_object_listed_first_then_ancestor_denied(self, mock_hou):
        handlers = _import_handler_module()
        ancestor, descendant = self._make_two_nested_objnodes(
            mock_hou, "/obj/subnet", "/obj/subnet/geo_b"
        )
        mock_hou.node.side_effect = _node_lookup({
            "/obj/subnet": ancestor, "/obj/subnet/geo_b": descendant,
        })

        objects_wire = [
            _obj_wire("child_obj", "/obj/subnet/geo_b", fixed=False),
            _obj_wire("parent_obj", "/obj/subnet", fixed=False),
        ]

        with pytest.raises(mock_hou.OperationFailed):
            handlers._preview_solve_layout({
                "objects": objects_wire, "relations": [], "bounds": None,
                "apply": True, "max_iters": 200,
            })
