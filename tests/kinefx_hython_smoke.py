"""
kinefx_hython_smoke.py — headless Houdini smoke for PP12-110b KineFX/APEX tools.

Dual-mode:
  - Standalone: hython kinefx_hython_smoke.py  (requires hython; pytest NOT required)
  - pytest: collected as a test file; @pytest.mark.hython_smoke tests SKIP
    automatically when hou is not importable (bare CI / off-DCC).

Covers:
  1. All 7 KineFX/APEX node types are creatable in a headless hython session.
  2. A synthetic Skeleton serialises to the §7.3 envelope with correct structure.

testVerificationSurface: hython-smoke
unitId: pp12-110b
"""

from __future__ import annotations

import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
_PKG_PYTHON = os.path.join(_REPO_ROOT, "python")
for _p in [_REPO_ROOT, _PKG_PYTHON]:
    _abs = os.path.abspath(_p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

# ---------------------------------------------------------------------------
# pytest availability guard
# Houdini's bundled hython does NOT have pytest installed.  Provide no-op shims
# so that:
#   - Under hython:  decorated functions/classes still define; __main__ runner
#                    exercises the checks directly.
#   - Under pytest:  real pytest is used; @pytest.mark.skipif guards skip
#                    everything when hou is absent.
# ---------------------------------------------------------------------------
try:
    import pytest  # type: ignore[import-not-found]
    _PYTEST_AVAILABLE = True
except ImportError:
    _PYTEST_AVAILABLE = False

    class _SkipIfDecorator:
        """No-op replacement for pytest.mark.skipif (hython standalone mode)."""
        def __call__(self, condition, reason=""):
            def decorator(fn_or_cls):
                return fn_or_cls
            return decorator

    class _ParametrizeDecorator:
        """No-op replacement for pytest.mark.parametrize (hython standalone mode)."""
        def __call__(self, *args, **kwargs):
            def decorator(fn_or_cls):
                return fn_or_cls
            return decorator

    class _MarkNS:
        """Minimal namespace shim for pytest.mark (hython standalone mode)."""
        skipif = _SkipIfDecorator()
        parametrize = _ParametrizeDecorator()
        hython_smoke = staticmethod(lambda fn_or_cls: fn_or_cls)

        def __getattr__(self, name):
            # Any other mark attribute returns an identity decorator factory.
            def decorator_factory(*args, **kwargs):
                def decorator(fn_or_cls):
                    return fn_or_cls
                return decorator
            return decorator_factory

    class pytest:  # type: ignore[no-redef]
        """Minimal pytest shim for hython standalone execution."""
        mark = _MarkNS()

        @staticmethod
        def fail(msg: str) -> None:
            raise AssertionError(msg)


# ---------------------------------------------------------------------------
# hou availability guard
# ---------------------------------------------------------------------------
try:
    import hou  # type: ignore[import-not-found]
    HOU_AVAILABLE = True
except ImportError:
    HOU_AVAILABLE = False

# pytestmark is only meaningful when collected by pytest.
if _PYTEST_AVAILABLE:
    pytestmark = pytest.mark.hython_smoke

# ===========================================================================
# The 7 KineFX / APEX node types under test (§9.1)
# ===========================================================================

# Each entry: (node_type, context_type)
# All live under /obj or OBJ-level containers reachable from /obj.
_KINEFX_NODE_TYPES = [
    ("kinefx::fbxcharacterimport", "sop"),
    ("kinefx::fbxanimimport",      "sop"),
    ("bonedeform",                 "sop"),
    ("rigmatchpose",               "sop"),
    ("motiontransform",            "sop"),
    ("kinefx::secondarymotion",    "sop"),
    ("apex::autorigcomponent",     "sop"),
]


# ===========================================================================
# Tests: node creatability (hython_smoke)
# ===========================================================================

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestKinefxNodeTypes:
    """Each KineFX/APEX node type must be creatable inside hython."""

    @classmethod
    def setup_class(cls):
        """Create a fresh scene and an OBJ geo container."""
        hou.hipFile.clear(suppress_save_prompt=True)
        cls._obj = hou.node("/obj")
        cls._geo = cls._obj.createNode("geo", "smoke_geo")

    @classmethod
    def teardown_class(cls):
        try:
            cls._geo.destroy()
        except Exception:
            pass

    @pytest.mark.parametrize("node_type,ctx", _KINEFX_NODE_TYPES)
    def test_node_type_createable(self, node_type, ctx):
        """Node type must be installed and creatable without an error."""
        parent = self.__class__._geo
        node = None
        try:
            node = parent.createNode(node_type, f"smoke_{node_type.replace('::', '_')}")
            assert node is not None, f"createNode returned None for {node_type}"
            assert node.type().name() != "", f"Created node has empty type name: {node_type}"
        except hou.OperationFailed as exc:
            pytest.fail(f"createNode failed for {node_type}: {exc}")
        finally:
            if node is not None:
                try:
                    node.destroy()
                except Exception:
                    pass


# ===========================================================================
# Tests: skeleton serialisation (§7.3 shape, hython_smoke)
# ===========================================================================

@pytest.mark.skipif(not HOU_AVAILABLE, reason="hou not available — requires hython")
class TestSkeletonSerialiseHython:
    """
    Drive the kinefx_model public surface inside hython and verify §7.3 shape.

    Unlike the pure-model pytest (test_serialize_skeleton.py, which runs off-DCC),
    this smoke runs inside hython to confirm the model is importable and correct
    in the Houdini Python environment.
    """

    def test_skeleton_to_json_shape_in_hython(self):
        """Synthetic 5-joint skeleton must serialise to §7.3 envelope."""
        from fxhoudinimcp.kinefx_model import (
            Joint, Skeleton, derive_parents, pack_trs, skeleton_to_json,
        )

        def identity_4x4():
            return [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]

        names = ["Hips", "Spine", "Head", "LeftUpLeg", "RightUpLeg"]
        edges = [
            ("Hips", "Spine"),
            ("Spine", "Head"),
            ("Hips", "LeftUpLeg"),
            ("Hips", "RightUpLeg"),
        ]

        parents = derive_parents(edges, names)
        joints = []
        for name in names:
            t, r, s = pack_trs(identity_4x4())
            rest = {"t": list(t), "r": list(r), "s": list(s)}
            joints.append(Joint(name=name, parent=parents[name], rest=rest))

        skeleton = Skeleton(joints=joints)
        out = skeleton_to_json(skeleton)

        # §7.3 top-level keys
        assert set(out.keys()) == {"count", "joints"}, f"Unexpected keys: {set(out.keys())}"
        assert out["count"] == 5

        # Every joint has name, parent, rest
        by_name = {}
        for j in out["joints"]:
            assert "name" in j
            assert "parent" in j
            assert "rest" in j
            rest = j["rest"]
            assert "t" in rest and "r" in rest and "s" in rest
            by_name[j["name"]] = j

        # Parent topology
        assert by_name["Hips"]["parent"] is None
        assert by_name["Spine"]["parent"] == "Hips"
        assert by_name["Head"]["parent"] == "Spine"
        assert by_name["LeftUpLeg"]["parent"] == "Hips"
        assert by_name["RightUpLeg"]["parent"] == "Hips"

        # anim key absent
        for j in out["joints"]:
            assert "anim" not in j, f"anim must be absent; found in {j['name']}"

        # JSON serialisable
        serialised = json.dumps(out)
        reloaded = json.loads(serialised)
        assert reloaded["count"] == 5

    def test_quaternion_normalised_in_hython(self):
        from fxhoudinimcp.kinefx_model import pack_trs
        identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        t, r, s = pack_trs(identity)
        qx, qy, qz, qw = r
        norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        assert abs(norm - 1.0) < 1e-6, f"Quaternion not normalised: norm={norm}"


# ===========================================================================
# Standalone runner (hython kinefx_hython_smoke.py)
# ===========================================================================

def main() -> int:
    if not HOU_AVAILABLE:
        print("ERROR: hou not importable — run this script with hython, not python.")
        return 1

    failures: list[str] = []

    # --- Node creatability ---
    print("\n--- KineFX / APEX node creatability ---")
    hou.hipFile.clear(suppress_save_prompt=True)
    geo = hou.node("/obj").createNode("geo", "smoke_geo_standalone")
    try:
        for node_type, _ctx in _KINEFX_NODE_TYPES:
            node = None
            try:
                node = geo.createNode(node_type, f"smoke_{node_type.replace('::', '_')}")
                print(f"  PASS  {node_type}")
            except Exception as exc:
                print(f"  FAIL  {node_type}: {exc}")
                failures.append(f"createNode({node_type}): {exc}")
            finally:
                if node:
                    try:
                        node.destroy()
                    except Exception:
                        pass
    finally:
        try:
            geo.destroy()
        except Exception:
            pass

    # --- Skeleton serialisation ---
    print("\n--- Skeleton serialisation (§7.3) ---")
    try:
        from fxhoudinimcp.kinefx_model import (
            Joint, Skeleton, derive_parents, pack_trs, skeleton_to_json,
        )
        names = ["Hips", "Spine", "Head", "LeftUpLeg", "RightUpLeg"]
        edges = [
            ("Hips", "Spine"), ("Spine", "Head"),
            ("Hips", "LeftUpLeg"), ("Hips", "RightUpLeg"),
        ]
        parents = derive_parents(edges, names)
        joints = []
        for name in names:
            identity = [
                [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
            ]
            t, r, s = pack_trs(identity)
            rest = {"t": list(t), "r": list(r), "s": list(s)}
            joints.append(Joint(name=name, parent=parents[name], rest=rest))
        out = skeleton_to_json(Skeleton(joints=joints))
        assert out["count"] == 5
        assert set(out.keys()) == {"count", "joints"}
        json.dumps(out)  # confirms serialisability
        print(f"  PASS  skeleton_to_json: count={out['count']}, keys={sorted(out.keys())}")
    except Exception as exc:
        print(f"  FAIL  skeleton_to_json: {exc}")
        failures.append(f"skeleton_to_json: {exc}")

    print()
    if failures:
        print(f"SMOKE FAILED — {len(failures)} failure(s):")
        for f in failures:
            print(f"  {f}")
        return 1
    else:
        print(f"SMOKE PASSED — all {len(_KINEFX_NODE_TYPES)} node types + skeleton serialisation OK")
        return 0


if __name__ == "__main__":
    sys.exit(main())
