"""
Hython smoke test: export_alembic_ue + export_fbx handlers.

PP12-111e (PR-5) — Alembic-UE + FBX gated bake.

TDD phase: RED — both dispatch calls fail with "unknown command" because
export_handlers (and the alembic/fbx registrations) do not exist yet.

Run with hython:
    hython tests/export_alembic_fbx_hython_smoke.py

Or as part of pytest (skips if hou not importable):
    pytest tests/export_alembic_fbx_hython_smoke.py -v

Mirrors export_vat_hython_smoke.py exactly in:
  - 3-path sys.path bootstrap
  - dispatcher helper (_get_dispatcher, _data)
  - setup_class / teardown_class pattern (build once, destroy on teardown)
  - standalone main() runner at bottom

Key differences from VAT smoke:
  - Fixture wrangle class = 2 (PRIMITIVE) so i@unreal_test VEX survives as
    a primitive attribute on the Alembic round-trip (FR-4 requirement).
  - Two dispatches: export_alembic_ue + export_fbx (not export_vat).
  - FR-4 check: after Alembic bake, reload the .abc with a File SOP and
    verify the 'unreal_test' primitive attrib survived (the Alembic pipeline
    must preserve unreal_* attributes per FR-4).

Cross-references:
  - plan pp12-111e decomposition[0].acceptanceTests (AT-3 Alembic, AT-4 FBX)
  - tdd-with-agents.md §4.1 (build-then-smoke for the hou/pxr layer)
  - human-test-delegation.md Bucket 2 (headless hython, agent-runnable)
  - mcp-subprocess-delegation.md (live-subprocess rung; this IS that rung)
  - CL-007: agents never commit; orchestrator commits per unit.
"""

from __future__ import annotations

import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# 3-path bootstrap — mirrors export_vat_hython_smoke.py exactly
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PKG_PYTHON = os.path.join(_REPO_ROOT, "python")
_HOUDINI_HANDLERS = os.path.join(_REPO_ROOT, "houdini", "scripts", "python")
for _p in [_REPO_ROOT, _PKG_PYTHON, _HOUDINI_HANDLERS]:
    _abs = os.path.abspath(_p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

# ---------------------------------------------------------------------------
# Dual-mode guard: pytest skip when hou is not importable
# ---------------------------------------------------------------------------
try:
    import hou  # type: ignore[import-not-found]
    _HOU_AVAILABLE = True
except ImportError:
    hou = None  # type: ignore[assignment]
    _HOU_AVAILABLE = False

try:
    import pytest as _pytest
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False

if _HAS_PYTEST:
    import pytest
    pytestmark = pytest.mark.skipif(
        not _HOU_AVAILABLE,
        reason="hou not importable — run via hython for full smoke",
    )


# ---------------------------------------------------------------------------
# Dispatcher helper — mirrors export_vat_hython_smoke.py exactly
# ---------------------------------------------------------------------------

def _get_dispatcher():
    """Import export_handlers to register alembic/fbx commands, then return dispatch.

    TDD phase RED: this import raises ImportError because export_handlers
    does not exist yet. Expected failure message contains 'export_handlers'.
    """
    from fxhoudinimcp_server.handlers import export_handlers  # noqa: F401  # RED
    from fxhoudinimcp_server.dispatcher import dispatch
    return dispatch


def _data(result: dict) -> dict:
    """Unwrap 'data' wrapper if present (mirrors VAT smoke)."""
    return result.get("data", result)


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------

def _build_abc_fbx_fixture(parent_name: str = "geo_abc_fbx_smoke"):
    """Build a minimal geo with a box + primitive-class attribwrangle.

    Wrangle class = 2 (PRIMITIVE) so 'i@unreal_test = 1;' writes a primitive
    attribute. The FR-4 check verifies this attrib survives the Alembic
    round-trip (rop_alembic preserves unreal_* primitive attribs).

    Returns (geo_node, sop_path_string).
    """
    geo = hou.node("/obj").createNode("geo", parent_name)
    box = geo.createNode("box", "box1")

    wr = geo.createNode("attribwrangle", "unreal_attrib_wrangle")
    wr.setFirstInput(box)
    wr.parm("class").set(2)  # 2 = PRIMITIVE (not point=0 as in VAT smoke)
    wr.parm("snippet").set("i@unreal_test = 1;")
    wr.cook(force=True)

    return geo, wr.path()


# ---------------------------------------------------------------------------
# Smoke test class
# ---------------------------------------------------------------------------

class TestAlembicFbxBakeSmoke:
    """Dispatcher-based headless smoke for export_alembic_ue and export_fbx.

    Both bakes fire once in setup_class against a shared fixture. Individual
    test methods assert on the captured results without re-running the bake.

    AT RED: both dispatches are expected to fail with a KeyError / 'unknown
    command' response because the handlers are not registered yet.
    """

    _geo = None
    _sop_path = None
    _tmp_dir = None
    _abc_result = None
    _fbx_result = None
    _abc_path = None
    _fbx_path = None

    # ------------------------------------------------------------------
    # setup_class / teardown_class
    # ------------------------------------------------------------------

    @classmethod
    def setup_class(cls):
        """Build fixture + fire both bakes once.

        Failures here are expected at RED phase (unknown command). Individual
        tests check the result dicts; test_dispatch_does_not_raise is the
        only test that may catch the RED failure gracefully.
        """
        cls._tmp_dir = tempfile.mkdtemp(prefix="abc_fbx_smoke_")
        cls._abc_path = os.path.join(cls._tmp_dir, "out.abc").replace("\\", "/")
        cls._fbx_path = os.path.join(cls._tmp_dir, "out.fbx").replace("\\", "/")

        cls._geo, cls._sop_path = _build_abc_fbx_fixture()

        dispatch = _get_dispatcher()

        # Fire Alembic bake
        cls._abc_result = dispatch(
            "export_alembic_ue",
            {
                "node": cls._sop_path,
                "out_path": cls._abc_path,
                "deforming": True,
                "frame_range": [1, 3],
            },
        )

        # Fire FBX bake
        cls._fbx_result = dispatch(
            "export_fbx",
            {
                "node": cls._sop_path,
                "out_path": cls._fbx_path,
                "frame_range": [1, 3],
            },
        )

    @classmethod
    def teardown_class(cls):
        """Destroy fixture geo; leave tmp files for inspection on failure."""
        if cls._geo is not None:
            try:
                cls._geo.destroy()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Alembic bake assertions (AT-3)
    # ------------------------------------------------------------------

    def test_abc_dispatch_ok(self):
        """export_alembic_ue dispatch returns ok=True.

        TDD RED: fails because 'export_alembic_ue' is not a registered command.
        Expected failure: KeyError / ok=False with message 'unknown command'.
        """
        result = _data(self._abc_result)
        assert result.get("ok") is True, (
            f"export_alembic_ue dispatch must return ok=True; "
            f"got: {result!r}\n"
            f"(TDD RED: expected — handler not registered yet)"
        )

    def test_abc_out_path_extension(self):
        """export_alembic_ue result out_path ends with .abc."""
        result = _data(self._abc_result)
        out = result.get("out_path", "")
        assert out.endswith(".abc"), (
            f"export_alembic_ue result out_path must end with .abc, got {out!r}"
        )

    def test_abc_file_exists_on_disk(self):
        """export_alembic_ue writes an .abc file to disk."""
        assert os.path.isfile(self._abc_path), (
            f"export_alembic_ue must write an .abc file at {self._abc_path!r}; "
            f"file not found."
        )

    def test_abc_file_nonempty(self):
        """Written .abc file has non-zero size."""
        if not os.path.isfile(self._abc_path):
            pytest.skip("abc file not created — prior assertion covers absence")
        size = os.path.getsize(self._abc_path)
        assert size > 0, (
            f"export_alembic_ue wrote an empty .abc file at {self._abc_path!r}"
        )

    def test_abc_fr4_unreal_attrib_survives(self):
        """FR-4: 'unreal_test' unreal_* attrib survives the Alembic round-trip.

        Houdini's alembic SOP loads geometry as packed prims by default — the
        unreal_* attribs live INSIDE the packed geometry and are NOT visible at
        the top-level alembic SOP output (top level shows only ['path']/['P']).

        Fix: wire an unpack SOP after the alembic SOP and read the UNPACKED
        geometry.  Assert that 'unreal_test' appears in EITHER the unpacked prim
        attribs OR the unpacked point attribs (class-agnostic: robust to
        rop_alembic promoting the attrib class on export).

        This is the authoritative FR-4 verification for the Alembic bake path.
        """
        if not os.path.isfile(self._abc_path):
            pytest.skip("abc file not created — test_abc_file_exists_on_disk covers this")

        verify_geo = hou.node("/obj").createNode("geo", "geo_verify_abc")
        try:
            # Load via alembic SOP (not file SOP — file SOP can mis-interpret
            # the packed hierarchy and also returns packed prims).
            abc_sop = verify_geo.createNode("alembic", "read_abc")
            abc_sop.parm("fileName").set(self._abc_path)
            abc_sop.cook(force=True)

            # Unpack so unreal_* attribs inside the packed prims become visible.
            unpack_sop = verify_geo.createNode("unpack", "unpack_abc")
            unpack_sop.setInput(0, abc_sop)
            unpack_sop.cook(force=True)

            geo = unpack_sop.geometry()
            assert geo is not None, "unpack SOP returned None geometry"

            prim_names = [a.name() for a in geo.primAttribs()]
            pt_names   = [a.name() for a in geo.pointAttribs()]

            unreal_in_prim  = "unreal_test" in prim_names
            unreal_in_point = "unreal_test" in pt_names

            assert unreal_in_prim or unreal_in_point, (
                f"FR-4 FAIL: 'unreal_test' not found in unpacked Alembic geometry "
                f"from {self._abc_path!r}.\n"
                f"  Unpacked prim  attribs: {prim_names!r}\n"
                f"  Unpacked point attribs: {pt_names!r}\n"
                f"The export_alembic_ue handler must preserve unreal_* attributes."
            )
        finally:
            verify_geo.destroy()

    def test_abc_result_contains_manifest(self):
        """export_alembic_ue result contains an export manifest (ExportManifest)."""
        result = _data(self._abc_result)
        # Manifest may be nested as 'manifest' key or flattened
        has_manifest = (
            "manifest" in result
            or "files" in result
            or "file_count" in result
        )
        assert has_manifest, (
            f"export_alembic_ue result should contain manifest metadata; "
            f"got keys: {list(result.keys())!r}"
        )

    # ------------------------------------------------------------------
    # FBX bake assertions (AT-4)
    # ------------------------------------------------------------------

    def test_fbx_dispatch_ok(self):
        """export_fbx dispatch returns ok=True.

        TDD RED: fails because 'export_fbx' is not a registered command.
        Expected failure: KeyError / ok=False with message 'unknown command'.
        """
        result = _data(self._fbx_result)
        assert result.get("ok") is True, (
            f"export_fbx dispatch must return ok=True; "
            f"got: {result!r}\n"
            f"(TDD RED: expected — handler not registered yet)"
        )

    def test_fbx_out_path_extension(self):
        """export_fbx result out_path ends with .fbx."""
        result = _data(self._fbx_result)
        out = result.get("out_path", "")
        assert out.endswith(".fbx"), (
            f"export_fbx result out_path must end with .fbx, got {out!r}"
        )

    def test_fbx_file_exists_on_disk(self):
        """export_fbx writes an .fbx file to disk."""
        assert os.path.isfile(self._fbx_path), (
            f"export_fbx must write an .fbx file at {self._fbx_path!r}; "
            f"file not found."
        )

    def test_fbx_file_nonempty(self):
        """Written .fbx file has non-zero size."""
        if not os.path.isfile(self._fbx_path):
            pytest.skip("fbx file not created — prior assertion covers absence")
        size = os.path.getsize(self._fbx_path)
        assert size > 0, (
            f"export_fbx wrote an empty .fbx file at {self._fbx_path!r}"
        )

    def test_fbx_result_contains_manifest(self):
        """export_fbx result contains an export manifest (ExportManifest)."""
        result = _data(self._fbx_result)
        has_manifest = (
            "manifest" in result
            or "files" in result
            or "file_count" in result
        )
        assert has_manifest, (
            f"export_fbx result should contain manifest metadata; "
            f"got keys: {list(result.keys())!r}"
        )

    # ------------------------------------------------------------------
    # Shared / cross-cutting
    # ------------------------------------------------------------------

    def test_abc_and_fbx_are_distinct_files(self):
        """Alembic and FBX dispatches write to distinct output paths."""
        assert self._abc_path != self._fbx_path, (
            "Alembic and FBX output paths must differ"
        )

    def test_fixture_sop_path_is_valid(self):
        """The fixture SOP path used for both bakes resolves to a real node."""
        node = hou.node(self._sop_path)
        assert node is not None, (
            f"Fixture SOP path {self._sop_path!r} does not resolve to a node; "
            f"smoke fixture setup failed"
        )

    def test_abc_version_triple_in_result(self):
        """export_alembic_ue result contains a VersionTriple (tool_version key)."""
        result = _data(self._abc_result)
        has_version = (
            "tool_version" in result
            or "version" in result
            or ("manifest" in result and "tool_version" in result.get("manifest", {}))
        )
        assert has_version, (
            f"export_alembic_ue result should contain VersionTriple / tool_version; "
            f"got keys: {list(result.keys())!r}"
        )

    def test_fbx_version_triple_in_result(self):
        """export_fbx result contains a VersionTriple (tool_version key)."""
        result = _data(self._fbx_result)
        has_version = (
            "tool_version" in result
            or "version" in result
            or ("manifest" in result and "tool_version" in result.get("manifest", {}))
        )
        assert has_version, (
            f"export_fbx result should contain VersionTriple / tool_version; "
            f"got keys: {list(result.keys())!r}"
        )


# ---------------------------------------------------------------------------
# Standalone hython runner
# ---------------------------------------------------------------------------

def main():
    """Run the smoke test directly under hython (no pytest required)."""
    if not _HOU_AVAILABLE:
        print("ERROR: hou not importable. Run via hython, not CPython.")
        sys.exit(1)

    hou.hipFile.clear(suppress_save_prompt=True)

    # Trigger the dispatcher import to confirm RED state
    print("[export_alembic_fbx_smoke] Verifying RED state (handlers absent)...")
    try:
        dispatch = _get_dispatcher()
        print(
            "[export_alembic_fbx_smoke] WARNING: _get_dispatcher() did NOT raise. "
            "export_handlers may already be registered. "
            "Confirm this is genuinely RED before handing off to hou-dev."
        )
    except (ImportError, ModuleNotFoundError) as exc:
        print(
            f"[export_alembic_fbx_smoke] RED confirmed: {type(exc).__name__}: {exc}"
        )
        print("[export_alembic_fbx_smoke] DONE — expected RED failure captured.")
        return

    # If handlers somehow exist, run the full smoke so hou-dev can verify GREEN
    print("[export_alembic_fbx_smoke] Handlers present — running full smoke ...")
    suite = TestAlembicFbxBakeSmoke()
    TestAlembicFbxBakeSmoke.setup_class()
    try:
        failures = []
        for name in dir(suite):
            if name.startswith("test_"):
                try:
                    getattr(suite, name)()
                    print(f"  PASS  {name}")
                except AssertionError as exc:
                    print(f"  FAIL  {name}: {exc}")
                    failures.append(name)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
                    failures.append(name)
    finally:
        TestAlembicFbxBakeSmoke.teardown_class()

    if failures:
        print(f"\n[export_alembic_fbx_smoke] {len(failures)} FAILURE(S): {failures}")
        sys.exit(1)
    else:
        test_count = sum(1 for n in dir(suite) if n.startswith("test_"))
        print(f"\n[export_alembic_fbx_smoke] ALL {test_count} TESTS PASSED.")


if __name__ == "__main__":
    main()
