"""Handler-level mock-hou pytest for usd_export_rop's edge branches
(PP12-112 PR-4, pp12-112d) -- the two REQUIRED cap-Blocker red assertions
(plan-5-cap, plan-7-cap) plus the guard-clause assertions.

Unit: pp12-112d
testVerificationSurface: pytest-model (mock-hou rung, per
    test-fixture-conventions.md section 2.3 -- these edge branches cannot be
    reliably forced through a REAL ROP cook, so they are pinned here rather
    than in the hython-smoke file)
planSha: 27f1d7b8428d108f13de6cad095a4f476de5f41319d4b8972c481244cd5c67ff

Why a mock-hou file, not hython-smoke, for these two branches
---------------------------------------------------------------
The plan rev3 lockedFieldContract identifies TWO cap-round Blockers
(round-2 plan-review found rev2's fold of round-1 plan-5/plan-7 was only
COSMETIC) that MUST be independently red-tested:

  plan-5-cap: "the ROP cooks with clean errors() but NO file lands at
      out_path" -- a silent no-op at the FILE layer. This cannot be forced
      through a real Houdini ROP cook (a real cook that reports no errors()
      normally DOES write the file); it requires monkeypatching
      os.path.exists to simulate "errors()==[] but no file appeared".

  plan-7-cap: "the written file's magic-bytes format != the out_path
      extension format" -- e.g. a .usdc out_path whose actual bytes are
      ASCII (#usda). A real ROP always writes bytes matching its own
      lopoutput extension resolution, so this mismatch cannot be forced by
      driving a real ROP either; it requires monkeypatching the file-read
      step (format_from_magic_bytes) to return a format that disagrees with
      the extension-derived format.

Per test-fixture-conventions.md section 2.3, both `hou` AND `pxr` are
installed into sys.modules as configured mocks (MagicMock(name=...) with an
explicit `hou.OperationFailed` real-Exception-subclass override, per the
"configure return values explicitly" discipline -- a bare MagicMock() would
silently accept `hou.OperationFailed` as a non-raising mock class, defeating
any test that expects a raise). The module-under-test is imported INSIDE
each test, AFTER the mocks are installed (test-fixture-conventions.md
section 2.3's "import inside the test" rule).

This file also carries the guard-clause assertions (empty lop_node/out_path,
malformed frame_range shapes, a non-LOP/uncooked node) since those, too, are
edge branches best pinned against a controlled mock rather than a real ROP.
"""

from __future__ import annotations

import importlib
import os
import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# hou / pxr mock installation (module-scoped fixture — installed BEFORE any
# import of usd_export_handlers, removed after each test to avoid leaking a
# stale mock into a later test file in the same pytest session).
# ---------------------------------------------------------------------------

class _FakeOperationFailed(Exception):
    """A REAL Exception subclass standing in for hou.OperationFailed.

    test-fixture-conventions.md section 2.3: a bare MagicMock() for
    hou.OperationFailed would not behave as a raisable/catchable exception
    class -- it must be a real Exception subclass so `raise hou.OperationFailed(...)`
    and `except hou.OperationFailed:` work exactly as they do against the
    real hou module.
    """


@pytest.fixture()
def mock_hou_and_pxr(monkeypatch):
    """Install configured mocks for `hou` and `pxr` into sys.modules.

    Returns a namespace object exposing the configured mocks so tests can
    further configure per-test behavior (e.g. a node's .stage() return, or
    hou.text.expandString's behavior).
    """
    hou_mock = MagicMock(name="hou")
    hou_mock.OperationFailed = _FakeOperationFailed
    # hou.text.expandString: identity passthrough is the simplest useful
    # default (out_path strings in these tests already look expanded).
    hou_mock.text = MagicMock(name="hou.text")
    hou_mock.text.expandString = MagicMock(side_effect=lambda s: s)

    pxr_mock = MagicMock(name="pxr")
    pxr_mock.Usd = MagicMock(name="pxr.Usd")
    pxr_mock.UsdShade = MagicMock(name="pxr.UsdShade")

    monkeypatch.setitem(sys.modules, "hou", hou_mock)
    monkeypatch.setitem(sys.modules, "pxr", pxr_mock)

    # Ensure the fork's non-standard package roots are importable (mirrors
    # the sys.path bootstrap usd_export_handlers.py itself performs, plus
    # the houdini/scripts/python root the handlers module lives under).
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    _pkg_python = os.path.join(_repo_root, "python")
    _houdini_handlers = os.path.join(_repo_root, "houdini", "scripts", "python")
    for _p in (_pkg_python, _houdini_handlers):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    class _Namespace:
        hou = hou_mock
        pxr = pxr_mock

    return _Namespace()


def _fresh_import_usd_export_handlers():
    """Import (or re-import) usd_export_handlers fresh, after mocks are set.

    Uses importlib.reload when the module is already cached from a prior
    test in the same session (module-level state like the dispatcher
    registry is process-global; reload re-runs the register_handler calls,
    which is idempotent for this module's registration pattern).
    """
    mod_name = "fxhoudinimcp_server.handlers.usd_export_handlers"
    if mod_name in sys.modules:
        return importlib.reload(sys.modules[mod_name])
    return importlib.import_module(mod_name)


# ---------------------------------------------------------------------------
# Guard: confirm the mock-hou rung itself is sound (module imports, and
# usd_export_rop is present once hou-dev implements it -- this file's own
# red gate is the same ImportError-adjacent AttributeError as the wrapper
# pytest, surfaced as "module has no attribute 'usd_export_rop'").
# ---------------------------------------------------------------------------

class TestModuleImportsWithMockedHouAndPxr:
    """Sanity: usd_export_handlers must import cleanly under the mock-hou
    rung (HAS_PXR becomes True once pxr is mocked, exercising the REAL code
    path rather than the _require_pxr() early-raise short-circuit)."""

    def test_module_imports_and_has_pxr_true(self, mock_hou_and_pxr):
        mod = _fresh_import_usd_export_handlers()
        assert mod.HAS_PXR is True, (
            "With pxr mocked into sys.modules, usd_export_handlers.HAS_PXR "
            "must be True (exercising the real code path, not the "
            "_require_pxr() early-raise short-circuit)."
        )


# ---------------------------------------------------------------------------
# PLAN-5-CAP (REQUIRED, cap-round Blocker): ROP cooks with clean errors()
#   but NO file lands at out_path -> {ok: False} whose error does NOT claim
#   the file was written.
# ---------------------------------------------------------------------------

class TestPlan5CapNoFileSilentNoOp:
    """PLAN-5-CAP (REQUIRED): a clean cook (rop.errors() == []) that produces
    NO file at out_path must be reported as a FAILURE naming the no-op --
    NOT as a success, and NOT with an error message that falsely claims the
    file was written.

    This is the round-2 cap-adjudication finding: rev2 set
    written_path = expanded UNCONDITIONALLY right after destroy(), BEFORE
    confirming the file actually landed on disk. rev3 requires an explicit
    os.path.exists(expanded) check BEFORE written_path is ever set, and a
    dedicated {ok: False, error: 'produced no file'} branch when the check
    fails.
    """

    def test_clean_cook_no_file_returns_ok_false_not_claiming_written(
        self, mock_hou_and_pxr, monkeypatch
    ):
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "usd_export_rop"):
            pytest.fail(
                "usd_export_handlers.usd_export_rop does not exist yet -- "
                "RED (expected before hou-dev implements it)."
            )

        hou_mock = mock_hou_and_pxr.hou

        # Build a fake cooked LOP node: hasattr(node, 'stage') True,
        # node.stage() returns a non-None stage-ish object.
        fake_stage = MagicMock(name="fake_stage")
        fake_node = MagicMock(name="fake_lop_node")
        fake_node.stage = MagicMock(return_value=fake_stage)

        # Fake /out network + fake ROP node the handler will create.
        fake_rop = MagicMock(name="fake_rop")
        fake_rop.parm = MagicMock(side_effect=lambda name: MagicMock(name=f"parm.{name}"))
        fake_rop.parmTuple = MagicMock(side_effect=lambda name: MagicMock(name=f"parmTuple.{name}"))
        # CLEAN cook: errors() returns an empty list (no cook error at all).
        fake_rop.errors = MagicMock(return_value=[])
        fake_rop.destroy = MagicMock()

        fake_out_net = MagicMock(name="fake_out_net")
        fake_out_net.createNode = MagicMock(return_value=fake_rop)

        def _fake_hou_node(path):
            if path == "/stage/fake_lop1":
                return fake_node
            if path == "/out":
                return fake_out_net
            return None

        hou_mock.node = MagicMock(side_effect=_fake_hou_node)

        # THE SILENT-NO-OP FORCE: os.path.exists must report False for the
        # expected out_path even though the cook reported zero errors.
        real_exists = os.path.exists

        def _fake_exists(path):
            if path == "$HIP/no_file_ever_lands.usdc":
                return False
            return real_exists(path)

        # Patch os.path.exists as imported/used inside the handlers module
        # (the module does `import os as _os` -- patch the shared os module
        # object's exists function, which both the test and the handler
        # module resolve through the same os module instance).
        monkeypatch.setattr(os.path, "exists", _fake_exists)

        result = mod.usd_export_rop(
            lop_node="/stage/fake_lop1",
            out_path="$HIP/no_file_ever_lands.usdc",
            frame_range=None,
        )

        assert result.get("ok") is False, (
            f"A clean cook (errors()==[]) that produces NO file at out_path must "
            f"return ok=False (a silent no-op is a FAILURE, not a success). "
            f"Got result={result!r}."
        )
        error_text = str(result.get("error", "")).lower()
        assert "no file" in error_text or "nothing written" in error_text, (
            f"plan-5-cap: the error message for a clean-cook-but-no-file case must "
            f"say something like 'produced no file' / 'nothing written' -- NOT "
            f"claim the file was written. Got error={result.get('error')!r}."
        )
        assert "file written" not in error_text, (
            f"plan-5-cap regression guard: the error message must NOT contain "
            f"'file written' when no file actually landed on disk (that would be "
            f"the exact cosmetic-fold bug the cap round found in rev2). "
            f"Got error={result.get('error')!r}."
        )
        # written_path must not be truthily reported as the successful path
        # when nothing was written.
        assert not result.get("out_path"), (
            f"plan-5-cap: out_path in the failure envelope must NOT report the "
            f"target path as if it were successfully written when no file "
            f"actually landed. Got out_path={result.get('out_path')!r}."
        )


# ---------------------------------------------------------------------------
# PLAN-7-CAP (REQUIRED, cap-round Blocker): written file's magic-bytes
#   format != out_path extension format -> {ok: False} naming the mismatch,
#   NOT {ok: True} with the mismatch merely nested in validator_post.
# ---------------------------------------------------------------------------

class TestPlan7CapFormatMismatchIsHardFailure:
    """PLAN-7-CAP (REQUIRED): when the ROP-written file's magic-bytes-derived
    actual_format disagrees with the out_path-extension-derived format, the
    handler must return {ok: False} NAMING the mismatch as the top-level
    error -- NOT {ok: True} with the discrepancy merely buried inside
    validator_post's nested checks (the round-2 cap-adjudication finding:
    rev2 returned ok=True unconditionally with the mismatch only visible if
    the caller inspects validator_post.checks).
    """

    def test_format_mismatch_returns_ok_false_naming_mismatch(
        self, mock_hou_and_pxr, monkeypatch, tmp_path
    ):
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "usd_export_rop"):
            pytest.fail(
                "usd_export_handlers.usd_export_rop does not exist yet -- "
                "RED (expected before hou-dev implements it)."
            )

        hou_mock = mock_hou_and_pxr.hou

        fake_stage = MagicMock(name="fake_stage")
        fake_node = MagicMock(name="fake_lop_node")
        fake_node.stage = MagicMock(return_value=fake_stage)

        fake_rop = MagicMock(name="fake_rop")
        fake_rop.parm = MagicMock(side_effect=lambda name: MagicMock(name=f"parm.{name}"))
        fake_rop.parmTuple = MagicMock(side_effect=lambda name: MagicMock(name=f"parmTuple.{name}"))
        fake_rop.errors = MagicMock(return_value=[])
        fake_rop.destroy = MagicMock()

        fake_out_net = MagicMock(name="fake_out_net")
        fake_out_net.createNode = MagicMock(return_value=fake_rop)

        # A real temp file on disk with ASCII (#usda) content, but the
        # requested out_path extension is .usdc -- this is the crafted
        # format mismatch. The handler's actual open()+read() of the file
        # will therefore genuinely observe ASCII magic bytes regardless of
        # extension, no format_from_magic_bytes mock needed (real function,
        # real mismatch input).
        mismatched_path = str(tmp_path / "mismatched.usdc")
        with open(mismatched_path, "wb") as fh:
            fh.write(b"#usda 1.0\n(\n)\n")

        def _fake_hou_node(path):
            if path == "/stage/fake_lop2":
                return fake_node
            if path == "/out":
                return fake_out_net
            return None

        hou_mock.node = MagicMock(side_effect=_fake_hou_node)
        # expandString must resolve to our crafted mismatched file so the
        # handler's os.path.exists + open() calls hit the real crafted bytes.
        hou_mock.text.expandString = MagicMock(
            side_effect=lambda s: mismatched_path if s == "$HIP/mismatch_target.usdc" else s
        )

        result = mod.usd_export_rop(
            lop_node="/stage/fake_lop2",
            out_path="$HIP/mismatch_target.usdc",
            frame_range=None,
        )

        assert result.get("ok") is False, (
            f"plan-7-cap: a format mismatch (requested usdc by extension, but "
            f"actual written bytes are usda/ASCII) must return ok=False as a "
            f"HARD top-level failure -- NOT ok=True with the mismatch merely "
            f"nested inside validator_post. Got result={result!r}."
        )
        error_text = str(result.get("error", "")).lower()
        assert "format" in error_text and "mismatch" in error_text, (
            f"plan-7-cap: the top-level error must NAME the format mismatch "
            f"explicitly (e.g. 'format mismatch: requested usdc ... but ROP "
            f"wrote usda'). Got error={result.get('error')!r}."
        )
        # Regression guard: this must NOT be the cosmetic rev2 bug where
        # ok=True was returned with the mismatch only visible in a nested
        # validator_post field.
        assert result.get("ok") is not True, (
            "plan-7-cap regression guard: ok must not be True when format != "
            "actual_format -- that was exactly the cosmetic-fold bug the cap "
            "round found in rev2 (mismatch nested in validator_post only)."
        )


# ---------------------------------------------------------------------------
# Guard clauses: empty args, malformed frame_range shapes, non-LOP/uncooked
#   node -- all -> {ok: False, error: ...} with NO ROP leaked.
# ---------------------------------------------------------------------------

class TestGuardClauses:
    """FR-2-style guard clauses. Every case here must return
    {ok: False, error: <str>} WITHOUT creating (or leaking) a ROP node."""

    @pytest.mark.parametrize("lop_node,out_path", [
        ("", "$HIP/x.usdc"),
        ("   ", "$HIP/x.usdc"),
        ("/stage/sphere1", ""),
        ("/stage/sphere1", "   "),
    ])
    def test_empty_lop_node_or_out_path_rejected(
        self, mock_hou_and_pxr, lop_node, out_path
    ):
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "usd_export_rop"):
            pytest.fail("usd_export_rop does not exist yet -- RED (expected).")

        hou_mock = mock_hou_and_pxr.hou

        # RED-2 (REQUIRED strengthening): the empty-arg guard fires at the
        # TOP of the handler, BEFORE any /out lookup or ROP creation. Wire a
        # fake /out network whose createNode we can assert was NEVER called
        # -- proving no ROP is created (and therefore none can be leaked) on
        # this path. A regression that reordered the guard after ROP
        # creation would create+leak a real /out node; this assertion
        # catches exactly that class of bug, matching the discipline the
        # adjacent malformed-frame_range guard test already applies (its
        # hou.node-must-not-be-called assertion).
        fake_out_net = MagicMock(name="fake_out_net")
        fake_out_net.createNode = MagicMock(
            side_effect=AssertionError(
                "out_net.createNode(...) must NOT be called when lop_node/out_path "
                "is empty -- the empty-arg guard must reject BEFORE any ROP is "
                "created (and therefore before any ROP could be leaked)."
            )
        )

        def _fake_hou_node(path):
            if path == "/out":
                return fake_out_net
            return None

        hou_mock.node = MagicMock(side_effect=_fake_hou_node)

        result = mod.usd_export_rop(lop_node=lop_node, out_path=out_path, frame_range=None)
        assert result.get("ok") is False, (
            f"lop_node={lop_node!r}, out_path={out_path!r} must be rejected "
            f"(ok=False). Got result={result!r}."
        )
        assert isinstance(result.get("error"), str) and result["error"], (
            f"A rejected empty-arg call must carry a non-empty string error. "
            f"Got error={result.get('error')!r}."
        )
        # RED-2: explicit no-leak assertion -- createNode was never invoked
        # on the /out network for an empty-arg call.
        fake_out_net.createNode.assert_not_called()

    @pytest.mark.parametrize("frame_range", [
        ["foo"],
        [1],
        [1, 2, 3],
        [True, 2],
        [1, True],
        [float("nan"), 2],
        [1, float("nan")],
        [float("inf"), 2],
        [1, float("inf")],
        [3, 1],  # reversed: start > end
        "not-a-list",
        {"start": 1, "end": 2},
    ])
    def test_malformed_frame_range_rejected_before_rop_creation(
        self, mock_hou_and_pxr, frame_range
    ):
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "usd_export_rop"):
            pytest.fail("usd_export_rop does not exist yet -- RED (expected).")

        hou_mock = mock_hou_and_pxr.hou

        # If the handler incorrectly proceeds past the frame_range guard, it
        # would call hou.node(...) -- fail loudly by making hou.node raise if
        # invoked, proving the guard short-circuits BEFORE any node lookup /
        # ROP creation.
        def _node_should_not_be_called(path):
            raise AssertionError(
                f"hou.node({path!r}) must NOT be called when frame_range={frame_range!r} "
                f"is malformed -- the shape guard must reject BEFORE any node lookup "
                f"or ROP creation."
            )

        hou_mock.node = MagicMock(side_effect=_node_should_not_be_called)

        result = mod.usd_export_rop(
            lop_node="/stage/sphere1",
            out_path="$HIP/x.usdc",
            frame_range=frame_range,
        )

        assert result.get("ok") is False, (
            f"frame_range={frame_range!r} must be rejected (ok=False) before any "
            f"ROP is created. Got result={result!r}."
        )
        assert "frame_range" in str(result.get("error", "")).lower(), (
            f"The rejection error should reference frame_range. "
            f"Got error={result.get('error')!r}."
        )

    def test_non_lop_node_rejected(self, mock_hou_and_pxr):
        """A node that exists but has no stage() method (not a LOP node)
        must be rejected with ok=False, no ROP created."""
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "usd_export_rop"):
            pytest.fail("usd_export_rop does not exist yet -- RED (expected).")

        hou_mock = mock_hou_and_pxr.hou

        # A node object WITHOUT a .stage attribute at all (spec=[] gives a
        # MagicMock with a restricted attribute set so hasattr(node, 'stage')
        # is genuinely False, not just "returns a Mock").
        fake_non_lop_node = MagicMock(name="fake_non_lop_node", spec=[])

        def _fake_hou_node(path):
            if path == "/obj/geo1":
                return fake_non_lop_node
            return None

        hou_mock.node = MagicMock(side_effect=_fake_hou_node)

        result = mod.usd_export_rop(
            lop_node="/obj/geo1", out_path="$HIP/x.usdc", frame_range=None
        )
        assert result.get("ok") is False, (
            f"A non-LOP node (no stage() method) must be rejected. Got result={result!r}."
        )
        assert "lop" in str(result.get("error", "")).lower(), (
            f"The rejection error should mention it is not a LOP node. "
            f"Got error={result.get('error')!r}."
        )

    def test_uncooked_lop_node_rejected(self, mock_hou_and_pxr):
        """A LOP node whose stage() returns None (uncooked / no composed
        stage) must be rejected with ok=False, no ROP created."""
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "usd_export_rop"):
            pytest.fail("usd_export_rop does not exist yet -- RED (expected).")

        hou_mock = mock_hou_and_pxr.hou

        fake_uncooked_node = MagicMock(name="fake_uncooked_lop_node")
        fake_uncooked_node.stage = MagicMock(return_value=None)

        def _fake_hou_node(path):
            if path == "/stage/uncooked1":
                return fake_uncooked_node
            return None

        hou_mock.node = MagicMock(side_effect=_fake_hou_node)

        result = mod.usd_export_rop(
            lop_node="/stage/uncooked1", out_path="$HIP/x.usdc", frame_range=None
        )
        assert result.get("ok") is False, (
            f"An uncooked LOP node (stage() returns None) must be rejected. "
            f"Got result={result!r}."
        )
        error_text = str(result.get("error", "")).lower()
        assert "stage" in error_text or "cook" in error_text, (
            f"The rejection error should mention the missing/uncooked stage. "
            f"Got error={result.get('error')!r}."
        )

    def test_node_not_found_rejected(self, mock_hou_and_pxr):
        """A lop_node path that resolves to no node at all (hou.node(...)
        returns None) must be rejected with ok=False."""
        mod = _fresh_import_usd_export_handlers()
        if not hasattr(mod, "usd_export_rop"):
            pytest.fail("usd_export_rop does not exist yet -- RED (expected).")

        hou_mock = mock_hou_and_pxr.hou
        hou_mock.node = MagicMock(return_value=None)

        result = mod.usd_export_rop(
            lop_node="/stage/does_not_exist", out_path="$HIP/x.usdc", frame_range=None
        )
        assert result.get("ok") is False, (
            f"A non-existent lop_node must be rejected. Got result={result!r}."
        )
        assert "not found" in str(result.get("error", "")).lower(), (
            f"The rejection error should say the node was not found. "
            f"Got error={result.get('error')!r}."
        )
