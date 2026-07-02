"""Hython-smoke tests for the usd_export_rop dispatcher path (PP12-112 PR-4, pp12-112d).

Unit: pp12-112d
testVerificationSurface: hython-smoke
planSha: 27f1d7b8428d108f13de6cad095a4f476de5f41319d4b8972c481244cd5c67ff

Run under hython (Houdini's headless Python interpreter):
    hython tests/usd_export_rop_hython_smoke.py

Two-mode guard (mirrors usd_export_layer_hython_smoke.py, pp12-112c):
  - If fxhoudinimcp_server.handlers.usd_export_handlers does not yet register
    'usd_export_rop' (hou-dev has not implemented it), dispatch() returns
    the dispatcher's own {"status": "error", "error": {"code": "UNKNOWN_COMMAND", ...}}
    shape -- this IS the expected RED signal for this file (the dispatcher
    module itself already exists from pp12-112b/c; only the NEW command is red).
  - Once hou-dev implements the gated handler + preview_fn + registration,
    the tests run and PASS GREEN.

Dispatcher-based (NOT direct handler calls) -- exercises the REAL
handler(**params) calling convention:
  - Importing usd_export_handlers triggers register_handler('usd_export_rop', ...)
    as a side effect (alongside the pp12-112b/c registrations already shipped).
  - Tests call dispatch('usd_export_rop', {'lop_node': ..., 'out_path': ...,
    'frame_range': ...}) -- the real dispatcher path, matching every other
    PP12 hython smoke.

STATICMETHOD HARNESS FIX (112c lesson, mandatory here too): dispatcher
functions assigned as class attrs in setup_class MUST be wrapped with
staticmethod(), otherwise `self` is injected as an extra positional arg when
called as `self._dispatch(...)` from an instance method, breaking the call
signature. See setup_class in every test class below.

Fixture (mirrors pp12-112b/c non-vacuous recipe -- orchestrator-confirmed):
  import hou
  sph = hou.node("/stage").createNode("sphere")   # a real Solaris LOP
  sph.cook(force=True)
  # drive sph.path() (e.g. '/stage/sphere1') through the /out `usd` ROP,
  # NOT '/stage' itself -- the ROP renders a specific LOP node's composed
  # stage via the `loppath` parm.

Acceptance tests covered (plan rev3 decomposition[hou-test].acceptanceTests):
  - dispatch('usd_export_rop', {'lop_node': <fixture>, 'out_path': '$HIP/x.usdc',
    'frame_range': None}) writes crate-magic bytes (PXR-USDC) via the /out
    `usd` ROP, re-opens in pxr WITH THE FIXTURE PRIM CONTENT PRESENT (plan-1
    no-silent-no-op -- assert real content, not just file existence), root has
    NO /World or /root, format==actual_format=='usdc', a validator_post
    postwrite verdict is present.
  - The same to '$HIP/x.usda' -> ascii header (#usda), format=='usda'.
  - frame_range=[1, 3] is honored (a range export, trange=1).
  - capability_of('usd_export_rop') == Capability.MUTATING AND it has a
    registered preview_fn with preview_required=True (mirrors usd_export_layer).
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap -- same pattern as usd_export_layer_hython_smoke.py
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_PKG_PYTHON = os.path.join(_REPO_ROOT, "python")
_HOUDINI_HANDLERS = os.path.join(_REPO_ROOT, "houdini", "scripts", "python")

for _p in (_REPO_ROOT, _PKG_PYTHON, _HOUDINI_HANDLERS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# pytest / hython compat shim
# ---------------------------------------------------------------------------
try:
    import pytest
    _PYTEST_AVAILABLE = True
except ImportError:
    _PYTEST_AVAILABLE = False

    class _FakePytest:
        """Minimal pytest shim so decorator syntax works under plain hython."""
        class mark:
            class parametrize:
                def __init__(self, *a, **kw): pass
                def __call__(self, f): return f
            asyncio = lambda f: f  # noqa: E731

        @staticmethod
        def skip(reason: str = ""):
            raise SystemExit(f"SKIP: {reason}")

    pytest = _FakePytest()

# ---------------------------------------------------------------------------
# Pass/fail counters (used when running directly under hython)
# ---------------------------------------------------------------------------
_PASS_COUNT = 0
_FAIL_COUNT = 0
_ERRORS: list[tuple[str, str]] = []


def _record_pass(name: str) -> None:
    global _PASS_COUNT
    _PASS_COUNT += 1
    print(f"  PASS  {name}")


def _record_fail(name: str, reason: str) -> None:
    global _FAIL_COUNT
    _FAIL_COUNT += 1
    _ERRORS.append((name, reason))
    print(f"  FAIL  {name}: {reason}")


# ---------------------------------------------------------------------------
# Dispatcher import (usd_export_handlers module already exists from
# pp12-112b/c; 'usd_export_rop' registration is the NEW red-gate surface for
# this file)
# ---------------------------------------------------------------------------
def _get_dispatcher_and_registry():
    """Import usd_export_handlers (triggers handler registration) and return
    (dispatch, capability_of, preview_of).

    Unlike a from-scratch module, the handlers module itself is expected to
    ALREADY exist (pp12-112b/c shipped it) -- the red signal for THIS file is
    the 'usd_export_rop' COMMAND being unregistered, which dispatch()
    surfaces as {"status": "error", "error": {"code": "UNKNOWN_COMMAND"}}
    rather than an ImportError. Tests below assert on that dispatch-level
    signal directly (see TestUsdExportRopRedGate).
    """
    import fxhoudinimcp_server.handlers.usd_export_handlers as _handlers  # noqa: F401
    from fxhoudinimcp_server.dispatcher import dispatch, capability_of, preview_of
    return dispatch, capability_of, preview_of


# ---------------------------------------------------------------------------
# Real cooked LOP fixture (mirrors pp12-112b/c non-vacuous recipe)
# ---------------------------------------------------------------------------

def _build_sphere_lop() -> str:
    """Create a real cooked sphere LOP in /stage and return its node path.

    Recipe (orchestrator-confirmed against live hython, pp12-112b/c):
        sph = hou.node("/stage").createNode("sphere")
        sph.cook(force=True)
        return sph.path()  # e.g. '/stage/sphere1'
    """
    import hou

    stage_net = hou.node("/stage")
    if stage_net is None:
        raise RuntimeError(
            "hou.node('/stage') returned None -- /stage LOP network is missing. "
            "This should not occur in a headless hython 21 session. "
            "Cannot build non-vacuous fixture."
        )
    sph = stage_net.createNode("sphere")
    sph.cook(force=True)
    return sph.path()


def _destroy_node(path: str) -> None:
    try:
        import hou
        node = hou.node(path)
        if node is not None:
            node.destroy()
    except Exception:
        pass


def _destroy_rop_leftovers() -> None:
    """Best-effort cleanup of any /out ROP the handler failed to destroy on
    an aborted test run (keeps repeated local runs from accumulating stale
    mcp_usd_rop_export nodes)."""
    try:
        import hou
        out_net = hou.node("/out")
        if out_net is None:
            return
        rop = out_net.node("mcp_usd_rop_export")
        if rop is not None:
            rop.destroy()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tiny result helpers (mirroring usd_export_layer_hython_smoke.py)
# ---------------------------------------------------------------------------

def _unwrap(result: dict) -> dict:
    """Unwrap the dispatcher envelope.

    The dispatcher wraps every handler result in:
        {'status': 'success'|'error', 'data': <handler_payload>, 'timing_ms': ...}
    The handler's real {ok, error, ...} dict is under result["data"].
    """
    if isinstance(result, dict) and "data" in result:
        return result["data"]
    return result


def _is_unknown_command(result: dict) -> bool:
    """True when dispatch() reports the command itself is unregistered (RED gate)."""
    return (
        isinstance(result, dict)
        and result.get("status") == "error"
        and isinstance(result.get("error"), dict)
        and result["error"].get("code") == "UNKNOWN_COMMAND"
    )


# ---------------------------------------------------------------------------
# Test class: URC-D1 -- RED gate: 'usd_export_rop' must be a registered command
# ---------------------------------------------------------------------------

class TestUsdExportRopRedGate:
    """URC-D1 (RED GATE): 'usd_export_rop' must be registered in the dispatcher.

    Before hou-dev implements the handler, dispatch('usd_export_rop', {...})
    returns {"status": "error", "error": {"code": "UNKNOWN_COMMAND", ...}} --
    this is the expected RED failure for this file. Once hou-dev registers
    the handler via register_handler('usd_export_rop', ...), this command
    resolves and the remaining test classes exercise real behavior.
    """

    @classmethod
    def setup_class(cls):
        _dispatch, _capability_of, _preview_of = _get_dispatcher_and_registry()
        # STATICMETHOD FIX (112c lesson): wrap with staticmethod() so `self`
        # is NOT injected as an extra positional arg when called via
        # `self._dispatch(...)` from an instance method below.
        cls._dispatch = staticmethod(_dispatch)
        cls._capability_of = staticmethod(_capability_of)
        cls._preview_of = staticmethod(_preview_of)

    def test_usd_export_rop_is_registered(self):
        """dispatch('usd_export_rop', ...) must NOT return UNKNOWN_COMMAND.

        FAILS RED (as intended) until hou-dev calls
        register_handler('usd_export_rop', usd_export_rop, Capability.MUTATING,
                          preview_fn=_preview_export_rop, preview_required=True).
        """
        result = self._dispatch("usd_export_rop", {
            "lop_node": "/dummy/probe",
            "out_path": "$HIP/probe.usdc",
            "frame_range": None,
        })
        assert not _is_unknown_command(result), (
            "'usd_export_rop' not found in the dispatcher registry. hou-dev must call "
            "register_handler('usd_export_rop', usd_export_rop, Capability.MUTATING, "
            "preview_fn=_preview_export_rop, preview_required=True) in "
            "usd_export_handlers.py. "
            f"Got dispatch result: {result!r}."
        )


# ---------------------------------------------------------------------------
# Test class: URC-D2 -- .usdc write via the /out `usd` ROP: crate magic,
#          no /World|/root, format match, REAL PRIM CONTENT (plan-1)
# ---------------------------------------------------------------------------

class TestUsdcRopExportWrite:
    """URC-D2: driving the /out `usd` ROP against a real cooked sphere LOP to
    a .usdc path must produce a genuine USD crate-binary file with no
    injected /World or /root wrapper, and the FIXTURE PRIM must actually be
    present in the written file (plan-1 no-silent-no-op guard -- a source-mode
    toggle silently gating loppath would otherwise produce a valid-but-empty
    file that passes a magic-bytes-only check).
    """

    @classmethod
    def setup_class(cls):
        _dispatch, _capability_of, _preview_of = _get_dispatcher_and_registry()
        cls._dispatch = staticmethod(_dispatch)
        cls._capability_of = staticmethod(_capability_of)
        cls._preview_of = staticmethod(_preview_of)
        cls._sphere_path = _build_sphere_lop()

        import hou
        cls._out_path_expr = "$HIP/pp12_112d_test.usdc"
        cls._out_path_expanded = hou.text.expandString(cls._out_path_expr)

        raw = cls._dispatch("usd_export_rop", {
            "lop_node": cls._sphere_path,
            "out_path": cls._out_path_expr,
            "frame_range": None,
        })
        cls._payload = _unwrap(raw)

    def test_export_reports_ok_true(self):
        """The export must succeed (ok=True) on a real cooked sphere LOP via the ROP."""
        assert self._payload.get("ok") is True, (
            f"usd_export_rop on a real cooked sphere LOP -> .usdc must return ok=True. "
            f"Got ok={self._payload.get('ok')!r}, error={self._payload.get('error')!r}."
        )

    def test_written_file_has_crate_magic_bytes(self):
        """The written file's first bytes must be the USD crate magic (PXR-USDC)."""
        assert os.path.exists(self._out_path_expanded), (
            f"Expected a written file at {self._out_path_expanded!r} but it does not exist."
        )
        with open(self._out_path_expanded, "rb") as fh:
            header = fh.read(16)
        assert header[:8] == b"PXR-USDC", (
            f"Written .usdc file must begin with the USD crate magic bytes b'PXR-USDC'. "
            f"Got header={header!r}."
        )

    def test_out_path_in_payload_matches_expanded_path(self):
        """The success payload's 'out_path' must match the $HIP-expanded path
        actually written to disk."""
        assert self._payload.get("out_path") == self._out_path_expanded, (
            f"usd_export_rop success payload must include 'out_path' matching "
            f"the $HIP-expanded write target. Expected {self._out_path_expanded!r}, "
            f"got {self._payload.get('out_path')!r}."
        )

    def test_file_reopens_in_pxr_with_fixture_prim_present_and_no_world_or_root(self):
        """PLAN-1 NO-SILENT-NO-OP (REQUIRED): the written file must re-open as
        a valid USD stage containing the FIXTURE PRIM (a 'sphere' or 'Sphere'
        typed prim, or at minimum a non-empty root-prim set derived from the
        cooked sphere LOP) -- not just a technically-valid-but-empty stage.
        A silently-gated loppath (an unconfirmed source-mode toggle) would
        write a valid, non-empty-bytes, empty-content file that a magic-bytes
        -only check would miss; re-opening and asserting real content is what
        catches that class of silent no-op.

        Also asserts NO /World or /root wrapper prim (usd-publish-discipline.md).
        """
        from pxr import Usd

        stage = Usd.Stage.Open(self._out_path_expanded)
        assert stage is not None, (
            f"Written file at {self._out_path_expanded!r} must re-open as a valid USD stage."
        )
        root_prim_paths = [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()]
        assert "/World" not in root_prim_paths, (
            f"Written layer must NOT contain a '/World' wrapper prim. "
            f"Got root prims={root_prim_paths!r}."
        )
        assert "/root" not in root_prim_paths, (
            f"Written layer must NOT contain a '/root' wrapper prim. "
            f"Got root prims={root_prim_paths!r}."
        )
        # PLAN-1: real content must be present -- the composed stage from the
        # cooked sphere LOP must have produced at least one real prim.
        assert len(root_prim_paths) > 0, (
            f"PLAN-1 no-silent-no-op: the written file has NO root prims at all "
            f"-- this indicates the ROP wrote an empty/near-empty stage (a "
            f"silently-gated loppath, or the ROP never actually rendered the "
            f"LOP's composed content). Expected at least one prim derived from "
            f"the cooked sphere LOP fixture. Got root_prim_paths={root_prim_paths!r}."
        )
        # A stronger content check: at least one prim in the FULL stage
        # traversal (not just root children) must be a defined, valid prim --
        # guards against a root prim list that is non-empty but whose prims
        # are all inactive/undefined placeholders.
        traversed = [str(p.GetPath()) for p in stage.Traverse()]
        assert len(traversed) > 0, (
            f"PLAN-1 no-silent-no-op: Usd.Stage.Traverse() over the written file "
            f"returned NO prims at all -- the stage has no real composed content. "
            f"Got traversed={traversed!r}."
        )
        # RED-3 (fixture-specific identity, REQUIRED strengthening): the
        # FIXTURE'S SPECIFIC prim path must exist and be valid in the
        # re-opened stage -- not merely "traversal is non-empty" (which
        # could technically be satisfied by an unrelated placeholder prim
        # under a different name than what the sphere LOP fixture actually
        # cooked). The sphere LOP conventionally cooks a prim whose leaf
        # name matches the source node's name (e.g. '/sphere1'); assert the
        # concrete fixture prim by name is present among the root children
        # AND that its GetPrimAtPath(...) resolves to a valid prim.
        fixture_leaf_name = self._sphere_path.rsplit("/", 1)[-1]
        fixture_root_candidates = [
            p for p in root_prim_paths if p.rsplit("/", 1)[-1] == fixture_leaf_name
        ]
        assert fixture_root_candidates, (
            f"RED-3: expected the fixture's specific prim (leaf name "
            f"{fixture_leaf_name!r}, derived from the cooked sphere LOP node "
            f"{self._sphere_path!r}) to appear among the written stage's root "
            f"prims -- not just SOME non-empty traversal. "
            f"Got root_prim_paths={root_prim_paths!r}."
        )
        fixture_prim = stage.GetPrimAtPath(fixture_root_candidates[0])
        assert fixture_prim.IsValid(), (
            f"RED-3: stage.GetPrimAtPath({fixture_root_candidates[0]!r}) (the "
            f"fixture's specific prim) must resolve to a VALID prim in the "
            f"re-opened stage."
        )

    def test_format_and_actual_format_are_usdc(self):
        """Both the extension-derived format and the magic-bytes-derived
        actual_format must report 'usdc' for a .usdc export (plan-7 equality
        gate)."""
        assert self._payload.get("format") == "usdc", (
            f"format (extension-derived) must be 'usdc'. Got {self._payload.get('format')!r}."
        )
        assert self._payload.get("actual_format") == "usdc", (
            f"actual_format (magic-bytes-derived, via format_from_magic_bytes reuse) "
            f"must be 'usdc'. Got {self._payload.get('actual_format')!r}."
        )
        assert self._payload.get("format") == self._payload.get("actual_format"), (
            f"format and actual_format must be EQUAL on a correctly-driven ROP export "
            f"(plan-7-cap gate): format={self._payload.get('format')!r}, "
            f"actual_format={self._payload.get('actual_format')!r}."
        )

    def test_validator_post_postwrite_verdict_present(self):
        """The handler must run an INLINE post-write usd_validate and embed its
        result under 'validator_post' with mode=='postwrite'."""
        validator_post = self._payload.get("validator_post")
        assert isinstance(validator_post, dict), (
            f"validator_post must be a dict (the inline post-write usd_validate result). "
            f"Got {type(validator_post)!r}: {validator_post!r}."
        )
        assert validator_post.get("mode") == "postwrite", (
            f"validator_post.mode must be 'postwrite' (out_path + actual_format both set). "
            f"Got mode={validator_post.get('mode')!r}."
        )
        assert "verdict" in validator_post, (
            f"validator_post must carry a 'verdict' key. Got keys={list(validator_post.keys())!r}."
        )

    @classmethod
    def teardown_class(cls):
        _destroy_node(cls._sphere_path)
        _destroy_rop_leftovers()
        try:
            if os.path.exists(cls._out_path_expanded):
                os.remove(cls._out_path_expanded)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test class: URC-D3 -- .usda write via the ROP: ascii header, format=='usda'
# ---------------------------------------------------------------------------

class TestUsdaRopExportWrite:
    """URC-D3: driving the /out `usd` ROP to a .usda path must produce an
    ASCII-format USD file (magic prefix b'#usda') and the handler must report
    format=='usda'.
    """

    @classmethod
    def setup_class(cls):
        _dispatch, _capability_of, _preview_of = _get_dispatcher_and_registry()
        cls._dispatch = staticmethod(_dispatch)
        cls._capability_of = staticmethod(_capability_of)
        cls._preview_of = staticmethod(_preview_of)
        cls._sphere_path = _build_sphere_lop()

        import hou
        cls._out_path_expr = "$HIP/pp12_112d_test.usda"
        cls._out_path_expanded = hou.text.expandString(cls._out_path_expr)

        raw = cls._dispatch("usd_export_rop", {
            "lop_node": cls._sphere_path,
            "out_path": cls._out_path_expr,
            "frame_range": None,
        })
        cls._payload = _unwrap(raw)

    def test_export_reports_ok_true(self):
        """The export must succeed (ok=True) on a real cooked sphere LOP -> .usda via the ROP."""
        assert self._payload.get("ok") is True, (
            f"usd_export_rop on a real cooked sphere LOP -> .usda must return ok=True. "
            f"Got ok={self._payload.get('ok')!r}, error={self._payload.get('error')!r}."
        )

    def test_written_file_has_ascii_header(self):
        """The written file's first bytes must be the USD ASCII magic (#usda)."""
        assert os.path.exists(self._out_path_expanded), (
            f"Expected a written file at {self._out_path_expanded!r} but it does not exist."
        )
        with open(self._out_path_expanded, "rb") as fh:
            header = fh.read(8)
        assert header[:5] == b"#usda", (
            f"Written .usda file must begin with the USD ASCII magic bytes b'#usda'. "
            f"Got header={header!r}."
        )

    def test_format_is_usda(self):
        """format must report 'usda' for a .usda export, and match actual_format."""
        assert self._payload.get("format") == "usda", (
            f"format (extension-derived) must be 'usda'. Got {self._payload.get('format')!r}."
        )
        assert self._payload.get("actual_format") == "usda", (
            f"actual_format (magic-bytes-derived) must be 'usda'. "
            f"Got {self._payload.get('actual_format')!r}."
        )

    def test_out_path_in_payload_matches_expanded_path(self):
        """The success payload's 'out_path' must match the $HIP-expanded path
        actually written to disk."""
        assert self._payload.get("out_path") == self._out_path_expanded, (
            f"usd_export_rop success payload must include 'out_path' matching "
            f"the $HIP-expanded write target. Expected {self._out_path_expanded!r}, "
            f"got {self._payload.get('out_path')!r}."
        )

    @classmethod
    def teardown_class(cls):
        _destroy_node(cls._sphere_path)
        _destroy_rop_leftovers()
        try:
            if os.path.exists(cls._out_path_expanded):
                os.remove(cls._out_path_expanded)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test class: URC-D4 -- frame_range=[1, 3] is honored (trange=1, a real range)
# ---------------------------------------------------------------------------

def _build_time_varying_sphere_lop():
    """Create a real cooked sphere LOP in /stage whose transform is
    TIME-VARYING (tx driven by an expression over $F) and return its node
    path. Used by TestFrameRangeExportWrite (red-1) to PROVE that
    frame_range=[1, 3] actually drove multiple frames through the ROP --
    a STATIC fixture would produce identical bytes whether range mode
    (trange=1) drove frames 1-3 or silently collapsed to the current frame,
    so a static fixture cannot distinguish the two. A time-varying xformOp
    can: it only gets multiple USD time samples if the ROP genuinely
    cooked+wrote more than one frame.

    Recipe: an xform LOP downstream of the sphere, with its translate X
    parameter set to the expression `$F` (varies every frame). Falls back
    to a plain sphere.setParm-based transform channel on the sphere itself
    if the `xform` LOP type or its translate parm name differs from
    expectation (defensive -- see the shape guard below).

    BUGFIX (pp12-112d red-1 fix, grounded against shipped Houdini 21.0.729
    HOM source at $HFS/houdini/python3.11libs/houpythonportion/{Parm,ParmTuple}.py
    and confirmed live via hython): hou.Parm ALWAYS has a public `.set()`
    method (so `hasattr(tx_parm, "set")` is unconditionally True for BOTH
    hou.Parm and hou.ParmTuple -- it does not discriminate the two), and
    hou.Parm has NO `__getitem__` at all (only hou.ParmTuple defines
    `__getitem__`). The original code's `hasattr(tx_parm, "set")` branch
    therefore always ran the ParmTuple-shaped `tx_parm[0].setExpression(...)`
    path even when `xform.parm("tx")` (a plain hou.Parm, confirmed live:
    `/stage/xform1.parm("tx")` resolves directly, no ParmTuple needed) was
    returned -- raising TypeError (not caught by the narrower
    `except AttributeError`), aborting the fixture builder uncaught.

    Fix: `xform.parm("tx")` already resolves to a real hou.Parm for the
    `xform` LOP type (verified live) -- call `.setExpression(...)` on it
    DIRECTLY, no indexing. The isinstance check below distinguishes
    hou.ParmTuple (needs component indexing) from hou.Parm (call directly)
    instead of the broken hasattr(..., "set") probe.
    """
    import hou

    stage_net = hou.node("/stage")
    if stage_net is None:
        raise RuntimeError(
            "hou.node('/stage') returned None -- /stage LOP network is missing. "
            "Cannot build the time-varying fixture."
        )
    sph = stage_net.createNode("sphere")
    xform = stage_net.createNode("xform")
    xform.setInput(0, sph)

    # Locate a translate-X parm on the xform LOP and drive it with $F so the
    # authored xformOp is genuinely time-varying across frames.
    tx_parm = xform.parm("tx") or xform.parm("t1") or xform.parmTuple("t")
    if tx_parm is None:
        raise RuntimeError(
            "Could not find a translate parm ('tx'/'t1'/'t') on the 'xform' LOP -- "
            "cannot build the time-varying fixture; Houdini's xform LOP parm "
            "naming may have changed."
        )
    if isinstance(tx_parm, hou.ParmTuple):
        # hou.ParmTuple: component indexing is valid -- drive the first
        # component (hou.Parm), which itself has no __getitem__.
        tx_parm[0].setExpression("$F", hou.exprLanguage.Hscript)
    else:
        # hou.Parm (the common case for xform's "tx"): call setExpression
        # DIRECTLY -- hou.Parm has no __getitem__, indexing it raises TypeError.
        tx_parm.setExpression("$F", hou.exprLanguage.Hscript)

    xform.cook(force=True)
    return xform.path()


class TestFrameRangeExportWrite:
    """URC-D4 (red-1 STRENGTHENED): a frame_range=[1, 3] request must be
    honored by the ROP (a real range export, trange=1) rather than
    silently collapsing to a current-frame-only export.

    A STATIC fixture cannot distinguish "range mode drove frames 1-3" from
    "range mode silently collapsed to the current frame" -- both produce
    byte-identical crate files. This class proves range mode genuinely
    drove multiple frames via a TIME-VARYING fixture (an xform LOP whose
    translate-X is driven by the $F expression): the written stage's
    time-varying attribute must carry MULTIPLE time samples, which only a
    real multi-frame cook can produce.

    As a second, independent proof (the plan's documented fallback for
    when animating a LOP fixture proves impractical), the SAME fixture is
    also exported at frame_range=None and the two outputs' time-sample
    counts are asserted to DIFFER -- a current-frame collapse would make
    them identical.
    """

    @classmethod
    def setup_class(cls):
        _dispatch, _capability_of, _preview_of = _get_dispatcher_and_registry()
        cls._dispatch = staticmethod(_dispatch)
        cls._capability_of = staticmethod(_capability_of)
        cls._preview_of = staticmethod(_preview_of)
        cls._fixture_path = _build_time_varying_sphere_lop()

        import hou
        cls._out_path_expr = "$HIP/pp12_112d_test_range.usdc"
        cls._out_path_expanded = hou.text.expandString(cls._out_path_expr)
        cls._out_path_current_expr = "$HIP/pp12_112d_test_range_current.usdc"
        cls._out_path_current_expanded = hou.text.expandString(cls._out_path_current_expr)

        raw = cls._dispatch("usd_export_rop", {
            "lop_node": cls._fixture_path,
            "out_path": cls._out_path_expr,
            "frame_range": [1, 3],
        })
        cls._payload = _unwrap(raw)

        # Second export of the SAME fixture at frame_range=None (current
        # frame only) -- the comparison baseline for the differs-from-range
        # assertion below.
        raw_current = cls._dispatch("usd_export_rop", {
            "lop_node": cls._fixture_path,
            "out_path": cls._out_path_current_expr,
            "frame_range": None,
        })
        cls._payload_current = _unwrap(raw_current)

    def test_export_reports_ok_true(self):
        """A frame_range=[1, 3] export must succeed (ok=True)."""
        assert self._payload.get("ok") is True, (
            f"usd_export_rop with frame_range=[1, 3] must return ok=True. "
            f"Got ok={self._payload.get('ok')!r}, error={self._payload.get('error')!r}."
        )

    def test_written_file_exists_with_crate_magic(self):
        """The ranged export must still produce a valid crate file (the
        format contract does not change for a range export)."""
        assert os.path.exists(self._out_path_expanded), (
            f"Expected a written file at {self._out_path_expanded!r} but it does not exist."
        )
        with open(self._out_path_expanded, "rb") as fh:
            header = fh.read(16)
        assert header[:8] == b"PXR-USDC", (
            f"Written .usdc file (frame_range export) must begin with the USD crate "
            f"magic bytes b'PXR-USDC'. Got header={header!r}."
        )

    def test_range_export_produces_multiple_time_samples(self):
        """RED-1 (REQUIRED, cap-round Major): the time-varying translate
        attribute in the frame_range=[1, 3]-exported stage must carry
        MULTIPLE time samples. A current-frame collapse (range mode
        silently ignored) can only ever produce a single (or zero) time
        sample -- this is the assertion a static fixture could never make,
        because it directly proves multiple frames were actually cooked
        and written by the ROP.
        """
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(self._out_path_expanded)
        assert stage is not None, (
            f"Ranged export file at {self._out_path_expanded!r} must re-open as a "
            f"valid USD stage."
        )
        xformable = None
        for prim in stage.Traverse():
            xformable = UsdGeom.Xformable(prim)
            if xformable and xformable.GetOrderedXformOps():
                break
            xformable = None
        assert xformable is not None, (
            "Expected at least one Xformable prim with an authored xformOp in the "
            "ranged-export stage -- cannot verify time-sample count without one. "
            f"Traversed prims: {[str(p.GetPath()) for p in stage.Traverse()]!r}."
        )
        num_samples = 0
        for op in xformable.GetOrderedXformOps():
            attr = op.GetAttr()
            num_samples = max(num_samples, attr.GetNumTimeSamples())
        assert num_samples > 1, (
            f"RED-1: frame_range=[1, 3] must produce a xformOp attribute with "
            f"MULTIPLE ({'>1'}) time samples -- got {num_samples} time sample(s). "
            f"A value of <=1 means the ROP silently collapsed to a single "
            f"(current) frame instead of honoring the requested range."
        )

    def test_range_export_differs_from_current_frame_export(self):
        """RED-1 (fallback proof): exporting the SAME time-varying fixture
        at frame_range=None (current frame) vs frame_range=[1, 3] (range)
        must produce a DIFFERENT time-sample count on the written xformOp
        attribute. If range mode silently collapsed to the current frame,
        the two outputs would be time-sample-identical."""
        from pxr import Usd, UsdGeom

        assert self._payload_current.get("ok") is True, (
            f"The current-frame comparison export must also succeed (ok=True). "
            f"Got ok={self._payload_current.get('ok')!r}, "
            f"error={self._payload_current.get('error')!r}."
        )

        def _max_time_samples(path):
            stage = Usd.Stage.Open(path)
            assert stage is not None, f"{path!r} must re-open as a valid USD stage."
            best = 0
            for prim in stage.Traverse():
                xformable = UsdGeom.Xformable(prim)
                if not xformable:
                    continue
                for op in xformable.GetOrderedXformOps():
                    best = max(best, op.GetAttr().GetNumTimeSamples())
            return best

        range_samples = _max_time_samples(self._out_path_expanded)
        current_samples = _max_time_samples(self._out_path_current_expanded)
        assert range_samples != current_samples, (
            f"RED-1 fallback proof: a frame_range=[1, 3] export and a "
            f"frame_range=None export of the IDENTICAL time-varying fixture must "
            f"differ in time-sample count (range={range_samples}, "
            f"current={current_samples}) -- identical counts would mean range "
            f"mode silently collapsed to the current frame."
        )

    @classmethod
    def teardown_class(cls):
        _destroy_node(cls._fixture_path)
        _destroy_rop_leftovers()
        for _p in (cls._out_path_expanded, cls._out_path_current_expanded):
            try:
                if os.path.exists(_p):
                    os.remove(_p)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Test class: URC-D5 -- Capability.MUTATING + registered preview_fn (109 gate)
# ---------------------------------------------------------------------------

class TestUsdExportRopGateRegistration:
    """URC-D5: 'usd_export_rop' must be registered as Capability.MUTATING
    WITH a preview_fn (preview_required=True) -- mirroring usd_export_layer's
    registration exactly.
    """

    @classmethod
    def setup_class(cls):
        _dispatch, _capability_of, _preview_of = _get_dispatcher_and_registry()
        cls._dispatch = staticmethod(_dispatch)
        cls._capability_of = staticmethod(_capability_of)
        cls._preview_of = staticmethod(_preview_of)

    def test_capability_is_mutating(self):
        """capability_of('usd_export_rop') must be Capability.MUTATING."""
        from fxhoudinimcp_server.dispatcher import Capability

        cap = self._capability_of("usd_export_rop")
        assert cap == Capability.MUTATING, (
            f"'usd_export_rop' must be registered with Capability.MUTATING "
            f"(it drives a ROP that writes a USD layer to disk). Got cap={cap!r}."
        )

    def test_preview_fn_is_registered_and_required(self):
        """preview_of('usd_export_rop') must have a non-None preview_fn with
        preview_required=True (mirrors usd_export_layer's registration)."""
        preview_record = self._preview_of("usd_export_rop")
        assert preview_record.get("preview_fn") is not None, (
            "'usd_export_rop' must register a preview_fn (_preview_export_rop) "
            "so the 109 gate can show a pre-flight preview before approval. "
            f"Got preview_record={preview_record!r}."
        )
        assert preview_record.get("preview_required") is True, (
            f"'usd_export_rop' must set preview_required=True (a preview_fn failure "
            f"must DENY the call, not silently queue without a preview). "
            f"Got preview_required={preview_record.get('preview_required')!r}."
        )

    def test_preview_fn_returns_approval_payload_dict_and_is_read_only(self):
        """Calling the registered preview_fn(params) directly must return a
        dict (the 109-gate approval payload), not raise, for a valid params
        dict -- and it must NOT write anything to disk (read-only)."""
        preview_record = self._preview_of("usd_export_rop")
        preview_fn = preview_record.get("preview_fn")
        assert preview_fn is not None, "preview_fn must be registered (see prior test)."

        sphere_path = _build_sphere_lop()
        try:
            import hou
            out_path_expr = "$HIP/pp12_112d_preview_test.usdc"
            expanded = hou.text.expandString(out_path_expr)

            # preview_fn is called POSITIONALLY with a single params dict --
            # NOT with **kwargs (the opposite convention from the handler,
            # matching _preview_export_layer's shape exactly).
            params = {
                "lop_node": sphere_path,
                "out_path": out_path_expr,
                "frame_range": None,
            }
            result = preview_fn(params)

            assert isinstance(result, dict), (
                f"preview_fn(params) must return a dict (the 109-gate approval payload). "
                f"Got {type(result)!r}: {result!r}."
            )

            assert result.get("out_path") == expanded, (
                f"preview_fn result 'out_path' must be the $HIP-expanded path "
                f"(hou.text.expandString(out_path)). Expected {expanded!r}, "
                f"got {result.get('out_path')!r}."
            )
            assert result.get("resolved_format") == "usdc", (
                f"resolved_format for a .usdc out_path must be 'usdc'. "
                f"Got {result.get('resolved_format')!r}."
            )
            assert "pre_validation" in result, (
                f"preview_fn result must include 'pre_validation' (the pre-flight "
                f"usd_validate reused in preflight mode). Got keys={list(result.keys())!r}."
            )
            assert isinstance(result["pre_validation"], dict), (
                f"pre_validation must be a dict. Got {type(result['pre_validation'])!r}."
            )
            assert result.get("frame_range") == params["frame_range"], (
                f"preview_fn result 'frame_range' must echo the request's frame_range "
                f"value ({params['frame_range']!r}). Got {result.get('frame_range')!r}."
            )
            assert result.get("no_world_wrapper") is True, (
                f"preview_fn result 'no_world_wrapper' must be True (the preview attests "
                f"no /World or /root wrapper will be injected). "
                f"Got {result.get('no_world_wrapper')!r}."
            )

            # The preview_fn must be genuinely READ-ONLY -- it must write
            # NOTHING at out_path (no ROP created, no file written). Assert
            # this BEFORE the finally cleanup so a preview that accidentally
            # writes the file is caught as a real failure, not silently
            # cleaned up first.
            assert not os.path.exists(expanded), (
                f"preview_fn must be READ-ONLY and must not write any file at "
                f"out_path. Found an unexpected file at {expanded!r} after calling "
                f"preview_fn -- the preview_fn must not perform the export."
            )
            # No ROP node should have been created by a read-only preview.
            import hou as _hou2
            out_net = _hou2.node("/out")
            leaked_rop = out_net.node("mcp_usd_rop_export") if out_net is not None else None
            assert leaked_rop is None, (
                f"preview_fn must be READ-ONLY and must not create a ROP node. "
                f"Found a leftover /out node: {leaked_rop.path() if leaked_rop else None!r}."
            )
        finally:
            _destroy_node(sphere_path)
            _destroy_rop_leftovers()
            try:
                import hou as _hou
                expanded = _hou.text.expandString(out_path_expr)
                if os.path.exists(expanded):
                    os.remove(expanded)
            except Exception:
                pass

    def test_preview_fn_denies_on_non_lop_or_uncooked_node(self):
        """FOLD plan-6/plan-9: the preview_fn must validate the SAME domain as
        the handler -- a node that is not a LOP node (no stage()), or a LOP
        node with no composed stage (uncooked), must cause the preview_fn to
        RAISE (which the gate interprets as DENY). This proves preview and
        handler agree on what is a valid target (they do NOT reuse
        _get_stage's file-open fallback branch -- the ROP renders a NODE)."""
        preview_record = self._preview_of("usd_export_rop")
        preview_fn = preview_record.get("preview_fn")
        assert preview_fn is not None, "preview_fn must be registered (see prior test)."

        raised = False
        try:
            preview_fn({
                "lop_node": "/definitely/not/a/real/node/path",
                "out_path": "$HIP/pp12_112d_deny_test.usdc",
                "frame_range": None,
            })
        except Exception:
            raised = True
        assert raised, (
            "preview_fn must RAISE (causing gate DENY) when lop_node does not "
            "resolve to a real Houdini node -- a file path or missing node must "
            "be rejected at preview time, not accepted and deferred to the handler."
        )


# ---------------------------------------------------------------------------
# Test class: URC-D6 -- no regression: existing usd_export family tools
#          (usd_inspect_layer, usd_validate READONLY; usd_export_layer
#          MUTATING) stay unchanged
# ---------------------------------------------------------------------------

class TestExistingToolsUnchanged:
    """URC-D6: usd_inspect_layer/usd_validate (pp12-112b, READONLY) and
    usd_export_layer (pp12-112c, MUTATING) must remain unchanged after the
    second gated sibling (usd_export_rop) is added. A regression here would
    mean the new registration accidentally clobbered or altered an existing
    registration.
    """

    @classmethod
    def setup_class(cls):
        _dispatch, _capability_of, _preview_of = _get_dispatcher_and_registry()
        cls._dispatch = staticmethod(_dispatch)
        cls._capability_of = staticmethod(_capability_of)
        cls._preview_of = staticmethod(_preview_of)

    def test_usd_inspect_layer_still_readonly(self):
        from fxhoudinimcp_server.dispatcher import Capability

        cap = self._capability_of("usd_inspect_layer")
        assert cap == Capability.READONLY, (
            f"'usd_inspect_layer' (pp12-112b) must remain Capability.READONLY. Got cap={cap!r}."
        )

    def test_usd_validate_still_readonly(self):
        from fxhoudinimcp_server.dispatcher import Capability

        cap = self._capability_of("usd_validate")
        assert cap == Capability.READONLY, (
            f"'usd_validate' (pp12-112b) must remain Capability.READONLY. Got cap={cap!r}."
        )

    def test_usd_export_layer_still_mutating_with_preview(self):
        from fxhoudinimcp_server.dispatcher import Capability

        cap = self._capability_of("usd_export_layer")
        assert cap == Capability.MUTATING, (
            f"'usd_export_layer' (pp12-112c) must remain Capability.MUTATING. Got cap={cap!r}."
        )
        preview_record = self._preview_of("usd_export_layer")
        assert preview_record.get("preview_fn") is not None, (
            "'usd_export_layer' (pp12-112c) must retain its registered preview_fn "
            "after the usd_export_rop sibling is added."
        )


# ---------------------------------------------------------------------------
# Test registry and runner (for direct hython execution)
# ---------------------------------------------------------------------------

_TEST_CLASSES = [
    TestUsdExportRopRedGate,
    TestUsdcRopExportWrite,
    TestUsdaRopExportWrite,
    TestFrameRangeExportWrite,
    TestUsdExportRopGateRegistration,
    TestExistingToolsUnchanged,
]


def _run_class(cls):
    """Instantiate the class and run all test_* methods; record pass/fail."""
    obj = cls()
    setup_failed = False

    if hasattr(cls, "setup_class"):
        try:
            cls.setup_class()
        except Exception as exc:
            _record_fail(f"{cls.__name__}.setup_class", repr(exc))
            setup_failed = True

    if setup_failed:
        return

    for name in dir(obj):
        if not name.startswith("test_"):
            continue
        method = getattr(obj, name)
        if not callable(method):
            continue
        full_name = f"{cls.__name__}.{name}"
        try:
            method()
            _record_pass(full_name)
        except SystemExit as exc:
            print(f"  SKIP  {full_name}: {exc}")
        except AssertionError as exc:
            _record_fail(full_name, str(exc))
        except Exception as exc:
            _record_fail(full_name, repr(exc))

    if hasattr(cls, "teardown_class"):
        try:
            cls.teardown_class()
        except Exception:
            pass


def main() -> int:
    print("=" * 72)
    print("usd_export_rop_hython_smoke.py -- pp12-112d")
    print("=" * 72)

    try:
        dispatch, capability_of, preview_of = _get_dispatcher_and_registry()
    except ImportError as exc:
        print(f"[mode] RED -- usd_export_handlers module itself is not importable: {exc}")
        print("Expected only if pp12-112b/c's module is somehow absent. "
              "For pp12-112d's own red gate, the module import succeeds but "
              "'usd_export_rop' is UNKNOWN_COMMAND -- see TestUsdExportRopRedGate.")
        return 1

    probe = dispatch("usd_export_rop", {
        "lop_node": "/dummy/probe", "out_path": "$HIP/probe.usdc",
        "frame_range": None,
    })
    if _is_unknown_command(probe):
        print("[mode] RED -- 'usd_export_rop' not yet registered (expected before hou-dev).")
        print("Running TestUsdExportRopRedGate only (the other classes require the real handler).")
        _run_class(TestUsdExportRopRedGate)
        print("\n" + "=" * 72)
        print(f"Results: {_PASS_COUNT} passed, {_FAIL_COUNT} failed")
        return 1  # RED gate confirmed -- exit non-zero

    print("[mode] GREEN -- 'usd_export_rop' registered; running all tests.")
    for cls in _TEST_CLASSES:
        print(f"\n--- {cls.__name__} ---")
        _run_class(cls)

    print("\n" + "=" * 72)
    print(f"Results: {_PASS_COUNT} passed, {_FAIL_COUNT} failed")

    if _ERRORS:
        print("\nFailures:")
        for name, reason in _ERRORS:
            print(f"  FAIL  {name}")
            print(f"        {reason}")

    return 0 if _FAIL_COUNT == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
