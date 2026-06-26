"""kinefx_gate_smoke.py — RED gate-integration hython-smoke for pp12-110g.

Covers the §4.3 approval payload + FR-12 verify-after-mutate contract for
the 5 shipped gated KineFX character tools.

Representative tool: apply_secondarymotion
  - Simplest fixture: needs only a skeleton SOP at input 0
  - The REAL handler in character_handlers.py is driven — NO stub re-registration
  - RED signals (pre-PR-8):
      check-1: "§4.3 preview missing: 'preview' key absent — real handler has no
               preview_fn registered (character_handlers.py:1032 has no preview_fn kwarg)"
      check-2: ImportError: cannot import name 'node_plan' from
               'fxhoudinimcp.kinefx_model' (+ preview missing from queue response)
      check-3: "FR-12 MISSING: approved apply_secondarymotion result has no
               'validator' key. Full approved_data: {'ok': True, 'node': '...',
               'affected_joints': N, 'frame_range': [s, e]}"
      check-4: "FR-12 APPROVE-PATH: apply_secondarymotion approved return missing
               'validator' key, OR ok=False (cook failed headlessly), OR
               validator.skeleton_summary.count == 0 (fake count, not real re-query)"
               This is the REAL approve->cook->verify-after-mutate path.

Anti-mirror-test discipline (tdd-with-agents.md §2):
  - This smoke imports fxhoudinimcp_server.handlers.character_handlers to trigger
    REAL module-level handler registrations.  It does NOT call register_handler()
    itself at any point.
  - check-2 asserts that preview["node_plan"] equals node_plan() imported from
    kinefx_model (the pure model) — not a hardcoded dict.
  - check-2 asserts preview["skeleton_summary"]["count"] against the REAL fixture
    via query_skeleton(), not a hardcoded {"joints": 3}.
  - check-3 asserts that validator["skeleton_summary"]["count"] > 0 (real
    post-cook re-query) and note == "verify-after-mutate" — a static placeholder
    dict cannot satisfy both conditions simultaneously without real query data.
  - check-4 APPROVES the real handler and asserts the REAL post-cook re-query
    in the approved RETURN.  A static {note: "verify-after-mutate"} in the handler
    return that lacks a real skeleton_summary.count > 0 cannot satisfy this check.

hou-dev PR-8 must:
  (a) Add a real preview_fn to each of the 5 mutating register_handler calls in
      character_handlers.py — the preview_fn must call query_skeleton on the
      source node and import node_plan from kinefx_model to build the §4.3 payload.
  (b) Add a real validator sub-dict to apply_secondarymotion (and the other 4
      mutating handlers) that embeds a post-cook re-query of query_skeleton on the
      result node — NOT a static {note: "verify-after-mutate"} placeholder.
  (c) Add node_plan(tool, args) -> dict to kinefx_model.py (pure, hou-free, CL-015).

Run:
    "C:/Program Files/Side Effects Software/Houdini 21.0.729/bin/hython.exe" ^
        C:/Users/husma/development/fxhoudinimcp/tests/kinefx_gate_smoke.py

Exit 0 = all checks PASSED.
Exit 1 = one or more checks FAILED (expected in RED phase).

Verification surface: hython-smoke
TDD phase: RED (strengthened pp12-110g) — drives REAL handler, no stubs.
Author: hou-test (pp12-110g, session 3 — restored check-4 per task instructions)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# sys.path: Homedini pure core first, then the fork server package.
# Mirrors export_handlers.py sys.path bootstrap pattern.
# ---------------------------------------------------------------------------
sys.path.insert(0, "C:/Users/husma/development/HoudiniUtilTools/scripts/python")
sys.path.insert(0, "C:/Users/husma/development/fxhoudinimcp/houdini/scripts/python")

# The gate must already exist (pp12-109 landed).
from fxhoudinimcp_server.gate import install_gate

import hou
import fxhoudinimcp_server.dispatcher as _dispatcher

# ---------------------------------------------------------------------------
# Import the REAL character_handlers module.
#
# This single import triggers ALL module-level register_handler() calls in
# character_handlers.py (lines 1025-1032), which registers the REAL handlers.
# The smoke NEVER calls register_handler() itself — that is the core
# anti-mirror discipline.  If we re-registered stubs here, check-3 could
# never go green (the stub lacks validator, not the real handler).
# ---------------------------------------------------------------------------
import fxhoudinimcp_server.handlers.character_handlers as _char_handlers  # noqa: F401

# ---------------------------------------------------------------------------
# Fixture: build a minimal kinefx skeleton in a fresh /obj network.
#
# A single `skeleton` SOP (or a fallback `line`) produces a valid KineFX
# point skeleton sufficient for apply_secondarymotion (input 0 = skeleton).
#
# query_skeleton() must be called on the DIRECT SOP node path
# (mcp_test_skel), not the geo container (/obj/kinefx_gate_test).
# ---------------------------------------------------------------------------

def _build_skeleton_fixture() -> tuple[hou.Node, str]:
    """Create a minimal /obj geo node with a skeleton SOP populated with named joints.

    Approach (a): injects a programmatic hou.Geometry() with a @name point string
    attribute into the skeleton SOP stash parameter.  A bare skeleton SOP has an
    empty default stash in hython headless mode (0 points, no @name attribute),
    so query_skeleton() would raise ValueError without this injection.

    Returns (skel_sop_node, skel_sop_path).
    The SOP path is what query_skeleton() expects as node_path.
    """
    _JOINT_NAMES = ("hip", "spine", "chest")

    obj_net = hou.node("/obj")
    old = obj_net.node("kinefx_gate_test")
    if old:
        old.destroy()

    geo = obj_net.createNode("geo", "kinefx_gate_test")

    try:
        skel = geo.createNode("skeleton", "mcp_test_skel")
    except hou.OperationFailed:
        # `skeleton` SOP type unavailable — fall back to a null SOP.
        # We inject the geometry directly via the stash parm in either case.
        skel = geo.createNode("null", "mcp_test_skel")

    # Build a minimal in-memory geometry with named KineFX joints.
    # Approach (a): populate points + @name attribute, then inject into the stash.
    # NOTE: hou.Geometry.createPoints(int) is NOT a valid HOM overload in H21 —
    # the method only accepts a list of Vector3 or list-of-lists.  Use the
    # singular createPoint() in a loop instead.
    fake_geo = hou.Geometry()
    # Add the @name string point attribute (default "").
    fake_geo.addAttrib(hou.attribType.Point, "name", "")
    # Create one point per joint, set position and name.
    for i, jname in enumerate(_JOINT_NAMES):
        pt = fake_geo.createPoint()
        pt.setAttribValue("name", jname)
        pt.setPosition(hou.Vector3(0.0, float(i), 0.0))

    # Inject into the skeleton SOP stash parameter.
    # hou.Parm.set() accepts a hou.Geometry object for stash-type parms.
    try:
        skel.parm("stash").set(fake_geo)
    except (hou.OperationFailed, AttributeError):
        # If the parm is absent (e.g. on the null fallback) we leave the node
        # as-is; check-2 will still fail loud rather than silently pass.
        pass

    skel.setDisplayFlag(True)
    # Force a cook so query_skeleton can read geometry immediately.
    try:
        skel.cook(force=True)
    except Exception:
        pass

    geo.layoutChildren()
    skel_path = skel.path()  # e.g. /obj/kinefx_gate_test/mcp_test_skel
    return skel, skel_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prefs(mode: str) -> str:
    """Write a fresh temp prefs dir with mcp_gate.json for the given mode."""
    prefs = tempfile.mkdtemp(prefix="kinefx_gate_test_")
    gate_dir = os.path.join(prefs, "mcp_gate")
    os.makedirs(gate_dir, exist_ok=True)
    cfg_path = os.path.join(gate_dir, "mcp_gate.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump({"mode": mode}, fh)
    return prefs


def _reset_gate() -> None:
    """Tear down any installed gate + dispatcher state before each check."""
    if hasattr(hou.session, "_fxhoudinimcp_gate"):
        delattr(hou.session, "_fxhoudinimcp_gate")
    orig = getattr(_dispatcher, "_ORIGINAL_DISPATCH", None)
    if orig is not None and callable(orig):
        _dispatcher.dispatch = orig
    if hasattr(_dispatcher.dispatch, "_is_gated"):
        try:
            del _dispatcher.dispatch._is_gated
        except (AttributeError, TypeError):
            pass
    if hasattr(hou.session, "_fxhoudinimcp_preview_registry"):
        delattr(hou.session, "_fxhoudinimcp_preview_registry")


def _install(mode: str) -> str:
    """Reset, write prefs, set env var, and install the gate. Returns prefs dir."""
    _reset_gate()
    prefs = _make_prefs(mode)
    os.environ["HOUDINI_USER_PREF_DIR"] = prefs
    install_gate()
    return prefs


def _dispatch(cmd: str, params: dict | None = None) -> dict:
    """Dispatch a command through the (potentially gated) dispatcher."""
    return _dispatcher.dispatch(cmd, params or {})


def _data(r: dict) -> dict:
    """Unwrap dispatcher envelope: {status, data, timing_ms} -> data dict."""
    return r.get("data", r)


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Results runner
# ---------------------------------------------------------------------------

RESULTS: list[tuple[str, bool, str]] = []


def _run(label: str, fn) -> None:
    try:
        fn()
        RESULTS.append((label, True, ""))
        print(f"  PASS  {label}")
    except Exception as exc:
        RESULTS.append((label, False, str(exc)))
        print(f"  FAIL  {label}: {exc}")


# ===========================================================================
# check-1: dispatch the REAL apply_secondarymotion in propose mode and
#          assert that the queue response contains a 'preview' key.
#
# RED today: character_handlers.py:1032 registers apply_secondarymotion with
#   register_handler("apply_secondarymotion", apply_secondarymotion,
#                    Capability.MUTATING)
#   — NO preview_fn kwarg, so the gate never invokes a preview_fn and
#   the queue response has no 'preview' key.
#
# GREEN after PR-8: the registration adds preview_fn=_preview_secondarymotion,
#   the gate invokes it, and the queue response carries 'preview'.
#
# ANTI-MIRROR: this check NEVER calls register_handler().  The real handler
#   is already registered by the character_handlers module import above.
# ===========================================================================

def _check1_preview_key_in_queue_response():
    """check-1: REAL apply_secondarymotion dispatch in propose mode has 'preview'.

    RED failure:
        §4.3 preview missing from queue response: 'preview' key absent —
        real handler has no preview_fn registered
        (character_handlers.py:1032 has no preview_fn kwarg)
    """
    _install("propose")

    skel_sop, skel_sop_path = _build_skeleton_fixture()

    # Dispatch the REAL registered command through the gated dispatcher.
    # No register_handler() call here — the real handler is already registered.
    # Handler signature: apply_secondarymotion(node, joints=None, params=None, dest=None)
    # 'effect' is a kinefx::secondarymotion parm name — it goes inside 'params'.
    dispatch_args = {
        "node": skel_sop_path,
        "params": {"effect": 0.5},
    }
    r = _dispatch("apply_secondarymotion", dispatch_args)

    # Must queue in propose mode.
    _assert(
        r.get("gate") == "queued",
        f"expected gate=queued, got: {r!r}",
    )

    pending_id = r.get("pending_id")
    _assert(
        pending_id is not None,
        f"no pending_id in queued response: {r!r}",
    )

    # The gate must surface a 'preview' key — this requires a real preview_fn
    # to be registered on the handler.  RED: no preview_fn → no 'preview'.
    _assert(
        "preview" in r,
        (
            f"§4.3 preview missing from queue response: 'preview' key absent — "
            f"real handler has no preview_fn registered "
            f"(character_handlers.py:1032 has no preview_fn kwarg). "
            f"Full queue response: {r!r}"
        ),
    )


# ===========================================================================
# check-2: §4.3 payload content — node_plan from kinefx_model AND real
#          skeleton_summary from query_skeleton on the fixture.
#
# RED signals:
#   (a) ImportError: cannot import name 'node_plan' from
#       'fxhoudinimcp.kinefx_model'  (node_plan not yet in kinefx_model.py)
#   (b) 'preview' key absent from queue response (check-1 RED still applies)
#   (c) preview["node_plan"] != node_plan("apply_secondarymotion", dispatch_args)
#   (d) preview["skeleton_summary"]["count"] == 0 (hardcoded placeholder instead
#       of real query_skeleton call on the fixture)
#
# GREEN after PR-8:
#   - node_plan exists in kinefx_model.py (hou-free, CL-015)
#   - preview["node_plan"] == node_plan("apply_secondarymotion", dispatch_args)
#   - preview["skeleton_summary"]["count"] > 0 (real fixture skeleton data)
#
# ANTI-MIRROR: skeleton_summary is asserted against the REAL query_skeleton()
#   return value on the fixture — not a hardcoded {"joints": 3}.
#   node_plan is asserted against the REAL import from kinefx_model — not
#   a hardcoded dict.
# ===========================================================================

def _check2_payload_matches_real_model_and_real_fixture():
    """check-2: §4.3 payload contains REAL node_plan + REAL skeleton_summary.

    RED failures (all apply simultaneously):
        ImportError: cannot import name 'node_plan' from 'fxhoudinimcp.kinefx_model'
        §4.3 preview missing from queue response (no preview_fn on real handler)
    """
    _install("propose")

    skel_sop, skel_sop_path = _build_skeleton_fixture()

    # Import node_plan from the pure model layer.
    # RED: this raises ImportError until hou-dev adds node_plan to kinefx_model.py.
    # node_plan must be hou-free (CL-015): no import hou anywhere in kinefx_model.py.
    from fxhoudinimcp.kinefx_model import node_plan  # RED: ImportError today

    # Build the expected node_plan from the pure model.
    # Handler signature: apply_secondarymotion(node, joints=None, params=None, dest=None)
    dispatch_args = {
        "node": skel_sop_path,
        "params": {"effect": 0.5},
    }
    expected_node_plan = node_plan("apply_secondarymotion", dispatch_args)

    # The expected skeleton_summary comes from the REAL query_skeleton call.
    # This is a READONLY call against the fixture SOP, not a hardcoded literal.
    qs_result = _char_handlers.query_skeleton(node_path=skel_sop_path)
    expected_skeleton_count = qs_result.get("count", 0)
    _assert(
        expected_skeleton_count > 0,
        f"Fixture skeleton has 0 joints — fixture build may have failed. "
        f"query_skeleton returned: {qs_result!r}",
    )

    # Dispatch the REAL handler.
    r = _dispatch("apply_secondarymotion", dispatch_args)
    _assert(
        r.get("gate") == "queued",
        f"expected gate=queued, got: {r!r}",
    )

    # 'preview' must be present (depends on check-1 being GREEN).
    _assert(
        "preview" in r,
        (
            f"§4.3 preview missing from queue response: cannot assert payload. "
            f"Full queue response: {r!r}"
        ),
    )
    preview = r["preview"]

    # Assert §4.3 payload shape.
    for key in ("tool", "args", "node_plan", "skeleton_summary"):
        _assert(
            key in preview,
            f"§4.3 payload missing '{key}' key in preview; got keys: {list(preview.keys())}",
        )

    _assert(
        preview["tool"] == "apply_secondarymotion",
        f"preview.tool should be 'apply_secondarymotion', got: {preview['tool']!r}",
    )

    # node_plan must match the pure model's output exactly.
    # This fails if hou-dev hardcodes the dict instead of importing node_plan.
    _assert(
        preview["node_plan"] == expected_node_plan,
        (
            f"preview.node_plan does not match kinefx_model.node_plan() output.\n"
            f"  expected: {expected_node_plan!r}\n"
            f"  got:      {preview['node_plan']!r}"
        ),
    )

    # skeleton_summary must reflect REAL fixture data — not a hardcoded literal.
    skel_summary = preview.get("skeleton_summary", {})
    _assert(
        isinstance(skel_summary, dict),
        f"preview.skeleton_summary must be a dict, got: {type(skel_summary)!r}",
    )
    actual_count = skel_summary.get("count", 0)
    _assert(
        actual_count > 0,
        (
            f"preview.skeleton_summary.count must be > 0 (real fixture joint data). "
            f"A hardcoded placeholder or zero-count summary indicates the preview_fn "
            f"is NOT calling query_skeleton on the fixture. "
            f"Got skeleton_summary: {skel_summary!r}; "
            f"fixture query_skeleton returned count={expected_skeleton_count}"
        ),
    )

    # §4.3 preview must ALSO appear in list_pending_calls entry.
    pending_id = r.get("pending_id")
    list_r = _dispatch("gate.list_pending_calls", {})
    entries = _data(list_r).get("pending", [])
    matched = [e for e in entries if e.get("id") == pending_id]
    _assert(
        len(matched) == 1,
        f"pending_id {pending_id!r} not found in list_pending_calls: {entries!r}",
    )
    entry_preview = matched[0].get("preview", {})
    _assert(
        "node_plan" in entry_preview,
        f"§4.3 payload missing 'node_plan' in list_pending_calls entry: {matched[0]!r}",
    )
    _assert(
        "skeleton_summary" in entry_preview,
        f"§4.3 payload missing 'skeleton_summary' in list_pending_calls entry: {matched[0]!r}",
    )
    entry_count = entry_preview.get("skeleton_summary", {}).get("count", 0)
    _assert(
        entry_count > 0,
        (
            f"list_pending_calls entry skeleton_summary.count must be > 0. "
            f"Got: {entry_preview.get('skeleton_summary')!r}"
        ),
    )


# ===========================================================================
# check-3: FR-12 verify-after-mutate — assert that the preview payload
#          declares a 'validator_contract' key, signalling the handler will
#          carry the post-cook re-query in the approved return.
#
# FR-12 contract (pp12-110 spec §7.3 + FR-12):
#   Approved result must carry:
#     {
#       ok: true,
#       node: <path>,
#       affected_joints: <int>,
#       frame_range: [s, e],
#       validator: {
#         skeleton_summary: {"count": N, "joints": [...]},  # real post-cook re-query
#         note: "verify-after-mutate"
#       }
#     }
#
# Headless hython note: kinefx::secondarymotion cannot cook headlessly in hython
#   because it requires a full Houdini session for motion-clip evaluation.
#   This check asserts STRUCTURAL DECLARATION in the PREVIEW payload — the
#   'validator_contract' key that signals the approved return will carry
#   'validator'.  The actual post-cook re-query data is verified by check-4
#   (if cook succeeds headlessly) or on the operator-smoke rung (Hamza in
#   interactive Houdini, if cook fails headlessly).
#
# GREEN after PR-8 (hython-smoke rung):
#   The preview payload contains 'validator_contract' key listing the fields
#   the handler will populate in the approved return.
#
# ANTI-MIRROR: the smoke drives the REAL handler (registered by the
#   character_handlers module import).  It does NOT re-register a stub.
#   The 'validator_contract' must come from the REAL preview_fn.
#
# FR-12 RED signal (hython-smoke):
#   - 'preview' missing from queue response (no preview_fn on real handler)
#   - 'validator_contract' missing from preview (preview_fn doesn't declare
#      the validator shape)
# ===========================================================================

def _check3_fr12_validator_contract_in_preview():
    """check-3: FR-12 validator_contract declared in preview payload.

    The hython-smoke rung cannot run the full cook (kinefx::secondarymotion
    requires interactive Houdini).  Instead this check asserts that the
    §4.3 preview payload declares the 'validator_contract' key — the structural
    signal that hou-dev has wired up the post-cook re-query in the handler.

    The operator-smoke rung (Hamza in interactive Houdini) verifies that the
    actual approved return carries validator.skeleton_summary with real data.

    RED failure:
        §4.3 preview missing from queue response: 'preview' key absent —
        real handler has no preview_fn registered.
        (Even if preview is present, 'validator_contract' absence means
        hou-dev has not declared the FR-12 post-cook re-query contract.)

    hou-dev PR-8 must include 'validator_contract' in the preview_fn return:
        {
          "tool": "apply_secondarymotion",
          "args": <params>,
          "node_plan": node_plan("apply_secondarymotion", params),
          "skeleton_summary": <query_skeleton result on source node>,
          "validator_contract": {
              "fields": ["skeleton_summary", "note"],
              "note": "verify-after-mutate"
          }
        }
    """
    _install("propose")

    skel_sop, skel_sop_path = _build_skeleton_fixture()

    # Dispatch the REAL apply_secondarymotion in propose mode.
    # No register_handler() call — the real handler is already registered.
    dispatch_args = {
        "node": skel_sop_path,
        "params": {"effect": 0.5},
    }
    r = _dispatch("apply_secondarymotion", dispatch_args)
    _assert(
        r.get("gate") == "queued",
        f"expected gate=queued, got: {r!r}",
    )

    # 'preview' must be present — check-1 GREEN is a prerequisite.
    _assert(
        "preview" in r,
        (
            f"FR-12 check-3 prerequisite FAILED: 'preview' absent from queue response. "
            f"check-1 must pass first (preview_fn on real handler). "
            f"Full queue response: {r!r}"
        ),
    )
    preview = r["preview"]

    # FR-12 structural contract: the preview_fn must declare 'validator_contract'
    # in its return dict, signalling that the handler will call query_skeleton
    # on the cooked result node.  A static placeholder {note: "verify-after-mutate"}
    # does NOT include 'validator_contract' — this assertion catches it.
    _assert(
        "validator_contract" in preview,
        (
            f"FR-12 MISSING validator_contract in preview: hou-dev must include "
            f"'validator_contract' in the preview_fn return for apply_secondarymotion. "
            f"This declares the post-cook re-query contract to the gate client. "
            f"A static {{note: 'verify-after-mutate'}} in the handler return is NOT "
            f"sufficient — the contract must be declared in the preview payload. "
            f"Full preview: {preview!r}"
        ),
    )
    vc = preview["validator_contract"]
    _assert(
        isinstance(vc, dict),
        f"FR-12: validator_contract must be a dict, got: {type(vc)!r}",
    )
    _assert(
        vc.get("note") == "verify-after-mutate",
        (
            f"FR-12: validator_contract.note must be 'verify-after-mutate', "
            f"got: {vc.get('note')!r}. Full validator_contract: {vc!r}"
        ),
    )
    # The contract must declare that skeleton_summary will be in the validator.
    declared_fields = vc.get("fields", [])
    _assert(
        "skeleton_summary" in declared_fields,
        (
            f"FR-12: validator_contract.fields must include 'skeleton_summary', "
            f"indicating the handler will re-query query_skeleton post-cook. "
            f"Got fields: {declared_fields!r}"
        ),
    )


# ===========================================================================
# check-4: FR-12 APPROVE-PATH verify-after-mutate — the REAL approve->cook
#          path.  Approves the pending call and asserts that the approved
#          RETURN carries:
#            result["ok"] is True
#            result["validator"]["skeleton_summary"]["count"] > 0
#            result["validator"]["note"] == "verify-after-mutate"
#
# This is the check that was silently weakened in session 2 (see audit-lessons-
# loaded.md for the post-mortem).  check-3 asserts the DECLARATION (the
# validator_contract in the preview payload) but NOT the actual approved return.
# check-4 exercises the FULL path: queue -> approve -> cook -> re-query.
#
# FAIL paths and what they mean:
#
#   (A) ok=False in approved return:
#       kinefx::secondarymotion could not cook headlessly.  This is a KNOWN
#       headless limitation (hython lacks the motion-clip evaluation engine
#       available in interactive Houdini).  If this is the failure, the check
#       FAILS with an informative escalation message rather than silently
#       dropping the assertion.  The FR-12 approve-path return must be verified
#       on the operator-smoke rung (Hamza in interactive Houdini).
#
#   (B) 'validator' absent from approved return:
#       hou-dev has NOT added the FR-12 verify-after-mutate block to the handler.
#       RED signal pre-PR-8.
#
#   (C) validator.skeleton_summary.count == 0 or absent:
#       The post-cook re-query returned 0 joints (the handler used a fake/static
#       count instead of calling query_skeleton on the cooked result node).
#
#   (D) validator.note != "verify-after-mutate":
#       The validator dict does not carry the expected note field.
#
# ANTI-MIRROR: does NOT assert the internal call sequence (which hou.node
#   was created, which parm was set).  Asserts ONLY the public output: the
#   approved RETURN dict's validator.skeleton_summary.count > 0.
#
# NOTE ON HEADLESS COOK:
#   kinefx::secondarymotion typically fails cook in hython because it needs
#   the Houdini viewport / motion-clip evaluation context.  If ok=False is
#   returned, this check fails with an explicit escalation message referencing
#   the cook error.  The orchestrator must then route the FR-12-return
#   verification to the operator-smoke rung EXPLICITLY (not silently dropped).
#
# VERIFICATION SURFACE: hython-smoke (agent-runnable)
#   If kinefx::secondarymotion genuinely cannot cook headlessly and this check
#   always fails at (A), the orchestrator MUST escalate to operator-smoke for
#   the FR-12 approve-path return (NOT silently demote check-4 to a weaker
#   assertion).  See tdd-with-agents.md §6 re-classification mid-story.
# ===========================================================================

def _check4_fr12_approve_path_real_validator_return():
    """check-4: FR-12 APPROVE-PATH — approve the pending call and assert
    the REAL approved RETURN carries validator.skeleton_summary.count > 0.

    RED failures:
        (A) ok=False — kinefx::secondarymotion cook failed headlessly.
            ESCALATION: "FR-12-return must be verified on operator-smoke rung."
        (B) 'validator' absent from approved return — handler not yet FR-12 patched.
        (C) validator.skeleton_summary.count == 0 — fake count, not real re-query.
        (D) validator.note != "verify-after-mutate".
    """
    _install("propose")

    skel_sop, skel_sop_path = _build_skeleton_fixture()

    # Dispatch the REAL apply_secondarymotion in propose mode.
    # No register_handler() call — the real handler is already registered.
    dispatch_args = {
        "node": skel_sop_path,
        "params": {"effect": 0.5},
    }
    r = _dispatch("apply_secondarymotion", dispatch_args)
    _assert(
        r.get("gate") == "queued",
        f"check-4 prerequisite FAILED: expected gate=queued, got: {r!r}",
    )
    pending_id = r.get("pending_id")
    _assert(
        pending_id is not None,
        f"check-4 prerequisite FAILED: no pending_id in queued response: {r!r}",
    )

    # Approve the pending call via the REAL gate.approve_pending_call handler.
    # This runs the thunk (lambda h=handler, p=captured: h(**p)) synchronously,
    # which calls apply_secondarymotion(**dispatch_args) and cooks
    # kinefx::secondarymotion on the fixture skeleton.
    #
    # The gate wraps the handler result:
    #   {gate="approved", status="success", data={ok, node, affected_joints,
    #    frame_range, validator}, timing_ms}
    # _data() unwraps "data" -> the handler's dict.
    approve_r = _dispatch("gate.approve_pending_call", {"pending_id": pending_id})
    approved_data = _data(approve_r)

    # ── (A) ok check ─────────────────────────────────────────────────────────
    ok_val = approved_data.get("ok")
    if ok_val is False:
        cook_error = approved_data.get("error", "<no error key>")
        raise AssertionError(
            f"FR-12 APPROVE-PATH ESCALATION: apply_secondarymotion ok=False after "
            f"approve — kinefx::secondarymotion cook failed headlessly.\n"
            f"  Cook error: {cook_error!r}\n"
            f"  Full approved_data: {approved_data!r}\n"
            f"\n"
            f"  REQUIRED ACTION: the FR-12 approve-path return (result[\"validator\"])"
            f" cannot be verified in hython-smoke because the node cook requires"
            f" interactive Houdini.  The orchestrator MUST route the FR-12-return"
            f" verification to the operator-smoke rung EXPLICITLY — do NOT silently"
            f" weaken check-4 to a declaration-only check.  See tdd-with-agents.md"
            f" §6 (re-classification mid-story) and the audit-lessons-loaded.md"
            f" post-mortem for this unit."
        )
    _assert(
        ok_val is True,
        (
            f"FR-12 APPROVE-PATH: approved return ok must be True, got: {ok_val!r}. "
            f"Full approved_data: {approved_data!r}"
        ),
    )

    # ── (B) validator key present ────────────────────────────────────────────
    _assert(
        "validator" in approved_data,
        (
            f"FR-12 APPROVE-PATH: approved apply_secondarymotion result has no "
            f"'validator' key — hou-dev has not added the FR-12 verify-after-mutate "
            f"block to the handler (character_handlers.py). "
            f"Full approved_data: {approved_data!r}"
        ),
    )
    validator = approved_data["validator"]
    _assert(
        isinstance(validator, dict),
        f"FR-12 APPROVE-PATH: validator must be a dict, got: {type(validator)!r}",
    )

    # ── (C) validator.skeleton_summary.count > 0 ────────────────────────────
    # A static placeholder {note: "verify-after-mutate"} lacks skeleton_summary.
    # The masking-fix discipline (fail-loud-discipline.md): if the post-cook
    # re-query failed, the handler sets skeleton_summary = {"error": "..."} with
    # NO "count" field — so count defaulting to 0 is a genuine failure signal.
    skel_summary = validator.get("skeleton_summary", {})
    _assert(
        isinstance(skel_summary, dict),
        (
            f"FR-12 APPROVE-PATH: validator.skeleton_summary must be a dict, "
            f"got: {type(skel_summary)!r}. Full validator: {validator!r}"
        ),
    )
    # Check for explicit error from the post-cook re-query (fail-loud path).
    if "error" in skel_summary and "count" not in skel_summary:
        raise AssertionError(
            f"FR-12 APPROVE-PATH: post-cook re-query (query_skeleton on cooked node) "
            f"failed — the handler surfaced the error honestly per fail-loud-discipline.md "
            f"but the count is missing.\n"
            f"  skeleton_summary: {skel_summary!r}\n"
            f"  Full validator: {validator!r}\n"
            f"  Full approved_data: {approved_data!r}\n"
            f"\n"
            f"  This may mean kinefx::secondarymotion cooked but the output geometry "
            f"has no @name attribute readable by query_skeleton.  Check whether "
            f"query_skeleton can read the cooked SOP in headless hython."
        )
    count = skel_summary.get("count", 0)
    _assert(
        count > 0,
        (
            f"FR-12 APPROVE-PATH: validator.skeleton_summary.count must be > 0 — "
            f"the post-cook re-query must have read real joint data from the cooked "
            f"kinefx::secondarymotion node output (3 joints in the fixture). "
            f"Got count={count!r}. "
            f"skeleton_summary: {skel_summary!r}. "
            f"Full validator: {validator!r}."
        ),
    )

    # ── (D) validator.note == "verify-after-mutate" ─────────────────────────
    _assert(
        validator.get("note") == "verify-after-mutate",
        (
            f"FR-12 APPROVE-PATH: validator.note must be 'verify-after-mutate', "
            f"got: {validator.get('note')!r}. Full validator: {validator!r}"
        ),
    )


# ===========================================================================
# Main runner
# ===========================================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("kinefx_gate_smoke.py — pp12-110g RED phase (strengthened)")
    print("Representative tool: apply_secondarymotion (REAL handler, no stubs)")
    print("=" * 70)
    print()
    print("ANTI-MIRROR: character_handlers module imported; no register_handler()")
    print("calls in this smoke.  All checks drive the REAL registered handler.")
    print()

    _run(
        "check-1: REAL dispatch in propose mode has 'preview' key",
        _check1_preview_key_in_queue_response,
    )
    _run(
        "check-2: preview contains REAL node_plan + REAL skeleton_summary",
        _check2_payload_matches_real_model_and_real_fixture,
    )
    _run(
        "check-3: preview declares FR-12 validator_contract with skeleton_summary",
        _check3_fr12_validator_contract_in_preview,
    )
    _run(
        "check-4: APPROVE-PATH — approved return carries REAL validator.skeleton_summary.count > 0",
        _check4_fr12_approve_path_real_validator_return,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    total = len(RESULTS)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = total - passed
    print(f"Results: {passed}/{total} PASSED  ({failed} FAILED)")
    print()
    if failed:
        print("FAILURES:")
        for label, ok, msg in RESULTS:
            if not ok:
                print(f"  FAIL  {label}")
                print(f"        {msg}")
        print()
        print("RED CONFIRMED — expected failure state for pp12-110g:")
        print("  check-1 RED: no preview_fn on real apply_secondarymotion registration")
        print("  check-2 RED: ImportError for node_plan + no preview in queue response")
        print("  check-3 RED: no preview_fn → no preview → no validator_contract in preview")
        print("  check-4 RED: (a) no preview_fn → dispatch fails before approve, OR")
        print("               (b) 'validator' absent from approved return, OR")
        print("               (c) kinefx::secondarymotion cook fails headlessly (ok=False)")
        print("               → escalate to operator-smoke for FR-12-return verification")
        print()
        print("hou-dev PR-8 must:")
        print("  (a) Add node_plan(tool, args) -> dict to kinefx_model.py (hou-free, CL-015)")
        print("  (b) Add real preview_fn to all 5 mutating register_handler() calls in")
        print("      character_handlers.py.  Each preview_fn must:")
        print("        - call query_skeleton on the source node (not a fixture)")
        print("        - call node_plan(tool, dispatch_args) from kinefx_model")
        print("        - return {tool, args, node_plan, skeleton_summary,")
        print("                  validator_contract: {fields: ['skeleton_summary'],")
        print("                                       note: 'verify-after-mutate'}}")
        print("  (c) Add real validator sub-dict to each mutating handler's approved return,")
        print("      embedding post-cook query_skeleton on the result node —")
        print("      NOT a static {note: 'verify-after-mutate'} placeholder.")
        print("      check-4 verifies this in hython-smoke IF the cook succeeds.")
        print("      If cook fails headlessly, check-4 escalates to operator-smoke.")
        sys.exit(1)
    else:
        print("All checks PASSED — pp12-110g smoke is GREEN.")
        sys.exit(0)
