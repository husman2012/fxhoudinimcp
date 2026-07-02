"""Gate preview-hook hython-smoke -- RED build gate for pp12-111g.

Covers 7 checks for ADR 0005 preview-hook contract:
  check-1: PREVIEW SURFACES in list_pending_calls
  check-2: DENY-ON-PREVIEW-ERROR (preview_required=True + raising preview_fn)
  check-3: DEGRADE (preview_required=False + raising preview_fn -> queued with preview_error)
  check-4: APPROVE-REVALIDATE-DIVERGENCE (approve when preview verdict changed)
  check-5: READONLY GUARD (B1 root fix -- hou.undos.disabler() applied in _run_preview)
  check-6: BUDGET_VERDICT IN EXPORT PREVIEW (B2 root fix -- validate_budget called)
  check-7: NO SPURIOUS DIVERGENCE (B3 root fix -- stored params used at approve time)

Round-1 checks (1-4): GREEN after Slice A landed (register_handler preview_fn kwarg).
Round-2 checks (5-7): INTENTIONALLY RED today.
  check-5 RED: sentinel detects disabler() never called in _run_preview (B1 confirmed).
  check-6 RED: _preview_fbx calls rop_plan() which has no budget_verdict key (B2 confirmed).
  check-7 RED: middleware uses params_hint={} at approve time (B3 confirmed).

Run:
    "C:/Program Files/Side Effects Software/Houdini 21.0.729/bin/hython.exe" \
        C:/Users/husma/development/fxhoudinimcp/tests/gate_preview_hython_smoke.py

Exit 0 = all checks PASSED.
Exit 1 = one or more checks FAILED.

Verification surface: hython-smoke
Author: hou-test (pp12-111g round-2)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

# --- sys.path: Homedini pure core first, then the fork ---
sys.path.insert(0, "C:/Users/husma/development/HoudiniUtilTools/scripts/python")
sys.path.insert(0, "C:/Users/husma/development/fxhoudinimcp/houdini/scripts/python")

# The gate must already exist (pp12-109b landed). If not, the smoke fails with
# a clear ModuleNotFoundError before check-1 even registers.
from fxhoudinimcp_server.gate import install_gate

import hou
import fxhoudinimcp_server.dispatcher as _dispatcher

# ---------------------------------------------------------------------------
# Helpers (mirror gate_hython_smoke.py)
# ---------------------------------------------------------------------------

def _make_prefs(mode: str) -> str:
    """Write a fresh temp prefs dir with mcp_gate.json for the given mode."""
    prefs = tempfile.mkdtemp(prefix="hython_preview_test_")
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
    # Also clear any preview registry stashed on hou.session (ADV-003 / CL-005).
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


# ---------------------------------------------------------------------------
# Check-1: PREVIEW SURFACES in list_pending_calls
# ---------------------------------------------------------------------------
# register_handler with preview_fn=<callable> + preview_required=False.
# Dispatch in propose mode -> should QUEUE (gate=queued) with preview payload
# visible both in the queue response and in list_pending_calls.
#
# RED failure: TypeError: register_handler() got an unexpected keyword argument 'preview_fn'
# ---------------------------------------------------------------------------

def _check1_preview_surfaces():
    _install("propose")

    call_log: list[dict] = []

    def _preview_fn(params: dict) -> dict:
        call_log.append(dict(params))
        return {"preview_field": "check1_value", "budget_ok": True}

    # RED: 'preview_fn' kwarg does not exist on register_handler yet.
    _dispatcher.register_handler(
        "test.preview_surfaces",
        lambda: {"status": "success", "data": {"done": True}},
        capability=_dispatcher.Capability.MUTATING,
        preview_fn=_preview_fn,
        preview_required=False,
    )

    r = _dispatch("test.preview_surfaces", {"arg": "x"})

    # Must queue in propose mode.
    _assert(r.get("gate") == "queued", f"expected gate=queued, got {r!r}")
    pending_id = r.get("pending_id")
    _assert(pending_id is not None, f"no pending_id in queued response: {r!r}")

    # Preview payload must be surfaced in the queue response under 'preview'.
    preview_in_response = r.get("preview", {})
    _assert(
        preview_in_response.get("preview_field") == "check1_value",
        f"preview not surfaced in queue response: {r!r}",
    )

    # preview_fn must have been called at least once (at queue time).
    _assert(len(call_log) >= 1, f"preview_fn was never called; call_log={call_log!r}")

    # Preview payload must also appear in list_pending_calls.
    list_r = _dispatch("gate.list_pending_calls", {})
    entries = list_r.get("data", {}).get("pending", [])
    matched = [e for e in entries if e.get("id") == pending_id]
    _assert(len(matched) == 1, f"pending_id not found in list_pending_calls: {entries!r}")
    entry_preview = matched[0].get("preview", {})
    _assert(
        entry_preview.get("preview_field") == "check1_value",
        f"preview not in list_pending_calls entry: {matched[0]!r}",
    )


# ---------------------------------------------------------------------------
# Check-2: DENY-ON-PREVIEW-ERROR (preview_required=True + raising preview_fn)
# ---------------------------------------------------------------------------
# When preview_required=True and the preview_fn raises, the call must be DENIED
# (not queued). gate=denied and no pending_id in the response. (ADV-007)
#
# RED failure: TypeError: register_handler() got an unexpected keyword argument 'preview_fn'
# ---------------------------------------------------------------------------

def _check2_deny_on_preview_error():
    _install("propose")

    def _failing_preview(params: dict) -> dict:
        raise RuntimeError("Simulated budget validation failure")

    # RED: 'preview_fn' / 'preview_required' kwargs do not exist yet.
    _dispatcher.register_handler(
        "test.deny_on_preview_error",
        lambda: {"status": "success", "data": {}},
        capability=_dispatcher.Capability.MUTATING,
        preview_fn=_failing_preview,
        preview_required=True,  # DENY when preview fails
    )

    r = _dispatch("test.deny_on_preview_error", {})

    # Must be DENIED — not queued, not approved.
    _assert(
        r.get("gate") == "denied",
        f"expected gate=denied when preview_required=True and preview_fn raises, got {r!r}",
    )
    _assert(
        r.get("pending_id") is None,
        f"denied call must not have a pending_id, got {r!r}",
    )

    # The denial reason should reference the preview failure.
    reason = r.get("reason") or r.get("error") or ""
    _assert(
        reason,
        f"denied response should carry a reason/error, got {r!r}",
    )


# ---------------------------------------------------------------------------
# Check-3: DEGRADE (preview_required=False + raising preview_fn -> queued with preview_error)
# ---------------------------------------------------------------------------
# When preview_required=False and the preview_fn raises, the call is still QUEUED
# but the queue entry carries a 'preview_error' field instead of a 'preview' payload.
# (ADR 0005 §3.4d degrade path)
#
# RED failure: TypeError: register_handler() got an unexpected keyword argument 'preview_fn'
# ---------------------------------------------------------------------------

def _check3_degrade_on_preview_error():
    _install("propose")

    def _failing_preview(params: dict) -> dict:
        raise RuntimeError("Simulated transient preview error")

    # RED: kwarg does not exist yet.
    _dispatcher.register_handler(
        "test.degrade_preview_error",
        lambda: {"status": "success", "data": {}},
        capability=_dispatcher.Capability.MUTATING,
        preview_fn=_failing_preview,
        preview_required=False,  # DEGRADE: queue even if preview fails
    )

    r = _dispatch("test.degrade_preview_error", {})

    # Must still QUEUE (not deny).
    _assert(
        r.get("gate") == "queued",
        f"expected gate=queued when preview_required=False even with failing preview_fn, got {r!r}",
    )
    pending_id = r.get("pending_id")
    _assert(pending_id is not None, f"no pending_id for degraded queue: {r!r}")

    # The queued entry must carry 'preview_error' (not a 'preview' dict).
    list_r = _dispatch("gate.list_pending_calls", {})
    entries = list_r.get("data", {}).get("pending", [])
    matched = [e for e in entries if e.get("id") == pending_id]
    _assert(len(matched) == 1, f"pending_id not in list_pending_calls: {entries!r}")
    entry = matched[0]

    # preview_error must be present and non-empty.
    _assert(
        entry.get("preview_error"),
        f"degraded entry must carry preview_error, got entry={entry!r}",
    )
    # preview payload should be None or absent (no valid preview was produced).
    preview_val = entry.get("preview")
    _assert(
        preview_val is None,
        f"degraded entry must not carry a preview payload, got preview={preview_val!r}",
    )


# ---------------------------------------------------------------------------
# Check-4: APPROVE-REVALIDATE-DIVERGENCE
# ---------------------------------------------------------------------------
# Approve-time re-validate: when the preview_fn returns a different verdict on
# the second call (re-validate at approve time vs queue time), the approved
# response carries a 'divergence_warning'. The call is still approved (not
# auto-blocked) per ADR 0005 revision-2 §3.4f — the operator is warned, not
# hard-blocked.
#
# A stateful preview_fn (via a mutable list call_count) returns {"verdict": "ok"}
# on call[0] (queue time) and {"verdict": "fail", "reason": "scene_changed"} on
# call[1] (approve time re-validate).
#
# RED failure: TypeError: register_handler() got an unexpected keyword argument 'preview_fn'
# ---------------------------------------------------------------------------

def _check4_approve_revalidate_divergence():
    _install("propose")

    call_count: list[int] = [0]

    def _stateful_preview(params: dict) -> dict:
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: queue-time preview passes.
            return {"verdict": "ok", "scene_hash": "abc"}
        # Second call: approve-time re-validate sees changed scene.
        return {"verdict": "fail", "reason": "scene_changed", "scene_hash": "xyz"}

    # RED: kwarg does not exist yet.
    _dispatcher.register_handler(
        "test.revalidate_divergence",
        lambda: {"status": "success", "data": {"bake": "done"}},
        capability=_dispatcher.Capability.MUTATING,
        preview_fn=_stateful_preview,
        preview_required=False,  # DEGRADE path — divergence is a warning, not a block
    )

    # Queue the call.
    r = _dispatch("test.revalidate_divergence", {})
    _assert(r.get("gate") == "queued", f"expected queued, got {r!r}")
    pending_id = r.get("pending_id")
    _assert(pending_id is not None, f"no pending_id: {r!r}")

    # preview_fn called once at queue time.
    _assert(call_count[0] == 1, f"preview_fn call_count after queue: {call_count[0]}")

    # Approve the call — middleware re-runs preview_fn at approve time.
    a = _dispatch("gate.approve_pending_call", {"pending_id": pending_id})

    # The call must be approved (not denied) — ADR 0005 revision-2 is warn-not-block.
    _assert(
        a.get("gate") == "approved",
        f"expected gate=approved even on divergence, got {a!r}",
    )

    # But the approved response must carry a divergence_warning.
    _assert(
        a.get("divergence_warning"),
        f"expected divergence_warning in approved response, got {a!r}",
    )

    # preview_fn must have been called a second time at approve time.
    _assert(
        call_count[0] >= 2,
        f"preview_fn should be called at approve time (re-validate); call_count={call_count[0]}",
    )


# ---------------------------------------------------------------------------
# Check-5: READONLY GUARD (B1 root fix)
# ---------------------------------------------------------------------------
# ADR 0005 rev2 §3.4a: _run_preview MUST wrap the preview_fn call with
# hou.undos.disabler() so that no undo stack entries are created during
# preview (the preview is a DRY RUN, not an actual mutation).
#
# Mechanism: monkey-patch hou.undos.disabler with a sentinel context manager
# that records invocations in a list. After the dispatch, assert the sentinel
# was called >= 1 times. This is hython-safe — does not rely on
# areEnabled() returning a specific value (hython default is False regardless),
# and does not use hou.undos.enable() which does not exist on the API surface.
#
# RED failure today: _run_preview has NO hou.undos.disabler() (confirmed by
# grep: 0 matches for 'undos.disabler|_readonly_dispatch' in middleware.py).
# The sentinel list stays empty, triggering:
#   AssertionError: hou.undos.disabler() was never called during _run_preview --
#     B1 guard not implemented. ADR 0005 rev2 §3.4a requires disabler() in _run_preview.
# ---------------------------------------------------------------------------

def _check5_readonly_guard():
    """B1: _run_preview must apply hou.undos.disabler() before invoking preview_fn.

    Mechanism: monkey-patch hou.undos.disabler with a sentinel context manager
    that records whether it was invoked during the dispatch. After the dispatch,
    assert the sentinel was called at least once.

    This approach is hython-safe: it does not rely on hou.undos.areEnabled()
    returning a particular value (hython starts with undos disabled regardless),
    and does not require hou.undos.enable() (which does not exist on the API).

    RED failure today: _run_preview has NO hou.undos.disabler() guard (confirmed
    by grep: 0 matches for 'undos.disabler|_readonly_dispatch' in middleware.py).
    The sentinel will never be appended to disabler_called[], triggering:
      AssertionError: hou.undos.disabler() was never called during _run_preview --
        B1 guard not implemented. ADR 0005 rev2 §3.4a requires disabler() in _run_preview.
    """
    import contextlib

    _install("propose")

    # Sentinel: tracks whether hou.undos.disabler() was invoked during the dispatch.
    disabler_called: list[bool] = []
    original_disabler = hou.undos.disabler

    @contextlib.contextmanager
    def _sentinel_disabler():
        disabler_called.append(True)
        with original_disabler():
            yield

    # Monkey-patch for the duration of this check.
    hou.undos.disabler = _sentinel_disabler
    try:
        def _preview_fn(params: dict) -> dict:
            return {"readonly_check": "ok"}

        _dispatcher.register_handler(
            "test.readonly_guard",
            lambda: {"status": "success", "data": {}},
            capability=_dispatcher.Capability.MUTATING,
            preview_fn=_preview_fn,
            preview_required=False,
        )

        r = _dispatch("test.readonly_guard", {"x": 1})

        # Must have queued (so preview_fn ran).
        _assert(r.get("gate") == "queued", f"expected gate=queued, got {r!r}")

        # The sentinel MUST have been called: _run_preview must invoke disabler().
        # RED today: _run_preview has no disabler(), so disabler_called stays empty.
        _assert(
            len(disabler_called) >= 1,
            f"hou.undos.disabler() was never called during _run_preview -- "
            f"B1 guard not implemented. ADR 0005 rev2 §3.4a requires hou.undos.disabler() "
            f"to be applied in _run_preview before invoking preview_fn.",
        )
    finally:
        # Always restore the original disabler.
        hou.undos.disabler = original_disabler


# ---------------------------------------------------------------------------
# Check-6: BUDGET_VERDICT IN EXPORT PREVIEW (B2 root fix)
# ---------------------------------------------------------------------------
# ADR 0005 rev2 §3.5: the export preview shape MUST be
# {version_triple, budget_verdict, out_paths, rop_plan}.
# validate_budget() MUST be called from each _preview_* function and its
# result stored under the key "budget_verdict".
#
# Mechanism: dispatch houdini.export_fbx in propose mode. In hython context
# hou.applicationVersionString() and rop_plan() are available; the preview
# will be computed and stored in the queue entry. The test then inspects the
# stored preview for the "budget_verdict" key.
#
# RED failure today: _preview_fbx calls rop_plan() which returns
# {tool, params, out_paths, version_triple, rop_plan_schema_version} with
# NO budget_verdict key. validate_budget() is never called from any _preview_*.
#
# Note: houdini.export_fbx requires a real scene node. We use minimal params
# that will fail the inner ROP execution (the actual bake), but the PREVIEW
# should still compute since rop_plan() does not require the node to exist.
# If the preview itself fails (preview_required=True on fbx), the response
# will be gate=denied with a preview_error; we inspect the preview_error
# payload for the budget_verdict instead.
# ---------------------------------------------------------------------------

def _check6_budget_verdict_in_export_preview():
    """B2: export preview shape must include budget_verdict (validate_budget called)."""
    _install("propose")

    # "export_fbx" is registered in export_handlers.py with preview_required=True
    # and preview_fn=_preview_fbx.
    # The _preview_fbx function is called before queuing.
    # If rop_plan() raises or preview is unavailable, the call is DENIED
    # (preview_required=True). The denial response may carry a preview_error.
    # Either way, we inspect whatever preview payload was produced.
    #
    # Minimal params: node="/obj/geo1" may not exist in a fresh hython scene,
    # which means rop_plan() might fail gracefully. We still assert on the key.
    minimal_params = {
        "node": "/obj/geo1",
        "out_path": "/tmp/hython_test_check6.fbx",
    }

    r = _dispatch("export_fbx", minimal_params)

    # The call will be queued (propose mode) or denied (preview_required=True + preview error).
    gate = r.get("gate")
    _assert(
        gate in ("queued", "denied"),
        f"expected gate=queued or gate=denied for export_fbx in propose mode, got {r!r}",
    )

    # Extract the preview payload from whichever path it came through.
    preview_payload = r.get("preview") or {}
    preview_error = r.get("preview_error") or ""

    # If the preview succeeded (gate=queued), budget_verdict must be in it.
    # If preview failed (gate=denied), the preview ran but raised — that's B2
    # being present differently; we still fail the check because validate_budget
    # was not called (the preview raised for other reasons, or budget_verdict is absent).
    #
    # The core assertion: "budget_verdict" must be a key in the preview dict.
    _assert(
        "budget_verdict" in preview_payload,
        f"'budget_verdict' key missing from export_fbx preview payload.\n"
        f"  gate={gate!r}\n"
        f"  preview={preview_payload!r}\n"
        f"  preview_error={preview_error!r}\n"
        f"  ADR 0005 rev2 §3.5 requires {{version_triple, budget_verdict, out_paths, rop_plan}}.\n"
        f"  Root cause: _preview_fbx calls rop_plan() which returns no budget_verdict key;\n"
        f"  validate_budget() is never called from any _preview_* function (B2 unimplemented).",
    )


# ---------------------------------------------------------------------------
# Check-7: NO SPURIOUS DIVERGENCE (B3 root fix)
# ---------------------------------------------------------------------------
# ADR 0005 rev2 §3.4f: approve-time re-validate MUST use the STORED params
# from the queue entry, NOT a synthetic hint (params_hint={}).
#
# Mechanism: register a params-dependent preview_fn that hashes the 'scene_token'
# param. Queue with params={"scene_token": "A"} -> preview stores {"hash": "A"}.
# Approve: if middleware uses params_hint={}, re-validate gets {"hash": ""} (or
# the default) which != {"hash": "A"}, triggering a spurious divergence_warning.
# If middleware uses the STORED params, re-validate gets {"hash": "A"} == {"hash": "A"}
# -> no divergence_warning (or an empty/absent one).
#
# RED failure today: middleware.py:651 sets params_hint: dict = {} and passes it
# to _run_preview at approve time, causing the hash to differ:
#   AssertionError: Got unexpected divergence_warning in approved response --
#     middleware used params_hint={} instead of stored params (B3 unimplemented).
# ---------------------------------------------------------------------------

def _check7_no_spurious_divergence():
    """B3: approve-time re-validate must use stored params, not params_hint={}."""
    _install("propose")

    def _params_dependent_preview(params: dict) -> dict:
        # Returns a hash based on the scene_token param.
        # If approve-time uses params_hint={}, scene_token is "" -> hash differs.
        token = params.get("scene_token", "")
        return {"hash": token, "verdict": "ok" if token else "missing_token"}

    _dispatcher.register_handler(
        "test.no_spurious_divergence",
        lambda scene_token="": {"status": "success", "data": {"scene_token": scene_token}},
        capability=_dispatcher.Capability.MUTATING,
        preview_fn=_params_dependent_preview,
        preview_required=False,  # divergence is a warning, not a block
    )

    # Queue with specific params including scene_token.
    # At queue time: preview_fn({"scene_token": "A"}) -> {"hash": "A", "verdict": "ok"}
    queue_params = {"scene_token": "A"}
    r = _dispatch("test.no_spurious_divergence", queue_params)
    _assert(r.get("gate") == "queued", f"expected gate=queued, got {r!r}")
    pending_id = r.get("pending_id")
    _assert(pending_id is not None, f"no pending_id: {r!r}")

    # Verify the stored preview reflects the original params.
    preview_at_queue = r.get("preview", {})
    _assert(
        preview_at_queue.get("hash") == "A",
        f"queue-time preview should have hash='A', got {preview_at_queue!r}",
    )

    # Approve: approve-time re-validate should use stored params {"scene_token": "A"}.
    # With stored params: preview_fn({"scene_token": "A"}) -> {"hash": "A"} == queue preview
    # With params_hint={}: preview_fn({}) -> {"hash": ""} != {"hash": "A"} -> spurious divergence
    a = _dispatch("gate.approve_pending_call", {"pending_id": pending_id})

    _assert(
        a.get("gate") == "approved",
        f"expected gate=approved, got {a!r}",
    )

    # NO divergence_warning should appear when params haven't changed.
    # RED today: middleware uses params_hint={} -> hash mismatch -> spurious divergence_warning.
    divergence = a.get("divergence_warning")
    _assert(
        not divergence,
        f"Got unexpected divergence_warning in approved response -- middleware used "
        f"params_hint={{}} instead of stored params (B3 unimplemented).\n"
        f"  divergence_warning={divergence!r}\n"
        f"  approved response={a!r}\n"
        f"  ADR 0005 rev2 §3.4f: approve-time re-validate must use STORED params "
        f"from queue entry, not a synthetic empty hint.",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    print("=== gate_preview_hython_smoke.py -- pp12-111g RED build gate (round-2) ===")
    print()
    print("-- Round-1: ADR 0005 preview-hook dispatcher checks (1-4) --")
    _run("check-1: PREVIEW SURFACES in list_pending_calls", _check1_preview_surfaces)
    _run("check-2: DENY-ON-PREVIEW-ERROR (preview_required=True)", _check2_deny_on_preview_error)
    _run("check-3: DEGRADE (preview_required=False + raising preview_fn)", _check3_degrade_on_preview_error)
    _run("check-4: APPROVE-REVALIDATE-DIVERGENCE", _check4_approve_revalidate_divergence)

    print()
    print("-- Round-2: B1/B2/B3 root-fix checks (5-7) --")
    _run("check-5: READONLY GUARD (B1 -- hou.undos.disabler in _run_preview)", _check5_readonly_guard)
    _run("check-6: BUDGET_VERDICT IN EXPORT PREVIEW (B2 -- validate_budget called)", _check6_budget_verdict_in_export_preview)
    _run("check-7: NO SPURIOUS DIVERGENCE (B3 -- stored params at approve time)", _check7_no_spurious_divergence)

    print()
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"=== {passed}/{total} PASSED ===")

    for label, ok, msg in RESULTS:
        if not ok:
            print(f"  FAIL  {label}: {msg}")

    if passed < total:
        print("RESULT: RED (expected for round-2 checks -- B1/B2/B3 not yet implemented)")
        return 1
    print("RESULT: GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
