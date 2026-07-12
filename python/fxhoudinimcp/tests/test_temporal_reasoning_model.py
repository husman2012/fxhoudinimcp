"""
Tests for temporal_reasoning_model.py — pure-logic core (PP12-117 PR-1,
VERIFICATION half only).

TDD phase: RED — temporal_reasoning_model.py does NOT exist yet.
Expected failure: ModuleNotFoundError on import (a single collection error
for the whole file — see plan pp12-117a acceptanceTests).

Contract (plan pp12-117a lockedFieldContract, REVISION 2 — folds all 11
codex-adversarial-reviewer findings, 2 Blockers + 9 Majors): imports NO hou,
NO Qt/PySide6, NO pxr, NO FastMCP. Makes NO MCP call. NEVER cooks/reads a
sim itself — per-frame series arrive as plain [[frame,value], ...] lists;
reading them off a real cook is the (deferred) PR-2 handler's job (CL-015
extended). numpy is PERMITTED but NOT mandated — the fork .venv does not
have numpy installed (verified for the sibling 116 PR-1, 2026-07-09), so a
correct stdlib-only implementation satisfies the locked behavioral contract
equally well; this suite asserts ONLY the behavioral contract, never a
numpy-API usage. Plain Python, pytest-able off-DCC (CL-015).

SCOPE SPLIT (BINDING — verification half ONLY, see plan pp12-117a): this PR
builds the event-timeline REPRESENTATION (EventSpec/EventGraph — the
causal DAG, topo order, cycle rejection, lossless params round-trip) plus
the Gate-1 ASSERTION-AGGREGATION oracle math (aggregate_assertion /
evaluate_assertions) plus the describe_sim_events() vocabulary. It does
NOT build the timeline->setup TRANSLATION (compile_timeline and its
keyframe/CHOP/DOP-parm compiled plan) — that is DEFERRED to a later PR
because its output shape co-designs with the applying handler. This suite
asserts NONE of: compile_timeline, an MCP registration, the 109 gate,
hdefereval marshaling, or a live cook/DOP read.

Covers the public contract of:
  - EventSpec    (dataclass) — id, type, target='', at_frame=0, params={},
                                 trigger={}, causes=[]; __post_init__
                                 validates type/trigger-kind vocab + field
                                 types; to_dict() with a LOSSLESS params
                                 payload (Blocker-2)
  - EventGraph   (dataclass) — events=[]; topo_order() (causal-DAG order,
                                 events-list-order tie-break, cycle/
                                 dangling/duplicate-id rejection);
                                 to_dict()/from_dict() (Major-9: duplicate
                                 ids raise in ALL THREE of topo_order/
                                 to_dict/from_dict)
  - describe_sim_events() -> dict (FR-C)      — exact SPEC 4.1 vocabulary
  - aggregate_assertion(series, expect) -> dict (FR-B core; the pinned
                                 per-predicate metric semantics, Blocker-1
                                 jump_gt scan)
  - evaluate_assertions(assertions) -> dict (FR-B, the assert_simulation
                                 model core; exact SPEC 4.1 {results,pass}
                                 shape, input-order results, Major-11
                                 extra-key ignoring)

Cross-references:
  - Plan pp12-117a lockedFieldContract (BINDING, revision 2)
  - docs/portfolio_projects/PP12_houdini_mcp_agentic_bridge/
    117_mcp_temporal_sim_reasoning_surface/spec.md Sections 4.1, 6, 9
  - tdd-with-agents.md Sec.4: hou-test writes red; hou-dev turns green
  - CL-015 (extended): pure-logic module — no hou/Qt/pxr/FastMCP, no MCP
    call, no cook/DOP read, no timeline->setup translation

Out of scope for PR-1 (deferred): compile_timeline, the timeline->setup
translation, the bbox/field_stats scalar-reduction (handler-side), MCP
registration, the 109 security gate, hdefereval marshaling, the
handlers/tools layer, any live cook/DOP read. This suite asserts NONE of
those.

NOTE on Hypothesis: the fork .venv does not have the `hypothesis` package
installed (verified 2026-07-09 for the sibling 116 PR-1) and this suite
deliberately does not add it as a new dependency mid-PR. The "assert the
invariant holds across many inputs" discipline a Hypothesis strategy would
give is instead delivered via `@pytest.mark.parametrize` over a
representative set of hand-authored fixtures (see
TestRoundTripAndDeterminismProperties below).
"""

from __future__ import annotations

import json
import sys
import os

# ---------------------------------------------------------------------------
# Path bootstrap — allow running standalone and via pytest.
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_PKG_ROOT))

import pytest

# ---------------------------------------------------------------------------
# The module under test.
# This import MUST fail (ModuleNotFoundError) in the RED phase because
# temporal_reasoning_model.py does not exist yet. Every test below is
# therefore expected to report as a single collection error, not an
# individual per-test failure, until hou-dev lands the green implementation.
# ---------------------------------------------------------------------------
from fxhoudinimcp.temporal_reasoning_model import (
    EventSpec,
    EventGraph,
    describe_sim_events,
    aggregate_assertion,
    evaluate_assertions,
)


# ===========================================================================
# Shared helper — recursive plain-JSON-type checker.
#
# Catches a numpy-scalar leak WITHOUT importing numpy (numpy is not
# installed in this venv, and must not be a hard test dependency either —
# the contract requires numpy be PERMITTED, not MANDATED). Asserts strict
# Python types (`type(x) is float`, not `isinstance`), which rejects
# numpy.float64 / numpy.int64 exactly as it rejects any other non-plain
# type, since numpy scalars have a DIFFERENT `type()` than the plain
# builtins even when they subclass them.
# ===========================================================================

def _assert_plain_json_types(obj):
    """Recursively assert every value is a plain JSON-safe Python type."""
    if isinstance(obj, dict):
        assert type(obj) is dict, f"expected a plain dict, got {type(obj)!r}"
        for k, v in obj.items():
            assert type(k) is str, f"dict key must be a plain str, got {type(k)!r}"
            _assert_plain_json_types(v)
    elif isinstance(obj, list):
        assert type(obj) is list, f"expected a plain list, got {type(obj)!r}"
        for item in obj:
            _assert_plain_json_types(item)
    elif obj is None:
        pass
    elif isinstance(obj, bool):
        assert type(obj) is bool, f"expected a plain bool, got {type(obj)!r}"
    elif isinstance(obj, int):
        assert type(obj) is int, f"expected a plain int (no numpy int64 leak), got {type(obj)!r}"
    elif isinstance(obj, float):
        assert type(obj) is float, f"expected a plain float (no numpy float64 leak), got {type(obj)!r}"
    elif isinstance(obj, str):
        assert type(obj) is str, f"expected a plain str, got {type(obj)!r}"
    else:
        pytest.fail(f"Non-JSON-plain type leaked into output: {type(obj)!r} (value={obj!r})")


# ===========================================================================
# Section 1 — EventSpec dataclass
#
# Locked contract:
#   EventSpec(id: str, type: str, target: str = '', at_frame: int = 0,
#             params: dict = field(default_factory=dict),
#             trigger: dict = field(default_factory=dict),
#             causes: list = field(default_factory=list))
#   type in {fracture, emit, tear, ignite, keyframe} else ValueError.
#   trigger, when non-empty, MUST carry a `kind` in
#   {stress_gt, collision_with, frame_eq, field_gt} else ValueError; an
#   empty trigger {} is allowed (unconditional/at_frame event).
#   id non-empty str; target str; at_frame int (NOT bool); params dict;
#   trigger dict; causes a list of non-empty str ids.
#   to_dict() -> EXACT keys {id, type, target, at_frame, params, trigger,
#   causes}. params is the LOSSLESS event-specific payload (Blocker-2).
# ===========================================================================

class TestEventSpec:
    """EventSpec — locked fields, defaults, type/trigger-kind vocab
    validation, lossless params, to_dict()."""

    def test_minimal_construction_and_defaults(self):
        e = EventSpec(id="e1", type="fracture")
        assert e.id == "e1"
        assert e.type == "fracture"
        assert e.target == ""
        assert e.at_frame == 0
        assert e.params == {}
        assert e.trigger == {}
        assert e.causes == []

    def test_full_construction(self):
        e = EventSpec(
            id="e1", type="emit", target="/obj/rbd_sim/debris", at_frame=40,
            params={"rate": 500.0}, trigger={"kind": "stress_gt", "value": 1200},
            causes=["e2"],
        )
        assert e.target == "/obj/rbd_sim/debris"
        assert e.at_frame == 40
        assert e.params == {"rate": 500.0}
        assert e.trigger == {"kind": "stress_gt", "value": 1200}
        assert e.causes == ["e2"]

    def test_to_dict_exact_keys(self):
        e = EventSpec(id="e1", type="fracture", target="/obj/x", at_frame=40,
                       params={"a": 1}, trigger={"kind": "frame_eq"}, causes=["e2"])
        d = e.to_dict()
        assert set(d.keys()) == {"id", "type", "target", "at_frame", "params", "trigger", "causes"}, (
            f"Unexpected keys in EventSpec.to_dict(): {set(d.keys())!r}"
        )

    def test_params_payload_preserved_losslessly_emit_rate(self):
        """Blocker-2 — params carries the event-specific payload so the
        representation is lossless (this is what the deferred timeline->
        setup translation will consume)."""
        e = EventSpec(id="e1", type="emit", target="/obj/x", params={"rate": 500.0, "unit": "particles/frame"})
        d = e.to_dict()
        assert d["params"] == {"rate": 500.0, "unit": "particles/frame"}, (
            "EventSpec.params must be preserved LOSSLESSLY through to_dict()"
        )

    def test_params_payload_preserved_losslessly_keyframe_frames(self):
        frames_payload = {"node": "/obj/x", "parm": "tx", "frames": [[1, 0.0], [10, 5.0]]}
        e = EventSpec(id="e1", type="keyframe", params=frames_payload)
        d = e.to_dict()
        assert d["params"] == frames_payload

    def test_params_payload_preserved_losslessly_tear_threshold(self):
        e = EventSpec(id="e1", type="tear", target="/obj/cloth", params={"threshold": 0.75})
        assert e.to_dict()["params"] == {"threshold": 0.75}

    def test_params_payload_preserved_losslessly_ignite_region(self):
        e = EventSpec(id="e1", type="ignite", params={"region": "north_wall"})
        assert e.to_dict()["params"] == {"region": "north_wall"}

    def test_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError):
            EventSpec(id="e1", type="explode")

    def test_empty_string_type_raises(self):
        with pytest.raises(ValueError):
            EventSpec(id="e1", type="")

    def test_all_five_event_types_are_constructible(self):
        for event_type in ("fracture", "emit", "tear", "ignite", "keyframe"):
            EventSpec(id="e1", type=event_type)

    def test_empty_trigger_is_allowed(self):
        """An empty trigger {} means unconditional/at_frame — must not raise."""
        e = EventSpec(id="e1", type="fracture", trigger={})
        assert e.trigger == {}

    def test_all_four_trigger_kinds_are_constructible(self):
        for kind in ("stress_gt", "collision_with", "frame_eq", "field_gt"):
            EventSpec(id="e1", type="fracture", trigger={"kind": kind})

    def test_trigger_missing_kind_raises(self):
        with pytest.raises(ValueError):
            EventSpec(id="e1", type="fracture", trigger={"value": 1200})

    def test_trigger_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            EventSpec(id="e1", type="fracture", trigger={"kind": "teleport"})

    def test_empty_string_id_raises(self):
        with pytest.raises(ValueError):
            EventSpec(id="", type="fracture")

    def test_bool_at_frame_raises(self):
        """at_frame must be int and NOT bool (bool is a subclass of int in
        Python, so this must be an explicit exclusion)."""
        with pytest.raises(ValueError):
            EventSpec(id="e1", type="fracture", at_frame=True)

    def test_non_int_at_frame_raises(self):
        with pytest.raises(ValueError):
            EventSpec(id="e1", type="fracture", at_frame=1.5)
        with pytest.raises(ValueError):
            EventSpec(id="e1", type="fracture", at_frame="40")

    def test_non_str_target_raises(self):
        with pytest.raises(ValueError):
            EventSpec(id="e1", type="fracture", target=123)

    def test_non_dict_params_raises(self):
        with pytest.raises(ValueError):
            EventSpec(id="e1", type="fracture", params="rate=500")

    def test_non_dict_trigger_raises(self):
        with pytest.raises(ValueError):
            EventSpec(id="e1", type="fracture", trigger="stress_gt")

    def test_causes_non_list_raises(self):
        with pytest.raises(ValueError):
            EventSpec(id="e1", type="fracture", causes="e2")

    def test_causes_containing_non_str_raises(self):
        with pytest.raises(ValueError):
            EventSpec(id="e1", type="fracture", causes=[1, 2])

    def test_causes_containing_empty_str_raises(self):
        with pytest.raises(ValueError):
            EventSpec(id="e1", type="fracture", causes=["e2", ""])

    def test_default_params_trigger_causes_are_independent_per_instance(self):
        """A bare mutable default ({} or [] as a class attribute) would
        leak across instances -- the locked contract requires
        field(default_factory=...)."""
        e1 = EventSpec(id="e1", type="fracture")
        e2 = EventSpec(id="e2", type="fracture")
        e1.params["x"] = 1
        e1.trigger["kind"] = "frame_eq"
        e1.causes.append("ghost")
        assert e2.params == {}
        assert e2.trigger == {}
        assert e2.causes == []

    def test_is_json_serialisable(self):
        e = EventSpec(id="e1", type="keyframe", params={"frames": [[1, 0.0]]})
        json.dumps(e.to_dict())


# ===========================================================================
# Section 2 — EventGraph dataclass
#
# Locked contract:
#   EventGraph(events: list = field(default_factory=list))
#   DUPLICATE event ids raise ValueError in topo_order(), to_dict(), AND
#   from_dict() (Major-9).
#   topo_order() -> list[str] ordered so a cause precedes the events it
#   causes; STABLE tie-break = original events-list order; traverses each
#   event's causes in causes-LIST order. A dangling causes edge (an id not
#   present in events) raises ValueError. A CYCLE raises ValueError whose
#   message CONTAINS "cycle" and at least one participating id.
#   to_dict() -> EXACT keys {events:[...], nodes:int, edges:int}.
#   from_dict rebuilds EventSpec from each event dict (preserving params);
#   ignores unknown top-level keys.
# ===========================================================================

class TestEventGraph:
    """EventGraph — defaults, topo order, cycle/dangling/duplicate
    rejection, to_dict()/from_dict()."""

    def test_defaults(self):
        g = EventGraph()
        assert g.events == []

    def test_default_events_list_is_independent_per_instance(self):
        g1 = EventGraph()
        g2 = EventGraph()
        g1.events.append(EventSpec(id="x", type="fracture"))
        assert g2.events == [], (
            "EventGraph.events default must be independent per instance "
            "(field(default_factory=list), not a shared mutable default)"
        )

    def test_to_dict_exact_keys(self):
        g = EventGraph(events=[EventSpec(id="e1", type="fracture", causes=["e2"]),
                                EventSpec(id="e2", type="emit")])
        d = g.to_dict()
        assert set(d.keys()) == {"events", "nodes", "edges"}, (
            f"Unexpected keys in EventGraph.to_dict(): {set(d.keys())!r}"
        )

    def test_to_dict_nodes_and_edges_counts(self):
        g = EventGraph(events=[
            EventSpec(id="e1", type="fracture", causes=["e2", "e3"]),
            EventSpec(id="e2", type="emit"),
            EventSpec(id="e3", type="emit"),
        ])
        d = g.to_dict()
        assert d["nodes"] == 3
        assert d["edges"] == 2

    def test_to_dict_events_are_dicts_preserving_params(self):
        g = EventGraph(events=[EventSpec(id="e1", type="emit", params={"rate": 250.0})])
        d = g.to_dict()
        assert isinstance(d["events"], list) and isinstance(d["events"][0], dict)
        assert d["events"][0]["params"] == {"rate": 250.0}

    def test_topo_order_fracture_then_emit_chain(self):
        """A fracture->emit chain: the cause precedes the effect."""
        g = EventGraph(events=[
            EventSpec(id="e1", type="fracture", causes=["e2"]),
            EventSpec(id="e2", type="emit"),
        ])
        assert g.topo_order() == ["e1", "e2"]

    def test_topo_order_stable_tie_break_is_events_list_order(self):
        """Three fully-independent (no causal edges) events must come back
        in exactly their original events-list order."""
        g = EventGraph(events=[
            EventSpec(id="e_b", type="fracture"),
            EventSpec(id="e_a", type="emit"),
            EventSpec(id="e_c", type="tear"),
        ])
        assert g.topo_order() == ["e_b", "e_a", "e_c"]

    def test_topo_order_mixed_branch_preserves_list_order_tie_break(self):
        """e1 causes e3; e2 is independent. Both e1 (cause precedes e3)
        and the tie-break-by-list-order for the remaining zero-indegree
        set at each step must hold: the only valid order consistent with
        BOTH constraints is [e1, e2, e3]."""
        g = EventGraph(events=[
            EventSpec(id="e1", type="fracture", causes=["e3"]),
            EventSpec(id="e2", type="emit"),
            EventSpec(id="e3", type="emit"),
        ])
        assert g.topo_order() == ["e1", "e2", "e3"]

    def test_topo_order_rejects_cycle(self):
        g = EventGraph(events=[
            EventSpec(id="e1", type="fracture", causes=["e2"]),
            EventSpec(id="e2", type="emit", causes=["e1"]),
        ])
        with pytest.raises(ValueError) as excinfo:
            g.topo_order()
        message = str(excinfo.value)
        assert "cycle" in message.lower(), (
            f"a cyclic causes graph's error message must contain 'cycle', got {message!r}"
        )
        assert ("e1" in message) or ("e2" in message), (
            f"a cyclic causes graph's error message must name a participating id, got {message!r}"
        )

    def test_topo_order_rejects_dangling_edge(self):
        g = EventGraph(events=[EventSpec(id="e1", type="fracture", causes=["ghost"])])
        with pytest.raises(ValueError):
            g.topo_order()

    def test_topo_order_rejects_duplicate_ids(self):
        g = EventGraph(events=[
            EventSpec(id="dup", type="fracture"),
            EventSpec(id="dup", type="emit"),
        ])
        with pytest.raises(ValueError):
            g.topo_order()

    # -----------------------------------------------------------------
    # M3 (REVISION 3 FOLD): a dangling or cyclic causes edge MUST raise
    # at to_dict() AND from_dict() too -- NOT only at topo_order(). This
    # matches the error-taxonomy '[at topo_order/to_dict/from_dict]'.
    # -----------------------------------------------------------------

    def test_to_dict_rejects_dangling_edge(self):
        g = EventGraph(events=[EventSpec(id="e1", type="fracture", causes=["ghost"])])
        with pytest.raises(ValueError):
            g.to_dict()

    def test_from_dict_rejects_dangling_edge(self):
        raw = {
            "events": [
                {"id": "e1", "type": "fracture", "target": "", "at_frame": 0,
                 "params": {}, "trigger": {}, "causes": ["ghost"]},
            ],
            "nodes": 1, "edges": 1,
        }
        with pytest.raises(ValueError):
            EventGraph.from_dict(raw)

    def test_to_dict_rejects_cycle(self):
        g = EventGraph(events=[
            EventSpec(id="e1", type="fracture", causes=["e2"]),
            EventSpec(id="e2", type="emit", causes=["e1"]),
        ])
        with pytest.raises(ValueError) as excinfo:
            g.to_dict()
        message = str(excinfo.value)
        assert "cycle" in message.lower(), (
            f"a cyclic causes graph must raise a 'cycle'-naming error at "
            f"to_dict() too, got {message!r}"
        )
        assert ("e1" in message) or ("e2" in message)

    def test_from_dict_rejects_cycle(self):
        raw = {
            "events": [
                {"id": "e1", "type": "fracture", "target": "", "at_frame": 0,
                 "params": {}, "trigger": {}, "causes": ["e2"]},
                {"id": "e2", "type": "emit", "target": "", "at_frame": 0,
                 "params": {}, "trigger": {}, "causes": ["e1"]},
            ],
            "nodes": 2, "edges": 2,
        }
        with pytest.raises(ValueError) as excinfo:
            EventGraph.from_dict(raw)
        message = str(excinfo.value)
        assert "cycle" in message.lower(), (
            f"a cyclic causes graph must raise a 'cycle'-naming error at "
            f"from_dict() too, got {message!r}"
        )
        assert ("e1" in message) or ("e2" in message)

    def test_to_dict_rejects_duplicate_ids(self):
        g = EventGraph(events=[
            EventSpec(id="dup", type="fracture"),
            EventSpec(id="dup", type="emit"),
        ])
        with pytest.raises(ValueError):
            g.to_dict()

    def test_from_dict_rejects_duplicate_ids(self):
        raw = {
            "events": [
                {"id": "dup", "type": "fracture", "target": "", "at_frame": 0, "params": {}, "trigger": {}, "causes": []},
                {"id": "dup", "type": "emit", "target": "", "at_frame": 0, "params": {}, "trigger": {}, "causes": []},
            ],
            "nodes": 2, "edges": 0,
        }
        with pytest.raises(ValueError):
            EventGraph.from_dict(raw)

    # -----------------------------------------------------------------
    # M2 (REVISION 3 FOLD, load-bearing): EventGraph.from_dict /
    # _event_from_dict must pass RAW params/trigger/causes STRAIGHT to
    # EventSpec(...) WITHOUT pre-coercion (no list(raw_causes), no
    # dict(raw_trigger)/dict(raw_params)) so EventSpec.__post_init__
    # validation actually fires on a malformed raw field. A coercing
    # implementation silently "fixes" bad input instead of rejecting it.
    # -----------------------------------------------------------------

    def test_from_dict_does_not_coerce_string_causes_into_char_list(self):
        """A raw `causes` value of the STRING 'bc' must raise ValueError
        (causes must be a list) -- NOT silently become ['b', 'c'] via an
        implementation that does list(raw_causes)."""
        raw = {
            "events": [
                {"id": "e1", "type": "fracture", "target": "", "at_frame": 0,
                 "params": {}, "trigger": {}, "causes": "bc"},
            ],
            "nodes": 1, "edges": 0,
        }
        with pytest.raises(ValueError):
            EventGraph.from_dict(raw)

    def test_from_dict_does_not_coerce_list_params_into_empty_dict(self):
        """A raw `params` value of [] (a list, not a dict) must raise
        ValueError (params must be a dict) -- NOT silently become {} via
        an implementation that does dict(raw_params)."""
        raw = {
            "events": [
                {"id": "e1", "type": "fracture", "target": "", "at_frame": 0,
                 "params": [], "trigger": {}, "causes": []},
            ],
            "nodes": 1, "edges": 0,
        }
        with pytest.raises(ValueError):
            EventGraph.from_dict(raw)

    def test_from_dict_does_not_coerce_list_trigger_into_empty_dict(self):
        """A raw `trigger` value of [] (a list, not a dict) must raise
        ValueError (trigger must be a dict) -- NOT silently become {} via
        an implementation that does dict(raw_trigger), which would also
        bypass the trigger-kind vocabulary validation entirely."""
        raw = {
            "events": [
                {"id": "e1", "type": "fracture", "target": "", "at_frame": 0,
                 "params": {}, "trigger": [], "causes": []},
            ],
            "nodes": 1, "edges": 0,
        }
        with pytest.raises(ValueError):
            EventGraph.from_dict(raw)

    def test_from_dict_rebuilds_events_preserving_params(self):
        raw = {
            "events": [
                {"id": "e1", "type": "fracture", "target": "/obj/wall", "at_frame": 40,
                 "params": {}, "trigger": {"kind": "stress_gt", "value": 1200}, "causes": ["e2"]},
                {"id": "e2", "type": "emit", "target": "/obj/debris", "at_frame": 0,
                 "params": {"rate": 500.0}, "trigger": {}, "causes": []},
            ],
            "nodes": 2, "edges": 1,
        }
        g = EventGraph.from_dict(raw)
        assert len(g.events) == 2
        assert all(isinstance(e, EventSpec) for e in g.events)
        by_id = {e.id: e for e in g.events}
        assert by_id["e1"].causes == ["e2"]
        assert by_id["e2"].params == {"rate": 500.0}

    def test_from_dict_ignores_unknown_top_level_keys(self):
        raw = {
            "events": [
                {"id": "e1", "type": "fracture", "target": "", "at_frame": 0, "params": {}, "trigger": {}, "causes": []},
            ],
            "nodes": 1, "edges": 0, "future_field": 123,
        }
        g = EventGraph.from_dict(raw)  # must not raise
        assert len(g.events) == 1

    def test_from_dict_propagates_bad_event_type_error(self):
        raw = {
            "events": [
                {"id": "e1", "type": "not_a_real_event", "target": "", "at_frame": 0, "params": {}, "trigger": {}, "causes": []},
            ],
            "nodes": 1, "edges": 0,
        }
        with pytest.raises(ValueError):
            EventGraph.from_dict(raw)

    def test_round_trip_preserves_params_and_causes(self):
        g = EventGraph(events=[
            EventSpec(id="e1", type="fracture", target="/obj/wall", at_frame=40,
                      trigger={"kind": "stress_gt", "value": 1200}, causes=["e2"]),
            EventSpec(id="e2", type="emit", target="/obj/debris", params={"rate": 500.0}),
        ])
        rebuilt = EventGraph.from_dict(g.to_dict())
        assert rebuilt.to_dict() == g.to_dict()

    def test_is_json_serialisable(self):
        g = EventGraph(events=[EventSpec(id="e1", type="fracture", causes=["e2"]),
                                EventSpec(id="e2", type="emit")])
        json.dumps(g.to_dict())


# ===========================================================================
# Section 3 — describe_sim_events() -> dict (FR-C)
#
# Locked contract: EXACTLY {events:[{type,context,params}...], triggers:
# [...], assertions:[...]} per SPEC 4.1. events lists the 5 EVENT_VOCAB
# types each with its context (fracture->rbd, emit->pop|pyro, tear->
# vellum, ignite->pyro, keyframe->any); triggers = the 4 TRIGGER_VOCAB
# names; assertions = the 7-member assertion-metric vocabulary.
# ===========================================================================

class TestDescribeSimEvents:
    """describe_sim_events — the exact FR-C wire shape."""

    def test_returns_exact_top_level_keys(self):
        result = describe_sim_events()
        assert set(result.keys()) == {"events", "triggers", "assertions"}, (
            f"describe_sim_events() must return exactly {{events, triggers, assertions}}, "
            f"got keys {set(result.keys())!r}"
        )

    def test_all_five_event_types_present_with_correct_context(self):
        result = describe_sim_events()
        context_by_type = {e["type"]: e["context"] for e in result["events"]}
        assert context_by_type == {
            "fracture": "rbd",
            "emit": "pop|pyro",
            "tear": "vellum",
            "ignite": "pyro",
            "keyframe": "any",
        }, f"describe_sim_events() type->context mapping mismatch: {context_by_type!r}"

    def test_each_event_entry_has_type_context_params_keys(self):
        result = describe_sim_events()
        for entry in result["events"]:
            assert set(entry.keys()) == {"type", "context", "params"}, (
                f"Unexpected keys in a describe_sim_events() event entry: {set(entry.keys())!r}"
            )
            assert isinstance(entry["params"], dict)

    def test_events_have_exact_spec_4_1_params_per_type(self):
        """B1 (REVISION 3 FOLD, the load-bearing pin): each event's
        `params` must be the SPEC 4.1 LITERAL per-event params dict
        EXACTLY -- not merely 'a dict', and not a differently-shaped
        params schema of the implementation's own invention. The param
        VALUES are the spec's own type-hint strings, verbatim, per
        spec.md Section 4.1's describe_sim_events() worked example."""
        result = describe_sim_events()
        params_by_type = {e["type"]: e["params"] for e in result["events"]}
        assert params_by_type == {
            "fracture": {"at_frame": "int", "trigger": "str?"},
            "emit": {"at_frame": "int", "rate": "float"},
            "tear": {"threshold": "float"},
            "ignite": {"at_frame": "int", "region": "str?"},
            "keyframe": {"node": "str", "parm": "str", "frames": "[[f,v]]"},
        }, (
            f"describe_sim_events() params drift from the SPEC 4.1 literal "
            f"per-event params: {params_by_type!r}"
        )

    def test_four_triggers_present(self):
        result = describe_sim_events()
        assert set(result["triggers"]) == {"stress_gt", "collision_with", "frame_eq", "field_gt"}, (
            f"describe_sim_events() must list exactly the 4 trigger vocabulary names, "
            f"got {set(result['triggers'])!r}"
        )

    def test_seven_assertion_metrics_present(self):
        result = describe_sim_events()
        assert set(result["assertions"]) == {
            "piece_count", "constraint_count", "point_count", "velocity_bounds",
            "bbox_over_time", "field_stats", "mass_conservation",
        }, f"describe_sim_events() must list exactly the 7 assertion-metric names, got {set(result['assertions'])!r}"

    def test_order_is_stable_across_calls(self):
        r1 = describe_sim_events()
        r2 = describe_sim_events()
        assert [e["type"] for e in r1["events"]] == [e["type"] for e in r2["events"]]
        assert r1["triggers"] == r2["triggers"]
        assert r1["assertions"] == r2["assertions"]

    def test_is_json_serialisable(self):
        json.dumps(describe_sim_events())


# ===========================================================================
# Section 4 — aggregate_assertion(series, expect) -> dict
#              max / min predicates (global extremum + EARLIEST frame)
# ===========================================================================

class TestAggregateAssertionMaxMin:
    """max/min predicates: report the GLOBAL extremum and its EARLIEST
    occurrence frame (not the first-encountered value that happens to
    equal the extremum after further samples, and not a later-occurring
    tie)."""

    def test_max_breach_reports_global_max_and_earliest_frame(self):
        """410 occurs at frame 20 AND frame 40 (a tie) -- the diagnostic
        at_frame must be the EARLIEST occurrence (20), not the last (40).
        M4 (REVISION 3 FOLD): since exactly ONE frame-bearing predicate
        is present here, the spec-compatible bare `at_frame` is ALSO
        emitted (spec 4.1's velocity_bounds example shows a bare
        at_frame) -- but the namespaced `max_at_frame` MUST also be
        present so a composite expect never has to disambiguate."""
        series = [[10, 100], [20, 410], [30, 200], [40, 410]]
        result = aggregate_assertion(series, {"max": 250})
        assert result["pass"] is False
        assert result["max_seen"] == 410
        assert result["at_frame"] == 20
        assert result["max_at_frame"] == 20

    def test_max_within_bound_passes(self):
        series = [[1, 10], [2, 50], [3, 30]]
        result = aggregate_assertion(series, {"max": 100})
        assert result["pass"] is True

    def test_min_breach_reports_global_min_and_earliest_frame(self):
        """M4: single-predicate case emits both the bare `at_frame`
        (spec-compat) and the namespaced `min_at_frame`."""
        series = [[1, 5], [2, 1], [3, 1], [4, 10]]
        result = aggregate_assertion(series, {"min": 2})
        assert result["pass"] is False
        assert result["min_seen"] == 1
        assert result["at_frame"] == 2
        assert result["min_at_frame"] == 2

    def test_min_within_bound_passes(self):
        series = [[1, 5], [2, 3]]
        result = aggregate_assertion(series, {"min": 1})
        assert result["pass"] is True

    def test_max_and_min_together_use_and_semantics(self):
        """A metric passes iff EVERY predicate key present holds."""
        series = [[1, 10], [2, 50], [3, 30]]
        assert aggregate_assertion(series, {"max": 100, "min": 5})["pass"] is True
        assert aggregate_assertion(series, {"max": 40, "min": 5})["pass"] is False


# ===========================================================================
# Section 5 — aggregate_assertion: jump_gt (Blocker-1, the LOAD-BEARING pin)
#
# jump_gt:N with at_frame F -> before = value at the GREATEST sampled
# frame strictly < F (if none, the first sample's value); scan samples
# with frame >= F in ASCENDING order; pass iff the FIRST sample whose
# (value - before) > N + _TOL exists; else pass:false (NOT an error).
# ===========================================================================

class TestAggregateAssertionJumpGt:
    """jump_gt -- pins the SPEC 4.1 worked example exactly, plus the
    before/after scan edge cases the rev-2 Blocker-1 fold corrected."""

    def test_spec_4_1_example_passes(self):
        """THE load-bearing pin: SPEC 4.1's own worked example
        ([[38,1],[40,1],[41,73]] / {at_frame:40,jump_gt:50}) MUST PASS
        ('fractured at 41'). A naive before='value at F' rule (rev-1,
        Blocker-1) would compare 73-1=72>50 against a before of 1 (the
        SAME as the corrected rule here, since frame 40's value IS 1) --
        but the rule must generalize to the harder case below where
        before != the value exactly at F."""
        series = [[38, 1], [40, 1], [41, 73]]
        result = aggregate_assertion(series, {"at_frame": 40, "jump_gt": 50})
        assert result["pass"] is True

    def test_before_uses_greatest_frame_strictly_less_than_at_frame(self):
        """before must come from the greatest frame STRICTLY < F, not
        from the sample AT F itself, when both exist and differ."""
        # before (frame 38, value 1) is used; frame 40's OWN value (5) is
        # NOT before -- scanning from frame>=40 ascending: 40:(5-1)=4 not
        # >50; 41:(80-1)=79>50 -> passes at frame 41.
        series = [[38, 1], [40, 5], [41, 80]]
        result = aggregate_assertion(series, {"at_frame": 40, "jump_gt": 50})
        assert result["pass"] is True

    def test_before_uses_first_sample_when_no_earlier_frame_exists(self):
        """No frame strictly < 40 exists -- before falls back to the
        FIRST sample's value (at frame 40 itself, value 1)."""
        series = [[40, 1], [41, 73]]
        result = aggregate_assertion(series, {"at_frame": 40, "jump_gt": 50})
        assert result["pass"] is True

    def test_no_qualifying_jump_after_at_frame_returns_pass_false_not_raise(self):
        series = [[38, 1], [40, 1], [41, 10]]
        result = aggregate_assertion(series, {"at_frame": 40, "jump_gt": 50})
        assert result["pass"] is False  # must NOT raise

    def test_no_sample_at_or_after_at_frame_returns_pass_false_not_raise(self):
        series = [[10, 1], [20, 2]]
        result = aggregate_assertion(series, {"at_frame": 100, "jump_gt": 5})
        assert result["pass"] is False  # must NOT raise


# ===========================================================================
# Section 6 — aggregate_assertion: tolerance (mass_conservation idiom)
#
# tolerance:t -> value0 = first frame's value; drift = max over frames of
# abs(value-value0)/abs(value0), EXCEPT value0==0 -> drift = max over
# frames of abs(value) (absolute drift, div-by-zero guard); pass iff
# drift <= t + _TOL.
# ===========================================================================

class TestAggregateAssertionTolerance:
    """tolerance -- relative drift from the first sample, with the
    value0==0 absolute-drift guard (div-by-zero safety)."""

    def test_relative_drift_breach(self):
        series = [[1, 10.0], [2, 10.5], [3, 12.0]]
        result = aggregate_assertion(series, {"tolerance": 0.05})
        assert result["pass"] is False
        assert result["drift"] == pytest.approx(0.2, abs=1e-6)

    def test_relative_drift_within_bound_passes(self):
        series = [[1, 10.0], [2, 10.1]]
        result = aggregate_assertion(series, {"tolerance": 0.05})
        assert result["pass"] is True

    def test_value0_zero_uses_absolute_drift_no_div_by_zero(self):
        """value0 == 0 -- drift must be the ABSOLUTE max |value|, never a
        ZeroDivisionError from a relative-drift computation."""
        series = [[1, 0.0], [2, 5.0], [3, -3.0]]
        result = aggregate_assertion(series, {"tolerance": 1.0})
        assert result["pass"] is False
        assert result["drift"] == pytest.approx(5.0, abs=1e-6)

    def test_value0_zero_within_absolute_tolerance_passes(self):
        series = [[1, 0.0], [2, 0.5], [3, -0.3]]
        result = aggregate_assertion(series, {"tolerance": 1.0})
        assert result["pass"] is True


# ===========================================================================
# Section 7 — aggregate_assertion: max_gt / eq
# ===========================================================================

class TestAggregateAssertionMaxGtEq:
    """max_gt: pass iff the series max exceeds x. eq: pass iff the value
    at the least sampled frame >= F matches v within tolerance; a missing
    frame returns pass:false with a note (NOT an error)."""

    def test_max_gt_passes(self):
        series = [[1, 1], [2, 2], [3, 3]]
        result = aggregate_assertion(series, {"max_gt": 2})
        assert result["pass"] is True

    def test_max_gt_fails(self):
        series = [[1, 1], [2, 2]]
        result = aggregate_assertion(series, {"max_gt": 5})
        assert result["pass"] is False

    def test_eq_passes_at_least_sampled_frame_gte_f(self):
        series = [[10, 5.0], [20, 5.0], [30, 7.0]]
        result = aggregate_assertion(series, {"at_frame": 20, "eq": 5.0})
        assert result["pass"] is True

    def test_eq_fails_when_value_mismatches(self):
        """M4: single-predicate case -- eq also emits the namespaced
        eq_at_frame/eq_observed diagnostic keys."""
        series = [[10, 5.0], [20, 5.0], [30, 7.0]]
        result = aggregate_assertion(series, {"at_frame": 20, "eq": 6.0})
        assert result["pass"] is False  # must NOT raise
        assert result["eq_at_frame"] == 20
        assert result["eq_observed"] == 5.0

    def test_eq_no_sample_at_or_after_frame_returns_pass_false_with_note_not_raise(self):
        series = [[10, 5.0], [20, 6.0]]
        result = aggregate_assertion(series, {"at_frame": 50, "eq": 5.0})
        assert result["pass"] is False  # must NOT raise
        assert "note" in result and isinstance(result["note"], str) and result["note"], (
            "an eq predicate whose at_frame has no sample >= F must report a "
            "non-empty note naming the missing requested frame"
        )
        assert "50" in result["note"], (
            f"the eq missing-frame note must name the missing requested frame (50), got {result['note']!r}"
        )


# ===========================================================================
# Section 7b — aggregate_assertion: COMPOSITE-expect diagnostics (M4,
#              REVISION 3 FOLD -- the load-bearing collision fix flagged
#              by BOTH the Codex and Claude reviewers). A composite
#              expect (2+ frame-bearing predicates) must emit ONLY
#              namespaced per-predicate diagnostic keys -- never a
#              shared bare `at_frame` that the second predicate would
#              silently overwrite.
# ===========================================================================

class TestAggregateAssertionCompositeDiagnostics:
    """M4: composite expects (2+ frame-bearing predicates present) must
    not collide on a shared bare `at_frame` key."""

    def test_composite_max_and_min_emit_only_namespaced_keys_no_bare_at_frame(self):
        """THE load-bearing pin: max breaches at frame 1 (value 100 > 50)
        and min breaches at frame 2 (value 0 < 10) -- a rev-1-style
        implementation that writes a SHARED `at_frame` key would have
        the min predicate's write (frame 2) silently overwrite the max
        predicate's write (frame 1), losing the max diagnostic. The
        namespaced keys must each carry their OWN correct frame, and NO
        bare `at_frame` key may be present at all (it would be
        ambiguous which predicate it belongs to)."""
        result = aggregate_assertion([[1, 100], [2, 0]], {"max": 50, "min": 10})
        assert result["pass"] is False
        assert result["max_seen"] == 100
        assert result["max_at_frame"] == 1
        assert result["min_seen"] == 0
        assert result["min_at_frame"] == 2
        assert "at_frame" not in result, (
            "a COMPOSITE expect (2+ frame-bearing predicates) must emit ONLY "
            "namespaced diagnostic keys -- a bare 'at_frame' key must not be "
            "present when it would be ambiguous which predicate it names, "
            f"got keys {set(result.keys())!r}"
        )


# ===========================================================================
# Section 8 — aggregate_assertion: validation errors (malformed series /
#              invalid expect) -- construction/validation raises, NEVER a
#              mere expectation failure.
# ===========================================================================

class TestAggregateAssertionValidation:
    """Malformed series or an invalid expect RAISES ValueError. A
    well-formed series that simply fails its well-formed expectation
    returns pass:false and does NOT raise (see prior sections)."""

    def test_empty_series_raises(self):
        with pytest.raises(ValueError):
            aggregate_assertion([], {"max": 100})

    def test_non_ascending_series_raises(self):
        with pytest.raises(ValueError):
            aggregate_assertion([[2, 1], [1, 2]], {"max": 100})

    def test_duplicate_frame_raises(self):
        with pytest.raises(ValueError):
            aggregate_assertion([[1, 1], [1, 2]], {"max": 100})

    def test_bool_frame_raises(self):
        with pytest.raises(ValueError):
            aggregate_assertion([[True, 1], [2, 2]], {"max": 100})

    def test_bool_value_raises(self):
        with pytest.raises(ValueError):
            aggregate_assertion([[1, True], [2, False]], {"max": 100})

    def test_non_scalar_value_list_raises(self):
        with pytest.raises(ValueError):
            aggregate_assertion([[1, [2, 3]]], {"max": 100})

    def test_non_scalar_value_dict_raises(self):
        with pytest.raises(ValueError):
            aggregate_assertion([[1, {"x": 1}]], {"max": 100})

    def test_non_pair_series_item_raises(self):
        with pytest.raises(ValueError):
            aggregate_assertion([[1, 2, 3]], {"max": 100})
        with pytest.raises(ValueError):
            aggregate_assertion([1, 2], {"max": 100})

    def test_expect_not_a_dict_raises(self):
        with pytest.raises(ValueError):
            aggregate_assertion([[1, 1], [2, 2]], None)
        with pytest.raises(ValueError):
            aggregate_assertion([[1, 1], [2, 2]], ["max", 5])

    def test_expect_unknown_key_raises(self):
        with pytest.raises(ValueError):
            aggregate_assertion([[1, 1], [2, 2]], {"mx": 250})

    def test_expect_zero_predicate_keys_raises(self):
        """at_frame alone is NOT a predicate -- an expect with zero
        predicate keys must raise."""
        with pytest.raises(ValueError):
            aggregate_assertion([[1, 1], [2, 2]], {"at_frame": 10})

    def test_expect_jump_gt_missing_at_frame_raises(self):
        with pytest.raises(ValueError):
            aggregate_assertion([[1, 1], [2, 2]], {"jump_gt": 50})

    def test_expect_eq_missing_at_frame_raises(self):
        with pytest.raises(ValueError):
            aggregate_assertion([[1, 1], [2, 2]], {"eq": 5.0})


# ===========================================================================
# Section 9 — aggregate_assertion: output normalization + JSON safety
# ===========================================================================

class TestAggregateAssertionOutputNormalization:
    """Outputs normalize frames via int(frame) and values via
    float(value); everything is JSON-serializable plain Python types."""

    def test_series_normalizes_frame_to_int_and_value_to_float(self):
        series = [[1, 5], [2, 10]]  # int values in
        result = aggregate_assertion(series, {"max": 100})
        assert result["series"] == [[1, 5.0], [2, 10.0]]
        for frame, value in result["series"]:
            assert type(frame) is int
            assert type(value) is float

    def test_output_is_plain_json_safe_on_pass(self):
        result = aggregate_assertion([[1, 1.0], [2, 2.0]], {"max": 100.0})
        _assert_plain_json_types(result)
        json.dumps(result)

    def test_output_is_plain_json_safe_on_breach(self):
        result = aggregate_assertion([[10, 100], [20, 410]], {"max": 250})
        _assert_plain_json_types(result)
        json.dumps(result)

    def test_output_is_plain_json_safe_with_drift(self):
        result = aggregate_assertion([[1, 10.0], [2, 10.08]], {"tolerance": 0.02})
        _assert_plain_json_types(result)
        json.dumps(result)


# ===========================================================================
# Section 10 — evaluate_assertions(assertions) -> dict (FR-B, the
#               assert_simulation model core; exact SPEC 4.1 return)
#
# Locked contract: input list of {metric, series, expect, ...}. Validates
# each metric against the 7-member vocabulary (unknown -> ValueError);
# each assertion MUST contain metric/series/expect. Processes and returns
# `results` in INPUT order (Major-11). Returns EXACTLY {results:[...],
# pass: bool} where pass is the AND over all per-metric pass (empty ->
# pass:true, results:[]). Handler-only extra top-level keys (node, field)
# are IGNORED and NOT copied into result entries (Major-11).
# ===========================================================================

class TestEvaluateAssertions:
    """evaluate_assertions -- the assert_simulation model core."""

    def test_empty_assertions_returns_pass_true_empty_results(self):
        result = evaluate_assertions([])
        assert result == {"results": [], "pass": True}

    def test_ands_all_metrics_true_when_all_pass(self):
        assertions = [
            {"metric": "velocity_bounds", "series": [[1, 10], [2, 50]], "expect": {"max": 100}},
            {"metric": "mass_conservation", "series": [[1, 10.0], [2, 10.05]], "expect": {"tolerance": 0.1}},
        ]
        result = evaluate_assertions(assertions)
        assert result["pass"] is True
        assert set(result.keys()) == {"results", "pass"}

    def test_pass_is_false_if_any_metric_fails(self):
        assertions = [
            {"metric": "velocity_bounds", "series": [[1, 10], [2, 50]], "expect": {"max": 100}},
            {"metric": "mass_conservation", "series": [[1, 10.0], [2, 20.0]], "expect": {"tolerance": 0.01}},
        ]
        result = evaluate_assertions(assertions)
        assert result["pass"] is False

    def test_results_are_in_input_order_not_sorted(self):
        """Input order deliberately differs from alphabetical AND from
        any plausible vocabulary-declaration order, so a sort-by-name or
        sort-by-vocab-position bug would be caught."""
        assertions = [
            {"metric": "velocity_bounds", "series": [[1, 1], [2, 2]], "expect": {"max_gt": -1}},
            {"metric": "piece_count", "series": [[1, 1], [2, 2]], "expect": {"max_gt": -1}},
            {"metric": "mass_conservation", "series": [[1, 1.0], [2, 1.0]], "expect": {"max_gt": -1}},
        ]
        result = evaluate_assertions(assertions)
        assert [r["metric"] for r in result["results"]] == [
            "velocity_bounds", "piece_count", "mass_conservation",
        ]

    def test_ignores_handler_only_extra_keys_node_field(self):
        """node/field are handler-only routing keys -- they must NOT leak
        into the per-metric result entry."""
        assertions = [
            {"metric": "field_stats", "node": "/obj/pyro", "field": "density",
             "series": [[1, 0.1], [2, 0.2]], "expect": {"max_gt": 0.05}},
        ]
        result = evaluate_assertions(assertions)
        entry = result["results"][0]
        assert set(entry.keys()) == {"metric", "pass", "series"}, (
            f"result entry must not leak handler-only keys (node/field), got {set(entry.keys())!r}"
        )
        assert "node" not in entry
        assert "field" not in entry

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            evaluate_assertions([{"metric": "not_a_real_metric", "series": [[1, 1]], "expect": {"max_gt": 0}}])

    def test_assertion_missing_metric_raises(self):
        with pytest.raises(ValueError):
            evaluate_assertions([{"series": [[1, 1]], "expect": {"max_gt": 0}}])

    def test_assertion_missing_series_raises(self):
        with pytest.raises(ValueError):
            evaluate_assertions([{"metric": "piece_count", "expect": {"max_gt": 0}}])

    def test_assertion_missing_expect_raises(self):
        with pytest.raises(ValueError):
            evaluate_assertions([{"metric": "piece_count", "series": [[1, 1]]}])

    def test_all_seven_metric_names_are_valid(self):
        for metric in (
            "piece_count", "constraint_count", "point_count", "velocity_bounds",
            "bbox_over_time", "field_stats", "mass_conservation",
        ):
            result = evaluate_assertions([
                {"metric": metric, "series": [[1, 0.0], [2, 1.0]], "expect": {"max_gt": -1}},
            ])
            assert result["results"][0]["metric"] == metric

    def test_spec_4_1_combined_scenario(self):
        """Mirrors the SPEC 4.1 worked assert_simulation example: a
        passing piece_count jump, a FAILING velocity_bounds breach, and a
        passing mass_conservation drift -- overall pass:false because one
        assertion fails (the AND semantics), exactly as SPEC 4.1 shows."""
        assertions = [
            {"metric": "piece_count", "node": "/obj/rbd_sim/wall",
             "series": [[38, 1], [40, 1], [41, 73]], "expect": {"at_frame": 40, "jump_gt": 50}},
            {"metric": "velocity_bounds", "series": [[10, 50], [52, 410]], "expect": {"max": 250}},
            {"metric": "mass_conservation", "series": [[1, 10.0], [2, 10.08]], "expect": {"tolerance": 0.02}},
        ]
        result = evaluate_assertions(assertions)
        assert result["pass"] is False
        by_metric = {r["metric"]: r for r in result["results"]}
        assert by_metric["piece_count"]["pass"] is True
        assert by_metric["velocity_bounds"]["pass"] is False
        assert by_metric["velocity_bounds"]["max_seen"] == 410
        assert by_metric["velocity_bounds"]["at_frame"] == 52
        assert by_metric["mass_conservation"]["pass"] is True

    def test_output_is_plain_json_safe(self):
        assertions = [
            {"metric": "piece_count", "series": [[38, 1], [41, 73]], "expect": {"at_frame": 38, "jump_gt": 1}},
        ]
        result = evaluate_assertions(assertions)
        _assert_plain_json_types(result)
        json.dumps(result)


# ===========================================================================
# Section 11 — round-trip + determinism, over a representative input set
#              (property-style coverage; Hypothesis is not installed in
#              this venv, see module docstring)
# ===========================================================================

def _sample_graphs():
    """Representative EventGraph configurations spanning: empty, a single
    event, a fracture->emit causal chain, and a lossless-params event --
    used as the input space for round-trip checks."""
    return [
        EventGraph(),
        EventGraph(events=[EventSpec(id="solo", type="keyframe", params={"frames": [[1, 0.0]]})]),
        EventGraph(events=[
            EventSpec(id="e1", type="fracture", target="/obj/wall", at_frame=40,
                      trigger={"kind": "stress_gt", "value": 1200}, causes=["e2"]),
            EventSpec(id="e2", type="emit", target="/obj/debris", params={"rate": 500.0}),
        ]),
        EventGraph(events=[
            EventSpec(id="e1", type="ignite", params={"region": "north_wall"}),
            EventSpec(id="e2", type="tear", target="/obj/cloth", params={"threshold": 0.75}),
            EventSpec(id="e3", type="keyframe", params={"node": "/obj/x", "parm": "tx", "frames": [[1, 0.0], [10, 5.0]]}),
        ]),
    ]


class TestRoundTripProperties:
    """Parametrized property-style coverage over the round-trip invariant
    (stands in for a Hypothesis strategy)."""

    @pytest.mark.parametrize("graph", _sample_graphs())
    def test_from_dict_to_dict_round_trips_exactly(self, graph):
        rebuilt = EventGraph.from_dict(graph.to_dict())
        assert rebuilt.to_dict() == graph.to_dict(), (
            "EventGraph.from_dict(graph.to_dict()) must round-trip to an "
            "identical to_dict() output"
        )


# ===========================================================================
# Section 12 — Module purity (CL-015 extended): no hou/Qt/pxr/FastMCP
#              imports, no MCP call, no cook/DOP read.
# ===========================================================================

class TestModulePurity:
    """temporal_reasoning_model.py must import NO hou, Qt/PySide6, pxr, or
    FastMCP, and must never make an MCP call or a cook/DOP read (CL-015
    extended / plan Scope guard)."""

    def test_source_has_no_forbidden_imports_or_calls(self):
        import fxhoudinimcp.temporal_reasoning_model as mod
        with open(mod.__file__, "r", encoding="utf-8") as f:
            source = f.read()
        forbidden_tokens = [
            "import hou",
            "from hou",
            "import PySide6",
            "import PySide2",
            "import pxr",
            "from pxr",
            "import fastmcp",
            "from fastmcp",
            "FastMCP",
            "mcp__fxhoudini",
        ]
        for token in forbidden_tokens:
            assert token not in source, (
                f"temporal_reasoning_model.py must not contain {token!r} "
                f"(CL-015 extended / PR-1 scope guard)"
            )


# ===========================================================================
# Section 13 — compile_plan(events, network=None) -> dict
#              (PP12-117 PR-3, the AUTHORING-half pure translation,
#              NARROWED SCOPE per plan pp12-117c lockedFieldContract
#              REVISION 2)
#
# TDD phase: RED — compile_plan does NOT exist yet on
# temporal_reasoning_model.py. Every test below imports compile_plan
# LOCALLY (via _get_compile_plan(), never added to the module-level import
# block at the top of this file) so the ImportError is scoped to THESE
# new tests only — the PR-1/PR-2 tests above (Sections 1-12) must stay
# green throughout this red phase.
#
# NARROWED SCOPE (rev-2, per the adversarial fold): compile_plan compiles
# ONLY two grounded event shapes into compiled.keyframes — (1) a bare
# keyframe event, (2) an activation event (emit/fracture/ignite/tear) that
# carries an EXPLICIT params.parm + EXPLICIT params.frames. Everything
# else (type-inferred activation, any threshold-trigger event, a
# causally-impossible frame edge) routes to unresolved[] — NEVER raised,
# NEVER invented. chop_triggers/dop_parms stay [] always (their content is
# a later, DEFERRED PR). A malformed event (bad type, cyclic/dangling
# causes) still raises ValueError — reusing PR-1's EventGraph validation
# unchanged.
#
# Cross-references:
#   - Plan pp12-117c lockedFieldContract (BINDING, revision 2)
#   - docs/portfolio_projects/PP12_houdini_mcp_agentic_bridge/
#     117_mcp_temporal_sim_reasoning_surface/spec.md Section 4.1
#   - tdd-with-agents.md Sec.4/Sec.2: hou-test writes red (public-contract
#     assertions only, never a mirror test on impl internals)
#   - CL-015 (extended): compile_plan is pure — no hou/Qt/pxr/MCP call
# ===========================================================================

def _get_compile_plan():
    """Local (not top-of-file) import of compile_plan — scopes the RED
    ImportError to only the tests in this section, keeping the PR-1/PR-2
    suite above green while this PR-3 addition is red."""
    from fxhoudinimcp.temporal_reasoning_model import compile_plan
    return compile_plan


class TestCompilePlanKeyframeEvent:
    """A bare keyframe event compiles DIRECTLY into compiled.keyframes —
    the SPEC 4.1 {compiled:{keyframes,chop_triggers,dop_parms},
    event_graph, unresolved} shape, exactly."""

    def test_keyframe_event_compiles_to_exact_shape(self):
        compile_plan = _get_compile_plan()
        events = [{
            "id": "kf1", "type": "keyframe", "target": "/obj/rbd_sim",
            "params": {"parm": "tx", "frames": [[1, 0], [10, 5]]},
        }]
        result = compile_plan(events)
        assert result["compiled"]["keyframes"] == [
            {"node": "/obj/rbd_sim", "parm": "tx", "frames": [[1, 0], [10, 5]]},
        ], f"expected a single compiled keyframe entry, got {result['compiled']!r}"
        assert result["compiled"]["chop_triggers"] == [], (
            f"chop_triggers must ALWAYS be [] in PR-3 (deferred), got "
            f"{result['compiled']['chop_triggers']!r}"
        )
        assert result["compiled"]["dop_parms"] == [], (
            f"dop_parms must ALWAYS be [] in PR-3 (deferred), got "
            f"{result['compiled']['dop_parms']!r}"
        )
        assert result["event_graph"] == {"nodes": 1, "edges": 0}
        assert result["unresolved"] == []

    def test_top_level_result_has_exact_keys(self):
        compile_plan = _get_compile_plan()
        events = [{"id": "kf1", "type": "keyframe", "target": "/obj/x",
                   "params": {"parm": "tx", "frames": [[1, 0.0]]}}]
        result = compile_plan(events)
        assert set(result.keys()) == {"compiled", "event_graph", "unresolved"}, (
            f"compile_plan must return exactly {{compiled, event_graph, unresolved}}, "
            f"got {set(result.keys())!r}"
        )
        assert set(result["compiled"].keys()) == {"keyframes", "chop_triggers", "dop_parms"}, (
            f"compiled must have exactly {{keyframes, chop_triggers, dop_parms}}, "
            f"got {set(result['compiled'].keys())!r}"
        )

    def test_multiple_independent_keyframe_events_all_compile(self):
        compile_plan = _get_compile_plan()
        events = [
            {"id": "kf1", "type": "keyframe", "target": "/obj/a",
             "params": {"parm": "tx", "frames": [[1, 0]]}},
            {"id": "kf2", "type": "keyframe", "target": "/obj/b",
             "params": {"parm": "ty", "frames": [[5, 2]]}},
        ]
        result = compile_plan(events)
        assert len(result["compiled"]["keyframes"]) == 2
        nodes = {kf["node"] for kf in result["compiled"]["keyframes"]}
        assert nodes == {"/obj/a", "/obj/b"}
        assert result["unresolved"] == []

    def test_compile_plan_accepts_optional_network_argument(self):
        """compile_plan is PURE translation — it does NOT verify nodes/
        parms exist or check network-scope (that is the handler's
        apply-time job), so passing `network` must not change the
        compiled output."""
        compile_plan = _get_compile_plan()
        events = [{"id": "kf1", "type": "keyframe", "target": "/obj/x",
                   "params": {"parm": "tx", "frames": [[1, 0.0]]}}]
        result_no_network = compile_plan(events)
        result_with_network = compile_plan(events, network="/obj/rbd_sim")
        assert result_no_network["compiled"] == result_with_network["compiled"]
        assert result_no_network["unresolved"] == result_with_network["unresolved"]


class TestCompilePlanExplicitParmActivation:
    """An EXPLICIT-parm-EXPLICIT-frames activation event (emit/fracture/
    ignite/tear) compiles to a compiled.keyframes entry EXACTLY like a
    keyframe event — the agent authored the exact parm + explicit frames
    (e.g. an on/off pair) and compile_plan does NOT synthesize on/off
    defaults (Major-5)."""

    @pytest.mark.parametrize("event_type", ["emit", "fracture", "ignite", "tear"])
    def test_explicit_parm_activation_compiles_to_keyframe_entry(self, event_type):
        compile_plan = _get_compile_plan()
        events = [{
            "id": "act1", "type": event_type, "target": "/obj/rbd_sim/wall",
            "params": {"parm": "activation", "frames": [[4, 0], [5, 1]]},
        }]
        result = compile_plan(events)
        assert result["compiled"]["keyframes"] == [
            {"node": "/obj/rbd_sim/wall", "parm": "activation", "frames": [[4, 0], [5, 1]]},
        ], (
            f"expected the explicit-parm activation ({event_type!r}) to compile "
            f"directly into a keyframe entry, got {result['compiled']!r}"
        )
        assert result["unresolved"] == []


class TestCompilePlanActivationWithoutExplicitParmDeferred:
    """DEFERRED (Blocker-2): an activation event WITHOUT an explicit
    params.parm — type-inference is out of scope for PR-3 — routes to
    unresolved. Likewise a params.parm present but params.frames absent
    or empty."""

    def test_activation_missing_parm_is_unresolved(self):
        compile_plan = _get_compile_plan()
        events = [{"id": "e1", "type": "emit", "target": "/obj/x",
                   "params": {"rate": 500.0}}]  # no 'parm' — type-inference deferred
        result = compile_plan(events)
        assert result["compiled"]["keyframes"] == []
        assert result["unresolved"] == ["e1"]

    def test_activation_with_parm_but_no_frames_key_is_unresolved(self):
        compile_plan = _get_compile_plan()
        events = [{"id": "e1", "type": "ignite", "target": "/obj/x",
                   "params": {"parm": "activation"}}]  # no 'frames' key at all
        result = compile_plan(events)
        assert result["compiled"]["keyframes"] == []
        assert result["unresolved"] == ["e1"]

    def test_activation_with_empty_frames_list_is_unresolved(self):
        compile_plan = _get_compile_plan()
        events = [{"id": "e1", "type": "tear", "target": "/obj/cloth",
                   "params": {"parm": "threshold", "frames": []}}]
        result = compile_plan(events)
        assert result["compiled"]["keyframes"] == []
        assert result["unresolved"] == ["e1"]

    def test_activation_with_non_string_parm_is_unresolved(self):
        """params.parm must be a non-empty str — a non-str value is not a
        valid explicit parm name, so this must degrade to unresolved
        (never raise — the event is well-formed EventSpec-wise, just
        unmappable)."""
        compile_plan = _get_compile_plan()
        events = [{"id": "e1", "type": "emit", "target": "/obj/x",
                   "params": {"parm": 123, "frames": [[1, 0]]}}]
        result = compile_plan(events)
        assert result["unresolved"] == ["e1"]


class TestCompilePlanThresholdTriggerDeferred:
    """DEFERRED: ANY event whose trigger.kind is in {stress_gt, field_gt,
    collision_with} routes to unresolved — NEVER raised, NEVER compiled —
    and chop_triggers stays [] (their compilation is a later PR)."""

    @pytest.mark.parametrize("kind", ["stress_gt", "field_gt", "collision_with"])
    def test_threshold_trigger_is_unresolved_not_raised(self, kind):
        compile_plan = _get_compile_plan()
        events = [{
            "id": "e1", "type": "fracture", "target": "/obj/wall",
            "trigger": {"kind": kind, "value": 1200},
        }]
        result = compile_plan(events)  # must NOT raise
        assert result["compiled"]["keyframes"] == []
        assert result["compiled"]["chop_triggers"] == []
        assert result["unresolved"] == ["e1"]

    def test_threshold_trigger_is_unresolved_even_with_explicit_parm_and_frames(self):
        """A stress_gt/field_gt/collision_with trigger ALWAYS routes to
        unresolved, even when the event ALSO carries an explicit
        params.parm + params.frames — only frame_eq gets the at_frame-
        keyframe treatment when explicit parm+frames are present."""
        compile_plan = _get_compile_plan()
        events = [{
            "id": "e1", "type": "emit", "target": "/obj/wall",
            "params": {"parm": "activation", "frames": [[4, 0], [5, 1]]},
            "trigger": {"kind": "stress_gt", "value": 1200},
        }]
        result = compile_plan(events)
        assert result["compiled"]["keyframes"] == []
        assert result["unresolved"] == ["e1"]

    def test_frame_eq_trigger_without_explicit_parm_is_unresolved(self):
        """frame_eq is treated as an at_frame keyframe ONLY IF it carries
        explicit parm+frames; here they are absent, so it is unresolved
        too (not a raise)."""
        compile_plan = _get_compile_plan()
        events = [{
            "id": "e1", "type": "fracture", "target": "/obj/wall",
            "trigger": {"kind": "frame_eq"},
        }]
        result = compile_plan(events)
        assert result["unresolved"] == ["e1"]

    def test_frame_eq_trigger_with_explicit_parm_and_frames_compiles(self):
        """frame_eq WITH explicit parm+frames IS treated as an at_frame
        keyframe -- it compiles (distinct from the other three threshold
        kinds, which NEVER compile regardless of explicit parm+frames)."""
        compile_plan = _get_compile_plan()
        events = [{
            "id": "e1", "type": "fracture", "target": "/obj/wall",
            "trigger": {"kind": "frame_eq"},
            "params": {"parm": "activation", "frames": [[10, 1]]},
        }]
        result = compile_plan(events)
        assert result["compiled"]["keyframes"] == [
            {"node": "/obj/wall", "parm": "activation", "frames": [[10, 1]]},
        ]
        assert result["unresolved"] == []


class TestCompilePlanMalformedFramesAndNoTargetAreUnresolved:
    """FIX-PASS (codex-reviewer Major-4): compile_plan must NOT emit a
    malformed/no-target compiled.keyframes entry. `_translate_event` must
    validate BOTH (a) a non-empty `target`, AND (b) that EVERY item in
    `frames` is a genuine [frame, value] pair -- not merely 'frames is a
    non-empty list'. A malformed/no-target event must route to
    unresolved, contributing NOTHING to compiled.keyframes."""

    def test_malformed_frame_shape_and_no_target_is_unresolved(self):
        """The EXACT codex-reviewer repro: no target + a frames list whose
        sole item is not an [f,v] pair at all."""
        compile_plan = _get_compile_plan()
        events = [{
            "id": "bad", "type": "keyframe",
            "params": {"parm": "tx", "frames": ["not-a-pair"]},
        }]
        result = compile_plan(events)
        assert "bad" in result["unresolved"], (
            f"a malformed/no-target keyframe event must be routed to unresolved, "
            f"got unresolved={result['unresolved']!r}"
        )
        assert result["compiled"]["keyframes"] == [], (
            f"compile_plan must NOT emit a malformed compiled.keyframes entry "
            f"(e.g. {{node:'', frames:['not-a-pair']}}), got "
            f"{result['compiled']['keyframes']!r}"
        )

    def test_no_target_with_well_formed_frames_is_unresolved(self):
        """A well-formed frames payload but NO target (target defaults to
        '') must ALSO be unresolved -- an empty node path is never a
        valid keyframe-apply target."""
        compile_plan = _get_compile_plan()
        events = [{
            "id": "e1", "type": "keyframe",
            "params": {"parm": "tx", "frames": [[1, 0.0]]},
        }]  # no 'target' key at all -> EventSpec.target defaults to ""
        result = compile_plan(events)
        assert result["unresolved"] == ["e1"], (
            f"an event with no target must be unresolved even with well-formed "
            f"frames, got unresolved={result['unresolved']!r}"
        )
        assert result["compiled"]["keyframes"] == []

    def test_valid_target_but_malformed_frame_item_is_unresolved(self):
        """A non-empty `target` but a frames list containing a non-pair
        item must also route to unresolved -- frames must be validated
        item-by-item, not merely checked for non-empty-list-ness."""
        compile_plan = _get_compile_plan()
        events = [{
            "id": "e1", "type": "keyframe", "target": "/obj/x",
            "params": {"parm": "tx", "frames": ["not-a-pair"]},
        }]
        result = compile_plan(events)
        assert result["unresolved"] == ["e1"], (
            f"a malformed frame item must route the event to unresolved, got "
            f"unresolved={result['unresolved']!r}"
        )
        assert result["compiled"]["keyframes"] == []


class TestCompilePlanCausalTimeOrder:
    """Major-10: for a `causes` edge between two events that BOTH carry a
    frame (keyframe-frame or at_frame), the caused event's frame must be
    >= the causing event's frame; a violation routes the CAUSED event to
    unresolved (NOT a raise) — a threshold/deferred side skips the check."""

    def test_causally_impossible_edge_routes_caused_event_to_unresolved(self):
        """cause@40 causes effect@20 — backward in time — effect is
        routed to unresolved; the cause itself still compiles."""
        compile_plan = _get_compile_plan()
        events = [
            {"id": "cause", "type": "keyframe", "target": "/obj/a", "at_frame": 40,
             "params": {"parm": "tx", "frames": [[40, 1]]}, "causes": ["effect"]},
            {"id": "effect", "type": "keyframe", "target": "/obj/b", "at_frame": 20,
             "params": {"parm": "tx", "frames": [[20, 1]]}},
        ]
        result = compile_plan(events)
        assert "effect" in result["unresolved"], (
            f"a caused event at an EARLIER frame than its cause must be routed to "
            f"unresolved, got unresolved={result['unresolved']!r}"
        )
        compiled_nodes = {kf["node"] for kf in result["compiled"]["keyframes"]}
        assert "/obj/a" in compiled_nodes, "the CAUSE event must still compile"
        assert "/obj/b" not in compiled_nodes, (
            "the causally-impossible CAUSED event must NOT appear in compiled.keyframes"
        )

    def test_causally_consistent_edge_both_compile(self):
        """cause@20 causes effect@40 — forward in time — BOTH compile."""
        compile_plan = _get_compile_plan()
        events = [
            {"id": "cause", "type": "keyframe", "target": "/obj/a", "at_frame": 20,
             "params": {"parm": "tx", "frames": [[20, 1]]}, "causes": ["effect"]},
            {"id": "effect", "type": "keyframe", "target": "/obj/b", "at_frame": 40,
             "params": {"parm": "tx", "frames": [[40, 1]]}},
        ]
        result = compile_plan(events)
        assert result["unresolved"] == []
        compiled_nodes = {kf["node"] for kf in result["compiled"]["keyframes"]}
        assert compiled_nodes == {"/obj/a", "/obj/b"}

    def test_equal_frames_are_allowed(self):
        """caused-frame >= cause-frame — EQUAL frames must be ALLOWED
        (not a violation)."""
        compile_plan = _get_compile_plan()
        events = [
            {"id": "cause", "type": "keyframe", "target": "/obj/a", "at_frame": 30,
             "params": {"parm": "tx", "frames": [[30, 1]]}, "causes": ["effect"]},
            {"id": "effect", "type": "keyframe", "target": "/obj/b", "at_frame": 30,
             "params": {"parm": "tx", "frames": [[30, 1]]}},
        ]
        result = compile_plan(events)
        assert result["unresolved"] == []

    def test_threshold_trigger_side_skips_time_check(self):
        """A threshold/deferred side skips the causal-time check
        entirely: the caused (keyframe) event must still compile despite
        its causing event's at_frame being LATER than its own — because
        the causing event is itself unresolved on threshold-trigger
        grounds, not on causal-time grounds, and that deferred status must
        not ALSO push the other side into unresolved via the time check."""
        compile_plan = _get_compile_plan()
        events = [
            {"id": "trigger_event", "type": "fracture", "target": "/obj/wall",
             "at_frame": 100, "trigger": {"kind": "stress_gt", "value": 1200},
             "causes": ["effect"]},
            {"id": "effect", "type": "keyframe", "target": "/obj/b", "at_frame": 20,
             "params": {"parm": "tx", "frames": [[20, 1]]}},
        ]
        result = compile_plan(events)
        assert "trigger_event" in result["unresolved"], (
            "the threshold-trigger event itself is unresolved on its own deferred-"
            "trigger grounds"
        )
        compiled_nodes = {kf["node"] for kf in result["compiled"]["keyframes"]}
        assert "/obj/b" in compiled_nodes, (
            "the causal-time check must be SKIPPED on a threshold/deferred side -- "
            "'effect' must still compile despite its causing event's at_frame (100) "
            "being later than its own (20)"
        )


class TestCompilePlanCausalTimeOrderUsesExplicitFramesWhenAtFrameAbsent:
    """FIX-PASS (codex-reviewer Major-5): the causal time-order check must
    derive each side's causal timestamp from its EXPLICIT keyframe frames
    when `at_frame` is ABSENT (EventSpec.at_frame defaults to 0) -- it
    must NOT silently compare two defaulted-to-0 at_frame values, which
    would miss a real causally-impossible edge whenever neither event
    sets at_frame explicitly."""

    def test_causally_impossible_edge_via_explicit_frames_no_at_frame(self):
        """cause's OWN explicit frame is 40; effect's OWN explicit frame is
        20 -- backward in time -- despite NEITHER event setting at_frame
        (both default to 0, so a naive at_frame-only comparison sees
        0 < 0, a false negative). effect must be routed to unresolved."""
        compile_plan = _get_compile_plan()
        events = [
            {"id": "cause", "type": "keyframe", "target": "/obj/a",
             "params": {"parm": "tx", "frames": [[40, 1]]}, "causes": ["effect"]},
            {"id": "effect", "type": "keyframe", "target": "/obj/b",
             "params": {"parm": "tx", "frames": [[20, 1]]}},
        ]
        result = compile_plan(events)
        assert "effect" in result["unresolved"], (
            f"a caused event whose EXPLICIT frames (20) are earlier than its "
            f"cause's EXPLICIT frames (40) must be routed to unresolved, even "
            f"when neither event sets at_frame explicitly -- got "
            f"unresolved={result['unresolved']!r}"
        )
        compiled_nodes = {kf["node"] for kf in result["compiled"]["keyframes"]}
        assert "/obj/a" in compiled_nodes, "the CAUSE event must still compile"
        assert "/obj/b" not in compiled_nodes, (
            "the causally-impossible CAUSED event must NOT appear in "
            "compiled.keyframes"
        )


class TestCompilePlanTopLevelFieldNormalization:
    """Major-9: a raw event with parm/frames given at TOP LEVEL (sibling
    to id/type/target), not nested under params, is NORMALIZED into
    params BEFORE EventSpec construction — it compiles exactly as if the
    caller had nested them under params (mirrors PR-2's
    expect-normalization pattern)."""

    def test_top_level_parm_and_frames_normalized_into_params(self):
        compile_plan = _get_compile_plan()
        events = [{
            "id": "e1", "type": "keyframe", "target": "/obj/x",
            "parm": "tx", "frames": [[1, 0], [10, 5]],  # TOP LEVEL, not under params
        }]
        result = compile_plan(events)
        assert result["compiled"]["keyframes"] == [
            {"node": "/obj/x", "parm": "tx", "frames": [[1, 0], [10, 5]]},
        ], (
            f"a top-level parm/frames pair must be normalized into params and compile "
            f"exactly as the nested-params form, got {result['compiled']!r}"
        )
        assert result["unresolved"] == []

    def test_top_level_activation_fields_normalized_into_params(self):
        compile_plan = _get_compile_plan()
        events = [{
            "id": "e1", "type": "emit", "target": "/obj/wall",
            "parm": "activation", "frames": [[4, 0], [5, 1]],
        }]
        result = compile_plan(events)
        assert result["compiled"]["keyframes"] == [
            {"node": "/obj/wall", "parm": "activation", "frames": [[4, 0], [5, 1]]},
        ]
        assert result["unresolved"] == []


class TestCompilePlanGraphValidationErrors:
    """A cyclic or dangling causes graph, a duplicate event id, or an
    unknown event type RAISES ValueError (reusing PR-1's EventGraph/
    EventSpec construction-time validation unchanged) — a construction/
    validation error, NEVER an unresolved-routing case."""

    def test_cyclic_causes_raises_value_error(self):
        compile_plan = _get_compile_plan()
        events = [
            {"id": "e1", "type": "fracture", "causes": ["e2"]},
            {"id": "e2", "type": "emit", "causes": ["e1"]},
        ]
        with pytest.raises(ValueError):
            compile_plan(events)

    def test_dangling_causes_edge_raises_value_error(self):
        compile_plan = _get_compile_plan()
        events = [{"id": "e1", "type": "fracture", "causes": ["ghost"]}]
        with pytest.raises(ValueError):
            compile_plan(events)

    def test_duplicate_event_id_raises_value_error(self):
        compile_plan = _get_compile_plan()
        events = [
            {"id": "dup", "type": "fracture"},
            {"id": "dup", "type": "emit"},
        ]
        with pytest.raises(ValueError):
            compile_plan(events)

    def test_unknown_event_type_raises_value_error(self):
        compile_plan = _get_compile_plan()
        events = [{"id": "e1", "type": "not_a_real_event"}]
        with pytest.raises(ValueError):
            compile_plan(events)


class TestCompilePlanDoesNotRegressPriorSurface:
    """PR-3 is ADDITIVE — the PR-1/PR-2 public surface (EventSpec,
    EventGraph, describe_sim_events, aggregate_assertion,
    evaluate_assertions) must remain importable and behave unchanged."""

    def test_prior_public_surface_still_importable_and_callable(self):
        from fxhoudinimcp.temporal_reasoning_model import (
            EventSpec, EventGraph, describe_sim_events,
            aggregate_assertion, evaluate_assertions,
        )
        assert describe_sim_events()["assertions"] == [
            "piece_count", "constraint_count", "point_count", "velocity_bounds",
            "bbox_over_time", "field_stats", "mass_conservation",
        ]
        e = EventSpec(id="e1", type="fracture")
        assert e.type == "fracture"
        g = EventGraph(events=[e])
        assert g.topo_order() == ["e1"]
        assert evaluate_assertions([]) == {"results": [], "pass": True}
        assert aggregate_assertion([[1, 1.0], [2, 2.0]], {"max": 100.0})["pass"] is True
