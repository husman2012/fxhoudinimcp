"""
temporal_reasoning_model.py — pure-logic core for the Temporal / Sim-Reasoning
MCP surface (PP12-117 PR-1, VERIFICATION HALF ONLY).

Contract: imports NO hou, NO Qt/PySide6, NO pxr, NO MCP server framework.
Makes NO MCP call. NEVER cooks or reads a simulation itself -- per-frame
physics series arrive as plain [[frame, value], ...] lists; reading them off
a live cook is the (deferred) PR-2 handler's job (CL-015 extended). numpy is
PERMITTED but NOT mandated -- this module deliberately uses plain Python
stdlib (dataclasses, heapq) only, since numpy is not installed in the
fxhoudinimcp fork .venv (verified 2026-07-09 for the sibling 116 PR-1) and a
correct stdlib implementation satisfies the locked behavioral contract
equally well. Pytest-able off-DCC (CL-015) -- this module never touches
Houdini, Qt, USD, an MCP server framework, or numpy, so it runs on bare CI
with plain pytest.

SCOPE SPLIT (BINDING, plan pp12-117a lockedFieldContract revision 2): this PR
builds the VERIFICATION half only:
  (1) the event-timeline REPRESENTATION -- EventSpec (with a lossless
      `params` payload) + EventGraph (a causal DAG with topo_order() that
      rejects a cycle/dangling-edge/duplicate-id).
  (2) the Gate-1 assertion-aggregation ORACLE MATH -- aggregate_assertion
      per metric + evaluate_assertions ANDing them into the exact SPEC 4.1
      assert_simulation return.
  (3) describe_sim_events() -- the exact SPEC 4.1 vocabulary wire shape.
DEFERRED to a later PR (the authoring half): the timeline->setup
TRANSLATION (events -> concrete keyframes/CHOP/DOP-parm plan),
compile_timeline, the bbox/field_stats scalar-reduction (handler-side), any
tool wrapper, handlers/temporal_reasoning_handlers.py, the registry edit,
the 109 gate, hdefereval marshaling, and any cook/DOP read.

Classes
-------
EventSpec   — one event's id/type/target/at_frame/params/trigger/causes
EventGraph  — a causal DAG of EventSpec (topo_order/to_dict/from_dict)

Functions
---------
describe_sim_events() -> dict
    The exact SPEC 4.1 {events, triggers, assertions} vocabulary shape.
aggregate_assertion(series, expect) -> dict
    Turns ONE metric's per-frame scalar series + its expect predicate into
    a deterministic pass/fail + diagnostics.
evaluate_assertions(assertions) -> dict
    ANDs a list of {metric, series, expect} assertions into the exact
    SPEC 4.1 assert_simulation return {results, pass}.

Error taxonomy (mirrors 116; REVISION 2 expanded)
--------------------------------------------------
CONSTRUCTION/VALIDATION errors RAISE ValueError: an unknown EventSpec
`type`, an unknown trigger `kind`, a malformed EventSpec field, a duplicate
event id, a dangling or cyclic `causes` edge, an unknown assertion
`metric`, an assertion missing metric/series/expect, a malformed series, or
an invalid expect (unknown key / zero predicate keys / jump_gt-eq missing
at_frame). A metric whose WELL-FORMED series simply FAILS its (well-formed)
expectation is NOT an error -- it returns pass:false. An eq/jump_gt whose
requested at_frame has no qualifying sample also returns pass:false (with a
note on eq), NOT an error.

Cross-references
-----------------
Plan pp12-117a lockedFieldContract (BINDING, revision 2 -- folds all 11
  codex-adversarial-reviewer findings)
docs/portfolio_projects/PP12_houdini_mcp_agentic_bridge/
  117_mcp_temporal_sim_reasoning_surface/spec.md Sections 4.1, 6, 9
CL-015 (extended): pure-logic module, no hou/Qt/pxr/MCP-server-framework
  import, no MCP call, no cook/DOP read, no timeline->setup translation
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TOL = 1e-4

_EVENT_ORDER = ("fracture", "emit", "tear", "ignite", "keyframe")
_EVENT_TYPES = frozenset(_EVENT_ORDER)

_TRIGGER_ORDER = ("stress_gt", "collision_with", "frame_eq", "field_gt")
_TRIGGER_KINDS = frozenset(_TRIGGER_ORDER)

_ASSERT_ORDER = (
    "piece_count", "constraint_count", "point_count", "velocity_bounds",
    "bbox_over_time", "field_stats", "mass_conservation",
)
_ASSERT_METRICS = frozenset(_ASSERT_ORDER)

_EVENT_CONTEXT = {
    "fracture": "rbd",
    "emit": "pop|pyro",
    "tear": "vellum",
    "ignite": "pyro",
    "keyframe": "any",
}

_EVENT_PARAMS_SCHEMA = {
    # B1 (REVISION 3 FOLD): these are the SPEC 4.1 LITERAL per-event params
    # dicts, verbatim -- the values are the spec's own type-hint strings,
    # not a descriptive prose schema of this module's own invention.
    "fracture": {"at_frame": "int", "trigger": "str?"},
    "emit": {"at_frame": "int", "rate": "float"},
    "tear": {"threshold": "float"},
    "ignite": {"at_frame": "int", "region": "str?"},
    "keyframe": {"node": "str", "parm": "str", "frames": "[[f,v]]"},
}

_PREDICATE_KEYS = frozenset({"max", "min", "jump_gt", "tolerance", "max_gt", "eq"})
_SUPPORT_KEYS = frozenset({"at_frame"})
_ALLOWED_EXPECT_KEYS = _PREDICATE_KEYS | _SUPPORT_KEYS

# M4 (REVISION 3 FOLD): the frame-bearing predicates -- counted to decide
# whether the spec-compatible BARE `at_frame` diagnostic is emitted. Only
# max/min/eq ever assign the bare at_frame (they carry a single scalar
# frame); jump_gt is counted here too (it is frame-bearing per the locked
# contract) so a jump_gt+max/min/eq combination also suppresses the bare
# key, even though jump_gt itself never sets one (its own diagnostic is a
# `note` naming before_frame/after_frame, not a single at_frame).
_FRAME_BEARING_KEYS = ("max", "min", "eq", "jump_gt")


# ---------------------------------------------------------------------------
# EventSpec
# ---------------------------------------------------------------------------

@dataclass
class EventSpec:
    """One event's id/type/target/at_frame/params/trigger/causes.

    `params` (Blocker-2) carries the event-specific payload (emit.rate,
    tear.threshold, ignite.region, keyframe.node/parm/frames, etc.) so the
    representation is LOSSLESS -- the deferred timeline->setup translation
    depends on this payload surviving to_dict()/from_dict() unchanged.

    `causes` = the ids of downstream EventSpecs this event causes (the
    causal edges consumed by EventGraph.topo_order()).
    """

    id: str
    type: str
    target: str = ""
    at_frame: int = 0
    params: dict = field(default_factory=dict)
    trigger: dict = field(default_factory=dict)
    causes: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError(f"EventSpec: id must be a non-empty str, got {self.id!r}")
        if self.type not in _EVENT_TYPES:
            raise ValueError(
                f"EventSpec {self.id!r}: unknown event type {self.type!r}; "
                f"must be one of {sorted(_EVENT_TYPES)!r}"
            )
        if not isinstance(self.target, str):
            raise ValueError(f"EventSpec {self.id!r}: target must be a str, got {self.target!r}")
        if isinstance(self.at_frame, bool) or not isinstance(self.at_frame, int):
            raise ValueError(
                f"EventSpec {self.id!r}: at_frame must be an int (not bool), got {self.at_frame!r}"
            )
        if not isinstance(self.params, dict):
            raise ValueError(f"EventSpec {self.id!r}: params must be a dict, got {self.params!r}")
        if not isinstance(self.trigger, dict):
            raise ValueError(f"EventSpec {self.id!r}: trigger must be a dict, got {self.trigger!r}")
        if self.trigger:
            kind = self.trigger.get("kind")
            if kind not in _TRIGGER_KINDS:
                raise ValueError(
                    f"EventSpec {self.id!r}: trigger kind {kind!r} must be one of "
                    f"{sorted(_TRIGGER_KINDS)!r}"
                )
        if not isinstance(self.causes, list):
            raise ValueError(f"EventSpec {self.id!r}: causes must be a list, got {self.causes!r}")
        for c in self.causes:
            if not isinstance(c, str) or not c:
                raise ValueError(
                    f"EventSpec {self.id!r}: causes must be a list of non-empty str "
                    f"ids, got {self.causes!r}"
                )

    def to_dict(self) -> dict:
        """Serialize to a plain dict with exactly seven keys, LOSSLESS params."""
        return {
            "id": self.id,
            "type": self.type,
            "target": self.target,
            "at_frame": self.at_frame,
            "params": dict(self.params),
            "trigger": dict(self.trigger),
            "causes": list(self.causes),
        }


def _event_from_dict(d: dict) -> EventSpec:
    """Rebuild one EventSpec from its to_dict() shape (or a compatible raw dict).

    M2 (REVISION 3 FOLD, load-bearing): params/trigger/causes are passed
    RAW straight to EventSpec(...) -- NO list(...)/dict(...) pre-coercion.
    A raw `causes` of the string "bc" or a raw `params`/`trigger` of []
    must reach EventSpec.__post_init__ unchanged so its field-type
    validation actually fires (raises ValueError) instead of a coercing
    implementation silently "fixing" bad input (e.g. list("bc") ->
    ['b','c'], dict([]) -> {}).
    """
    return EventSpec(
        id=d["id"],
        type=d["type"],
        target=d.get("target", ""),
        at_frame=d.get("at_frame", 0),
        params=d.get("params", {}),
        trigger=d.get("trigger", {}),
        causes=d.get("causes", []),
    )


# ---------------------------------------------------------------------------
# EventGraph
# ---------------------------------------------------------------------------

@dataclass
class EventGraph:
    """A causal DAG of EventSpec.

    DUPLICATE event ids raise ValueError in topo_order(), to_dict(), AND
    from_dict() (Major-9). topo_order() orders events so every cause
    precedes the events it causes, with a STABLE tie-break (the original
    `events` list order) whenever multiple events are simultaneously
    available; a dangling `causes` edge or a CYCLE both raise ValueError
    (the cycle message names the word "cycle" plus a participating id).
    """

    events: list = field(default_factory=list)

    def _check_duplicate_ids(self) -> None:
        seen = set()
        for e in self.events:
            if e.id in seen:
                raise ValueError(f"EventGraph: duplicate event id {e.id!r}")
            seen.add(e.id)

    def _compute_topo_order(self) -> list:
        """The single shared duplicate-id + dangling-edge + cycle check,
        returning the causal-DAG order.

        M3 (REVISION 3 FOLD): this is the ONE implementation of the
        validation; topo_order(), to_dict(), and from_dict() all invoke it
        (via _validate_graph()) so a dangling or cyclic causes graph
        raises at all three entry points, not only topo_order().
        """
        self._check_duplicate_ids()
        by_id = {e.id: e for e in self.events}
        for e in self.events:
            for c in e.causes:
                if c not in by_id:
                    raise ValueError(
                        f"EventGraph: causes edge from {e.id!r} to "
                        f"{c!r} does not reference a known event id (dangling edge)"
                    )

        order_index = {e.id: i for i, e in enumerate(self.events)}
        in_degree = {e.id: 0 for e in self.events}
        for e in self.events:
            for c in e.causes:
                in_degree[c] += 1

        heap = [(order_index[eid], eid) for eid, deg in in_degree.items() if deg == 0]
        heapq.heapify(heap)
        order: list = []
        while heap:
            _, eid = heapq.heappop(heap)
            order.append(eid)
            for c in by_id[eid].causes:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    heapq.heappush(heap, (order_index[c], c))

        if len(order) != len(self.events):
            remaining = [eid for eid, deg in in_degree.items() if deg > 0]
            raise ValueError(
                f"EventGraph: the causes graph contains a cycle; "
                f"participating event ids include {remaining!r}"
            )
        return order

    def _validate_graph(self) -> None:
        """Validate duplicate ids + dangling causes edges + cycles.

        M3 (REVISION 3 FOLD): invoked from topo_order(), to_dict(), AND
        from_dict() -- a dangling or cyclic causes graph raises at all
        three entry points, matching the error-taxonomy
        '[at topo_order/to_dict/from_dict]'.
        """
        self._compute_topo_order()

    def topo_order(self) -> list:
        """Causal-DAG order: every cause precedes the events it causes.

        Stable tie-break = the original `events` list order among all
        events simultaneously available (zero remaining in-degree); each
        event's `causes` are traversed in their own list order. Raises
        ValueError on a duplicate id, a dangling causes edge (an id not
        present in `events`), or a cycle (message contains "cycle" plus a
        participating id).
        """
        return self._compute_topo_order()

    def to_dict(self) -> dict:
        """Serialize to a plain dict with exactly three keys.

        Raises ValueError on a duplicate event id, a dangling causes
        edge, or a cycle (M3 -- not only a duplicate id, Major-9).
        """
        self._validate_graph()
        return {
            "events": [e.to_dict() for e in self.events],
            "nodes": len(self.events),
            "edges": sum(len(e.causes) for e in self.events),
        }

    @staticmethod
    def from_dict(d: dict) -> "EventGraph":
        """Rebuild an EventGraph from its to_dict() shape.

        Rebuilds EventSpec instances (propagating any ValueError a bad
        event raises); raises ValueError on a duplicate event id, a
        dangling causes edge, or a cycle (M3 -- not only a duplicate id,
        Major-9); ignores unknown top-level keys.
        """
        events = [_event_from_dict(ed) for ed in d.get("events", [])]
        graph = EventGraph(events=events)
        graph._validate_graph()
        return graph


# ---------------------------------------------------------------------------
# describe_sim_events (FR-C)
# ---------------------------------------------------------------------------

def describe_sim_events() -> dict:
    """Return the exact SPEC 4.1 {events, triggers, assertions} vocabulary
    shape, in a STABLE order across calls."""
    return {
        "events": [
            {
                "type": t,
                "context": _EVENT_CONTEXT[t],
                "params": dict(_EVENT_PARAMS_SCHEMA[t]),
            }
            for t in _EVENT_ORDER
        ],
        "triggers": list(_TRIGGER_ORDER),
        "assertions": list(_ASSERT_ORDER),
    }


# ---------------------------------------------------------------------------
# Series + expect validation (aggregate_assertion's error taxonomy)
# ---------------------------------------------------------------------------

def _validate_series(series) -> list:
    """Validate + normalize a `series` into [[int frame, float value], ...].

    Raises ValueError on: not a non-empty list; a non-[frame,value] item; a
    bool frame/value; a non-scalar (list/tuple/dict) value; a duplicate
    frame; frames not strictly ascending.
    """
    if not isinstance(series, list) or len(series) == 0:
        raise ValueError(
            f"series must be a non-empty list of [frame, value] pairs, got {series!r}"
        )

    normalized = []
    seen_frames = set()
    prev_frame = None
    for item in series:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"series item must be a [frame, value] pair, got {item!r}")
        frame, value = item
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise ValueError(f"series frame must be an int (not bool), got {frame!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"series value must be a scalar int/float (not bool/list/tuple/dict), "
                f"got {value!r}"
            )
        if frame in seen_frames:
            raise ValueError(f"series contains a duplicate frame {frame!r}")
        if prev_frame is not None and frame <= prev_frame:
            raise ValueError(
                f"series frames must be strictly ascending, got {frame!r} after {prev_frame!r}"
            )
        seen_frames.add(frame)
        prev_frame = frame
        normalized.append([int(frame), float(value)])
    return normalized


def _validate_expect(expect) -> None:
    """Validate an `expect` predicate dict.

    Raises ValueError on: not a dict; an unknown key (predicate keys are
    exactly {max,min,jump_gt,tolerance,max_gt,eq}, the only support key is
    {at_frame}); zero predicate keys present (at_frame alone is not a
    predicate); jump_gt or eq present without the required at_frame.
    """
    if not isinstance(expect, dict):
        raise ValueError(f"expect must be a dict, got {expect!r}")
    unknown = set(expect.keys()) - _ALLOWED_EXPECT_KEYS
    if unknown:
        raise ValueError(
            f"expect contains unknown key(s) {sorted(unknown)!r}; allowed keys "
            f"are {sorted(_ALLOWED_EXPECT_KEYS)!r}"
        )
    predicate_keys_present = set(expect.keys()) & _PREDICATE_KEYS
    if not predicate_keys_present:
        raise ValueError(
            "expect must contain at least one predicate key (max/min/jump_gt/"
            "tolerance/max_gt/eq); at_frame alone is not a predicate"
        )
    if "jump_gt" in expect and "at_frame" not in expect:
        raise ValueError("expect['jump_gt'] requires 'at_frame'")
    if "eq" in expect and "at_frame" not in expect:
        raise ValueError("expect['eq'] requires 'at_frame'")


# ---------------------------------------------------------------------------
# aggregate_assertion (FR-B core)
# ---------------------------------------------------------------------------

def aggregate_assertion(series, expect) -> dict:
    """Aggregate ONE metric's SCALAR series against its expect predicate.

    Returns EXACTLY {pass: bool, series: [[frame,value]...]} plus the
    applicable NAMESPACED diagnostic keys (max_seen+max_at_frame on a
    `max` breach; min_seen+min_at_frame on a `min` breach; eq_observed+
    eq_at_frame on an `eq` breach; a descriptive `note` when a `jump_gt`
    scan finds a qualifying jump, or when an eq/jump_gt at_frame has no
    qualifying sample; `drift` for `tolerance`). M4 (REVISION 3 FOLD):
    additionally, ONLY when EXACTLY ONE frame-bearing predicate
    (max/min/eq/jump_gt) is present in `expect`, the spec-compatible bare
    `at_frame` (= that predicate's own frame) is ALSO emitted -- a
    COMPOSITE expect (2+ frame-bearing predicates) emits ONLY the
    namespaced keys so two predicates never collide on a shared bare key.
    All outputs are JSON-serializable plain float/int/list/str/bool.
    Raises ValueError on a malformed series or an invalid expect (see
    _validate_series/_validate_expect); does NOT raise on a metric that
    simply fails its well-formed expectation.
    """
    normalized = _validate_series(series)
    _validate_expect(expect)

    values = [v for _, v in normalized]
    result: dict = {}
    passes: list = []

    # M4 (REVISION 3 FOLD): count the frame-bearing predicates PRESENT in
    # `expect` (not merely the ones that end up breaching) -- a COMPOSITE
    # expect (2+ present) emits ONLY namespaced diagnostic keys; a SINGLE
    # frame-bearing predicate ALSO gets the spec-compatible bare
    # `at_frame` so the SPEC 4.1 single-predicate examples keep working.
    _frame_bearing_present = sum(1 for k in _FRAME_BEARING_KEYS if k in expect)
    _single_frame_bearing = _frame_bearing_present == 1
    _bare_at_frame = None

    if "max" in expect:
        max_seen = max(values)
        at_frame = next(f for f, v in normalized if v == max_seen)
        ok = max_seen <= float(expect["max"]) + _TOL
        passes.append(ok)
        if not ok:
            result["max_seen"] = float(max_seen)
            result["max_at_frame"] = int(at_frame)
            if _single_frame_bearing:
                _bare_at_frame = int(at_frame)

    if "min" in expect:
        min_seen = min(values)
        at_frame = next(f for f, v in normalized if v == min_seen)
        ok = min_seen >= float(expect["min"]) - _TOL
        passes.append(ok)
        if not ok:
            result["min_seen"] = float(min_seen)
            result["min_at_frame"] = int(at_frame)
            if _single_frame_bearing:
                _bare_at_frame = int(at_frame)

    if "jump_gt" in expect:
        at_f = expect["at_frame"]
        n = float(expect["jump_gt"])
        before = None
        before_frame = None
        for f, v in normalized:
            if f < at_f:
                before, before_frame = v, f
            else:
                break
        if before is None:
            before, before_frame = normalized[0][1], normalized[0][0]

        ok = False
        after_frame = after_val = None
        for f, v in normalized:
            if f >= at_f and (v - before) > n + _TOL:
                ok, after_frame, after_val = True, f, v
                break
        passes.append(ok)
        if ok:
            result["note"] = (
                f"jump from {before} at frame {before_frame} to {after_val} at "
                f"frame {after_frame} exceeds jump_gt={n}"
            )

    if "tolerance" in expect:
        t = float(expect["tolerance"])
        value0 = values[0]
        if value0 == 0:
            drift = max(abs(v) for v in values)
        else:
            drift = max(abs(v - value0) / abs(value0) for v in values)
        ok = drift <= t + _TOL
        passes.append(ok)
        result["drift"] = float(drift)

    if "max_gt" in expect:
        x = float(expect["max_gt"])
        ok = max(values) > x + _TOL
        passes.append(ok)

    if "eq" in expect:
        at_f = expect["at_frame"]
        v_expected = float(expect["eq"])
        sample = None
        for f, v in normalized:
            if f >= at_f:
                sample = (f, v)
                break
        if sample is None:
            passes.append(False)
            result["note"] = (
                f"eq predicate: no sample at or after the requested at_frame={at_f}"
            )
        else:
            f, v = sample
            ok = abs(v - v_expected) <= _TOL
            passes.append(ok)
            if not ok:
                result["eq_at_frame"] = int(f)
                result["eq_observed"] = float(v)
                if _single_frame_bearing:
                    _bare_at_frame = int(f)

    if _bare_at_frame is not None:
        result["at_frame"] = _bare_at_frame

    result["pass"] = bool(all(passes))
    result["series"] = normalized
    return result


# ---------------------------------------------------------------------------
# evaluate_assertions (FR-B, the assert_simulation model core)
# ---------------------------------------------------------------------------

def evaluate_assertions(assertions) -> dict:
    """AND a list of {metric, series, expect} assertions into the exact
    SPEC 4.1 assert_simulation return {results, pass}.

    Processes assertions in INPUT order and returns `results` in that same
    order (Major-11). Each result entry carries ONLY {metric, pass, series}
    plus the applicable predicate diagnostics -- handler-only extra
    top-level keys on the input (e.g. node, field) are IGNORED and never
    copied into a result entry. Empty assertions -> pass:true, results:[].
    Raises ValueError on an unknown metric or an assertion missing
    metric/series/expect.
    """
    if not isinstance(assertions, list):
        raise ValueError(f"assertions must be a list, got {assertions!r}")

    results = []
    for a in assertions:
        if not isinstance(a, dict):
            raise ValueError(f"each assertion must be a dict, got {a!r}")
        if "metric" not in a:
            raise ValueError("assertion is missing required key 'metric'")
        metric = a["metric"]
        if metric not in _ASSERT_METRICS:
            raise ValueError(
                f"unknown assertion metric {metric!r}; must be one of "
                f"{sorted(_ASSERT_METRICS)!r}"
            )
        if "series" not in a:
            raise ValueError(f"assertion for metric {metric!r} is missing required key 'series'")
        if "expect" not in a:
            raise ValueError(f"assertion for metric {metric!r} is missing required key 'expect'")

        agg = aggregate_assertion(a["series"], a["expect"])
        entry = {"metric": metric}
        entry.update(agg)
        results.append(entry)

    overall_pass = all(r["pass"] for r in results) if results else True
    return {"results": results, "pass": bool(overall_pass)}
