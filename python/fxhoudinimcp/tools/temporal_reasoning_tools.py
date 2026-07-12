"""MCP wrappers: houdini_describe_sim_events, houdini_assert_simulation.

Both are READ-ONLY, UNGATED (require_approval=False, Capability.READONLY
handler-side) -- the Gate-1 read surface of the Temporal/Sim-Reasoning MCP
member (PP12-117 PR-2). compile_timeline (the MUTATING/gated authoring
tool) and the hypothesis eval are DEFERRED, out of scope here.

houdini_describe_sim_events -- the anti-hallucination vocabulary tool: a
                                pure delegate to
                                temporal_reasoning_model.describe_sim_events()
                                over the bridge (no params).
houdini_assert_simulation   -- the Gate-1 temporal oracle: steps an
                                inclusive frame range, reads each requested
                                assertion's per-frame scalar off the live
                                sim, and returns the model's exact SPEC 4.1
                                assert_simulation dict.

Each wrapper delegates to the correspondingly named handler registered on
the Houdini side via bridge.execute. No domain logic lives here.

Contract: imports NO hou, NO pxr -- this module must be importable
off-DCC for the wrapper pytest suite (CL-015).

PP12-117 / pp12-117b (houdini_describe_sim_events, houdini_assert_simulation
                       -- PR-2 of member 117)
"""
from __future__ import annotations

from mcp.server.fastmcp import Context

import fxhoudinimcp.server as _fxserver

# mcp is used by the @mcp.tool() decorator at module import time.
mcp = _fxserver.mcp


@mcp.tool(meta={"require_approval": False})
async def houdini_describe_sim_events(ctx: Context) -> dict:
    """Return the sim-event vocabulary -- the anti-hallucination reference
    for every event type, trigger kind, and assertion metric
    assert_simulation/compile_timeline accept.

    READ-ONLY / UNGATED -- a pure vocabulary read that cannot fail on scene
    state.

    Returns::

        {
            "events": [{"type": str, "context": str, "params": {...}}, ...],
            "triggers": [str, ...],
            "assertions": [str, ...],
        }

    Args:
        ctx: MCP lifespan context -- injected by FastMCP; hidden from client schema.
    """
    # Access _get_bridge through the module reference so that
    # `patch("fxhoudinimcp.server._get_bridge", ...)` intercepts it correctly
    # in tests (a local import would cache the original function object).
    bridge = _fxserver._get_bridge(ctx)
    return await bridge.execute("describe_sim_events", {})


@mcp.tool(meta={"require_approval": False})
async def houdini_assert_simulation(
    ctx: Context,
    network: str,
    frame_range: list,
    assertions: list,
    cook_job: "str | None" = None,
) -> dict:
    """The Gate-1 temporal oracle: assert per-frame simulation behavior
    (piece counts, point counts, velocity bounds, world-space bbox extent,
    mass conservation) across an inclusive frame range, resolving each
    assertion's read source from its own `node` (or `network` when absent)
    from the live scene. READ-ONLY / UNGATED -- frames are stepped to read
    per-frame scalars but the pre-call frame is ALWAYS restored, and no
    node/parm/userData state is ever written (the explicit
    reversible-frame-evaluation exception).

    Returns the pure model's exact SPEC 4.1 {results, pass} dict VERBATIM
    on success, or {"ok": False, "error": "<reason>"} on a scene-resolution
    failure (an unresolvable node) or a metric the live scene cannot supply
    (a missing required attribute, or a deferred metric -- field_stats /
    constraint_count -- unsupported in PR-2). A caller-contract error (a
    malformed frame_range, an unknown assertion metric, a malformed
    expect, or the expect/top-level-predicate conflict) propagates as the
    dispatcher's standard error envelope, not a normal return value.

    A SINGLE bridge.execute call -- the wrapper performs no result
    interpretation and returns bridge.execute's result VERBATIM.

    Args:
        ctx: MCP lifespan context -- injected by FastMCP; hidden from client schema.
        network: The default node path assertions read from when an
            assertion supplies no `node` of its own.
        frame_range: [start, end] inclusive frame range (ints, start <=
            end; a single-frame [f, f] is valid).
        assertions: List of assertion wire dicts. Each carries `metric`
            (one of piece_count/point_count/velocity_bounds/
            bbox_over_time/mass_conservation/field_stats/constraint_count),
            an optional `node` overriding `network` as the read source, and
            either a nested `expect` predicate dict OR the predicate
            key(s) supplied directly at the top level (e.g.
            {"metric": "velocity_bounds", "max": 250}) -- exactly one of
            the two forms, never both.
        cook_job: Reserved for a future cook-registry integration (the 115
            surface); currently always returns the documented
            unavailable-message when non-null. Omit to read the current
            synchronous sim state.
    """
    bridge = _fxserver._get_bridge(ctx)
    return await bridge.execute(
        "assert_simulation",
        {
            "network": network,
            "frame_range": frame_range,
            "assertions": assertions,
            "cook_job": cook_job,
        },
    )


@mcp.tool(meta={"require_approval": True})
async def houdini_compile_timeline(
    ctx: Context,
    network: str,
    events: list,
    frame_range: list,
    apply: bool = True,
) -> dict:
    """Compile an agent-authored event-timeline into concrete Houdini
    KEYFRAMES on the EXISTING sim network (PP12-117 PR-3, the AUTHORING
    half). Compiles ONLY two grounded event shapes -- a bare `keyframe`
    event, and an activation event (emit/fracture/ignite/tear) that
    carries an EXPLICIT params.parm + EXPLICIT params.frames -- into a
    compiled.keyframes entry. Everything this narrowed, honest translation
    cannot map (type-inferred activation without an explicit parm, a
    threshold-triggered event, a causally-impossible caused event, an
    out-of-network or missing target) is routed to `unresolved` instead of
    being invented or silently dropped.

    GATED (require_approval=True -- Capability.MUTATING handler-side; the
    first MUTATING tool of the Temporal/Sim-Reasoning MCP member).
    apply=False STILL goes through the gate (fail-safe -- gate capability
    is per-COMMAND, not per-argument, mirroring houdini_solve_layout).

    A SINGLE bridge.execute call -- the wrapper performs no result
    interpretation and returns bridge.execute's result VERBATIM, including
    the 109-gate pending-approval/preview shape (a normal, valid return --
    never reinterpreted, never raised).

    Args:
        ctx: MCP lifespan context -- injected by FastMCP; hidden from client schema.
        network: The sim network node every compiled keyframe target must
            be `network` itself or a descendant under `network + '/'` --
            an out-of-scope target is routed to `unresolved`, never
            written.
        events: List of event-timeline wire dicts (see
            houdini_describe_sim_events for the vocabulary). Only a
            keyframe event, or an activation event carrying an explicit
            `parm` + `frames`, compiles; everything else is unresolved.
        frame_range: [start, end] inclusive frame range -- reserved for
            parity with the read-side tools; compile_timeline only WRITES
            keyframes, it does not cook or step frames.
        apply: When True (default), atomically set the preflight-validated
            keyframes on the resolved target parms. When False, only the
            compiled plan is returned -- no scene mutation.
    """
    bridge = _fxserver._get_bridge(ctx)
    return await bridge.execute(
        "compile_timeline",
        {
            "network": network,
            "events": events,
            "frame_range": frame_range,
            "apply": apply,
        },
    )
