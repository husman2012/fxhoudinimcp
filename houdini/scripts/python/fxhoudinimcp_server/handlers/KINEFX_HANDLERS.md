# KineFX / APEX Handlers — Reference

Source of record for the 8 MCP KineFX/APEX tools added in the `pp12-111-engine-export` branch (pp12-110 PR-8, code committed at fork `d00c74d`).

---

## §1 — Tool List

| MCP Tool | Handler | Capability | Gated (`preview_required`) | Purpose |
|---|---|---|---|---|
| `kinefx_probe` | `kinefx_probe` | READONLY | No | Confirm Houdini build + verify every KineFX/APEX node type resolves (`kinefx::fbxcharacterimport`, `kinefx::fbxanimimport`, `bonedeform`, `rigmatchpose`, `motiontransform`, `kinefx::secondarymotion`, `apex::autorigcomponent`) |
| `query_skeleton` | `query_skeleton` | READONLY | No | Serialize a skeleton's joints (name, parent, rest + optional animated transform at `frame`) from its KineFX point graph to JSON — reads only, writes nothing |
| `inspect_apex` | `inspect_apex` | READONLY | No | Summarize an APEX `.rig` / APEX node graph (node list, types, ports, wires, parms) to JSON — reads only, writes nothing |
| `import_fbx_character` | `import_fbx_character` | MUTATING | Yes | Drive `kinefx::fbxcharacterimport`: import a skeleton (+ optional skin geo) from an FBX file |
| `import_fbx_animation` | `import_fbx_animation` | MUTATING | Yes | Drive `kinefx::fbxanimimport`: import an animated skeleton / MotionClip from an FBX; Cascadeur FBX is first-class (`cascadeur=True` flag) |
| `setup_bonedeform` | `setup_bonedeform` | MUTATING | Yes | Wire a `bonedeform` SOP with rest-skeleton, anim-skeleton, and skinned-geo inputs |
| `setup_retarget` | `setup_retarget` | MUTATING | Yes | Build a `kinefx::rigmatchpose` → `kinefx::mappoints` → `kinefx::fullbodyik` retarget network from a source clip/skeleton onto a target skeleton; optional `match_size` (Match Bounds) and explicit joint `mapping` |
| `apply_secondarymotion` | `apply_secondarymotion` | MUTATING | Yes | Drive `kinefx::secondarymotion`: add overshoot / lag / jiggle / spring on selected joints — **no DOP / sim network involved** |

All 8 tools are registered in `kinefx_tools.py` (`python/fxhoudinimcp/tools/`).
All 8 handlers are registered in `character_handlers.py` (`houdini/scripts/python/fxhoudinimcp_server/handlers/`).

**APEX BUILD — H22-BLOCKED:** `inspect_apex` (read-only) ships in this branch and is fully functional. The full APEX graph-authoring tool (`build_apex_graph` / `apex_graph_plan`, PR-7) is blocked pending the official Houdini 22 release and the interop contract with SideFX's official APEX Script MCP (~mid-July 2026). See `docs/homedini/plans/_agentic/architecture/0003-110-apex-extend-and-interop.adr.md` and `docs/portfolio_projects/PP12_houdini_mcp_agentic_bridge/110_mcp_kinefx_apex_surface/spec.md` §Non-Goals + §H22 Rescope.

---

## §2 — Agent↔Operator Call Sequence (§4.2)

This section documents the **live-verified** flow for a representative KineFX session. The sequence below matches `spec.md §4.2` and `pp12-110g live-verify-result.md`.

### 2.1 Sequence (Cascadeur import → skeleton query → retarget → secondary motion)

```
1. agent → kinefx_probe()
   ← { houdini: "21.0.729",
       nodes: { fbxcharacterimport: true, fbxanimimport: true, bonedeform: true,
                rigmatchpose: true, motiontransform: true, secondarymotion: true,
                apex_autorigcomponent: true } }

2. agent → import_fbx_animation(path="$HIP/in/cascadeur_run.fbx", cascadeur=true, dest="/obj/anim")
   ⏸  GATE — operator sees §4.3 payload:
       { tool: "import_fbx_animation",
         args: { path: "$HIP/in/cascadeur_run.fbx", cascadeur: true, dest: "/obj/anim" },
         node_plan: "kinefx::fbxanimimport @ /obj/anim",
         skeleton_summary: { count: 0, note: "to-be-created" },
         validator_contract: { fields: ["skeleton_summary"], note: "verify-after-mutate" } }
   operator → APPROVE
   ← { ok: true, node: "/obj/anim/fbxanimimport1",
       skeleton: { joints: 64, frame_range: [1, 90], convention_hint: "cascadeur" } }

3. agent → query_skeleton(node="/obj/anim/fbxanimimport1", frame=1)   [ungated — read-only]
   ← { joints: [ { name: "Hips", parent: null, rest: { t: [0,1.02,0], r: [...], s: [1,1,1] } },
                  { name: "Spine", parent: "Hips", ... }, ... ],
       count: 64 }

4. agent → import_fbx_character(path="$HIP/in/creature.fbx", dest="/obj/creature")
   ⏸  GATE — operator sees §4.3 payload (skeleton_summary: to-be-created) → APPROVE
   ← { ok: true, node: "/obj/creature/fbxcharacterimport1",
       skeleton: { joints: 58, has_skin_geo: true } }

5. agent → query_skeleton(node="/obj/creature/fbxcharacterimport1")   [ungated — read-only]
   ← { joints: [ { name: "root", parent: null, ... },
                  { name: "spine_01", parent: "root", ... }, ... ],
       count: 58 }
   → agent reasons on both joint sets; proposes source→target mapping to operator in plain text

6. agent → setup_retarget(source="/obj/anim/fbxanimimport1",
                           target="/obj/creature/fbxcharacterimport1",
                           method="rigmatchpose+fullbodyik", match_size=true,
                           mapping=[["Hips","root"],["Spine","spine_01"],["LeftUpLeg","thigh_l"],...])
   ⏸  GATE — operator sees §4.3 payload:
       { tool: "setup_retarget", args: { ... },
         node_plan: "kinefx::rigmatchpose → kinefx::mappoints → kinefx::fullbodyik",
         skeleton_summary: { source_joints: 64, target_joints: 58, frame_range: [1,90],
                             mapping_count: 57 },
         validator_contract: { fields: ["target_skeleton"], note: "verify-after-mutate" } }
   operator → APPROVE
   ← { ok: true, retarget_node: "/obj/creature/mcp_fullbodyik",
       target_skeleton: { joints: 58, frame_range: [1,90] },
       validator: { unmapped_target_joints: ["tail_05"], note: "tail not driven by source" } }

   → agent verifies motion landed (FR-12 verify-after-mutate re-query):
   agent → query_skeleton(node="/obj/creature/mcp_fullbodyik", frame=45)   [ungated]
   ← { joints: [ ... animated transforms differ from rest ... ] }   # loop closed

7. agent → apply_secondarymotion(node="/obj/creature/mcp_fullbodyik",
                                  joints=["tail_01","tail_02","tail_03","tail_04","tail_05","ear_L","ear_R"],
                                  params={ overshoot: 0.6, lag: 0.3, stiffness: 0.4 })
   ⏸  GATE — operator sees §4.3 payload:
       { tool: "apply_secondarymotion", args: { ... },
         node_plan: "kinefx::secondarymotion @ /obj/creature/mcp_fullbodyik",
         skeleton_summary: { count: 58, joints: [ ... real joint transforms from pre-cook query ] },
         validator_contract: { fields: ["skeleton_summary"], note: "verify-after-mutate" } }
   operator → APPROVE
   ← { ok: true, node: "/obj/creature/secondarymotion1",
       affected_joints: 7, frame_range: [1,90] }

8. Operator scrubs the timeline in the viewport and confirms the creature runs with a lagging tail.
   (Viewport visual confirmation is operator-driven; the agent verifies via query_skeleton.)
```

### 2.2 §4.3 Approval Payload (the operator-facing gate dialog)

When the 109 gate queues a mutating KineFX call, `list_pending_calls` returns the payload the operator reviews **before** approving:

```json
{
  "tool":              "<kinefx-tool-name>",
  "args":              { ... },
  "node_plan":         "<node_type(s) + destination path + key parms the agent proposes>",
  "skeleton_summary":  "<joint count + frame_range + relevant context (mapping, convention hint)>",
  "validator_contract": { "fields": ["<per-tool — see the §2.3 table>"], "note": "verify-after-mutate" }
}
```

The `validator_contract.fields` differ per tool (e.g. `skeleton_summary` for the import / secondary-motion tools, `target_skeleton` + `unmapped_target_joints` for retarget, `deformed_points` for bonedeform) — see the §2.3 return-shape table.

`preview_required=True` on all 5 mutating handlers means: if the pre-cook `query_skeleton` cannot find the source node (node missing, no `@name` point attribute), the gate **DENY**s immediately without queuing — the operator never sees a payload for a missing-node call. The `skeleton_summary` on the two FBX-import tools carries `{ count: 0, note: "to-be-created" }` because there is no pre-existing skeleton to query before import.

### 2.3 FR-12 Verify-After-Mutate (return shape)

After operator approval and successful handler execution, every mutating tool returns a structured result that includes a **re-queried post-cook state** — the agent never receives a success without verified scene evidence:

| Tool | FR-12 return shape |
|---|---|
| `import_fbx_character` | `{ ok, node, skeleton: { joints, has_skin_geo } }` |
| `import_fbx_animation` | `{ ok, node, skeleton: { joints, frame_range, convention_hint } }` |
| `setup_bonedeform` | `{ ok, node, deformed_points: <point count from output geo> }` |
| `setup_retarget` | `{ ok, retarget_node, target_skeleton: { joints, frame_range }, validator: { unmapped_target_joints, note } }` |
| `apply_secondarymotion` | `{ ok, node, affected_joints, frame_range }` |

**Live-verified behavior (pp12-110g):** `apply_secondarymotion` on a static (non-animated) skeleton returns `{ ok: false, error: "The attempted operation failed." }` — `kinefx::secondarymotion` requires an animated skeleton, not a static pose. The handler failed loud, returned the error, and cleaned up the stray node. The FR-12 success path (cook succeeds → real post-cook `query_skeleton` embedded in return) requires a real animated production scene and is operator-smoke-verified.

---

## §3 — Node-Type Table

| Tool | KineFX / Houdini Node Types Created |
|---|---|
| `import_fbx_character` | `kinefx::fbxcharacterimport` |
| `import_fbx_animation` | `kinefx::fbxanimimport` |
| `query_skeleton` | None (read-only — reads cooked geo from an existing node via `hou.Geometry`) |
| `inspect_apex` | None (read-only — summarizes an existing APEX `.rig` / `apex::*` node graph) |
| `setup_bonedeform` | `bonedeform` (SOP context; wired with three inputs: rest-skel, anim-skel, skinned-geo) |
| `setup_retarget` | `kinefx::rigmatchpose` → `kinefx::mappoints` → `kinefx::fullbodyik` |
| `apply_secondarymotion` | `kinefx::secondarymotion` |
| `kinefx_probe` | None (read-only — calls `_resolve_node_type` per type, writes nothing) |

All KineFX node types resolve via `_resolve_node_type` at the running Houdini 21 build before any handler executes. A node type that does not resolve (renamed or absent in the installed H21 dot-release) is reported as `false` in the probe result, not silently skipped. KineFX node naming can drift across H21 dot-releases; always run `kinefx_probe` first in a new session.

---

## §4 — Operator-Confirms-in-Viewport

The agent can query skeleton state before and after every mutating call (`query_skeleton` / `inspect_apex` are ungated read-only tools the agent drives freely). However, **visual confirmation of motion quality — "does the retarget look right?", "is the tail lagging naturally?", "did the secondary motion explode?" — is always operator-driven in the Houdini viewport**. The agent closes the verify loop on structural facts (joint counts, frame ranges, unmapped-joint lists, cook errors); the operator closes it on visual quality.

The canonical confirmation point is step 8 of the §2.1 sequence: after `apply_secondarymotion` returns `ok`, the operator scrubs the timeline in the Houdini viewport before the session moves on to the next operation.

This is not a limitation of the gate design — it is intentional. `kinefx::secondarymotion`, `kinefx::fullbodyik`, and `bonedeform` all produce motion that only a human eye can assess for correctness on a production character. The gate exists precisely to keep the operator in the loop at every mutation; the viewport scrub at the end of each mutating call is the operator's half of that loop.
