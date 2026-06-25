# Engine-Export Handlers — Reference

Source of record for the 7 MCP engine-export tools added in the `pp12-111-engine-export` branch.

---

## §1 — Tool List

| MCP Tool | Handler | Capability | Gated (`require_approval`) | Purpose |
|---|---|---|---|---|
| `houdini_export_probe_versions` | `probe_versions` | READONLY | No | Probe installed Houdini, SideFX Labs VAT, and UE versions; returns a `VersionTriple` + `skew_table` verdict |
| `houdini_export_validate_budget` | `validate_budget` | READONLY | No | Server-side budget check for a node + target engine: validates tri count (≤500 000) and frame-range length (≤600 frames); returns `{verdict, checks[]}` |
| `houdini_export_vat` | `export_vat` | MUTATING | Yes | Bake a Vertex Animation Texture sequence to disk via a `labs::vertex_animation_textures` ROP under `/out`; writes an `ExportManifest` sidecar |
| `houdini_export_alembic_ue` | `export_alembic_ue` | MUTATING | Yes | Bake a UE-compatible Alembic (`rop_alembic`) inside the parent geo node (SOP context); writes an `ExportManifest` sidecar |
| `houdini_export_fbx` | `export_fbx` | MUTATING | Yes | Bake FBX (SOP-context `rop_fbx` ROP inside the parent geo node); writes an `ExportManifest` sidecar |
| `houdini_export_chaos_gc` | `export_chaos_gc` | MUTATING | Yes | Bake a Chaos Geometry Cache Alembic; validates `unreal_gc_piece` prim-attribute contiguity (FR-7 gap check) BEFORE any ROP creation — non-contiguous pieces are refused without writing any file; writes an `ExportManifest` sidecar on success |
| `houdini_export_niagara` | `export_niagara` | MUTATING | Yes | Bake a Niagara cache via `labs::niagara_rop` under `/out`; output is normalised to `.hbjson` via `niagara_normalize_output()`; writes an `ExportManifest` sidecar |

All 7 tools are registered in `export_tools.py` (`python/fxhoudinimcp/tools/`).
All 7 handlers are registered in `export_handlers.py` (`houdini/scripts/python/fxhoudinimcp_server/handlers/`).

---

## §2 — Version-Skew Compatibility Table

Source: `python/fxhoudinimcp/skew_table.py`. Versions are normalised before lookup: Houdini → major only; UE and SideFX Labs VAT → major.minor.

| Houdini | Labs VAT | UE | Verdict | Notes |
|---|---|---|---|---|
| 21 | 3.0 | 5.4 | **ok** | Validated support window: Houdini 21 / SideFX Labs VAT 3.0 / UE 5.4. This combination is production-ready. |
| 21 | 1.0 | 5.4 | **block** | SideFX Labs VAT 1.x not designed for modern Houdini 21 and UE 5.x |
| 21 | 3.0 | 4.0 | **block** | UE 4.x not compatible with VAT 3.x shader interface |
| 18 | 3.0 | 5.4 | **block** | Houdini 18 not supported by modern Labs VAT 3.x pipeline |
| 18 | 1.0 | 4.0 | **block** | Houdini 18 + Labs VAT 1.x + UE 4.x: doubly blocked |
| 17 | 3.0 | 5.4 | **block** | Ancient Houdini 17 — not supported |
| any other | any other | any other | **warn** | Unknown combination — proceed with caution; run `houdini_export_probe_versions` and verify shader compatibility manually |

`probe_versions` embeds the real version-triple from the running Houdini + installed Labs package, then looks it up in this table and returns the `verdict` + `notes` to the caller before any bake is attempted.

---

## §3 — Gated Call Sequence (§4.2)

This section documents the **live-verified** flow for a MUTATING export tool. All five MUTATING tools follow the same sequence. The evidence below is from the `pp12-111g` live-verify run against Houdini 21.0.729 (pid 19576), gate mode `propose`.

### 3.1 Sequence

```
1. Client calls a gated MCP tool (e.g. houdini_export_fbx)
   gate mode == "propose"
   → dispatcher queues the call; returns a pending_id WITHOUT executing the handler
   → preview is computed server-side at queue time:
       validate_budget() runs immediately (even if the agent forgot to call it first)
       _preview_fbx() assembles: { version_triple, budget_verdict, out_paths, rop_plan_schema_version }

2. Client calls list_pending_calls
   → response includes the queued call with:
       preview = {
           version_triple:   { houdini: "21.0.729", labs_vat: "3.0", ue: "5.4" },
           budget_verdict:   { verdict: "pass", checks: [ ... ] }  -- OR {ok:false, error:"..."} on failure
           out_paths:        [ "/path/to/output.fbx" ],
           rop_plan_schema_version: 1
       }
   → operator sees the budget verdict BEFORE approving
   → if budget_verdict shows a failure (e.g. "Node not found"), the operator rejects; no bake occurs

3. Operator calls approve_pending_call(pending_id)
   → dispatcher re-validates the budget at approve time (divergence check)
   → if scene changed between queue and approve and the verdict diverged, the response carries divergence_warning
   → if no divergence: handler executes in Houdini's main thread via hdefereval
       ROP is created, cooked, then deleted (cleanup)
       Output file is written to disk
       ExportManifest sidecar (.export.json) is written beside the output:
           {
               "tool":       "houdini_export_fbx",
               "args":       { ... },
               "out_paths":  [ "/path/to/output.fbx" ],
               "version_triple": { ... },
               "validator":  { "verdict": "pass", "checks": [ ... ] }   ← validate_budget() result
           }

4. approve_pending_call returns { ok: true } on success
   Gate returns to 0 pending calls
```

### 3.2 Negative-case behavior (live-verified)

When `validate_budget()` finds the node missing or budget exceeded, it returns `{ok: false, error: "Node not found"}` (or a checks-failed result) in the `preview.budget_verdict`. The operator sees this in `list_pending_calls` **before** approving. The call may still be approved (the handler will re-validate and fail fast), or rejected. The gate does NOT hard-DENY on a bad budget_verdict — `validate_budget` handles missing nodes gracefully by returning a structured error rather than raising, so the DENY-on-preview-exception path is reserved for exceptions in the preview function itself.

### 3.3 Chaos-GC FR-7 contiguity pre-check

`export_chaos_gc` performs a contiguity check on the `unreal_gc_piece` prim attribute **before any ROP creation**. If the piece indices are non-contiguous (e.g. pieces 0, 2 with gap at 1), the handler returns `{ok: false, error: "unreal_gc_piece gap at 1"}` and writes nothing to disk. This check runs inside the handler at approve-time, not at queue/preview time.

---

## §4 — UE-Side Manual Verification

After a successful bake, the TD verifies the exported asset in UE5 as follows:

**FBX / Alembic:**
1. In the UE5 Content Browser, drag the exported file into the project (or use **File > Import into Level**).
2. Confirm the import dialog shows the expected geometry (mesh count, bone hierarchy for FBX with skeleton).
3. Place the asset in a level and verify it renders without missing materials or broken transforms.

**VAT (Vertex Animation Textures):**
1. Import the exported `.exr` position and normal textures plus the base mesh.
2. Assign the SideFX Labs VAT material (VAT 3.0 shader) and wire the textures to the appropriate slots.
3. Play or scrub the timeline — geometry should animate via the vertex shader without a skeleton.

**Chaos Geometry Cache (Alembic):**
1. Enable the **Chaos Geometry Collection** or **Alembic Geometry Cache** plugin in the UE5 project.
2. Import the exported `.abc` file and verify piece count matches the Houdini source (`unreal_gc_piece` max + 1).
3. Trigger the destruction in PIE and confirm all pieces separate correctly.

**Niagara (`.hbjson`):**
1. In the UE5 Niagara editor, open the target Niagara System and locate the Houdini data interface.
2. Point the data interface at the exported `.hbjson` file.
3. Play the simulation — particles should match the Houdini cache in count and trajectory.

The `ExportManifest` sidecar (`.export.json`) beside each output records `tool`, `args`, `out_paths`, `version_triple`, and `validator` (the `validate_budget` result) for audit purposes.
