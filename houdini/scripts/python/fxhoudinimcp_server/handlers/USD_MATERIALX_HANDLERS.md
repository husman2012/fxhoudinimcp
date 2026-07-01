# USD/MaterialX Export Handlers — Reference

Source of record for the 6 MCP USD/MaterialX export tools added across the
`pp12-112-usd-export` branch (member 112, PRs pp12-112b through pp12-112e;
this document is pp12-112f, PR-6, the final PR that closes member 112).

---

## §1 — Tool List

| MCP Tool | Handler | Capability | Gated (`require_approval`) | Purpose |
|---|---|---|---|---|
| `houdini_usd_inspect_layer` | `usd_inspect_layer` | READONLY | No | Read a composed USD layer/stage summary via `pxr` — default prim, root prims, sublayers, current format, MaterialX-material presence. No write. |
| `houdini_usd_validate` | `usd_validate` | READONLY | No | Run the `usd-publish-discipline` check set against a layer/stage (minimal / preflight / postwrite mode); structured `verdict` + per-check results. No write. |
| `houdini_mtlx_inspect` | `mtlx_inspect` | READONLY | No | Parse a `.mtlx` file via the `MaterialX` Python API; list nodegraphs, surface nodes, inputs with absolute paths; run `Document.validate()`. No write. |
| `houdini_usd_export_layer` | `usd_export_layer` | MUTATING | Yes | `Sdf.Layer.Export()` of a composed/edited layer; format chosen **by file extension** (magic-bytes-by-extension). |
| `houdini_usd_export_rop` | `usd_export_rop` | MUTATING | Yes | Drive the `/out`-context `usd` ROP to write a chosen LOP node's composed stage to disk (current frame or a `[start, end]` range). |
| `houdini_mtlx_edit` | `mtlx_edit` | MUTATING | Yes | Apply node/input value edits to an existing `.mtlx` via the `MaterialX` API and write with `writeToXmlFile` — existing inputs only, never creates a new input. |

All 6 tools are registered in
`python/fxhoudinimcp/tools/usd_export_tools.py` with
`@mcp.tool(meta={"require_approval": ...})` — `False` for the 3 reads,
`True` for the 3 writes.
All 6 handlers are registered in
`houdini/scripts/python/fxhoudinimcp_server/handlers/usd_export_handlers.py`
via `register_handler(...)` — `Capability.READONLY` for the 3 reads;
`Capability.MUTATING` + `preview_fn=...` + `preview_required=True` for the
3 writes.

This observed state was confirmed by grep against the shipped source
(§3 below) and matches `spec.md` §4.1's gated/ungated table exactly — no
mismatch found.

---

## §2 — Tool Signatures and Returns

### 2.1 `houdini_usd_inspect_layer`

**MCP wrapper params:**

| Param | Type | Default | Notes |
|---|---|---|---|
| `node_or_layer` | `str` | — | Houdini LOP node path (e.g. `"/stage/lop1"`) or a USD file path; supports Houdini variable expansion (`$HIP/...`) |

**Success return shape (`LayerSummary`):**

```json
{
  "ok": true,
  "default_prim": "/asset",
  "root_prims": ["/asset"],
  "sublayers": ["<layer path>", ...],
  "current_format": "in-memory",
  "has_mtlx_material": true
}
```

**Error shape:** `{"ok": false, "error": "<reason>"}` (FR-2 on an
unresolvable node/path, FR-5 on any unexpected exception).

---

### 2.2 `houdini_usd_validate`

**MCP wrapper params:**

| Param | Type | Default | Notes |
|---|---|---|---|
| `target` | `str` | — | Houdini LOP node path or USD file path |
| `out_path` | `str \| None` | `None` | Setting this alone enables **preflight** mode |
| `actual_format` | `str \| None` | `None` | Setting this + `out_path` enables **postwrite** mode |
| `texture_paths` | `list \| None` | `None` | Texture path list for the abs-path portability check (postwrite mode) |
| `checks` | `list \| None` | `None` | Reserved — MUST be `None`; a non-`None` value is rejected fail-loud (M-3, not yet implemented) |

**Success return shape (B-1 compliant):**

```json
{
  "ok": true,
  "mode": "minimal | preflight | postwrite",
  "omitted_checks": ["<check id>", ...],
  "verdict": "pass | warn | fail",
  "checks": [
    {"id": "no_world_wrapper", "status": "pass"},
    {"id": "format_matches_ext", "status": "pass"},
    {"id": "abs_texture_paths", "status": "warn", "msg": "<reason>"}
  ],
  "wrote_files": false
}
```

- `mode="minimal"` (no `out_path`) omits `format_extension_known`,
  `format_matches_ext`, `abs_texture_paths`.
- `mode="preflight"` (`out_path` set, no `actual_format`) omits
  `format_matches_ext`, `abs_texture_paths`.
- `mode="postwrite"` (both `out_path` and `actual_format` set) omits nothing.

**Error shape:** `{"ok": false, "error": "<reason>"}` (FR-2 on an empty
`target`, FR-5 on any unexpected exception, or the M-3 `checks`-not-`None`
rejection).

---

### 2.3 `houdini_mtlx_inspect`

**MCP wrapper params:**

| Param | Type | Default | Notes |
|---|---|---|---|
| `mtlx_path_or_doc` | `str` | — | A `.mtlx` **file path** (Houdini `$`-var expansion supported). v1 accepts a file path only — an inline MaterialX doc-string is NOT supported, despite the parameter name (kept per spec.md §4.1 signature). |

**Success return shape (`MtlxSummary`):**

```json
{
  "ok": true,
  "nodegraphs": ["NG_asset"],
  "surface_nodes": ["standard_surface1"],
  "inputs_with_abs_paths": ["base_color"],
  "validate": {"ok": true, "errors": []}
}
```

**Error shape:** `{"ok": false, "error": "<reason>"}` (FR-2 on a missing/
unreadable path, FR-5 on an unexpected exception; MaterialX-unavailable
degrades to a clear error via `_require_mtlx()`, per R-4).

---

### 2.4 `houdini_usd_export_layer` (GATED)

**MCP wrapper params:**

| Param | Type | Default | Notes |
|---|---|---|---|
| `node` | `str` | — | LOP node path or USD file path |
| `out_path` | `str` | — | Output file path; format resolved by extension |
| `flatten` | `bool` | `False` | `True` exports a flattened single-layer composition; `False` exports the stage's root layer as-is |
| `default_prim` | `str \| None` | `None` | Optional prim path set as the layer's defaultPrim before export |

**Success return shape:**

```json
{
  "ok": true,
  "out_path": "<expanded out_path>",
  "format": "usda | usdc | usdz",
  "actual_format": "<format from magic-byte detection of the written file>",
  "validator_post": { "ok": true, "mode": "postwrite", "verdict": "...", "checks": [...] }
}
```

**Error shape:** `{"ok": false, "error": "<reason>"}`.

A SINGLE `bridge.execute` call — the wrapper performs no result
interpretation and returns the result **verbatim**, including a
pending-approval / preview response shape from the PP12-109 gate (that is a
normal, valid return, not a failure).

---

### 2.5 `houdini_usd_export_rop` (GATED)

**MCP wrapper params:**

| Param | Type | Default | Notes |
|---|---|---|---|
| `lop_node` | `str` | — | LOP node path whose composed stage the ROP renders (named `lop_node`, not `node`, to be precise — this drives a specific LOP's composed stage) |
| `out_path` | `str` | — | Output file path; format resolved by extension |
| `frame_range` | `list \| None` | `None` | `None` exports the current frame only; `[start, end]` exports that range (drives `trange=1` + `parmTuple('f')`) |

**Success return shape:** same envelope as `houdini_usd_export_layer`
(`ok`, `out_path`, `format`, `validator_post`); on a post-write validation
failure after the file was already written, `out_path` is still returned
alongside the error so the caller knows a partial write occurred.

**Error shape:** `{"ok": false, "error": "<reason>", "out_path": "<written path or null>"}`.

Not a full-time-history flatten like `houdini_usd_export_layer` — this
drives the `/out`-context `usd` ROP node (`loppath` / `lopoutput` /
`trange` / `parmTuple('f')` / `execute`), current-frame by default.

---

### 2.6 `houdini_mtlx_edit` (GATED)

**MCP wrapper params:**

| Param | Type | Default | Notes |
|---|---|---|---|
| `mtlx_path` | `str` | — | Path to the existing `.mtlx` to edit |
| `edits` | `list` | — | List of edit ops (each targets an existing node + input by name; never creates a new input) |
| `out_path` | `str` | — | Path to write the edited document |

**Success return shape:**

```json
{
  "ok": true,
  "out_path": "<expanded out_path>",
  "edits_applied": 1,
  "validate": {"ok": true, "errors": []}
}
```

**Error shape:** `{"ok": false, "error": "<reason>"}` (includes the
`_preview_mtlx_edit` `source_parseable: False` denial payload when the
source document cannot be parsed at all — see §3).

---

## §3 — Gate Wiring + Pre/Post Validation

All source-of-record confirmation below is grep-verified against the
shipped registrations in
`houdini/scripts/python/fxhoudinimcp_server/handlers/usd_export_handlers.py`
and the `@mcp.tool` wrappers in
`python/fxhoudinimcp/tools/usd_export_tools.py`.

| Tool | `register_handler(...)` (line) | Capability | `preview_fn` | `preview_required` | `require_approval` |
|---|---|---|:--:|---|:--:|:--:|
| `usd_inspect_layer` | line 303 | READONLY | — | — | `False` |
| `usd_validate` | line 403 | READONLY | — | — | `False` |
| `usd_export_layer` | lines 575–581 | MUTATING | `_preview_export_layer` | `True` | `True` |
| `usd_export_rop` | lines 917–923 | MUTATING | `_preview_export_rop` | `True` | `True` |
| `mtlx_inspect` | line 1121 | READONLY | — | — | `False` |
| `mtlx_edit` | lines 1401–1407 | MUTATING | `_preview_mtlx_edit` | `True` | `True` |

The 3 reads bypass the PP12-109 gate entirely (`Capability.READONLY`); the
3 writes route through it (`Capability.MUTATING` + `preview_fn` +
`preview_required=True`), matching `spec.md` §4.1's gated/ungated table
with zero deviation.

### 3.1 — Pre-validation embedded in each write's preview

Each write's `preview_fn` runs a **read-only, non-mutating** pre-check and
raising inside it causes the PP12-109 gate to **DENY** the call before it
is ever queued for operator approval (`preview_required=True`):

- **`_preview_export_layer`** (usd_export_layer) — resolves
  `format_for_extension(out_path)` (raises `ValueError` → DENY on an
  unrecognized extension) and calls `usd_validate(target=node_path,
  out_path=out_path)` in **preflight** mode, embedding the result under
  `"pre_validation"` in the returned approval payload alongside
  `"out_path"`, `"resolved_format"`, `"flatten"`, `"default_prim"`, and
  `"no_world_wrapper": true`.
- **`_preview_export_rop`** (usd_export_rop) — validates that `lop_node`
  resolves to a real, cooked Houdini node with a non-`None` composed
  `.stage()` (raises `hou.OperationFailed` → DENY if not), then calls the
  same preflight-mode `usd_validate(...)` and embeds it under
  `"pre_validation"`, alongside `"resolved_format"`, `"frame_range"`,
  `"driven_via"`, and `"no_world_wrapper": true`.
- **`_preview_mtlx_edit`** (mtlx_edit) — attempts to parse the source
  `.mtlx` document; on failure returns an operator-visible
  `"source_parseable": False` denial payload. On success it returns
  `"source_parseable": True`, `"edits_shape_ok": True`,
  `"edits_preview": [...]` (a per-edit preview of what will change), and
  `"pre_validation": {"ok": bool, "errors": [...]}` — the
  `doc.validate()` result of the **unedited** source document.

Every write's `preview_fn` therefore embeds a pre-check the operator sees
**before** approving — FR-7's pre-validation requirement — matching the
spec's naming: `usd_validate` preflight for the two USD writes,
`source_parseable`/`edits_preview` for `mtlx_edit`.

### 3.2 — Post-write validation / round-trip verify returned by each handler

- **`usd_export_layer`** returns `"validator_post"` — a **postwrite**-mode
  `usd_validate(...)` call (both `out_path` and `actual_format` set,
  `actual_format` read back from the written file's magic bytes) run
  after the `Sdf.Layer.Export()` call succeeds.
- **`usd_export_rop`** likewise returns `"validator_post"` — the same
  postwrite-mode `usd_validate(...)` call after the ROP cook/write
  completes. If validation fails after the file was already written, the
  handler still returns `"out_path"` (the partial-write path) alongside
  the error so the caller is never left guessing whether a file landed.
- **`mtlx_edit`** returns a **round-trip verify** rather than a
  `usd_validate` call: after writing via `mx.writeToXmlFile`, it
  re-parses the just-written document (`doc2`) and re-resolves each
  edited node by the **canonical path recorded during PASS 1** (never by
  re-running a name search, so the resolution is unambiguous), then
  returns `"validate": {"ok": bool, "errors": [...]}` from
  `doc2.validate()` — confirming the edit landed and the document is
  still well-formed post-write.

This is the FR-7/FR-8 contract: every disk write is pre-validated (shown
to the operator in the gate payload) and post-validated (returned to the
agent) — never a silent write.

---

## §4 — Agent↔Operator Call Sequence

Matches `spec.md` §4.2 and §4.3. Representative sequence: the agent has
already edited a Solaris stage via fxhoudinimcp's existing in-memory LOP
tools (`set_usd_attribute` / `create_lop_node`, outside this family) and
now publishes it plus its MaterialX material.

```text
1. agent → houdini_usd_inspect_layer(node_or_layer="/stage/OUT")   [ungated]
   ← { ok:true, default_prim:"/asset", root_prims:["/asset"], sublayers:[...],
       current_format:"in-memory", has_mtlx_material:true }

2. agent → houdini_usd_validate(target="/stage/OUT")   [ungated, pre-flight/minimal]
   ← { ok:true, mode:"minimal", verdict:"warn",
       checks:[ {id:"no_world_wrapper", status:"pass"},
                {id:"abs_texture_paths", status:"warn", msg:"2 texture inputs use absolute paths"} ] }
   → agent relays the warn to the operator; proposes fixing paths or proceeding.

3. agent → houdini_mtlx_inspect(mtlx_path_or_doc="$HIP/mat/asset.mtlx")   [ungated]
   ← { ok:true, nodegraphs:["NG_asset"], surface_nodes:["standard_surface1"],
       validate:{ok:true}, inputs_with_abs_paths:["base_color"] }

4. agent → houdini_usd_export_layer(node="/stage/OUT",
                                    out_path="$HIP/publish/asset.usdc",
                                    flatten=false, default_prim="/asset")
   ⏸  GATE (PP12-109) — operator sees the _preview_export_layer payload:
       { out_path:".../asset.usdc", resolved_format:"usdc",
         pre_validation:{ verdict:"warn", checks:[...] },
         flatten:false, default_prim:"/asset", no_world_wrapper:true }
   operator → APPROVE
   ← { ok:true, out_path:".../asset.usdc", format:"usdc", actual_format:"usdc",
       validator_post:{ ok:true, mode:"postwrite", verdict:"warn", checks:[...] } }

5. agent → houdini_mtlx_edit(mtlx_path="$HIP/mat/asset.mtlx",
                             edits=[{node:"standard_surface1", input:"base_color",
                                     set:"value", value:"./tex/base.exr"}],
                             out_path="$HIP/publish/asset.mtlx")
   ⏸  GATE (PP12-109) — operator sees the _preview_mtlx_edit payload:
       { source_parseable:true, edits_shape_ok:true,
         edits_preview:[{node:"standard_surface1", input:"base_color", ...}],
         pre_validation:{ok:true, errors:[]} }
   operator → APPROVE
   ← { ok:true, out_path:".../asset.mtlx", edits_applied:1,
       validate:{ok:true, errors:[]} }
```

### §4.1 — Approval flow (§4.3)

```
agent calls a gated tool (usd_export_layer / usd_export_rop / mtlx_edit)
        │
        ▼
PRE-VALIDATE (preview_fn, read-only)  ──► embed pre_validation / source_parseable
        │                                  in the approval payload
        ▼
APPROVAL PAYLOAD (PP12-109 Security Gate)
        │
   operator review ──reject──► {ok:false}; nothing written
        │ approve
        ▼
hdefereval.executeInMainThreadWithResult(_do)   # pxr/MaterialX-only, main thread
        │
        ▼
POST-VALIDATE on the written file (validator_post) OR round-trip verify (mtlx_edit)
        │
        ▼
return { ok, out_path, format/edits_applied, validator_post/validate } ──► agent
```

Read-only tools (`houdini_usd_inspect_layer`, `houdini_usd_validate`,
`houdini_mtlx_inspect`) bypass the gate entirely and never produce a
pending-approval response.

---

## §5 — usd-publish-discipline Note

This family exists to encode the repo's `usd-publish-discipline` rule as
executable behavior, not just a lint pass:

- **pxr/MaterialX-API only — NO regex** on `.usda`/`.usdc`/`.mtlx` text.
  Every write goes through `Sdf.Layer.Export()`, the `/out` `usd` ROP, or
  the `MaterialX` Python API (`readFromXmlFile` / `Input.setValueString` /
  `writeToXmlFile`). There is no text-substitution code path anywhere in
  this handler family — the discipline validator (`usd_validate`) exists
  specifically to catch hand-authored junk that *would* result from a
  regex shortcut.
- **Magic-bytes-by-extension via `Sdf.Layer.Export()`** — the output
  format is never forced by a flag; it is resolved from the `out_path`
  file extension (`format_for_extension()`): `.usda` → ascii, `.usdc` /
  `.usd` → crate, `.usdz` → packaged. The `usd_validate` `postwrite` mode
  re-derives `actual_format` from the **written file's real magic bytes**
  and checks it against the extension (`format_matches_ext`), so a
  mismatch is caught after the fact, not just asserted before it.
- **No `/World` or `/root` wrapper injected.** Neither `usd_export_layer`
  nor `usd_export_rop` auto-injects a root wrapper prim; the authored root
  structure is preserved as-is. Both preview functions report
  `"no_world_wrapper": true` in the approval payload, and `usd_validate`'s
  `no_world_wrapper` check fails loud if a layer is found to be wrapped.
- **Automatic pre + post hython/pxr validation on every write** — FR-7.
  Every gated write's `preview_fn` embeds a `pre_validation` (or
  `source_parseable`/`edits_preview` for `mtlx_edit`) result in the
  operator-facing approval payload, and every write handler returns a
  post-write `validator_post` (or round-trip `validate`) result to the
  agent. The operator is never asked to approve a write without seeing
  its own discipline-check verdict first, and the agent never receives a
  bare `{"ok": true}` without confirmation the written result is still
  valid.

---

## §6 — Verification

### Headless unit suite (mock `hou`/`pxr`, off-DCC)

`python/fxhoudinimcp/tests/test_mtlx_tools.py` and
`python/fxhoudinimcp/tests/test_mtlx_handler_edge_cases.py` — **83/83
tests passing** as of this document's authoring (docs-only change; zero
`.py` delta from this PR, so this count is unchanged from the prior PR).
Covers `format_for_extension` table cases, `/World`-wrapper detection,
default-prim-missing, gated-tool-refuses-without-approval, MaterialX edit
node/input resolution (`_resolve_node`), round-trip re-validation, and the
`_preview_mtlx_edit` `source_parseable: False` denial path.

### `hython` integration (live Houdini, headless)

Per `spec.md` §9.2: `.usdc`/`.usda` round-trip export with binary
crate/ascii-header assertion and no `/World` root; `usd_export_rop` writing
from a `/stage` LOP; `mtlx_inspect`/`mtlx_edit` node/input round-trip with
re-validation; a deliberately `/World`-wrapped layer failing
`no_world_wrapper`; and a write-without-approval producing
`{ok:false, pending_approval:true}` with nothing written to disk.

### Phase-0 hython probes cited in the shipped source

`_preview_export_rop`'s module comment records a live Houdini
21.0.729 hython probe (2026-07-01) confirming the `/out` `usd` ROP's
`loppath`/`lopoutput`/`trange`/`parmTuple('f')`/`execute` parm surface,
`node.errors()` tuple-truthiness, and real time-varying multi-sample
writes. `_mtlx_validate`'s docstring records a matching probe confirming
Houdini's bundled MaterialX (1.39.3) binds `doc.validate()` as a
`(bool, str)` tuple.
