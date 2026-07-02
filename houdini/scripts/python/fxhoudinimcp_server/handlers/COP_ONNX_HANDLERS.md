# Copernicus-ONNX Handlers — Reference

Source of record for the 6 MCP Copernicus-ONNX tools added across the
`pp12-113-cop-onnx` branch (member 113, PRs pp12-113a through pp12-113e;
this document is pp12-113f, PR-7, the final PR that closes member 113).

---

## §1 — Tool List

| MCP Tool | Handler | Capability | Gated (`require_approval`) | Purpose |
|---|---|---|---|---|
| `houdini_cop_onnx_list_models` | `cop_onnx_list_models` | READONLY | No | Enumerate `.onnx` files under configured/given filesystem roots (path/size/mtime). No node, no onnx parse. |
| `houdini_cop_onnx_inspect_model` | `cop_onnx_inspect_model` | READONLY | No | Read a model's input/output tensor contract — either read-only off an EXISTING node (`node_path`), or via a scratch cop/onnx node that is ALWAYS destroyed in a `finally` (`model_path`). No node persists. |
| `houdini_cop_onnx_setup_node` | `cop_onnx_setup_node` | MUTATING | Yes | Create a PERSISTENT `cop/onnx` node, set `modelfile`, press `setupshapes`, optionally set vertical-flip parms, return the bound input/output tensor mapping. |
| `houdini_cop_onnx_set_provider` | `cop_onnx_set_provider` | MUTATING | Yes | Set the node's Execution Provider parm from the RUNTIME, platform-filtered menu; report `available_providers` + `will_bind`. |
| `houdini_cop_onnx_run_inference` | `cop_onnx_run_inference` | MUTATING | Yes | Cook an ALREADY-CONFIGURED `cop/onnx` node at a frame; return cook outcome + output-plane manifest + bound provider. |
| `houdini_cop_onnx_read_pixels` | `cop_onnx_read_pixels` | READONLY | No | Read a named output plane of a COOKED `cop/onnx` node as context-safe numeric data (summary/roi/sample). Genuinely read-only — refuses a stale node rather than cooking it. |

All 6 tools are registered in
`python/fxhoudinimcp/tools/cop_onnx_tools.py` with
`@mcp.tool(meta={"require_approval": ...})` — `False` for the 3 reads
(`list_models`, `inspect_model`, `read_pixels`), `True` for the 3 writes
(`setup_node`, `set_provider`, `run_inference`).
All 6 handlers are registered in
`houdini/scripts/python/fxhoudinimcp_server/handlers/cop_onnx_handlers.py`
via `register_handler(...)` — `Capability.READONLY` for the 3 reads;
`Capability.MUTATING` + `preview_fn=...` + `preview_required=True` for the
3 writes.

This observed state was confirmed by reading the shipped source (§8 below)
and matches `spec.md` §4.1's gated/ungated table exactly — no mismatch
found.

---

## §2 — Per-Tool Signatures and Returns

### 2.1 `houdini_cop_onnx_list_models`

**MCP wrapper params:**

| Param | Type | Default | Notes |
|---|---|---|---|
| `roots` | `list \| None` | `None` | Houdini-expandable root paths to scan for `.onnx` files. Defaults to `["$HIP/models", "$HOUDINI_USER_PREF_DIR/onnx"]` when `None` (see §5). |

**Success return shape:**

```json
{
  "ok": true,
  "models": [{"path": "<forward-slash path>", "size": 123456, "mtime": 1751328000.0}, ...],
  "roots_scanned": ["$HIP/models"],
  "missing_roots": ["$HOUDINI_USER_PREF_DIR/onnx"]
}
```

Filesystem-metadata-only — no node created, no model parsed. A missing
root is noted in `missing_roots` and never raises. A per-file `stat()`
failure is logged and the file is skipped, not fatal to the call.

**Error shape:** `{"ok": false, "error": "<reason>"}` (unexpected
exception only — a missing root is not an error).

---

### 2.2 `houdini_cop_onnx_inspect_model`

**MCP wrapper params:**

| Param | Type | Default | Notes |
|---|---|---|---|
| `model_path` | `str` | — | Path to the `.onnx` file (Houdini-expandable). Optional when `node_path` is given; **IGNORED** when `node_path` is given (`node_path` wins). |
| `node_path` | `str \| None` | `None` | Optional existing `cop/onnx` node path to read in place instead of the scratch-node mechanism. |

**TWO mutually exclusive paths:**

- **(A) `node_path` given** — READ-ONLY on the EXISTING node. Never sets `modelfile`, never presses `setupshapes`; only reads whatever `model_inputs`/`model_outputs` the node already has populated. An empty `modelfile` parm returns an informative `{ok:false, error}` rather than mutating the node to populate it (a freshly-created `onnx` node's `model_inputs`/`model_outputs` multiparms default to 1 each with empty-named placeholder entries — multiparm count alone is NOT a reliable "unconfigured" signal; an empty `modelfile` is). The returned contract's `model_path` field reports the NODE's own `modelfile` value, never the caller's `model_path` argument.
- **(B) `node_path` absent, `model_path` given** — the SCRATCH-NODE mechanism: creates a scratch `copnet` + scratch `onnx` node under `/obj`, sets `modelfile`, presses `setupshapes`, reads back the populated parms, then GUARANTEES both the scratch node AND the scratch net are destroyed in a `finally` — even on exception, even when the copnet was created but the onnx `createNode` then failed. Nothing persists in `/obj` after this path runs. This guaranteed cleanup is what keeps the tool READONLY/ungated.

At least one of `{model_path, node_path}` is required.

**Success return shape (`OnnxContract.to_dict()` merged with `ok: true`):**

```json
{
  "ok": true,
  "model_path": "<resolved path>",
  "inputs": [{"name": "input", "shape": [1, 3, "dynamic", "dynamic"], "dtype": "float32", "layout_guess": "NCHW"}],
  "outputs": [{"name": "output", "shape": [1, 3, "dynamic", "dynamic"], "dtype": "float32", "layout_guess": "unknown"}],
  "opset": null,
  "producer": null,
  "loadable": true,
  "error": null
}
```

**Error shape:** `{"ok": false, "error": "<reason>"}`. A scratch-cleanup
failure (path B only) surfaces `cleanup_failed: true` + `orphaned_path`
alongside `error` — fail-loud on a broken READONLY guarantee, never a
bare `ok: true` when cleanup failed even though the inspection read
itself succeeded:

```json
{"ok": false, "error": "read OK but scratch cleanup FAILED — orphaned <path>", "cleanup_failed": true, "orphaned_path": "<path>"}
```

If the inspection read had already failed AND cleanup also failed, the
ORIGINAL inspection error is preserved as `error` (never overwritten by
a misleading success string), with `cleanup_failed`/`orphaned_path` still
surfaced.

---

### 2.3 `houdini_cop_onnx_setup_node` (GATED)

**MCP wrapper params:**

| Param | Type | Default | Notes |
|---|---|---|---|
| `parent_path` | `str` | — | Path to an existing COP network (`parent.childTypeCategory() == hou.copNodeTypeCategory()`) the onnx node is created under. |
| `model_path` | `str` | — | Path to the `.onnx` file (Houdini-expandable). |
| `node_name` | `str` | `"agent_onnx"` | Name for the created node. |
| `setup_shapes` | `bool` | `True` | When `True`, press `setupshapes` after setting `modelfile` so the tensor mapping is populated. |
| `flip_input` | `bool \| None` | `None` | When not `None`, sets every input-instance `input_flip{i}` parm to `int(flip_input)`. |
| `flip_output` | `bool \| None` | `None` | When not `None`, sets every output-instance `output_flip{i}` parm to `int(flip_output)`. |

**Success return shape:**

```json
{
  "ok": true,
  "node_path": "/img/comp1/agent_onnx",
  "model_path": "<expanded model_path>",
  "input_tensors": [{"name": "input1", "shape": [1, 3, 64, 64], "dtype": "float32", "cop_input_index": 1}],
  "output_tensors": [{"name": "output1", "shape": [1, 3, 64, 64], "dtype": "float32", "cop_plane": "n_output"}],
  "warnings": [],
  "applied": true
}
```

**Error shape:** `{"ok": false, "error": "<reason>"}`.

The created node **PERSISTS** — unlike `inspect_model`'s scratch node
(always destroyed), this node is the agent's node. This is precisely why
the tool is GATED: it mutates the scene by design. Parent validation is
`parent.childTypeCategory() == hou.copNodeTypeCategory()` — NEVER
inferred from `createNode()` raising, because `createNode('onnx', ...)`
under a non-COP parent (e.g. a geo SOP network) can SILENTLY SUCCEED by
resolving the bare name `onnx` to an unrelated `Sop/onnx` node type (see
§7). NEVER presses `reload` (houdini-001, §7).

A SINGLE `bridge.execute` call — the wrapper performs no result
interpretation and returns the result **verbatim**, including a
pending-approval / preview response shape from the PP12-109 gate (a
normal, valid return, not a failure).

---

### 2.4 `houdini_cop_onnx_set_provider` (GATED)

**MCP wrapper params:**

| Param | Type | Default | Notes |
|---|---|---|---|
| `node_path` | `str` | — | Path to an existing `onnx` node. |
| `provider` | `str` | — | Requested Execution Provider token (any case). |

**Success return shape:**

```json
{
  "ok": true,
  "node_path": "/img/comp1/agent_onnx",
  "requested": "CUDA",
  "available_providers": ["automatic", "cpu", "cuda", "directml"],
  "will_bind": "cuda",
  "warnings": []
}
```

**Error shape:** `{"ok": false, "error": "<reason>"}`. The ONLY error
case is when the `onnx` node exposes NO Execution Provider options at
all (`available == []`) — an unavailable *requested* value NEVER errors;
it falls back to `automatic` (or the first available provider) with a
non-empty warning, never a raise (FR-4).

Reads the RUNTIME, platform-filtered `provider` menu
(`node.parm('provider').menuItems()` — e.g. `('automatic', 'cpu', 'cuda',
'directml')` on Windows, no `coreml`) and delegates the requested/
available mapping to the pure `choose_provider` helper
(`cop_onnx_model.py`). NEVER hardcodes the provider list.

---

### 2.5 `houdini_cop_onnx_run_inference` (GATED)

**MCP wrapper params:**

| Param | Type | Default | Notes |
|---|---|---|---|
| `node_path` | `str` | — | Path to an existing, already-configured `cop/onnx` node. |
| `frame` | `int \| float \| None` | `None` | Frame to cook at. Defaults to the current `hou.frame()` when `None`. |

**Success return shape (cook attempted, whether it succeeded or failed):**

```json
{
  "ok": true,
  "cooked": true,
  "node_path": "/img/comp1/agent_onnx",
  "bound_provider": "automatic",
  "cook_ms": 412.3,
  "output_planes": [{"name": "output1", "xres": 64, "yres": 64, "channels": 3, "dtype": "float32"}],
  "errors": [],
  "warnings": []
}
```

**Error shape (bad TARGET / unexpected failure only):** `{"ok": false, "error": "<reason>"}`.

**LOCKED ok/cooked split (FR-5 no-silent-success):** a shape-mismatched /
mis-wired / unconfigured cook is `ok: true` + `cooked: false` +
`errors: [...]` (reported, NOT raised) — never a silent `cooked: true`
and never an unhandled raise. Only a bad target (node not found / not a
`cop/onnx` node) or an unexpected non-cook exception is `ok: false`.
`output_planes` is DETERMINISTICALLY `[]` on a failed cook (the
manifest-assembly loop only runs `if cooked:`) — a failed cook never
exposes stale/partial planes. `bound_provider` reads
`node.parm('provider').evalAsString()` (the token, e.g. `"automatic"`),
NEVER `.eval()` (returns the menu index — wrong). NEVER presses `reload`
(houdini-001) or `setupshapes` (that is `setup_node`'s job — pressing it
here would be scope creep + a re-run risk).

A SINGLE `bridge.execute` call — the wrapper returns the result
verbatim, including a failed-cook shape or a pending-approval / preview
shape from the 109 gate (both normal, valid returns, not errors).

---

### 2.6 `houdini_cop_onnx_read_pixels`

**MCP wrapper params:**

| Param | Type | Default | Notes |
|---|---|---|---|
| `node_path` | `str` | — | Path to an existing, cooked `cop/onnx` node. |
| `plane` | `str \| None` | `None` | Output plane name to read. Defaults to the node's first output plane (`node.outputNames()[0]`) when omitted. |
| `mode` | `str` | `"summary"` | One of `"summary"`, `"roi"`, `"sample"`. |
| `roi` | `list \| None` | `None` | Required for `mode="roi"` — a 4-element `[x0, y0, x1, y1]` box. |
| `max_pixels` | `int` | `4096` | Server-side pixel budget, hard-clamped to `ABS_MAX_PIXELS` regardless of this value (§6). |
| `downsample` | `int \| None` | `None` | Explicit stride for `mode="sample"`. When `None` or `<= 0`, the stride is derived from `max_pixels`. |
| `page` | `int` | `0` | Zero-based page index for roi/sample pagination. |
| `page_size` | `int` | `1024` | Page size for roi/sample pagination, also subject to the budget ceiling. |

**Success return shape — `summary` mode:**

```json
{
  "ok": true, "cooked": true, "mode": "summary",
  "sampled": true, "stride": 1,
  "plane": "output1", "xres": 64, "yres": 64, "channels": 3, "dtype": "float32",
  "stats": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0], "mean": [0.5, 0.5, 0.5], "nan_count": 0, "inf_count": 0},
  "histogram": {"bins": 32, "counts": [[...], [...], [...]]}
}
```

**Success return shape — `roi` / `sample` mode (merges `ReadbackPage.to_dict()`):**

```json
{
  "ok": true, "cooked": true,
  "xres": 64, "yres": 64, "channels": 3, "dtype": "float32",
  "stride": 1,
  "plane": "output1", "mode": "roi",
  "pixels": [[0.1, 0.2, 0.3], ...],
  "page": 0, "page_size": 1024, "total_pages": 1, "truncated": false
}
```

**A stale/uncooked node (read-only, no cook triggered) — a valid, reportable outcome, not an error:**

```json
{
  "ok": true, "cooked": false,
  "node_path": "/img/comp1/agent_onnx", "plane": null,
  "message": "node not cooked — run cop_onnx_run_inference first"
}
```

**Error shape (bad target, wrong category, bad plane, inverted roi box):** `{"ok": false, "error": "<reason>"}`.

A SINGLE `bridge.execute` call — the wrapper returns the result
verbatim, including the "not cooked" shape or an error shape (neither is
treated as a wrapper-level failure).

---

## §3 — The Tensor-Contract / Flip Trap (FR-7)

Copernicus's `cop/onnx` node's tensors carry no metadata that tells the
agent whether a 4D shape is channel-first or channel-last, and most 2D
models train in top-left-origin space while Houdini images are
bottom-left-origin. This family surfaces both hazards rather than
letting the agent guess:

- **NCHW vs NHWC (`layout_guess`, PR-1, pure).** `guess_layout(shape)`
  (`cop_onnx_model.py`) is a pure heuristic: a 4D shape is `"NCHW"` when
  `shape[1]` is a plausible channel count (`1`, `3`, or `4`) AND
  `shape[2]`/`shape[3]` are large-or-dynamic image dims; `"NHWC"` when
  `shape[-1]` is a plausible channel count AND `shape[1]`/`shape[2]` are
  large-or-dynamic. Ambiguous/non-4D/all-dynamic shapes degrade to
  `"unknown"` — it never raises. `layout_guess` is computed for INPUT
  tensors only; outputs always report `"unknown"` (cop/onnx exposes no
  automatic layout inference for outputs, and guessing on an output
  tensor is out of scope).
- **The Flip Input/Output Vertically origin trap.** The node's
  `input_flip{i}` / `output_flip{i}` parms convert between Houdini's
  bottom-left image origin and a typical 2D model's top-left origin —
  the classic correctness mistake FR-7 exists to head off. `setup_node`
  accepts `flip_input`/`flip_output` and, when not `None`, sets every
  per-instance flip parm so the agent can act on the trap directly
  rather than hand-tuning a single global flag.
- **`Setup Shapes from Model` is the ONLY authoritative populate.**
  There is NO static `onnxruntime`/`onnx` Python path in Houdini 21
  (`onnxruntime` is C++-only — `ModuleNotFoundError` confirmed live) and
  NO nodeless `hou` API for reading an `.onnx` tensor contract. Every
  read in this family — `inspect_model`'s scratch-node path, or
  `setup_node`'s persistent node — goes through
  `node.parm("setupshapes").pressButton()`. The agent never hand-types a
  shape; the node is the single source of truth.
- **Synthesized `float32` dtype.** `cop/onnx` exposes NO dtype field
  anywhere in the Setup-Shapes read-back surface for either inputs or
  outputs (confirmed live on a float32 fixture) — `model_input_data{i}`
  ("Data") is a plain internal Houdini token, not a dtype string. Every
  tensor's `dtype` in `inspect_model`/`setup_node` is therefore
  SYNTHESIZED as the literal `"float32"` — this matches ONNX's
  overwhelmingly common tensor dtype and is a stated, intentional
  stand-in. `run_inference`'s output-plane manifest is different: it
  READS a real dtype via `layer.storageType()` →
  `normalize_plane_dtype()` (e.g. `imageLayerStorageType.Float32` →
  `"float32"`), because a cooked `hou.ImageLayer` genuinely exposes a
  storage type where the pre-cook Setup-Shapes parms do not.

---

## §4 — The Gate Posture (member #109)

3 of the 6 tools are READONLY/ungated; 3 are MUTATING/gated. Ungated
tools run without a prompt; gated tools preview → queue for operator
approval → execute only on approval.

| Tool | Capability | `require_approval` | Why |
|---|---|---|---|
| `list_models` | READONLY | `False` | Filesystem metadata only — no node, no parse. |
| `inspect_model` | READONLY | `False` | Node-path mode never mutates; model-path mode's scratch node is GUARANTEED destroyed in a `finally` — the READONLY guarantee is structural, not a convention. |
| `setup_node` | MUTATING | `True` | Creates a node that PERSISTS in the scene. |
| `set_provider` | MUTATING | `True` | Mutates an existing node's `provider` parm. |
| `run_inference` | MUTATING | `True` | Cooks — burns GPU/CPU and can touch disk. |
| `read_pixels` | READONLY | `False` | Refuses to cook a stale node (§7 `needsToCook()` guard) — the expensive cook stays behind the GATED `run_inference`. |

**The gated flow:** a gated call's `preview_fn` runs READ-ONLY validation
(never mutates) and RAISES `hou.OperationFailed` on an invalid target —
the gate DENIES the call before it is ever queued for operator approval
(an invalid target is denied at the gate, not merely flagged for the
operator to reject):

- `_preview_setup_node` — validates `parent_path` resolves to a real COP
  network via `parent.childTypeCategory() == hou.copNodeTypeCategory()`.
  Raises on a missing parent or a non-COP parent.
- `_preview_set_provider` — validates `node_path` resolves to a real
  `onnx` node (`node.type().name() == "onnx"`).
- `_preview_run_inference` — validates `node_path` resolves to a real
  Copernicus `cop/onnx` node via BOTH `node.type().name() == "onnx"` AND
  `node.type().category() == hou.copNodeTypeCategory()` (a name-only
  check is insufficient — see §7). Does NOT cook; cooking in the preview
  would perform the exact mutation the gate exists to gate.

On approval, the handler executes and returns its normal success/error
shape. All three `preview_fn`s are called POSITIONALLY by the gate
middleware as `preview_fn(params)` — a single `params: dict` argument,
NOT `**params`.

---

## §5 — Model-Root Config

`list_models` scans, by default, two Houdini-expandable filesystem roots
(`_DEFAULT_MODEL_ROOTS` in `cop_onnx_handlers.py`):

```python
["$HIP/models", "$HOUDINI_USER_PREF_DIR/onnx"]
```

Expanded via `hou.text.expandString(raw_root)` at call time
(`asset-reference-discipline.md`). A caller-supplied `roots` list
overrides the defaults entirely (no merge). A root that does not exist
on disk is appended to `missing_roots` and skipped — never raises.

---

## §6 — The FR-6 Context-Safe Readback Budget (`read_pixels`)

`ABS_MAX_PIXELS = 4096` is an ABSOLUTE server ceiling applied to EVERY
mode regardless of the caller's `max_pixels`/`page_size` — no call ever
returns more pixels than this. `SUMMARY_HISTOGRAM_BINS = 32` is locked
per spec §4.2.

```
budget = min(max(1, max_pixels), ABS_MAX_PIXELS)
effective_page_size = min(max(1, page_size), budget)
```

- **`summary` (default)** — per-channel min/max/mean + a 32-bin
  histogram + nan/inf counts, over a budget-bounded STRIDED SAMPLE (via
  `clamp_readback(w, h, channels, budget)` → stride →
  `bounded_sample_coords`), reading pixels one at a time via
  `layer.bufferIndex(x, y)`. NEVER `layer.allBufferElements()` over the
  full plane (~192 MiB for a 4K plane) — the summary rung stays a
  bounded sample regardless of plane resolution, ~1-2 KB per call.
- **`roi`** — a bounded `[x0, y0, x1, y1]` crop, clamped to
  `[0, w] x [0, h]`. An out-of-bounds-but-valid box is CLAMPED, not
  rejected; an inverted/empty box (`x1 <= x0` or `y1 <= y0`) IS rejected
  (`ok: false`). Paginated via the LAZY `bounded_page_coords` helper —
  only the requested `[start, end)` page slice is ever computed, never
  the whole box materialized first and sliced after.
- **`sample`** — a strided sample; `downsample` (when `> 0`) sets the
  stride explicitly, otherwise the stride derives from
  `clamp_readback(...)`. Paginated via the LAZY `bounded_sample_coords`
  helper (same never-materialize-the-whole-grid discipline). A budget
  clamp beyond the true strided-grid size still marks `truncated: true`
  even when the clamped total fits in a single page.
- **NaN/Inf are COUNTED, never silently dropped** (`count_nan_inf`) —
  `nan_count`/`inf_count` are always reported; min/max/mean are computed
  over the finite values only, `None` when there are zero finite
  samples.

This is the FR-6 no-full-frame-dump invariant: a 2048² 3-channel float
plane in `summary` mode returns well under the naive ~48 MB a raw dump
would cost.

---

## §7 — Load-Bearing Gotchas

- **houdini-001 (catalog).** NEVER press the `"reload"` button
  (`parm("reload").pressButton()`) on a `cop/onnx` node BEFORE
  `"setupshapes"` — pressing `reload` on a freshly `modelfile`-set node
  segfaults Houdini (`COP_ONNXParms::buildFromOp`, confirmed live via a
  full native crash dump). Every handler in this family presses
  `"setupshapes"` alone (which always re-reads the file at `modelfile`)
  and NEVER `"reload"`.
- **The new Copernicus `CopNode` has NONE of the classic COP2 methods.**
  `xRes()`/`yRes()`/`planes()`/`depth()`/`stage()`/`imageBounds()` are
  ALL absent (`AttributeError`) — the fork's existing
  `cop_handlers.get_cop_info`/`get_cop_layer` (COP2-only) do NOT work on
  `cop/onnx`. The real surface: `node.outputNames()` → plane name
  tuple; `node.outputLabels()`; `node.outputDataTypes()`; `node.layer(i)`
  (an INT index ONLY — a string arg raises `TypeError`) → a
  `hou.ImageLayer` with `.bufferResolution()` → `(xres, yres)`,
  `.channelCount()` (NOT `numChannels`) → `int`, `.storageType()` → a
  `hou.imageLayerStorageType` enum (normalized via
  `normalize_plane_dtype`), and `.bufferIndex(x, y)` → a
  channel-count-tuple of floats for a single pixel.
- **Parent/target validation must use `childTypeCategory()` /
  `type().category()`, never rely on `createNode()` raising.** A
  `copnet`'s OWN `type().category()` is `"Object"` (the `/obj` context)
  — NOT `"Cop"`; the correct non-mutating check for `setup_node`'s
  preview is `parent.childTypeCategory() == hou.copNodeTypeCategory()`.
  Critically, `createNode('onnx')` under a NON-cop parent (e.g. a geo SOP
  network) can SILENTLY SUCCEED — Houdini resolves the bare name `'onnx'`
  to an UNRELATED `Sop/onnx` node type under a SOP context. Relying on
  `createNode()` raising as the parent-validation signal is THEREFORE
  WRONG. `run_inference`/`read_pixels` similarly validate BOTH
  `node.type().name() == "onnx"` AND
  `node.type().category() == hou.copNodeTypeCategory()` (a name-only
  check, as `set_provider` uses, is insufficient for these two — a bare
  name can resolve to the unrelated `Sop/onnx`).
- **`run_inference`'s cook semantics.** `node.cook(force=True,
  frame_range=(frame_to_cook, frame_to_cook))` RAISES
  `hou.OperationFailed` AND populates `node.errors()` on a cook error
  (BOTH, together) — live-confirmed on an input-less node, a
  shape-mismatched model, and a 2-dynamic-axis model. LOCKED algorithm
  order: capture `cook_exc`; read `errors()`/`warnings()` AFTER the cook
  attempt (whether or not it raised); if `cook_exc is not None and not
  errors`, fold `str(cook_exc)` into `errors`; THEN
  `cooked = cooked_from_errors(errors)` (pure: `cooked` iff `errors` is
  empty). A raised cook is therefore NEVER reported `cooked: true`.
- **`read_pixels` is read-only via the `needsToCook()` guard, NOT a
  `layer(idx) is None` check.** On this Houdini build,
  `node.layer(idx)` on a CONFIGURED-but-stale node does NOT simply
  return `None` — it triggers an IMPLICIT COOK as a side effect, which
  would violate the read-only guarantee. The correct read-only guard is
  `node.needsToCook()`, checked BEFORE ever calling `layer()`;
  `needsToCook()` is side-effect-free (verified live — never itself
  triggers a cook, however many times called). The handler trusts it
  only when it returns an actual `bool` (a mocked-hou node's
  `needsToCook()` returns a `MagicMock`, not a bool, so the pytest
  suite's `layer(idx) is None` staleness contract still governs there —
  both the mocked and the live paths are correct without either
  short-circuiting the other). `layer(idx) is None` remains as a
  defensive fallback for the mock case only; on live Houdini this
  fallback branch is never reached because `needsToCook()` already
  caught the stale case.
- **`multi_input.onnx` is the cookable fixture; `identity_dynamic.onnx`
  is NOT.** `identity_dynamic.onnx` has 2 dynamic axes (H, W) →
  Houdini's runtime error "Only can deduce a single dynamic axis, 2 axes
  are dynamic" — not cookable. `multi_input.onnx` (2 inputs, STATIC
  `[1,3,64,64]`) is the cookable fixture used by the hython-smoke and
  live-MCP verification.
- **The 4-bug wrapper convention class** (per `mcp-subprocess-delegation.md`
  and the MCP fork build lessons): `ctx: Context` (not `Any`);
  `_get_bridge(ctx)` accessed through the module reference so
  `patch("fxhoudinimcp.server._get_bridge", ...)` intercepts correctly
  in tests; `bridge.execute("cmd", {params})` (not `.call`); and the
  dispatcher calls `handler(**params)` (keyword-only handler
  signatures) — every handler/wrapper pair in this family mirrors the
  shipped `usd_export_handlers.py` / `usd_export_tools.py` pattern
  exactly rather than inventing the convention.

---

## §8 — grep-confirmed-against-source

The §1 tool-list table and this document's gate-posture claims were
verified directly against the shipped registrations:

- `houdini/scripts/python/fxhoudinimcp_server/handlers/cop_onnx_handlers.py`
  — all 6 `register_handler("cop_onnx_*", ..., Capability.{READONLY,MUTATING}[, preview_fn=..., preview_required=True])`
  calls read in full.
- `python/fxhoudinimcp/tools/cop_onnx_tools.py` — all 6
  `@mcp.tool(meta={"require_approval": ...})` wrappers read in full.

Both match `spec.md` §4.1's gated/ungated table with zero deviation: 3
READONLY/ungated (`list_models`, `inspect_model`, `read_pixels`) and 3
MUTATING/gated (`setup_node`, `set_provider`, `run_inference`), each
gated tool carrying a `preview_fn` + `preview_required=True`.

### Verification

**Headless unit suite (mock `hou`, off-DCC):** `python/fxhoudinimcp/tests/test_cop_onnx_model.py`,
`test_cop_onnx_tools.py`, `test_cop_onnx_setup_tools.py`,
`test_cop_onnx_run_inference_{model,handler,tools}.py`,
`test_cop_onnx_read_pixels_{model,handler,tools}.py` — **281/281 tests
passing** (docs-only change; zero `.py` delta from this PR, so this
count is unchanged from the prior PR).

**`hython` integration + live-MCP rungs:** validated per member 113's
PR-2 through PR-5 progress files — inspect → setup → provider → run →
read_pixels exercised against live Houdini 21.0.729 through the PP12-109
gate, including the `multi_input.onnx` cookable fixture and a
deliberately shape-mismatched sad-path cook.
