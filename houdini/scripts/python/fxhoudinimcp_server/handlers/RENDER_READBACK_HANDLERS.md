# Render-Readback Handlers — Reference

Source of record for the 4 MCP render-readback tools added across the
`pp12-114` branch (units pp12-114c, pp12-114e, pp12-114f).

---

## §1 — Tool List

| MCP Tool | Handler | Capability | Gated (`require_approval`) | Purpose |
|---|---|---|---|---|
| `render_lint_settings` | `render_lint_settings` | READONLY | No | Read a Karma render node's USD stage and run handoff_linter rule-preset checks; returns a `results[]` + `summary` + `ready_to_render` verdict |
| `render_parse_exr` | `render_parse_exr` | READONLY | No | Parse an on-disk EXR via `hoiiotool` and return a channel/metadata manifest (`ExrManifest`), including `crypto_layers` |
| `render_read_pixels` | `render_read_pixels` | READONLY | No | Read pixel data from an on-disk EXR via OIIO; supports `summary`, `roi`, and `sample` modes with pagination and `max_pixels` clamping |
| `render_compare` | `render_compare` | READONLY | No | Compare two EXR renders A and B plane-by-plane; returns `aovs_a`, `aovs_b`, `common`, `selected`, `per_plane` deltas, and a `verdict` string |

All 4 tools are registered in
`python/fxhoudinimcp/tools/render_readback_tools.py` with
`meta={"require_approval": False}`.
All 4 handlers are registered in
`houdini/scripts/python/fxhoudinimcp_server/handlers/render_readback_handlers.py`
via `register_handler(..., Capability.READONLY)`.

---

## §2 — Tool Signatures and Returns

### 2.1 `render_lint_settings`

**MCP wrapper params (client schema):**

| Param | Type | Default | Notes |
|---|---|---|---|
| `render_node` | `str` | — | Scene path of the Karma render node, e.g. `"/stage/karma1"` |
| `preset` | `str` | `"nuke_safe"` | Rule preset name |

**Success return shape:**

```json
{
  "render_node": "<str>",
  "preset": "<str>",
  "results": [
    {"rule": "<str>", "severity": "<ok|warn|error>", "message": "<str>", ...}
  ],
  "summary": {"ok": 0, "warn": 0, "error": 0},
  "ready_to_render": true
}
```

**Error shape (FR-2 / FR-5):**

```json
{"ok": false, "error": "<human-readable reason>"}
```

FR-2 applies when `render_node` is empty or does not resolve via `hou.node()`.

---

### 2.2 `render_parse_exr`

**MCP wrapper params:**

| Param | Type | Default | Notes |
|---|---|---|---|
| `exr_path` | `str` | — | Path to EXR; Houdini variable expansion supported |
| `subimage` | `int \| None` | `None` | When set, inspect only this subimage index |

**Success return shape (`ExrManifest`):**

```json
{
  "exr_path": "<str>",
  "is_multipart": false,
  "subimages": 1,
  "compression": "zip",
  "xres": 1920,
  "yres": 1080,
  "channels": [
    {"name": "R", "layer": null, "dtype": "float16"},
    {"name": "depth.Z", "layer": "depth", "dtype": "float32"}
  ],
  "crypto_layers": ["<layer_name>", ...],
  "metadata": {"<key>": "<value>"}
}
```

`crypto_layers` lists any CryptomatteV2 manifest layers found in the EXR
metadata — relevant for downstream cryptomatte handoff validation.

**Error shape:** `{"ok": false, "error": "<reason>"}` (FR-2 on missing path,
FR-5 on `hoiiotool` failure).

Note: `exr_path` in the response echoes the **original (unexpanded)** input so
callers can round-trip the value they passed in.

---

### 2.3 `render_read_pixels`

**MCP wrapper params:**

| Param | Type | Default | Notes |
|---|---|---|---|
| `source` | `str` | — | Path to EXR; Houdini variable expansion supported (e.g. `"$HIP/render/beauty.0001.exr"`) |
| `plane` | `str` | `"C"` | AOV plane name; `"C"` and `"beauty"` both select the top-level beauty channels |
| `mode` | `str` | `"summary"` | `"summary"` / `"roi"` / `"sample"` |
| `roi` | `list[int] \| None` | `None` | `[x0, y0, x1, y1]` half-open bounding box for `mode="roi"` |
| `max_pixels` | `int` | `4096` | Maximum pixel count before auto-downsampling |
| `downsample` | `int` | `1` | Manual stride factor for `mode="sample"` |
| `page` | `int` | `0` | Zero-based page index for paginated reads |
| `page_size` | `int` | `1024` | Pixels per page |

**Success return shape (`ReadbackResult`):**

> On success `render_read_pixels` returns the bare result dict below — there is **no `ok` key**. Failure returns `{ok: false, error}`. Callers MUST NOT gate on `result.get("ok")` being `True` (same convention as `render_compare`, §2.4).

```json
{
  "plane": "C",
  "xres": 1920,
  "yres": 1080,
  "channels": 3,
  "dtype": "float32",
  "mode": "summary",
  "stats": {
    "min": [0.0, 0.0, 0.0],
    "max": [1.0, 1.0, 1.0],
    "mean": [0.5, 0.5, 0.5],
    "nan_count": 0,
    "inf_count": 0
  },
  "histogram": {"bins": 16, "counts": [[...], [...], [...]]},
  "pixels": [],
  "page": 0,
  "page_size": 1024,
  "total_pages": 1,
  "truncated": false
}
```

Key semantics:

- `xres` / `yres` always reflect **source frame dimensions** — not the ROI
  dimensions — even in `mode="roi"`.
- `pixels` is `[]` in `mode="summary"`.
- `truncated: true` when the pixel list was longer than `max_pixels` before
  pagination; never a full-frame pixel dump.
- `stats` and `histogram` are always present in all three modes; they summarise
  the *selected* pixel subset (full frame for `sample`, ROI rect for `roi`,
  full frame for `summary`).

**Error shape:** `{"ok": false, "error": "<reason>"}` (FR-2 on empty `source`
or unknown `mode`, FR-5 on OIIO failure or file-not-found).

EXR-source v1: `source` must be an on-disk file path. In-scene COP node paths
are not yet supported and return `{ok: false, error: "..."}`.

---

### 2.4 `render_compare`

**MCP wrapper params:**

| Param | Type | Default | Notes |
|---|---|---|---|
| `a` | `str` | — | Path to render A; Houdini variable expansion supported |
| `b` | `str` | — | Path to render B |
| `planes` | `list[str] \| None` | `None` | AOV planes to compare; `None` compares all planes common to both renders |
| `metric` | `str` | `"stats"` | `"stats"` / `"mae"` / `"psnr"` — validated before any file I/O |

**Success return shape (`CompareReport`, returned directly — no `ok` key):**

```json
{
  "aovs_only_in_a": ["<plane>", ...],
  "aovs_only_in_b": ["<plane>", ...],
  "aovs_common":    ["<plane>", ...],
  "per_plane": [
    {
      "plane":         "C",
      "mean_delta":    [0.001, -0.002, 0.000],
      "max_abs_delta": [0.03, 0.02, 0.01],
      "mae":           0.0015,
      "psnr":          62.3,
      "moved":         true
    }
  ],
  "verdict": "C changed (mae 0.002); depth unchanged"
}
```

Key semantics:

- On **success** the shape has no `ok` key — callers **must not** gate on
  `result.get("ok")` being `True`.
- On **failure** the shape is `{"ok": false, "error": "<reason>"}`.
- `psnr` is `null` (JSON) when non-finite (`inf` or `nan`); this is the
  all-non-finite sentinel meaning *no numeric comparison was possible* —
  distinct from an unchanged plane (`moved: false`).
- `metric` is validated **before** any file I/O; an unknown value returns
  `{ok: false, error: ...}` immediately.
- In v1 all three metric values (`stats`, `mae`, `psnr`) return the full
  `per_plane` list with every field; the `metric` param is reserved for
  future server-side field filtering.
- `planes` that do not appear in the common set are silently skipped;
  `planes` that results in an empty intersection returns `{ok: false, error: ...}`.

---

## §3 — Reuse, Not Duplication (PP11/05 Engine)

The render-readback subsystem **does not reimplement** EXR parsing or linting
logic. It imports from the EXR cryptomatte handoff linter authored in **project
05** (PP11/05):

```
homedini.rendering.handoff_linter.stage_reader   — USD stage reader
homedini.rendering.handoff_linter.rules           — rule evaluation engine
homedini.rendering.handoff_linter.presets         — preset registry (nuke_safe, etc.)
homedini.rendering.handoff_linter.exr_inspector   — parse_exr_manifest / ExrManifest
```

These are loaded via `fxhoudinimcp.handoff_linter_loader.ensure_on_path()` in
the handlers. The linter is discovered at runtime via `$HOMEDINI_PYTHON` or
`$UT` — it is never vendored into the fork.

**Why this matters:** any rule bug fixed in project 05 propagates automatically
to `render_lint_settings` without a fork change. The
`crypto_layers` field in `render_parse_exr` is populated by
`exr_inspector.parse_exr_manifest`, which shares the PP11/05 cryptomatte
manifest-parsing logic. The `handoff_model` dataclasses defined in PP11/05 are
the canonical schema for `RuleResult.to_dict()` entries in the
`render_lint_settings` response.

---

## §4 — Readback Budget and Context Safety

`render_read_pixels` never dumps an entire frame into the MCP response. Three
mechanisms prevent context blowup:

1. **Mode selection:** `"summary"` returns only statistics and a histogram —
   `pixels: []`, zero pixel data transferred. Use this for a quick sanity check.
2. **`max_pixels` clamp (default 4096):** in `"roi"` and `"sample"` modes,
   the pixel list is truncated at `max_pixels` rows before pagination.
   When truncation occurs, `"truncated": true` is set in the response.
3. **Pagination:** `page` / `page_size` (default 1024) let callers retrieve
   pixel data in slices without re-reading the file.

Recommended workflow for agent use:

1. Call `render_read_pixels(source=..., mode="summary")` to confirm the
   frame dimensions, channel count, and value range without transferring
   pixels.
2. If specific pixel values are needed, call with `mode="roi"` and a tight
   `roi` rect, or `mode="sample"` with an appropriate `downsample` factor.
3. Never call with `mode="roi"` on a full-resolution frame without a narrow
   `roi` or a low `page_size` — the result may hit `max_pixels` and be
   truncated.

---

## §5 — Gate Posture (FR-10)

All four render-readback tools are `Capability.READONLY` and are registered
with `require_approval=False`. They are **ungated** — no PP12 member #109
queuing or operator approval is required.

This reflects FR-10: read-only inspection tools run without gating because they
make no scene mutations and write nothing to disk. Any future tool in this
family that **automates a render launch or writes files** must be registered
with `Capability.MUTATING` and `require_approval=True`, routing through the
PP12 member #109 gate.

---

## §6 — OIIO Discovery

**`render_parse_exr`** uses `hoiiotool` (the CLI) to parse the EXR manifest.
The handler in `render_readback_handlers.py` delegates to
`exr_inspector.parse_exr_manifest`, which discovers `hoiiotool` at runtime via
`$HB/hoiiotool` (Houdini's `$HB` bin directory). No pip install or system
OIIO path is required; `hoiiotool` is bundled with every Houdini install.

**`render_read_pixels` and `render_compare`** use the OIIO **Python bindings**
(`import OpenImageIO`) for pixel reads. The pp12-114g live-verify session
confirmed that `import OpenImageIO` works in Houdini 21 hython (OIIO 2.5.18.0).
The reader (`render_readback_reader.py`) imports `OpenImageIO` directly for
`ImageBuf`-based pixel reads and also imports `numpy` for the
`get_pixels(...).reshape(...)` step. Both are available in Houdini's bundled
hython — no external install needed.

`list_exr_planes` and `read_exr_plane` in `render_readback_reader.py` raise
`ImportError` if `OpenImageIO` is unavailable, surfacing a clear message: the
tool must be called from Houdini's hython, not from a plain CPython environment.

---

## §7 — Verification

### Per-tool hython smokes

Each tool has an individual hython smoke test under
`houdini/scripts/python/fxhoudinimcp_server/handlers/tests/`:

- `test_render_lint_settings_hython.py` — calls the handler with a real Karma
  node in a minimal `.hip` fixture; asserts `ready_to_render` is a bool and
  `summary` has the expected keys.
- `test_render_parse_exr_hython.py` — calls the handler with a test EXR;
  asserts `channels` list is non-empty, `xres`/`yres` are positive ints, and
  `crypto_layers` is a list.
- `test_render_read_pixels_hython.py` — exercises all three modes (`summary`,
  `roi`, `sample`) and asserts `truncated`, `total_pages`, and `pixels` shapes
  are consistent.
- `test_render_compare_hython.py` — compares a file with itself (expects all
  `mae == 0`, `moved == false`) and with a pixel-shifted copy (expects
  `moved == true` on at least one plane).

### Aggregate smoke

`houdini/scripts/python/fxhoudinimcp_server/handlers/tests/test_render_readback_hython.py`
runs all four handlers end-to-end in a single hython session to verify handler
registration and bridge dispatch work together.

### `render_compare` live FastMCP rung

`render_compare` was additionally verified via a `claude -p` live subprocess
call (`mcp-subprocess-delegation.md` pattern) that called the MCP tool against
a running Houdini 21.0.729 session in gate mode `propose`. The live rung
confirms the full client → wrapper → `bridge.execute` → dispatcher → handler
path — the only rung that exercises the FastMCP wire format end-to-end.
