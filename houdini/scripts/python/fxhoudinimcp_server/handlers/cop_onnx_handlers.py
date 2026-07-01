"""Handlers: cop_onnx_list_models, cop_onnx_inspect_model.

Both are READ-ONLY, UNGATED (Capability.READONLY) — the Copernicus-ONNX
inspection surface (PP12-113 PR-2).

cop_onnx_list_models   — enumerate .onnx files under configured/given
                          filesystem roots (path/size/mtime). No node, no
                          onnx parse.
cop_onnx_inspect_model — read a model's input/output tensor contract via
                          TWO mutually exclusive paths:
                            (A) node_path given -> READ-ONLY on the
                                EXISTING node: never sets modelfile, never
                                presses setupshapes -- reads whatever the
                                node's model_inputs/model_outputs already
                                hold. model_path is IGNORED when node_path
                                is given (node_path wins).
                            (B) node_path absent (model_path given) -> the
                                SCRATCH-NODE mechanism: create a scratch
                                cop/onnx node in a scratch copnet, set
                                modelfile, press setupshapes, read back the
                                populated model_inputs/model_outputs
                                multiparms, then GUARANTEE both the scratch
                                node AND the scratch net are destroyed in a
                                finally -- even on exception, and even when
                                the copnet was created but the onnx node
                                createNode then failed. Nothing persists in
                                /obj after a call on this path.
                          Both paths build the OnnxContract via the pure
                          contract_from_setup_shapes() helper. Guaranteed
                          cleanup on path B (and read-only-ness on path A)
                          is what keeps the tool READONLY/ungated (operator
                          decision 2026-07-01).

REV2 FOLD (codex tier-2 BLOCK 2026-07-01 -- 3 Blockers + 1 Minor closed):
  B1 -- node_path branch was mutating the caller's node (set modelfile +
        pressed setupshapes on an EXISTING node). Fixed: node_path is now
        strictly READ-ONLY -- it never touches modelfile/setupshapes, it
        only reads the node's ALREADY-populated parms. model_path is now
        Optional; at least one of {model_path, node_path} is required;
        node_path wins when both are given (model_path ignored, documented
        here and in the docstring below).
  B2 -- the scratch NET was reused/persisted across calls (never
        destroyed), and a copnet-created-but-onnx-createNode-failed path
        orphaned the net entirely. Fixed: DROPPED the persistent-net-reuse
        design. Both the scratch node AND the scratch net are ALWAYS
        created and destroyed together, tracked by a created_net flag set
        immediately after the copnet createNode succeeds, so the finally
        destroys the net on every path -- including create-copnet-then-
        onnx-createNode-fails.
  B3 -- a scratch destroy() failure in the finally was swallowed (logged,
        then the success-path ok=True return stood). Fixed: cleanup
        failure is now surfaced fail-loud -- ok=False with a
        cleanup_failed marker and the orphaned path, even though the
        inspection read itself succeeded, so the caller/operator knows the
        READONLY guarantee may have been broken for this call.
  Minor -- _normalize_shape_dims's trailing-zero-vs-padding heuristic now
        explicitly treats an all-zero read (no dim populated at all) as an
        empty/unsupported-v1 shape ([]) rather than silently returning [].

REV3 FOLD (codex round-2 re-review NEEDS_CHANGES 2026-07-01, threadId
019f1f73, operator-approved 3rd round -- 2 small correctness fixes;
B2/B3/Minor above are UNCHANGED/CLOSED):
  B1-metadata -- in the node_path branch, the returned OnnxContract.
        model_path was echoing the caller's model_path ARG instead of the
        NODE's actual configured model. Fixed: resolved_model_path in the
        node_path branch now reads the node's OWN modelfile parm value
        (already captured as modelfile_value during the "is a model
        configured" check) -- model_path is still ignored in node_path
        mode (REV2 FOLD B1), and now the returned contract's model_path
        field reflects that truthfully instead of echoing a possibly
        bogus/unrelated caller-supplied model_path.
  New defect -- the finally block's cleanup-failure return always claimed
        "inspection read succeeded but scratch cleanup FAILED", even when
        the try block's inspection had ALREADY failed (an exception was
        caught, or an early FR-2-style return fired) before cleanup ran.
        Fixed: a read_error_message tracker (None == read succeeded) is
        set on every non-success try-block outcome; the finally's
        cleanup-failure branch now returns (a) the "read OK but cleanup
        FAILED" message when read_error_message is None, or (b) the
        ORIGINAL inspection error message when it is not -- never
        overwriting a real failure with a misleading success string.
        Both cases still surface cleanup_failed=True + orphaned_path.

Grounded against (pp12-113b lockedFieldContract + the 2026-07-01
cop-onnx-inspection-api-memo.md, including the orchestrator's live hython
probe): there is NO static onnxruntime/onnx Python path in Houdini 21
(onnxruntime is C++-only, ModuleNotFoundError confirmed live) and NO
nodeless hou API for reading an .onnx tensor contract — Setup Shapes from
Model is the only documented mechanism.

Phase-0 hython probe findings that shape this handler (recorded verbatim
in the impl-bundle; see also docs/homedini/plans/_agentic/_artifacts/
hou-dev/pp12-113b/impl-bundle.json):
  - onnxruntime/onnx: BOTH ModuleNotFoundError in hython; zero onnx .pyd
    hits anywhere under $HFS. Mechanism = temp-node-Setup-Shapes, locked.
  - CRASH FOUND + WORKED AROUND: pressing the "reload" button
    (parm("reload").pressButton()) BEFORE "setupshapes" on a freshly
    modelfile-set cop/onnx node segfaults Houdini
    (COP_ONNXParms::buildFromOp, confirmed live via a full native crash
    dump). This handler NEVER presses "reload" — it sets modelfile then
    presses "setupshapes" directly, which is safe and sufficient (Setup
    Shapes always (re)reads the file at modelfile).
  - modelfile MUST be set with forward slashes (hou.text.expandString
    already normalizes to forward slashes on Windows; do not pass a
    raw backslash path).
  - model_input_data{i} ("Data") is a plain STRING parm holding an
    internal Houdini token (observed: "n_input") — it is NOT a dtype
    string. cop/onnx exposes NO dtype field anywhere in the Setup-Shapes
    read-back surface (confirmed for both inputs and outputs on a
    float32 fixture). Per Phase-0 item #3's documented fallback, dtype
    is SYNTHESIZED as the literal "float32" when unavailable — this
    matches ONNX's overwhelmingly common tensor dtype and is a stated,
    intentional stand-in that the ADR/spec should revisit if dtype-
    sensitive downstream behavior is ever needed (out of scope for this
    read-only inspection PR).
  - model_input_shape{i}{d} for d=1..9 is ALWAYS a fixed 9-slot block;
    unused trailing dims beyond the model's actual rank read back as the
    literal int 0 (NOT -1, NOT blank). This handler stops reading dims
    at the first 0 IF at least one dim has already been read AND no
    dynamic (-1) sentinel has yet appeared at that position — i.e. it
    reads sequentially and stops at the first 0 that is NOT part of the
    real declared shape. In practice for a rank-4 model dims 1-4 are
    populated (concrete ints or -1) and dims 5-9 are 0.
  - A symbolic/dynamic ONNX dim (declared as a string dim_param, e.g.
    "H"/"W") resolves to the literal int -1 in
    model_input_shape{i}{d}/model_output_shape{i}{d}. This handler
    normalizes -1 -> the literal string "dynamic" before calling
    contract_from_setup_shapes (matching the pure helper's + red tests'
    "dynamic" sentinel).
  - model_input_channelfirst{i} label is "Collate Channels Separately"
    (NOT a layout-guess signal in the sense PR-1's guess_layout()
    already computes) — NOT surfaced in the returned contract; PR-1's
    guess_layout() remains the sole layout-inference path per the
    lockedFieldContract.
  - model_inputs / model_outputs are multiparm Folder parms; a
    2-input fixture confirmed .eval() == 2 with model_input_name1/2 and
    model_input_shape1<d>/2<d> populated independently and in order —
    multi-input instance-count semantics are the straightforward
    per-index enumeration this handler implements.
  - Scratch node + scratch net node.destroy() cleanup is SAFE mid-session
    — confirmed live: children(/obj) count identical before create and
    after destroy, for both the single-input and 2-input fixtures.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# sys.path bootstrap — 5 levels up from this file reaches the fork root;
# +/python adds the FastMCP-side fxhoudinimcp package (contract_from_
# setup_shapes lives there) so the import below resolves when this module
# is imported from hython. Mirrors usd_export_handlers.py exactly.
#
#  __file__: .../fxhoudinimcp/houdini/scripts/python/fxhoudinimcp_server/handlers/cop_onnx_handlers.py
#   1 up -> .../handlers/
#   2 up -> .../fxhoudinimcp_server/
#   3 up -> .../python/
#   4 up -> .../scripts/
#   5 up -> .../houdini/
#   6 up -> .../fxhoudinimcp/             (fork root)
#  +/python -> .../fxhoudinimcp/python/
# ---------------------------------------------------------------------------
import glob as _glob
import logging as _logging
import os as _os
import sys as _sys

_PY = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..", "python")
)
if _PY not in _sys.path:
    _sys.path.insert(0, _PY)

import hou  # noqa: E402  (hython / Houdini-side interpreter only)
from fxhoudinimcp_server.dispatcher import Capability, register_handler  # noqa: E402
from fxhoudinimcp.cop_onnx_model import contract_from_setup_shapes  # noqa: E402

_log = _logging.getLogger(__name__)

# Default filesystem roots scanned by cop_onnx_list_models when the
# caller passes roots=None. Houdini-expandable strings, resolved via
# hou.text.expandString at call time (asset-reference-discipline.md).
_DEFAULT_MODEL_ROOTS = ["$HIP/models", "$HOUDINI_USER_PREF_DIR/onnx"]

# Fixed 9-slot shape block exposed by cop/onnx's Setup Shapes from Model
# (Phase-0 probe, confirmed live).
_MAX_SHAPE_DIMS = 9

# Synthesized dtype used when cop/onnx exposes no dtype field (Phase-0
# probe finding: model_input_data{i}/model_output_data{i} never carries a
# dtype string — see module docstring).
_SYNTHESIZED_DTYPE = "float32"

# Scratch net name used for the transient copnet created (and ALWAYS
# destroyed, REV2 FOLD B2) for each cop_onnx_inspect_model scratch-node
# call. Not reused/persisted across calls -- see the module docstring.
_SCRATCH_NET_NAME = "_mcp_onnx_inspect"


###### Helpers

def _normalize_shape_dims(onnx_node: "hou.Node", prefix: str, index: int) -> list:
    """Read model_{prefix}_shape{index}{d} for d=1.._MAX_SHAPE_DIMS and
    return the populated (non-padding) dims, normalizing the -1 dynamic
    sentinel to the literal string "dynamic".

    Guards the trailing-zero-vs-padding heuristic (REV2 FOLD Minor):
    cop/onnx always exposes a FIXED 9-slot shape block (Phase-0 probe,
    confirmed live) -- unused trailing dims beyond the model's actual
    rank read back as the literal int 0. This function strips ONLY that
    fixed trailing pad: once a dim reads 0 AFTER at least one real dim
    has already been read, every remaining slot is padding and reading
    stops.

    A read that is ALL zeros (dims stays empty through the whole loop --
    i.e. the very first dim1 already reads 0) is NOT treated as "one
    real 0-sized dim" -- cop/onnx's Setup Shapes never legitimately
    reports a rank-0/scalar tensor this way, so an all-zero read is
    handled explicitly as an unsupported v1 case and normalizes to the
    empty shape ([]), matching what the padding-strip would produce
    anyway. This makes the "no real dim has a legitimate value of 0"
    assumption an explicit, documented decision rather than an
    accidental consequence of the `and dims` guard.
    """
    dims: list = []
    for d in range(1, _MAX_SHAPE_DIMS + 1):
        parm = onnx_node.parm(f"model_{prefix}_shape{index}{d}")
        if parm is None:
            break
        raw = parm.eval()
        if raw == 0 and dims:
            # Padding slot past the model's actual declared rank.
            break
        if raw == 0 and not dims:
            # Unsupported v1 case: a rank-0/scalar shape (or dim1 itself
            # padding-zero with nothing read yet). Treat as empty shape
            # rather than guessing whether this is a real 0-sized dim.
            break
        dims.append("dynamic" if raw == -1 else raw)
    return dims


def _read_raw_tensors(onnx_node: "hou.Node", prefix: str, count: int) -> list:
    """Assemble the raw dicts contract_from_setup_shapes expects, reading
    model_{prefix}_name{i}/model_{prefix}_shape{i}{d} for i in 1..count.

    dtype is synthesized (see module docstring — cop/onnx exposes no
    dtype field for either inputs or outputs).
    """
    raw: list = []
    for i in range(1, count + 1):
        name_parm = onnx_node.parm(f"model_{prefix}_name{i}")
        name = name_parm.eval() if name_parm is not None else f"{prefix}{i}"
        shape = _normalize_shape_dims(onnx_node, prefix, i)
        raw.append({"name": name, "dtype": _SYNTHESIZED_DTYPE, "shape": shape})
    return raw


###### cop_onnx_list_models

def cop_onnx_list_models(*, roots: "list | None" = None) -> dict:
    """Enumerate .onnx files under the given (or default) filesystem roots.

    Filesystem-metadata-only — no node is created, no model is parsed.
    A missing root is NOTED in missing_roots and never raises (FR-1).

    Returns::

        {
            "ok": True,
            "models": [{"path": str, "size": int, "mtime": float}, ...],
            "roots_scanned": [str, ...],
            "missing_roots": [str, ...],
        }

    or an FR-5 error shape on unexpected failure::

        {"ok": False, "error": "<reason>"}

    Args:
        roots: Optional list of Houdini-expandable root paths to scan for
            .onnx files. Defaults to cop_onnx.model_roots-style defaults
            (``$HIP/models``, ``$HOUDINI_USER_PREF_DIR/onnx``) when None.
    """
    try:
        candidate_roots = roots if roots is not None else list(_DEFAULT_MODEL_ROOTS)

        models: list = []
        roots_scanned: list = []
        missing_roots: list = []

        for raw_root in candidate_roots:
            expanded = hou.text.expandString(raw_root)
            if not _os.path.isdir(expanded):
                missing_roots.append(raw_root)
                continue
            roots_scanned.append(raw_root)
            pattern = _os.path.join(expanded, "**", "*.onnx")
            for match in _glob.glob(pattern, recursive=True):
                try:
                    stat = _os.stat(match)
                    models.append({
                        "path": match.replace("\\", "/"),
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                    })
                except OSError as exc:
                    _log.warning("cop_onnx_list_models: could not stat %r: %s", match, exc)

        return {
            "ok": True,
            "models": models,
            "roots_scanned": roots_scanned,
            "missing_roots": missing_roots,
        }
    except Exception as exc:
        _log.warning("cop_onnx_list_models failed for roots=%r: %s", roots, exc)
        return {"ok": False, "error": str(exc)}


register_handler("cop_onnx_list_models", cop_onnx_list_models, Capability.READONLY)


###### cop_onnx_inspect_model

def cop_onnx_inspect_model(
    *, model_path: "str | None" = None, node_path: "str | None" = None
) -> dict:
    """Read an .onnx model's input/output tensor contract.

    TWO mutually exclusive paths (REV2 FOLD B1 -- node_path is now
    strictly READ-ONLY):

      (A) node_path given -> READ-ONLY on the EXISTING node. This handler
          NEVER sets modelfile and NEVER presses setupshapes on a
          node_path node -- it only reads whatever model_inputs/
          model_outputs the node ALREADY has populated. If the node has
          no model configured yet (an empty modelfile parm -- live-probe
          correction: a fresh onnx node's model_inputs/model_outputs
          multiparms default to 1 each with empty-named placeholder
          entries, so multiparm COUNT is not a reliable "unconfigured"
          signal; an empty modelfile is), this returns an informative
          {ok: False, error} rather than mutating the node to populate
          it. When BOTH model_path and node_path are given, node_path
          WINS and model_path is ignored (this is intentional and
          documented here -- see module docstring B1). The returned
          contract's model_path field reports the NODE's OWN modelfile
          parm value, never the caller's (possibly bogus/unrelated)
          model_path argument (REV3 FOLD B1-metadata).

      (B) node_path absent, model_path given -> the SCRATCH-NODE
          mechanism: create a scratch copnet + a scratch cop/onnx node
          under /obj, set modelfile, press setupshapes, read back the
          populated parms, then GUARANTEE both the scratch node AND the
          scratch net are destroyed in a finally — even on exception,
          and even when the copnet was created but the onnx node
          createNode then failed (REV2 FOLD B2). Nothing persists in
          /obj after this path runs. This guaranteed cleanup is what
          keeps the tool READONLY/ungated.

    At least one of {model_path, node_path} is required.

    IMPORTANT (Phase-0 finding, path B only): this handler NEVER presses
    the "reload" button before "setupshapes" — doing so segfaults
    Houdini (COP_ONNXParms::buildFromOp, confirmed live). Setting
    modelfile then pressing setupshapes directly is safe and sufficient;
    Setup Shapes always (re)reads the file at modelfile.

    Returns the OnnxContract.to_dict() shape merged with ok=True on
    success::

        {
            "ok": True,
            "model_path": str,
            "inputs": [...], "outputs": [...],
            "opset": int | None, "producer": str | None,
            "loadable": bool, "error": str | None,
        }

    or an FR-2/FR-5 error shape on failure::

        {"ok": False, "error": "<reason>"}

    or, if a scratch cleanup failure is detected in the finally block
    (REV2 FOLD B3 -- fail-loud, never a bare ok=True on a broken
    READONLY guarantee; REV3 FOLD new-defect -- the two cases below are
    now distinguished rather than always claiming "read succeeded")::

        # (a) the inspection read itself SUCCEEDED, then cleanup FAILED:
        {
            "ok": False,
            "error": "read OK but scratch cleanup FAILED — orphaned <path>",
            "cleanup_failed": True,
            "orphaned_path": "<path>",
        }

        # (b) the inspection read had ALREADY FAILED, and cleanup ALSO
        # failed -- the ORIGINAL inspection error is preserved verbatim
        # as `error`, never overwritten by a misleading success string:
        {
            "ok": False,
            "error": "<the ORIGINAL inspection error>",
            "cleanup_failed": True,
            "orphaned_path": "<path>",
        }

    Args:
        model_path: Path to the .onnx file (Houdini-expandable). Optional
            when node_path is given; IGNORED when node_path is given
            (node_path wins — see path A above).
        node_path: Optional existing cop/onnx node path to read in-place
            (READ-ONLY, path A) instead of the scratch-node mechanism
            (path B). Must already be of type "onnx"; a mismatched type
            is an FR-2-style error.
    """
    # FR-2: require at least one of {model_path, node_path}.
    has_model_path = bool(model_path and model_path.strip())
    has_node_path = bool(node_path and node_path.strip())
    if not has_model_path and not has_node_path:
        return {
            "ok": False,
            "error": "at least one of model_path or node_path is required",
        }

    scratch_node = None
    scratch_net = None
    created_node = False
    created_net = False

    # REV3 FOLD (new-defect): tracks whether the try-block's inspection
    # read ALREADY failed, and if so, what the ORIGINAL failure message
    # was -- so the finally block's cleanup-failure branch can tell apart
    # "read OK, cleanup then failed" from "read already FAILED, cleanup
    # ALSO failed" and never overwrite a real inspection failure with a
    # misleading "inspection read succeeded" string. None == read
    # succeeded (so far); a non-None string is the original error.
    read_error_message: "str | None" = None

    try:
        if has_node_path:
            # -----------------------------------------------------------
            # Path A: READ-ONLY on the existing node (REV2 FOLD B1).
            # Never set modelfile, never press setupshapes here.
            # model_path is ignored when node_path is given (documented).
            # -----------------------------------------------------------
            target_node = hou.node(node_path)
            if target_node is None:
                read_error_message = f"Node not found: {node_path}"
                return {"ok": False, "error": read_error_message}
            if target_node.type().name() != "onnx":
                read_error_message = (
                    f"Node at {node_path!r} is type "
                    f"{target_node.type().name()!r}, expected 'onnx'"
                )
                return {"ok": False, "error": read_error_message}
            # NOTE (live-probe correction, REV2 FOLD B1): a freshly
            # created 'onnx' node ships with model_inputs==1 /
            # model_outputs==1 as DEFAULT multiparm entries (empty-named
            # placeholders) — the multiparm count alone does NOT signal
            # "no model configured". The reliable signal is an empty
            # modelfile parm: cop/onnx only populates real tensor names/
            # shapes into the multiparm entries after Setup Shapes has
            # been run against a non-empty modelfile.
            modelfile_parm = target_node.parm("modelfile")
            modelfile_value = modelfile_parm.eval() if modelfile_parm is not None else ""
            if not modelfile_value or not modelfile_value.strip():
                read_error_message = (
                    f"Node at {node_path!r} has no model configured "
                    f"(modelfile is empty) — this is a READ-ONLY "
                    f"inspection and will not mutate the node to "
                    f"populate it. Run Setup Shapes on the node "
                    f"yourself, or pass model_path with no node_path "
                    f"to use the scratch-node mechanism."
                )
                return {"ok": False, "error": read_error_message}
            n_inputs_parm = target_node.parm("model_inputs")
            n_outputs_parm = target_node.parm("model_outputs")
            n_inputs = n_inputs_parm.eval() if n_inputs_parm is not None else 0
            n_outputs = n_outputs_parm.eval() if n_outputs_parm is not None else 0
            onnx_node = target_node
            # REV3 FOLD B1-metadata: report the NODE's actual configured
            # model (its own modelfile parm), NOT the caller's model_path
            # arg -- model_path is ignored/ignorable in node_path mode
            # (see REV2 FOLD B1 above), so the returned contract must
            # reflect what the node is ACTUALLY inspecting, not whatever
            # (possibly bogus/unrelated) model_path the caller passed
            # alongside node_path.
            resolved_model_path = modelfile_value

        else:
            # -----------------------------------------------------------
            # Path B: the scratch-node mechanism. Both the scratch node
            # AND the scratch net are always created and destroyed
            # together (REV2 FOLD B2 — no persistent-net-reuse).
            # -----------------------------------------------------------
            expanded_path = hou.text.expandString(model_path)
            obj = hou.node("/obj")

            scratch_net = obj.createNode("copnet", _SCRATCH_NET_NAME)
            created_net = True  # set immediately after createNode succeeds

            scratch_node = scratch_net.createNode("onnx")
            created_node = True
            onnx_node = scratch_node

            with hou.undos.group("cop_onnx_inspect_model"):
                onnx_node.parm("modelfile").set(expanded_path)
                # NOTE: do NOT press "reload" here -- see module docstring
                # (segfault, confirmed live). "setupshapes" alone (re)reads
                # the file at modelfile.
                onnx_node.parm("setupshapes").pressButton()

            n_inputs = onnx_node.parm("model_inputs").eval()
            n_outputs = onnx_node.parm("model_outputs").eval()
            resolved_model_path = model_path

        raw_inputs = _read_raw_tensors(onnx_node, "input", n_inputs)
        raw_outputs = _read_raw_tensors(onnx_node, "output", n_outputs)

        contract = contract_from_setup_shapes(
            model_path=resolved_model_path,
            raw_inputs=raw_inputs,
            raw_outputs=raw_outputs,
        )
        return {"ok": True, **contract.to_dict()}

    except Exception as exc:
        _log.warning(
            "cop_onnx_inspect_model failed for model_path=%r node_path=%r: %s",
            model_path, node_path, exc,
        )
        read_error_message = str(exc)
        return {"ok": False, "error": read_error_message}

    finally:
        # GUARANTEED cleanup on path B — this is what keeps the tool
        # READONLY. Destroy the scratch node first (child of the net),
        # then the scratch net, on EVERY path — including a copnet-
        # created-but-onnx-createNode-then-failed path (REV2 FOLD B2).
        cleanup_error: "Exception | None" = None
        orphaned_path = None

        if created_node and scratch_node is not None:
            try:
                scratch_node.destroy()
            except Exception as exc:
                cleanup_error = exc
                orphaned_path = scratch_node.path()
                _log.warning(
                    "cop_onnx_inspect_model: failed to destroy scratch node %r: %s",
                    scratch_node, exc,
                )

        if created_net and scratch_net is not None:
            try:
                scratch_net.destroy()
            except Exception as exc:
                cleanup_error = cleanup_error or exc
                orphaned_path = orphaned_path or scratch_net.path()
                _log.warning(
                    "cop_onnx_inspect_model: failed to destroy scratch net %r: %s",
                    scratch_net, exc,
                )

        if cleanup_error is not None:
            # REV2 FOLD B3: fail-loud on a broken READONLY guarantee —
            # never a bare ok=True when scratch cleanup failed, even if
            # the inspection read itself succeeded. This return REPLACES
            # any return already produced in the try block (Python
            # executes a finally-block return in preference to a
            # pending try-block return).
            #
            # REV3 FOLD (new-defect): distinguish the two cases rather
            # than always claiming "inspection read succeeded" --
            #   (a) read OK + cleanup failed -> the cleanup-failed message
            #       below (accurate: the read really did succeed).
            #   (b) read FAILED + cleanup ALSO failed -> preserve the
            #       ORIGINAL inspection error as the top-level `error`
            #       (never overwrite a real failure with a misleading
            #       success string); cleanup_failed + orphaned_path are
            #       still surfaced so the caller/operator knows about
            #       BOTH problems.
            if read_error_message is None:
                return {
                    "ok": False,
                    "error": (
                        "read OK but scratch cleanup FAILED — "
                        f"orphaned {orphaned_path}"
                    ),
                    "cleanup_failed": True,
                    "orphaned_path": orphaned_path,
                }
            return {
                "ok": False,
                "error": read_error_message,
                "cleanup_failed": True,
                "orphaned_path": orphaned_path,
            }


register_handler("cop_onnx_inspect_model", cop_onnx_inspect_model, Capability.READONLY)
