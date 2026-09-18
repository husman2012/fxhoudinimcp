"""VEX handlers for FXHoudini-MCP.

Provides tools for creating, reading, and validating VEX code
in Attribute Wrangle nodes and VEX expressions.
"""

from __future__ import annotations

# Built-in
import re

# Third-party
import hou

# Internal
from fxhoudinimcp_server.config import layout_if_enabled
from fxhoudinimcp_server.dispatcher import Capability, register_handler


###### Helpers

def _get_node(node_path: str) -> hou.Node:
    """Return a node or raise if not found."""
    node = hou.node(node_path)
    if node is None:
        raise ValueError(f"Node not found: {node_path}")
    return node


_VEX_CONTEXT_MARKERS = ("vex", "vop", "snippet", "syntax error", "undefined function")
_VEX_FAILURE_MARKERS = (
    "error", "undefined function", "unable to load shader", "failed to resolve",
)
_VEX_COMPILE_HEADER = "errors or warnings encountered during vex compile"


def _is_vex_compile_error(message: str) -> bool:
    """True if a node message reports a VEX compile FAILURE.

    DOP wrangles (Gas Field Wrangle, POP Wrangle) report compile failures
    as node *warnings* ("Error in VOP 'snippet1'.", "... Syntax error ...",
    "Call to undefined function ..."), so a check that only reads
    node.errors() calls broken code valid. Compile *warnings* such as
    "Implicit cast from float to int" are not failures.
    """
    low = str(message).lower()
    if not any(marker in low for marker in _VEX_CONTEXT_MARKERS):
        return False
    body = low.replace(_VEX_COMPILE_HEADER, "")
    return any(marker in body for marker in _VEX_FAILURE_MARKERS)


def _split_messages(errors, warnings) -> tuple[list[str], list[str]]:
    """Promote VEX compile failures reported as warnings to errors."""
    out_errors: list[str] = []
    out_warnings: list[str] = []
    for message in list(errors or []):
        if message not in out_errors:
            out_errors.append(message)
    for message in list(warnings or []):
        target = out_errors if _is_vex_compile_error(message) else out_warnings
        if message not in target:
            target.append(message)
    return out_errors, out_warnings


# DOP VEX node families that compile only when a solver runs them. Each maps
# to (object type, solver type, solver input for the wrangle) of a tiny
# scratch simulation that makes the solver compile a COPY of the node.
_DOP_SCRATCH_SIMS = (
    ("gas", ("smokeobject", "multisolver", 1)),
    ("pop", ("popobject", "popsolver", 1)),
)


def _dop_scratch_spec(node: hou.Node):
    type_name = node.type().name()
    for prefix, spec in _DOP_SCRATCH_SIMS:
        if type_name.startswith(prefix):
            return spec
    return None


def _compile_dop_vex(node: hou.Node) -> tuple[list[str], list[str]]:
    """Compile a DOP wrangle's VEX in an isolated scratch simulation.

    A DOP wrangle's VEX is compiled only when a solver runs it, so cooking
    the node itself proves nothing (broken code reported "valid"). A copy
    of the node (same parms) is attached to a tiny object in a throwaway
    DOP network under /obj and cooked over its first two frames with
    cook(frame_range=...), which runs one solve without moving the global
    frame or touching the user's simulation. The scratch network is
    always destroyed.
    """
    object_type, solver_type, solver_input = _dop_scratch_spec(node)
    scratch = None
    try:
        with hou.undos.disabler():
            scratch = hou.node("/obj").createNode("dopnet", "__fxh_vex_check")
            if scratch.parm("cacheenabled") is not None:
                scratch.parm("cacheenabled").set(0)
            obj = scratch.createNode(object_type)
            if object_type == "smokeobject":
                obj.parm("divsize").set(0.25)
                obj.parmTuple("size").set((1, 1, 1))
            copy = hou.copyNodesTo([node], scratch)[0]
            solver = scratch.createNode(solver_type)
            solver.setInput(0, obj)
            solver.setInput(solver_input, copy)
            out = scratch.createNode("output")
            out.setInput(0, solver)
            out.setDisplayFlag(True)
            start_parm = scratch.parm("startframe")
            start = int(start_parm.eval()) if start_parm is not None else 1
            try:
                out.cook(force=True, frame_range=(start, start + 1))
            except hou.OperationFailed:
                pass  # compile failures are read from the copy below
            return list(copy.errors()), list(copy.warnings())
    finally:
        if scratch is not None:
            with hou.undos.disabler():
                scratch.destroy()


def _compile_report(node: hou.Node) -> dict:
    """Compile *node*'s VEX and report errors/warnings.

    SOP-style nodes compile on cook. DOP wrangles are compiled in a
    scratch simulation (see _compile_dop_vex); DOP VEX nodes of other
    families are reported as not compile-checked instead of valid.
    """
    is_dop = node.type().category().name() == "Dop"
    method = "cook"
    compile_checked = True
    errors: list[str] = []
    warnings: list[str] = []
    if is_dop and node.parm("snippet") is not None:
        if _dop_scratch_spec(node) is not None:
            method = "scratch_dop_solve"
            errors, warnings = _compile_dop_vex(node)
        else:
            compile_checked = False
            method = "none"
    # Cooking the node compiles SOP-style VEX and clears messages left on a
    # DOP node by an earlier solve of code that has since been replaced.
    try:
        node.cook(force=True)
    except hou.OperationFailed:
        pass
    # Messages on the node itself (for DOP nodes: from the user's own sim).
    try:
        errors += list(node.errors() or [])
    except Exception:
        pass
    try:
        warnings += list(node.warnings() or [])
    except Exception:
        pass
    errors, warnings = _split_messages(errors, warnings)
    report = {
        "errors": errors,
        "warnings": warnings,
        "compile_checked": compile_checked,
        "compile_method": method,
    }
    if not compile_checked:
        report["note"] = (
            f"VEX was NOT compiled: {node.type().name()} compiles only while "
            "its simulation steps. Step the simulation, then call "
            "validate_vex or verify_network again."
        )
    return report


def _validate_vex_quick(node: hou.Node) -> dict:
    """Compile a wrangle node and return any VEX errors/warnings."""
    report = _compile_report(node)
    result = {
        "vex_valid": len(report["errors"]) == 0,
        "vex_errors": report["errors"],
        "vex_warnings": report["warnings"],
        "compile_checked": report["compile_checked"],
    }
    if "note" in report:
        result["vex_note"] = report["note"]
    return result


def _resolve_class_value(node: hou.Node, run_over: str) -> int:
    """Resolve a run_over string to the correct menu index on this node.

    Reads the ``class`` parameter's menu labels dynamically so the mapping
    is always correct regardless of Houdini version.
    """
    class_parm = node.parm("class")
    if class_parm is None:
        raise ValueError(
            f"Node {node.path()} has no 'class' parameter — "
            "is it an Attribute Wrangle?"
        )

    template = class_parm.parmTemplate()
    labels = list(template.menuLabels())
    items = list(template.menuItems())

    # Try exact match first (case-insensitive), then substring match
    target = run_over.strip().lower()
    for idx, label in enumerate(labels):
        if label.lower() == target:
            return int(items[idx]) if items[idx].isdigit() else idx
    for idx, label in enumerate(labels):
        if target in label.lower():
            return int(items[idx]) if items[idx].isdigit() else idx

    raise ValueError(
        f"Invalid run_over value '{run_over}'. "
        f"Available options: {labels}"
    )


def _reverse_class_label(node: hou.Node) -> str | None:
    """Return the human-readable label for the current class value."""
    class_parm = node.parm("class")
    if class_parm is None:
        return None
    value = class_parm.eval()
    template = class_parm.parmTemplate()
    labels = list(template.menuLabels())
    items = list(template.menuItems())
    for idx, item in enumerate(items):
        if (item.isdigit() and int(item) == value) or idx == value:
            return labels[idx] if idx < len(labels) else str(value)
    return str(value)


def _focus_network_editor(node: hou.Node) -> None:
    """Best-effort: layout the parent network, then pan the editor to *node*."""
    try:
        parent = node.parent()
        if parent is not None:
            layout_if_enabled(parent)
        for pane_tab in hou.ui.paneTabs():
            if pane_tab.type() == hou.paneTabType.NetworkEditor:
                if parent is not None:
                    pane_tab.cd(parent.path())
                pane_tab.setCurrentNode(node)
                pane_tab.homeToSelection()
                return
    except Exception:
        pass


# Regex pattern for detecting absolute channel paths in VEX code
_RE_ABS_CH = re.compile(r'ch[sfiv]?\s*\(\s*["\']/')


def _check_channel_paths(vex_code: str) -> list[str]:
    """Return warnings if the VEX code contains absolute channel refs."""
    warnings = []
    if _RE_ABS_CH.search(vex_code):
        warnings.append(
            "VEX contains absolute channel path (ch(\"/...\")). "
            "Prefer relative paths — use ../parm_name to reach the "
            "immediate parent, ../../parm_name for two levels up, etc. "
            "Absolute paths break when nodes are renamed or moved."
        )
    return warnings


###### vex.create_wrangle

def create_wrangle(
    parent_path: str,
    vex_code: str,
    run_over: str = "Points",
    name: str = None,
) -> dict:
    """Create an Attribute Wrangle node with VEX code.

    Args:
        parent_path: Path to the parent SOP network.
        vex_code: The VEX snippet code to set.
        run_over: What to run the wrangle over:
                  "Points", "Vertices", "Primitives", "Detail", or "Numbers".
        name: Optional explicit name for the node.
    """
    parent = hou.node(parent_path)
    if parent is None:
        raise ValueError(f"Parent node not found: {parent_path}")

    # Create the attribwrangle node
    try:
        if name:
            node = parent.createNode("attribwrangle", name)
        else:
            node = parent.createNode("attribwrangle")
    except hou.OperationFailed as e:
        raise ValueError(f"Failed to create attribwrangle node: {e}")

    # Set the VEX snippet
    snippet_parm = node.parm("snippet")
    if snippet_parm is None:
        raise ValueError(
            f"Created node {node.path()} does not have a 'snippet' parameter."
        )
    snippet_parm.set(vex_code)

    # Set the run_over class (resolved dynamically from menu labels)
    class_value = _resolve_class_value(node, run_over)
    node.parm("class").set(class_value)

    _focus_network_editor(node)

    # Build relative prefix map so the AI knows what each ../ level reaches.
    ancestors = {}
    current = node.parent()
    level = 1
    while current is not None:
        dots = "/".join([".."] * level)
        ancestors[dots] = current.path()
        current = current.parent()
        level += 1

    result = {
        "success": True,
        "node_path": node.path(),
        "node_name": node.name(),
        "run_over": run_over,
        "vex_code": vex_code,
        "channel_prefix": "../",
        "channel_ancestors": ancestors,
        "channel_hint": (
            "Use relative paths for channel references. "
            "See channel_ancestors to find the correct ../ depth "
            "for the node whose parameters you want to reference."
        ),
    }
    result.update(_validate_vex_quick(node))

    path_warnings = _check_channel_paths(vex_code)
    if path_warnings:
        result.setdefault("vex_warnings", []).extend(path_warnings)

    return result


###### vex.set_wrangle_code

def set_wrangle_code(node_path: str, vex_code: str) -> dict:
    """Set VEX code on an existing Attribute Wrangle node.

    Args:
        node_path: Path to the wrangle node.
        vex_code: The VEX snippet code to set.
    """
    node = _get_node(node_path)

    snippet_parm = node.parm("snippet")
    if snippet_parm is None:
        raise ValueError(
            f"Node {node_path} does not have a 'snippet' parameter. "
            "Is it an Attribute Wrangle?"
        )

    snippet_parm.set(vex_code)

    result = {
        "success": True,
        "node_path": node.path(),
        "vex_code": vex_code,
        "channel_prefix": "../",
    }
    result.update(_validate_vex_quick(node))

    path_warnings = _check_channel_paths(vex_code)
    if path_warnings:
        result.setdefault("vex_warnings", []).extend(path_warnings)

    return result


###### vex.get_wrangle_code

def get_wrangle_code(node_path: str) -> dict:
    """Read the VEX code from an Attribute Wrangle node.

    Args:
        node_path: Path to the wrangle node.
    """
    node = _get_node(node_path)

    snippet_parm = node.parm("snippet")
    if snippet_parm is None:
        raise ValueError(
            f"Node {node_path} does not have a 'snippet' parameter. "
            "Is it an Attribute Wrangle?"
        )

    vex_code = snippet_parm.eval()

    # Also get the run_over class (resolved dynamically from menu labels)
    run_over = _reverse_class_label(node)

    return {
        "node_path": node.path(),
        "vex_code": vex_code,
        "run_over": run_over,
    }


###### vex.create_vex_expression

def create_vex_expression(
    node_path: str,
    parm_name: str,
    vex_code: str,
) -> dict:
    """Set a VEX expression on a parameter.

    This sets the parameter's expression language to VEX and assigns
    the given expression code.

    Args:
        node_path: Path to the node.
        parm_name: Name of the parameter.
        vex_code: The VEX expression code.
    """
    node = _get_node(node_path)

    parm = node.parm(parm_name)
    if parm is None:
        raise ValueError(
            f"Parameter '{parm_name}' not found on node {node_path}."
        )

    try:
        parm.setExpression(vex_code, language=hou.exprLanguage.Hscript)
    except Exception:
        # If Hscript doesn't work, try setting as a Python expression
        try:
            parm.setExpression(vex_code, language=hou.exprLanguage.Python)
        except Exception as e:
            raise ValueError(
                f"Failed to set expression on {node_path}/{parm_name}: {e}"
            )

    return {
        "success": True,
        "node_path": node.path(),
        "parm_name": parm_name,
        "vex_code": vex_code,
    }


###### vex.validate_vex

def validate_vex(node_path: str) -> dict:
    """Validate VEX code by compiling the node and checking for errors.

    SOP wrangles compile when cooked. DOP wrangles (Gas Field Wrangle,
    POP Wrangle) compile only while a solver runs them, so their VEX is
    compiled in an isolated scratch simulation (the user's simulation and
    the global frame are not touched). VEX compile failures that Houdini
    reports as node warnings are returned as errors.

    Args:
        node_path: Path to the wrangle node to validate.
    """
    node = _get_node(node_path)

    # Read the current VEX code for reference
    vex_code = None
    snippet_parm = node.parm("snippet")
    if snippet_parm is not None:
        vex_code = snippet_parm.eval()

    report = _compile_report(node)
    errors = report["errors"]
    warnings = report["warnings"]
    is_valid = len(errors) == 0

    result = {
        "node_path": node.path(),
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "compile_checked": report["compile_checked"],
        "compile_method": report["compile_method"],
    }

    if vex_code is not None:
        result["vex_code"] = vex_code

    if not report["compile_checked"]:
        result["message"] = report["note"]
    elif is_valid:
        result["message"] = "VEX code is valid."
    else:
        result["message"] = f"VEX code has {len(errors)} error(s)."

    return result


###### Registration

register_handler("vex.create_wrangle", create_wrangle, Capability.CODE_EXEC)          # injects VEX snippet that executes on cook
register_handler("vex.set_wrangle_code", set_wrangle_code, Capability.CODE_EXEC)      # injects VEX snippet that executes on cook
register_handler("vex.get_wrangle_code", get_wrangle_code, Capability.READONLY)       # reads snippet parm; no execution
register_handler("vex.create_vex_expression", create_vex_expression, Capability.CODE_EXEC)  # setExpression → executes on every parm eval
register_handler("vex.validate_vex", validate_vex, Capability.CODE_EXEC)              # cook(force=True) executes injected VEX (ADR §3.3.2)
