"""Materials & Shaders handlers for FXHoudini-MCP.

Provides tools for listing, inspecting, creating, and assigning
materials and shader networks within Houdini.
"""

from __future__ import annotations

# Built-in
from typing import Any

# Third-party
import hou

# Internal
from fxhoudinimcp_server.config import layout_if_enabled
from fxhoudinimcp_server.dispatcher import Capability, register_handler


###### Helpers

def _get_node(node_path: str) -> hou.Node:
    """Resolve a node path and raise a clear error if it does not exist."""
    node = hou.node(node_path)
    if node is None:
        raise ValueError(f"Node not found: {node_path}")
    return node


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


def _material_summary(node: hou.Node) -> dict[str, Any]:
    """Return a compact summary dict for a material node."""
    return {
        "path": node.path(),
        "type": node.type().name(),
        "label": node.type().description(),
        "param_count": len(node.parms()),
    }


###### materials.list_materials

def _list_materials(*, root_path: str = "/mat", **_: Any) -> dict[str, Any]:
    """List all material nodes under the given root path.

    Walks children of the specified root (typically /mat) and also
    checks /stage if it exists, collecting material node summaries.

    Args:
        root_path: Root path to search for materials (default: "/mat").
    """
    materials: list[dict[str, Any]] = []
    search_paths = [root_path]

    # Also search /stage if it exists and is not already the root
    if root_path != "/stage" and hou.node("/stage") is not None:
        search_paths.append("/stage")

    for search_path in search_paths:
        root = hou.node(search_path)
        if root is None:
            continue

        for node in root.allSubChildren():
            type_name = node.type().name()
            category = node.type().category().name()

            # Match material-like node types
            if category == "Vop" or "material" in type_name.lower() or \
               "shader" in type_name.lower() or \
               type_name in ("principledshader::2.0", "principledshader",
                             "mtlxstandard_surface", "materialbuilder"):
                materials.append(_material_summary(node))

    return {
        "count": len(materials),
        "materials": materials,
    }

register_handler("materials.list_materials", _list_materials, Capability.READONLY)


###### materials.get_material_info

def _get_material_info(*, node_path: str, **_: Any) -> dict[str, Any]:
    """Get detailed information about a material node.

    Returns the material's type, all non-default parameters, shader VOP
    nodes inside (if it's a material builder), and geometry nodes that
    reference this material.

    Args:
        node_path: Absolute path to the material node.
    """
    node = _get_node(node_path)

    # Gather non-default parameters
    params: dict[str, Any] = {}
    for parm in node.parms():
        try:
            val = parm.eval()
            default = parm.parmTemplate().defaultValue()
            if isinstance(default, tuple) and len(default) == 1:
                default = default[0]
            if val != default:
                params[parm.name()] = val
        except Exception:
            pass

    # List shader VOP nodes inside (if this is a material builder)
    shaders: list[dict[str, str]] = []
    try:
        for child in node.children():
            if child.type().category().name() == "Vop":
                shaders.append({
                    "name": child.name(),
                    "path": child.path(),
                    "type": child.type().name(),
                })
    except Exception:
        pass

    # Find geometry nodes that reference this material
    assignments: list[str] = []
    mat_path = node.path()
    try:
        root = hou.node("/obj")
        if root is not None:
            for child in root.allSubChildren():
                for parm in child.parms():
                    try:
                        val = parm.eval()
                        if isinstance(val, str) and mat_path in val:
                            assignments.append(child.path())
                            break
                    except Exception:
                        continue
    except Exception:
        pass

    return {
        "path": node.path(),
        "type": node.type().name(),
        "params": params,
        "shaders": shaders,
        "assignments": assignments,
    }

register_handler("materials.get_material_info", _get_material_info, Capability.READONLY)


###### materials.create_material_network

def _create_material_network(
    *,
    name: str,
    shader_type: str = "principled",
    params: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Create a new material network in /mat.

    Creates a material node of the specified shader type and optionally
    sets parameter values from the provided dict.

    Args:
        name: Name for the new material node.
        shader_type: Type of shader to create. "principled" creates a
            principledshader::2.0; "materialx" creates mtlxstandard_surface.
        params: Optional dict of parameter name -> value to set.
    """
    mat_context = hou.node("/mat")
    if mat_context is None:
        # Create /mat if it doesn't exist
        mat_context = hou.node("/").createNode("matnet", "mat")

    # Map shader_type to actual Houdini node type
    type_map = {
        "principled": "principledshader::2.0",
        "materialx": "mtlxstandard_surface",
    }
    actual_type = type_map.get(shader_type, shader_type)

    try:
        node = mat_context.createNode(actual_type, node_name=name)
    except hou.OperationFailed as e:
        raise ValueError(
            f"Failed to create material of type '{actual_type}' in /mat: {e}"
        )

    # Set parameters if provided
    if params:
        for parm_name, parm_value in params.items():
            parm = node.parm(parm_name)
            if parm is not None:
                try:
                    parm.set(parm_value)
                except Exception:
                    pass

    node.moveToGoodPosition()
    _focus_network_editor(node)

    return {
        "material_path": node.path(),
        "shader_type": actual_type,
    }

register_handler("materials.create_material_network", _create_material_network)


###### materials.create_mtlx_material

def _create_mtlx_material(
    *,
    name: str,
    parent_path: str = "/stage",
    base_color: list[float] | None = None,
    metalness: float | None = None,
    roughness: float | None = None,
    textures: dict[str, str] | None = None,
    normal_map: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Create a COMPLETE Solaris MaterialX material and publish it as a USD Material prim.

    Unlike ``create_material_network`` (which makes a bare VOP node in ``/mat``), this builds a
    ``materiallibrary`` LOP containing an ``mtlxstandard_surface`` (+ optional textures and a
    normal map). The ``materiallibrary`` AUTO-PUBLISHES that surface as a USD ``Material`` prim at
    ``<matpathprefix>/<name>`` on the stage (default ``/materials/<name>``) — a ready-to-bind
    material in one call. Live-confirmed publishing mechanism (Houdini 21.0.729): no extra
    surface-output wiring is required; the materiallibrary collects the mtlx surface by name.

    Args:
        name: material name; the published ``Material`` prim is ``<matpathprefix>/<name>``.
        parent_path: LOP parent to build the ``materiallibrary`` under (default ``/stage``).
        base_color: ``[r, g, b]`` diffuse color (mtlxstandard_surface ``base_color``).
        metalness: 0..1 metalness.
        roughness: 0..1 (sets ``specular_roughness``).
        textures: ``{surface_input_name: file_path}`` — one ``mtlxUsdUVTexture`` per entry, wired
            to the named surface input (e.g. ``{"base_color": "$HIP/tex/albedo.<UDIM>.exr"}``).
        normal_map: a file path → ``mtlxUsdUVTexture`` → ``mtlxnormalmap`` → the surface ``normal``.

    Returns:
        ``{materiallibrary_path, surface_path, material_prim_path}``.
    """
    parent = _get_node(parent_path)

    # The materiallibrary LOP is the publish container; the mtlx surface created inside it is
    # auto-collected into a Material prim (Phase-0 hython-probed, 21.0.729).
    try:
        matlib = parent.createNode("materiallibrary", node_name=f"{name}_matlib")
        surf = matlib.createNode("mtlxstandard_surface", node_name=name)
    except hou.OperationFailed as e:
        raise ValueError(f"Failed to build materiallibrary/mtlx surface under {parent_path}: {e}")

    # Base surface parms — only those explicitly provided. FAIL-LOUD if an explicitly
    # requested parm is missing (a version mismatch / renamed parm): a silent no-op would
    # report success while ignoring the caller's material value (fail-loud-discipline).
    if base_color is not None:
        pt = surf.parmTuple("base_color")
        if pt is None:
            raise ValueError("mtlxstandard_surface has no 'base_color' parm tuple "
                             "(version mismatch?) — cannot honor the requested base_color")
        pt.set(base_color)
    if metalness is not None:
        p = surf.parm("metalness")
        if p is None:
            raise ValueError("mtlxstandard_surface has no 'metalness' parm (version mismatch?)")
        p.set(metalness)
    if roughness is not None:
        p = surf.parm("specular_roughness")
        if p is None:
            raise ValueError("mtlxstandard_surface has no 'specular_roughness' parm "
                             "(version mismatch?) — cannot honor the requested roughness")
        p.set(roughness)

    # Optional textures: one mtlxUsdUVTexture per surface input, wired by name (setNamedInput
    # raises on an unknown input name — fail-loud). A missing `file` parm also fails loud: a
    # wired texture node with no file path is a silent no-op that would report success.
    for input_name, file_path in (textures or {}).items():
        tex = matlib.createNode("mtlxUsdUVTexture", node_name=f"{name}_{input_name}")
        fp = tex.parm("file")
        if fp is None:
            raise ValueError(f"mtlxUsdUVTexture has no 'file' parm (version mismatch?) — "
                             f"cannot set the texture for input {input_name!r}")
        fp.set(file_path)
        surf.setNamedInput(input_name, tex, 0)

    # Optional normal map: texture -> mtlxnormalmap (input 0) -> surface 'normal'.
    if normal_map:
        ntex = matlib.createNode("mtlxUsdUVTexture", node_name=f"{name}_normal_tex")
        fp = ntex.parm("file")
        if fp is None:
            raise ValueError("mtlxUsdUVTexture has no 'file' parm (version mismatch?) — "
                             "cannot set the normal-map texture")
        fp.set(normal_map)
        nmap = matlib.createNode("mtlxnormalmap", node_name=f"{name}_normalmap")
        nmap.setInput(0, ntex)
        surf.setNamedInput("normal", nmap, 0)

    matlib.layoutChildren()
    matlib.setDisplayFlag(True)
    _focus_network_editor(matlib)

    # Material prim path = matpathprefix + surface node name (live-confirmed: /materials/<name>).
    prefix = "/materials/"
    mpp = matlib.parm("matpathprefix")
    if mpp is not None and mpp.eval():
        prefix = mpp.eval()
    material_prim_path = prefix.rstrip("/") + "/" + name

    return {
        "materiallibrary_path": matlib.path(),
        "surface_path": surf.path(),
        "material_prim_path": material_prim_path,
    }

register_handler("materials.create_mtlx_material", _create_mtlx_material)


###### materials.assign_material

def _assign_material(
    *,
    geo_path: str,
    material_path: str,
    **_: Any,
) -> dict[str, Any]:
    """Assign a material to a geometry node.

    Finds the SOP network inside the geo node, creates a material SOP
    after the last displayed node, sets the material path, and sets
    the display flag on the new material SOP.

    Args:
        geo_path: Path to the geometry (Object-level) node.
        material_path: Path to the material node to assign.
    """
    geo_node = _get_node(geo_path)

    # Find the SOP network inside the geo node
    # Look for the display node (the one with the display flag set)
    display_node = None
    sop_children = geo_node.children()

    for child in sop_children:
        try:
            if child.isDisplayFlagSet():
                display_node = child
                break
        except Exception:
            continue

    if display_node is None and sop_children:
        display_node = sop_children[-1]

    # Create a material SOP
    mat_sop = geo_node.createNode("material", node_name="assign_material")

    # Connect to the display node
    if display_node is not None:
        mat_sop.setInput(0, display_node)

    # Set the material path
    mat_parm = mat_sop.parm("shop_materialpath1")
    if mat_parm is not None:
        mat_parm.set(material_path)

    # Set display flag on the new material SOP
    mat_sop.setDisplayFlag(True)
    mat_sop.setRenderFlag(True)
    mat_sop.moveToGoodPosition()
    _focus_network_editor(mat_sop)

    return {
        "material_sop_path": mat_sop.path(),
    }

register_handler("materials.assign_material", _assign_material)


###### materials.list_material_types

def _list_material_types(
    *,
    filter: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """List available VOP/material node types.

    Inspects the Vop and Shop node type categories to find available
    material and shader types. Optionally filters by a substring.

    Args:
        filter: Optional substring to filter type names by.
    """
    results: list[dict[str, str]] = []

    categories_to_search = ["Vop", "Shop"]
    all_categories = hou.nodeTypeCategories()

    for cat_name in categories_to_search:
        category = all_categories.get(cat_name)
        if category is None:
            continue

        types_dict = category.nodeTypes()
        for type_name, node_type in sorted(types_dict.items()):
            # Skip hidden types
            try:
                if node_type.hidden():
                    continue
            except Exception:
                pass

            label = node_type.description()

            # Apply filter if provided
            if filter is not None:
                filter_lower = filter.lower()
                if filter_lower not in type_name.lower() and \
                   filter_lower not in label.lower():
                    continue

            results.append({
                "name": type_name,
                "label": label,
                "category": cat_name,
            })

    return {
        "count": len(results),
        "types": results,
    }

register_handler("materials.list_material_types", _list_material_types, Capability.READONLY)
