"""MCP tool wrappers for Houdini materials and shaders operations.

Each tool delegates to the corresponding handler running inside Houdini
via the HTTP bridge.
"""

from __future__ import annotations

# Built-in
from typing import Any, Optional

# Third-party
from mcp.server.fastmcp import Context

# Internal
from fxhoudinimcp.server import mcp, _get_bridge


@mcp.tool()
async def list_materials(
    ctx: Context,
    root_path: str = "/mat",
) -> dict:
    """List all material nodes under a root path.

    Args:
        ctx: MCP context.
        root_path: Root path to search for materials.
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute(
        "materials.list_materials",
        {
            "root_path": root_path,
        },
    )


@mcp.tool()
async def get_material_info(ctx: Context, node_path: str) -> dict:
    """Get detailed information about a material node.

    Args:
        ctx: MCP context.
        node_path: Absolute path to the material node.
    """
    bridge = _get_bridge(ctx)
    return await bridge.execute(
        "materials.get_material_info",
        {
            "node_path": node_path,
        },
    )


@mcp.tool()
async def create_material_network(
    ctx: Context,
    name: str,
    shader_type: str = "principled",
    params: Optional[dict[str, Any]] = None,
) -> dict:
    """Create a new material network in /mat.

    Args:
        ctx: MCP context.
        name: Name for the new material node.
        shader_type: Shader type name ("principled", "materialx", etc.).
        params: Parameter name-value pairs to set on the shader.
    """
    bridge = _get_bridge(ctx)
    p: dict[str, Any] = {
        "name": name,
        "shader_type": shader_type,
    }
    if params is not None:
        p["params"] = params
    return await bridge.execute("materials.create_material_network", p)


@mcp.tool()
async def create_mtlx_material(
    ctx: Context,
    name: str,
    parent_path: str = "/stage",
    base_color: Optional[list[float]] = None,
    metalness: Optional[float] = None,
    roughness: Optional[float] = None,
    textures: Optional[dict[str, str]] = None,
    normal_map: Optional[str] = None,
) -> dict:
    """Create a COMPLETE Solaris MaterialX material in one call.

    Builds a ``materiallibrary`` LOP containing an ``mtlxstandard_surface`` (+ optional textures
    and a normal map), which the materiallibrary auto-publishes as a USD ``Material`` prim at
    ``<matpathprefix>/<name>`` (default ``/materials/<name>``) — ready to bind with
    ``assign_material``. Unlike ``create_material_network`` (a bare VOP node in ``/mat``), the
    result is a real, bindable USD material.

    Args:
        ctx: MCP context.
        name: material name; the published ``Material`` prim is ``<matpathprefix>/<name>``.
        parent_path: LOP parent for the ``materiallibrary`` (default ``/stage``).
        base_color: ``[r, g, b]`` diffuse color.
        metalness: 0..1 metalness.
        roughness: 0..1 (sets ``specular_roughness``).
        textures: ``{surface_input_name: file_path}`` — an ``mtlxUsdUVTexture`` per entry wired to
            the named surface input.
        normal_map: file path → ``mtlxUsdUVTexture`` → ``mtlxnormalmap`` → the surface ``normal``.
    """
    bridge = _get_bridge(ctx)
    p: dict[str, Any] = {"name": name, "parent_path": parent_path}
    if base_color is not None:
        p["base_color"] = base_color
    if metalness is not None:
        p["metalness"] = metalness
    if roughness is not None:
        p["roughness"] = roughness
    if textures is not None:
        p["textures"] = textures
    if normal_map is not None:
        p["normal_map"] = normal_map
    return await bridge.execute("materials.create_mtlx_material", p)


@mcp.tool()
async def list_material_types(
    ctx: Context,
    filter: Optional[str] = None,
) -> dict:
    """List available VOP/material node types.

    Args:
        ctx: MCP context.
        filter: Substring to filter type names and labels by.
    """
    bridge = _get_bridge(ctx)
    p: dict[str, Any] = {}
    if filter is not None:
        p["filter"] = filter
    return await bridge.execute("materials.list_material_types", p)
