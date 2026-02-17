"""Pipeline operations for phenotastic.

This module contains all atomic operations that can be used in pipelines.
Operations are either auto-generated from decorated PhenoMesh methods or
manually defined for complex orchestration logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from phenotastic import domains
from phenotastic.exceptions import ConfigurationError
from phenotastic.mesh import contour, create_cellular_mesh, create_mesh
from phenotastic.pipeline_decorator import generate_operation_wrappers

if TYPE_CHECKING:
    from collections.abc import Callable

    from phenotastic.phenomesh import PhenoMesh
    from phenotastic.pipeline import PipelineContext


@dataclass
class OperationInfo:
    """Metadata about an operation."""

    name: str
    description: str
    category: str
    parameters: dict[str, ParameterInfo] = field(default_factory=dict)


@dataclass
class ParameterInfo:
    """Metadata about an operation parameter."""

    name: str
    param_type: str
    default: Any
    description: str


# =============================================================================
# Helper Functions
# =============================================================================


def _get_mesh(context: PipelineContext) -> PhenoMesh:
    """Validate that context has a mesh and return it (narrows type)."""
    if context.mesh is None:
        raise ConfigurationError("This operation requires a mesh in context")
    return context.mesh


def _extract_meristem_index(result: int | tuple[int, Any]) -> int:
    """Extract meristem index from define_meristem result."""
    return result[0] if isinstance(result, tuple) else result


# =============================================================================
# Image/Contour Operations (Manual - Complex)
# =============================================================================


def generate_contour(
    context: PipelineContext,
    iterations: int = 25,
    smoothing: int = 1,
    masking_factor: float = 0.75,
    target_resolution: list[float] | None = None,
    gaussian_sigma: list[float] | None = None,
    gaussian_iterations: int = 5,
    register_stack: bool = True,
    fill_slices: bool = True,
    fill_inland_threshold: float | None = None,
    verbose: bool = True,
) -> PipelineContext:
    """Generate binary contour from 3D image using morphological active contours."""
    if context.image is None:
        raise ConfigurationError("contour operation requires image in context")

    result = contour(
        context.image,
        iterations=iterations,
        smoothing=smoothing,
        masking_factor=masking_factor,
        resolution=context.resolution,
        target_resolution=target_resolution,
        gaussian_sigma=gaussian_sigma,
        gaussian_iterations=gaussian_iterations,
        register_stack=register_stack,
        fill_slices=fill_slices,
        fill_inland_threshold=fill_inland_threshold,
        return_resolution=True,
        verbose=verbose,
    )
    context.contour = result[0]
    context.resolution = list(result[1])

    return context


def create_mesh_from_contour(
    context: PipelineContext,
    step_size: int = 1,
) -> PipelineContext:
    """Create mesh from binary contour using marching cubes."""
    if context.contour is None:
        raise ConfigurationError("create_mesh operation requires contour in context")

    resolution = context.resolution or [1.0, 1.0, 1.0]
    context.mesh = create_mesh(context.contour, resolution=resolution, step_size=step_size)

    return context


def create_mesh_from_cells(
    context: PipelineContext,
    verbose: bool = True,
) -> PipelineContext:
    """Create mesh from segmented image with one mesh per cell."""
    if context.image is None:
        raise ConfigurationError("create_cellular_mesh requires image in context")

    resolution = context.resolution or [1.0, 1.0, 1.0]
    context.mesh = create_cellular_mesh(context.image, resolution=resolution, verbose=verbose)

    return context


# =============================================================================
# Rotation Operation (Manual - Special handling for axis)
# =============================================================================


def rotate_mesh(
    context: PipelineContext,
    axis: str = "x",
    angle: float = 0.0,
) -> PipelineContext:
    """Rotate mesh around specified axis.

    Args:
        context: Pipeline context with mesh
        axis: Rotation axis ('x', 'y', 'z')
        angle: Rotation angle in degrees

    Returns:
        Updated context with rotated mesh
    """
    mesh = _get_mesh(context)

    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ConfigurationError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'")

    if axis == "x":
        context.mesh = mesh.rotate_x(angle)
    elif axis == "y":
        context.mesh = mesh.rotate_y(angle)
    else:
        context.mesh = mesh.rotate_z(angle)

    context.neighbors = None
    return context


# =============================================================================
# Curvature Operations (Manual - Updates context.curvature)
# =============================================================================


def compute_curvature(
    context: PipelineContext,
    curvature_type: Literal["mean", "gaussian", "minimum", "maximum"] = "mean",
) -> PipelineContext:
    """Compute surface curvature and store in context.

    Args:
        context: Pipeline context with mesh
        curvature_type: Curvature type ('mean', 'gaussian', 'minimum', 'maximum')

    Returns:
        Updated context with curvature array
    """
    mesh = _get_mesh(context)

    valid_types = ("mean", "gaussian", "minimum", "maximum")
    if curvature_type not in valid_types:
        raise ConfigurationError(f"curvature_type must be one of {valid_types}")

    context.curvature = mesh.compute_curvature(curvature_type=curvature_type)
    mesh["curvature"] = context.curvature

    return context


def filter_scalars(
    context: PipelineContext,
    scalars: str = "curvature",
    filter_type: str = "median",
    iterations: int = 1,
) -> PipelineContext:
    """Apply filter to scalar field on mesh."""
    mesh = _get_mesh(context)

    if context.neighbors is None:
        context.neighbors = mesh.get_all_vertex_neighbors(include_self=True)

    if scalars == "curvature":
        if context.curvature is None:
            context.curvature = mesh.compute_curvature(curvature_type="mean")
        data = context.curvature
    else:
        if scalars not in mesh.point_data:
            raise ConfigurationError(f"Scalar field '{scalars}' not found")
        data = mesh.point_data[scalars]

    if filter_type == "median":
        result = domains.median(data, neighbors=context.neighbors, iterations=iterations)
    elif filter_type == "mean":
        result = domains.mean(data, neighbors=context.neighbors, iterations=iterations)
    else:
        raise ConfigurationError(f"Unknown filter_type: {filter_type}")

    if scalars == "curvature":
        context.curvature = result
        mesh["curvature"] = result
    else:
        mesh[scalars] = result

    return context


# =============================================================================
# Domain Operations (Manual - Complex context orchestration)
# =============================================================================


def segment_domains(
    context: PipelineContext,
    curvature_type: Literal["mean", "gaussian", "minimum", "maximum"] | None = None,
) -> PipelineContext:
    """Create domains via steepest ascent on curvature field."""
    mesh = _get_mesh(context)

    if context.curvature is None:
        curvature_type = curvature_type or "mean"
        context.curvature = mesh.compute_curvature(curvature_type=curvature_type)
        mesh["curvature"] = context.curvature

    if context.neighbors is None:
        context.neighbors = mesh.get_all_vertex_neighbors(include_self=True)

    context.domains = domains.steepest_ascent(
        mesh.to_polydata(),
        context.curvature,
        neighbours=context.neighbors,
    )
    mesh["domains"] = context.domains

    return context


def merge_by_angles(
    context: PipelineContext,
    threshold: float = 20.0,
    meristem_method: str = "center_of_mass",
) -> PipelineContext:
    """Merge domains within angular threshold from meristem."""
    mesh = _get_mesh(context)

    if context.domains is None:
        raise ConfigurationError("merge_angles requires domains in context")

    if context.meristem_index is None:
        result = domains.define_meristem(
            mesh.to_polydata(),
            context.domains,
            method=meristem_method,
        )
        context.meristem_index = _extract_meristem_index(result)

    context.domains = domains.merge_angles(
        mesh.to_polydata(),
        context.domains,
        context.meristem_index,
        threshold=threshold,
        method=meristem_method,
    )
    mesh["domains"] = context.domains

    return context


def merge_by_distance(
    context: PipelineContext,
    threshold: float = 50.0,
    metric: str = "euclidean",
    method: str = "center_of_mass",
) -> PipelineContext:
    """Merge domains within spatial distance threshold."""
    mesh = _get_mesh(context)

    if context.domains is None:
        raise ConfigurationError("merge_distance requires domains in context")

    context.domains = domains.merge_distance(
        mesh.to_polydata(),
        context.domains,
        threshold=threshold,
        scalars=context.curvature,
        method=method,
        metric=metric,
    )
    mesh["domains"] = context.domains

    return context


def merge_small_domains(
    context: PipelineContext,
    threshold: float = 50,
    metric: str = "points",
    mode: str = "border",
) -> PipelineContext:
    """Merge small domains into neighbors."""
    mesh = _get_mesh(context)

    if context.domains is None:
        raise ConfigurationError("merge_small requires domains in context")

    if context.neighbors is None:
        context.neighbors = mesh.get_all_vertex_neighbors(include_self=True)

    context.domains = domains.merge_small(
        mesh.to_polydata(),
        context.domains,
        threshold=threshold,
        metric=metric,
        mode=mode,
        neighbours=context.neighbors,
    )
    mesh["domains"] = context.domains

    return context


def merge_engulfing_domains(
    context: PipelineContext,
    threshold: float = 0.5,
) -> PipelineContext:
    """Merge domains that engulf others."""
    mesh = _get_mesh(context)

    if context.domains is None:
        raise ConfigurationError("merge_engulfing requires domains in context")

    if context.neighbors is None:
        context.neighbors = mesh.get_all_vertex_neighbors(include_self=True)

    context.domains = domains.merge_engulfing(
        mesh.to_polydata(),
        context.domains,
        threshold,
        neighbours=context.neighbors,
    )
    mesh["domains"] = context.domains

    return context


def merge_disconnected_domains(
    context: PipelineContext,
    method: str = "center_of_mass",
) -> PipelineContext:
    """Merge disconnected components of the same domain."""
    mesh = _get_mesh(context)

    if context.domains is None:
        raise ConfigurationError("merge_disconnected requires domains in context")

    if context.meristem_index is None:
        result = domains.define_meristem(
            mesh.to_polydata(),
            context.domains,
            method=method,
        )
        context.meristem_index = _extract_meristem_index(result)

    if context.neighbors is None:
        context.neighbors = mesh.get_all_vertex_neighbors(include_self=True)

    context.domains = domains.merge_disconnected(
        mesh.to_polydata(),
        context.domains,
        context.meristem_index,
        threshold=None,
        neighbours=context.neighbors,
    )
    mesh["domains"] = context.domains

    return context


def merge_by_depth(
    context: PipelineContext,
    threshold: float = 0.5,
    method: str = "center_of_mass",
) -> PipelineContext:
    """Merge domains based on depth from boundary."""
    mesh = _get_mesh(context)

    if context.domains is None:
        raise ConfigurationError("merge_depth requires domains in context")

    if context.curvature is None:
        context.curvature = mesh.compute_curvature(curvature_type="mean")

    if context.neighbors is None:
        context.neighbors = mesh.get_all_vertex_neighbors(include_self=True)

    context.domains = domains.merge_depth(
        mesh.to_polydata(),
        context.domains,
        context.curvature,
        threshold=threshold,
        neighbours=context.neighbors,
    )
    mesh["domains"] = context.domains

    return context


def define_meristem(
    context: PipelineContext,
    method: str = "center_of_mass",
) -> PipelineContext:
    """Identify which domain corresponds to the meristem."""
    mesh = _get_mesh(context)

    if context.domains is None:
        raise ConfigurationError("define_meristem requires domains in context")

    vertex_neighbors = None
    if method in ("n_neighs", "neighbors", "neighs", "n_neighbors"):
        if context.neighbors is None:
            context.neighbors = mesh.get_all_vertex_neighbors(include_self=True)
        vertex_neighbors = context.neighbors

    result = domains.define_meristem(
        mesh.to_polydata(),
        context.domains,
        method=method,
        neighbors=vertex_neighbors,
    )
    context.meristem_index = _extract_meristem_index(result)

    return context


def extract_domain_data(context: PipelineContext) -> PipelineContext:
    """Extract geometric and spatial data for each domain."""
    mesh = _get_mesh(context)

    if context.domains is None:
        raise ConfigurationError("extract_domain_data requires domains in context")

    if context.meristem_index is None:
        result = domains.define_meristem(
            mesh.to_polydata(),
            context.domains,
        )
        context.meristem_index = _extract_meristem_index(result)

    point_data = pd.DataFrame({"domain": context.domains})
    apex = mesh.compute_center_of_mass()

    context.domain_data = domains.extract_domain_data(
        point_data,
        mesh.to_polydata(),
        apex,
        context.meristem_index,
    )

    return context


# =============================================================================
# Filter by Curvature (Manual - Special parameter handling)
# =============================================================================


def filter_by_curvature(
    context: PipelineContext,
    threshold: float | list[float] = 1.0,
    curvature_type: Literal["mean", "gaussian", "minimum", "maximum"] = "mean",
) -> PipelineContext:
    """Filter mesh vertices by curvature threshold.

    Args:
        context: Pipeline context with mesh
        threshold: Single value for symmetric range [-t, t] or [min, max] list
        curvature_type: Curvature type to compute if not present

    Returns:
        Updated context with filtered mesh
    """
    mesh = _get_mesh(context)

    if context.curvature is None:
        context.curvature = mesh.compute_curvature(curvature_type=curvature_type)

    if isinstance(threshold, list):
        curvature_threshold = (threshold[0], threshold[1])
    else:
        curvature_threshold = (-abs(threshold), abs(threshold))

    context.mesh = mesh.filter_by_curvature(
        curvature_threshold=curvature_threshold,
        curvatures=context.curvature,
    )
    context.neighbors = None
    context.curvature = None

    return context


# =============================================================================
# Auto-Generated Operations from Decorated PhenoMesh Methods
# =============================================================================

# Import PhenoMesh here to avoid circular imports and trigger decorator registration
from phenotastic.phenomesh import PhenoMesh  # noqa: E402

# Generate wrappers from decorated methods
_AUTO_OPERATIONS: dict[str, Callable[..., PipelineContext]] = generate_operation_wrappers(PhenoMesh)


# =============================================================================
# Operations Registry
# =============================================================================

# Manual operations that require complex context orchestration
_MANUAL_OPERATIONS: dict[str, Callable[..., PipelineContext]] = {
    # Contour/Mesh creation
    "contour": generate_contour,
    "create_mesh": create_mesh_from_contour,
    "create_cellular_mesh": create_mesh_from_cells,
    # Rotation (special axis handling)
    "rotate": rotate_mesh,
    # Curvature operations
    "compute_curvature": compute_curvature,
    "filter_scalars": filter_scalars,
    "filter_curvature": filter_by_curvature,
    # Domain operations
    "segment_domains": segment_domains,
    "merge_angles": merge_by_angles,
    "merge_distance": merge_by_distance,
    "merge_small": merge_small_domains,
    "merge_engulfing": merge_engulfing_domains,
    "merge_disconnected": merge_disconnected_domains,
    "merge_depth": merge_by_depth,
    "define_meristem": define_meristem,
    "extract_domain_data": extract_domain_data,
    "extract_domaindata": extract_domain_data,  # Backwards compatibility
}

# Combine auto-generated and manual operations
# Manual operations override auto-generated ones (for special handling)
OPERATIONS: dict[str, Callable[..., PipelineContext]] = {
    **_AUTO_OPERATIONS,
    **_MANUAL_OPERATIONS,
}


# =============================================================================
# Operation Metadata (for documentation/CLI)
# =============================================================================

OPERATION_INFO: dict[str, OperationInfo] = {
    "smooth": OperationInfo(
        name="smooth",
        description="Laplacian smoothing",
        category="mesh",
        parameters={
            "iterations": ParameterInfo("iterations", "int", 100, "Number of iterations"),
            "relaxation_factor": ParameterInfo("relaxation_factor", "float", 0.01, "Relaxation factor"),
        },
    ),
    "remesh": OperationInfo(
        name="remesh",
        description="Regularize faces with ACVD",
        category="mesh",
        parameters={
            "n_clusters": ParameterInfo("n_clusters", "int", 10000, "Target number of faces"),
            "subdivisions": ParameterInfo("subdivisions", "int", 3, "Subdivisions for clustering"),
        },
    ),
}
