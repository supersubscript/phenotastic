"""Pipeline operations for phenotastic.

This module contains all atomic operations that can be used in pipelines.
Operations are either auto-generated from decorated PhenoMesh methods or
manually defined for complex orchestration logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from phenotastic import domains
from phenotastic.exceptions import ConfigurationError
from phenotastic.mesh import contour, create_cellular_mesh, create_mesh
from phenotastic.pipeline_decorator import generate_operation_wrappers

if TYPE_CHECKING:
    from collections.abc import Callable

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


def _validate_mesh_present(context: PipelineContext) -> None:
    """Validate that context has a mesh."""
    if context.mesh is None:
        raise ConfigurationError("This operation requires a mesh in context")


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
    _validate_mesh_present(context)

    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ConfigurationError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'")

    if axis == "x":
        context.mesh = context.mesh.rotate_x(angle)
    elif axis == "y":
        context.mesh = context.mesh.rotate_y(angle)
    else:
        context.mesh = context.mesh.rotate_z(angle)

    context.neighbors = None
    return context


# =============================================================================
# Curvature Operations (Manual - Updates context.curvature)
# =============================================================================


def compute_curvature(
    context: PipelineContext,
    curvature_type: str = "mean",
) -> PipelineContext:
    """Compute surface curvature and store in context.

    Args:
        context: Pipeline context with mesh
        curvature_type: Curvature type ('mean', 'gaussian', 'minimum', 'maximum')

    Returns:
        Updated context with curvature array
    """
    _validate_mesh_present(context)

    valid_types = ("mean", "gaussian", "minimum", "maximum")
    if curvature_type not in valid_types:
        raise ConfigurationError(f"curvature_type must be one of {valid_types}")

    context.curvature = context.mesh.compute_curvature(curvature_type=curvature_type)
    context.mesh["curvature"] = context.curvature

    return context


def filter_scalars(
    context: PipelineContext,
    scalars: str = "curvature",
    filter_type: str = "median",
    iterations: int = 1,
) -> PipelineContext:
    """Apply filter to scalar field on mesh."""
    _validate_mesh_present(context)

    if context.neighbors is None:
        context.neighbors = context.mesh.get_all_vertex_neighbors(include_self=True)

    if scalars == "curvature":
        if context.curvature is None:
            context.curvature = context.mesh.compute_curvature(curvature_type="mean")
        data = context.curvature
    else:
        if scalars not in context.mesh.point_data:
            raise ConfigurationError(f"Scalar field '{scalars}' not found")
        data = context.mesh.point_data[scalars]

    if filter_type == "median":
        result = domains.median(data, neighs=context.neighbors, iterations=iterations)
    elif filter_type == "mean":
        result = domains.mean(data, neighs=context.neighbors, iterations=iterations)
    else:
        raise ConfigurationError(f"Unknown filter_type: {filter_type}")

    if scalars == "curvature":
        context.curvature = result
        context.mesh["curvature"] = result
    else:
        context.mesh[scalars] = result

    return context


# =============================================================================
# Domain Operations (Manual - Complex context orchestration)
# =============================================================================


def segment_domains(
    context: PipelineContext,
    curvature_type: str | None = None,
) -> PipelineContext:
    """Create domains via steepest ascent on curvature field."""
    _validate_mesh_present(context)

    if context.curvature is None:
        curvature_type = curvature_type or "mean"
        context.curvature = context.mesh.compute_curvature(curvature_type=curvature_type)
        context.mesh["curvature"] = context.curvature

    if context.neighbors is None:
        context.neighbors = context.mesh.get_all_vertex_neighbors(include_self=True)

    context.domains = domains.steepest_ascent(
        context.mesh.to_polydata(),
        context.curvature,
        neighbours=context.neighbors,
    )
    context.mesh["domains"] = context.domains

    return context


def merge_by_angles(
    context: PipelineContext,
    threshold: float = 20.0,
    meristem_method: str = "center_of_mass",
) -> PipelineContext:
    """Merge domains within angular threshold from meristem."""
    _validate_mesh_present(context)

    if context.domains is None:
        raise ConfigurationError("merge_angles requires domains in context")

    if context.meristem_index is None:
        result = domains.define_meristem(
            context.mesh.to_polydata(),
            context.domains,
            method=meristem_method,
        )
        context.meristem_index = _extract_meristem_index(result)

    context.domains = domains.merge_angles(
        context.mesh.to_polydata(),
        context.domains,
        context.meristem_index,
        threshold=threshold,
        method=meristem_method,
    )
    context.mesh["domains"] = context.domains

    return context


def merge_by_distance(
    context: PipelineContext,
    threshold: float = 50.0,
    metric: str = "euclidean",
    method: str = "center_of_mass",
) -> PipelineContext:
    """Merge domains within spatial distance threshold."""
    _validate_mesh_present(context)

    if context.domains is None:
        raise ConfigurationError("merge_distance requires domains in context")

    if context.neighbors is None:
        context.neighbors = context.mesh.get_all_vertex_neighbors(include_self=True)

    context.domains = domains.merge_distance(
        context.mesh.to_polydata(),
        context.domains,
        threshold=threshold,
        scalars=context.curvature,
        method=method,
        metric=metric,
        neighbours=context.neighbors,
    )
    context.mesh["domains"] = context.domains

    return context


def merge_small_domains(
    context: PipelineContext,
    threshold: float = 50,
    metric: str = "points",
    mode: str = "border",
) -> PipelineContext:
    """Merge small domains into neighbors."""
    _validate_mesh_present(context)

    if context.domains is None:
        raise ConfigurationError("merge_small requires domains in context")

    if context.neighbors is None:
        context.neighbors = context.mesh.get_all_vertex_neighbors(include_self=True)

    context.domains = domains.merge_small(
        context.mesh.to_polydata(),
        context.domains,
        threshold=threshold,
        metric=metric,
        mode=mode,
        neighbours=context.neighbors,
    )
    context.mesh["domains"] = context.domains

    return context


def merge_engulfing_domains(
    context: PipelineContext,
    threshold: float = 0.5,
    method: str = "center_of_mass",
) -> PipelineContext:
    """Merge domains that engulf others."""
    _validate_mesh_present(context)

    if context.domains is None:
        raise ConfigurationError("merge_engulfing requires domains in context")

    if context.neighbors is None:
        context.neighbors = context.mesh.get_all_vertex_neighbors(include_self=True)

    context.domains = domains.merge_engulfing(
        context.mesh.to_polydata(),
        context.domains,
        threshold,
        neighbours=context.neighbors,
    )
    context.mesh["domains"] = context.domains

    return context


def merge_disconnected_domains(
    context: PipelineContext,
    method: str = "center_of_mass",
) -> PipelineContext:
    """Merge disconnected components of the same domain."""
    _validate_mesh_present(context)

    if context.domains is None:
        raise ConfigurationError("merge_disconnected requires domains in context")

    if context.neighbors is None:
        context.neighbors = context.mesh.get_all_vertex_neighbors(include_self=True)

    context.domains = domains.merge_disconnected(
        context.mesh.to_polydata(),
        context.domains,
        scalars=context.curvature,
        method=method,
        neighbours=context.neighbors,
    )
    context.mesh["domains"] = context.domains

    return context


def merge_by_depth(
    context: PipelineContext,
    threshold: float = 0.5,
    method: str = "center_of_mass",
) -> PipelineContext:
    """Merge domains based on depth from boundary."""
    _validate_mesh_present(context)

    if context.domains is None:
        raise ConfigurationError("merge_depth requires domains in context")

    if context.meristem_index is None:
        result = domains.define_meristem(
            context.mesh.to_polydata(),
            context.domains,
            method=method,
        )
        context.meristem_index = _extract_meristem_index(result)

    if context.neighbors is None:
        context.neighbors = context.mesh.get_all_vertex_neighbors(include_self=True)

    context.domains = domains.merge_depth(
        context.mesh.to_polydata(),
        context.domains,
        context.meristem_index,
        threshold=threshold,
        method=method,
        neighbours=context.neighbors,
    )
    context.mesh["domains"] = context.domains

    return context


def define_meristem(
    context: PipelineContext,
    method: str = "center_of_mass",
) -> PipelineContext:
    """Identify which domain corresponds to the meristem."""
    _validate_mesh_present(context)

    if context.domains is None:
        raise ConfigurationError("define_meristem requires domains in context")

    neighs = None
    if method in ("n_neighs", "neighbors", "neighs", "n_neighbors"):
        if context.neighbors is None:
            context.neighbors = context.mesh.get_all_vertex_neighbors(include_self=True)
        neighs = context.neighbors

    result = domains.define_meristem(
        context.mesh.to_polydata(),
        context.domains,
        method=method,
        neighs=neighs,
    )
    context.meristem_index = _extract_meristem_index(result)

    return context


def extract_domain_data(context: PipelineContext) -> PipelineContext:
    """Extract geometric and spatial data for each domain."""
    _validate_mesh_present(context)

    if context.domains is None:
        raise ConfigurationError("extract_domaindata requires domains in context")

    if context.meristem_index is None:
        result = domains.define_meristem(
            context.mesh.to_polydata(),
            context.domains,
        )
        context.meristem_index = _extract_meristem_index(result)

    pdata = pd.DataFrame({"domain": context.domains})
    apex = context.mesh.compute_center_of_mass()

    context.domain_data = domains.extract_domaindata(
        pdata,
        context.mesh.to_polydata(),
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
    curvature_type: str = "mean",
) -> PipelineContext:
    """Filter mesh vertices by curvature threshold.

    Args:
        context: Pipeline context with mesh
        threshold: Single value for symmetric range [-t, t] or [min, max] list
        curvature_type: Curvature type to compute if not present

    Returns:
        Updated context with filtered mesh
    """
    _validate_mesh_present(context)

    if context.curvature is None:
        context.curvature = context.mesh.compute_curvature(curvature_type=curvature_type)

    if isinstance(threshold, list):
        curvature_threshold = (threshold[0], threshold[1])
    else:
        curvature_threshold = (-abs(threshold), abs(threshold))

    context.mesh = context.mesh.filter_by_curvature(
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
    "extract_domaindata": extract_domain_data,
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
