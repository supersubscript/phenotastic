"""Pipeline operations for phenotastic.

This module contains all atomic operations that can be used in pipelines.
Each operation takes a PipelineContext and parameters, modifies the context,
and returns it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from phenotastic import domains
from phenotastic.exceptions import ConfigurationError
from phenotastic.mesh import contour, create_cellular_mesh, create_mesh

if TYPE_CHECKING:
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
    """Extract meristem index from define_meristem result.

    Args:
        result: Either an integer index or tuple of (index, coordinates)

    Returns:
        The meristem domain index
    """
    return result[0] if isinstance(result, tuple) else result


# =============================================================================
# Image/Contour Operations
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
    chan_vese_lambda1: float = 1.0,
    chan_vese_lambda2: float = 1.0,
    fill_slices: bool = True,
    crop: bool = True,
) -> PipelineContext:
    """Generate binary contour from 3D image using morphological active contours.

    Args:
        context: Pipeline context with image data
        iterations: Number of morphological Chan-Vese iterations
        smoothing: Smoothing iterations per cycle
        masking_factor: Initial mask threshold factor
        target_resolution: Target resolution for resampling
        gaussian_sigma: Gaussian filter sigma
        gaussian_iterations: Number of Gaussian smoothing iterations
        register_stack: Apply stack registration
        chan_vese_lambda1: Chan-Vese lambda1 parameter
        chan_vese_lambda2: Chan-Vese lambda2 parameter
        fill_slices: Fill holes in XY slices
        crop: Automatically crop image

    Returns:
        Updated context with contour
    """
    if context.image is None:
        raise ConfigurationError("contour operation requires image in context")

    if target_resolution is None:
        target_resolution = [0.5, 0.5, 0.5]
    if gaussian_sigma is None:
        gaussian_sigma = [1.0, 1.0, 1.0]

    result = contour(
        context.image,
        iterations=iterations,
        smoothing=smoothing,
        masking_factor=masking_factor,
        target_resolution=target_resolution,
        gaussian_sigma=gaussian_sigma,
        gaussian_iterations=gaussian_iterations,
        register_stack=register_stack,
        chan_vese_lambda1=chan_vese_lambda1,
        chan_vese_lambda2=chan_vese_lambda2,
        fill_slices=fill_slices,
        crop=crop,
        return_resolution=True,
    )

    if isinstance(result, tuple):
        context.contour = result[0]
        context.resolution = list(result[1])
    else:
        context.contour = result

    return context


def create_mesh_from_contour(
    context: PipelineContext,
    step_size: int = 1,
) -> PipelineContext:
    """Create mesh from contour using marching cubes.

    Args:
        context: Pipeline context with contour data
        step_size: Step size for marching cubes

    Returns:
        Updated context with mesh
    """
    if context.contour is None:
        raise ConfigurationError("create_mesh operation requires contour in context")

    resolution = context.resolution or [1.0, 1.0, 1.0]
    context.mesh = create_mesh(context.contour, resolution=resolution, step_size=step_size)

    return context


def create_mesh_from_cells(
    context: PipelineContext,
    verbose: bool = True,
) -> PipelineContext:
    """Create mesh from segmented image with one mesh per cell.

    Args:
        context: Pipeline context with segmented image
        verbose: Print progress information

    Returns:
        Updated context with mesh
    """
    if context.image is None:
        raise ConfigurationError("create_cellular_mesh requires image in context")

    resolution = context.resolution or [1.0, 1.0, 1.0]
    context.mesh = create_cellular_mesh(context.image, resolution=resolution, verbose=verbose)

    return context


# =============================================================================
# Mesh Processing Operations
# =============================================================================


def smooth_mesh(
    context: PipelineContext,
    iterations: int = 100,
    relaxation_factor: float = 0.01,
    feature_smoothing: bool = False,
    boundary_smoothing: bool = True,
) -> PipelineContext:
    """Smooth mesh using Laplacian smoothing.

    Args:
        context: Pipeline context with mesh
        iterations: Number of smoothing iterations
        relaxation_factor: Relaxation factor (0-1)
        feature_smoothing: Smooth along features
        boundary_smoothing: Smooth boundary edges

    Returns:
        Updated context with smoothed mesh
    """
    _validate_mesh_present(context)

    if iterations < 0:
        raise ConfigurationError("iterations must be non-negative")
    if not 0 <= relaxation_factor <= 1:
        raise ConfigurationError("relaxation_factor must be in [0, 1]")

    if iterations == 0:
        return context

    context.mesh = context.mesh.smoothen(
        iterations=iterations,
        relaxation_factor=relaxation_factor,
        feature_smoothing=feature_smoothing,
        boundary_smoothing=boundary_smoothing,
    )
    context.neighbors = None

    return context


def smooth_mesh_taubin(
    context: PipelineContext,
    iterations: int = 100,
    pass_band: float = 0.1,
    feature_smoothing: bool = False,
    boundary_smoothing: bool = True,
) -> PipelineContext:
    """Smooth mesh using Taubin smoothing (less shrinkage than Laplacian).

    Args:
        context: Pipeline context with mesh
        iterations: Number of smoothing iterations
        pass_band: Pass band for filter (0-2)
        feature_smoothing: Smooth along features
        boundary_smoothing: Smooth boundary edges

    Returns:
        Updated context with smoothed mesh
    """
    _validate_mesh_present(context)

    if iterations < 0:
        raise ConfigurationError("iterations must be non-negative")

    if iterations == 0:
        return context

    context.mesh = context.mesh.smooth_taubin(
        iterations=iterations,
        pass_band=pass_band,
        feature_smoothing=feature_smoothing,
        boundary_smoothing=boundary_smoothing,
    )
    context.neighbors = None

    return context


def smooth_boundary(
    context: PipelineContext,
    iterations: int = 20,
    sigma: float = 0.1,
) -> PipelineContext:
    """Smooth only the boundary edges of the mesh.

    Args:
        context: Pipeline context with mesh
        iterations: Number of smoothing iterations
        sigma: Smoothing sigma

    Returns:
        Updated context with boundary-smoothed mesh
    """
    _validate_mesh_present(context)

    if iterations <= 0:
        return context

    context.mesh = context.mesh.smooth_boundary(iterations=iterations, sigma=sigma)
    context.neighbors = None

    return context


def remesh(
    context: PipelineContext,
    n_clusters: int = 10000,
    subdivisions: int = 3,
) -> PipelineContext:
    """Regularize mesh faces using ACVD algorithm.

    Args:
        context: Pipeline context with mesh
        n_clusters: Target number of faces
        subdivisions: Number of subdivisions for clustering

    Returns:
        Updated context with remeshed mesh
    """
    _validate_mesh_present(context)

    if n_clusters <= 0:
        raise ConfigurationError("n_clusters must be positive")

    context.mesh = context.mesh.remesh(n=n_clusters, sub=subdivisions)
    context.neighbors = None

    return context


def decimate_mesh(
    context: PipelineContext,
    target_reduction: float = 0.5,
    volume_preservation: bool = True,
) -> PipelineContext:
    """Reduce mesh complexity by removing faces.

    Args:
        context: Pipeline context with mesh
        target_reduction: Fraction of faces to remove (0-1)
        volume_preservation: Preserve mesh volume

    Returns:
        Updated context with decimated mesh
    """
    _validate_mesh_present(context)

    if not 0 <= target_reduction < 1:
        raise ConfigurationError("target_reduction must be in [0, 1)")

    context.mesh = context.mesh.decimate(
        target_reduction=target_reduction,
        volume_preservation=volume_preservation,
    )
    context.neighbors = None

    return context


def subdivide_mesh(
    context: PipelineContext,
    n_subdivisions: int = 1,
    subfilter: str = "linear",
) -> PipelineContext:
    """Increase mesh resolution by subdividing faces.

    Args:
        context: Pipeline context with mesh
        n_subdivisions: Number of subdivision iterations
        subfilter: Subdivision filter ('linear', 'butterfly', 'loop')

    Returns:
        Updated context with subdivided mesh
    """
    _validate_mesh_present(context)

    if n_subdivisions <= 0:
        return context

    context.mesh = context.mesh.subdivide(n_subdivisions=n_subdivisions, subfilter=subfilter)
    context.neighbors = None

    return context


def repair_holes(
    context: PipelineContext,
    max_hole_edges: int = 100,
    refine: bool = True,
) -> PipelineContext:
    """Fill small holes in the mesh.

    Args:
        context: Pipeline context with mesh
        max_hole_edges: Maximum hole size to fill (in edges)
        refine: Refine the filled region

    Returns:
        Updated context with repaired mesh
    """
    _validate_mesh_present(context)

    context.mesh = context.mesh.repair_small(nbe=max_hole_edges, refine=refine)
    context.neighbors = None

    return context


def repair_mesh(context: PipelineContext) -> PipelineContext:
    """Full mesh repair using MeshFix.

    Args:
        context: Pipeline context with mesh

    Returns:
        Updated context with repaired mesh
    """
    _validate_mesh_present(context)

    context.mesh = context.mesh.repair()
    context.neighbors = None

    return context


def make_manifold(
    context: PipelineContext,
    hole_edges: int = 300,
) -> PipelineContext:
    """Remove non-manifold edges from mesh.

    Args:
        context: Pipeline context with mesh
        hole_edges: Size of holes to fill after removal

    Returns:
        Updated context with manifold mesh
    """
    _validate_mesh_present(context)

    context.mesh = context.mesh.make_manifold(hole_edges=hole_edges)
    context.neighbors = None

    return context


def filter_by_curvature(
    context: PipelineContext,
    threshold: float | list[float] = 0.4,
) -> PipelineContext:
    """Remove vertices outside curvature threshold range.

    Args:
        context: Pipeline context with mesh
        threshold: Single value for symmetric range or [min, max] list

    Returns:
        Updated context with filtered mesh
    """
    _validate_mesh_present(context)

    if isinstance(threshold, list):
        if len(threshold) != 2:
            raise ConfigurationError("threshold list must have exactly 2 elements [min, max]")
        threshold_tuple = (threshold[0], threshold[1])
    else:
        threshold_tuple = (-threshold, threshold)

    context.mesh = context.mesh.filter_curvature(curvature_threshold=threshold_tuple)
    context.neighbors = None

    return context


def remove_by_normals(
    context: PipelineContext,
    threshold_angle: float = 60.0,
    flip: bool = False,
    angle_type: str = "polar",
) -> PipelineContext:
    """Remove vertices based on normal angle.

    Args:
        context: Pipeline context with mesh
        threshold_angle: Angle threshold in degrees
        flip: Flip normal orientation before filtering
        angle_type: Type of angle ('polar' or 'azimuth')

    Returns:
        Updated context with filtered mesh
    """
    _validate_mesh_present(context)

    context.mesh = context.mesh.remove_normals(
        threshold_angle=threshold_angle,
        flip=flip,
        angle=angle_type,
    )
    context.neighbors = None

    return context


def remove_bridges(context: PipelineContext) -> PipelineContext:
    """Remove triangles where all vertices are on the boundary.

    Args:
        context: Pipeline context with mesh

    Returns:
        Updated context with bridges removed
    """
    _validate_mesh_present(context)

    context.mesh = context.mesh.remove_bridges(verbose=False)
    context.neighbors = None

    return context


def remove_tongues(
    context: PipelineContext,
    radius: float = 50.0,
    threshold: float = 6.0,
    hole_edges: int = 100,
) -> PipelineContext:
    """Remove tongue-like artifacts from mesh.

    Args:
        context: Pipeline context with mesh
        radius: Radius for boundary point neighborhood
        threshold: Threshold for boundary/euclidean distance ratio
        hole_edges: Size of holes to fill after removal

    Returns:
        Updated context with tongues removed
    """
    _validate_mesh_present(context)

    context.mesh = context.mesh.remove_tongues(
        radius=radius,
        threshold=threshold,
        hole_edges=hole_edges,
        verbose=False,
    )
    context.neighbors = None

    return context


def extract_largest(context: PipelineContext) -> PipelineContext:
    """Keep only the largest connected component.

    Args:
        context: Pipeline context with mesh

    Returns:
        Updated context with largest component only
    """
    _validate_mesh_present(context)

    context.mesh = context.mesh.extract_largest()
    context.neighbors = None

    return context


def clean_mesh(
    context: PipelineContext,
    tolerance: float | None = None,
) -> PipelineContext:
    """Clean mesh by removing degenerate cells.

    Args:
        context: Pipeline context with mesh
        tolerance: Tolerance for point merging

    Returns:
        Updated context with cleaned mesh
    """
    _validate_mesh_present(context)

    context.mesh = context.mesh.clean(tolerance=tolerance)
    context.neighbors = None

    return context


def triangulate_mesh(context: PipelineContext) -> PipelineContext:
    """Convert all faces to triangles.

    Args:
        context: Pipeline context with mesh

    Returns:
        Updated context with triangulated mesh
    """
    _validate_mesh_present(context)

    context.mesh = context.mesh.triangulate()
    context.neighbors = None

    return context


def compute_normals(
    context: PipelineContext,
    flip: bool = False,
    consistent: bool = True,
    auto_orient: bool = False,
) -> PipelineContext:
    """Compute surface normals.

    Args:
        context: Pipeline context with mesh
        flip: Flip all normals
        consistent: Make normals consistent
        auto_orient: Orient normals outward

    Returns:
        Updated context with computed normals
    """
    _validate_mesh_present(context)

    context.mesh = context.mesh.compute_normals(
        flip_normals=flip,
        consistent_normals=consistent,
        auto_orient_normals=auto_orient,
    )

    return context


def flip_normals(context: PipelineContext) -> PipelineContext:
    """Flip all surface normals.

    Args:
        context: Pipeline context with mesh

    Returns:
        Updated context with flipped normals
    """
    _validate_mesh_present(context)

    context.mesh = context.mesh.flip_normals()

    return context


def rotate_mesh(
    context: PipelineContext,
    axis: str = "x",
    angle: float = 0.0,
) -> PipelineContext:
    """Rotate mesh around an axis.

    Args:
        context: Pipeline context with mesh
        axis: Axis to rotate around ('x', 'y', 'z')
        angle: Rotation angle in degrees

    Returns:
        Updated context with rotated mesh
    """
    _validate_mesh_present(context)

    if angle == 0:
        return context

    axis_lower = axis.lower()
    if axis_lower == "x":
        context.mesh = context.mesh.rotate_x(angle)
    elif axis_lower == "y":
        context.mesh = context.mesh.rotate_y(angle)
    elif axis_lower == "z":
        context.mesh = context.mesh.rotate_z(angle)
    else:
        raise ConfigurationError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")

    return context


def clip_mesh(
    context: PipelineContext,
    normal: str | list[float] = "x",
    origin: list[float] | None = None,
    invert: bool = True,
) -> PipelineContext:
    """Clip mesh with a plane.

    Args:
        context: Pipeline context with mesh
        normal: Plane normal ('x', 'y', 'z', '-x', '-y', '-z') or [nx, ny, nz]
        origin: Point on clipping plane
        invert: Invert clipping direction

    Returns:
        Updated context with clipped mesh
    """
    _validate_mesh_present(context)

    context.mesh = context.mesh.clip(normal=normal, origin=origin, invert=invert)
    context.neighbors = None

    return context


def erode_mesh(
    context: PipelineContext,
    iterations: int = 1,
) -> PipelineContext:
    """Erode mesh by removing boundary points.

    Args:
        context: Pipeline context with mesh
        iterations: Number of erosion iterations

    Returns:
        Updated context with eroded mesh
    """
    _validate_mesh_present(context)

    if iterations <= 0:
        return context

    context.mesh = context.mesh.erode(iterations=iterations)
    context.neighbors = None

    return context


def extract_clean_fill_triangulate(
    context: PipelineContext,
    hole_edges: int = 300,
) -> PipelineContext:
    """ExtractLargest, Clean, FillHoles, Triangulate.

    Args:
        context: Pipeline context with mesh
        hole_edges: Size of holes to fill

    Returns:
        Updated context with processed mesh
    """
    _validate_mesh_present(context)

    context.mesh = context.mesh.ecft(hole_edges=hole_edges)
    context.neighbors = None

    return context


def correct_normal_orientation(
    context: PipelineContext,
    relative: str = "x",
) -> PipelineContext:
    """Correct normal orientation relative to an axis.

    Args:
        context: Pipeline context with mesh
        relative: Axis to use for orientation ('x', 'y', 'z')

    Returns:
        Updated context with corrected normals
    """
    _validate_mesh_present(context)

    result = context.mesh.correct_normal_orientation(relative=relative, inplace=False)
    if result is not None:
        context.mesh = result

    return context


# =============================================================================
# Domain Operations
# =============================================================================


def compute_curvature(
    context: PipelineContext,
    curvature_type: str = "mean",
) -> PipelineContext:
    """Compute mesh curvature.

    Args:
        context: Pipeline context with mesh
        curvature_type: Type of curvature ('mean', 'gaussian', 'minimum', 'maximum')

    Returns:
        Updated context with curvature array
    """
    _validate_mesh_present(context)

    valid_types = ["mean", "gaussian", "minimum", "maximum"]
    if curvature_type not in valid_types:
        raise ConfigurationError(f"curvature_type must be one of {valid_types}")

    context.curvature = context.mesh.curvature(curv_type=curvature_type)
    context.mesh["curvature"] = context.curvature

    return context


def filter_scalars(
    context: PipelineContext,
    function: str = "median",
    iterations: int = 1,
) -> PipelineContext:
    """Apply filter to scalar field (curvature).

    Args:
        context: Pipeline context with curvature
        function: Filter function ('median', 'mean', 'minmax', 'maxmin')
        iterations: Number of filter iterations

    Returns:
        Updated context with filtered curvature
    """
    _validate_mesh_present(context)

    if context.curvature is None:
        raise ConfigurationError("filter_scalars requires curvature in context")

    if context.neighbors is None:
        context.neighbors = context.mesh.vertex_neighbors_all(include_self=True)

    func_map = {
        "median": domains.median,
        "mean": domains.mean,
        "minmax": domains.minmax,
        "maxmin": domains.maxmin,
    }

    if function not in func_map:
        raise ConfigurationError(f"function must be one of {list(func_map.keys())}")

    context.curvature = func_map[function](context.curvature, context.neighbors, iterations)
    context.mesh["curvature"] = context.curvature

    return context


def segment_domains(
    context: PipelineContext,
    curvature_type: str | None = None,
) -> PipelineContext:
    """Create domains via steepest ascent on curvature field.

    Args:
        context: Pipeline context with mesh and curvature
        curvature_type: Curvature type to compute (if not already in context)

    Returns:
        Updated context with domain labels
    """
    _validate_mesh_present(context)

    if context.curvature is None:
        curvature_type = curvature_type or "mean"
        context.curvature = context.mesh.curvature(curv_type=curvature_type)
        context.mesh["curvature"] = context.curvature

    if context.neighbors is None:
        context.neighbors = context.mesh.vertex_neighbors_all(include_self=True)

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
    """Merge domains within angular threshold from meristem.

    Args:
        context: Pipeline context with domains
        threshold: Angular threshold in degrees
        meristem_method: Method for calculating meristem center

    Returns:
        Updated context with merged domains
    """
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
    """Merge domains within spatial distance threshold.

    Args:
        context: Pipeline context with domains
        threshold: Distance threshold
        metric: Distance metric ('euclidean' or 'geodesic')
        method: Method for calculating domain center

    Returns:
        Updated context with merged domains
    """
    _validate_mesh_present(context)

    if context.domains is None:
        raise ConfigurationError("merge_distance requires domains in context")

    context.domains = domains.merge_distance(
        context.mesh.to_polydata(),
        context.domains,
        threshold=threshold,
        scalars=context.curvature,
        method=method,
        metric=metric,
    )
    context.mesh["domains"] = context.domains

    return context


def merge_small_domains(
    context: PipelineContext,
    threshold: int = 100,
    metric: str = "points",
    mode: str = "border",
) -> PipelineContext:
    """Merge small domains to their largest neighbor.

    Args:
        context: Pipeline context with domains
        threshold: Size threshold for merging
        metric: Size metric ('points' or 'area')
        mode: Merge strategy ('border' or 'area')

    Returns:
        Updated context with merged domains
    """
    _validate_mesh_present(context)

    if context.domains is None:
        raise ConfigurationError("merge_small requires domains in context")

    if context.neighbors is None:
        context.neighbors = context.mesh.vertex_neighbors_all(include_self=True)

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
    threshold: float = 0.9,
) -> PipelineContext:
    """Merge domains mostly encircled by a neighbor.

    Args:
        context: Pipeline context with domains
        threshold: Fraction of boundary that must be shared (0-1)

    Returns:
        Updated context with merged domains
    """
    _validate_mesh_present(context)

    if context.domains is None:
        raise ConfigurationError("merge_engulfing requires domains in context")

    if context.neighbors is None:
        context.neighbors = context.mesh.vertex_neighbors_all(include_self=True)

    context.domains = domains.merge_engulfing(
        context.mesh.to_polydata(),
        context.domains,
        threshold=threshold,
        neighbours=context.neighbors,
    )
    context.mesh["domains"] = context.domains

    return context


def merge_disconnected_domains(
    context: PipelineContext,
    meristem_method: str = "center_of_mass",
) -> PipelineContext:
    """Connect domains isolated from meristem.

    Args:
        context: Pipeline context with domains
        meristem_method: Method for identifying meristem

    Returns:
        Updated context with connected domains
    """
    _validate_mesh_present(context)

    if context.domains is None:
        raise ConfigurationError("merge_disconnected requires domains in context")

    if context.neighbors is None:
        context.neighbors = context.mesh.vertex_neighbors_all(include_self=True)

    if context.meristem_index is None:
        result = domains.define_meristem(
            context.mesh.to_polydata(),
            context.domains,
            method=meristem_method,
        )
        context.meristem_index = _extract_meristem_index(result)

    context.domains = domains.merge_disconnected(
        context.mesh.to_polydata(),
        context.domains,
        context.meristem_index,
        threshold=None,
        neighbours=context.neighbors,
    )
    context.mesh["domains"] = context.domains

    return context


def merge_by_depth(
    context: PipelineContext,
    threshold: float = 0.0,
    mode: str = "max",
    exclude_boundary: bool = False,
    min_points: int = 0,
) -> PipelineContext:
    """Merge domains with similar depth values.

    Args:
        context: Pipeline context with domains and curvature
        threshold: Maximum depth difference for merging
        mode: Aggregation mode ('min', 'max', 'median', 'mean')
        exclude_boundary: Exclude boundary vertices from calculation
        min_points: Minimum border points required for merging

    Returns:
        Updated context with merged domains
    """
    _validate_mesh_present(context)

    if context.domains is None:
        raise ConfigurationError("merge_depth requires domains in context")
    if context.curvature is None:
        raise ConfigurationError("merge_depth requires curvature in context")

    if context.neighbors is None:
        context.neighbors = context.mesh.vertex_neighbors_all(include_self=True)

    context.domains = domains.merge_depth(
        context.mesh.to_polydata(),
        context.domains,
        context.curvature,
        threshold=threshold,
        neighbours=context.neighbors,
        exclude_boundary=exclude_boundary,
        min_points=min_points,
        mode=mode,
    )
    context.mesh["domains"] = context.domains

    return context


def define_meristem(
    context: PipelineContext,
    method: str = "center_of_mass",
) -> PipelineContext:
    """Identify the meristem domain.

    Args:
        context: Pipeline context with domains
        method: Method for meristem identification

    Returns:
        Updated context with meristem index
    """
    _validate_mesh_present(context)

    if context.domains is None:
        raise ConfigurationError("define_meristem requires domains in context")

    if context.neighbors is None:
        context.neighbors = context.mesh.vertex_neighbors_all(include_self=True)

    result = domains.define_meristem(
        context.mesh.to_polydata(),
        context.domains,
        method=method,
        neighs=context.neighbors,
    )
    context.meristem_index = _extract_meristem_index(result)

    return context


def extract_domain_data(context: PipelineContext) -> PipelineContext:
    """Extract geometric measurements for each domain.

    Args:
        context: Pipeline context with mesh and domains

    Returns:
        Updated context with domain_data DataFrame
    """
    _validate_mesh_present(context)

    if context.domains is None:
        raise ConfigurationError("extract_domaindata requires domains in context")

    pdata = pd.DataFrame({"domain": context.domains})

    # Get apex/meristem coordinates
    if context.meristem_index is None:
        result = domains.define_meristem(
            context.mesh.to_polydata(),
            context.domains,
            return_coordinates=True,
        )
        if isinstance(result, tuple):
            context.meristem_index = result[0]
            apex = result[1]
        else:
            context.meristem_index = result
            apex = np.array(context.mesh.center_of_mass())
    else:
        apex = np.array(context.mesh.center_of_mass())

    context.domain_data = domains.extract_domaindata(
        pdata,
        context.mesh.to_polydata(),
        apex,
        context.meristem_index,
    )

    return context


# =============================================================================
# Operation Registry
# =============================================================================

# Map of YAML operation names to functions
OPERATIONS: dict[str, Any] = {
    # Image/Contour
    "contour": generate_contour,
    "create_mesh": create_mesh_from_contour,
    "create_cellular_mesh": create_mesh_from_cells,
    # Mesh Processing
    "smooth": smooth_mesh,
    "smooth_taubin": smooth_mesh_taubin,
    "smooth_boundary": smooth_boundary,
    "remesh": remesh,
    "decimate": decimate_mesh,
    "subdivide": subdivide_mesh,
    "repair_holes": repair_holes,
    "repair": repair_mesh,
    "make_manifold": make_manifold,
    "filter_curvature": filter_by_curvature,
    "remove_normals": remove_by_normals,
    "remove_bridges": remove_bridges,
    "remove_tongues": remove_tongues,
    "extract_largest": extract_largest,
    "clean": clean_mesh,
    "triangulate": triangulate_mesh,
    "compute_normals": compute_normals,
    "flip_normals": flip_normals,
    "rotate": rotate_mesh,
    "clip": clip_mesh,
    "erode": erode_mesh,
    "ecft": extract_clean_fill_triangulate,
    "correct_normal_orientation": correct_normal_orientation,
    # Domain Operations
    "compute_curvature": compute_curvature,
    "filter_scalars": filter_scalars,
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

# Operation metadata for documentation
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
