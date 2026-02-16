"""Pipeline operations for phenotastic.

This module contains all atomic operations that can be used in pipelines.
Each operation takes a PipelineContext and parameters, modifies the context,
and returns it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from phenotastic.exceptions import ConfigurationError

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
# Image/Contour Operations
# =============================================================================


def contour_op(
    ctx: PipelineContext,
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
        ctx: Pipeline context with image data
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
    from phenotastic.mesh import contour

    if ctx.image is None:
        raise ConfigurationError("contour operation requires image in context")

    if target_resolution is None:
        target_resolution = [0.5, 0.5, 0.5]
    if gaussian_sigma is None:
        gaussian_sigma = [1.0, 1.0, 1.0]

    result = contour(
        ctx.image,
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
        ctx.contour = result[0]
        ctx.resolution = list(result[1])
    else:
        ctx.contour = result

    return ctx


def create_mesh_op(
    ctx: PipelineContext,
    step_size: int = 1,
) -> PipelineContext:
    """Create mesh from contour using marching cubes.

    Args:
        ctx: Pipeline context with contour data
        step_size: Step size for marching cubes

    Returns:
        Updated context with mesh
    """
    from phenotastic.mesh import create_mesh

    if ctx.contour is None:
        raise ConfigurationError("create_mesh operation requires contour in context")

    resolution = ctx.resolution or [1.0, 1.0, 1.0]
    ctx.mesh = create_mesh(ctx.contour, resolution=resolution, step_size=step_size)

    return ctx


def create_cellular_mesh_op(
    ctx: PipelineContext,
    verbose: bool = True,
) -> PipelineContext:
    """Create mesh from segmented image with one mesh per cell.

    Args:
        ctx: Pipeline context with segmented image
        verbose: Print progress information

    Returns:
        Updated context with mesh
    """
    from phenotastic.mesh import create_cellular_mesh

    if ctx.image is None:
        raise ConfigurationError("create_cellular_mesh requires image in context")

    resolution = ctx.resolution or [1.0, 1.0, 1.0]
    ctx.mesh = create_cellular_mesh(ctx.image, resolution=resolution, verbose=verbose)

    return ctx


# =============================================================================
# Mesh Processing Operations
# =============================================================================


def _require_mesh(ctx: PipelineContext) -> None:
    """Validate that context has a mesh."""
    if ctx.mesh is None:
        raise ConfigurationError("This operation requires a mesh in context")


def smooth_op(
    ctx: PipelineContext,
    iterations: int = 100,
    relaxation_factor: float = 0.01,
    feature_smoothing: bool = False,
    boundary_smoothing: bool = True,
) -> PipelineContext:
    """Smooth mesh using Laplacian smoothing.

    Args:
        ctx: Pipeline context with mesh
        iterations: Number of smoothing iterations
        relaxation_factor: Relaxation factor (0-1)
        feature_smoothing: Smooth along features
        boundary_smoothing: Smooth boundary edges

    Returns:
        Updated context with smoothed mesh
    """
    _require_mesh(ctx)

    if iterations < 0:
        raise ConfigurationError("iterations must be non-negative")
    if not 0 <= relaxation_factor <= 1:
        raise ConfigurationError("relaxation_factor must be in [0, 1]")

    if iterations == 0:
        return ctx

    ctx.mesh = ctx.mesh.smoothen(
        iterations=iterations,
        relaxation_factor=relaxation_factor,
        feature_smoothing=feature_smoothing,
        boundary_smoothing=boundary_smoothing,
    )
    # Clear cached neighbors since mesh changed
    ctx.neighbors = None

    return ctx


def smooth_taubin_op(
    ctx: PipelineContext,
    iterations: int = 100,
    pass_band: float = 0.1,
    feature_smoothing: bool = False,
    boundary_smoothing: bool = True,
) -> PipelineContext:
    """Smooth mesh using Taubin smoothing (less shrinkage than Laplacian).

    Args:
        ctx: Pipeline context with mesh
        iterations: Number of smoothing iterations
        pass_band: Pass band for filter (0-2)
        feature_smoothing: Smooth along features
        boundary_smoothing: Smooth boundary edges

    Returns:
        Updated context with smoothed mesh
    """
    _require_mesh(ctx)

    if iterations < 0:
        raise ConfigurationError("iterations must be non-negative")

    if iterations == 0:
        return ctx

    ctx.mesh = ctx.mesh.smooth_taubin(
        iterations=iterations,
        pass_band=pass_band,
        feature_smoothing=feature_smoothing,
        boundary_smoothing=boundary_smoothing,
    )
    ctx.neighbors = None

    return ctx


def smooth_boundary_op(
    ctx: PipelineContext,
    iterations: int = 20,
    sigma: float = 0.1,
) -> PipelineContext:
    """Smooth only the boundary edges of the mesh.

    Args:
        ctx: Pipeline context with mesh
        iterations: Number of smoothing iterations
        sigma: Smoothing sigma

    Returns:
        Updated context with boundary-smoothed mesh
    """
    _require_mesh(ctx)

    if iterations <= 0:
        return ctx

    ctx.mesh = ctx.mesh.smooth_boundary(iterations=iterations, sigma=sigma)
    ctx.neighbors = None

    return ctx


def remesh_op(
    ctx: PipelineContext,
    n_clusters: int = 10000,
    subdivisions: int = 3,
) -> PipelineContext:
    """Regularize mesh faces using ACVD algorithm.

    Args:
        ctx: Pipeline context with mesh
        n_clusters: Target number of faces
        subdivisions: Number of subdivisions for clustering

    Returns:
        Updated context with remeshed mesh
    """
    _require_mesh(ctx)

    if n_clusters <= 0:
        raise ConfigurationError("n_clusters must be positive")

    ctx.mesh = ctx.mesh.remesh(n=n_clusters, sub=subdivisions)
    ctx.neighbors = None

    return ctx


def decimate_op(
    ctx: PipelineContext,
    target_reduction: float = 0.5,
    volume_preservation: bool = True,
) -> PipelineContext:
    """Reduce mesh complexity by removing faces.

    Args:
        ctx: Pipeline context with mesh
        target_reduction: Fraction of faces to remove (0-1)
        volume_preservation: Preserve mesh volume

    Returns:
        Updated context with decimated mesh
    """
    _require_mesh(ctx)

    if not 0 <= target_reduction < 1:
        raise ConfigurationError("target_reduction must be in [0, 1)")

    ctx.mesh = ctx.mesh.decimate(
        target_reduction=target_reduction,
        volume_preservation=volume_preservation,
    )
    ctx.neighbors = None

    return ctx


def subdivide_op(
    ctx: PipelineContext,
    n_subdivisions: int = 1,
    subfilter: str = "linear",
) -> PipelineContext:
    """Increase mesh resolution by subdividing faces.

    Args:
        ctx: Pipeline context with mesh
        n_subdivisions: Number of subdivision iterations
        subfilter: Subdivision filter ('linear', 'butterfly', 'loop')

    Returns:
        Updated context with subdivided mesh
    """
    _require_mesh(ctx)

    if n_subdivisions <= 0:
        return ctx

    ctx.mesh = ctx.mesh.subdivide(n_subdivisions=n_subdivisions, subfilter=subfilter)
    ctx.neighbors = None

    return ctx


def repair_holes_op(
    ctx: PipelineContext,
    max_hole_edges: int = 100,
    refine: bool = True,
) -> PipelineContext:
    """Fill small holes in the mesh.

    Args:
        ctx: Pipeline context with mesh
        max_hole_edges: Maximum hole size to fill (in edges)
        refine: Refine the filled region

    Returns:
        Updated context with repaired mesh
    """
    _require_mesh(ctx)

    ctx.mesh = ctx.mesh.repair_small(nbe=max_hole_edges, refine=refine)
    ctx.neighbors = None

    return ctx


def repair_op(ctx: PipelineContext) -> PipelineContext:
    """Full mesh repair using MeshFix.

    Args:
        ctx: Pipeline context with mesh

    Returns:
        Updated context with repaired mesh
    """
    _require_mesh(ctx)

    ctx.mesh = ctx.mesh.repair()
    ctx.neighbors = None

    return ctx


def make_manifold_op(
    ctx: PipelineContext,
    hole_edges: int = 300,
) -> PipelineContext:
    """Remove non-manifold edges from mesh.

    Args:
        ctx: Pipeline context with mesh
        hole_edges: Size of holes to fill after removal

    Returns:
        Updated context with manifold mesh
    """
    _require_mesh(ctx)

    ctx.mesh = ctx.mesh.make_manifold(hole_edges=hole_edges)
    ctx.neighbors = None

    return ctx


def filter_curvature_op(
    ctx: PipelineContext,
    threshold: float | list[float] = 0.4,
) -> PipelineContext:
    """Remove vertices outside curvature threshold range.

    Args:
        ctx: Pipeline context with mesh
        threshold: Single value for symmetric range or [min, max] tuple

    Returns:
        Updated context with filtered mesh
    """
    _require_mesh(ctx)

    threshold_tuple = (threshold[0], threshold[1]) if isinstance(threshold, list) else (-threshold, threshold)

    ctx.mesh = ctx.mesh.filter_curvature(curvature_threshold=threshold_tuple)
    ctx.neighbors = None

    return ctx


def remove_normals_op(
    ctx: PipelineContext,
    threshold_angle: float = 60.0,
    flip: bool = False,
    angle_type: str = "polar",
) -> PipelineContext:
    """Remove vertices based on normal angle.

    Args:
        ctx: Pipeline context with mesh
        threshold_angle: Angle threshold in degrees
        flip: Flip normal orientation before filtering
        angle_type: Type of angle ('polar' or 'azimuth')

    Returns:
        Updated context with filtered mesh
    """
    _require_mesh(ctx)

    ctx.mesh = ctx.mesh.remove_normals(
        threshold_angle=threshold_angle,
        flip=flip,
        angle=angle_type,
    )
    ctx.neighbors = None

    return ctx


def remove_bridges_op(ctx: PipelineContext) -> PipelineContext:
    """Remove triangles where all vertices are on the boundary.

    Args:
        ctx: Pipeline context with mesh

    Returns:
        Updated context with bridges removed
    """
    _require_mesh(ctx)

    ctx.mesh = ctx.mesh.remove_bridges(verbose=False)
    ctx.neighbors = None

    return ctx


def remove_tongues_op(
    ctx: PipelineContext,
    radius: float = 50.0,
    threshold: float = 6.0,
    hole_edges: int = 100,
) -> PipelineContext:
    """Remove tongue-like artifacts from mesh.

    Args:
        ctx: Pipeline context with mesh
        radius: Radius for boundary point neighborhood
        threshold: Threshold for boundary/euclidean distance ratio
        hole_edges: Size of holes to fill after removal

    Returns:
        Updated context with tongues removed
    """
    _require_mesh(ctx)

    ctx.mesh = ctx.mesh.remove_tongues(
        radius=radius,
        threshold=threshold,
        hole_edges=hole_edges,
        verbose=False,
    )
    ctx.neighbors = None

    return ctx


def extract_largest_op(ctx: PipelineContext) -> PipelineContext:
    """Keep only the largest connected component.

    Args:
        ctx: Pipeline context with mesh

    Returns:
        Updated context with largest component only
    """
    _require_mesh(ctx)

    ctx.mesh = ctx.mesh.extract_largest()
    ctx.neighbors = None

    return ctx


def clean_op(
    ctx: PipelineContext,
    tolerance: float | None = None,
) -> PipelineContext:
    """Clean mesh by removing degenerate cells.

    Args:
        ctx: Pipeline context with mesh
        tolerance: Tolerance for point merging

    Returns:
        Updated context with cleaned mesh
    """
    _require_mesh(ctx)

    ctx.mesh = ctx.mesh.clean(tolerance=tolerance)
    ctx.neighbors = None

    return ctx


def triangulate_op(ctx: PipelineContext) -> PipelineContext:
    """Convert all faces to triangles.

    Args:
        ctx: Pipeline context with mesh

    Returns:
        Updated context with triangulated mesh
    """
    _require_mesh(ctx)

    ctx.mesh = ctx.mesh.triangulate()
    ctx.neighbors = None

    return ctx


def compute_normals_op(
    ctx: PipelineContext,
    flip: bool = False,
    consistent: bool = True,
    auto_orient: bool = False,
) -> PipelineContext:
    """Compute surface normals.

    Args:
        ctx: Pipeline context with mesh
        flip: Flip all normals
        consistent: Make normals consistent
        auto_orient: Orient normals outward

    Returns:
        Updated context with computed normals
    """
    _require_mesh(ctx)

    ctx.mesh = ctx.mesh.compute_normals(
        flip_normals=flip,
        consistent_normals=consistent,
        auto_orient_normals=auto_orient,
    )

    return ctx


def flip_normals_op(ctx: PipelineContext) -> PipelineContext:
    """Flip all surface normals.

    Args:
        ctx: Pipeline context with mesh

    Returns:
        Updated context with flipped normals
    """
    _require_mesh(ctx)

    ctx.mesh = ctx.mesh.flip_normals()

    return ctx


def rotate_op(
    ctx: PipelineContext,
    axis: str = "x",
    angle: float = 0.0,
) -> PipelineContext:
    """Rotate mesh around an axis.

    Args:
        ctx: Pipeline context with mesh
        axis: Axis to rotate around ('x', 'y', 'z')
        angle: Rotation angle in degrees

    Returns:
        Updated context with rotated mesh
    """
    _require_mesh(ctx)

    if angle == 0:
        return ctx

    if axis.lower() == "x":
        ctx.mesh = ctx.mesh.rotate_x(angle)
    elif axis.lower() == "y":
        ctx.mesh = ctx.mesh.rotate_y(angle)
    elif axis.lower() == "z":
        ctx.mesh = ctx.mesh.rotate_z(angle)
    else:
        raise ConfigurationError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")

    return ctx


def clip_op(
    ctx: PipelineContext,
    normal: str | list[float] = "x",
    origin: list[float] | None = None,
    invert: bool = True,
) -> PipelineContext:
    """Clip mesh with a plane.

    Args:
        ctx: Pipeline context with mesh
        normal: Plane normal ('x', 'y', 'z', '-x', '-y', '-z') or [nx, ny, nz]
        origin: Point on clipping plane
        invert: Invert clipping direction

    Returns:
        Updated context with clipped mesh
    """
    _require_mesh(ctx)

    ctx.mesh = ctx.mesh.clip(normal=normal, origin=origin, invert=invert)
    ctx.neighbors = None

    return ctx


def erode_op(
    ctx: PipelineContext,
    iterations: int = 1,
) -> PipelineContext:
    """Erode mesh by removing boundary points.

    Args:
        ctx: Pipeline context with mesh
        iterations: Number of erosion iterations

    Returns:
        Updated context with eroded mesh
    """
    _require_mesh(ctx)

    if iterations <= 0:
        return ctx

    ctx.mesh = ctx.mesh.erode(iterations=iterations)
    ctx.neighbors = None

    return ctx


def ecft_op(
    ctx: PipelineContext,
    hole_edges: int = 300,
) -> PipelineContext:
    """ExtractLargest, Clean, FillHoles, Triangulate.

    Args:
        ctx: Pipeline context with mesh
        hole_edges: Size of holes to fill

    Returns:
        Updated context with processed mesh
    """
    _require_mesh(ctx)

    ctx.mesh = ctx.mesh.ecft(hole_edges=hole_edges)
    ctx.neighbors = None

    return ctx


def correct_normal_orientation_op(
    ctx: PipelineContext,
    relative: str = "x",
) -> PipelineContext:
    """Correct normal orientation relative to an axis.

    Args:
        ctx: Pipeline context with mesh
        relative: Axis to use for orientation ('x', 'y', 'z')

    Returns:
        Updated context with corrected normals
    """
    _require_mesh(ctx)

    result = ctx.mesh.correct_normal_orientation(relative=relative, inplace=False)
    if result is not None:
        ctx.mesh = result

    return ctx


# =============================================================================
# Domain Operations
# =============================================================================


def compute_curvature_op(
    ctx: PipelineContext,
    curvature_type: str = "mean",
) -> PipelineContext:
    """Compute mesh curvature.

    Args:
        ctx: Pipeline context with mesh
        curvature_type: Type of curvature ('mean', 'gaussian', 'minimum', 'maximum')

    Returns:
        Updated context with curvature array
    """
    _require_mesh(ctx)

    valid_types = ["mean", "gaussian", "minimum", "maximum"]
    if curvature_type not in valid_types:
        raise ConfigurationError(f"curvature_type must be one of {valid_types}")

    ctx.curvature = ctx.mesh.curvature(curv_type=curvature_type)
    ctx.mesh["curvature"] = ctx.curvature

    return ctx


def filter_scalars_op(
    ctx: PipelineContext,
    function: str = "median",
    iterations: int = 1,
) -> PipelineContext:
    """Apply filter to scalar field (curvature).

    Args:
        ctx: Pipeline context with curvature
        function: Filter function ('median', 'mean', 'min', 'max')
        iterations: Number of filter iterations

    Returns:
        Updated context with filtered curvature
    """
    _require_mesh(ctx)

    if ctx.curvature is None:
        raise ConfigurationError("filter_scalars requires curvature in context")

    from phenotastic import domains

    # Ensure neighbors are computed
    if ctx.neighbors is None:
        ctx.neighbors = ctx.mesh.vertex_neighbors_all(include_self=True)

    func_map = {
        "median": domains.median,
        "mean": domains.mean,
        "minmax": domains.minmax,
        "maxmin": domains.maxmin,
    }

    if function not in func_map:
        raise ConfigurationError(f"function must be one of {list(func_map.keys())}")

    ctx.curvature = func_map[function](ctx.curvature, ctx.neighbors, iterations)
    ctx.mesh["curvature"] = ctx.curvature

    return ctx


def segment_domains_op(
    ctx: PipelineContext,
    curvature_type: str | None = None,
) -> PipelineContext:
    """Create domains via steepest ascent on curvature field.

    Args:
        ctx: Pipeline context with mesh and curvature
        curvature_type: Curvature type to compute (if not already in context)

    Returns:
        Updated context with domain labels
    """
    _require_mesh(ctx)

    from phenotastic import domains

    # Compute curvature if needed
    if ctx.curvature is None:
        if curvature_type is None:
            curvature_type = "mean"
        ctx.curvature = ctx.mesh.curvature(curv_type=curvature_type)
        ctx.mesh["curvature"] = ctx.curvature

    # Ensure neighbors are computed
    if ctx.neighbors is None:
        ctx.neighbors = ctx.mesh.vertex_neighbors_all(include_self=True)

    ctx.domains = domains.steepest_ascent(
        ctx.mesh.to_polydata(),
        ctx.curvature,
        neighbours=ctx.neighbors,
    )
    ctx.mesh["domains"] = ctx.domains

    return ctx


def merge_angles_op(
    ctx: PipelineContext,
    threshold: float = 20.0,
    meristem_method: str = "center_of_mass",
) -> PipelineContext:
    """Merge domains within angular threshold from meristem.

    Args:
        ctx: Pipeline context with domains
        threshold: Angular threshold in degrees
        meristem_method: Method for calculating meristem center

    Returns:
        Updated context with merged domains
    """
    _require_mesh(ctx)

    if ctx.domains is None:
        raise ConfigurationError("merge_angles requires domains in context")

    from phenotastic import domains

    # Find meristem if not set
    if ctx.meristem_index is None:
        ctx.meristem_index = domains.define_meristem(
            ctx.mesh.to_polydata(),
            ctx.domains,
            method=meristem_method,
        )
        if isinstance(ctx.meristem_index, tuple):
            ctx.meristem_index = ctx.meristem_index[0]

    ctx.domains = domains.merge_angles(
        ctx.mesh.to_polydata(),
        ctx.domains,
        ctx.meristem_index,
        threshold=threshold,
        method=meristem_method,
    )
    ctx.mesh["domains"] = ctx.domains

    return ctx


def merge_distance_op(
    ctx: PipelineContext,
    threshold: float = 50.0,
    metric: str = "euclidean",
    method: str = "center_of_mass",
) -> PipelineContext:
    """Merge domains within spatial distance threshold.

    Args:
        ctx: Pipeline context with domains
        threshold: Distance threshold
        metric: Distance metric ('euclidean' or 'geodesic')
        method: Method for calculating domain center

    Returns:
        Updated context with merged domains
    """
    _require_mesh(ctx)

    if ctx.domains is None:
        raise ConfigurationError("merge_distance requires domains in context")

    from phenotastic import domains

    ctx.domains = domains.merge_distance(
        ctx.mesh.to_polydata(),
        ctx.domains,
        threshold=threshold,
        scalars=ctx.curvature,
        method=method,
        metric=metric,
    )
    ctx.mesh["domains"] = ctx.domains

    return ctx


def merge_small_op(
    ctx: PipelineContext,
    threshold: int = 100,
    metric: str = "points",
    mode: str = "border",
) -> PipelineContext:
    """Merge small domains to their largest neighbor.

    Args:
        ctx: Pipeline context with domains
        threshold: Size threshold for merging
        metric: Size metric ('points' or 'area')
        mode: Merge strategy ('border' or 'area')

    Returns:
        Updated context with merged domains
    """
    _require_mesh(ctx)

    if ctx.domains is None:
        raise ConfigurationError("merge_small requires domains in context")

    from phenotastic import domains

    if ctx.neighbors is None:
        ctx.neighbors = ctx.mesh.vertex_neighbors_all(include_self=True)

    ctx.domains = domains.merge_small(
        ctx.mesh.to_polydata(),
        ctx.domains,
        threshold=threshold,
        metric=metric,
        mode=mode,
        neighbours=ctx.neighbors,
    )
    ctx.mesh["domains"] = ctx.domains

    return ctx


def merge_engulfing_op(
    ctx: PipelineContext,
    threshold: float = 0.9,
) -> PipelineContext:
    """Merge domains mostly encircled by a neighbor.

    Args:
        ctx: Pipeline context with domains
        threshold: Fraction of boundary that must be shared (0-1)

    Returns:
        Updated context with merged domains
    """
    _require_mesh(ctx)

    if ctx.domains is None:
        raise ConfigurationError("merge_engulfing requires domains in context")

    from phenotastic import domains

    if ctx.neighbors is None:
        ctx.neighbors = ctx.mesh.vertex_neighbors_all(include_self=True)

    ctx.domains = domains.merge_engulfing(
        ctx.mesh.to_polydata(),
        ctx.domains,
        threshold=threshold,
        neighbours=ctx.neighbors,
    )
    ctx.mesh["domains"] = ctx.domains

    return ctx


def merge_disconnected_op(
    ctx: PipelineContext,
    meristem_method: str = "center_of_mass",
) -> PipelineContext:
    """Connect domains isolated from meristem.

    Args:
        ctx: Pipeline context with domains
        meristem_method: Method for identifying meristem

    Returns:
        Updated context with connected domains
    """
    _require_mesh(ctx)

    if ctx.domains is None:
        raise ConfigurationError("merge_disconnected requires domains in context")

    from phenotastic import domains

    if ctx.neighbors is None:
        ctx.neighbors = ctx.mesh.vertex_neighbors_all(include_self=True)

    if ctx.meristem_index is None:
        ctx.meristem_index = domains.define_meristem(
            ctx.mesh.to_polydata(),
            ctx.domains,
            method=meristem_method,
        )
        if isinstance(ctx.meristem_index, tuple):
            ctx.meristem_index = ctx.meristem_index[0]

    ctx.domains = domains.merge_disconnected(
        ctx.mesh.to_polydata(),
        ctx.domains,
        ctx.meristem_index,
        threshold=None,
        neighbours=ctx.neighbors,
    )
    ctx.mesh["domains"] = ctx.domains

    return ctx


def merge_depth_op(
    ctx: PipelineContext,
    threshold: float = 0.0,
    mode: str = "max",
    exclude_boundary: bool = False,
    min_points: int = 0,
) -> PipelineContext:
    """Merge domains with similar depth values.

    Args:
        ctx: Pipeline context with domains and curvature
        threshold: Maximum depth difference for merging
        mode: Aggregation mode ('min', 'max', 'median', 'mean')
        exclude_boundary: Exclude boundary vertices from calculation
        min_points: Minimum border points required for merging

    Returns:
        Updated context with merged domains
    """
    _require_mesh(ctx)

    if ctx.domains is None:
        raise ConfigurationError("merge_depth requires domains in context")
    if ctx.curvature is None:
        raise ConfigurationError("merge_depth requires curvature in context")

    from phenotastic import domains

    if ctx.neighbors is None:
        ctx.neighbors = ctx.mesh.vertex_neighbors_all(include_self=True)

    ctx.domains = domains.merge_depth(
        ctx.mesh.to_polydata(),
        ctx.domains,
        ctx.curvature,
        threshold=threshold,
        neighbours=ctx.neighbors,
        exclude_boundary=exclude_boundary,
        min_points=min_points,
        mode=mode,
    )
    ctx.mesh["domains"] = ctx.domains

    return ctx


def define_meristem_op(
    ctx: PipelineContext,
    method: str = "center_of_mass",
) -> PipelineContext:
    """Identify the meristem domain.

    Args:
        ctx: Pipeline context with domains
        method: Method for meristem identification

    Returns:
        Updated context with meristem index
    """
    _require_mesh(ctx)

    if ctx.domains is None:
        raise ConfigurationError("define_meristem requires domains in context")

    from phenotastic import domains

    if ctx.neighbors is None:
        ctx.neighbors = ctx.mesh.vertex_neighbors_all(include_self=True)

    result = domains.define_meristem(
        ctx.mesh.to_polydata(),
        ctx.domains,
        method=method,
        neighs=ctx.neighbors,
    )
    if isinstance(result, tuple):
        ctx.meristem_index = result[0]
    else:
        ctx.meristem_index = result

    return ctx


def extract_domaindata_op(ctx: PipelineContext) -> PipelineContext:
    """Extract geometric measurements for each domain.

    Args:
        ctx: Pipeline context with mesh and domains

    Returns:
        Updated context with domain_data DataFrame
    """
    _require_mesh(ctx)

    if ctx.domains is None:
        raise ConfigurationError("extract_domaindata requires domains in context")

    import pandas as pd

    from phenotastic import domains

    # Create point data frame
    pdata = pd.DataFrame({"domain": ctx.domains})

    # Get apex/meristem coordinates
    if ctx.meristem_index is None:
        result = domains.define_meristem(
            ctx.mesh.to_polydata(),
            ctx.domains,
            return_coordinates=True,
        )
        if isinstance(result, tuple):
            ctx.meristem_index = result[0]
            apex = result[1]
        else:
            ctx.meristem_index = result
            apex = np.array(ctx.mesh.center_of_mass())
    else:
        apex = np.array(ctx.mesh.center_of_mass())

    ctx.domain_data = domains.extract_domaindata(
        pdata,
        ctx.mesh.to_polydata(),
        apex,
        ctx.meristem_index,
    )

    return ctx


# =============================================================================
# Operation Registry
# =============================================================================

# Map of operation names to functions
OPERATIONS: dict[str, Any] = {
    # Image/Contour
    "contour": contour_op,
    "create_mesh": create_mesh_op,
    "create_cellular_mesh": create_cellular_mesh_op,
    # Mesh Processing
    "smooth": smooth_op,
    "smooth_taubin": smooth_taubin_op,
    "smooth_boundary": smooth_boundary_op,
    "remesh": remesh_op,
    "decimate": decimate_op,
    "subdivide": subdivide_op,
    "repair_holes": repair_holes_op,
    "repair": repair_op,
    "make_manifold": make_manifold_op,
    "filter_curvature": filter_curvature_op,
    "remove_normals": remove_normals_op,
    "remove_bridges": remove_bridges_op,
    "remove_tongues": remove_tongues_op,
    "extract_largest": extract_largest_op,
    "clean": clean_op,
    "triangulate": triangulate_op,
    "compute_normals": compute_normals_op,
    "flip_normals": flip_normals_op,
    "rotate": rotate_op,
    "clip": clip_op,
    "erode": erode_op,
    "ecft": ecft_op,
    "correct_normal_orientation": correct_normal_orientation_op,
    # Domain Operations
    "compute_curvature": compute_curvature_op,
    "filter_scalars": filter_scalars_op,
    "segment_domains": segment_domains_op,
    "merge_angles": merge_angles_op,
    "merge_distance": merge_distance_op,
    "merge_small": merge_small_op,
    "merge_engulfing": merge_engulfing_op,
    "merge_disconnected": merge_disconnected_op,
    "merge_depth": merge_depth_op,
    "define_meristem": define_meristem_op,
    "extract_domaindata": extract_domaindata_op,
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
    # Add more as needed...
}
