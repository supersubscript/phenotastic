"""PhenoMesh: Extension of PyVista PolyData with phenotyping-specific operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
import pyvista as pv

from phenotastic.mesh import (
    correct_bad_mesh,
    correct_normal_orientation,
    define_meristem,
    drop_skirt,
    erode,
    extract_clean_fill_triangulate,
    filter_by_curvature,
    fit_paraboloid_mesh,
    get_boundary_edges,
    get_boundary_points,
    get_feature_edges,
    get_manifold_edges,
    get_non_manifold_edges,
    get_vertex_cycles,
    get_vertex_neighbors,
    get_vertex_neighbors_all,
    label_from_image,
    make_manifold,
    process,
    project_to_surface,
    remesh,
    remesh_decimate,
    remove_bridges,
    remove_by_normals,
    remove_inland_under,
    remove_tongues,
    repair,
    repair_small_holes,
    smooth_boundary,
)
from phenotastic.pipeline_decorator import pipeline_operation

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from numpy.typing import NDArray


class PhenoMesh(pv.PolyData):
    """Extension of PyVista PolyData with phenotyping-specific operations.

    PhenoMesh inherits from pv.PolyData, so it can be used anywhere a PolyData is expected.
    It adds convenience methods for 3D plant phenotyping workflows including smoothing,
    repair, curvature analysis, and domain segmentation.

    Args:
        var_inp: Input data - can be a PolyData, file path, or vertex array
        *args: Additional positional arguments passed to pv.PolyData
        contour: Optional binary contour array the mesh was generated from
        resolution: Optional spatial resolution [z, y, x]
        **kwargs: Additional keyword arguments passed to pv.PolyData

    Example:
        >>> mesh = PhenoMesh(pv.Sphere())
        >>> smoothed = mesh.smooth(iterations=100)
        >>> isinstance(mesh, pv.PolyData)  # True - PhenoMesh is a PolyData
        True
    """

    _phenotastic_contour: NDArray[np.bool_] | None
    _phenotastic_resolution: list[float] | None

    def __init__(
        self,
        var_inp: pv.PolyData | str | Path | NDArray[np.floating[Any]] | None = None,
        *args: Any,
        contour: NDArray[np.bool_] | None = None,
        resolution: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize PhenoMesh with optional PolyData, file, or vertex array."""
        super().__init__(var_inp, *args, **kwargs)  # type: ignore[arg-type]
        self._phenotastic_contour = contour
        self._phenotastic_resolution = resolution

    # =========================================================================
    # Custom attributes
    # =========================================================================

    @property  # type: ignore[override]
    def contour(self) -> NDArray[np.bool_] | None:
        """Binary contour array the mesh was generated from."""
        return self._phenotastic_contour

    @contour.setter
    def contour(self, value: NDArray[np.bool_] | None) -> None:
        """Set contour array."""
        self._phenotastic_contour = value

    @property
    def resolution(self) -> list[float] | None:
        """Spatial resolution [z, y, x]."""
        return self._phenotastic_resolution

    @resolution.setter
    def resolution(self, value: list[float] | None) -> None:
        """Set resolution."""
        self._phenotastic_resolution = value

    # =========================================================================
    # Result wrapping helper
    # =========================================================================

    def _wrap_result(self, polydata: pv.PolyData) -> PhenoMesh:
        """Wrap a PolyData result in a PhenoMesh, preserving custom attributes."""
        return PhenoMesh(polydata, contour=self._phenotastic_contour, resolution=self._phenotastic_resolution)

    # =========================================================================
    # Conversion methods
    # =========================================================================

    @classmethod
    def from_polydata(
        cls,
        polydata: pv.PolyData,
        contour: NDArray[np.bool_] | None = None,
        resolution: list[float] | None = None,
    ) -> PhenoMesh:
        """Create PhenoMesh from PyVista PolyData.

        Args:
            polydata: PyVista PolyData mesh
            contour: Optional binary contour array
            resolution: Optional spatial resolution

        Returns:
            New PhenoMesh instance
        """
        return cls(polydata, contour=contour, resolution=resolution)

    def to_polydata(self) -> pv.PolyData:
        """Return a plain PolyData copy (without PhenoMesh attributes).

        Returns:
            A new pv.PolyData copy of this mesh
        """
        return pv.PolyData(self)

    def copy(self, deep: bool = True) -> PhenoMesh:
        """Create a copy of this PhenoMesh.

        Args:
            deep: If True, create a deep copy

        Returns:
            New PhenoMesh with copied data
        """
        result = super().copy(deep=deep)
        return PhenoMesh(result, contour=self._phenotastic_contour, resolution=self._phenotastic_resolution)

    # =========================================================================
    # Additional properties
    # =========================================================================

    @property
    def n_faces(self) -> int:
        """Number of faces in the mesh (alias for n_cells)."""
        return int(self.n_cells)

    # =========================================================================
    # Smoothing operations
    # =========================================================================

    @pipeline_operation(
        name="smooth",
        validators={
            "iterations": lambda x: x >= 0,
            "relaxation_factor": lambda x: 0 <= x <= 1,
        },
    )
    def smooth(
        self,
        iterations: int = 100,
        relaxation_factor: float = 0.01,
        feature_smoothing: bool = False,
        boundary_smoothing: bool = True,
        edge_angle: float = 15.0,
        feature_angle: float = 45.0,
    ) -> PhenoMesh:
        """Smooth the mesh using Laplacian smoothing.

        Args:
            iterations: Number of smoothing iterations
            relaxation_factor: Relaxation factor for smoothing (0-1)
            feature_smoothing: Smooth along features
            boundary_smoothing: Smooth boundary edges
            edge_angle: Angle for edge detection
            feature_angle: Angle for feature detection

        Returns:
            New PhenoMesh with smoothed surface
        """
        result = super().smooth(
            n_iter=iterations,
            relaxation_factor=relaxation_factor,
            feature_smoothing=feature_smoothing,
            boundary_smoothing=boundary_smoothing,
            edge_angle=edge_angle,
            feature_angle=feature_angle,
        )
        return self._wrap_result(result)

    @pipeline_operation(
        name="smooth_taubin",
        validators={"iterations": lambda x: x >= 0},
    )
    def smooth_taubin(
        self,
        iterations: int = 100,
        pass_band: float = 0.1,
        edge_angle: float = 15.0,
        feature_angle: float = 45.0,
        feature_smoothing: bool = False,
        boundary_smoothing: bool = True,
        non_manifold_smoothing: bool = True,
        normalize_coordinates: bool = True,
    ) -> PhenoMesh:
        """Smooth the mesh using Taubin smoothing (low-pass filter).

        Taubin smoothing is less prone to shrinkage than Laplacian smoothing.

        Args:
            iterations: Number of smoothing iterations
            pass_band: Pass band for the filter (0-2)
            edge_angle: Angle for edge detection
            feature_angle: Angle for feature detection
            feature_smoothing: Smooth along features
            boundary_smoothing: Smooth boundary edges
            non_manifold_smoothing: Smooth non-manifold edges
            normalize_coordinates: Normalize coordinates before smoothing

        Returns:
            New PhenoMesh with smoothed surface
        """
        result = super().smooth_taubin(
            n_iter=iterations,
            pass_band=pass_band,
            edge_angle=edge_angle,
            feature_angle=feature_angle,
            feature_smoothing=feature_smoothing,
            boundary_smoothing=boundary_smoothing,
            non_manifold_smoothing=non_manifold_smoothing,
            normalize_coordinates=normalize_coordinates,
        )
        return self._wrap_result(result)

    @pipeline_operation(name="smooth_boundary")
    def smooth_boundary(self, iterations: int = 20, sigma: float = 0.1) -> PhenoMesh:
        """Smooth the boundary of the mesh using Laplacian smoothing.

        Args:
            iterations: Number of smoothing iterations
            sigma: Smoothing sigma

        Returns:
            Smoothed PhenoMesh
        """
        result = smooth_boundary(self, iterations, sigma, inplace=False)
        return self._wrap_result(result) if result is not None else self.copy()

    # =========================================================================
    # Mesh simplification and refinement
    # =========================================================================

    @pipeline_operation(
        name="decimate",
        validators={"target_reduction": lambda x: 0 <= x < 1},
    )
    def decimate(self, target_reduction: float = 0.5, volume_preservation: bool = True) -> PhenoMesh:
        """Reduce mesh complexity by decimating faces.

        Args:
            target_reduction: Fraction of faces to remove (0-1)
            volume_preservation: Preserve mesh volume during decimation

        Returns:
            New PhenoMesh with reduced complexity
        """
        result = super().decimate(
            target_reduction=target_reduction,
            volume_preservation=volume_preservation,
        )
        return self._wrap_result(result)

    @pipeline_operation(
        name="subdivide",
        validators={"n_subdivisions": lambda x: x >= 0},
    )
    def subdivide(self, n_subdivisions: int = 1, subfilter: str = "linear") -> PhenoMesh:
        """Subdivide mesh faces to increase resolution.

        Args:
            n_subdivisions: Number of subdivision iterations
            subfilter: Subdivision filter ('linear', 'butterfly', 'loop')

        Returns:
            New PhenoMesh with subdivided faces
        """
        result = super().subdivide(n_subdivisions, subfilter)
        return self._wrap_result(result)

    @pipeline_operation(
        name="remesh",
        validators={"n_clusters": lambda x: x > 0},
    )
    def remesh(self, n_clusters: int, subdivisions: int = 3) -> PhenoMesh:
        """Regularise the mesh faces using the ACVD algorithm.

        Args:
            n_clusters: The number of clusters (i.e. output number of faces)
            subdivisions: The number of subdivisions to use when clustering

        Returns:
            Regularised PhenoMesh
        """
        return self._wrap_result(remesh(self, n_clusters, subdivisions))

    def remesh_decimate(
        self,
        iterations: int,
        upscale_factor: float = 2,
        downscale_factor: float = 0.5,
        verbose: bool = True,
    ) -> PhenoMesh:
        """Iterative remeshing/decimation.

        Args:
            iterations: Number of iterations
            upscale_factor: Factor with which to upsample
            downscale_factor: Factor with which to downsample
            verbose: Print operation steps

        Returns:
            Processed PhenoMesh
        """
        return self._wrap_result(remesh_decimate(self, iterations, upscale_factor, downscale_factor, verbose))

    # =========================================================================
    # Mesh cleanup and repair
    # =========================================================================

    @pipeline_operation(name="triangulate")
    def triangulate(self) -> PhenoMesh:
        """Convert all faces to triangles.

        Returns:
            New PhenoMesh with triangulated faces
        """
        result = super().triangulate()
        return self._wrap_result(result)

    @pipeline_operation(name="clean")
    def clean(
        self,
        point_merging: bool = True,
        tolerance: float | None = None,
        lines_to_points: bool = True,
        polys_to_lines: bool = True,
        strips_to_polys: bool = True,
        absolute: bool = True,
    ) -> PhenoMesh:
        """Clean mesh by removing degenerate cells and merging duplicate points.

        Args:
            point_merging: Merge coincident points
            tolerance: Tolerance for point merging
            lines_to_points: Convert degenerate lines to points
            polys_to_lines: Convert degenerate polygons to lines
            strips_to_polys: Convert strips to polygons
            absolute: Use absolute tolerance

        Returns:
            Cleaned PhenoMesh
        """
        result = super().clean(
            point_merging=point_merging,
            tolerance=tolerance,
            lines_to_points=lines_to_points,
            polys_to_lines=polys_to_lines,
            strips_to_polys=strips_to_polys,
            absolute=absolute,
        )
        return self._wrap_result(result)

    @pipeline_operation(name="extract_largest")
    def extract_largest(self) -> PhenoMesh:
        """Extract the largest connected component.

        Returns:
            New PhenoMesh containing only the largest connected region
        """
        result = super().extract_largest()
        return self._wrap_result(result)

    @pipeline_operation(name="fill_holes")
    def fill_holes(self, hole_size: float = 1000.0) -> PhenoMesh:
        """Fill holes in the mesh.

        Args:
            hole_size: Maximum hole size to fill

        Returns:
            New PhenoMesh with holes filled
        """
        result = super().fill_holes(hole_size)
        return self._wrap_result(result)

    @pipeline_operation(name="repair")
    def repair(self) -> PhenoMesh:
        """Repair the mesh using MeshFix.

        Returns:
            Repaired PhenoMesh
        """
        return self._wrap_result(repair(self))

    @pipeline_operation(name="repair_holes")
    def repair_small_holes(self, max_hole_edges: int | None = 100, refine: bool = True) -> PhenoMesh:
        """Repair small holes in the mesh based on the number of edges.

        Args:
            max_hole_edges: Maximum number of edges for holes to repair
            refine: Refine the mesh after repair

        Returns:
            Repaired PhenoMesh
        """
        return self._wrap_result(repair_small_holes(self, max_hole_edges, refine))

    @pipeline_operation(name="make_manifold")
    def make_manifold(self, hole_edges: int = 300) -> PhenoMesh:
        """Make the mesh manifold by removing non-manifold edges.

        Args:
            hole_edges: Size of holes to fill

        Returns:
            Manifold PhenoMesh
        """
        return self._wrap_result(make_manifold(self, hole_edges))

    def correct_bad_mesh(self, verbose: bool = True) -> PhenoMesh:
        """Correct a bad (non-manifold) mesh.

        Args:
            verbose: Print processing steps

        Returns:
            Corrected PhenoMesh
        """
        return self._wrap_result(correct_bad_mesh(self, verbose))

    @pipeline_operation(name="extract_clean_fill_triangulate")
    def extract_clean_fill_triangulate(self, hole_edges: int = 300) -> PhenoMesh:
        """Perform ExtractLargest, Clean, FillHoles, and TriFilter operations.

        Args:
            hole_edges: Size of holes to fill

        Returns:
            Processed PhenoMesh
        """
        return self._wrap_result(extract_clean_fill_triangulate(self, hole_edges))

    # Backwards compatibility alias
    ecft = extract_clean_fill_triangulate

    # =========================================================================
    # Normal operations
    # =========================================================================

    @pipeline_operation(name="compute_normals", invalidates_neighbors=False)
    def compute_normals(
        self,
        cell_normals: bool = True,
        point_normals: bool = True,
        flip_normals: bool = False,
        consistent_normals: bool = True,
        auto_orient_normals: bool = False,
        non_manifold_traversal: bool = True,
        feature_angle: float = 30.0,
    ) -> PhenoMesh:
        """Compute surface normals.

        Args:
            cell_normals: Compute cell normals
            point_normals: Compute point normals
            flip_normals: Flip all normals
            consistent_normals: Make normals consistent
            auto_orient_normals: Orient normals outward
            non_manifold_traversal: Allow traversal across non-manifold edges
            feature_angle: Feature angle for splitting

        Returns:
            New PhenoMesh with computed normals
        """
        result = super().compute_normals(
            cell_normals=cell_normals,
            point_normals=point_normals,
            flip_normals=flip_normals,
            consistent_normals=consistent_normals,
            auto_orient_normals=auto_orient_normals,
            non_manifold_traversal=non_manifold_traversal,
            feature_angle=feature_angle,
        )
        return self._wrap_result(result)

    @pipeline_operation(name="flip_normals", invalidates_neighbors=False)
    def flip_normals(self) -> PhenoMesh:
        """Flip all surface normals.

        Returns:
            New PhenoMesh with flipped normals
        """
        result = self.copy()
        super(PhenoMesh, result).flip_normals()  # type: ignore[no-untyped-call]
        return result

    @pipeline_operation(name="correct_normal_orientation", invalidates_neighbors=False)
    def correct_normal_orientation(self, relative: str = "x", inplace: bool = False) -> PhenoMesh | None:
        """Correct the orientation of the normals.

        Args:
            relative: Axis to use for orientation ('x', 'y', 'z')
            inplace: Modify in place

        Returns:
            PhenoMesh with corrected normals, or None if inplace
        """
        if inplace:
            correct_normal_orientation(self, relative, inplace=True)
            return None
        result = correct_normal_orientation(self, relative, inplace=False)
        return self._wrap_result(result) if result is not None else self.copy()

    # =========================================================================
    # Curvature and filtering
    # =========================================================================

    def compute_curvature(self, curvature_type: str = "mean") -> NDArray[np.floating[Any]]:
        """Compute surface curvature.

        Args:
            curvature_type: Curvature type ('mean', 'gaussian', 'minimum', 'maximum')

        Returns:
            Array of curvature values per vertex
        """
        result: NDArray[np.floating[Any]] = self.curvature(curvature_type)
        return result

    @pipeline_operation(name="filter_curvature")
    def filter_by_curvature(
        self,
        curvature_threshold: tuple[float, float] | float,
        curvatures: NDArray[np.floating[Any]] | None = None,
    ) -> PhenoMesh:
        """Remove mesh vertices outside curvature threshold range.

        Args:
            curvature_threshold: Tuple (min, max) defining valid curvature range,
                or single value for symmetric range
            curvatures: Optional curvature array. If None, computes mean curvature

        Returns:
            Filtered PhenoMesh
        """
        return self._wrap_result(filter_by_curvature(self, curvature_threshold, curvatures))

    # =========================================================================
    # Point and vertex operations
    # =========================================================================

    def remove_points(self, mask: NDArray[np.bool_], keep_scalars: bool = True) -> tuple[PhenoMesh, NDArray[np.intp]]:  # type: ignore[override]
        """Remove points from the mesh.

        Args:
            mask: Boolean mask indicating points to remove (True = remove)
            keep_scalars: Preserve scalar data

        Returns:
            Tuple of (new PhenoMesh with points removed, indices of removed points)
        """
        result = super().remove_points(mask, keep_scalars=keep_scalars)
        return self._wrap_result(result[0]), result[1]

    @pipeline_operation(name="remove_normals")
    def remove_by_normals(
        self,
        threshold_angle: float = 0,
        flip: bool = False,
        angle_type: str = "polar",
    ) -> PhenoMesh:
        """Remove points based on the point normal angle.

        Args:
            threshold_angle: Threshold for the polar angle
            flip: Flip normal orientation
            angle_type: Type of angle to use ('polar' or 'azimuth')

        Returns:
            PhenoMesh with vertices removed
        """
        return self._wrap_result(remove_by_normals(self, threshold_angle, flip, angle_type))

    @pipeline_operation(name="erode")
    def erode(self, iterations: int = 1) -> PhenoMesh:
        """Erode the mesh by removing boundary points iteratively.

        Args:
            iterations: Number of erosion iterations

        Returns:
            Eroded PhenoMesh
        """
        return self._wrap_result(erode(self, iterations))

    # =========================================================================
    # Artifact removal
    # =========================================================================

    @pipeline_operation(name="remove_bridges")
    def remove_bridges(self, verbose: bool = True) -> PhenoMesh:
        """Remove triangles where all vertices are part of the mesh boundary.

        Args:
            verbose: Print processing steps

        Returns:
            PhenoMesh after bridge removal
        """
        return self._wrap_result(remove_bridges(self, verbose))

    @pipeline_operation(name="remove_tongues")
    def remove_tongues(
        self,
        radius: float,
        threshold: float = 6,
        hole_edges: int = 100,
        verbose: bool = True,
    ) -> PhenoMesh:
        """Remove "tongues" in mesh.

        Args:
            radius: Radius for boundary point neighbourhood
            threshold: Threshold for fraction between boundary and euclidean distance
            hole_edges: Size of holes to fill after removal
            verbose: Print processing steps

        Returns:
            PhenoMesh with tongues removed
        """
        return self._wrap_result(remove_tongues(self, radius, threshold, hole_edges, verbose))

    def remove_inland_under(
        self,
        contour: NDArray[np.bool_],
        threshold: int,
        resolution: list[float] | None = None,
        invert: bool = False,
    ) -> PhenoMesh:
        """Remove the part of the mesh that is under the contour.

        Args:
            contour: Contour to use for the removal
            threshold: Threshold distance from the contour XY periphery
            resolution: Resolution of the image
            invert: Invert the mesh normals

        Returns:
            PhenoMesh with the inland part removed
        """
        return self._wrap_result(remove_inland_under(self, contour, threshold, resolution, invert))

    # =========================================================================
    # Geometric operations
    # =========================================================================

    @pipeline_operation(name="clip")
    def clip(
        self,
        normal: str | Sequence[float] = "x",
        origin: Sequence[float] | None = None,
        invert: bool = True,
    ) -> PhenoMesh:
        """Clip mesh with a plane.

        Args:
            normal: Plane normal direction ('x', 'y', 'z', '-x', '-y', '-z')
                or 3-element normal vector
            origin: Point on the clipping plane
            invert: Invert clipping direction

        Returns:
            Clipped PhenoMesh
        """
        result = super().clip(normal=normal, origin=origin, invert=invert)  # type: ignore[arg-type]
        return self._wrap_result(result)

    def rotate_x(self, angle: float, inplace: bool = False) -> Self:  # type: ignore[override]
        """Rotate mesh around X axis.

        Args:
            angle: Rotation angle in degrees
            inplace: Modify in place

        Returns:
            Rotated PhenoMesh (or self if inplace)
        """
        if inplace:
            super().rotate_x(angle, inplace=True)
            return self
        result = self.copy()
        super(PhenoMesh, result).rotate_x(angle, inplace=True)
        return cast("Self", result)

    def rotate_y(self, angle: float, inplace: bool = False) -> Self:  # type: ignore[override]
        """Rotate mesh around Y axis.

        Args:
            angle: Rotation angle in degrees
            inplace: Modify in place

        Returns:
            Rotated PhenoMesh (or self if inplace)
        """
        if inplace:
            super().rotate_y(angle, inplace=True)
            return self
        result = self.copy()
        super(PhenoMesh, result).rotate_y(angle, inplace=True)
        return cast("Self", result)

    def rotate_z(self, angle: float, inplace: bool = False) -> Self:  # type: ignore[override]
        """Rotate mesh around Z axis.

        Args:
            angle: Rotation angle in degrees
            inplace: Modify in place

        Returns:
            Rotated PhenoMesh (or self if inplace)
        """
        if inplace:
            super().rotate_z(angle, inplace=True)
            return self
        result = self.copy()
        super(PhenoMesh, result).rotate_z(angle, inplace=True)
        return cast("Self", result)

    @pipeline_operation(name="drop_skirt")
    def drop_skirt(self, max_distance: float, flip: bool = False) -> PhenoMesh:
        """Downprojects the boundary to the lowest point in the z-direction.

        Args:
            max_distance: Distance in z-direction from the lowest point to consider
            flip: If True, flip the direction

        Returns:
            PhenoMesh with boundary downprojected
        """
        return self._wrap_result(drop_skirt(self, max_distance, flip))

    # =========================================================================
    # Geometric queries
    # =========================================================================

    def compute_center_of_mass(self) -> NDArray[np.floating[Any]]:
        """Compute center of mass.

        Returns:
            3-element array of center of mass coordinates
        """
        return np.array(self.center_of_mass())

    def compute_geodesic_distance(self, start_vertex: int, end_vertex: int) -> float:
        """Compute geodesic distance between two vertices.

        Args:
            start_vertex: Index of start vertex
            end_vertex: Index of end vertex

        Returns:
            Geodesic distance along the mesh surface
        """
        return float(self.geodesic_distance(start_vertex, end_vertex))

    def find_closest_point(self, point: Sequence[float]) -> int:  # type: ignore[override]
        """Find the index of the closest point.

        Args:
            point: Query point coordinates

        Returns:
            Index of the closest point
        """
        return int(self.FindPoint(list(point)))

    def ray_trace(  # type: ignore[override]
        self,
        origin: Sequence[float],
        end_point: Sequence[float],
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.intp]]:
        """Perform ray tracing on the mesh.

        Args:
            origin: Ray origin point
            end_point: Ray end point

        Returns:
            Tuple of (intersection points, cell indices)
        """
        result = super().ray_trace(origin, end_point)
        return (np.asarray(result[0]), np.asarray(result[1]))

    # =========================================================================
    # Boundary and edge queries
    # =========================================================================

    def get_boundary_points(self) -> NDArray[np.intp]:
        """Get vertex indices of points in the boundary.

        Returns:
            Array of boundary vertex indices
        """
        return get_boundary_points(self)

    def get_boundary_edges(self) -> PhenoMesh:
        """Get boundary edges.

        Returns:
            PhenoMesh containing boundary edges
        """
        return self._wrap_result(get_boundary_edges(self))

    def get_non_manifold_edges(self) -> PhenoMesh:
        """Get non-manifold edges.

        Returns:
            PhenoMesh containing non-manifold edges
        """
        return self._wrap_result(get_non_manifold_edges(self))

    def get_manifold_edges(self) -> PhenoMesh:
        """Get manifold edges.

        Returns:
            PhenoMesh containing manifold edges
        """
        return self._wrap_result(get_manifold_edges(self))

    def get_feature_edges(self, angle: float = 30) -> PhenoMesh:
        """Get feature edges defined by given angle.

        Args:
            angle: Feature angle threshold

        Returns:
            PhenoMesh containing feature edges
        """
        return self._wrap_result(get_feature_edges(self, angle))

    # =========================================================================
    # Vertex connectivity
    # =========================================================================

    def get_vertex_neighbors(self, index: int, include_self: bool = True) -> NDArray[np.intp]:
        """Get the indices of the vertices connected to a given vertex.

        Args:
            index: Index of the vertex
            include_self: Include the vertex itself in the list

        Returns:
            Array of connected vertex indices
        """
        return get_vertex_neighbors(self, index, include_self)

    def get_all_vertex_neighbors(self, include_self: bool = True) -> list[NDArray[np.intp]]:
        """Get all vertex neighbors.

        Args:
            include_self: Include each vertex in its own neighbor list

        Returns:
            List of arrays of connected vertex indices
        """
        return list(get_vertex_neighbors_all(self, include_self))

    def get_vertex_cycles(self) -> list[list[int]]:
        """Find cycles (holes/boundaries) in the mesh.

        Returns:
            List of cycles, each cycle is a list of vertex indices
        """
        return [list(cycle) for cycle in get_vertex_cycles(self)]

    # =========================================================================
    # Labeling and projection
    # =========================================================================

    def label_from_image(
        self,
        segmented_image: NDArray[np.integer[Any]],
        resolution: list[float] | None = None,
        background: int = 0,
        mode: str = "point",
    ) -> NDArray[np.integer[Any]]:
        """Label mesh vertices or faces using nearest voxel in segmented image.

        Args:
            segmented_image: 3D segmented image with integer labels
            resolution: Spatial resolution of segmented image
            background: Background label value to ignore
            mode: Labeling mode ('point' for vertices, 'face' for cell centers)

        Returns:
            Label array
        """
        return label_from_image(self, segmented_image, resolution, background, mode)

    def project_to_surface(
        self,
        intensity_image: NDArray[Any],
        distance_threshold: float,
        mask: NDArray[np.bool_] | None = None,
        resolution: list[float] | None = None,
        aggregation_function: Callable[..., Any] = np.sum,
        background: float = 0,
    ) -> NDArray[np.floating[Any]]:
        """Project image intensity values onto mesh surface.

        Args:
            intensity_image: 3D intensity image array
            distance_threshold: Maximum distance from surface for projection
            mask: Optional mask array for image
            resolution: Spatial resolution of intensity image
            aggregation_function: Aggregation function
            background: Background value to ignore in image

        Returns:
            Array of projected values, one per mesh vertex
        """
        return project_to_surface(
            self, intensity_image, distance_threshold, mask, resolution, aggregation_function, background
        )

    # =========================================================================
    # Phenotyping-specific operations
    # =========================================================================

    def define_meristem(
        self,
        method: str = "central_mass",
        resolution: tuple[float, float, float] = (1, 1, 1),
        return_coordinates: bool = False,
    ) -> int | tuple[int, NDArray[np.floating[Any]]]:
        """Determine which domain corresponds to the meristem.

        Args:
            method: Method for defining the meristem
            resolution: Resolution of the dimensions
            return_coordinates: If True, return coordinates as well

        Returns:
            Domain index of the meristem, and optionally the center coordinates
        """
        result = define_meristem(self, method, resolution, return_coordinates)
        if return_coordinates:
            return (int(result[0]), np.asarray(result[1]))  # type: ignore[index]
        return int(result)  # type: ignore[arg-type]

    def fit_paraboloid(
        self,
        init: list[float] | None = None,
        return_success: bool = False,
    ) -> NDArray[np.floating[Any]] | tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Fit a paraboloid to the mesh.

        Args:
            init: Initial parameters for the paraboloid
            return_success: If True, return apex coordinates as well

        Returns:
            Parameters for the paraboloid, and optionally the apex coordinates
        """
        return fit_paraboloid_mesh(self, return_coord=return_success)

    def process(
        self,
        hole_repair_threshold: int = 100,
        downscaling: float = 0.01,
        upscaling: float = 2,
        threshold_angle: float = 60,
        top_cut: str | tuple[float, float, float] = "center",
        tongues_radius: float | None = None,
        tongues_ratio: float = 4,
        smooth_iterations: int = 200,
        smooth_relaxation: float = 0.01,
        curvature_threshold: float = 0.4,
        inland_threshold: float | None = None,
        contour: NDArray[np.bool_] | None = None,
    ) -> PhenoMesh:
        """Convenience function for postprocessing the mesh.

        Args:
            hole_repair_threshold: Threshold for the hole repair algorithm
            downscaling: Downscaling factor for the mesh
            upscaling: Upscaling factor for the mesh
            threshold_angle: Threshold for the polar angle
            top_cut: Top cut location
            tongues_radius: Radius of the tongues
            tongues_ratio: Ratio of the tongues
            smooth_iterations: Number of smoothing iterations
            smooth_relaxation: Smoothing relaxation factor
            curvature_threshold: Threshold for the curvature
            inland_threshold: Threshold for the inland removal
            contour: Contour to use for the inland removal

        Returns:
            Processed PhenoMesh
        """
        return self._wrap_result(
            process(
                self,
                hole_repair_threshold,
                downscaling,
                upscaling,
                threshold_angle,
                top_cut,
                tongues_radius,
                tongues_ratio,
                smooth_iterations,
                smooth_relaxation,
                curvature_threshold,
                inland_threshold,
                contour,
            ),
        )

    # =========================================================================
    # Magic methods
    # =========================================================================

    def __add__(self, other: PhenoMesh | pv.PolyData) -> PhenoMesh:
        """Concatenate meshes."""
        if isinstance(other, pv.PolyData):
            result = super().__add__(other)
            return self._wrap_result(result)
        raise TypeError(f"Cannot add PhenoMesh and {type(other).__name__}")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PhenoMesh(n_points={self.n_points}, n_faces={self.n_faces})"

    def __str__(self) -> str:
        """Return string representation."""
        return self.__repr__()

    # =========================================================================
    # Backwards compatibility aliases
    # =========================================================================

    # Keep old names as aliases for backwards compatibility
    smoothen = smooth
    filter_curvature = filter_by_curvature
    remove_normals = remove_by_normals
    repair_small = repair_small_holes
    label_mesh = label_from_image
    project2surface = project_to_surface
    process_mesh = process
    boundary_points = get_boundary_points
    boundary_edges = get_boundary_edges
    non_manifold_edges = get_non_manifold_edges
    manifold_edges = get_manifold_edges
    feature_edges = get_feature_edges
    vertex_neighbors = get_vertex_neighbors
    vertex_neighbors_all = get_all_vertex_neighbors
    vertex_cycles = get_vertex_cycles
    find_point = find_closest_point
