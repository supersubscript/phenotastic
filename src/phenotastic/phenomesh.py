"""PhenoMesh: Wrapper class for PyVista PolyData with phenotyping-specific operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv

from phenotastic.mesh import remesh

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray


class PhenoMesh:
    """Wrapper class for PyVista PolyData with phenotyping-specific operations.

    Provides a convenient interface for mesh operations commonly used in
    3D plant phenotyping, including smoothing, repair, and feature extraction.

    Args:
        polydata: PyVista PolyData mesh to wrap

    Example:
        >>> mesh = PhenoMesh(pv.Sphere())
        >>> smoothed = mesh.smoothen(iterations=100)
        >>> polydata = mesh.to_polydata()
    """

    mesh: pv.PolyData
    contour: NDArray[np.bool_] | None = None
    resolution: list[float] | None = None

    def __init__(
        self,
        polydata: pv.PolyData,
        contour: NDArray[np.bool_] | None = None,
        resolution: list[float] | None = None,
    ) -> None:
        """Initialize PhenoMesh with a PyVista PolyData object."""
        if not isinstance(polydata, pv.PolyData):
            raise TypeError(f"Expected pv.PolyData, got {type(polydata).__name__}")
        self.mesh = polydata
        self.contour = contour
        self.resolution = resolution

    @classmethod
    def from_polydata(cls, polydata: pv.PolyData) -> PhenoMesh:
        """Create PhenoMesh from PyVista PolyData.

        Args:
            polydata: PyVista PolyData mesh

        Returns:
            New PhenoMesh instance
        """
        return cls(polydata)

    def to_polydata(self) -> pv.PolyData:
        """Return the underlying PyVista PolyData object.

        Returns:
            The wrapped PyVista PolyData mesh
        """
        return self.mesh

    def copy(self) -> PhenoMesh:
        """Create a deep copy of this PhenoMesh.

        Returns:
            New PhenoMesh with copied data
        """
        return PhenoMesh(self.mesh.copy())

    # =========================================================================
    # Properties - delegate to underlying PolyData
    # =========================================================================

    @property
    def n_points(self) -> int:
        """Number of vertices in the mesh."""
        return int(self.mesh.n_points)

    @property
    def n_cells(self) -> int:
        """Number of cells (faces) in the mesh."""
        return int(self.mesh.n_cells)

    @property
    def n_faces(self) -> int:
        """Number of faces in the mesh."""
        return int(self.mesh.n_cells)

    @property
    def points(self) -> NDArray[np.floating[Any]]:
        """Vertex coordinates as Nx3 array."""
        return self.mesh.points

    @points.setter
    def points(self, value: NDArray[np.floating[Any]]) -> None:
        """Set vertex coordinates."""
        self.mesh.points = value

    @property
    def faces(self) -> NDArray[np.intp]:
        """Face connectivity array."""
        return self.mesh.faces

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """Bounding box as (xmin, xmax, ymin, ymax, zmin, zmax)."""
        bounds = self.mesh.bounds
        return (
            float(bounds[0]),
            float(bounds[1]),
            float(bounds[2]),
            float(bounds[3]),
            float(bounds[4]),
            float(bounds[5]),
        )

    @property
    def center(self) -> tuple[float, float, float]:
        """Geometric center of the mesh."""
        center = self.mesh.center
        return (float(center[0]), float(center[1]), float(center[2]))

    @property
    def area(self) -> float:
        """Surface area of the mesh."""
        return float(self.mesh.area)

    @property
    def volume(self) -> float:
        """Volume enclosed by the mesh."""
        return float(self.mesh.volume)

    @property
    def point_normals(self) -> NDArray[np.floating[Any]]:
        """Point normals as Nx3 array."""
        return self.mesh.point_normals

    # =========================================================================
    # Array data access - delegate to underlying PolyData
    # =========================================================================

    def __getitem__(self, key: str) -> NDArray[Any]:
        """Get array data by name."""
        return self.mesh[key]

    def __setitem__(self, key: str, value: NDArray[Any]) -> None:
        """Set array data by name."""
        self.mesh[key] = value

    def clear_data(self) -> None:
        """Clear all point and cell data arrays."""
        self.mesh.clear_data()

    # =========================================================================
    # Convenience methods
    # =========================================================================

    def smoothen(
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
        smoothed = self.mesh.smooth(
            n_iter=iterations,
            relaxation_factor=relaxation_factor,
            feature_smoothing=feature_smoothing,
            boundary_smoothing=boundary_smoothing,
            edge_angle=edge_angle,
            feature_angle=feature_angle,
        )
        return PhenoMesh(smoothed)

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
        smoothed = self.mesh.smooth_taubin(
            n_iter=iterations,
            pass_band=pass_band,
            edge_angle=edge_angle,
            feature_angle=feature_angle,
            feature_smoothing=feature_smoothing,
            boundary_smoothing=boundary_smoothing,
            non_manifold_smoothing=non_manifold_smoothing,
            normalize_coordinates=normalize_coordinates,
        )
        return PhenoMesh(smoothed)

    def decimate(
        self,
        target_reduction: float = 0.5,
        volume_preservation: bool = True,
    ) -> PhenoMesh:
        """Reduce mesh complexity by decimating faces.

        Args:
            target_reduction: Fraction of faces to remove (0-1)
            volume_preservation: Preserve mesh volume during decimation

        Returns:
            New PhenoMesh with reduced complexity
        """
        decimated = self.mesh.decimate(
            target_reduction=target_reduction,
            volume_preservation=volume_preservation,
        )
        return PhenoMesh(decimated)

    def subdivide(self, n_subdivisions: int = 1, subfilter: str = "linear") -> PhenoMesh:
        """Subdivide mesh faces to increase resolution.

        Args:
            n_subdivisions: Number of subdivision iterations
            subfilter: Subdivision filter ('linear', 'butterfly', 'loop')

        Returns:
            New PhenoMesh with subdivided faces
        """
        subdivided = self.mesh.subdivide(n_subdivisions, subfilter)
        return PhenoMesh(subdivided)

    def triangulate(self) -> PhenoMesh:
        """Convert all faces to triangles.

        Returns:
            New PhenoMesh with triangulated faces
        """
        return PhenoMesh(self.mesh.triangulate())

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
        cleaned = self.mesh.clean(
            point_merging=point_merging,
            tolerance=tolerance,
            lines_to_points=lines_to_points,
            polys_to_lines=polys_to_lines,
            strips_to_polys=strips_to_polys,
            absolute=absolute,
        )
        return PhenoMesh(cleaned)

    def extract_largest(self) -> PhenoMesh:
        """Extract the largest connected component.

        Returns:
            New PhenoMesh containing only the largest connected region
        """
        return PhenoMesh(self.mesh.extract_largest())

    def fill_holes(self, hole_size: float = 1000.0) -> PhenoMesh:
        """Fill holes in the mesh.

        Args:
            hole_size: Maximum hole size to fill

        Returns:
            New PhenoMesh with holes filled
        """
        return PhenoMesh(self.mesh.fill_holes(hole_size))

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
        with_normals = self.mesh.compute_normals(
            cell_normals=cell_normals,
            point_normals=point_normals,
            flip_normals=flip_normals,
            consistent_normals=consistent_normals,
            auto_orient_normals=auto_orient_normals,
            non_manifold_traversal=non_manifold_traversal,
            feature_angle=feature_angle,
        )
        return PhenoMesh(with_normals)

    def flip_normals(self) -> PhenoMesh:
        """Flip all surface normals.

        Returns:
            New PhenoMesh with flipped normals
        """
        flipped = self.mesh.copy()
        flipped.flip_normals()
        return PhenoMesh(flipped)

    def curvature(self, curv_type: str = "mean") -> NDArray[np.floating[Any]]:
        """Compute surface curvature.

        Args:
            curv_type: Curvature type ('mean', 'gaussian', 'minimum', 'maximum')

        Returns:
            Array of curvature values per vertex
        """
        return self.mesh.curvature(curv_type)

    def remove_points(
        self,
        mask: NDArray[np.bool_],
        keep_scalars: bool = True,
    ) -> PhenoMesh:
        """Remove points from the mesh.

        Args:
            mask: Boolean mask indicating points to remove (True = remove)
            keep_scalars: Preserve scalar data

        Returns:
            New PhenoMesh with points removed
        """
        result = self.mesh.remove_points(mask, keep_scalars=keep_scalars)
        return PhenoMesh(result[0])

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
        clipped = self.mesh.clip(normal=normal, origin=origin, invert=invert)
        return PhenoMesh(clipped)

    def center_of_mass(self) -> NDArray[np.floating[Any]]:
        """Compute center of mass.

        Returns:
            3-element array of center of mass coordinates
        """
        return np.array(self.mesh.center_of_mass())

    def geodesic_distance(self, start_vertex: int, end_vertex: int) -> float:
        """Compute geodesic distance between two vertices.

        Args:
            start_vertex: Index of start vertex
            end_vertex: Index of end vertex

        Returns:
            Geodesic distance along the mesh surface
        """
        return float(self.mesh.geodesic_distance(start_vertex, end_vertex))

    def ray_trace(
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
        result = self.mesh.ray_trace(origin, end_point)
        return (np.asarray(result[0]), np.asarray(result[1]))

    def rotate_x(self, angle: float, inplace: bool = False) -> PhenoMesh:
        """Rotate mesh around X axis.

        Args:
            angle: Rotation angle in degrees
            inplace: Modify in place

        Returns:
            Rotated PhenoMesh (or self if inplace)
        """
        if inplace:
            self.mesh.rotate_x(angle, inplace=True)
            return self
        rotated = self.mesh.copy()
        rotated.rotate_x(angle, inplace=True)
        return PhenoMesh(rotated)

    def rotate_y(self, angle: float, inplace: bool = False) -> PhenoMesh:
        """Rotate mesh around Y axis.

        Args:
            angle: Rotation angle in degrees
            inplace: Modify in place

        Returns:
            Rotated PhenoMesh (or self if inplace)
        """
        if inplace:
            self.mesh.rotate_y(angle, inplace=True)
            return self
        rotated = self.mesh.copy()
        rotated.rotate_y(angle, inplace=True)
        return PhenoMesh(rotated)

    def rotate_z(self, angle: float, inplace: bool = False) -> PhenoMesh:
        """Rotate mesh around Z axis.

        Args:
            angle: Rotation angle in degrees
            inplace: Modify in place

        Returns:
            Rotated PhenoMesh (or self if inplace)
        """
        if inplace:
            self.mesh.rotate_z(angle, inplace=True)
            return self
        rotated = self.mesh.copy()
        rotated.rotate_z(angle, inplace=True)
        return PhenoMesh(rotated)

    # =========================================================================
    # Wrapped functions from mesh.py
    # =========================================================================

    def filter_curvature(
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
        from phenotastic.mesh import filter_curvature as _filter_curvature

        return PhenoMesh(_filter_curvature(self.mesh, curvature_threshold, curvatures))

    def label_mesh(
        self,
        segm_img: NDArray[np.integer[Any]],
        resolution: list[float] | None = None,
        background: int = 0,
        mode: str = "point",
    ) -> NDArray[np.integer[Any]]:
        """Label mesh vertices or faces using nearest voxel in segmented image.

        Args:
            segm_img: 3D segmented image with integer labels
            resolution: Spatial resolution of segmented image
            background: Background label value to ignore
            mode: Labeling mode ('point' for vertices, 'face' for cell centers)

        Returns:
            Label array
        """
        from phenotastic.mesh import label_mesh as _label_mesh

        return _label_mesh(self.mesh, segm_img, resolution, background, mode)

    def project2surface(
        self,
        int_img: NDArray[Any],
        distance_threshold: float,
        mask: NDArray[np.bool_] | None = None,
        resolution: list[float] | None = None,
        fct: Callable[..., Any] = np.sum,
        background: float = 0,
    ) -> NDArray[np.floating[Any]]:
        """Project image intensity values onto mesh surface.

        Args:
            int_img: 3D intensity image array
            distance_threshold: Maximum distance from surface for projection
            mask: Optional mask array for image
            resolution: Spatial resolution of intensity image
            fct: Aggregation function (currently only np.sum supported)
            background: Background value to ignore in image

        Returns:
            Array of projected values, one per mesh vertex
        """
        from phenotastic.mesh import project2surface as _project2surface

        return _project2surface(self.mesh, int_img, distance_threshold, mask, resolution, fct, background)

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
        from phenotastic.mesh import remove_inland_under as _remove_inland_under

        return PhenoMesh(_remove_inland_under(self.mesh, contour, threshold, resolution, invert))

    def repair_small(self, nbe: int | None = 100, refine: bool = True) -> PhenoMesh:
        """Repair small holes in the mesh based on the number of edges.

        Args:
            nbe: Number of edges to use for the repair
            refine: Refine the mesh

        Returns:
            Repaired PhenoMesh
        """
        from phenotastic.mesh import repair_small as _repair_small

        return PhenoMesh(_repair_small(self.mesh, nbe, refine))

    def correct_bad_mesh(self, verbose: bool = True) -> PhenoMesh:
        """Correct a bad (non-manifold) mesh.

        Args:
            verbose: Print processing steps

        Returns:
            Corrected PhenoMesh
        """
        from phenotastic.mesh import correct_bad_mesh as _correct_bad_mesh

        return PhenoMesh(_correct_bad_mesh(self.mesh, verbose))

    def remove_bridges(self, verbose: bool = True) -> PhenoMesh:
        """Remove triangles where all vertices are part of the mesh boundary.

        Args:
            verbose: Print processing steps

        Returns:
            PhenoMesh after bridge removal
        """
        from phenotastic.mesh import remove_bridges as _remove_bridges

        return PhenoMesh(_remove_bridges(self.mesh, verbose))

    def remove_normals(
        self,
        threshold_angle: float = 0,
        flip: bool = False,
        angle: str = "polar",
    ) -> PhenoMesh:
        """Remove points based on the point normal angle.

        Args:
            threshold_angle: Threshold for the polar angle
            flip: Flip normal orientation
            angle: Type of angle to use ('polar' or 'azimuth')

        Returns:
            PhenoMesh with vertices removed
        """
        from phenotastic.mesh import remove_normals as _remove_normals

        return PhenoMesh(_remove_normals(self.mesh, threshold_angle, flip, angle))

    def smooth_boundary(
        self,
        iterations: int = 20,
        sigma: float = 0.1,
    ) -> PhenoMesh:
        """Smooth the boundary of the mesh using Laplacian smoothing.

        Args:
            iterations: Number of smoothing iterations
            sigma: Smoothing sigma

        Returns:
            Smoothed PhenoMesh
        """
        from phenotastic.mesh import smooth_boundary as _smooth_boundary

        result = _smooth_boundary(self.mesh, iterations, sigma, inplace=False)
        if result is None:
            return self.copy()
        return PhenoMesh(result)

    def process_mesh(
        self,
        hole_repair_threshold: int = 100,
        downscaling: float = 0.01,
        upscaling: float = 2,
        threshold_angle: float = 60,
        top_cut: str | tuple[float, float, float] = "center",
        tongues_radius: float | None = None,
        tongues_ratio: float = 4,
        smooth_iter: int = 200,
        smooth_relax: float = 0.01,
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
            smooth_iter: Number of smoothing iterations
            smooth_relax: Smoothing relaxation factor
            curvature_threshold: Threshold for the curvature
            inland_threshold: Threshold for the inland removal
            contour: Contour to use for the inland removal

        Returns:
            Processed PhenoMesh
        """
        from phenotastic.mesh import process_mesh as _process_mesh

        return PhenoMesh(
            _process_mesh(
                self.mesh,
                hole_repair_threshold,
                downscaling,
                upscaling,
                threshold_angle,
                top_cut,
                tongues_radius,
                tongues_ratio,
                smooth_iter,
                smooth_relax,
                curvature_threshold,
                inland_threshold,
                contour,
            ),
        )

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

        Returns:
            PhenoMesh with tongues removed
        """
        from phenotastic.mesh import remove_tongues as _remove_tongues

        return PhenoMesh(_remove_tongues(self.mesh, radius, threshold, hole_edges, verbose))

    def repair(self) -> PhenoMesh:
        """Repair the mesh using MeshFix.

        Returns:
            Repaired PhenoMesh
        """
        from phenotastic.mesh import repair as _repair

        return PhenoMesh(_repair(self.mesh))

    def remesh(self, n: int, sub: int = 3) -> PhenoMesh:
        """Regularise the mesh faces using the ACVD algorithm.

        Args:
            n: The number of clusters (i.e. output number of faces) to use
            sub: The number of subdivisions to use when clustering

        Returns:
            Regularised PhenoMesh
        """

        return PhenoMesh(remesh(self.mesh, n, sub))

    def make_manifold(self, hole_edges: int = 300) -> PhenoMesh:
        """Make the mesh manifold by removing non-manifold edges.

        Args:
            hole_edges: Size of holes to fill

        Returns:
            Manifold PhenoMesh
        """
        from phenotastic.mesh import make_manifold as _make_manifold

        return PhenoMesh(_make_manifold(self.mesh, hole_edges))

    def drop_skirt(self, maxdist: float, flip: bool = False) -> PhenoMesh:
        """Downprojects the boundary to the lowest point in the z-direction.

        Args:
            maxdist: Distance in z-direction from the lowest point to consider
            flip: If True, flip the direction

        Returns:
            PhenoMesh with boundary downprojected
        """
        from phenotastic.mesh import drop_skirt as _drop_skirt

        return PhenoMesh(_drop_skirt(self.mesh, maxdist, flip))

    def boundary_points(self) -> NDArray[np.intp]:
        """Get vertex indices of points in the boundary.

        Returns:
            Array of boundary vertex indices
        """
        from phenotastic.mesh import get_boundary_points

        return get_boundary_points(self.mesh)

    def boundary_edges(self) -> PhenoMesh:
        """Get boundary edges.

        Returns:
            PhenoMesh containing boundary edges
        """
        from phenotastic.mesh import get_boundary_edges

        return PhenoMesh(get_boundary_edges(self.mesh))

    def non_manifold_edges(self) -> PhenoMesh:
        """Get non-manifold edges.

        Returns:
            PhenoMesh containing non-manifold edges
        """
        from phenotastic.mesh import get_non_manifold_edges

        return PhenoMesh(get_non_manifold_edges(self.mesh))

    def manifold_edges(self) -> PhenoMesh:
        """Get manifold edges.

        Returns:
            PhenoMesh containing manifold edges
        """
        from phenotastic.mesh import get_manifold_edges

        return PhenoMesh(get_manifold_edges(self.mesh))

    def feature_edges(self, angle: float = 30) -> PhenoMesh:
        """Get feature edges defined by given angle.

        Args:
            angle: Feature angle threshold

        Returns:
            PhenoMesh containing feature edges
        """
        from phenotastic.mesh import get_feature_edges

        return PhenoMesh(get_feature_edges(self.mesh, angle))

    def vertex_neighbors(self, index: int, include_self: bool = True) -> NDArray[np.intp]:
        """Get the indices of the vertices connected to a given vertex.

        Args:
            index: Index of the vertex
            include_self: Include the vertex itself in the list

        Returns:
            Array of connected vertex indices
        """
        from phenotastic.mesh import get_vertex_neighbors

        return get_vertex_neighbors(self.mesh, index, include_self)

    def vertex_neighbors_all(self, include_self: bool = True) -> list[NDArray[np.intp]]:
        """Get all vertex neighbors.

        Args:
            include_self: Include each vertex in its own neighbor list

        Returns:
            List of arrays of connected vertex indices
        """
        from phenotastic.mesh import get_vertex_neighbors_all

        result = get_vertex_neighbors_all(self.mesh, include_self)
        return list(result)

    def vertex_cycles(self) -> list[list[int]]:
        """Find cycles (holes/boundaries) in the mesh.

        Returns:
            List of cycles, each cycle is a list of vertex indices
        """
        from phenotastic.mesh import get_vertex_cycles

        result = get_vertex_cycles(self.mesh)
        return [list(cycle) for cycle in result]

    def erode(self, iterations: int = 1) -> PhenoMesh:
        """Erode the mesh by removing boundary points iteratively.

        Args:
            iterations: Number of erosion iterations

        Returns:
            Eroded PhenoMesh
        """
        from phenotastic.mesh import erode as _erode

        return PhenoMesh(_erode(self.mesh, iterations))

    def ecft(self, hole_edges: int = 300) -> PhenoMesh:
        """Perform ExtractLargest, Clean, FillHoles, and TriFilter operations.

        Args:
            hole_edges: Size of holes to fill

        Returns:
            Processed PhenoMesh
        """
        from phenotastic.mesh import ecft as _ecft

        return PhenoMesh(_ecft(self.mesh, hole_edges))

    def correct_normal_orientation(self, relative: str = "x", inplace: bool = False) -> PhenoMesh | None:
        """Correct the orientation of the normals.

        Args:
            relative: Axis to use for orientation ('x', 'y', 'z')
            inplace: Modify in place

        Returns:
            PhenoMesh with corrected normals, or None if inplace
        """
        from phenotastic.mesh import correct_normal_orientation as _correct_normal_orientation

        if inplace:
            _correct_normal_orientation(self.mesh, relative, inplace=True)
            return None
        result = _correct_normal_orientation(self.mesh, relative, inplace=False)
        if result is None:
            return self.copy()
        return PhenoMesh(result)

    def fit_paraboloid(
        self,
        init: list[float] | None = None,
        return_success: bool = False,
    ) -> NDArray[np.floating[Any]] | tuple[NDArray[np.floating[Any]], bool]:
        """Fit a paraboloid to the mesh.

        Args:
            init: Initial parameters for the paraboloid
            return_success: If True, return success status as well

        Returns:
            Parameters for the paraboloid, and optionally success status
        """
        from phenotastic.mesh import fit_paraboloid_mesh as _fit_paraboloid_mesh

        return _fit_paraboloid_mesh(self.mesh, return_coord=return_success)

    def remesh_decimate(
        self,
        iters: int,
        upfactor: float = 2,
        downfactor: float = 0.5,
        verbose: bool = True,
    ) -> PhenoMesh:
        """Iterative remeshing/decimation.

        Args:
            iters: Number of iterations
            upfactor: Factor with which to upsample
            downfactor: Factor with which to downsample
            verbose: Print operation steps

        Returns:
            Processed PhenoMesh
        """
        from phenotastic.mesh import remesh_decimate as _remesh_decimate

        return PhenoMesh(_remesh_decimate(self.mesh, iters, upfactor, downfactor, verbose))

    def define_meristem(
        self,
        method: str = "central_mass",
        resolution: tuple[float, float, float] = (1, 1, 1),
        return_coordinates: bool = False,
    ) -> int | tuple[int, NDArray[np.floating[Any]]]:
        """Determine which domain corresponds to the meristem.

        Args:
            method: Method for defining the meristem
            res: Resolution of the dimensions
            return_coord: If True, return coordinates as well

        Returns:
            Domain index of the meristem, and optionally the center coordinates
        """
        from phenotastic.mesh import define_meristem as _define_meristem

        result = _define_meristem(self.mesh, method, resolution, return_coordinates)
        if return_coordinates:
            return (int(result[0]), np.asarray(result[1]))  # type: ignore[index]
        return int(result)  # type: ignore[arg-type]

    def find_point(self, point: Sequence[float]) -> int:
        """Find the index of the closest point.

        Args:
            point: Query point coordinates

        Returns:
            Index of the closest point
        """
        return int(self.mesh.FindPoint(point))

    # =========================================================================
    # Visualization helpers
    # =========================================================================

    def plot(self, **kwargs: Any) -> Any:
        """Plot the mesh using PyVista.

        Args:
            **kwargs: Arguments passed to pyvista.plot()

        Returns:
            PyVista plotter object
        """
        return self.mesh.plot(**kwargs)

    # =========================================================================
    # Magic methods
    # =========================================================================
    def __add__(self, other: PhenoMesh | pv.PolyData) -> PhenoMesh:
        """Concatenate meshes."""
        if isinstance(other, PhenoMesh):
            return PhenoMesh(self.mesh + other.mesh)
        if isinstance(other, pv.PolyData):
            return PhenoMesh(self.mesh + other)
        raise TypeError(f"Cannot add PhenoMesh and {type(other).__name__}")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PhenoMesh(n_points={self.n_points}, n_faces={self.n_faces})"

    def __str__(self) -> str:
        """Return string representation."""
        return self.__repr__()
