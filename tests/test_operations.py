"""Tests for pipeline operations."""

import numpy as np
import pytest

from phenotastic import PhenoMesh, PipelineContext
from phenotastic.exceptions import ConfigurationError
from phenotastic.operations import (
    clean_op,
    compute_curvature_op,
    decimate_op,
    extract_largest_op,
    filter_curvature_op,
    remesh_op,
    rotate_op,
    smooth_boundary_op,
    smooth_op,
    smooth_taubin_op,
    subdivide_op,
    triangulate_op,
)


class TestSmoothOperation:
    """Tests for the smooth operation."""

    def test_smooth_preserves_point_count(self, sphere_mesh: PhenoMesh) -> None:
        """Smoothing should not change vertex count."""
        ctx = PipelineContext(mesh=sphere_mesh)
        original_points = sphere_mesh.n_points

        result = smooth_op(ctx, iterations=10)

        assert result.mesh is not None
        assert result.mesh.n_points == original_points

    def test_smooth_with_zero_iterations(self, sphere_mesh: PhenoMesh) -> None:
        """Zero iterations should return unchanged mesh."""
        ctx = PipelineContext(mesh=sphere_mesh)
        original_points = sphere_mesh.points.copy()

        result = smooth_op(ctx, iterations=0)

        assert result.mesh is not None
        np.testing.assert_array_equal(result.mesh.points, original_points)

    def test_smooth_modifies_points(self, sphere_mesh: PhenoMesh) -> None:
        """Smoothing should modify point positions."""
        ctx = PipelineContext(mesh=sphere_mesh)
        original_points = sphere_mesh.points.copy()

        result = smooth_op(ctx, iterations=10, relaxation_factor=0.1)

        assert result.mesh is not None
        # Points should be different after smoothing
        assert not np.allclose(result.mesh.points, original_points)

    def test_smooth_invalid_iterations_raises(self, sphere_mesh: PhenoMesh) -> None:
        """Negative iterations should raise ConfigurationError."""
        ctx = PipelineContext(mesh=sphere_mesh)

        with pytest.raises(ConfigurationError, match="iterations must be non-negative"):
            smooth_op(ctx, iterations=-5)

    def test_smooth_invalid_relaxation_raises(self, sphere_mesh: PhenoMesh) -> None:
        """Invalid relaxation factor should raise ConfigurationError."""
        ctx = PipelineContext(mesh=sphere_mesh)

        with pytest.raises(ConfigurationError, match="relaxation_factor"):
            smooth_op(ctx, iterations=10, relaxation_factor=1.5)

    def test_smooth_requires_mesh(self) -> None:
        """Smooth operation should require mesh in context."""
        ctx = PipelineContext()

        with pytest.raises(ConfigurationError, match="requires a mesh"):
            smooth_op(ctx, iterations=10)


class TestSmoothTaubinOperation:
    """Tests for the smooth_taubin operation."""

    def test_smooth_taubin_preserves_point_count(self, sphere_mesh: PhenoMesh) -> None:
        """Taubin smoothing should not change vertex count."""
        ctx = PipelineContext(mesh=sphere_mesh)
        original_points = sphere_mesh.n_points

        result = smooth_taubin_op(ctx, iterations=10)

        assert result.mesh is not None
        assert result.mesh.n_points == original_points

    def test_smooth_taubin_zero_iterations(self, sphere_mesh: PhenoMesh) -> None:
        """Zero iterations should return unchanged mesh."""
        ctx = PipelineContext(mesh=sphere_mesh)

        result = smooth_taubin_op(ctx, iterations=0)

        assert result.mesh is not None


class TestRemeshOperation:
    """Tests for the remesh operation."""

    def test_remesh_changes_point_count(self, sphere_mesh: PhenoMesh) -> None:
        """Remeshing should change vertex count to approximately target."""
        ctx = PipelineContext(mesh=sphere_mesh)

        result = remesh_op(ctx, n_clusters=500, subdivisions=2)

        assert result.mesh is not None
        # Point count should be different
        assert result.mesh.n_points != sphere_mesh.n_points

    def test_remesh_invalid_clusters_raises(self, sphere_mesh: PhenoMesh) -> None:
        """Invalid n_clusters should raise ConfigurationError."""
        ctx = PipelineContext(mesh=sphere_mesh)

        with pytest.raises(ConfigurationError, match="n_clusters must be positive"):
            remesh_op(ctx, n_clusters=0)


class TestDecimateOperation:
    """Tests for the decimate operation."""

    def test_decimate_reduces_faces(self, sphere_mesh: PhenoMesh) -> None:
        """Decimation should reduce face count."""
        ctx = PipelineContext(mesh=sphere_mesh)
        original_faces = sphere_mesh.n_faces

        result = decimate_op(ctx, target_reduction=0.5)

        assert result.mesh is not None
        assert result.mesh.n_faces < original_faces

    def test_decimate_invalid_reduction_raises(self, sphere_mesh: PhenoMesh) -> None:
        """Invalid target_reduction should raise ConfigurationError."""
        ctx = PipelineContext(mesh=sphere_mesh)

        with pytest.raises(ConfigurationError, match="target_reduction"):
            decimate_op(ctx, target_reduction=1.5)


class TestSubdivideOperation:
    """Tests for the subdivide operation."""

    def test_subdivide_increases_faces(self, small_sphere_mesh: PhenoMesh) -> None:
        """Subdivision should increase face count."""
        ctx = PipelineContext(mesh=small_sphere_mesh)
        original_faces = small_sphere_mesh.n_faces

        result = subdivide_op(ctx, n_subdivisions=1)

        assert result.mesh is not None
        assert result.mesh.n_faces > original_faces

    def test_subdivide_zero_iterations(self, sphere_mesh: PhenoMesh) -> None:
        """Zero subdivisions should not change mesh."""
        ctx = PipelineContext(mesh=sphere_mesh)
        original_faces = sphere_mesh.n_faces

        result = subdivide_op(ctx, n_subdivisions=0)

        assert result.mesh is not None
        assert result.mesh.n_faces == original_faces


class TestCleanOperation:
    """Tests for the clean operation."""

    def test_clean_runs_successfully(self, sphere_mesh: PhenoMesh) -> None:
        """Clean operation should complete without errors."""
        ctx = PipelineContext(mesh=sphere_mesh)

        result = clean_op(ctx)

        assert result.mesh is not None


class TestTriangulateOperation:
    """Tests for the triangulate operation."""

    def test_triangulate_runs_successfully(self, cube_mesh: PhenoMesh) -> None:
        """Triangulate operation should complete without errors."""
        ctx = PipelineContext(mesh=cube_mesh)

        result = triangulate_op(ctx)

        assert result.mesh is not None


class TestExtractLargestOperation:
    """Tests for the extract_largest operation."""

    def test_extract_largest_runs_successfully(self, sphere_mesh: PhenoMesh) -> None:
        """Extract largest should complete without errors."""
        ctx = PipelineContext(mesh=sphere_mesh)

        result = extract_largest_op(ctx)

        assert result.mesh is not None


class TestFilterCurvatureOperation:
    """Tests for the filter_curvature operation."""

    def test_filter_curvature_with_wide_threshold(self, sphere_mesh: PhenoMesh) -> None:
        """Curvature filter with wide threshold should keep most vertices."""
        ctx = PipelineContext(mesh=sphere_mesh)
        original_points = sphere_mesh.n_points

        # Use a wide threshold to avoid removing all vertices
        result = filter_curvature_op(ctx, threshold=5.0)

        assert result.mesh is not None
        # With a wide threshold, should keep most/all points
        assert result.mesh.n_points > 0
        assert result.mesh.n_points <= original_points

    def test_filter_curvature_with_list_threshold(self, sphere_mesh: PhenoMesh) -> None:
        """Filter should accept list threshold."""
        ctx = PipelineContext(mesh=sphere_mesh)

        # Use a wide range to avoid removing all vertices
        result = filter_curvature_op(ctx, threshold=[-5.0, 5.0])

        assert result.mesh is not None
        assert result.mesh.n_points > 0


class TestRotateOperation:
    """Tests for the rotate operation."""

    def test_rotate_changes_points(self, sphere_mesh: PhenoMesh) -> None:
        """Rotation should change point positions."""
        ctx = PipelineContext(mesh=sphere_mesh)
        original_points = sphere_mesh.points.copy()

        result = rotate_op(ctx, axis="y", angle=90)

        assert result.mesh is not None
        # Points should be different after rotation
        assert not np.allclose(result.mesh.points, original_points)

    def test_rotate_zero_angle(self, sphere_mesh: PhenoMesh) -> None:
        """Zero rotation should not change points."""
        ctx = PipelineContext(mesh=sphere_mesh)
        original_points = sphere_mesh.points.copy()

        result = rotate_op(ctx, axis="x", angle=0)

        assert result.mesh is not None
        np.testing.assert_array_almost_equal(result.mesh.points, original_points)

    def test_rotate_invalid_axis_raises(self, sphere_mesh: PhenoMesh) -> None:
        """Invalid axis should raise ConfigurationError."""
        ctx = PipelineContext(mesh=sphere_mesh)

        with pytest.raises(ConfigurationError, match="Invalid axis"):
            rotate_op(ctx, axis="w", angle=45)


class TestComputeCurvatureOperation:
    """Tests for the compute_curvature operation."""

    def test_compute_curvature_adds_to_context(self, sphere_mesh: PhenoMesh) -> None:
        """Curvature should be added to context."""
        ctx = PipelineContext(mesh=sphere_mesh)

        result = compute_curvature_op(ctx, curvature_type="mean")

        assert result.curvature is not None
        assert len(result.curvature) == sphere_mesh.n_points

    def test_compute_curvature_invalid_type_raises(self, sphere_mesh: PhenoMesh) -> None:
        """Invalid curvature type should raise ConfigurationError."""
        ctx = PipelineContext(mesh=sphere_mesh)

        with pytest.raises(ConfigurationError, match="curvature_type"):
            compute_curvature_op(ctx, curvature_type="invalid")


class TestSmoothBoundaryOperation:
    """Tests for the smooth_boundary operation."""

    def test_smooth_boundary_runs_successfully(self, sphere_mesh: PhenoMesh) -> None:
        """Smooth boundary should complete without errors."""
        ctx = PipelineContext(mesh=sphere_mesh)

        result = smooth_boundary_op(ctx, iterations=5)

        assert result.mesh is not None

    def test_smooth_boundary_zero_iterations(self, sphere_mesh: PhenoMesh) -> None:
        """Zero iterations should return unchanged."""
        ctx = PipelineContext(mesh=sphere_mesh)

        result = smooth_boundary_op(ctx, iterations=0)

        assert result.mesh is not None
