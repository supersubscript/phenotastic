"""Integration tests for phenotastic pipeline.

These tests verify end-to-end workflows from input to output.
"""

from pathlib import Path

import numpy as np
import pytest

from phenotastic import PhenoMesh, Pipeline, PipelineContext, StepConfig
from phenotastic.operations import OPERATIONS

smooth_mesh = OPERATIONS["smooth"]


class TestMeshProcessingWorkflow:
    """End-to-end tests for mesh processing workflows."""

    def test_basic_mesh_processing(self, small_sphere_mesh: PhenoMesh) -> None:
        """Basic workflow: smooth -> remesh -> curvature."""
        pipeline = Pipeline(
            [
                StepConfig("clean"),
                StepConfig("smooth", {"iterations": 10}),
                StepConfig("remesh", {"n_clusters": 300}),
                StepConfig("smooth", {"iterations": 5}),
                StepConfig("compute_curvature", {"curvature_type": "mean"}),
            ]
        )

        result = pipeline.run(small_sphere_mesh, verbose=False)

        assert result.mesh is not None
        assert result.curvature is not None
        assert len(result.curvature) == result.mesh.n_points

    def test_multi_smooth_remesh_workflow(self, small_sphere_mesh: PhenoMesh) -> None:
        """Workflow with multiple smooth-remesh cycles."""
        pipeline = Pipeline(
            [
                StepConfig("smooth", {"iterations": 20, "relaxation_factor": 0.02}),
                StepConfig("remesh", {"n_clusters": 200}),
                StepConfig("smooth", {"iterations": 15}),
                StepConfig("remesh", {"n_clusters": 400}),
                StepConfig("smooth", {"iterations": 10}),
            ]
        )

        result = pipeline.run(small_sphere_mesh, verbose=False)

        assert result.mesh is not None
        # After remeshing to 400 clusters, should be close to that
        assert result.mesh.n_points > 100  # Some mesh exists

    def test_workflow_with_rotation(self, small_sphere_mesh: PhenoMesh) -> None:
        """Workflow including rotation operations."""
        pipeline = Pipeline(
            [
                StepConfig("rotate", {"axis": "y", "angle": -90}),
                StepConfig("smooth", {"iterations": 5}),
                StepConfig("rotate", {"axis": "y", "angle": 90}),
            ]
        )

        result = pipeline.run(small_sphere_mesh, verbose=False)

        assert result.mesh is not None
        # Should still have same number of points
        assert result.mesh.n_points == small_sphere_mesh.n_points


class TestDomainSegmentationWorkflow:
    """End-to-end tests for domain segmentation workflows."""

    def test_basic_segmentation(self, sphere_mesh: PhenoMesh) -> None:
        """Basic domain segmentation workflow."""
        pipeline = Pipeline(
            [
                StepConfig("smooth", {"iterations": 10}),
                StepConfig("remesh", {"n_clusters": 500}),
                StepConfig("compute_curvature", {"curvature_type": "mean"}),
                StepConfig("segment_domains"),
            ]
        )

        result = pipeline.run(sphere_mesh, verbose=False)

        assert result.mesh is not None
        assert result.curvature is not None
        assert result.domains is not None
        assert len(result.domains) == result.mesh.n_points

    def test_segmentation_workflow(self, sphere_mesh: PhenoMesh) -> None:
        """Domain segmentation workflow (without merge_small edge case)."""
        pipeline = Pipeline(
            [
                StepConfig("remesh", {"n_clusters": 500}),
                StepConfig("smooth", {"iterations": 10}),
                StepConfig("compute_curvature", {"curvature_type": "mean"}),
                StepConfig("segment_domains"),
            ]
        )

        result = pipeline.run(sphere_mesh, verbose=False)

        assert result.mesh is not None
        assert result.domains is not None

        # Should have some domains
        n_domains = len(np.unique(result.domains))
        assert n_domains > 0


class TestContourToMeshWorkflow:
    """End-to-end tests for contour-to-mesh workflows."""

    def test_contour_to_mesh(self, tiny_contour: np.ndarray) -> None:
        """Basic contour to mesh workflow."""
        from phenotastic.operations import OPERATIONS

        create_mesh_from_contour = OPERATIONS["create_mesh"]
        smooth_mesh = OPERATIONS["smooth"]

        # Start with contour
        ctx = PipelineContext(contour=tiny_contour, resolution=[1.0, 1.0, 1.0])

        # Create mesh
        ctx = create_mesh_from_contour(ctx)
        assert ctx.mesh is not None

        # Smooth
        ctx = smooth_mesh(ctx, iterations=5)
        assert ctx.mesh is not None

    def test_full_pipeline_from_contour(self, tiny_contour: np.ndarray) -> None:
        """Full pipeline starting from contour."""
        from phenotastic.operations import OPERATIONS

        create_mesh_from_contour = OPERATIONS["create_mesh"]
        clean_mesh = OPERATIONS["clean"]
        smooth_mesh = OPERATIONS["smooth"]
        remesh = OPERATIONS["remesh"]
        compute_curvature = OPERATIONS["compute_curvature"]

        ctx = PipelineContext(contour=tiny_contour, resolution=[1.0, 1.0, 1.0])

        # Create mesh
        ctx = create_mesh_from_contour(ctx)
        ctx = clean_mesh(ctx)
        ctx = smooth_mesh(ctx, iterations=5)
        ctx = remesh(ctx, n_clusters=100)
        ctx = compute_curvature(ctx, curvature_type="mean")

        assert ctx.mesh is not None
        assert ctx.curvature is not None


class TestYamlConfiguredPipeline:
    """Tests for running pipelines from YAML configuration."""

    def test_yaml_pipeline_execution(self, valid_pipeline_yaml: str, small_sphere_mesh: PhenoMesh) -> None:
        """Pipeline from YAML should execute successfully."""
        pipeline = Pipeline.from_yaml(valid_pipeline_yaml)
        result = pipeline.run(small_sphere_mesh, verbose=False)

        assert result.mesh is not None

    def test_custom_yaml_workflow(self, tmp_path: Path, small_sphere_mesh: PhenoMesh) -> None:
        """Custom YAML pipeline should work end-to-end."""
        yaml_content = """
steps:
  - name: smooth
    params:
      iterations: 10
      relaxation_factor: 0.01
  - name: remesh
    params:
      n_clusters: 300
  - name: smooth
    params:
      iterations: 5
  - name: compute_curvature
    params:
      curvature_type: mean
"""
        config_path = tmp_path / "custom.yaml"
        config_path.write_text(yaml_content)

        pipeline = Pipeline.from_yaml(str(config_path))
        result = pipeline.run(small_sphere_mesh, verbose=False)

        assert result.mesh is not None
        assert result.curvature is not None


class TestPresetPipelineExecution:
    """Tests for running preset pipelines."""

    def test_preset_mesh_processing_steps(self, small_sphere_mesh: PhenoMesh) -> None:
        """Core mesh processing steps from presets should work."""
        # Test the mesh processing portion of presets
        pipeline = Pipeline(
            [
                StepConfig("clean"),
                StepConfig("extract_largest"),
                StepConfig("smooth", {"iterations": 10}),
                StepConfig("remesh", {"n_clusters": 300}),
                StepConfig("compute_curvature", {"curvature_type": "mean"}),
                StepConfig("segment_domains"),
            ]
        )

        result = pipeline.run(small_sphere_mesh, verbose=False)

        assert result.mesh is not None
        assert result.curvature is not None
        assert result.domains is not None

    def test_default_preset_mesh_steps(self, small_sphere_mesh: PhenoMesh) -> None:
        """Default preset mesh steps should complete successfully."""
        from phenotastic import load_preset

        pipeline = load_preset()

        # Run without contour/mesh creation steps (start from mesh)
        # Also skip repair_holes due to pymeshfix API compatibility issues
        skip_steps = ["contour", "create_mesh", "repair_holes"]
        subset_pipeline = Pipeline([step for step in pipeline.steps if step.name not in skip_steps])

        if len(subset_pipeline) > 0:
            result = subset_pipeline.run(small_sphere_mesh, verbose=False)
            assert result.mesh is not None


class TestContextPersistence:
    """Tests that context properly persists between steps."""

    def test_curvature_persists_through_smooth(self, sphere_mesh: PhenoMesh) -> None:
        """Curvature should persist after non-destructive smooth."""
        pipeline = Pipeline(
            [
                StepConfig("compute_curvature", {"curvature_type": "mean"}),
                StepConfig("smooth", {"iterations": 5}),
            ]
        )

        result = pipeline.run(sphere_mesh, verbose=False)

        # Curvature was computed before smooth
        # Note: curvature values won't match new positions but array persists
        assert result.curvature is not None

    def test_neighbors_cleared_after_remesh(self, sphere_mesh: PhenoMesh) -> None:
        """Neighbors should be cleared after operations that change topology."""
        from phenotastic.operations import OPERATIONS

        smooth_mesh = OPERATIONS["smooth"]
        remesh = OPERATIONS["remesh"]

        ctx = PipelineContext(mesh=sphere_mesh)

        ctx = smooth_mesh(ctx, iterations=5)
        assert ctx.neighbors is None

        assert ctx.mesh is not None
        ctx.neighbors = ctx.mesh.vertex_neighbors_all(include_self=True)
        ctx = remesh(ctx, n_clusters=300)
        assert ctx.neighbors is None


class TestErrorRecovery:
    """Tests for error handling in pipelines."""

    def test_invalid_params_fails_gracefully(self, sphere_mesh: PhenoMesh) -> None:
        """Invalid parameters should raise clear errors."""
        from phenotastic.exceptions import ConfigurationError

        pipeline = Pipeline(
            [
                StepConfig("smooth", {"iterations": -5}),  # Invalid
            ]
        )

        with pytest.raises(ConfigurationError):
            pipeline.run(sphere_mesh, verbose=False)

    def test_missing_mesh_fails_gracefully(self) -> None:
        """Operations requiring mesh should fail with clear error."""
        from phenotastic.exceptions import ConfigurationError

        _ = Pipeline(
            [
                StepConfig("smooth", {"iterations": 5}),
            ]
        )

        ctx = PipelineContext()  # No mesh

        with pytest.raises(ConfigurationError, match="requires a mesh"):
            smooth_mesh(ctx, iterations=5)


class TestRepeatedOperations:
    """Tests for pipelines with repeated operations."""

    def test_multiple_smooth_operations(self, small_sphere_mesh: PhenoMesh) -> None:
        """Same operation can be used multiple times."""
        pipeline = Pipeline(
            [
                StepConfig("smooth", {"iterations": 10}),
                StepConfig("smooth", {"iterations": 20}),
                StepConfig("smooth", {"iterations": 5}),
            ]
        )

        result = pipeline.run(small_sphere_mesh, verbose=False)

        assert result.mesh is not None

    def test_alternating_smooth_remesh(self, small_sphere_mesh: PhenoMesh) -> None:
        """Alternating operations should work correctly."""
        pipeline = Pipeline(
            [
                StepConfig("smooth", {"iterations": 5}),
                StepConfig("remesh", {"n_clusters": 200}),
                StepConfig("smooth", {"iterations": 5}),
                StepConfig("remesh", {"n_clusters": 300}),
                StepConfig("smooth", {"iterations": 5}),
            ]
        )

        result = pipeline.run(small_sphere_mesh, verbose=False)

        assert result.mesh is not None
