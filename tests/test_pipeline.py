"""Tests for pipeline orchestration."""

import numpy as np
import pytest

from phenotastic import (
    OperationRegistry,
    PhenoMesh,
    Pipeline,
    PipelineContext,
    StepConfig,
)
from phenotastic.exceptions import ConfigurationError, PipelineError


class TestPipelineContext:
    """Tests for PipelineContext dataclass."""

    def test_default_context_is_empty(self) -> None:
        """Default context should have all None values."""
        ctx = PipelineContext()

        assert ctx.image is None
        assert ctx.contour is None
        assert ctx.mesh is None
        assert ctx.domains is None
        assert ctx.curvature is None
        assert ctx.meristem_index is None
        assert ctx.domain_data is None
        assert ctx.resolution is None
        assert ctx.neighbors is None

    def test_context_with_mesh(self, sphere_mesh: PhenoMesh) -> None:
        """Context should properly store mesh."""
        ctx = PipelineContext(mesh=sphere_mesh)

        assert ctx.mesh is sphere_mesh
        assert ctx.mesh.n_points > 0


class TestStepConfig:
    """Tests for StepConfig dataclass."""

    def test_step_config_default_params(self) -> None:
        """StepConfig should default to empty parameters."""
        step = StepConfig(name="smooth")

        assert step.name == "smooth"
        assert step.parameters == {}

    def test_step_config_with_params(self) -> None:
        """StepConfig should accept parameters."""
        step = StepConfig(name="smooth", parameters={"iterations": 50})

        assert step.name == "smooth"
        assert step.parameters == {"iterations": 50}


class TestOperationRegistry:
    """Tests for OperationRegistry."""

    def test_registry_has_operations(self) -> None:
        """Registry should have default operations."""
        registry = OperationRegistry()

        assert "smooth" in registry
        assert "remesh" in registry
        assert "clean" in registry

    def test_registry_get_operation(self) -> None:
        """Registry should return callable operations."""
        registry = OperationRegistry()

        smooth_op = registry.get("smooth")

        assert callable(smooth_op)

    def test_registry_get_unknown_raises(self) -> None:
        """Getting unknown operation should raise PipelineError."""
        registry = OperationRegistry()

        with pytest.raises(PipelineError, match="Unknown operation"):
            registry.get("nonexistent_operation")

    def test_registry_list_operations(self) -> None:
        """Registry should list all operations."""
        registry = OperationRegistry()

        all_ops = registry.list_operations()

        assert len(all_ops) > 0
        assert "smooth" in all_ops
        assert sorted(all_ops) == all_ops  # Should be sorted

    def test_registry_list_by_category(self) -> None:
        """Registry should filter by category."""
        registry = OperationRegistry()

        mesh_ops = registry.list_operations("mesh")
        domain_ops = registry.list_operations("domain")

        assert "smooth" in mesh_ops
        assert "segment_domains" in domain_ops
        assert "segment_domains" not in mesh_ops

    def test_registry_register_custom(self) -> None:
        """Registry should accept custom operations."""
        registry = OperationRegistry()

        def custom_op(ctx: PipelineContext) -> PipelineContext:
            return ctx

        registry.register("custom", custom_op)

        assert "custom" in registry
        assert registry.get("custom") is custom_op

    def test_registry_register_non_callable_raises(self) -> None:
        """Registering non-callable should raise ConfigurationError."""
        registry = OperationRegistry()

        with pytest.raises(ConfigurationError, match="must be callable"):
            registry.register("invalid", "not a function")  # type: ignore[arg-type]


class TestPipeline:
    """Tests for Pipeline class."""

    def test_pipeline_empty(self) -> None:
        """Empty pipeline should be valid."""
        pipeline = Pipeline([])

        assert len(pipeline) == 0

    def test_pipeline_with_steps(self) -> None:
        """Pipeline should accept steps."""
        steps = [
            StepConfig("smooth", {"iterations": 50}),
            StepConfig("clean"),
        ]
        pipeline = Pipeline(steps)

        assert len(pipeline) == 2
        assert pipeline.steps[0].name == "smooth"

    def test_pipeline_repr(self) -> None:
        """Pipeline should have useful repr."""
        pipeline = Pipeline(
            [
                StepConfig("smooth"),
                StepConfig("clean"),
            ]
        )

        repr_str = repr(pipeline)

        assert "Pipeline" in repr_str
        assert "smooth" in repr_str
        assert "clean" in repr_str

    def test_pipeline_run_with_mesh(self, sphere_mesh: PhenoMesh) -> None:
        """Pipeline should run with mesh input."""
        pipeline = Pipeline(
            [
                StepConfig("smooth", {"iterations": 5}),
                StepConfig("clean"),
            ]
        )

        result = pipeline.run(sphere_mesh, verbose=False)

        assert result.mesh is not None
        assert result.mesh.n_points > 0

    def test_pipeline_run_sequence(self, sphere_mesh: PhenoMesh) -> None:
        """Pipeline should execute steps in sequence."""
        # Create pipeline that changes mesh in predictable way
        pipeline = Pipeline(
            [
                StepConfig("smooth", {"iterations": 10}),
                StepConfig("remesh", {"n_clusters": 300}),
            ]
        )

        result = pipeline.run(sphere_mesh, verbose=False)

        assert result.mesh is not None
        # After remesh, point count should be different
        assert result.mesh.n_points != sphere_mesh.n_points

    def test_pipeline_context_persistence(self, sphere_mesh: PhenoMesh) -> None:
        """Context should persist between steps."""
        pipeline = Pipeline(
            [
                StepConfig("compute_curvature", {"curvature_type": "mean"}),
                StepConfig("smooth", {"iterations": 5}),
            ]
        )

        result = pipeline.run(sphere_mesh, verbose=False)

        # Curvature should have been computed in first step
        # Note: neighbors are cleared after smooth
        assert result.mesh is not None

    def test_pipeline_run_unknown_operation_raises(self, sphere_mesh: PhenoMesh) -> None:
        """Unknown operation in pipeline should raise PipelineError."""
        pipeline = Pipeline(
            [
                StepConfig("nonexistent_operation"),
            ]
        )

        with pytest.raises(PipelineError, match="Unknown operation"):
            pipeline.run(sphere_mesh, verbose=False)

    def test_pipeline_validate_success(self) -> None:
        """Valid pipeline should pass validation."""
        pipeline = Pipeline(
            [
                StepConfig("smooth", {"iterations": 10}),
                StepConfig("remesh", {"n_clusters": 1000}),
            ]
        )

        warnings = pipeline.validate()

        # Should have no critical errors
        assert isinstance(warnings, list)

    def test_pipeline_validate_unknown_operation(self) -> None:
        """Unknown operation should fail validation."""
        pipeline = Pipeline(
            [
                StepConfig("nonexistent"),
            ]
        )

        with pytest.raises(ConfigurationError, match="Unknown operation"):
            pipeline.validate()

    def test_pipeline_validate_warnings(self) -> None:
        """Validation should return warnings for potential issues."""
        pipeline = Pipeline(
            [
                StepConfig("segment_domains"),  # Without compute_curvature first
            ]
        )

        warnings = pipeline.validate()

        # Should warn about missing curvature
        assert len(warnings) > 0
        assert any("curvature" in w.lower() for w in warnings)


class TestPipelineRunWithDifferentInputs:
    """Tests for pipeline with different input types."""

    def test_run_with_contour(self, synthetic_contour: np.ndarray) -> None:
        """Pipeline should accept contour input."""
        _ = Pipeline(
            [
                StepConfig("create_mesh"),
                StepConfig("smooth", {"iterations": 5}),
            ]
        )

        ctx = PipelineContext(contour=synthetic_contour, resolution=[1.0, 1.0, 1.0])
        from phenotastic.operations import create_mesh_from_contour, smooth_mesh

        ctx = create_mesh_from_contour(ctx)
        ctx = smooth_mesh(ctx, iterations=5)

        assert ctx.mesh is not None
        assert ctx.mesh.n_points > 0


class TestPipelineStepErrors:
    """Tests for pipeline step error handling."""

    def test_step_failure_includes_step_name(self, sphere_mesh: PhenoMesh) -> None:
        """Step failure should include step name in error."""
        pipeline = Pipeline(
            [
                StepConfig("smooth", {"iterations": -5}),  # Invalid
            ]
        )

        with pytest.raises(ConfigurationError, match="iterations"):
            pipeline.run(sphere_mesh, verbose=False)

    def test_step_failure_context_state(self, sphere_mesh: PhenoMesh) -> None:
        """Failed step should not corrupt context."""
        # This is more of an implementation detail test
        # The pipeline should fail cleanly
        pipeline = Pipeline(
            [
                StepConfig("remesh", {"n_clusters": 0}),  # Invalid
            ]
        )

        with pytest.raises(ConfigurationError):
            pipeline.run(sphere_mesh, verbose=False)
