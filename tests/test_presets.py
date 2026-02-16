"""Tests for pipeline presets."""

import numpy as np
import pytest

from phenotastic import PhenoMesh, Pipeline, get_preset_yaml, list_presets, load_preset


class TestListPresets:
    """Tests for list_presets function."""

    def test_list_presets_returns_list(self) -> None:
        """Should return a list of preset names."""
        presets = list_presets()

        assert isinstance(presets, list)
        assert len(presets) > 0

    def test_list_presets_contains_expected(self) -> None:
        """Should contain expected preset names."""
        presets = list_presets()

        assert "standard" in presets
        assert "high_quality" in presets
        assert "mesh_only" in presets
        assert "quick" in presets
        assert "full" in presets


class TestLoadPreset:
    """Tests for load_preset function."""

    def test_load_standard_preset(self) -> None:
        """Should load standard preset."""
        pipeline = load_preset("standard")

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline) > 0

    def test_load_high_quality_preset(self) -> None:
        """Should load high_quality preset."""
        pipeline = load_preset("high_quality")

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline) > 0

    def test_load_mesh_only_preset(self) -> None:
        """Should load mesh_only preset."""
        pipeline = load_preset("mesh_only")

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline) > 0

    def test_load_quick_preset(self) -> None:
        """Should load quick preset."""
        pipeline = load_preset("quick")

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline) > 0

    def test_load_full_preset(self) -> None:
        """Should load full preset."""
        pipeline = load_preset("full")

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline) > 0

    def test_load_unknown_preset_raises(self) -> None:
        """Should raise ValueError for unknown preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            load_preset("nonexistent")

    def test_load_unknown_preset_shows_available(self) -> None:
        """Error message should list available presets."""
        with pytest.raises(ValueError, match="standard"):
            load_preset("nonexistent")


class TestGetPresetYaml:
    """Tests for get_preset_yaml function."""

    def test_get_standard_yaml(self) -> None:
        """Should return YAML string for standard preset."""
        yaml_str = get_preset_yaml("standard")

        assert isinstance(yaml_str, str)
        assert "steps:" in yaml_str
        assert "smooth" in yaml_str

    def test_get_high_quality_yaml(self) -> None:
        """Should return YAML string for high_quality preset."""
        yaml_str = get_preset_yaml("high_quality")

        assert isinstance(yaml_str, str)
        assert "steps:" in yaml_str

    def test_get_unknown_preset_yaml_raises(self) -> None:
        """Should raise ValueError for unknown preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_yaml("nonexistent")


class TestPresetValidation:
    """Tests that presets are valid and can be executed."""

    @pytest.mark.parametrize("preset_name", ["standard", "high_quality", "mesh_only", "quick", "full"])
    def test_preset_validates(self, preset_name: str) -> None:
        """All presets should pass validation."""
        pipeline = load_preset(preset_name)

        # Should not raise - all operations should exist
        warnings = pipeline.validate()

        # Warnings are OK, errors are not
        assert isinstance(warnings, list)

    @pytest.mark.parametrize("preset_name", ["standard", "high_quality", "mesh_only", "quick", "full"])
    def test_preset_has_all_valid_operations(self, preset_name: str) -> None:
        """All operations in presets should be registered."""
        from phenotastic import OperationRegistry

        pipeline = load_preset(preset_name)
        registry = OperationRegistry()

        for step in pipeline.steps:
            assert step.name in registry, f"Operation '{step.name}' not found in registry"


class TestPresetExecution:
    """Tests for actually running presets."""

    def test_quick_preset_mesh_steps(self, tiny_contour: np.ndarray) -> None:
        """Quick preset mesh steps should execute successfully."""
        # Create mesh from contour first
        from phenotastic import PipelineContext, StepConfig
        from phenotastic.operations import create_mesh_from_contour

        ctx = PipelineContext(contour=tiny_contour, resolution=[1.0, 1.0, 1.0])
        ctx = create_mesh_from_contour(ctx)

        # Run just the mesh processing steps (skip domain ops on tiny mesh)
        pipeline = Pipeline(
            [
                StepConfig("clean"),
                StepConfig("remesh", {"n_clusters": 100}),
                StepConfig("smooth", {"iterations": 5}),
            ]
        )
        result = pipeline.run(ctx.mesh, verbose=False)

        assert result.mesh is not None
        assert result.mesh.n_points > 0

    def test_standard_preset_on_mesh(self, small_sphere_mesh: PhenoMesh) -> None:
        """Standard preset should work when skipping contour steps."""
        from phenotastic import StepConfig

        # Create a subset of standard that works with mesh input
        pipeline = Pipeline(
            [
                StepConfig("clean"),
                StepConfig("smooth", {"iterations": 5}),
                StepConfig("remesh", {"n_clusters": 200}),
                StepConfig("compute_curvature", {"curvature_type": "mean"}),
            ]
        )

        result = pipeline.run(small_sphere_mesh, verbose=False)

        assert result.mesh is not None
        assert result.curvature is not None

    def test_mesh_processing_preset_steps(self, sphere_mesh: PhenoMesh) -> None:
        """Mesh processing steps from presets should work on sphere mesh."""
        from phenotastic import StepConfig

        # Test core mesh processing from mesh_only preset
        pipeline = Pipeline(
            [
                StepConfig("clean"),
                StepConfig("extract_largest"),
                StepConfig("smooth", {"iterations": 10}),
                StepConfig("remesh", {"n_clusters": 500}),
                StepConfig("compute_curvature", {"curvature_type": "mean"}),
            ]
        )
        result = pipeline.run(sphere_mesh, verbose=False)

        assert result.mesh is not None
        assert result.curvature is not None


class TestPresetContents:
    """Tests for specific preset contents."""

    def test_standard_has_smoothing(self) -> None:
        """Standard preset should include smoothing."""
        pipeline = load_preset("standard")
        step_names = [s.name for s in pipeline.steps]

        assert "smooth" in step_names

    def test_standard_has_remesh(self) -> None:
        """Standard preset should include remeshing."""
        pipeline = load_preset("standard")
        step_names = [s.name for s in pipeline.steps]

        assert "remesh" in step_names

    def test_high_quality_has_multiple_smooth(self) -> None:
        """High-quality preset should have multiple smoothing steps."""
        pipeline = load_preset("high_quality")
        smooth_count = sum(1 for s in pipeline.steps if "smooth" in s.name)

        assert smooth_count >= 2

    def test_quick_is_shorter(self) -> None:
        """Quick preset should have fewer steps than standard."""
        quick = load_preset("quick")
        standard = load_preset("standard")

        assert len(quick) < len(standard)

    def test_full_has_contour(self) -> None:
        """Full preset should include contour step."""
        pipeline = load_preset("full")
        step_names = [s.name for s in pipeline.steps]

        assert "contour" in step_names
