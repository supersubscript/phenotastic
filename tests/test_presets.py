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

    def test_list_presets_contains_default(self) -> None:
        """Should contain default preset."""
        presets = list_presets()

        assert presets == ["default"]


class TestLoadPreset:
    """Tests for load_preset function."""

    def test_load_default_preset(self) -> None:
        """Should load default preset."""
        pipeline = load_preset("default")

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline) > 0

    def test_load_preset_without_argument(self) -> None:
        """Should load default preset when no argument given."""
        pipeline = load_preset()

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline) > 0

    def test_load_unknown_preset_raises(self) -> None:
        """Should raise ValueError for unknown preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            load_preset("nonexistent")

    def test_load_unknown_preset_shows_available(self) -> None:
        """Error message should list available presets."""
        with pytest.raises(ValueError, match="default"):
            load_preset("nonexistent")


class TestGetPresetYaml:
    """Tests for get_preset_yaml function."""

    def test_get_default_yaml(self) -> None:
        """Should return YAML string for default preset."""
        yaml_str = get_preset_yaml("default")

        assert isinstance(yaml_str, str)
        assert "steps:" in yaml_str
        assert "contour" in yaml_str
        assert "smooth" in yaml_str

    def test_get_yaml_without_argument(self) -> None:
        """Should return default preset YAML when no argument given."""
        yaml_str = get_preset_yaml()

        assert isinstance(yaml_str, str)
        assert "steps:" in yaml_str

    def test_get_unknown_preset_yaml_raises(self) -> None:
        """Should raise ValueError for unknown preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_yaml("nonexistent")


class TestPresetValidation:
    """Tests that presets are valid and can be executed."""

    def test_default_preset_validates(self) -> None:
        """Default preset should pass validation."""
        pipeline = load_preset("default")

        # Should not raise - all operations should exist
        warnings = pipeline.validate()

        # Warnings are OK, errors are not
        assert isinstance(warnings, list)

    def test_default_preset_has_all_valid_operations(self) -> None:
        """All operations in default preset should be registered."""
        from phenotastic import OperationRegistry

        pipeline = load_preset("default")
        registry = OperationRegistry()

        for step in pipeline.steps:
            assert step.name in registry, f"Operation '{step.name}' not found in registry"


class TestPresetExecution:
    """Tests for actually running presets."""

    def test_default_preset_mesh_steps(self, tiny_contour: np.ndarray) -> None:
        """Default preset mesh steps should execute successfully."""
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

    def test_default_preset_on_mesh(self, small_sphere_mesh: PhenoMesh) -> None:
        """Default preset processing should work when starting with mesh input."""
        from phenotastic import StepConfig

        # Create a subset of default that works with mesh input
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

    def test_mesh_processing_steps(self, sphere_mesh: PhenoMesh) -> None:
        """Mesh processing steps from default preset should work on sphere mesh."""
        from phenotastic import StepConfig

        # Test core mesh processing from default preset
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

    def test_default_has_contour(self) -> None:
        """Default preset should include contour step."""
        pipeline = load_preset("default")
        step_names = [s.name for s in pipeline.steps]

        assert "contour" in step_names

    def test_default_has_smoothing(self) -> None:
        """Default preset should include smoothing."""
        pipeline = load_preset("default")
        step_names = [s.name for s in pipeline.steps]

        assert "smooth" in step_names

    def test_default_has_remesh(self) -> None:
        """Default preset should include remeshing."""
        pipeline = load_preset("default")
        step_names = [s.name for s in pipeline.steps]

        assert "remesh" in step_names

    def test_default_has_domain_segmentation(self) -> None:
        """Default preset should include domain segmentation."""
        pipeline = load_preset("default")
        step_names = [s.name for s in pipeline.steps]

        assert "segment_domains" in step_names
