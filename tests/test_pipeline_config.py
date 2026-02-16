"""Tests for pipeline YAML configuration loading."""

from pathlib import Path

import pytest

from phenotastic import Pipeline
from phenotastic.exceptions import ConfigurationError


class TestPipelineFromYaml:
    """Tests for loading pipelines from YAML files."""

    def test_load_valid_yaml(self, valid_pipeline_yaml: str) -> None:
        """Should load valid YAML file."""
        pipeline = Pipeline.from_yaml(valid_pipeline_yaml)

        assert len(pipeline) == 3
        assert pipeline.steps[0].name == "smooth"
        assert pipeline.steps[0].parameters == {"iterations": 10}
        assert pipeline.steps[1].name == "remesh"
        assert pipeline.steps[2].name == "clean"

    def test_load_yaml_with_params(self, valid_pipeline_yaml: str) -> None:
        """Should correctly parse params from YAML."""
        pipeline = Pipeline.from_yaml(valid_pipeline_yaml)

        smooth_step = pipeline.steps[0]
        assert smooth_step.parameters["iterations"] == 10

        remesh_step = pipeline.steps[1]
        assert remesh_step.parameters["n_clusters"] == 500

    def test_load_yaml_without_params(self, pipeline_yaml_no_params: str) -> None:
        """Should handle steps without params."""
        pipeline = Pipeline.from_yaml(pipeline_yaml_no_params)

        assert len(pipeline) == 3
        assert pipeline.steps[0].parameters == {}
        assert pipeline.steps[1].parameters == {}

    def test_load_nonexistent_file_raises(self) -> None:
        """Should raise ConfigurationError for missing file."""
        with pytest.raises(ConfigurationError, match="not found"):
            Pipeline.from_yaml("/nonexistent/path/config.yaml")

    def test_load_invalid_syntax_raises(self, invalid_pipeline_yaml_syntax: str) -> None:
        """Should raise ConfigurationError for invalid YAML syntax."""
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            Pipeline.from_yaml(invalid_pipeline_yaml_syntax)

    def test_load_no_steps_key_raises(self, invalid_pipeline_yaml_no_steps: str) -> None:
        """Should raise ConfigurationError when 'steps' key is missing."""
        with pytest.raises(ConfigurationError, match="must have 'steps'"):
            Pipeline.from_yaml(invalid_pipeline_yaml_no_steps)

    def test_load_empty_file_raises(self, tmp_path: Path) -> None:
        """Should raise ConfigurationError for empty file."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        with pytest.raises(ConfigurationError, match="Empty configuration"):
            Pipeline.from_yaml(str(empty_file))

    def test_load_steps_not_list_raises(self, tmp_path: Path) -> None:
        """Should raise ConfigurationError when steps is not a list."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("steps: 'not a list'")

        with pytest.raises(ConfigurationError, match="'steps' must be a list"):
            Pipeline.from_yaml(str(bad_yaml))

    def test_load_step_without_name_raises(self, tmp_path: Path) -> None:
        """Should raise ConfigurationError when step has no name."""
        bad_yaml = tmp_path / "no_name.yaml"
        bad_yaml.write_text("""
steps:
  - params:
      iterations: 10
""")

        with pytest.raises(ConfigurationError, match="must have 'name'"):
            Pipeline.from_yaml(str(bad_yaml))

    def test_load_step_params_not_dict_raises(self, tmp_path: Path) -> None:
        """Should raise ConfigurationError when params is not a dict."""
        bad_yaml = tmp_path / "bad_params.yaml"
        bad_yaml.write_text("""
steps:
  - name: smooth
    params: [1, 2, 3]
""")

        with pytest.raises(ConfigurationError, match="'params' must be a dictionary"):
            Pipeline.from_yaml(str(bad_yaml))

    def test_load_step_not_dict_raises(self, tmp_path: Path) -> None:
        """Should raise ConfigurationError when step is not a dict."""
        bad_yaml = tmp_path / "step_not_dict.yaml"
        bad_yaml.write_text("""
steps:
  - smooth
  - clean
""")

        with pytest.raises(ConfigurationError, match="must be a dictionary"):
            Pipeline.from_yaml(str(bad_yaml))


class TestPipelineFromYamlString:
    """Tests for loading pipelines from YAML strings."""

    def test_load_valid_string(self) -> None:
        """Should load valid YAML string."""
        yaml_string = """
steps:
  - name: smooth
    params:
      iterations: 25
  - name: clean
"""
        pipeline = Pipeline.from_yaml_string(yaml_string)

        assert len(pipeline) == 2
        assert pipeline.steps[0].name == "smooth"
        assert pipeline.steps[0].parameters["iterations"] == 25

    def test_load_empty_string_raises(self) -> None:
        """Should raise ConfigurationError for empty string."""
        with pytest.raises(ConfigurationError, match="must have 'steps'"):
            Pipeline.from_yaml_string("")

    def test_load_invalid_string_raises(self) -> None:
        """Should raise ConfigurationError for invalid YAML."""
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            Pipeline.from_yaml_string("steps: [invalid")


class TestSavePipelineYaml:
    """Tests for saving pipeline configuration."""

    def test_save_pipeline_yaml(self, tmp_path: Path) -> None:
        """Should save pipeline to YAML file."""
        from phenotastic import StepConfig, save_pipeline_yaml

        pipeline = Pipeline(
            [
                StepConfig("smooth", {"iterations": 100}),
                StepConfig("clean"),
            ]
        )

        output_path = tmp_path / "output.yaml"
        save_pipeline_yaml(pipeline, output_path)

        # Verify file was created
        assert output_path.exists()

        # Verify content can be loaded back
        loaded = Pipeline.from_yaml(str(output_path))
        assert len(loaded) == 2
        assert loaded.steps[0].name == "smooth"
        assert loaded.steps[0].parameters["iterations"] == 100

    def test_roundtrip_preserves_config(self, tmp_path: Path) -> None:
        """Saving and loading should preserve configuration."""
        from phenotastic import StepConfig, save_pipeline_yaml

        original = Pipeline(
            [
                StepConfig("smooth", {"iterations": 50, "relaxation_factor": 0.02}),
                StepConfig("remesh", {"n_clusters": 5000}),
                StepConfig("compute_curvature", {"curvature_type": "mean"}),
            ]
        )

        output_path = tmp_path / "roundtrip.yaml"
        save_pipeline_yaml(original, output_path)
        loaded = Pipeline.from_yaml(str(output_path))

        assert len(loaded) == len(original)
        for orig_step, load_step in zip(original.steps, loaded.steps, strict=True):
            assert orig_step.name == load_step.name
            assert orig_step.parameters == load_step.parameters


class TestYamlEdgeCases:
    """Tests for YAML edge cases."""

    def test_null_params_handled(self, tmp_path: Path) -> None:
        """Should handle null params gracefully."""
        yaml_file = tmp_path / "null_params.yaml"
        yaml_file.write_text("""
steps:
  - name: smooth
    params: null
  - name: clean
    params:
""")

        pipeline = Pipeline.from_yaml(str(yaml_file))

        assert len(pipeline) == 2
        assert pipeline.steps[0].parameters == {}
        assert pipeline.steps[1].parameters == {}

    def test_nested_params_preserved(self, tmp_path: Path) -> None:
        """Should preserve nested param structures."""
        yaml_file = tmp_path / "nested.yaml"
        yaml_file.write_text("""
steps:
  - name: contour
    params:
      target_resolution: [0.5, 0.5, 0.5]
      gaussian_sigma: [1.0, 1.0, 1.0]
""")

        pipeline = Pipeline.from_yaml(str(yaml_file))

        assert pipeline.steps[0].parameters["target_resolution"] == [0.5, 0.5, 0.5]
        assert pipeline.steps[0].parameters["gaussian_sigma"] == [1.0, 1.0, 1.0]

    def test_boolean_params_preserved(self, tmp_path: Path) -> None:
        """Should preserve boolean param values."""
        yaml_file = tmp_path / "booleans.yaml"
        yaml_file.write_text("""
steps:
  - name: smooth
    params:
      feature_smoothing: false
      boundary_smoothing: true
""")

        pipeline = Pipeline.from_yaml(str(yaml_file))

        assert pipeline.steps[0].parameters["feature_smoothing"] is False
        assert pipeline.steps[0].parameters["boundary_smoothing"] is True
