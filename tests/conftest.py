"""Shared test fixtures for phenotastic tests."""

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from phenotastic import PhenoMesh, Pipeline, PipelineContext, StepConfig


@pytest.fixture
def sphere_mesh() -> PhenoMesh:
    """Simple sphere mesh for testing."""
    return PhenoMesh(pv.Sphere(radius=1.0, theta_resolution=20, phi_resolution=20))


@pytest.fixture
def cube_mesh() -> PhenoMesh:
    """Simple cube mesh for testing."""
    return PhenoMesh(pv.Box())


@pytest.fixture
def small_sphere_mesh() -> PhenoMesh:
    """Very small sphere mesh for fast tests."""
    return PhenoMesh(pv.Sphere(radius=1.0, theta_resolution=8, phi_resolution=8))


@pytest.fixture
def synthetic_contour() -> np.ndarray:
    """Small synthetic binary contour for testing."""
    contour = np.zeros((20, 20, 20), dtype=bool)
    contour[5:15, 5:15, 5:15] = True
    return contour


@pytest.fixture
def tiny_contour() -> np.ndarray:
    """Very small contour for fast tests."""
    contour = np.zeros((10, 10, 10), dtype=bool)
    contour[3:7, 3:7, 3:7] = True
    return contour


@pytest.fixture
def pipeline_context(sphere_mesh: PhenoMesh) -> PipelineContext:
    """Pre-initialized pipeline context with a mesh."""
    return PipelineContext(mesh=sphere_mesh)


@pytest.fixture
def simple_pipeline() -> Pipeline:
    """Simple two-step pipeline for testing."""
    return Pipeline(
        [
            StepConfig("smooth", {"iterations": 5}),
            StepConfig("clean"),
        ]
    )


@pytest.fixture
def valid_pipeline_yaml(tmp_path: Path) -> str:
    """Create a valid pipeline YAML file."""
    yaml_content = """
steps:
  - name: smooth
    params:
      iterations: 10
  - name: remesh
    params:
      n_clusters: 500
  - name: clean
"""
    config_path = tmp_path / "valid_pipeline.yaml"
    config_path.write_text(yaml_content)
    return str(config_path)


@pytest.fixture
def invalid_pipeline_yaml_no_steps(tmp_path: Path) -> str:
    """Create an invalid pipeline YAML file (missing steps)."""
    yaml_content = """
other_key: value
"""
    config_path = tmp_path / "invalid_no_steps.yaml"
    config_path.write_text(yaml_content)
    return str(config_path)


@pytest.fixture
def invalid_pipeline_yaml_syntax(tmp_path: Path) -> str:
    """Create an invalid pipeline YAML file (bad syntax)."""
    yaml_content = """
steps:
  - name: [invalid
"""
    config_path = tmp_path / "invalid_syntax.yaml"
    config_path.write_text(yaml_content)
    return str(config_path)


@pytest.fixture
def pipeline_yaml_no_params(tmp_path: Path) -> str:
    """Create a pipeline YAML with steps that have no params."""
    yaml_content = """
steps:
  - name: smooth
  - name: clean
  - name: extract_largest
"""
    config_path = tmp_path / "no_params.yaml"
    config_path.write_text(yaml_content)
    return str(config_path)
