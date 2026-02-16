"""Phenotastic: 3D plant phenotyping package.

This package provides methods for contouring 3D data, generating 2.5D meshes,
and segmenting features based on mesh curvature for analysis of early flower
organs (primordia) from shoot apical meristems in 3D images.
"""

__version__ = "0.3.0"

from phenotastic.exceptions import (
    ConfigurationError,
    InvalidImageError,
    InvalidMeshError,
    PhenotasticError,
    PipelineError,
)
from phenotastic.phenomesh import PhenoMesh
from phenotastic.pipeline import (
    OperationRegistry,
    Pipeline,
    PipelineContext,
    StepConfig,
    save_pipeline_yaml,
)
from phenotastic.presets import get_preset_yaml, list_presets, load_preset

__all__ = [
    "ConfigurationError",
    "InvalidImageError",
    "InvalidMeshError",
    "OperationRegistry",
    "PhenoMesh",
    "PhenotasticError",
    "Pipeline",
    "PipelineContext",
    "PipelineError",
    "StepConfig",
    "get_preset_yaml",
    "list_presets",
    "load_preset",
    "save_pipeline_yaml",
]
