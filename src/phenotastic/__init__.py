"""Phenotastic: 3D plant phenotyping package.

This package provides methods for contouring 3D data, generating 2.5D meshes,
and segmenting features based on mesh curvature for analysis of early flower
organs (primordia) from shoot apical meristems in 3D images.
"""

from dotenv import load_dotenv

from phenotastic.utils.project import dev_root

__version__ = "0.5.1"

if (dev_root() / ".env").exists():
    load_dotenv(dev_root() / ".env")

# Internal imports after .env loading so config can affect them
from phenotastic.exceptions import (  # noqa: E402
    ConfigurationError,
    InvalidImageError,
    InvalidMeshError,
    PhenotasticError,
    PipelineError,
)
from phenotastic.phenomesh import PhenoMesh  # noqa: E402
from phenotastic.pipeline import (  # noqa: E402
    OperationRegistry,
    Pipeline,
    PipelineContext,
    StepConfig,
    save_pipeline_yaml,
)
from phenotastic.presets import get_preset_yaml, list_presets, load_preset  # noqa: E402

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
