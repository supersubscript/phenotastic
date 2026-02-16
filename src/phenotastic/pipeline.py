"""Pipeline orchestration for phenotastic.

This module provides the Pipeline class for executing sequences of operations
defined in YAML configuration files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml  # type: ignore[import-untyped]
from loguru import logger

from phenotastic.exceptions import ConfigurationError, PipelineError
from phenotastic.operations import OPERATIONS

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    from numpy.typing import NDArray

    from phenotastic.phenomesh import PhenoMesh


@dataclass
class PipelineContext:
    """Holds state passed between pipeline steps.

    Attributes:
        image: Input 3D image array
        contour: Binary contour array
        mesh: PhenoMesh object
        domains: Domain labels array
        curvature: Curvature values array
        meristem_index: Index of the meristem domain
        domain_data: DataFrame with domain measurements
        resolution: Spatial resolution [x, y, z]
        neighbors: Cached vertex neighbors for performance
    """

    image: NDArray[Any] | None = None
    contour: NDArray[np.bool_] | None = None
    mesh: PhenoMesh | None = None
    domains: NDArray[np.integer[Any]] | None = None
    curvature: NDArray[np.floating[Any]] | None = None
    meristem_index: int | None = None
    domain_data: pd.DataFrame | None = None
    resolution: list[float] | None = None
    neighbors: list[NDArray[np.intp]] | None = None


@dataclass
class StepConfig:
    """Configuration for a single pipeline step.

    Attributes:
        name: Name of the operation to execute
        params: Parameters to pass to the operation
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


class OperationRegistry:
    """Registry of available pipeline operations.

    Manages the mapping from operation names to callable functions.
    Supports registering custom operations.
    """

    def __init__(self) -> None:
        """Initialize with default operations."""
        self._operations: dict[str, Callable[..., PipelineContext]] = dict(OPERATIONS)

    def register(self, name: str, func: Callable[..., PipelineContext]) -> None:
        """Register a custom operation.

        Args:
            name: Operation name for use in pipeline configs
            func: Callable that takes (PipelineContext, **params) and returns PipelineContext
        """
        if not callable(func):
            raise ConfigurationError(f"Operation {name} must be callable")
        self._operations[name] = func

    def get(self, name: str) -> Callable[..., PipelineContext]:
        """Get an operation by name.

        Args:
            name: Operation name

        Returns:
            The operation callable

        Raises:
            PipelineError: If operation name is not found
        """
        if name not in self._operations:
            available = ", ".join(sorted(self._operations.keys()))
            raise PipelineError(f"Unknown operation: {name}. Available operations: {available}")
        return self._operations[name]

    def list_operations(self, category: str = "all") -> list[str]:
        """List available operation names.

        Args:
            category: Filter by category ('all', 'contour', 'mesh', 'domain')

        Returns:
            List of operation names
        """
        # Define categories
        contour_ops = {"contour", "create_mesh", "create_cellular_mesh"}
        domain_ops = {
            "compute_curvature",
            "filter_scalars",
            "segment_domains",
            "merge_angles",
            "merge_distance",
            "merge_small",
            "merge_engulfing",
            "merge_disconnected",
            "merge_depth",
            "define_meristem",
            "extract_domaindata",
        }

        if category == "contour":
            return sorted([op for op in self._operations if op in contour_ops])
        if category == "domain":
            return sorted([op for op in self._operations if op in domain_ops])
        if category == "mesh":
            return sorted([op for op in self._operations if op not in contour_ops and op not in domain_ops])
        return sorted(self._operations.keys())

    def __contains__(self, name: str) -> bool:
        """Check if operation is registered."""
        return name in self._operations


class Pipeline:
    """Recipe-style pipeline executor.

    Executes a sequence of operations defined as StepConfig objects.
    Each operation receives the PipelineContext, modifies it, and returns it.

    Example:
        >>> steps = [
        ...     StepConfig("smooth", {"iterations": 100}),
        ...     StepConfig("remesh", {"n_clusters": 5000}),
        ...     StepConfig("smooth", {"iterations": 50}),
        ... ]
        >>> pipeline = Pipeline(steps)
        >>> result = pipeline.run(mesh)
    """

    def __init__(
        self,
        steps: list[StepConfig],
        registry: OperationRegistry | None = None,
    ) -> None:
        """Initialize pipeline with steps.

        Args:
            steps: List of step configurations
            registry: Operation registry (uses default if not provided)
        """
        self.steps = steps
        self._registry = registry if registry is not None else OperationRegistry()

    @classmethod
    def from_yaml(cls, path: str | Path) -> Pipeline:
        """Load pipeline from YAML configuration file.

        Args:
            path: Path to YAML file

        Returns:
            Configured Pipeline instance

        Raises:
            ConfigurationError: If YAML is invalid or missing required keys
        """
        path = Path(path)

        if not path.exists():
            raise ConfigurationError(f"Pipeline config file not found: {path}")

        try:
            with open(path) as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML syntax: {e}") from e

        if config is None:
            raise ConfigurationError("Empty configuration file")

        if "steps" not in config:
            raise ConfigurationError("Configuration must have 'steps' key")

        if not isinstance(config["steps"], list):
            raise ConfigurationError("'steps' must be a list")

        steps: list[StepConfig] = []
        for i, step_dict in enumerate(config["steps"]):
            if not isinstance(step_dict, dict):
                raise ConfigurationError(f"Step {i} must be a dictionary")

            if "name" not in step_dict:
                raise ConfigurationError(f"Step {i} must have 'name' key")

            name = step_dict["name"]
            params = step_dict.get("params", {})

            if params is None:
                params = {}

            if not isinstance(params, dict):
                raise ConfigurationError(f"Step {i} 'params' must be a dictionary")

            steps.append(StepConfig(name=name, params=params))

        return cls(steps)

    @classmethod
    def from_yaml_string(cls, yaml_string: str) -> Pipeline:
        """Load pipeline from YAML string.

        Args:
            yaml_string: YAML configuration as string

        Returns:
            Configured Pipeline instance
        """
        try:
            config = yaml.safe_load(yaml_string)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML syntax: {e}") from e

        if config is None or "steps" not in config:
            raise ConfigurationError("Configuration must have 'steps' key")

        steps: list[StepConfig] = []
        for step_dict in config["steps"]:
            name = step_dict["name"]
            params = step_dict.get("params", {}) or {}
            steps.append(StepConfig(name=name, params=params))

        return cls(steps)

    def run(
        self,
        input_data: str | Path | NDArray[Any] | PhenoMesh,
        verbose: bool = True,
        resolution: list[float] | None = None,
    ) -> PipelineContext:
        """Execute all pipeline steps in sequence.

        Args:
            input_data: Input data - can be:
                - Path to image file
                - NumPy array (image or contour)
                - PhenoMesh object
            verbose: Print progress information
            resolution: Spatial resolution [x, y, z] (for image/contour input)

        Returns:
            PipelineContext with all results

        Raises:
            PipelineError: If a step fails or input is invalid
        """
        import tifffile as tiff

        from phenotastic.phenomesh import PhenoMesh

        # Initialize context based on input type
        context = PipelineContext(resolution=resolution)

        if isinstance(input_data, PhenoMesh):
            context.mesh = input_data
        elif isinstance(input_data, np.ndarray):
            # Determine if it's an image or contour based on dtype
            if input_data.dtype == bool:
                context.contour = input_data
            else:
                context.image = input_data
        elif isinstance(input_data, (str, Path)):
            # Load image from file
            try:
                context.image = tiff.imread(str(input_data))
                context.image = np.squeeze(context.image)
            except Exception as e:
                raise PipelineError(f"Failed to load image: {e}") from e
        else:
            raise PipelineError(f"Unsupported input type: {type(input_data)}")

        # Execute steps
        n_steps = len(self.steps)
        for i, step in enumerate(self.steps):
            if verbose:
                logger.info(f"Step {i + 1}/{n_steps}: {step.name}")

            try:
                operation = self._registry.get(step.name)
                context = operation(context, **step.params)
            except (ConfigurationError, PipelineError):
                raise
            except Exception as e:
                raise PipelineError(f"Step '{step.name}' failed: {e}") from e

        return context

    def validate(self) -> list[str]:
        """Validate pipeline configuration without running.

        Returns:
            List of validation warnings (empty if valid)

        Raises:
            ConfigurationError: If configuration is invalid
        """
        warnings: list[str] = []

        for i, step in enumerate(self.steps):
            # Check operation exists
            if step.name not in self._registry:
                raise ConfigurationError(f"Step {i}: Unknown operation '{step.name}'")

            # Check for common issues
            if step.name == "create_mesh" and i == 0:
                warnings.append("create_mesh as first step requires contour input")

            if step.name == "segment_domains":
                # Check if compute_curvature comes before
                has_curvature = any(s.name == "compute_curvature" for s in self.steps[:i])
                if not has_curvature:
                    warnings.append(f"Step {i}: segment_domains may need compute_curvature before it")

        return warnings

    def __len__(self) -> int:
        """Return number of steps."""
        return len(self.steps)

    def __repr__(self) -> str:
        """Return string representation."""
        step_names = [s.name for s in self.steps]
        return f"Pipeline(steps={step_names})"


def save_pipeline_yaml(pipeline: Pipeline, path: str | Path) -> None:
    """Save pipeline configuration to YAML file.

    Args:
        pipeline: Pipeline to save
        path: Output file path
    """
    config = {"steps": [{"name": step.name, "params": step.params} for step in pipeline.steps]}

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
