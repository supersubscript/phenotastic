"""Pipeline operation decorator for auto-generating operation wrappers.

This module provides the @pipeline_operation decorator that marks PhenoMesh methods
as pipeline-compatible, enabling automatic wrapper generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from phenotastic.phenomesh import PhenoMesh
    from phenotastic.pipeline import PipelineContext


@dataclass
class OperationMeta:
    """Metadata about a pipeline operation."""

    name: str
    invalidates_neighbors: bool = True
    requires_mesh: bool = True
    category: str = "mesh"
    validators: dict[str, Callable[[Any], bool]] = field(default_factory=dict)
    description: str = ""


# Global registry of decorated operations
_REGISTERED_OPERATIONS: dict[str, OperationMeta] = {}


def pipeline_operation(
    name: str | None = None,
    *,
    invalidates_neighbors: bool = True,
    requires_mesh: bool = True,
    category: str = "mesh",
    validators: dict[str, Callable[[Any], bool]] | None = None,
    description: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a PhenoMesh method as a pipeline operation.

    Args:
        name: Operation name for pipeline configs. Defaults to method name.
        invalidates_neighbors: Whether this operation invalidates cached neighbors.
        requires_mesh: Whether this operation requires a mesh in context.
        category: Operation category ('mesh', 'domain', 'contour').
        validators: Dict mapping parameter names to validation functions.
        description: Human-readable description of the operation.

    Returns:
        Decorator function.
    """

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        op_name = name or method.__name__
        _REGISTERED_OPERATIONS[op_name] = OperationMeta(
            name=op_name,
            invalidates_neighbors=invalidates_neighbors,
            requires_mesh=requires_mesh,
            category=category,
            validators=validators or {},
            description=description or method.__doc__ or "",
        )
        method._pipeline_operation = op_name  # type: ignore[attr-defined]
        return method

    return decorator


def get_registered_operations() -> dict[str, OperationMeta]:
    """Get all registered pipeline operations."""
    return _REGISTERED_OPERATIONS.copy()


def generate_operation_wrappers(
    mesh_class: type[PhenoMesh],
) -> dict[str, Callable[[PipelineContext, Any], PipelineContext]]:
    """Generate pipeline operation wrappers from decorated PhenoMesh methods.

    Args:
        mesh_class: The PhenoMesh class to introspect.

    Returns:
        Dict mapping operation names to wrapper functions.
    """

    operations: dict[str, Callable[..., PipelineContext]] = {}

    for method_name in dir(mesh_class):
        method = getattr(mesh_class, method_name)
        if not hasattr(method, "_pipeline_operation"):
            continue

        op_name: str = method._pipeline_operation
        meta = _REGISTERED_OPERATIONS.get(op_name)
        if meta is None:
            continue

        operations[op_name] = _make_wrapper(method_name, meta)

    return operations


def _make_wrapper(
    method_name: str,
    meta: OperationMeta,
) -> Callable[..., PipelineContext]:
    """Create a pipeline wrapper for a PhenoMesh method.

    Args:
        method_name: Name of the method on PhenoMesh.
        meta: Operation metadata.

    Returns:
        Wrapper function that operates on PipelineContext.
    """
    from phenotastic.exceptions import ConfigurationError
    from phenotastic.phenomesh import PhenoMesh

    def wrapper(context: PipelineContext, **kwargs: Any) -> PipelineContext:
        # Validate mesh presence
        if meta.requires_mesh and context.mesh is None:
            raise ConfigurationError(f"Operation '{meta.name}' requires a mesh in context")

        # Run parameter validators
        for param, validator in meta.validators.items():
            if param in kwargs and not validator(kwargs[param]):
                raise ConfigurationError(f"Invalid value for parameter '{param}'")

        # Get the bound method from the mesh instance
        method = getattr(context.mesh, method_name)

        # Call method
        result = method(**kwargs)

        # Update context based on result type
        if isinstance(result, PhenoMesh):
            context.mesh = result

        # Invalidate neighbors if needed
        if meta.invalidates_neighbors:
            context.neighbors = None

        return context

    # Preserve metadata for introspection
    wrapper.__name__ = meta.name
    wrapper.__doc__ = meta.description

    return wrapper
