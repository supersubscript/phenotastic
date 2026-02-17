"""Project utility functions."""

import functools
from collections.abc import Callable
from pathlib import Path
from typing import overload

from loguru import logger

PathLike = str | Path


def dev_root() -> Path:
    """Return the development root directory of this project."""
    return Path(__file__).resolve().parents[3]


def package_root() -> Path:
    """Return the package root directory."""
    return Path(__file__).resolve().parents[1]


@overload
def deprecated[T: Callable[..., object]](func: T) -> T: ...
@overload
def deprecated[T: Callable[..., object]](*, use_instead: str) -> Callable[[T], T]: ...
def deprecated[T: Callable[..., object]](func: T | None = None, *, use_instead: str = "") -> T | Callable[[T], T]:
    """Decorator for indicating deprecated functions.

    Can be used as `@deprecated` or `@deprecated(use_instead="...")`.

    Args:
        func: The function to wrap (when used without parentheses).
        use_instead: The function to use instead.

    Returns:
        The wrapped function, or a decorator if called with arguments.
    """

    def decorator(fn: T) -> T:
        @functools.wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> object:
            message = f"Function '{fn.__name__}' is deprecated and will be removed in a future version."
            if use_instead:
                message += f" Use `{use_instead}` instead."
            logger.warning(message)
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator


@overload
def validate_paths(inputs: PathLike | None) -> Path: ...
@overload
def validate_paths(inputs: list[PathLike | None]) -> list[Path]: ...
def validate_paths(
    inputs: PathLike | list[PathLike | None] | None = None,
) -> Path | list[Path]:
    """Validate and create directory paths.

    Ensures that input paths are properly formatted as Path objects and creates
    any missing parent directories.

    Args:
        inputs: Single path, list of paths, or None.

    Returns:
        Validated Path object(s). For file paths, parent directories are
        created if they don't exist. For directory paths, the directories
        themselves are created.

    Example:
        >>> validate_paths("data/output")
        >>> validate_paths(["file1.csv", "file2.csv"])
    """
    if isinstance(inputs, list):
        return [validate_paths(p) for p in inputs]

    path = Path(inputs or ".")
    if not path.suffix:
        path.mkdir(exist_ok=True, parents=True)
    else:
        path.parent.mkdir(exist_ok=True, parents=True)
    return path
