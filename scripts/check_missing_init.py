#!/usr/bin/env -S uv run --script
#
# /// script
# dependencies = [
#   "loguru",
#   "gitpython",
# ]
# ///

"""
Check for missing __init__.py files in Python packages.

This script checks all directories under packages/ and ensures that any directory
containing .py files also has an __init__.py file to make it a proper Python module.
"""

import sys
from collections.abc import Iterable
from pathlib import Path

import git
from git.exc import GitCommandError
from loguru import logger


def find_missing_inits(files: Iterable[Path]) -> list[Path]:
    """Find all directories that contain .py files but are missing __init__.py.

    Args:
        files: List of files to check.

    Returns:
        List of directories missing __init__.py files
    """
    repo = git.Repo(Path(__file__).parent.parent)

    try:
        ignored = repo.git.check_ignore(*files)
        ignored_paths = [Path(line) for line in ignored.splitlines()]
    except GitCommandError:
        ignored_paths = []

    missing_init_dirs = []
    for directory in {Path(file).parent for file in files if file.suffix == ".py" and file not in ignored_paths}:
        if not (directory / "__init__.py").exists():
            missing_init_dirs.append(directory)

    return missing_init_dirs


def main(files: Iterable[Path]) -> int:
    """Main function to check for missing __init__.py files.

    This function checks the provided list of files and reports any directories containing .py files
    that are missing an __init__.py file, which is required for Python package/module recognition.

    Returns:
        0 if all directories with .py files have __init__.py, 1 otherwise
    """
    if not files:
        logger.error("Usage: uv run --script check_missing_init.py <filename1> <filename2> ...")
        return 0

    missing_init_dirs = find_missing_inits(files)
    if missing_init_dirs:
        logger.error(
            "\n❌ Found directories with .py files missing __init__.py:\n\n"
            + "\n".join(f"  {directory}" for directory in sorted(missing_init_dirs))
            + "\n\n💡 To fix this, add __init__.py files to the directories above.\n"
            "   You can create empty __init__.py files with:\n\n"
            + "\n".join(f"   touch {directory}/__init__.py" for directory in sorted(missing_init_dirs))
            + "\n"
        )
        return 1
    logger.success("✅ All directories with .py files have __init__.py files")
    return 0


if __name__ == "__main__":
    """How to run: uv run --script check_missing_init.py <filename1> <filename2> ..."""
    files = [Path(file) for file in sys.argv[1:]]
    sys.exit(main(files))
