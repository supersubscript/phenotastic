# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

SHELL := /usr/bin/bash
.ONESHELL:
.SHELLFLAGS := -euo pipefail -c

PROJECT      := phenotastic
SOURCES      := src/phenotastic
TESTS        := tests
WORK_DIR     := $(CURDIR)

DOCKER_IMAGE := supersubscript/$(PROJECT)
DOCKER_TARGET ?= phenotastic_base
DOCKER_PLATFORM ?= linux/amd64
DOCKER_BUILD_FLAGS ?= --platform $(DOCKER_PLATFORM)
DOCKER_RUN_FLAGS ?= --rm --shm-size=1024m

UV      := uv run
RUFF    := $(UV) ruff
MYPY    := $(UV) mypy
PYTEST  := $(UV) pytest
PRECOMMIT := $(UV) pre-commit

COV_FLAGS := --cov=$(SOURCES) --cov-branch --cov-report=term-missing

DOCS_DIR      ?= docs

# ------------------------------------------------------------------------------
# Phony targets
# ------------------------------------------------------------------------------

.PHONY: help install test format lint type-check \
        docker_build docker_dev docker_release \
        pre-commit clean docs build publish-testpypi publish-pypi

# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------

help:
	@echo "Phenotastic Development Commands"
	@echo "================================="
	@echo ""
	@echo "Setup:"
	@echo "  install            Install dependencies with uv"
	@echo ""
	@echo "Code Quality:"
	@echo "  test               Run tests with coverage"
	@echo "  format             Format code with ruff"
	@echo "  lint               Run linter checks"
	@echo "  type-check         Run mypy type checker"
	@echo "  pre-commit         Run all pre-commit hooks"
	@echo ""
	@echo "Build & Publish:"
	@echo "  build              Build sdist and wheel packages"
	@echo "  publish-testpypi   Publish to Test PyPI (for testing)"
	@echo "  publish-pypi       Publish to PyPI (production)"
	@echo ""
	@echo "Docker:"
	@echo "  docker_build       Build the base Docker image"
	@echo "  docker_dev         Build and run interactive dev container"
	@echo "  docker_release     Build the release Docker image"
	@echo ""
	@echo "Documentation:"
	@echo "  docs               Build Sphinx documentation"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean              Remove build and cache artefacts"

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------

install:
	uv sync --group dev

# ------------------------------------------------------------------------------
# Docker
# ------------------------------------------------------------------------------

docker_build:
	docker build \
		--target $(DOCKER_TARGET) \
		-t $(DOCKER_IMAGE) \
		-f Dockerfile \
		$(DOCKER_BUILD_FLAGS) \
		$(WORK_DIR)

docker_dev:
	IMAGE_ID=$$(docker build \
		--target $(DOCKER_TARGET) \
		$(DOCKER_BUILD_FLAGS) \
		-q $(WORK_DIR))
	docker run -it $(DOCKER_RUN_FLAGS) $$IMAGE_ID /bin/bash

docker_release:
	docker build \
		--target phenotastic_release \
		-t $(DOCKER_IMAGE):release \
		-f Dockerfile \
		$(DOCKER_BUILD_FLAGS) \
		$(WORK_DIR)

# ------------------------------------------------------------------------------
# Code quality
# ------------------------------------------------------------------------------

test:
	$(PYTEST) --disable-warnings $(COV_FLAGS) $(TESTS)

format:
	$(RUFF) format $(SOURCES) $(TESTS)

lint:
	$(RUFF) check $(SOURCES) $(TESTS)

type-check:
	$(MYPY) $(SOURCES)

pre-commit:
	$(PRECOMMIT) run --all-files

# ------------------------------------------------------------------------------
# Build & Publish
# ------------------------------------------------------------------------------

build:
	@echo "Building sdist and wheel..."
	rm -rf dist/
	uv build
	@echo "Build complete. Packages in dist/"
	@ls -la dist/

publish-testpypi: build
	@echo "Publishing to Test PyPI..."
	@echo "Make sure you have configured ~/.pypirc or use TWINE_* env vars"
	uv run --group release twine upload --repository testpypi dist/* --verbose
	@echo "Published to https://test.pypi.org/project/phenotastic/"

publish-pypi: build
	@echo "Publishing to PyPI..."
	@echo "WARNING: This will publish to production PyPI!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	uv run --group release twine upload dist/* --verbose
	@echo "Published to https://pypi.org/project/phenotastic/"

# ------------------------------------------------------------------------------
# Documentation
# ------------------------------------------------------------------------------

docs:
	$(UV) --extra docs sphinx-build -b html $(DOCS_DIR) $(DOCS_DIR)/_build/html

# ------------------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------------------

clean:
	@echo "Cleaning build and cache artefacts..."
	@test "$$(pwd)" != "/"  # basic safety guard

	# Python cache + editor artefacts (single traversal)
	find . -depth \
		\( \
			-type d \( \
				-name "__pycache__" -o \
				-name "*.egg-info" -o \
				-name ".eggs" -o \
				-name ".pytest_cache" -o \
				-name ".mypy_cache" -o \
				-name ".ruff_cache" -o \
				-name ".tox" -o \
				-name ".nox" -o \
				-name ".idea" -o \
				-name ".hypothesis" -o \
				-name ".ipynb_checkpoints" \
			\) -o \
			-type f \( \
				-name "*.py[co]" -o \
				-name "*.swo" -o \
				-name "*.swp" -o \
				-name ".coverage" -o \
				-name "coverage.xml" \
			\) \
		\) -exec rm -rf {} +

	# Top-level build artefacts
	rm -rf build dist htmlcov $(DOCS_DIR)/_build
