ARG BASE_IMAGE=python:3.12-slim
ARG UV_VERSION=0.9.8

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv_base

# stage: phenotastic_base
# Build environment with main dependencies and package using uv
FROM ${BASE_IMAGE} AS phenotastic_base
COPY --from=uv_base /uv /bin/

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_LOCKED=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Install system dependencies required for compiled packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

WORKDIR /app

# Install the main dependencies without the project to optimise cache
COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project

# Copy the readme
COPY README.rst ./

# Install phenotastic
COPY src ./src/
RUN uv sync --extra docs

ARG PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}/app/src/"


# stage: phenotastic_release
# Create a venv for semantic-release CI
FROM phenotastic_base AS phenotastic_release

RUN uv sync --group release

ENV UV_LOCKED=0
