"""Preset pipeline configurations for phenotastic.

This module provides the default pipeline configuration that can be loaded
directly without creating a YAML file.
"""

from __future__ import annotations

from phenotastic.pipeline import Pipeline

# =============================================================================
# Default Pipeline (full image-to-domains workflow)
# =============================================================================

PRESET_DEFAULT = """
steps:
  # Image processing
  - name: contour
    params:
      iterations: 25
      target_resolution: [0.5, 0.5, 0.5]
      gaussian_iterations: 5
      register_stack: true
      masking_factor: 0.75

  # Mesh creation
  - name: create_mesh
    params:
      step_size: 1

  # Cleanup and processing
  - name: clean
  - name: extract_largest
  - name: repair_holes
    params:
      max_hole_edges: 100

  # Smoothing
  - name: smooth
    params:
      iterations: 100
      relaxation_factor: 0.01

  - name: remesh
    params:
      n_clusters: 10000
      subdivisions: 3

  - name: smooth
    params:
      iterations: 50
      relaxation_factor: 0.01

  # Domain segmentation
  - name: compute_curvature
    params:
      curvature_type: mean

  - name: segment_domains

  - name: merge_small
    params:
      threshold: 50
      metric: points

  - name: define_meristem

  - name: extract_domain_data
"""

# =============================================================================
# Preset Registry
# =============================================================================

PRESETS: dict[str, str] = {
    "default": PRESET_DEFAULT,
}


def load_preset(name: str = "default") -> Pipeline:
    """Load a preset pipeline configuration.

    Args:
        name: Preset name (defaults to 'default')

    Returns:
        Configured Pipeline instance

    Raises:
        ValueError: If preset name is not found
    """
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown preset: {name}. Available presets: {available}")

    return Pipeline.from_yaml_string(PRESETS[name])


def list_presets() -> list[str]:
    """List available preset names.

    Returns:
        List of preset names
    """
    return list(PRESETS.keys())


def get_preset_yaml(name: str = "default") -> str:
    """Get the YAML configuration for a preset.

    Args:
        name: Preset name (defaults to 'default')

    Returns:
        YAML configuration string

    Raises:
        ValueError: If preset name is not found
    """
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown preset: {name}. Available presets: {available}")

    return PRESETS[name]
