"""Preset pipeline configurations for phenotastic.

This module provides common pipeline configurations that can be loaded
directly without creating a YAML file.
"""

from __future__ import annotations

from phenotastic.pipeline import Pipeline

# =============================================================================
# Standard Pipeline
# =============================================================================

PRESET_STANDARD = """
steps:
  # Create mesh from contour
  - name: create_mesh
    params:
      step_size: 1

  # Initial cleanup
  - name: clean
  - name: extract_largest

  # Smoothing and remeshing
  - name: smooth
    params:
      iterations: 100
      relaxation_factor: 0.01

  - name: remesh
    params:
      n_clusters: 10000
      subdivisions: 3

  # Final smoothing
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
    params:
      method: center_of_mass

  - name: extract_domaindata
"""

# =============================================================================
# High Quality Pipeline
# =============================================================================

PRESET_HIGH_QUALITY = """
steps:
  # Create mesh from contour
  - name: create_mesh
    params:
      step_size: 1

  # Aggressive cleanup
  - name: clean
  - name: extract_largest
  - name: repair_holes
    params:
      max_hole_edges: 100
      refine: true

  # Multi-pass smoothing with remeshing
  - name: smooth
    params:
      iterations: 150
      relaxation_factor: 0.01

  - name: remesh
    params:
      n_clusters: 15000
      subdivisions: 3

  - name: smooth
    params:
      iterations: 100
      relaxation_factor: 0.008

  - name: remesh
    params:
      n_clusters: 20000
      subdivisions: 3

  - name: smooth_taubin
    params:
      iterations: 50
      pass_band: 0.1

  # Normal-based filtering
  - name: rotate
    params:
      axis: y
      angle: -90

  - name: remove_normals
    params:
      threshold_angle: 60
      angle_type: polar

  - name: rotate
    params:
      axis: y
      angle: 90

  # Repair after filtering
  - name: make_manifold
    params:
      hole_edges: 100

  - name: extract_largest
  - name: repair_holes
    params:
      max_hole_edges: 100

  # Final smoothing
  - name: smooth
    params:
      iterations: 50
      relaxation_factor: 0.005

  - name: smooth_boundary
    params:
      iterations: 20
      sigma: 0.1

  # Domain segmentation with filtering
  - name: compute_curvature
    params:
      curvature_type: mean

  - name: filter_scalars
    params:
      function: median
      iterations: 3

  - name: segment_domains

  - name: merge_small
    params:
      threshold: 100
      metric: points

  - name: merge_angles
    params:
      threshold: 15

  - name: merge_engulfing
    params:
      threshold: 0.85

  - name: define_meristem
    params:
      method: center_of_mass

  - name: extract_domaindata
"""

# =============================================================================
# Mesh-Only Pipeline (for when you already have a mesh)
# =============================================================================

PRESET_MESH_ONLY = """
steps:
  # Cleanup
  - name: clean
  - name: extract_largest

  # Smoothing
  - name: smooth
    params:
      iterations: 100
      relaxation_factor: 0.01

  - name: remesh
    params:
      n_clusters: 10000
      subdivisions: 3

  # Domain segmentation
  - name: compute_curvature
    params:
      curvature_type: mean

  - name: segment_domains

  - name: merge_small
    params:
      threshold: 50

  - name: extract_domaindata
"""

# =============================================================================
# Quick Pipeline (for fast previews)
# =============================================================================

PRESET_QUICK = """
steps:
  - name: create_mesh
  - name: clean
  - name: remesh
    params:
      n_clusters: 5000
      subdivisions: 2
  - name: smooth
    params:
      iterations: 50
  - name: compute_curvature
  - name: segment_domains
"""

# =============================================================================
# Full Pipeline (from image to domains)
# =============================================================================

PRESET_FULL = """
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

  - name: extract_domaindata
"""

# =============================================================================
# Preset Registry
# =============================================================================

PRESETS: dict[str, str] = {
    "standard": PRESET_STANDARD,
    "high_quality": PRESET_HIGH_QUALITY,
    "mesh_only": PRESET_MESH_ONLY,
    "quick": PRESET_QUICK,
    "full": PRESET_FULL,
}


def load_preset(name: str) -> Pipeline:
    """Load a preset pipeline configuration.

    Args:
        name: Preset name ('standard', 'high_quality', 'mesh_only', 'quick', 'full')

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


def get_preset_yaml(name: str) -> str:
    """Get the YAML configuration for a preset.

    Args:
        name: Preset name

    Returns:
        YAML configuration string

    Raises:
        ValueError: If preset name is not found
    """
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown preset: {name}. Available presets: {available}")

    return PRESETS[name]
