===========
Phenotastic
===========

| `Documentation <https://supersubscript.github.io/phenotastic/>`_

3D plant phenotyping package for segmentation of early flower organs (primordia)
from shoot apical meristems in 3D images.

Features
--------

- **3D Image Contouring**: Morphological active contour methods for extracting surfaces from 3D image stacks
- **Mesh Processing**: Smoothing, remeshing, and repair operations for 3D meshes
- **Domain Segmentation**: Curvature-based segmentation of meshes into regions (domains)
- **Pipeline System**: Configurable recipe-style pipelines for reproducible workflows

Installation
------------

.. code-block:: bash

    uv pip install phenotastic

Or install from source:

.. code-block:: bash

    git clone https://github.com/supersubscript/phenotastic.git
    cd phenotastic
    uv pip install -e ".[dev]"

Quick Start
-----------

Using the Python API
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from phenotastic import PhenoMesh, Pipeline, load_preset
    import pyvista as pv

    # Load a mesh
    polydata = pv.read("my_mesh.vtk")
    mesh = PhenoMesh(polydata)

    # Process with the default pipeline
    pipeline = load_preset()
    result = pipeline.run(mesh)

    # Access results
    print(f"Mesh has {result.mesh.n_points} points")
    print(f"Found {len(result.domains.unique())} domains")

Using the CLI
~~~~~~~~~~~~~

.. code-block:: bash

    # Run with default pipeline
    phenotastic run image.tif --output results/

    # Run with custom config
    phenotastic run image.tif --config my_pipeline.yaml

    # Generate a config template
    phenotastic init-config my_pipeline.yaml

    # List available operations
    phenotastic list-operations

    # List available presets
    phenotastic list-presets

    # Validate configuration
    phenotastic validate my_pipeline.yaml

    # View a mesh interactively
    phenotastic view mesh.vtk --scalars curvature

Pipeline Configuration
----------------------

Phenotastic uses a recipe-style YAML configuration for defining pipelines.
Each step specifies an operation name and optional parameters.

Example Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    steps:
      # Create mesh from contour
      - name: create_mesh
        params:
          step_size: 1

      # Smoothing
      - name: smooth
        params:
          iterations: 100
          relaxation_factor: 0.01

      # Remesh to regularize faces
      - name: remesh
        params:
          n_clusters: 10000

      # More smoothing
      - name: smooth
        params:
          iterations: 50

      # Domain segmentation
      - name: compute_curvature
        params:
          curvature_type: mean

      - name: segment_domains

      - name: merge_small
        params:
          threshold: 50

      - name: extract_domaindata

Default Pipeline
~~~~~~~~~~~~~~~~

Phenotastic provides a default pipeline that includes the full workflow from
3D image to domain analysis. The default pipeline is automatically used when
calling ``load_preset()`` without arguments or when running the CLI.

Available Operations
--------------------

Image/Contour Operations
~~~~~~~~~~~~~~~~~~~~~~~~

- ``contour``: Generate binary contour from 3D image using morphological active contours
- ``create_mesh``: Create mesh from contour using marching cubes
- ``create_cellular_mesh``: Create mesh from segmented image (one mesh per cell)

Mesh Processing Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``smooth``: Laplacian smoothing
- ``smooth_taubin``: Taubin smoothing (less shrinkage than Laplacian)
- ``smooth_boundary``: Smooth only boundary edges
- ``remesh``: Regularize faces using ACVD algorithm
- ``decimate``: Reduce mesh complexity by removing faces
- ``subdivide``: Increase mesh resolution by subdividing faces
- ``repair_holes``: Fill small holes in the mesh
- ``repair``: Full mesh repair using MeshFix
- ``make_manifold``: Remove non-manifold edges
- ``filter_curvature``: Remove vertices outside curvature threshold range
- ``remove_normals``: Remove vertices based on normal angle
- ``remove_bridges``: Remove triangles where all vertices are on the boundary
- ``remove_tongues``: Remove tongue-like artifacts
- ``extract_largest``: Keep only the largest connected component
- ``clean``: Remove degenerate cells
- ``triangulate``: Convert all faces to triangles
- ``compute_normals``: Compute surface normals
- ``flip_normals``: Flip all surface normals
- ``correct_normal_orientation``: Correct normal orientation relative to an axis
- ``rotate``: Rotate mesh around an axis
- ``clip``: Clip mesh with a plane
- ``erode``: Erode mesh by removing boundary points
- ``ecft``: ExtractLargest, Clean, FillHoles, Triangulate (combined operation)

Domain Operations
~~~~~~~~~~~~~~~~~

- ``compute_curvature``: Compute mesh curvature (mean, gaussian, minimum, maximum)
- ``filter_scalars``: Apply filter to curvature field (median, mean, minmax, maxmin)
- ``segment_domains``: Create domains via steepest ascent on curvature field
- ``merge_angles``: Merge domains within angular threshold from meristem
- ``merge_distance``: Merge domains within spatial distance threshold
- ``merge_small``: Merge small domains to their largest neighbor
- ``merge_engulfing``: Merge domains mostly encircled by a neighbor
- ``merge_disconnected``: Connect domains isolated from meristem
- ``merge_depth``: Merge domains with similar depth values
- ``define_meristem``: Identify the meristem domain
- ``extract_domaindata``: Extract geometric measurements for each domain

PhenoMesh Class
---------------

``PhenoMesh`` extends PyVista's ``PolyData`` class, adding convenient methods
for 3D plant phenotyping workflows. It can be used anywhere a ``PolyData`` is
expected.

.. code-block:: python

    from phenotastic import PhenoMesh
    import pyvista as pv

    # Create from PyVista mesh
    mesh = PhenoMesh(pv.Sphere())

    # PhenoMesh is a PolyData
    isinstance(mesh, pv.PolyData)  # True

    # Process
    mesh = mesh.smooth(iterations=100)
    mesh = mesh.remesh(n_clusters=5000)
    curvature = mesh.compute_curvature(curvature_type="mean")

    # Visualize
    mesh.plot(scalars=curvature, cmap="coolwarm")

    # Convert to plain PyVista PolyData if needed
    polydata = mesh.to_polydata()

Development
-----------

.. code-block:: bash

    # Install development dependencies
    uv sync --group dev

    # Run tests
    uv run pytest

    # Type checking
    uv run mypy src/phenotastic/

    # Linting
    uv run ruff check src/phenotastic/

    # Pre-commit hooks
    uv run pre-commit run --all-files

License
-------

GNU General Public License v3

Author
------

Henrik Ahl (henrikaahl@gmail.com)
