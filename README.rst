===========
Phenotastic
===========

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

    git clone https://github.com/superWhsubscript/phenotastic.git
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

    # Validate configuration
    phenotastic validate my_pipeline.yaml

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

- ``contour``: Generate binary contour from 3D image
- ``create_mesh``: Create mesh from contour using marching cubes
- ``create_cellular_mesh``: Create mesh from segmented image

Mesh Processing Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``smooth``: Laplacian smoothing
- ``smooth_taubin``: Taubin smoothing (less shrinkage)
- ``smooth_boundary``: Smooth boundary edges only
- ``remesh``: Regularize faces with ACVD
- ``decimate``: Reduce mesh complexity
- ``subdivide``: Increase mesh resolution
- ``repair_holes``: Fill small holes
- ``repair``: Full mesh repair
- ``make_manifold``: Remove non-manifold edges
- ``filter_curvature``: Remove vertices outside curvature range
- ``remove_normals``: Remove vertices based on normal angle
- ``remove_bridges``: Remove bridge triangles
- ``remove_tongues``: Remove tongue-like artifacts
- ``extract_largest``: Keep largest connected component
- ``clean``: Remove degenerate cells
- ``triangulate``: Convert to triangles
- ``compute_normals``: Compute surface normals
- ``flip_normals``: Flip normal direction
- ``rotate``: Rotate around axis
- ``clip``: Clip with plane
- ``erode``: Remove boundary points
- ``ecft``: ExtractLargest, Clean, FillHoles, Triangulate

Domain Operations
~~~~~~~~~~~~~~~~~

- ``compute_curvature``: Compute mesh curvature
- ``filter_scalars``: Apply filter to curvature field
- ``segment_domains``: Steepest ascent segmentation
- ``merge_angles``: Merge domains within angular threshold
- ``merge_distance``: Merge domains within distance threshold
- ``merge_small``: Merge small domains
- ``merge_engulfing``: Merge encircled domains
- ``merge_disconnected``: Connect isolated domains
- ``merge_depth``: Merge domains with similar depth
- ``define_meristem``: Identify meristem domain
- ``extract_domaindata``: Extract domain measurements

PhenoMesh Class
---------------

``PhenoMesh`` is a wrapper around PyVista's ``PolyData`` that provides
convenient methods for mesh processing.

.. code-block:: python

    from phenotastic import PhenoMesh
    import pyvista as pv

    # Create from PyVista mesh
    mesh = PhenoMesh(pv.Sphere())

    # Process
    mesh = mesh.smoothen(iterations=100)
    mesh = mesh.remesh(n=5000)
    curvature = mesh.curvature(curv_type="mean")

    # Visualize
    mesh.plot(scalars=curvature, cmap="coolwarm")

    # Convert back to PyVista
    polydata = mesh.to_polydata()

Development
-----------

.. code-block:: bash

    # Install development dependencies
    uv pip install -e ".[dev]"

    # Run tests
    uv run pytest

    # Type checking
    uv run mypy phenotastic/

    # Linting
    uv run ruff check phenotastic/

    # Pre-commit hooks
    uv run pre-commit run --all-files

License
-------

GNU General Public License v3

Author
------

Henrik Ahl (henrikaahl@gmail.com)
