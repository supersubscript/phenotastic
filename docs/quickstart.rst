==========
Quickstart
==========

This guide will help you get started with Phenotastic for 3D plant phenotyping.

Basic Workflow
--------------

The typical Phenotastic workflow consists of:

1. **Load** a 3D image or mesh
2. **Process** using a pipeline
3. **Analyze** the resulting domains

Working with PhenoMesh
----------------------

``PhenoMesh`` extends PyVista's ``PolyData`` class with phenotyping-specific methods:

.. code-block:: python

   from phenotastic import PhenoMesh
   import pyvista as pv

   # Create from a PyVista mesh
   mesh = PhenoMesh(pv.Sphere())

   # PhenoMesh is a PolyData - use anywhere PolyData is expected
   isinstance(mesh, pv.PolyData)  # True

   # Apply mesh operations
   mesh = mesh.smooth(iterations=100)
   mesh = mesh.remesh(n_clusters=5000)

   # Compute curvature
   curvature = mesh.compute_curvature(curvature_type="mean")

   # Visualize
   mesh.plot(scalars=curvature, cmap="coolwarm")

Using Pipelines
---------------

Pipelines provide a reproducible way to process meshes:

.. code-block:: python

   from phenotastic import PhenoMesh, load_preset
   import pyvista as pv

   # Load a mesh
   mesh = PhenoMesh(pv.read("my_mesh.vtk"))

   # Use the default pipeline
   pipeline = load_preset()

   # Run the pipeline
   result = pipeline.run(mesh)

   # Access results
   print(f"Mesh: {result.mesh.n_points} points")
   print(f"Domains: {len(result.domains.unique())} found")
   print(result.domain_data)  # DataFrame with measurements

Custom Pipeline Configuration
-----------------------------

Create custom pipelines using YAML:

.. code-block:: yaml

   # my_pipeline.yaml
   steps:
     - name: smooth
       params:
         iterations: 100
         relaxation_factor: 0.01

     - name: remesh
       params:
         n_clusters: 10000

     - name: compute_curvature
       params:
         curvature_type: mean

     - name: segment_domains

     - name: merge_small
       params:
         threshold: 50

Load and run your custom pipeline:

.. code-block:: python

   from phenotastic import Pipeline

   pipeline = Pipeline.from_yaml("my_pipeline.yaml")
   result = pipeline.run(mesh)

Command Line Interface
----------------------

Phenotastic provides a CLI for common operations:

.. code-block:: bash

   # Run with default pipeline
   phenotastic run image.tif --output results/

   # Run with custom config
   phenotastic run mesh.vtk --config my_pipeline.yaml

   # Generate a config template
   phenotastic init-config my_pipeline.yaml

   # View a mesh
   phenotastic view mesh.vtk --scalars curvature

See :doc:`cli` for the full CLI reference.

Next Steps
----------

- :doc:`cli` - Command line interface reference
- :doc:`pipeline` - Pipeline configuration guide
- :doc:`operations` - Available operations reference
- :doc:`api/index` - Full API documentation
