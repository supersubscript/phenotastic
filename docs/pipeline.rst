======================
Pipeline Configuration
======================

Phenotastic uses a recipe-style YAML configuration for defining reproducible
processing pipelines.

Pipeline Structure
------------------

A pipeline is a sequence of steps, each specifying an operation and optional parameters:

.. code-block:: yaml

   steps:
     - name: operation_name
       params:
         param1: value1
         param2: value2

     - name: another_operation
       # params are optional for operations with defaults

Loading Pipelines
-----------------

From YAML File
~~~~~~~~~~~~~~

.. code-block:: python

   from phenotastic import Pipeline

   pipeline = Pipeline.from_yaml("my_pipeline.yaml")

From Preset
~~~~~~~~~~~

.. code-block:: python

   from phenotastic import load_preset

   # Load default preset
   pipeline = load_preset()

   # Or explicitly
   pipeline = load_preset("default")

Programmatically
~~~~~~~~~~~~~~~~

.. code-block:: python

   from phenotastic import Pipeline, StepConfig

   steps = [
       StepConfig(name="smooth", parameters={"iterations": 100}),
       StepConfig(name="remesh", parameters={"n_clusters": 5000}),
       StepConfig(name="compute_curvature"),
   ]
   pipeline = Pipeline(steps)

Running Pipelines
-----------------

.. code-block:: python

   from phenotastic import PhenoMesh, load_preset
   import pyvista as pv

   # Load input
   mesh = PhenoMesh(pv.read("mesh.vtk"))

   # Run pipeline
   pipeline = load_preset()
   result = pipeline.run(mesh, verbose=True)

   # Access results
   result.mesh          # Processed PhenoMesh
   result.domains       # Domain labels array
   result.curvature     # Curvature array
   result.domain_data   # DataFrame with measurements

Pipeline Context
----------------

The ``PipelineContext`` holds state as operations are applied:

.. code-block:: python

   from phenotastic import PipelineContext

   # Context attributes
   context.image        # Input 3D image (optional)
   context.contour      # Binary contour array
   context.mesh         # Current PhenoMesh
   context.resolution   # Spatial resolution [z, y, x]
   context.curvature    # Curvature values
   context.domains      # Domain labels
   context.neighbors    # Vertex neighbor indices
   context.meristem_index  # Meristem domain index
   context.domain_data  # Domain measurements DataFrame

Example Configuration
---------------------

Full Pipeline
~~~~~~~~~~~~~

A complete pipeline from 3D image to domain analysis:

.. code-block:: yaml

   steps:
     # Generate contour from image
     - name: contour
       params:
         iterations: 25
         smoothing: 1

     # Create mesh
     - name: create_mesh
       params:
         step_size: 1

     # Mesh processing
     - name: smooth
       params:
         iterations: 100
         relaxation_factor: 0.01

     - name: remesh
       params:
         n_clusters: 10000

     - name: smooth
       params:
         iterations: 50

     - name: repair_holes
       params:
         max_hole_edges: 100

     # Domain segmentation
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
         threshold: 50

     - name: merge_engulfing
       params:
         threshold: 0.9

     - name: define_meristem

     - name: extract_domaindata

Mesh-Only Pipeline
~~~~~~~~~~~~~~~~~~

For processing an existing mesh:

.. code-block:: yaml

   steps:
     - name: smooth
       params:
         iterations: 200

     - name: remesh
       params:
         n_clusters: 8000

     - name: compute_curvature
       params:
         curvature_type: gaussian

     - name: segment_domains

     - name: merge_small
       params:
         threshold: 100

Saving Pipelines
----------------

Export a pipeline to YAML:

.. code-block:: python

   from phenotastic import save_pipeline_yaml

   save_pipeline_yaml(pipeline, "exported_pipeline.yaml")

Or get the YAML string:

.. code-block:: python

   from phenotastic import get_preset_yaml

   yaml_content = get_preset_yaml()
   print(yaml_content)

Validating Configurations
-------------------------

Check a configuration file before running:

.. code-block:: bash

   phenotastic validate my_pipeline.yaml

Or programmatically:

.. code-block:: python

   pipeline = Pipeline.from_yaml("my_pipeline.yaml")
   warnings = pipeline.validate()

   for warning in warnings:
       print(f"Warning: {warning}")

See :doc:`operations` for all available operations and their parameters.
