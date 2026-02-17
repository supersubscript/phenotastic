======================
Command Line Interface
======================

Phenotastic provides a command line interface for running pipelines, managing
configurations, and visualizing meshes.

Global Options
--------------

.. code-block:: bash

   phenotastic --version  # Show version
   phenotastic --help     # Show help

Commands
--------

run
~~~

Run the phenotyping pipeline on an input file.

.. code-block:: bash

   phenotastic run INPUT_FILE [OPTIONS]

**Arguments:**

- ``INPUT_FILE``: 3D image (TIFF) or mesh file (VTK, PLY, STL)

**Options:**

- ``-c, --config PATH``: YAML pipeline configuration file
- ``-p, --preset [default]``: Use a preset pipeline (default: 'default')
- ``-o, --output PATH``: Output directory for results
- ``--save-mesh / --no-save-mesh``: Save output mesh (default: yes)
- ``--save-domains / --no-save-domains``: Save domain data CSV (default: yes)
- ``--verbose / --quiet``: Print progress information (default: verbose)

**Examples:**

.. code-block:: bash

   # Run with default pipeline
   phenotastic run image.tif --output results/

   # Run with custom config
   phenotastic run mesh.vtk --config my_pipeline.yaml -o output/

   # Quiet mode, mesh only
   phenotastic run image.tif --quiet --no-save-domains -o out/

init-config
~~~~~~~~~~~

Generate a pipeline configuration file from the default preset.

.. code-block:: bash

   phenotastic init-config OUTPUT_FILE

**Arguments:**

- ``OUTPUT_FILE``: Path for the generated YAML file

**Example:**

.. code-block:: bash

   phenotastic init-config my_pipeline.yaml

list-operations
~~~~~~~~~~~~~~~

List all available pipeline operations.

.. code-block:: bash

   phenotastic list-operations [OPTIONS]

**Options:**

- ``-c, --category [all|contour|mesh|domain]``: Filter by category (default: all)

**Examples:**

.. code-block:: bash

   # List all operations
   phenotastic list-operations

   # List only mesh operations
   phenotastic list-operations --category mesh

list-presets
~~~~~~~~~~~~

List available preset pipeline configurations.

.. code-block:: bash

   phenotastic list-presets

validate
~~~~~~~~

Validate a pipeline configuration file.

.. code-block:: bash

   phenotastic validate CONFIG_FILE

**Arguments:**

- ``CONFIG_FILE``: Path to the YAML configuration file

**Example:**

.. code-block:: bash

   phenotastic validate my_pipeline.yaml

view
~~~~

Visualize a mesh file interactively.

.. code-block:: bash

   phenotastic view MESH_FILE [OPTIONS]

**Arguments:**

- ``MESH_FILE``: Path to the mesh file (VTK, PLY, STL, OBJ)

**Options:**

- ``-s, --scalars NAME``: Scalar array to color by (e.g., 'curvature', 'domains')
- ``--cmap NAME``: Colormap for visualization (default: 'viridis')

**Examples:**

.. code-block:: bash

   # Basic view
   phenotastic view mesh.vtk

   # View with curvature coloring
   phenotastic view mesh.vtk --scalars curvature

   # View domains with custom colormap
   phenotastic view mesh.vtk --scalars domains --cmap tab20
