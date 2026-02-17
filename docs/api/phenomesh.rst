=========
PhenoMesh
=========

.. module:: phenotastic.phenomesh

The ``PhenoMesh`` class extends PyVista's ``PolyData`` with phenotyping-specific
operations for 3D plant analysis.

PhenoMesh Class
---------------

.. autoclass:: phenotastic.PhenoMesh
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Usage Examples
--------------

Creating a PhenoMesh
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from phenotastic import PhenoMesh
   import pyvista as pv

   # From a PyVista mesh
   mesh = PhenoMesh(pv.Sphere())

   # From a file
   mesh = PhenoMesh(pv.read("mesh.vtk"))

   # With custom attributes
   mesh = PhenoMesh(
       pv.Sphere(),
       contour=binary_contour_array,
       resolution=[1.0, 1.0, 1.0]
   )

Mesh Processing
~~~~~~~~~~~~~~~

.. code-block:: python

   # Smoothing
   mesh = mesh.smooth(iterations=100)
   mesh = mesh.smooth_taubin(iterations=50, pass_band=0.1)

   # Remeshing
   mesh = mesh.remesh(n_clusters=10000)

   # Repair
   mesh = mesh.repair()
   mesh = mesh.repair_small(nbe=100)

Curvature Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compute curvature
   curvature = mesh.compute_curvature(curvature_type="mean")

   # Filter by curvature
   filtered = mesh.filter_curvature(curvature_threshold=(-0.5, 0.5))

Interoperability
~~~~~~~~~~~~~~~~

.. code-block:: python

   # PhenoMesh is a PolyData
   isinstance(mesh, pv.PolyData)  # True

   # Use anywhere PolyData is expected
   plotter = pv.Plotter()
   plotter.add_mesh(mesh)

   # Convert to plain PolyData
   polydata = mesh.to_polydata()

   # Create from PolyData
   mesh = PhenoMesh.from_polydata(polydata)
