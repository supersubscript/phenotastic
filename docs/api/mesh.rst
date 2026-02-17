===========
mesh Module
===========

.. module:: phenotastic.mesh

The mesh module provides low-level functions for mesh processing, contour
generation, and mesh creation from 3D images.

Contour Generation
------------------

.. autofunction:: phenotastic.mesh.contour

Mesh Creation
-------------

.. autofunction:: phenotastic.mesh.create_mesh

.. autofunction:: phenotastic.mesh.create_cellular_mesh

Mesh Smoothing
--------------

.. autofunction:: phenotastic.mesh.smooth_boundary

Mesh Remeshing
--------------

.. autofunction:: phenotastic.mesh.remesh

.. autofunction:: phenotastic.mesh.remesh_decimate

Mesh Repair
-----------

.. autofunction:: phenotastic.mesh.repair

.. autofunction:: phenotastic.mesh.repair_small_holes

.. autofunction:: phenotastic.mesh.make_manifold

Mesh Filtering
--------------

.. autofunction:: phenotastic.mesh.filter_by_curvature

.. autofunction:: phenotastic.mesh.remove_by_normals

.. autofunction:: phenotastic.mesh.remove_bridges

.. autofunction:: phenotastic.mesh.remove_tongues

.. autofunction:: phenotastic.mesh.remove_inland_under

Mesh Processing
---------------

.. autofunction:: phenotastic.mesh.process

.. autofunction:: phenotastic.mesh.ecft

.. autofunction:: phenotastic.mesh.erode

.. autofunction:: phenotastic.mesh.drop_skirt

Mesh Analysis
-------------

.. autofunction:: phenotastic.mesh.fit_paraboloid_mesh

.. autofunction:: phenotastic.mesh.correct_normal_orientation

.. autofunction:: phenotastic.mesh.correct_bad_mesh

.. autofunction:: phenotastic.mesh.define_meristem

Vertex Operations
-----------------

.. autofunction:: phenotastic.mesh.get_vertex_neighbors

.. autofunction:: phenotastic.mesh.get_vertex_neighbors_all

.. autofunction:: phenotastic.mesh.get_vertex_cycles

Edge Operations
---------------

.. autofunction:: phenotastic.mesh.get_boundary_edges

.. autofunction:: phenotastic.mesh.get_boundary_points

.. autofunction:: phenotastic.mesh.get_feature_edges

.. autofunction:: phenotastic.mesh.get_manifold_edges

.. autofunction:: phenotastic.mesh.get_non_manifold_edges

Labeling
--------

.. autofunction:: phenotastic.mesh.label_from_image

.. autofunction:: phenotastic.mesh.project_to_surface
