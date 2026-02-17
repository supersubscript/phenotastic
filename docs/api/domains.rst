=============
domains Module
=============

.. module:: phenotastic.domains

The domains module provides functions for curvature-based segmentation of
meshes into domains and subsequent domain merging and analysis.

Domain Segmentation
-------------------

.. autofunction:: phenotastic.domains.steepest_ascent

Domain Merging
--------------

.. autofunction:: phenotastic.domains.merge_small

.. autofunction:: phenotastic.domains.merge_angles

.. autofunction:: phenotastic.domains.merge_distance

.. autofunction:: phenotastic.domains.merge_engulfing

.. autofunction:: phenotastic.domains.merge_disconnected

.. autofunction:: phenotastic.domains.merge_depth

Domain Analysis
---------------

.. autofunction:: phenotastic.domains.define_meristem

.. autofunction:: phenotastic.domains.extract_domaindata

Scalar Filtering
----------------

.. autofunction:: phenotastic.domains.median

.. autofunction:: phenotastic.domains.mean

.. autofunction:: phenotastic.domains.minmax

.. autofunction:: phenotastic.domains.maxmin
