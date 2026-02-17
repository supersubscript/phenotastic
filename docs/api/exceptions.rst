==========
Exceptions
==========

.. module:: phenotastic.exceptions

Phenotastic defines a hierarchy of exceptions for error handling.

Exception Hierarchy
-------------------

.. code-block:: text

   PhenotasticError (base)
   ├── PipelineError
   ├── ConfigurationError
   ├── InvalidImageError
   └── InvalidMeshError

Base Exception
--------------

.. autoexception:: phenotastic.PhenotasticError
   :members:
   :show-inheritance:

Pipeline Exceptions
-------------------

.. autoexception:: phenotastic.PipelineError
   :members:
   :show-inheritance:

.. autoexception:: phenotastic.ConfigurationError
   :members:
   :show-inheritance:

Data Validation Exceptions
--------------------------

.. autoexception:: phenotastic.InvalidImageError
   :members:
   :show-inheritance:

.. autoexception:: phenotastic.InvalidMeshError
   :members:
   :show-inheritance:

Usage Example
-------------

.. code-block:: python

   from phenotastic import Pipeline, ConfigurationError

   try:
       pipeline = Pipeline.from_yaml("config.yaml")
       result = pipeline.run(mesh)
   except ConfigurationError as e:
       print(f"Invalid configuration: {e}")
   except PipelineError as e:
       print(f"Pipeline failed: {e}")
