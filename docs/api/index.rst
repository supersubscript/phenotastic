=============
API Reference
=============

This section provides detailed API documentation for all public classes,
functions, and modules in Phenotastic.

.. toctree::
   :maxdepth: 2

   phenomesh
   pipeline
   mesh
   domains
   exceptions

Public API Overview
-------------------

Classes
~~~~~~~

- :class:`~phenotastic.PhenoMesh` - Extension of PyVista PolyData for phenotyping
- :class:`~phenotastic.Pipeline` - Pipeline executor
- :class:`~phenotastic.PipelineContext` - State container for pipeline execution
- :class:`~phenotastic.StepConfig` - Configuration for a pipeline step
- :class:`~phenotastic.OperationRegistry` - Registry of available operations

Functions
~~~~~~~~~

- :func:`~phenotastic.load_preset` - Load a preset pipeline
- :func:`~phenotastic.list_presets` - List available presets
- :func:`~phenotastic.get_preset_yaml` - Get preset configuration as YAML
- :func:`~phenotastic.save_pipeline_yaml` - Save pipeline to YAML file

Exceptions
~~~~~~~~~~

- :exc:`~phenotastic.PhenotasticError` - Base exception
- :exc:`~phenotastic.PipelineError` - Pipeline execution errors
- :exc:`~phenotastic.ConfigurationError` - Configuration errors
- :exc:`~phenotastic.InvalidImageError` - Invalid image data
- :exc:`~phenotastic.InvalidMeshError` - Invalid mesh data
