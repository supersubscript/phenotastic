===========
Phenotastic
===========

3D plant phenotyping package for segmentation of early flower organs (primordia)
from shoot apical meristems in 3D images.

.. image:: https://img.shields.io/pypi/v/phenotastic.svg
   :target: https://pypi.org/project/phenotastic/
   :alt: PyPI version

.. image:: https://img.shields.io/github/actions/workflow/status/supersubscript/phenotastic/dev.yml?branch=main
   :target: https://github.com/supersubscript/phenotastic/actions
   :alt: Build Status

Features
--------

- **3D Image Contouring**: Morphological active contour methods for extracting surfaces from 3D image stacks
- **Mesh Processing**: Smoothing, remeshing, and repair operations for 3D meshes
- **Domain Segmentation**: Curvature-based segmentation of meshes into regions (domains)
- **Pipeline System**: Configurable recipe-style pipelines for reproducible workflows

Quick Example
-------------

.. code-block:: python

   from phenotastic import PhenoMesh, load_preset
   import pyvista as pv

   # Load a mesh
   mesh = PhenoMesh(pv.read("my_mesh.vtk"))

   # Process with the default pipeline
   pipeline = load_preset()
   result = pipeline.run(mesh)

   # Access results
   print(f"Found {len(result.domains.unique())} domains")

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   cli
   pipeline
   operations

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Citation
--------

If you use Phenotastic in your research, please cite:

    Åhl, H., Zhang, Y., & Jönsson, H. (2022). High-throughput 3D phenotyping of plant shoot
    apical meristems from tissue-resolution data. *Frontiers in Plant Science*, 13, 827147.

BibTeX:

.. code-block:: bibtex

    @article{aahl2022high,
      title={High-throughput 3d phenotyping of plant shoot apical meristems from tissue-resolution data},
      author={{\AA}hl, Henrik and Zhang, Yi and J{\"o}nsson, Henrik},
      journal={Frontiers in Plant Science},
      volume={13},
      pages={827147},
      year={2022},
      publisher={Frontiers Media SA}
    }

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
