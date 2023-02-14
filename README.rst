===========
Phenotastic
===========

.. image:: https://img.shields.io/pypi/v/phenotastic.svg
        :target: https://pypi.python.org/pypi/phenotastic

.. image:: https://img.shields.io/travis/supersubscript/phenotastic.svg
        :target: https://travis-ci.org/supersubscript/phenotastic

.. image:: https://readthedocs.org/projects/phenotastic/badge/?version=latest
        :target: https://phenotastic.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Phenotastic is a Python package developed to facilitate the segmentation of early flower organs, primordia, from shoot apical meristems in 3D images. The package provides methods for contouring 3D data, generating a 2.5D mesh, and segmenting out features based on the mesh curvature. It is designed to work with a range of 3D imaging modalities, including confocal microscopy.

============
Installation
============

You can install Phenotastic using pip:

´´´bash
pip install phenotastic
´´´

Phenotastic requires the following dependencies:

- NumPy
- SciPy
- scikit-image
- scikit-learn
- PyVista

=====
Usage
=====

For detailed examples and explanations of the code, please refer to the documentation.

=============
Documentation
=============

The documentation for Phenotastic can be found on Read the Docs. The documentation provides detailed explanations of each method in the package, as well as example usage and workflows.

=============
Contributions
=============

Contributions to Phenotastic are welcome! If you find a bug or have a feature request, please open an issue on the GitHub repository. If you would like to contribute code, please open a pull request with a clear explanation of the changes you've made and their purpose.

===================
License and credits
===================

* Free software: GNU General Public License v3
* Documentation: https://phenotastic.readthedocs.io.

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
