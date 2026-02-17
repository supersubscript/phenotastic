============
Installation
============

Requirements
------------

Phenotastic requires Python 3.12 or later.

Install from PyPI
-----------------

The easiest way to install Phenotastic is via pip:

.. code-block:: bash

   pip install phenotastic

Or using uv (recommended):

.. code-block:: bash

   uv pip install phenotastic

Install from Source
-------------------

To install from source for development:

.. code-block:: bash

   git clone https://github.com/supersubscript/phenotastic.git
   cd phenotastic
   uv sync --group dev

Optional Dependencies
---------------------

Documentation
~~~~~~~~~~~~~

To build the documentation locally:

.. code-block:: bash

   uv sync --extra docs
   make docs

The documentation will be available at ``docs/_build/html/index.html``.

Verifying Installation
----------------------

After installation, verify that Phenotastic is working:

.. code-block:: python

   import phenotastic
   print(phenotastic.__version__)

Or via the command line:

.. code-block:: bash

   phenotastic --version
