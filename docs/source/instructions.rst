.. Blue Brain Search is a text mining toolbox focused on scientific use cases.
   Copyright (C) 2020  Blue Brain Project, EPFL.
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public License
   along with this program. If not, see <https://www.gnu.org/licenses/>.

Instructions
============

Installation
------------
Before installation, please make sure you have a recent :code:`pip` installed (:code:`>=19.1`)

Then you can easily install :code:`bluesearch` from PyPI:

.. code-block:: bash

   pip install bluesearch

You can also build from source if you prefer:

.. code-block:: bash

    pip install .  # use -e for editable install


Generating docs
---------------
All the versions of our documentation, both stable and latest,
`can be found on Read the Docs <https://blue-brain-search.readthedocs.io/en/stable/>`_.


To generate the documentation manually, we use :code:`sphinx` with a custom BBP theme.
Make sure to install the :code:`bluesearch` package with :code:`dev` extras to get
the necessary dependencies.

.. code-block:: bash

    pip install -e .[dev]

To generate autodoc directives one can run

.. code-block:: bash

    cd docs
    sphinx-apidoc -o source/api/ -f -e ../src/bluesearch/ ../src/bluesearch/entrypoint/*

Note that it only needs to be rerun when there are new subpackages/modules.

To generate the documentation run

.. code-block:: bash

    cd docs
    make clean && make html


Finally, one can also run doctests

.. code-block:: bash

    cd docs
    make doctest
