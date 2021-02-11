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

FAQ
===

This section describes how to handle common issues.


MySQL encoding issue
---------------------

When interacting in Python with the MySQL database, using SQLAlchemy and the
MySQL driver :code:`mysqldb`, one might run into the following error when
retrieving columns with text:

.. code-block:: text

    UnicodeDecodeError: 'charmap' codec can't decode byte 0x81 in position 239:
    character maps to <undefined>

The solution is to append :code:`?charset=utf8mb4` to the database URL.

So, if the database URL was:

.. code-block:: python

    f"mysql+mysqldb://{username}:{password}@{host}:{port}/{database}"

then the new URL would be:

.. code-block:: python

    f"mysql+mysqldb://{username}:{password}@{host}:{port}/{database}?charset=utf8mb4"

The database URL is what is passed as a first argument to create the engine:

.. code-block:: python

    import sqlalchemy

    engine = sqlalchemy.create_engine(f"{dialect}+{driver}://{username}:{password}@{host}:{port}/{database}")


DVC dataclasses issue
----------------------

When in a Python 3.7+ environment the package :code:`dataclasses` is installed,
one might run into the following error when doing :code:`dvc pull`:

.. code-block:: bash

    AttributeError: module 'typing' has no attribute '_ClassVar'

The solution is to uninstall the package :code:`dataclasses`:

.. code-block:: bash

    pip uninstall dataclasses


DVC pull issue
--------------

When launching mining_cache or mining_server entrypoints or even simply
:code:`dvc pull`, one might run into the following error:

.. code-block:: text

    WARNING: Some of the cache files do not exist neither locally nor on remote.
    Missing cache files:

In this case, the solution is to go to the :code:`.dvc` directory
and remove the file called `config.local`:

.. code-block:: bash

    cd .dvc
    rm config.local

Doing `dvc pull` again should work fine after this.
