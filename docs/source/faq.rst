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
