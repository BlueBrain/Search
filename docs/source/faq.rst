FAQ
===

This section describes how to handle common issues.


MySQL encoding issues
---------------------

When interacting in Python with the MySQL database, using SQLAlchemy and the
MySQL driver :code:`mysqldb`, one might ran into the following issue when
retrieving columns with text:

.. code-block:: bash

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
