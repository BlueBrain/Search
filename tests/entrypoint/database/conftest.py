import pathlib

import docker
import pytest
import sqlalchemy
from sqlalchemy.orm import sessionmaker

def fill_db(engine):
    metadata = sqlalchemy.MetaData()
    sqlalchemy.Table(
        "articles",
        metadata,
        sqlalchemy.Column(
            "article_id", sqlalchemy.Integer(), primary_key=True, autoincrement=True
        ),
        sqlalchemy.Column("title", sqlalchemy.Text()),
    )
    with engine.begin() as connection:
        metadata.create_all(connection)



@pytest.fixture(scope="session", params=["sqlite", "mysql"])
def bbs_database_backend(request):
    """Different backends for the database"""
    backend = request.param
    if backend == "mysql":
        # check docker daemon running
        try:
            client = docker.from_env()
            client.ping()

        except docker.errors.DockerException:
            pytest.skip()

    return backend


@pytest.fixture(scope="session")
def bbs_database_engine(tmp_path_factory, bbs_database_backend):
    if bbs_database_backend == "sqlite":
        db_path = tmp_path_factory.mktemp("db", numbered=False) / "bbs_database_test.db"
        pathlib.Path(db_path).touch()
        engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
        fill_db(engine)
        yield engine

    else:
        port_number = 22346
        client = docker.from_env()
        container = client.containers.run(
            image="mysql:latest",
            environment={"MYSQL_ROOT_PASSWORD": "my-secret-pw"},
            ports={"3306/tcp": port_number},
            detach=True,
            auto_remove=True,
        )

        max_waiting_time = 2 * 60
        start = time.perf_counter()

        while time.perf_counter() - start < max_waiting_time:
            try:
                engine = sqlalchemy.create_engine(
                    f"mysql+pymysql://root:my-secret-pw@127.0.0.1:{port_number}/"
                )
                # Container ready?
                engine.execute("show databases")
                break
            except OperationalError:
                # Container not ready, pause and then try again
                time.sleep(0.1)
                continue
        else:
            raise TimeoutError("Could not spawn the MySQL container.")

        engine.execute("create database test")
        engine.dispose()
        engine = sqlalchemy.create_engine(
            f"mysql+pymysql://root:my-secret-pw" f"@127.0.0.1:{port_number}/test"
        )
        fill_db(engine, metadata_path, test_parameters, entity_types)

        yield engine

        container.kill()
        client.close()


@pytest.fixture(scope='function')
def bbs_database_session(bbs_database_engine):
    """One does not commit to the database by using this fixture."""
    session = sessionmaker(bind=bbs_database_engine)()

    yield session

    session.rollback()
    session.close()
