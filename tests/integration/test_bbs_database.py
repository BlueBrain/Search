import pathlib
import pytest
import time

import docker
import sqlalchemy
from sqlalchemy.exc import OperationalError

from bluesearch.entrypoint.database.parent import main


@pytest.fixture(params=["sqlite", "mysql"])
def setup_backend(request, tmpdir):
    backend = request.param
    if backend == "sqlite":
        db_url = pathlib.Path(str(tmpdir)) / "db.sqlite"
        yield "sqlite", str(db_url)

    elif backend == "mysql":
        try:
            client = docker.from_env()
            client.ping()

        except docker.errors.DockerException:
            pytest.skip()

        port = 22346
        container = client.containers.run(
            image="mysql:latest",
            environment={"MYSQL_ROOT_PASSWORD": "my-secret-pw"},
            ports={"3306/tcp": port},
            detach=True,
            auto_remove=True,
        )

        max_waiting_time = 2 * 60
        start = time.perf_counter()

        while time.perf_counter() - start < max_waiting_time:
            try:
                engine = sqlalchemy.create_engine(
                    f"mysql+pymysql://root:my-secret-pw@127.0.0.1:{port}/"
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

        yield "mysql", f"root:my-secret-pw@127.0.0.1:{port}/test",

        container.kill()
        client.close()

    else:
        raise ValueError


def test_bbs_database(tmpdir, setup_backend, jsons_path):
    # Parameters
    db_type, db_url = setup_backend

    temp_path = pathlib.Path(str(tmpdir))
    parsed_files_dir = temp_path / "parsed"
    parsed_files_dir.mkdir()

    all_input_paths = sorted(jsons_path.rglob("*.json"))
    n_files = len(all_input_paths)

    # Initialization
    args_and_opts_init = [
        "init",
        str(db_url),
        f"--db-type={db_type}",
    ]
    main(args_and_opts_init)

    # Parsing all available articles
    for input_path in all_input_paths:
        args_and_opts_parse = [
            "parse",
            "CORD19ArticleParser",
            str(input_path),
            str(parsed_files_dir / f"{input_path.stem}.pkl"),
        ]
        main(args_and_opts_parse)

    # Adding parsed files to the database
    for parsed_file in parsed_files_dir.iterdir():
        args_and_opts_add = [
            "add",
            str(db_url),
            str(parsed_file),
            f"--db-type={db_type}",
        ]
        main(args_and_opts_add)

    # Asserts
    if db_type == "sqlite":
        engine = sqlalchemy.create_engine(f"sqlite:///{db_url}")

    elif db_type == "mysql":
        engine = sqlalchemy.create_engine(f"mysql+pymysql://{db_url}")

    query = "SELECT COUNT(*) FROM articles"
    (n_rows,) = engine.execute(query).fetchone()

    assert n_rows == n_files > 0
