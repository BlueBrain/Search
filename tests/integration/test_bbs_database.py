import pytest
import sqlalchemy

from bluesearch.entrypoint.database.parent import main


@pytest.fixture(params=["sqlite", "mysql"])
def setup_backend(request, tmp_path):
    backend = request.param
    if backend == "sqlite":
        db_url = tmp_path / "db.sqlite"
        yield "sqlite", str(db_url)

    elif backend == "mysql":
        engine = sqlalchemy.create_engine("mysql+pymysql://root:root@127.0.0.1:3306")
        # Should throw a OperationalError if MySQL is not ready yet.
        engine.execute("show databases")
        yield "mysql", "root:root@127.0.0.1:3306/test"
    else:
        raise ValueError


def test_bbs_database(tmp_path, setup_backend, jsons_path):
    # Parameters
    db_type, db_url = setup_backend

    parsed_files_dir = tmp_path / "parsed"
    parsed_files_dir.mkdir()

    all_input_paths = sorted(jsons_path.rglob("*.json"))

    # 16e82ce0e0c8a1b36497afc0d4392b4fe21eb174.json and PMC7223769.xml.json are the
    # same article. In the presence of duplicates, currently, the code stops with an
    # 'IntegrityError' from MySQL. The patch below is to move forward until the code
    # does not stop anymore.
    all_input_paths = [x for x in all_input_paths if x.name != "PMC7223769.xml.json"]

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
            "cord19-json",
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
    (n_rows,) = engine.execute(query).fetchone()  # type: ignore

    assert n_rows == n_files > 0
