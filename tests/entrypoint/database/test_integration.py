import pathlib

import sqlalchemy

from bluesearch.entrypoint.database.parent import main


def test_everything_sqlite(tmpdir, jsons_path):
    # Parameters
    temp_path = pathlib.Path(str(tmpdir))
    db_path = temp_path / "db.sqlite"
    parsed_files_dir = temp_path / "parsed"
    parsed_files_dir.mkdir()
    all_input_paths = sorted(jsons_path.rglob("*.json"))
    n_files = len(all_input_paths)

    # Initialization
    args_and_opts_init = [
        "init",
        str(db_path),
        "--db-type=sqlite",
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
            str(db_path),
            str(parsed_file),
            "--db-type=sqlite",
        ]
        main(args_and_opts_add)

    # Asserts
    engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")

    query = "SELECT COUNT(*) FROM articles"
    (n_rows,) = engine.execute(query).fetchone()

    assert n_rows == n_files
