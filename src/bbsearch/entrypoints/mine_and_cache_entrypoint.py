"""EntryPoint for the mining a database and saving of extracted items in a cache DB."""
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--db_type",
    default="mysql",
    type=str,
    help="Type of the database. Possible values: (sqlite, " "mysql)",
)
parser.add_argument(
    "--ee_models_library_file",
    default='/raid/sync/proj115/bbs_data/models_libraries/ee_models_library.csv"',
    type=str,
    help="The csv file with info on which model to use to mine which entity type.",
)
parser.add_argument(
    "--n_processes",
    default="4",
    type=int,
    help="Max n of processes to run the different requested mining models in parallel.",
)
args = parser.parse_args()


def main():
    """Mine all texts in database and save results in a cache."""
    import getpass
    import pathlib

    import sqlalchemy

    if args.db_type == "sqlite":
        database_path = "/raid/sync/proj115/bbs_data/cord19_v35/databases/cord19.db"
        if not pathlib.Path(database_path).exists():
            pathlib.Path(database_path).touch()
        engine = sqlalchemy.create_engine(f"sqlite:///{database_path}")
    elif args.db_type == "mysql":
        password = getpass.getpass("Password:")
        engine = sqlalchemy.create_engine(
            f"mysql+pymysql://guest:{password}" f"@dgx1.bbp.epfl.ch:8853/cord19_v35"
        )
    else:
        raise ValueError("This is not an handled db_type.")

    # create CORD19CacheCreation inside database.py or do everything here?
    metadata = sqlalchemy.MetaData()
    mining_cache_table = sqlalchemy.Table(
        "mining_cache",
        metadata,
        sqlalchemy.Column("entity", sqlalchemy.Text()),
        sqlalchemy.Column("entity_type", sqlalchemy.Text()),
        sqlalchemy.Column("property", sqlalchemy.Text()),
        sqlalchemy.Column("property_value", sqlalchemy.Text()),
        sqlalchemy.Column("property_type", sqlalchemy.Text()),
        sqlalchemy.Column("property_value_type", sqlalchemy.Text()),
        sqlalchemy.Column("ontology_source", sqlalchemy.Text()),
        sqlalchemy.Column("start_char", sqlalchemy.Integer()),
        sqlalchemy.Column("end_char", sqlalchemy.Integer()),
        sqlalchemy.Column(
            "article_id",
            sqlalchemy.Integer(),
            sqlalchemy.ForeignKey("articles.article_id"),
            nullable=False,
        ),
        sqlalchemy.Column(
            "paragraph_pos_in_article", sqlalchemy.Integer(), nullable=False
        ),
        sqlalchemy.Column(
            "pargraph_sha", sqlalchemy.Text(), nullable=False
        ),  # TODO: maybe Int?
        sqlalchemy.Column("mining_model", sqlalchemy.Text(), nullable=False),
    )

    mined_items_list_table = sqlalchemy.Table(
        "mined_items_list",
        metadata,
        sqlalchemy.Column(
            "article_id",
            sqlalchemy.Integer(),
            sqlalchemy.ForeignKey("articles.article_id"),
            nullable=False,
        ),
        sqlalchemy.Column(
            "paragraph_pos_in_article", sqlalchemy.Integer(), nullable=False
        ),
        sqlalchemy.Column(
            "pargraph_sha", sqlalchemy.Text(), nullable=False
        ),  # TODO: maybe Int?
        sqlalchemy.Column("mining_model", sqlalchemy.Text(), nullable=False),
    )

    # TODO: Are tables mining_cache and mined_items_list already there?
    # TODO: -- if No, then create tables!

    # TODO: all_texts = generator to go through (article_id, par_pos_in_article, text)
    # TODO: all_texts = (a, p, t, sha(p) for a, p, t in all_texts) # note: here is still a generator!
    # TODO: For model in the library ... [here potentially process pool]
    # TODO:     Are results of model in the table mined_items_list already?
    # TODO:         Yes: continue
    # TODO:     results_of_mining = model.pipe(all_texts, metadata) # note: here is still a generator!
    # TODO:
