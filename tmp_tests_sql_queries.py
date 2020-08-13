import sqlalchemy
import pandas as pd
import getpass
from io import StringIO

from bbsearch.utils import Timer


def get_sql_url():
    protocol = "mysql"
    host = "dgx1.bbp.epfl.ch"
    port = 8853
    user = "stan"
    pw = "letmein"
    db = "cord19_v35"

    return f"{protocol}://{user}:{pw}@{host}:{port}/{db}"


timer = Timer(verbose=True)
engine = sqlalchemy.create_engine(get_sql_url())

if engine.dialect.has_table(engine, "mining_cache"):
    print("--- TABLE mining_cache is already present!")
    engine.execute('DROP TABLE mining_cache')

metadata = sqlalchemy.MetaData()
articles_table = sqlalchemy.Table(
    "articles", metadata, autoload=True, autoload_with=engine)
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
    sqlalchemy.Column("paper_id", sqlalchemy.Text()),
    sqlalchemy.Column("start_char", sqlalchemy.Integer()),
    sqlalchemy.Column("end_char", sqlalchemy.Integer()),
    sqlalchemy.Column(
        "article_id",
        sqlalchemy.Integer(),
        sqlalchemy.ForeignKey(articles_table.columns.article_id),
    ),
    sqlalchemy.Column(
        "paragraph_pos_in_article", sqlalchemy.Integer(), nullable=False
    ),
    sqlalchemy.Column("mining_model", sqlalchemy.Text(), nullable=False),
)

# cols = ["entity",
#         "entity_type",
#         "property",
#         "property_value",
#         "property_type",
#         "property_value_type",
#         "ontology_source",  # TODO: fix issue inserting None!
#         "paper_id",
#         "start_char",
#         "end_char",
#         "article_id",
#         "paragraph_pos_in_article",
#         "mining_model"]


with engine.begin() as connection:
    metadata.create_all(connection)

df = pd.read_csv('assets/extractions_200.csv')
# df = df.reset_index(drop=True)
print(df.columns)
# import pdb; pdb.set_trace()
df['paragraph_pos_in_article'] = 9
df['article_id'] = 123
df['mining_model'] = 'some_model'

df = df[:10_000]
df = df.where(pd.notnull(df), None)


print(f'START INSERTING {len(df)} entries')

with timer("df.to_sql"):
    df.to_sql(name='mining_cache', con=engine, if_exists='append', index=False)


with timer("engine.execute"):
    engine.execute(mining_cache_table.insert(), df.to_dict("records"))

# df_out = pd.read_sql('SELECT * FROM mining_cache', engine)
# print(df_out)
