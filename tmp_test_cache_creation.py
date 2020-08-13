import sqlalchemy

from bbsearch.database import MiningCacheCreation


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

mining_cache_creation = MiningCacheCreation(engine=engine)

mining_cache_creation.construct()

