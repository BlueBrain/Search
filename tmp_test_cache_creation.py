import pandas as pd
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

ee_models_df = pd.read_csv('~/dev/BlueBrainSearch/ee_models_library.csv')
mining_cache_creation.construct(ee_models_library=ee_models_df,
                                n_processes=1,
                                always_mine=True)
