import argparse
import logging
from pathlib import Path

from bbsearch.embedding_models import EmbeddingModels

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--assets_path",
                    default="/raid/covid_data/assets/",
                    type=str,
                    help="The path to the assets needed to find the pretrained models")
parser.add_argument("--models_to_load",
                    default='USE,BSV,SBIOBERT,SBERT',
                    type=str,
                    help="List of all the models for which we need to compute the embeddings"
                         "Current possible models are: USE, BSV, SBIOBERT, SBERT")
parser.add_argument("--database_path",
                    default='/raid/covid_data/data/v7/cord19_v1.db',
                    type=str,
                    help="The version of the database")
parser.add_argument("--saving_directory",
                    default='/raid/covid_data/data/v7/embeddings/',
                    type=str,
                    help="The directory where the embeddings are going to be saved")
args = parser.parse_args()


def main():
    models_to_load = args.models_to_load.split(',')
    embeddings_models = EmbeddingModels(assets_path=Path(args.assets_path),
                                        models_to_load=models_to_load)
    embeddings_models.save_sentence_embeddings(database_path=Path(args.database_path),
                                               saving_directory='')


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    main()
