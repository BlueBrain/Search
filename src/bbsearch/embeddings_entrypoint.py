"""EntryPoint for the computation and saving of the embeddings."""
import argparse
from pathlib import Path
import sqlite3

import numpy as np

import bbsearch.embedding_models as embedding_models

parser = argparse.ArgumentParser()
parser.add_argument("--database",
                    default="/raid/covid_data/data/v7/databases/cord19.db",
                    type=str,
                    help="Database")
parser.add_argument("--saving_directory",
                    default='/raid/covid_data/data/v7/embeddings/',
                    type=str,
                    help="The directory path where the database is saved")
parser.add_argument("--models",
                    default='USE,SBERT,SBioBERT,BSV',
                    type=str,
                    help="Models for which we need to compute the embeddings. "
                         "Format should be comma separated list.")
parser.add_argument("--bsv_checkpoints",
                    default='/raid/covid_data/assets/BioSentVec_PubMed_MIMICIII-bigram_d700.bin',
                    type=str,
                    help="Path to file containing the checkpoints for the BSV model.")
args = parser.parse_args()


def main():
    """Compute Embeddings."""
    print(args)
    if Path(args.database).exists():
        db = sqlite3.connect(args.database).cursor()
    else:
        raise FileNotFoundError(f'The database {args.database} is not found!')

    for model in args.models.split(','):
        if model == 'BSV':
            embedding_model = embedding_models.BSV(checkpoint_model_path=Path(args.bsv_checkpoints))
        else:
            embedding_model = getattr(embedding_models, model)()
        embeddings = embedding_models.compute_database_embeddings(db, embedding_model)
        path = Path(args.saving_directory) / model / f'{model}.npy'
        np.save(path, embeddings)


if __name__ == "__main__":
    main()
