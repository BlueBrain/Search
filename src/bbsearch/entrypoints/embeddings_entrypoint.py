"""EntryPoint for the computation and saving of the embeddings."""
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--database_path",
                    default="/raid/bbs_data/cord19_v7/databases/cord19.db",
                    type=str,
                    help="Database containing at least 4 tables:  "
                         "articles, article_id_to_sha, paragraphs and sentences."
                         "This database is used to read all sentences "
                         "and compute the embeddings. ")
parser.add_argument("--out_dir",
                    default='/raid/bbs_data/cord19_v7/embeddings/',
                    type=str,
                    help="The directory path where the embeddings are saved.")
parser.add_argument("--models",
                    default='USE,SBERT,SBioBERT,BSV',
                    type=str,
                    help="Models for which we need to compute the embeddings. "
                         "Format should be comma separated list.")
parser.add_argument("--bsv_checkpoints",
                    default='/raid/bbs_data/trained_models/BioSentVec_PubMed_MIMICIII-bigram_d700.bin',
                    type=str,
                    help="Path to file containing the checkpoints for the BSV model.")
args = parser.parse_args()


def main():
    """Compute Embeddings."""
    from pathlib import Path
    import sqlalchemy
    import numpy as np
    from .. import embedding_models

    if Path(args.out_dir).exists():
        raise FileNotFoundError(f'The output directory {args.out_dir} does not exist!')

    if Path(args.database).exists():
        engine = sqlalchemy.create_engine(f"sqlite:////{args.database_path}")
    else:
        raise FileNotFoundError(f'The database {args.database} is not found!')

    for model in args.models.split(','):
        model = model.strip()
        if model == 'BSV':
            embedding_model = embedding_models.BSV(
                checkpoint_model_path=Path(args.bsv_checkpoints))
        else:
            try:
                embedding_model = getattr(embedding_models, model)()
            except AttributeError:
                print(f'The model {model} is not supported.')
                continue
        embeddings = embedding_models.compute_database_embeddings(engine, embedding_model)
        path = Path(args.out_dir) / f'{model}.npy'
        np.save(path, embeddings)


if __name__ == "__main__":
    main()
