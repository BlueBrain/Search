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
    import pathlib
    import sqlalchemy
    import numpy as np
    from .. import embedding_models

    out_dir = pathlib.Path(args.out_dir)
    database_path = pathlib.Path(args.database_path)
    bsv_checkpoints = pathlib.Path(args.bsv_checkpoints)

    if not out_dir.exists():
        raise FileNotFoundError(f'The output directory {out_dir} does not exist!')
    if not database_path.exists():
        raise FileNotFoundError(f'The database {database_path} is not found!')

    engine = sqlalchemy.create_engine(f"sqlite:////{database_path}")

    for model in args.models.split(','):
        model = model.strip()
        if model == 'BSV':
            embedding_model = embedding_models.BSV(
                checkpoint_model_path=bsv_checkpoints)
        else:
            try:
                embedding_model_cls = getattr(embedding_models, model)
                embedding_model = embedding_model_cls()
            except AttributeError:
                print(f'The model {model} is not supported.')
                continue

        embeddings = embedding_models.compute_database_embeddings(engine, embedding_model)
        path = out_dir / f'{model}.npy'
        np.save(path, embeddings)


if __name__ == "__main__":
    main()
