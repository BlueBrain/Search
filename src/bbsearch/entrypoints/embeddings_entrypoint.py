"""EntryPoint for the computation and saving of the embeddings."""
import argparse
import getpass
import logging
import pathlib

from ._helper import configure_logging

parser = argparse.ArgumentParser()
parser.add_argument("--db_type",
                    default="mysql",
                    type=str,
                    help="Type of the database. Possible values: (sqlite, "
                         "mysql)")
parser.add_argument("--out_dir",
                    default='/raid/sync/proj115/bbs_data/cord19_v47/embeddings/',
                    type=str,
                    help="The directory path where the embeddings are saved.")
parser.add_argument("--log_dir",
                    default="/raid/projects/bbs/logs/",
                    type=str,
                    help="The directory path where to save the logs.")
parser.add_argument("--log_name",
                    default="database_creation.log",
                    type=str,
                    help="The name of the log file.")
parser.add_argument("--models",
                    default='USE,SBERT,SBioBERT,BSV',
                    type=str,
                    help="Models for which we need to compute the embeddings. "
                         "Format should be comma separated list.")
parser.add_argument("--bsv_checkpoints",
                    default='/raid/sync/proj115/bbs_data/trained_models/BioSentVec_PubMed_MIMICIII-bigram_d700.bin',
                    type=str,
                    help="Path to file containing the checkpoints for the BSV model.")
parser.add_argument("--step",
                    default='1000',
                    type=int,
                    help="Batch size for the embeddings computation")
args = parser.parse_args()


def main():
    """Compute Embeddings."""
    # Configure logging
    log_file = pathlib.Path(args.log_dir) / args.log_name
    configure_logging(log_file, logging.INFO)
    logger = logging.getLogger(__name__)

    import pandas as pd
    import sqlalchemy

    from .. import embedding_models
    from ..utils import H5

    out_dir = pathlib.Path(args.out_dir)
    bsv_checkpoints = pathlib.Path(args.bsv_checkpoints)

    if not out_dir.exists():
        raise FileNotFoundError(f'The output directory {out_dir} does not exist!')
    if not bsv_checkpoints.exists():
        raise FileNotFoundError(f'The BSV checkpoints {bsv_checkpoints} does '
                                f'not exist!')

    embeddings_path = out_dir / 'embeddings.h5'

    logger.info('SQL Alchemy Engine creation ....')

    if args.db_type == 'sqlite':
        database_path = '/raid/sync/proj115/bbs_data/cord19_v47/databases/cord19.db'
        if not pathlib.Path(database_path).exists():
            pathlib.Path(database_path).touch()
        engine = sqlalchemy.create_engine(f'sqlite:///{database_path}')
    elif args.db_type == 'mysql':
        password = getpass.getpass('Password:')
        engine = sqlalchemy.create_engine(f'mysql+pymysql://guest:{password}'
                                          f'@dgx1.bbp.epfl.ch:8853/cord19_v47')
    else:
        raise ValueError('This is not an handled db_type.')

    logger.info('Sentences IDs retrieving....')

    sql_query = """SELECT sentence_id
                   FROM sentences"""

    sentence_ids = pd.read_sql(sql_query, engine)['sentence_id'].to_list()

    logger.info('Counting Number Total of sentences....')

    sql_query = """SELECT COUNT(*)
                   FROM sentences"""

    n_sentences = pd.read_sql(sql_query, engine).iloc[0, 0]

    logger.info(f'{len(sentence_ids)} to embed / '
          f'Total Number of sentences {n_sentences}')

    for model in args.models.split(','):
        model = model.strip()

        logger.info(f'Loading of the embedding model {model}')
        if model == 'BSV':
            embedding_model = embedding_models.BSV(
                checkpoint_model_path=bsv_checkpoints)
        else:
            try:
                embedding_model_cls = getattr(embedding_models, model)
                embedding_model = embedding_model_cls()
            except AttributeError:
                logger.warning(f'The model {model} is not supported.')
                continue

        logger.info(f'Creation of the H5 dataset for {model} ...')
        H5.create(embeddings_path, model, (n_sentences+1, embedding_model.dim))

        logger.info(f'Computation of the embeddings for {model} ...')
        for index in range(0, len(sentence_ids), args.step):
            try:
                final_embeddings, retrieved_indices = \
                    embedding_models.compute_database_embeddings(engine,
                                                                 embedding_model,
                                                                 sentence_ids[
                                                                     index:index+args.step])
                H5.write(embeddings_path, model, final_embeddings, retrieved_indices)

            except Exception as e:
                logger.error(f'Issues raised for sentence_ids[{index}'
                      f':{index+args.step}]')
                logger.error(e)
            logger.info(f'{index+args.step} sentences embeddings computed.')


if __name__ == "__main__":
    main()
