"""EntryPoint for the computation and saving of the embeddings."""
import argparse
import getpass
import logging
import os
import pathlib

import torch

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
                    default="embeddings_computation.log",
                    type=str,
                    help="The name of the log file.")
parser.add_argument("--models",
                    default='USE,SBERT,SBioBERT,BSV,Sent2Vec,BIOBERT NLI+STS',
                    type=str,
                    help="Models for which we need to compute the embeddings. "
                         "Format should be comma separated list.")
parser.add_argument("--bsv_checkpoints",
                    default='/raid/sync/proj115/bbs_data/trained_models/BioSentVec_PubMed_MIMICIII-bigram_d700.bin',
                    type=str,
                    help="Path to file containing the checkpoints for the BSV model.")
parser.add_argument("--sent2vec_checkpoints",
                    default='/raid/sync/proj115/bbs_data/trained_models/new_s2v_model.bin',
                    type=str,
                    help="Path to file containing the checkpoints for the sent2vec model.")
parser.add_argument("--step",
                    default='1000',
                    type=int,
                    help="Batch size for the embeddings computation")
args = parser.parse_args()


def get_embedding_model(model_name, checkpoint_path=None, device=None):
    """Construct an embedding model from its name.

    Parameters
    ----------
    model_name : str
        The name of the model.

    checkpoint_path : pathlib.Path
        Path to load the embedding models (Needed for BSV and Sent2Vec)

    device: str
        If GPU are available, device='cuda' (Useful for BIOBERT NLI+STS, SBioBERT, SBERT)

    Returns
    -------
    bbsearch.embedding_models.EmbeddingModel
        The embedding model instance.
    """
    from .. import embedding_models
    model_factories = {
        "BSV": lambda: embedding_models.BSV(checkpoint_model_path=checkpoint_path),
        "SBioBERT": lambda: embedding_models.SBioBERT(device=device),
        "USE": lambda: embedding_models.USE(),
        "SBERT": lambda: embedding_models.SentTransformer(model_name="bert-base-nli-mean-tokens",
                                                          device=device),
        "BIOBERT NLI+STS": lambda: embedding_models.SentTransformer(
            model_name="clagator/biobert_v1.1_pubmed_nli_sts", device=device),
        "Sent2Vec": lambda: embedding_models.Sent2VecModel(checkpoint_path=checkpoint_path)
    }

    if model_name not in model_factories:
        raise ValueError(f"Unknown model name: {model_name}")
    selected_factory = model_factories[model_name]

    return selected_factory()


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
    sent2vec_checkpoints = pathlib.Path(args.sent2vec_checkpoints)

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

    logger.info(f'{len(sentence_ids)} to embed / Total Number of sentences {n_sentences}')

    device = 'cpu'
    if torch.cuda.is_available():
        try:
            if os.environ['CUDA_VISIBLE_DEVICES']:
                device = 'cuda'
        except KeyError:
            logger.info('The environment variable CUDA_VISIBLE_DEVICES seems not specified.')

    logger.info(f'The device used for the embeddings computation is {device}.')

    for model in args.models.split(','):
        model = model.strip()

        logger.info(f'Loading of the embedding model {model}')

        checkpoint_path = None
        if model == 'BSV':
            checkpoint_path = bsv_checkpoints
        elif model == 'Sent2Vec':
            checkpoint_path = sent2vec_checkpoints

        try:
            embedding_model = get_embedding_model(model, checkpoint_path=checkpoint_path,
                                                  device=device)
        except ValueError:
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
                                                                     index:index+args.step],
                                                                 batch_size=args.step)
                H5.write(embeddings_path, model, final_embeddings, retrieved_indices)

            except Exception as e:
                logger.error(f'Issues raised for sentence_ids[{index}:{index+args.step}]')
                logger.error(e)
            logger.info(f'{index+args.step} sentences embeddings computed.')


if __name__ == "__main__":
    main()
