"""EntryPoint for the computation and saving of the embeddings."""
import argparse
import logging
import pathlib
import sys
from typing import Optional

import numpy as np
import sqlalchemy

from ._helper import configure_logging


def run_compute_embeddings(argv=None):
    """Run CLI."""
    # CLI setup
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "model_name_or_class",
        type=str,
        help="The name or class of the model for which to compute the embeddings."
        " Recognized model names are: 'BioBERT NLI+STS', 'SBioBERT', 'SBERT', 'USE'."
        " Recognized model classes are: 'SentTransformer', 'Sent2VecModel', 'BSV',"
        " 'SklearnVectorizer'."
        " See also 'get_embedding_model(...)'.",
    )
    parser.add_argument(
        "outfile",
        type=str,
        help="The path to where the embeddings are saved (h5 file)",
    )
    parser.add_argument(
        "--batch-size-inference",
        default=256,
        type=int,
        help="Batch size for embeddings computation",
    )
    parser.add_argument(
        "--batch-size-transfer",
        default=1000,
        type=int,
        help="Batch size for the concatenation of temp h5 files",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        help="If 'model_name_or_class' is the class, the path of the model to load."
        " Otherwise, this argument is ignored.",
    )
    parser.add_argument(
        "--db-url",
        default="dgx1.bbp.epfl.ch:8853/cord19_v47",
        type=str,
        help="Url of the database",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        help="Comma seperated list of GPU indices for each process. To only "
        "run on a CPU leave blank. For example '2,,3,' will use GPU 2 and 3 "
        "for the 1st and 3rd process respectively. The processes 2 and 4 will "
        "be run on a CPU. By default using CPU for all processes.",
    )
    parser.add_argument(
        "--h5-dataset-name",
        type=str,
        help="The name of the dataset in the H5 file."
        " Otherwise, the value of 'model_name_or_class' is used.",
    )
    parser.add_argument(
        "--indices-path",
        type=str,
        help="Path to a .npy file containing sentence ids to embed. Specifically, "
        "it is a 1D numpy array of integers representing the sentence ids. If "
        "not specified we embedd all sentences in the database.",
    )
    parser.add_argument(
        "--log-dir",
        default="/raid/projects/bbs/logs/",
        type=str,
        help="The directory path where to save the logs",
    )
    parser.add_argument(
        "--log-name",
        default="embeddings_computation.log",
        type=str,
        help="The name of the log file",
    )
    parser.add_argument(
        "-n",
        "--n-processes",
        default=4,
        type=int,
        help="Number of processes to use",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        help="The path to where temporary h5 files are saved. If not "
        "specified then identical to the folder in which the output h5 "
        "file is placed.",
    )
    args = parser.parse_args(argv)

    # Configure logging
    log_file = pathlib.Path(args.log_dir) / args.log_name
    configure_logging(log_file, logging.INFO)
    logger = logging.getLogger(__name__)

    # Imports (they are here to make --help quick)
    logger.info("Loading libraries")
    from ..embedding_models import MPEmbedder

    # Database related
    logger.info("SQL Alchemy Engine creation ....")
    full_url = f"mysql+mysqldb://guest:guest@{args.db_url}?charset=utf8mb4"
    engine = sqlalchemy.create_engine(full_url)

    # Path preparation and checking
    out_file = pathlib.Path(args.outfile)
    temp_dir = None if args.temp_dir is None else pathlib.Path(args.temp_dir)
    checkpoint_path: Optional[pathlib.Path] = None
    if args.checkpoint is not None:
        checkpoint_path = pathlib.Path(args.checkpoint)
    indices_path = (
        None if args.indices_path is None else pathlib.Path(args.indices_path)
    )

    # Parse GPUs
    if args.gpus is None:
        gpus = None
    else:
        gpus = [None if x == "" else int(x) for x in args.gpus.split(",")]

    if indices_path is not None:
        if indices_path.exists():
            indices = np.load(str(indices_path))
        else:
            raise FileNotFoundError(f"Indices file {indices_path} does not exist!")

    else:
        n_sentences = list(engine.execute("SELECT COUNT(*) FROM sentences"))[0][0]
        indices = np.arange(1, n_sentences + 1)

    logger.info("Instantiating MPEmbedder")
    mpe = MPEmbedder(
        engine.url,
        args.model_name_or_class,
        indices,
        out_file,
        batch_size_inference=args.batch_size_inference,
        batch_size_transfer=args.batch_size_transfer,
        n_processes=args.n_processes,
        checkpoint_path=checkpoint_path,
        gpus=gpus,
        temp_folder=temp_dir,
        h5_dataset_name=args.h5_dataset_name,
    )

    logger.info("Starting embedding")
    mpe.do_embedding()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_compute_embeddings())
