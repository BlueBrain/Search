"""EntryPoint for the computation and saving of the embeddings."""

# Blue Brain Search is a text mining toolbox focused on scientific use cases.
#
# Copyright (C) 2020  Blue Brain Project, EPFL.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import argparse
import logging
import pathlib
import sys
from typing import Optional

import numpy as np
import sqlalchemy

from ._helper import CombinedHelpFormatter, configure_logging, parse_args_or_environment


def run_compute_embeddings(argv=None):
    """Run CLI."""
    # CLI setup
    parser = argparse.ArgumentParser(
        formatter_class=CombinedHelpFormatter,
    )
    parser.add_argument(
        "model_name_or_class",
        type=str,
        help="""
        The name or class of the model for which to compute the embeddings.
        Recognized model names are: 'BioBERT NLI+STS', 'SBioBERT', 'SBERT'.
        Recognized model classes are: 'SentTransformer', 'SklearnVectorizer'.

        See also 'get_embedding_model(...)'.
        """,
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
        help="""
        If 'model_name_or_class' is the class, the path of the model to load.
        Otherwise, this argument is ignored.
        """,
    )
    parser.add_argument(
        "--db-url",
        type=str,
        help="""
        URL of the MySQL database. Generally, the scheme part of
        the URL should be omitted, i.e. the URL should be
        of the form 'my_sql_server.ch:1234/my_database'.

        If missing, then the environment variable DB_URL will be read.
        """,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gpus",
        type=str,
        help="""
        Comma separated list of GPU indices for each process. To only
        run on a CPU leave blank. For example '2,,3,' will use GPU 2 and 3
        for the 1st and 3rd process respectively. The processes 2 and 4 will
        be run on a CPU. By default using CPU for all processes.
        """,
    )
    parser.add_argument(
        "--h5-dataset-name",
        type=str,
        help="""
        The name of the dataset in the H5 file.  Otherwise, the value of
        'model_name_or_class' is used.
        """,
    )
    parser.add_argument(
        "--indices-path",
        type=str,
        help="""
        Path to a .npy file containing sentence ids to embed. Specifically,
        it is a 1D numpy array of integers representing the sentence ids. If
        not specified we embed all sentences in the database.
        """,
    )
    parser.add_argument(
        "--log-file",
        "-l",
        type=str,
        metavar="<filepath>",
        default=None,
        help="In addition to stderr, log messages to a file.",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="""
        The logging level. Possible values:
        - 50 for CRITICAL
        - 40 for ERROR
        - 30 for WARNING
        - 20 for INFO
        - 10 for DEBUG
        - 0 for NOTSET
        """,
    )
    parser.add_argument(
        "-n",
        "--n-processes",
        default=4,
        type=int,
        help="Number of processes to use",
    )
    parser.add_argument(
        "-s",
        "--start-method",
        default="forkserver",
        choices=["fork", "forkserver", "spawn"],
        type=str,
        help="""
        Multiprocessing starting method to be used. Note that using "fork" might
        lead to problems when doing GPU inference.
        """,
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        help="""
        The path to where temporary h5 files are saved. If not specified then
        identical to the folder in which the output h5 file is placed.
        """,
    )

    # Parse CLI arguments
    env_variable_names = {
        "db_url": "DB_URL",
    }
    args = parse_args_or_environment(parser, env_variable_names, argv=argv)

    # Configure logging
    configure_logging(args.log_file, args.log_level)
    logger = logging.getLogger(__name__)

    logger.info(" Configuration ".center(80, "-"))
    for k, v in vars(args).items():
        logger.info(f"{k:<32}: {v}")
    logger.info("-" * 80)

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
        start_method=args.start_method,
    )

    logger.info("Starting embedding")
    mpe.do_embedding()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_compute_embeddings())
