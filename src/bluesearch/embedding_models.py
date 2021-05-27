"""Model handling sentences embeddings."""

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

import logging
import multiprocessing as mp
import pathlib
import pickle  # nosec
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import sentence_transformers
import sqlalchemy

from .sql import retrieve_sentences_from_sentence_ids
from .utils import H5

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract interface for the Sentences Embeddings Models."""

    @property
    @abstractmethod
    def dim(self):
        """Return dimension of the embedding."""

    def preprocess(self, raw_sentence):
        """Preprocess the sentence (Tokenization, ...) if needed by the model.

        This is a default implementation that perform no preprocessing.
        Model specific preprocessing can be define within children classes.

        Parameters
        ----------
        raw_sentence : str
            Raw sentence to embed.

        Returns
        -------
        preprocessed_sentence
            Preprocessed sentence in the format expected by the model if needed.
        """
        return raw_sentence

    def preprocess_many(self, raw_sentences):
        """Preprocess multiple sentences.

        This is a default implementation and can be overridden by children classes.

        Parameters
        ----------
        raw_sentences : list of str
            List of str representing raw sentences that we want to embed.

        Returns
        -------
        preprocessed_sentences
            List of preprocessed sentences corresponding to `raw_sentences`.
        """
        return [self.preprocess(sentence) for sentence in raw_sentences]

    @abstractmethod
    def embed(self, preprocessed_sentence):
        """Compute the sentences embeddings for a given sentence.

        Parameters
        ----------
        preprocessed_sentence : str
            Preprocessed sentence to embed.

        Returns
        -------
        embedding : numpy.array
            One dimensional vector representing the embedding of the given sentence.
        """

    def embed_many(self, preprocessed_sentences):
        """Compute sentence embeddings for all provided sentences.

        This is a default implementation. Children classes can implement more
        sophisticated batching schemes.

        Parameters
        ----------
        preprocessed_sentences : list of str
            List of preprocessed sentences.

        Returns
        -------
        embeddings : np.ndarray
            2D numpy array with shape `(len(preprocessed_sentences), self.dim)`.
            Each row is an embedding of a sentence in `preprocessed_sentences`.
        """
        return np.array([self.embed(sentence) for sentence in preprocessed_sentences])


class SentTransformer(EmbeddingModel):
    """Sentence Transformer.

    Parameters
    ----------
    model_name_or_path : pathlib.Path or str
        The name or the path of the Transformer model to load.

    References
    ----------
    https://github.com/UKPLab/sentence-transformers
    """

    def __init__(self, model_name_or_path, device=None):

        self.senttransf_model = sentence_transformers.SentenceTransformer(
            str(model_name_or_path), device=device
        )

    @property
    def dim(self):
        """Return dimension of the embedding."""
        return 768

    def embed(self, preprocessed_sentence):
        """Compute the sentences embeddings for a given sentence.

        Parameters
        ----------
        preprocessed_sentence : str
            Preprocessed sentence to embed.

        Returns
        -------
        embedding : numpy.array
            Embedding of the given sentence of shape (768,).
        """
        return self.embed_many([preprocessed_sentence]).squeeze()

    def embed_many(self, preprocessed_sentences):
        """Compute sentence embeddings for multiple sentences.

        Parameters
        ----------
        preprocessed_sentences : list of str
            Preprocessed sentences to embed.

        Returns
        -------
        embedding : numpy.array
            Embedding of the specified sentences of shape
            `(len(preprocessed_sentences), 768)`.
        """
        embeddings = np.array(self.senttransf_model.encode(preprocessed_sentences))
        return embeddings


class SklearnVectorizer(EmbeddingModel):
    """Simple wrapper for sklearn vectorizer models.

    Parameters
    ----------
    checkpoint_path : pathlib.Path or str
        The path of the scikit-learn model to use for the embeddings in Pickle format.
    """

    def __init__(self, checkpoint_path):
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        with self.checkpoint_path.open("rb") as f:
            self.model = pickle.load(f)  # nosec

    @property
    def dim(self):
        """Return dimension of the embedding.

        Returns
        -------
        dim : int
            The dimension of the embedding.
        """
        if hasattr(self.model, "n_features"):  # e.g. HashingVectorizer
            return self.model.n_features
        elif hasattr(self.model, "vocabulary_"):  # e.g. TfIdfVectorizer
            return len(self.model.vocabulary_)
        else:
            raise NotImplementedError(
                f"Something went wrong, embedding dimension for class "
                f"{type(self.model)} could not be computed."
            )

    def embed(self, preprocessed_sentence):
        """Embed one given sentence.

        Parameters
        ----------
        preprocessed_sentence : str
            Preprocessed sentence to embed. Can by obtained using the
            `preprocess` or `preprocess_many` methods.

        Returns
        -------
        embedding : numpy.ndarray
            Array of shape `(dim,)` with the sentence embedding.
        """
        embedding = self.embed_many([preprocessed_sentence])
        return embedding.squeeze()

    def embed_many(self, preprocessed_sentences):
        """Compute sentence embeddings for multiple sentences.

        Parameters
        ----------
        preprocessed_sentences : iterable of str
            Preprocessed sentences to embed. Can by obtained using the
            `preprocess` or `preprocess_many` methods.

        Returns
        -------
        embeddings : numpy.ndarray
            Array of shape `(len(preprocessed_sentences), dim)` with the
            sentence embeddings.
        """
        embeddings = self.model.transform(preprocessed_sentences).toarray()
        return embeddings


def compute_database_embeddings(connection, model, indices, batch_size=10):
    """Compute sentences embeddings.

    The embeddings are computed for a given model and a given database
    (articles with covid19_tag True).

    Parameters
    ----------
    connection : sqlalchemy.engine.Engine
        Connection to the database.
    model : EmbeddingModel
        Instance of the EmbeddingModel of choice.
    indices : np.ndarray
        1D array storing the sentence_ids for which we want to perform the
        embedding.
    batch_size : int
        Number of sentences to preprocess and embed at the same time. Should
        lead to major speedups. Note that the last batch will have a length of
        `n_sentences % batch_size` (unless it is 0). Note that some models
        (SBioBERT) might perform padding to the longest sentence and bigger
        batch size might not lead to a speedup.

    Returns
    -------
    final_embeddings : np.array
        2D numpy array with all sentences embeddings for the given models. Its
        shape is `(len(retrieved_indices), dim)`.
    retrieved_indices : np.ndarray
        1D array of sentence_ids that we managed to embed. Note that the order
        corresponds exactly to the rows in `final_embeddings`.
    """
    sentences = retrieve_sentences_from_sentence_ids(indices, connection)
    n_sentences = len(sentences)

    all_embeddings = []
    all_ids = []

    for batch_ix in range((n_sentences // batch_size) + 1):
        start_ix = batch_ix * batch_size
        end_ix = min((batch_ix + 1) * batch_size, n_sentences)

        if start_ix == end_ix:
            continue

        sentences_text = sentences.iloc[start_ix:end_ix]["text"].to_list()
        sentences_id = sentences.iloc[start_ix:end_ix]["sentence_id"].to_list()

        preprocessed_sentences = model.preprocess_many(sentences_text)
        embeddings = model.embed_many(preprocessed_sentences)

        all_ids.extend(sentences_id)
        all_embeddings.append(embeddings)

    final_embeddings = np.concatenate(all_embeddings, axis=0)
    retrieved_indices = np.array(all_ids)

    return final_embeddings, retrieved_indices


def get_embedding_model(
    model_name_or_class: str,
    checkpoint_path: Optional[Union[pathlib.Path, str]] = None,
    device: str = "cpu",
) -> EmbeddingModel:
    """Load a sentence embedding model from its name or its class and checkpoint.

    Usage:

    - For defined models:
        - BioBERT NLI+STS:
          `get_embedding_model('BioBERT NLI+STS', device=<device>)`
        - SBioBERT:
          `get_embedding_model('SBioBERT', device=<device>)`
        - SBERT:
          `get_embedding_model('SBERT', device=<device>)`

    - For arbitrary models:
        - My Transformer model:
          `get_embedding_model('SentTransformer', <model_name_or_path>, <device>)`
        - My scikit-learn model:
          `get_embedding_model('SklearnVectorizer', <checkpoint_path>)`

    Parameters
    ----------
    model_name_or_class
        The name or class of the embedding model to load.
    checkpoint_path
        If 'model_name_or_class' is the class, this parameter is required and
        it is the path of the embedding model to load.
    device
        The target device to which load the model ('cpu' or 'cuda').

    Returns
    -------
    sentence_embedding_model : EmbeddingModel
        The sentence embedding model instance.
    """
    configs = {
        # Transformer models.
        "SentTransformer": lambda: SentTransformer(checkpoint_path, device),
        "BioBERT NLI+STS": lambda: SentTransformer(
            "clagator/biobert_v1.1_pubmed_nli_sts", device
        ),
        "SBioBERT": lambda: SentTransformer("gsarti/biobert-nli", device),
        "SBERT": lambda: SentTransformer("bert-base-nli-mean-tokens", device),
        # Scikit-learn models.
        "SklearnVectorizer": lambda: SklearnVectorizer(checkpoint_path),
    }
    if model_name_or_class not in configs:
        raise ValueError(f"Unknown model name or class: {model_name_or_class}")
    return configs[model_name_or_class]()


class MPEmbedder:
    """Embedding of sentences with multiprocessing.

    Parameters
    ----------
    database_url : str
        URL of the database.
    model_name_or_class : str
        The name or class of the model for which to compute the embeddings.
    indices : np.ndarray
        1D array storing the sentence_ids for which we want to compute the
        embedding.
    h5_path_output : pathlib.Path
        Path to where the output h5 file will be lying.
    batch_size_inference : int
        Number of sentences to preprocess and embed at the same time. Should
        lead to major speedups. Note that the last batch will have a length of
        `n_sentences % batch_size` (unless it is 0). Note that some models
        (SBioBERT) might perform padding to the longest sentence in the batch
        and bigger batch size might not lead to a speedup.
    batch_size_transfer : int
        Batch size to be used for transfering data from the temporary h5 files to the
        final h5 file.
    n_processes : int
        Number of processes to use. Note that each process gets
        `len(indices) / n_processes` sentences to embed.
    checkpoint_path : pathlib.Path or None
        If 'model_name_or_class' is the class, the path of the model to load.
        Otherwise, this argument is ignored.
    gpus : None or list
        If not specified, all processes will be using CPU. If not None, then
        it needs to be a list of length `n_processes` where each element
        represents the GPU id (integer) to be used. None elements will
        be interpreted as CPU.
    delete_temp : bool
        If True, the temporary h5 files are deleted after the final h5 is created.
        Disabling this flag is useful for testing and debugging purposes.
    temp_folder : None or pathlib.Path
        If None, then all temporary h5 files stored into the same folder as the output
        h5 file. Otherwise they are stored in the specified folder.
    h5_dataset_name : str or None
        The name of the dataset in the H5 file.
        Otherwise, the value of 'model_name_or_class' is used.
    start_method : str, {"fork", "forkserver", "spawn"}
        Start method for multiprocessing. Note that using "fork" might
        lead to problems when doing GPU inference.
    preinitialize : bool
        If True we instantiate the model before running multiprocessing
        in order to download any checkpoints. Once instantiated, the model
        will be deleted.
    """

    def __init__(
        self,
        database_url,
        model_name_or_class,
        indices,
        h5_path_output,
        batch_size_inference=16,
        batch_size_transfer=1000,
        n_processes=2,
        checkpoint_path=None,
        gpus=None,
        delete_temp=True,
        temp_folder=None,
        h5_dataset_name=None,
        start_method="forkserver",
        preinitialize=True,
    ):
        self.database_url = database_url
        self.model_name_or_class = model_name_or_class
        self.indices = indices
        self.h5_path_output = h5_path_output
        self.batch_size_inference = batch_size_inference
        self.batch_size_transfer = batch_size_transfer
        self.n_processes = n_processes
        self.checkpoint_path = checkpoint_path
        self.delete_temp = delete_temp
        self.temp_folder = temp_folder
        self.start_method = start_method
        self.preinitialize = preinitialize
        if h5_dataset_name is None:
            self.h5_dataset_name = model_name_or_class
        else:
            self.h5_dataset_name = h5_dataset_name

        self.logger = logging.getLogger(
            f"{self.__class__.__name__}[{self.h5_dataset_name}]"
        )

        if gpus is not None and len(gpus) != n_processes:
            raise ValueError("One needs to specify the GPU for each process separately")

        self.gpus = gpus

    def do_embedding(self):
        """Do the parallelized embedding."""
        if self.preinitialize:
            self.logger.info("Preinitializing model (download of checkpoints)")
            model_temp = get_embedding_model(  # noqa
                self.model_name_or_class, checkpoint_path=self.checkpoint_path
            )
            del model_temp

        self.logger.info("Starting multiprocessing")
        mp.set_start_method(self.start_method, force=True)

        output_folder = self.temp_folder or self.h5_path_output.parent

        self.h5_path_output.parent.mkdir(parents=True, exist_ok=True)
        output_folder.mkdir(parents=True, exist_ok=True)

        worker_processes = []
        splits = [
            x for x in np.array_split(self.indices, self.n_processes) if len(x) > 0
        ]
        h5_paths_temp = []

        for process_ix, split in enumerate(splits):
            temp_h5_path = (
                output_folder / f"{self.h5_path_output.stem}_temp{process_ix}.h5"
            )

            worker_process = mp.Process(
                name=f"worker_{process_ix}",
                target=self.run_embedding_worker,
                kwargs={
                    "database_url": self.database_url,
                    "model_name_or_class": self.model_name_or_class,
                    "indices": split,
                    "temp_h5_path": temp_h5_path,
                    "batch_size": self.batch_size_inference,
                    "checkpoint_path": self.checkpoint_path,
                    "gpu": None if self.gpus is None else self.gpus[process_ix],
                    "h5_dataset_name": self.h5_dataset_name,
                },
            )
            worker_process.start()
            worker_processes.append(worker_process)
            h5_paths_temp.append(temp_h5_path)

        self.logger.info("Waiting for children to be done")
        for process in worker_processes:
            process.join()

        self.logger.info("Concatenating children temp h5")
        H5.concatenate(
            self.h5_path_output,
            self.h5_dataset_name,
            h5_paths_temp,
            delete_inputs=self.delete_temp,
            batch_size=self.batch_size_transfer,
        )
        self.logger.info("Concatenation done!")

    @staticmethod
    def run_embedding_worker(
        database_url,
        model_name_or_class,
        indices,
        temp_h5_path,
        batch_size,
        checkpoint_path,
        gpu,
        h5_dataset_name,
    ):
        """Run per worker function.

        Parameters
        ----------
        database_url : str
            URL of the database.
        model_name_or_class : str
            The name or class of the model for which to compute the embeddings.
        indices : np.ndarray
            1D array of sentences ids indices representing what
            the worker needs to embed.
        temp_h5_path : pathlib.Path
            Path to where we store the temporary h5 file.
        batch_size : int
            Number of sentences in the batch.
        checkpoint_path : pathlib.Path or None
            If 'model_name_or_class' is the class, the path of the model to load.
            Otherwise, this argument is ignored.
        gpu : int or None
            If None, we are going to use a CPU. Otherwise, we use a GPU
            with the specified id.
        h5_dataset_name : str or None
            The name of the dataset in the H5 file.
        """
        current_process = mp.current_process()
        cname = current_process.name
        cpid = current_process.pid

        logger = logging.getLogger(f"{cname}({cpid})")
        logger.info(f"First index={indices[0]}")

        device = "cpu" if gpu is None else f"cuda:{gpu}"

        logger.info("Loading model")
        model = get_embedding_model(
            model_name_or_class,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        logger.info("Get sentences from the database")
        engine = sqlalchemy.create_engine(database_url)
        engine.dispose()

        if temp_h5_path.exists():
            raise FileExistsError(f"{temp_h5_path} already exists")

        n_indices = len(indices)
        logger.info("Create temporary h5 files.")
        H5.create(temp_h5_path, h5_dataset_name, shape=(n_indices, model.dim))
        H5.create(
            temp_h5_path,
            f"{h5_dataset_name}_indices",
            shape=(n_indices, 1),
            dtype="int32",
        )

        batch_size = min(n_indices, batch_size)

        logger.info("Populating h5 files")
        splits = np.array_split(np.arange(n_indices), n_indices / batch_size)
        splits = [split for split in splits if len(split) > 0]

        for split_ix, pos_indices in enumerate(splits):
            batch_indices = indices[pos_indices]

            try:
                embeddings, retrieved_indices = compute_database_embeddings(
                    engine, model, batch_indices, batch_size=len(batch_indices)
                )

                if not np.array_equal(retrieved_indices, batch_indices):
                    raise ValueError(
                        "The retrieved and requested indices do not agree."
                    )

                H5.write(temp_h5_path, h5_dataset_name, embeddings, pos_indices)

            except Exception as e:
                logger.error(f"Issues raised for sentence_ids[{batch_indices}]")
                logger.error(e)
                raise  # any error will lead to the child being stopped

            H5.write(
                temp_h5_path,
                f"{h5_dataset_name}_indices",
                batch_indices.reshape(-1, 1),
                pos_indices,
            )

            logger.info(f"Finished {(split_ix + 1) / len(splits):.2%}")

        logger.info("CHILD IS DONE")
