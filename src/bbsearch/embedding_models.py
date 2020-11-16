"""Model handling sentences embeddings."""
import importlib
import logging
import multiprocessing as mp
import os
import pathlib
import string
from abc import ABC, abstractmethod

import joblib
import numpy as np
import sent2vec
import sentence_transformers
import spacy
import sqlalchemy
import tensorflow_hub as hub
import torch
from nltk import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoModel, AutoTokenizer

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


class Sent2VecModel(EmbeddingModel):
    """A sent2vec model.

    Parameters
    ----------
    checkpoint_path : pathlib.Path or str
        The path of the Sent2Vec model to use for the embeddings.
    """

    def __init__(self, checkpoint_path):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"The checkpoint file {self.checkpoint_path} was not found."
            )

        self.logger.info(f"Loading the checkpoint from {self.checkpoint_path}")
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(str(self.checkpoint_path))

        self.logger.info("Loading the preprocessing spacy model")
        # We only need the tokenizer of the spacy model, so we disable
        # all the other components. Note that vocab takes by far the most
        # time to load. Internally roughly the following steps take place:
        #   nlp = spacy.lang.en.English()
        #   nlp.tokenizer.from_disk(tokenizer_path, exclude=["vocab"])
        # (See `spacy.language.Language.from_disk`, here the tokenizer path is
        # "../site-packages/en_core_sci_lg/en_core_sci_lg-x.x.x/tokenizer")
        # so it doesn't seem that the vocab is necessary for the tokenization.
        self.nlp = spacy.load(
            name="en_core_sci_lg",
            disable=["tagger", "parser", "ner", "vocab"],
        )

    @property
    def dim(self):
        """Return dimension of the embedding.

        Returns
        -------
        dim : int
            The dimension of the embedding.
        """
        return self.model.get_emb_size()

    def _generate_preprocessed(self, sentences):
        """Preprocess sentences and yield results one by one.

        Parameters
        ----------
        sentences : iterable of str or str
            The sentences to be processed.

        Yields
        ------
        preprocessed_sentence : str
            A preprocessed sentence.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        for sentence_doc in self.nlp.pipe(sentences):
            preprocessed_sentence = " ".join(
                token.lemma_.lower()
                for token in sentence_doc
                if not (
                    token.is_punct
                    or token.is_stop
                    or token.like_num
                    or token.like_url
                    or token.like_email
                    or token.is_bracket
                )
            )

            yield preprocessed_sentence

    def preprocess(self, raw_sentence):
        """Preprocess one sentence.

        Parameters
        ----------
        raw_sentence : str
            Raw sentence to embed.

        Returns
        -------
        preprocessed_sentence : str
            Preprocessed sentence.
        """
        return next(self._generate_preprocessed(raw_sentence))

    def preprocess_many(self, raw_sentences):
        """Preprocess multiple sentences.

        Parameters
        ----------
        raw_sentences : list of str
            Sentences as they are are extracted from a body of text.

        Returns
        -------
        preprocessed_sentences : list of str
            Preprocessed sentences. Intended to be fed ot the `embed` or
            `embed_many` methods.
        """
        return list(self._generate_preprocessed(raw_sentences))

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
            Array of shape `(700,)` with the sentence embedding.
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
            Array of shape `(len(preprocessed_sentences), 700)` with the
            sentence embeddings.
        """
        embeddings = self.model.embed_sentences(preprocessed_sentences)
        return embeddings


class BSV(EmbeddingModel):
    """BioSentVec.

    Parameters
    ----------
    checkpoint_path : pathlib.Path or str
        The path of the BioSentVec (BSV) model to use for the embeddings.

    References
    ----------
    https://github.com/ncbi-nlp/BioSentVec
    """

    def __init__(self, checkpoint_path):
        checkpoint_path = pathlib.Path(checkpoint_path)
        self.checkpoint_path = checkpoint_path
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"The file {self.checkpoint_path} was not found."
            )
        self.bsv_model = sent2vec.Sent2vecModel()
        self.bsv_model.load_model(str(self.checkpoint_path))
        self.bsv_stopwords = set(stopwords.words("english"))

    @property
    def dim(self):
        """Return dimension of the embedding."""
        return 700

    def preprocess(self, raw_sentence):
        """Preprocess the sentence (Tokenization, ...).

        Parameters
        ----------
        raw_sentence : str
            Raw sentence to embed.

        Returns
        -------
        preprocessed_sentence : str
            Preprocessed sentence.
        """
        raw_sentence = raw_sentence.replace("/", " / ")
        raw_sentence = raw_sentence.replace(".-", " .- ")
        raw_sentence = raw_sentence.replace(".", " . ")
        raw_sentence = raw_sentence.replace("'", " ' ")
        raw_sentence = raw_sentence.lower()
        tokens = [
            token
            for token in word_tokenize(raw_sentence)
            if token not in string.punctuation and token not in self.bsv_stopwords
        ]
        return " ".join(tokens)

    def embed(self, preprocessed_sentence):
        """Compute the sentences embeddings for a given sentence.

        Parameters
        ----------
        preprocessed_sentence : str
            Preprocessed sentence to embed.

        Returns
        -------
        embedding : numpy.array
            Embedding of the specified sentence of shape (700,).
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
            `(len(preprocessed_sentences), 700)`.
        """
        embeddings = self.bsv_model.embed_sentences(preprocessed_sentences)
        return embeddings


class SentTransformer(EmbeddingModel):
    """Sentence Transformer.

    Parameters
    ----------
    checkpoint_path : pathlib.Path or str
        The name or the path of the Transformer model to use for the embeddings.

    References
    ----------
    https://github.com/UKPLab/sentence-transformers
    """

    def __init__(self, checkpoint_path, device=None):

        self.senttransf_model = sentence_transformers.SentenceTransformer(
            str(checkpoint_path), device=device
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


class USE(EmbeddingModel):
    """Universal Sentence Encoder.

    References
    ----------
    https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder?hl=en
    """

    def __init__(self, use_version=5):
        self.use_version = use_version
        self.use_model = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/"
            + str(self.use_version)
        )

    @property
    def dim(self):
        """Return dimension of the embedding."""
        return 512

    def embed(self, preprocessed_sentence):
        """Compute the sentences embeddings for a given sentence.

        Parameters
        ----------
        preprocessed_sentence : str
            Preprocessed sentence to embed.

        Returns
        -------
        embedding : numpy.array
            Embedding of the specified sentence of shape (512,).
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
            `(len(preprocessed_sentences), 512)`.
        """
        embedding = self.use_model(preprocessed_sentences).numpy()
        return embedding


class SklearnVectorizer(EmbeddingModel):
    """Simple wrapper for sklearn vectorizer models.

    Parameters
    ----------
    checkpoint_path : pathlib.Path or str
        The path of the scikit-learn model to use for the embeddings in Pickle format.
    """

    def __init__(self, checkpoint_path):
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.model = joblib.load(self.checkpoint_path)

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

    all_embeddings = list()
    all_ids = list()

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


def get_embedding_model(model_name_or_class, checkpoint_path=None, device=None):
    """Load a sentence embedding model from its name or its class and checkpoint.

    Usage:

    - For defined models:
        - BioBERT NLI+STS: `get_embedding_model('BioBERT_NLI+STS', <device>)`
        - SBioBERT: `get_embedding_model('SBioBERT', <device>)`
        - SBERT: `get_embedding_model('SBERT', <device>)`
        - USE: `get_embedding_model('USE')`
        - BSV: `get_embedding_model('Sent2VecModel', <checkpoint_path>)`
        - Sent2Vec: `get_embedding_model('Sent2VecModel', <checkpoint_path>)`

    - For arbitrary models:
        - My Transformer model: `get_embedding_model('SentTransformer', <checkpoint_path>, <device>)`
        - My Sent2Vec model: `get_embedding_model('Sent2VecModel', <checkpoint_path>)`
        - My scikit-learn model: `get_embedding_model('SklearnVectorizer', <checkpoint_path>)`

    Parameters
    ----------
    model_name_or_class : str
        The name or the class of the model to load.
    checkpoint_path : pathlib.Path
        When {model_name_or_class} is the model name, the path of the model to load.
    device : str
        The target device to which load the model. Can be {None, 'cpu', 'cuda'}.

    Returns
    -------
    bbsearch.embedding_models.EmbeddingModel
        The sentence embedding model instance.
    """
    configs = {
        'BioBERT_NLI+STS': ('SentTransformer', 'clagator/biobert_v1.1_pubmed_nli_sts'),
        'SBioBERT': ('SentTransformer', 'gsarti/biobert-nli'),
        'SBERT': ('SentTransformer', 'bert-base-nli-mean-tokens'),
    }
    kwargs = {'device': device} if device else {}
    if model_name_or_class in configs:
        if checkpoint_path is not None:
            raise ValueError(f"Cannot use 'checkpoint_path' when using a model name!")
        model_class, model_path = configs[model_name_or_class]
        kwargs['checkpoint_path'] = pathlib.Path(model_path)
    else:
        model_class = model_name_or_class
        kwargs['checkpoint_path'] = checkpoint_path
    try:
        module = importlib.import_module('bbsearch.embedding_models')
        model = getattr(module, model_class)
        return model(**kwargs)
    except AttributeError:
        raise ValueError(f'Unknown model name or class: {model_name_or_class}')


class MPEmbedder:
    """Embedding of sentences with multiprocessing.

    Parameters
    ----------
    database_url : str
        URL of the database.
    model_name_or_class : str
        Name or class of the embedding model to use.
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
        If provided, it represents the path to the trained model. Note
        that for some embedding models it is not necessary (they have
        a standard caching directory).
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
    ):
        self.database_url = database_url
        self.indices = indices
        self.h5_path_output = h5_path_output
        self.batch_size_inference = batch_size_inference
        self.batch_size_transfer = batch_size_transfer
        self.n_processes = n_processes
        self.checkpoint_path = checkpoint_path
        self.delete_temp = delete_temp
        self.temp_folder = temp_folder

        if checkpoint_path is None:
            self.model_name = model_name_or_class
            self.model_class = None
        else:
            self.model_name = checkpoint_path.stem
            self.model_class = model_name_or_class

        self.logger = logging.getLogger(f"{self.__class__.__name__}[{self.model_name}]")

        if gpus is not None and len(gpus) != n_processes:
            raise ValueError("One needs to specify the GPU for each process separately")

        self.gpus = gpus

    def do_embedding(self):
        """Do the parallelized embedding."""
        self.logger.info("Starting multiprocessing")
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
                    "model_name": self.model_name,
                    "model_class": self.model_class,
                    "checkpoint_path": self.checkpoint_path,
                    "database_url": self.database_url,
                    "indices": split,
                    "temp_h5_path": temp_h5_path,
                    "batch_size": self.batch_size_inference,
                    "gpu": None if self.gpus is None else self.gpus[process_ix],
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
            self.model_name,
            h5_paths_temp,
            delete_inputs=self.delete_temp,
            batch_size=self.batch_size_transfer,
        )
        self.logger.info("Concatenation done!")

    @staticmethod
    def run_embedding_worker(
        model_name,
        model_class,
        checkpoint_path,
        database_url,
        indices,
        temp_h5_path,
        batch_size,
        gpu,
    ):
        """Run per worker function.

        Parameters
        ----------
        model_name : str
            Name of the model to use. `None` when using `model_class` and `checkpoint_path`.
        model_class : str
            Class of the model to use. `None` when using `model_name`.
        checkpoint_path : pathlib.Path
            Path of the model to use. `None` when using `model_name`.
        database_url : str
            URL of the database.
        indices : np.ndarray
            1D array of sentences ids indices representing what
            the worker needs to embed.
        temp_h5_path : pathlib.Path
            Path to where we store the temporary h5 file.
        batch_size : int
            Number of sentences in the batch.
        gpu : int or None
            If None, we are going to use a CPU. Otherwise, we use a GPU
            with the specified id.
        """
        current_process = mp.current_process()
        cname = current_process.name
        cpid = current_process.pid

        logger = logging.getLogger(f"{cname}({cpid})")
        logger.info(f"First index={indices[0]}")

        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        logger.info("Loading model")
        model = get_embedding_model(
            model_name or model_class,
            checkpoint_path=checkpoint_path,
            device="cpu" if gpu is None else "cuda",
        )
        logger.info("Get sentences from the database")
        engine = sqlalchemy.create_engine(database_url)
        engine.dispose()

        if temp_h5_path.exists():
            raise FileExistsError(f"{temp_h5_path} already exists")

        n_indices = len(indices)
        logger.info("Create temporary h5 files.")
        H5.create(temp_h5_path, model_name, shape=(n_indices, model.dim))
        H5.create(
            temp_h5_path, f"{model_name}_indices", shape=(n_indices, 1), dtype="int32"
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

                H5.write(temp_h5_path, model_name, embeddings, pos_indices)

            except Exception as e:
                logger.error(f"Issues raised for sentence_ids[{batch_indices}]")
                logger.error(e)
                raise  # any error will lead to the child being stopped

            H5.write(
                temp_h5_path,
                f"{model_name}_indices",
                batch_indices.reshape(-1, 1),
                pos_indices,
            )

            logger.info(f"Finished {(split_ix + 1) / len(splits):.2%}")

        logger.info("CHILD IS DONE")
