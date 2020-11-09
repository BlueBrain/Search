"""Model handling sentences embeddings."""
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


class SBioBERT(EmbeddingModel):
    """Sentence BioBERT.

    Parameters
    ----------
    device : str
        Available device for the model. Can be {'cuda', 'cpu', None}

    References
    ----------
    https://huggingface.co/gsarti/biobert-nli
    """

    def __init__(self, device=None):
        available_device = device or "cpu"
        self.device = torch.device(available_device)
        self.sbiobert_model = AutoModel.from_pretrained("gsarti/biobert-nli").to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained("gsarti/biobert-nli")

    @property
    def dim(self):
        """Return dimension of the embedding."""
        return 768

    def preprocess(self, raw_sentence):
        """Preprocess the sentence - tokenization and determining of token ids.

        Note that this method already works in batched way if we pass a list.

        Parameters
        ----------
        raw_sentence : str or list of str
            Raw sentence to embed. One can also provide multiple sentences.

        Returns
        -------
        encoding : transformers.BatchEncoding
            Dictionary like object that holds the following keys: 'input_ids',
            'token_type_ids' and 'attention_mask'. All of the corresponding
            values are going to be ``torch.Tensor`` of shape `(n_sentences, n_tokens)`.

        References
        ----------
        https://huggingface.co/transformers/model_doc/bert.html#transformers.BertTokenizer

        """
        encoding = self.tokenizer(
            raw_sentence, pad_to_max_length=True, return_tensors="pt"
        )
        return encoding

    def preprocess_many(self, raw_sentences):
        """Preprocess multiple sentences - tokenization and determining of token ids.

        Parameters
        ----------
        raw_sentences : list of str
            List of raw sentence to embed.

        Returns
        -------
        encodings : transformers.BatchEncoding
            Dictionary like object that holds the following keys: 'input_ids',
            'token_type_ids' and 'attention_mask'. All of the corresponding
            values are going to be ``torch.Tensor`` of shape `(n_sentences, n_tokens)`.

        References
        ----------
        https://huggingface.co/transformers/model_doc/bert.html#transformers.BertTokenizer

        """
        return self.preprocess(raw_sentences)

    def embed(self, preprocessed_sentence):
        """Compute the sentence embedding for a given sentence.

        Note that this method already works in batched way if we pass a
        `BatchEncoding` that contains batches.

        Parameters
        ----------
        preprocessed_sentence : transformers.BatchEncoding
            Dictionary like object that holds the following keys: 'input_ids',
            'token_type_ids' and 'attention_mask'. All of the corresponding
            values are going to be ``torch.Tensor`` of shape `(n_sentences, n_tokens)`.

        Returns
        -------
        embedding : numpy.array
            Embedding of the specified sentence of shape (768,) if only a single
            sample in the batch. Otherwise `(len(preprocessed_sentences), 768)`.

        References
        ----------
        https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel
        """
        with torch.no_grad():
            last_hidden_state = self.sbiobert_model(
                **preprocessed_sentence.to(self.device)
            )[0]
            embedding = self.masked_mean(
                last_hidden_state, preprocessed_sentence["attention_mask"]
            )

        return embedding.squeeze().cpu().numpy()

    def embed_many(self, preprocessed_sentences):
        """Compute the sentences embeddings for multiple sentences.

        Parameters
        ----------
        preprocessed_sentences : transformers.BatchEncoding
            Dictionary like object that holds the following keys: 'input_ids',
            'token_type_ids' and 'attention_mask'. All of the corresponding
            values are going to be ``torch.Tensor`` of shape `(n_sentences, n_tokens)`.

        Returns
        -------
        embedding : numpy.array
            Embedding of the specified sentence of shape
            `(len(preprocessed_sentences), 768)`

        References
        ----------
        https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel
        """
        return self.embed(preprocessed_sentences)

    @staticmethod
    def masked_mean(last_hidden_state, attention_mask):
        """Compute the mean of token embeddings while taking into account the padding.

        Note that the `sequence_length` is going to be the number of tokens of
        the longest sentence + 2 (CLS and SEP are added).

        Parameters
        ----------
        last_hidden_state : torch.Tensor
            Per sample and per token embeddings as returned by the model. Shape
            `(n_sentences, sequence_length, dim)`.
        attention_mask : torch.Tensor
            Boolean mask of what tokens were padded (0) or not (1). The dtype
            is `torch.int64` and the shape is `(n_sentences, sequence_length)`.

        Returns
        -------
        sentence_embeddings : torch.Tensor
            Mean of token embeddings taking into account the padding. The shape
            is `(n_sentences, dim)`.

        References
        ----------
        https://github.com/huggingface/transformers/blob/82dd96cae74797be0c1d330566df7f929214b278/model_cards/sentence-transformers/bert-base-nli-mean-tokens/README.md
        """
        token_embeddings = last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        sentence_embeddings = sum_embeddings / sum_mask
        return sentence_embeddings


class Sent2VecModel(EmbeddingModel):
    """A sent2vec model.

    Parameters
    ----------
    checkpoint_path : pathlib.Path or str
        Location of the model checkpoint.
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
    checkpoint_model_path : pathlib.Path or str
        Path to the file of the stored model BSV.

    References
    ----------
    https://github.com/ncbi-nlp/BioSentVec
    """

    def __init__(self, checkpoint_model_path):
        checkpoint_model_path = pathlib.Path(checkpoint_model_path)
        self.checkpoint_model_path = checkpoint_model_path
        if not self.checkpoint_model_path.is_file():
            raise FileNotFoundError(
                f"The file {self.checkpoint_model_path} was not found."
            )
        self.bsv_model = sent2vec.Sent2vecModel()
        self.bsv_model.load_model(str(self.checkpoint_model_path))
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
    model_name : str
        Name of the model to use for the embeddings.
        Currently:
        - 'bert-base-nli-mean-tokens' is the one we use as SBERT
        - 'clagator/biobert_v1.1_pubmed_nli_sts' is the one we named BIOBERT NLI+STS

    References
    ----------
    https://github.com/UKPLab/sentence-transformers
    """

    def __init__(self, model_name="bert-base-nli-mean-tokens", device=None):

        self.senttransf_model = sentence_transformers.SentenceTransformer(
            model_name, device=device
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
        Location of the model checkpoint in pickle format.
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
    num_errors = 0

    for batch_ix in range((n_sentences // batch_size) + 1):
        start_ix = batch_ix * batch_size
        end_ix = min((batch_ix + 1) * batch_size, n_sentences)

        if start_ix == end_ix:
            continue

        sentences_text = sentences.iloc[start_ix:end_ix]["text"].to_list()
        sentences_id = sentences.iloc[start_ix:end_ix]["sentence_id"].to_list()

        try:
            preprocessed_sentences = model.preprocess_many(sentences_text)
            embeddings = model.embed_many(preprocessed_sentences)
        except IndexError:
            # This could happen when the sentence is too long for example
            num_errors += 1
            continue

        all_ids.extend(sentences_id)
        all_embeddings.append(embeddings)

        if batch_ix % 10 == 0:
            logger.info(f"Embedded {batch_ix} batches with {num_errors} errors")

    final_embeddings = np.concatenate(all_embeddings, axis=0)
    retrieved_indices = np.array(all_ids)
    return final_embeddings, retrieved_indices


def get_embedding_model(model_name, checkpoint_path=None, device=None):
    """Construct an embedding model from its name.

    Parameters
    ----------
    model_name : str
        The name of the model.
    checkpoint_path : pathlib.Path
        Path to load the embedding models (Needed for BSV and Sent2Vec).
    device : str
        If GPU are available, device='cuda' (Useful for BIOBERT NLI+STS,
        SBioBERT, SBERT).

    Returns
    -------
    bbsearch.embedding_models.EmbeddingModel
        The embedding model instance.
    """
    model_factories = {
        "BSV": lambda: BSV(checkpoint_model_path=checkpoint_path),
        "SBioBERT": lambda: SBioBERT(device=device),
        "USE": lambda: USE(),
        "SBERT": lambda: SentTransformer(
            model_name="bert-base-nli-mean-tokens", device=device
        ),
        "BIOBERT NLI+STS": lambda: SentTransformer(
            model_name="clagator/biobert_v1.1_pubmed_nli_sts", device=device
        ),
        "Sent2Vec": lambda: Sent2VecModel(checkpoint_path=checkpoint_path),
    }

    if model_name not in model_factories:
        raise ValueError(f"Unknown model name: {model_name}")
    selected_factory = model_factories[model_name]

    return selected_factory()


class MPEmbedder:
    """Embedding of sentences with multiprocessing.

    Parameters
    ----------
    database_url : str
        URL of the database.
    model_name : str
        Name of the embedding model to be used.
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
        If None, then all temporary h5 files stored into the same folders as the the ouput h5.
        Otherwise they are stored in the specified folder.

    """

    def __init__(
        self,
        database_url,
        model_name,
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
        self.model_name = model_name
        self.indices = indices
        self.h5_path_output = h5_path_output
        self.batch_size_inference = batch_size_inference
        self.batch_size_transfer = batch_size_transfer
        self.n_processes = n_processes
        self.checkpoint_path = checkpoint_path
        self.delete_temp = delete_temp
        self.temp_folder = temp_folder

        if gpus is not None and len(gpus) != n_processes:
            raise ValueError("One needs to specify the GPU for each process separately")

        self.gpus = gpus

    def do_embedding(self):
        """Do the parallelized embedding."""
        output_folder = self.temp_folder or self.h5_path_output.parent

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
                target=self.create_and_embed,
                kwargs={
                    "database_url": self.database_url,
                    "model_name": self.model_name,
                    "indices": split,
                    "temp_h5_path": temp_h5_path,
                    "batch_size": self.batch_size_inference,
                    "checkpoint_path": self.checkpoint_path,
                    "gpu": None if self.gpus is None else self.gpus[process_ix],
                },
            )
            worker_process.start()
            worker_processes.append(worker_process)
            h5_paths_temp.append(temp_h5_path)

        # Wait for the temporary
        for process in worker_processes:
            process.join()

        # Create a big h5 file from the temporary h5 files
        H5.concatenate(
            self.h5_path_output,
            self.model_name,
            h5_paths_temp,
            delete_inputs=self.delete_temp,
            batch_size=self.batch_size_transfer,
        )

    @staticmethod
    def create_and_embed(
        database_url,
        model_name,
        indices,
        temp_h5_path,
        batch_size,
        checkpoint_path,
        gpu,
    ):
        """Run per worker function.

        Parameters
        ----------
        database_url : str
            URL of the database.
        model_name : str
            Name of the model to use for embedding.
        indices : np.ndarray
            1D array of sentences ids indices representing what
            the worker needs to embed.
        temp_h5_path : pathlib.Path
            Path to where we store the temporary h5 file.
        batch_size : int
            Number of sentences in the batch.
        checkpoint_path : None or pathlib.Path
            If provided, then used to load the pretrained embedding model.
        gpu : int or None
            If None, we are going to use a CPU. Otherwise, we use a GPU
            with the specified id.
        """
        model = get_embedding_model(
            model_name,
            checkpoint_path=checkpoint_path,
            device="cpu" if gpu is None else "cuda",
        )

        engine = sqlalchemy.create_engine(database_url)
        engine.dispose()

        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        if temp_h5_path.exists():
            raise FileExistsError(f"{temp_h5_path} already exists")

        n_indices = len(indices)
        H5.create(temp_h5_path, model_name, shape=(n_indices, model.dim))
        H5.create(temp_h5_path, f"{model_name}_indices", shape=(n_indices, 1),  dtype="int32")

        for pos_indices in np.array_split(np.arange(n_indices), n_indices / batch_size):
            batch_indices = indices[pos_indices]
            embeddings, retrieved_indices = compute_database_embeddings(
                engine, model, batch_indices, batch_size=len(batch_indices)
            )

            if not np.array_equal(retrieved_indices, batch_indices):
                raise ValueError("The retrieved and requested indices do not agree.")

            H5.write(temp_h5_path, model_name, embeddings, pos_indices)
            H5.write(temp_h5_path,
                    f"{model_name}_indices",
                    batch_indices.reshape(-1, 1),
                    pos_indices)
