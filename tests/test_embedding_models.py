"""Tests covering embedding models."""

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

import importlib
import pickle
from pathlib import Path
from unittest.mock import Mock

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from sentence_transformers import SentenceTransformer

from bluesearch.embedding_models import (
    EmbeddingModel,
    MPEmbedder,
    SentTransformer,
    SklearnVectorizer,
    compute_database_embeddings,
    get_embedding_model,
)

GPU_IS_AVAILABLE = torch.cuda.is_available()


class TestEmbeddingModels:
    """The included tests do not use real models."""

    def test_abstract_class(self):
        # Test that "EmbeddingModel" is abstract
        assert "__abstractmethods__" in EmbeddingModel.__dict__
        assert len(EmbeddingModel.__dict__["__abstractmethods__"]) > 0

        # Test not overriding all abstract methods
        class WrongModel(EmbeddingModel):
            def embed(self, _):
                pass

        assert "__abstractmethods__" in WrongModel.__dict__
        assert len(WrongModel.__dict__["__abstractmethods__"]) > 0

    @pytest.mark.parametrize("n_sentences", [1, 5])
    def test_senttransf_embedding(self, monkeypatch, n_sentences):
        sentence_transormer_class = Mock()
        senttrans_model = Mock(spec=SentenceTransformer)
        senttrans_model.encode.return_value = np.ones(
            [n_sentences, 768]
        )  # Need to check the dimensions
        sentence_transormer_class.return_value = senttrans_model

        monkeypatch.setattr(
            "bluesearch.embedding_models.sentence_transformers.SentenceTransformer",
            sentence_transormer_class,
        )
        sbert = SentTransformer("bert-base-nli-mean-tokens")

        # Preparations
        dummy_sentence = "This is a dummy sentence/test."

        if n_sentences != 1:
            dummy_sentence = n_sentences * [dummy_sentence]

        preprocess_method = getattr(
            sbert, "preprocess" if n_sentences == 1 else "preprocess_many"
        )
        embed_method = getattr(sbert, "embed" if n_sentences == 1 else "embed_many")

        # Assertions
        assert sbert.dim == 768

        preprocessed_sentence = preprocess_method(dummy_sentence)
        assert preprocessed_sentence == dummy_sentence

        embedding = embed_method(preprocessed_sentence)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == ((768,) if n_sentences == 1 else (n_sentences, 768))
        senttrans_model.encode.assert_called_once()

    @pytest.mark.parametrize(
        "backend", ["TfidfVectorizer", "CountVectorizer", "HashingVectorizer"]
    )
    @pytest.mark.parametrize("n_sentences", [1, 5])
    def test_sklearnvectorizer_embedding(self, tmpdir, backend, n_sentences):
        train_sentences = [
            "This is a sentence to train our model.",
            "Another one just for fun.",
            "This is also used to train the model.",
            "And this sentence completes the sentences dataset.",
        ]
        module = importlib.import_module("sklearn.feature_extraction.text")
        backend_cls = getattr(module, backend)
        model = backend_cls()
        model.fit(train_sentences)
        save_file = Path(tmpdir) / "model.pkl"
        print("Saving model to disk...")
        with open(save_file, "wb") as f:
            pickle.dump(model, f)

        skl_vectorizer = SklearnVectorizer(checkpoint_path=save_file)

        # Preparations
        dummy_sentence = "This is a dummy sentence/test."
        if n_sentences != 1:
            dummy_sentence = n_sentences * [dummy_sentence]

        preprocess_method = getattr(
            skl_vectorizer, "preprocess" if n_sentences == 1 else "preprocess_many"
        )
        embed_method = getattr(
            skl_vectorizer, "embed" if n_sentences == 1 else "embed_many"
        )

        # Assertions
        if backend in ("TfidfVectorizer", "CountVectorizer"):
            assert skl_vectorizer.dim == 19
        elif backend == "HashingVectorizer":
            assert skl_vectorizer.dim == 2 ** 20
        else:
            raise ValueError(f"Don't know what to do with backend {backend}")

        preprocessed_sentence = preprocess_method(dummy_sentence)
        assert preprocessed_sentence == dummy_sentence

        embedding = embed_method(preprocessed_sentence)
        assert isinstance(embedding, np.ndarray)

        if n_sentences == 1:
            assert embedding.shape == (skl_vectorizer.dim,)
        else:
            assert embedding.shape == (n_sentences, skl_vectorizer.dim)

    def test_default_preprocess_many(self, monkeypatch):
        class NewModel(EmbeddingModel):
            @property
            def dim(self):
                return 2

            # just to be able to instantiate
            def embed(self, preprocessed_sentence):
                return np.ones(self.dim)

        model = NewModel()
        fake_preprocess = Mock()
        fake_preprocess.return_value = "I am a preprocessed sentence"
        monkeypatch.setattr(model, "preprocess", fake_preprocess)

        preprocessed_sentences = model.preprocess_many(["A", "B", "C"])

        assert isinstance(preprocessed_sentences, list)
        assert len(preprocessed_sentences) == 3
        assert fake_preprocess.call_count == 3

    def test_default_embed_many(self, monkeypatch):
        class NewModel(EmbeddingModel):
            @property
            def dim(self):
                return 2

            # just to be able to instantiate
            def embed(self, preprocessed_sentence):
                return np.ones(self.dim)

        model = NewModel()
        fake_embed = Mock()
        fake_embed.return_value = np.ones(model.dim)
        monkeypatch.setattr(model, "embed", fake_embed)

        embeddings = model.embed_many(["A", "B", "C"])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, model.dim)
        assert fake_embed.call_count == 3


@pytest.mark.parametrize("batch_size", [1, 5, 1000])
def test_compute_database(
    fake_sqlalchemy_engine,
    test_parameters,
    metadata_path,
    batch_size,
):

    n_articles = pd.read_csv(metadata_path)["cord_uid"].notna().sum()
    n_sentences = (
        n_articles
        * test_parameters["n_sections_per_article"]
        * test_parameters["n_sentences_per_section"]
    )

    fake_embedder = Mock(spec=SentTransformer)
    fake_embedder.preprocess_many.side_effect = lambda raw_sentences: raw_sentences
    fake_embedder.embed_many.side_effect = lambda preprocessed_sentences: np.ones(
        (len(preprocessed_sentences), 768)
    )

    indices = np.arange(1, n_sentences + 1)
    final_embeddings, retrieved_indices = compute_database_embeddings(
        fake_sqlalchemy_engine, fake_embedder, indices, batch_size=batch_size
    )

    assert final_embeddings.shape == (n_sentences, 768)
    assert np.all(indices == retrieved_indices)

    assert fake_embedder.embed_many.call_count == (n_sentences // batch_size) + int(
        n_sentences % batch_size != 0
    )


@pytest.mark.slow
class TestSentTransformer:
    @pytest.mark.parametrize(
        "device",
        [
            pytest.param(torch.device("cpu"), id="CPU"),
            pytest.param(
                torch.device("cuda:0"),
                id="GPU",
                marks=pytest.mark.skipif(
                    not GPU_IS_AVAILABLE, reason="No GPU available"
                ),
            ),
        ],
    )
    def test_all(self, device):
        model_name = "bert-base-nli-mean-tokens"
        model = SentTransformer(model_name, device=device)

        sentence = "This is a sample sentence"
        sentences = [sentence]

        preprocessed_sentence = model.preprocess(sentence)
        preprocessed_sentences = model.preprocess_many(sentences)

        assert sentence == preprocessed_sentence
        assert sentences == preprocessed_sentences

        emb = model.embed(preprocessed_sentence)
        embs = model.embed_many(preprocessed_sentences)

        assert isinstance(emb, np.ndarray)
        assert isinstance(embs, np.ndarray)

        assert emb.shape == (model.dim,)
        assert embs.shape == (1, model.dim)

        # Check all parameters are on the right device
        # SentenceTransformer sends tensors to the device inside of `encode`
        # (not when instantiated)
        for p in model.senttransf_model.parameters():
            assert p.device == device


class TestGetEmbeddingModel:
    def test_invalid_key(self):
        with pytest.raises(ValueError):
            get_embedding_model("wrong_model_name")

    @pytest.mark.parametrize(
        "name, underlying_class",
        [
            ("BioBERT NLI+STS", "SentTransformer"),
            ("SentTransformer", "SentTransformer"),
            ("SklearnVectorizer", "SklearnVectorizer"),
            ("SBioBERT", "SentTransformer"),
            ("SBERT", "SentTransformer"),
        ],
    )
    def test_returns_instance(self, monkeypatch, name, underlying_class):
        fake_instance = Mock()
        fake_class = Mock(return_value=fake_instance)

        monkeypatch.setattr(
            f"bluesearch.embedding_models.{underlying_class}", fake_class
        )

        returned_instance = get_embedding_model(name)

        assert returned_instance is fake_instance


class TestMPEmbedder:
    @pytest.mark.parametrize("dim", [2, 5])
    @pytest.mark.parametrize("batch_size", [1, 2, 10])
    def test_run_embedding_worker(
        self, fake_sqlalchemy_engine, monkeypatch, tmpdir, dim, batch_size
    ):
        class Random(EmbeddingModel):
            def __init__(self, _dim):
                self._dim = _dim

            @property
            def dim(self):
                return self._dim

            def embed(self, *args):
                return np.random.random(self.dim)

        temp_h5_path = Path(str(tmpdir)) / "temp.h5"
        indices = np.array([1, 4, 5])

        fake_get_embedding_model = Mock(return_value=Random(dim))

        monkeypatch.setattr(
            "bluesearch.embedding_models.get_embedding_model", fake_get_embedding_model
        )

        MPEmbedder.run_embedding_worker(
            database_url=fake_sqlalchemy_engine.url,
            model_name_or_class="some_model",
            indices=indices,
            temp_h5_path=temp_h5_path,
            batch_size=batch_size,
            gpu=3,
            checkpoint_path=None,
            h5_dataset_name="some_model",
        )

        assert temp_h5_path.exists()
        with h5py.File(temp_h5_path, "r") as f:
            assert "some_model" in f.keys()
            assert "some_model_indices" in f.keys()

            # data checks
            assert f["some_model"].shape == (len(indices), dim)
            assert f["some_model"].dtype == "float32"

            assert not np.any(np.isnan(f["some_model"][:]))

            # indices checks
            assert f["some_model_indices"].shape == (len(indices), 1)
            assert f["some_model_indices"].dtype == "int32"

            assert not np.any(np.isnan(f["some_model_indices"][:]))

        with pytest.raises(FileExistsError):
            MPEmbedder.run_embedding_worker(
                database_url=fake_sqlalchemy_engine.url,
                model_name_or_class="some_model",
                indices=indices,
                temp_h5_path=temp_h5_path,
                batch_size=batch_size,
                gpu=None,
                checkpoint_path=None,
                h5_dataset_name="some_model",
            )

    @pytest.mark.parametrize("n_processes", [1, 2, 5])
    def test_do_embedding(self, monkeypatch, tmp_path, n_processes):
        # test 1 gpu per process or not specified
        with pytest.raises(ValueError):
            MPEmbedder(
                "some_url",
                "some_model",
                np.array([2, 5, 11]),
                tmp_path,
                n_processes=2,
                gpus=[1, 4, 8],
            )

        mpe = MPEmbedder(
            "some_url",
            "some_model",
            np.array([2, 5, 11, 523, 523523, 3243223, 23424234]),
            tmp_path,
            n_processes=n_processes,
        )

        fake_multiprocessing = Mock()
        fake_h5 = Mock()
        fake_get_embedding_model = Mock()
        monkeypatch.setattr("bluesearch.embedding_models.mp", fake_multiprocessing)
        monkeypatch.setattr("bluesearch.embedding_models.H5", fake_h5)
        monkeypatch.setattr(
            "bluesearch.embedding_models.get_embedding_model", fake_get_embedding_model
        )

        mpe.do_embedding()

        # checks
        fake_multiprocessing.set_start_method.asset_called_once()
        assert fake_multiprocessing.Process.call_count == n_processes
        fake_h5.concatenate.assert_called_once()

        args, _ = fake_h5.concatenate.call_args
        assert len(args[2]) == n_processes
