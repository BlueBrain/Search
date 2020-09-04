from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest
import sent2vec
import tensorflow as tf
import torch
import transformers
from sentence_transformers import SentenceTransformer

from bbsearch.embedding_models import (
    BSV,
    SBERT,
    USE,
    EmbeddingModel,
    SBioBERT,
    compute_database_embeddings,
)


class TestEmbeddingModels:

    def test_abstractclass(self):
        with pytest.raises(TypeError):
            EmbeddingModel()

        class WrongModel(EmbeddingModel):

            def embed(a):
                pass

        with pytest.raises(TypeError):
            WrongModel()

    @pytest.mark.parametrize('n_sentences', [1, 5])
    def test_sbiobert_embedding(self, monkeypatch, n_sentences):
        torch_model = MagicMock(spec=torch.nn.Module)
        torch_model.return_value = (torch.ones([n_sentences, 10, 768]), None)  # 10 tokens

        auto_model = Mock()
        auto_model.from_pretrained().to.return_value = torch_model

        tokenizer = Mock()
        be = MagicMock(spec=transformers.BatchEncoding)
        be.keys.return_value = ['input_ids', 'token_type_ids', 'attention_mask']
        be.__getitem__.side_effect = lambda x: torch.tensor(torch.ones([n_sentences, 10]))
        tokenizer.return_value = be

        auto_tokenizer = Mock()
        auto_tokenizer.from_pretrained.return_value = tokenizer

        monkeypatch.setattr('bbsearch.embedding_models.AutoTokenizer', auto_tokenizer)
        monkeypatch.setattr('bbsearch.embedding_models.AutoModel', auto_model)

        sbiobert = SBioBERT()

        # Preparations
        dummy_sentence = 'This is a dummy sentence'

        if n_sentences != 1:
            dummy_sentence = n_sentences * [dummy_sentence]

        preprocess_method = getattr(sbiobert, 'preprocess' if n_sentences == 1 else 'preprocess_many')
        embed_method = getattr(sbiobert, 'embed' if n_sentences == 1 else 'embed_many')

        preprocess_sentence = preprocess_method(dummy_sentence)
        embedding = embed_method(preprocess_sentence)

        # Assertions
        assert sbiobert.dim == 768
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == ((768,) if n_sentences == 1 else (n_sentences, 768))
        torch_model.assert_called_once()
        tokenizer.assert_called_once()

    @pytest.mark.parametrize('n_sentences', [1, 5])
    def test_bsv_embedding(self, monkeypatch, tmpdir, n_sentences):
        sent2vec_module = Mock()
        bsv_model = Mock(spec=sent2vec.Sent2vecModel)
        bsv_model.embed_sentences.return_value = np.ones([n_sentences, 700])
        sent2vec_module.Sent2vecModel.return_value = bsv_model

        monkeypatch.setattr('bbsearch.embedding_models.sent2vec', sent2vec_module)

        new_file_path = Path(str(tmpdir)) / 'test.txt'
        new_file_path.touch()
        with pytest.raises(FileNotFoundError):
            BSV(checkpoint_model_path=Path(''))
        bsv = BSV(Path(new_file_path))

        # Preparation
        dummy_sentence = 'This is a dummy sentence/test.'
        preprocess_truth = 'dummy sentence test'

        if n_sentences != 1:
            dummy_sentence = n_sentences * [dummy_sentence]
            preprocess_truth = n_sentences * [preprocess_truth]

        preprocess_method = getattr(bsv, 'preprocess' if n_sentences == 1 else 'preprocess_many')
        embed_method = getattr(bsv, 'embed' if n_sentences == 1 else 'embed_many')

        # Assertions
        assert bsv.dim == 700

        preprocess_sentence = preprocess_method(dummy_sentence)
        assert isinstance(preprocess_sentence, str if n_sentences == 1 else list)
        assert preprocess_sentence == preprocess_truth

        embedding = embed_method(preprocess_sentence)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == ((700,) if n_sentences == 1 else (n_sentences, 700))
        bsv_model.embed_sentences.assert_called_once()

    @pytest.mark.parametrize('n_sentences', [1, 5])
    def test_sbert_embedding(self, monkeypatch, n_sentences):
        sentence_transormer_class = Mock()
        sbert_model = Mock(spec=SentenceTransformer)
        sbert_model.encode.return_value = np.ones([n_sentences, 768])  # Need to check the dimensions
        sentence_transormer_class.return_value = sbert_model

        monkeypatch.setattr('bbsearch.embedding_models.SentenceTransformer', sentence_transormer_class)
        sbert = SBERT()

        # Preparations
        dummy_sentence = 'This is a dummy sentence/test.'

        if n_sentences != 1:
            dummy_sentence = n_sentences * [dummy_sentence]

        preprocess_method = getattr(sbert, 'preprocess' if n_sentences == 1 else 'preprocess_many')
        embed_method = getattr(sbert, 'embed' if n_sentences == 1 else 'embed_many')

        # Assertions
        assert sbert.dim == 768

        preprocessed_sentence = preprocess_method(dummy_sentence)
        assert preprocessed_sentence == dummy_sentence

        embedding = embed_method(preprocessed_sentence)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == ((768,) if n_sentences == 1 else (n_sentences, 768))
        sbert_model.encode.assert_called_once()

    @pytest.mark.parametrize('n_sentences', [1, 5])
    def test_use_embedding(self, monkeypatch, n_sentences):
        hub_module = Mock()
        use_model = Mock()
        hub_module.load.return_value = use_model
        use_model.return_value = tf.ones((n_sentences, 512))

        monkeypatch.setattr('bbsearch.embedding_models.hub', hub_module)
        use = USE()

        # Preparations
        dummy_sentence = 'This is a dummy sentence/test.'

        if n_sentences != 1:
            dummy_sentence = n_sentences * [dummy_sentence]

        preprocess_method = getattr(use, 'preprocess' if n_sentences == 1 else 'preprocess_many')
        embed_method = getattr(use, 'embed' if n_sentences == 1 else 'embed_many')

        # Assertions
        assert use.dim == 512

        preprocessed_sentence = preprocess_method(dummy_sentence)
        assert preprocessed_sentence == dummy_sentence

        embedding = embed_method(preprocessed_sentence)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == ((512,) if n_sentences == 1 else (n_sentences, 512))
        use_model.assert_called_once()

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
        monkeypatch.setattr(model, 'preprocess', fake_preprocess)

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
        monkeypatch.setattr(model, 'embed', fake_embed)

        embeddings = model.embed_many(["A", "B", "C"])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, model.dim)
        assert fake_embed.call_count == 3


@pytest.mark.parametrize("batch_size", [1, 5, 1000])
def test_compute_database(monkeypatch, fake_sqlalchemy_engine, test_parameters, metadata_path, tmpdir,
                          batch_size):

    n_articles = pd.read_csv(metadata_path)['cord_uid'].notna().sum()
    n_sentences = n_articles * test_parameters['n_sections_per_article'] * test_parameters[
        'n_sentences_per_section']

    # We use the BSV model to make sure it works
    sent2vec_module = Mock()
    bsv_model = Mock(spec=sent2vec.Sent2vecModel, side_effect=lambda x:  np.ones([len(x), 700]))
    bsv_model.embed_sentences.side_effect = lambda x:  np.ones([len(x), 700])
    sent2vec_module.Sent2vecModel.return_value = bsv_model

    new_file_path = Path(str(tmpdir)) / 'test.txt'
    new_file_path.touch()

    monkeypatch.setattr('bbsearch.embedding_models.sent2vec', sent2vec_module)

    bsv = BSV(Path(new_file_path))

    indices = np.arange(1, n_sentences + 1)
    final_embeddings, retrieved_indices = compute_database_embeddings(fake_sqlalchemy_engine,
                                                                      bsv,
                                                                      indices,
                                                                      batch_size=batch_size)

    assert final_embeddings.shape == (n_sentences, 700)
    assert np.all(indices == retrieved_indices)

    assert bsv_model.embed_sentences.call_count == (n_sentences // batch_size) + 1
