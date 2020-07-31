from pathlib import Path
import pytest
from unittest.mock import Mock, MagicMock

import numpy as np
import pandas as pd
import sent2vec
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import torch

from bbsearch.embedding_models import EmbeddingModel, SBioBERT, BSV, SBERT, USE, compute_database_embeddings


class TestEmbeddingModels:

    def test_abstractclass(self):
        with pytest.raises(TypeError):
            EmbeddingModel()

        class WrongModel(EmbeddingModel):

            def embed(a):
                pass

        with pytest.raises(TypeError):
            WrongModel()

    def test_sbiobert_embedding(self, monkeypatch, fake_sqlalchemy_engine, test_parameters, metadata_path):
        torch_model = MagicMock(spec=torch.nn.Module)
        torch_model.return_value = (torch.ones([1, 8, 768]), torch.ones([1, 768]))

        auto_model = Mock()
        auto_model.from_pretrained().bert.to.return_value = torch_model

        auto_tokenizer = Mock()

        monkeypatch.setattr('bbsearch.embedding_models.AutoTokenizer', auto_tokenizer)
        monkeypatch.setattr('bbsearch.embedding_models.AutoModelWithLMHead', auto_model)
        monkeypatch.setattr('bbsearch.embedding_models.torch', MagicMock())

        sbiobert = SBioBERT()
        dummy_sentence = 'This is a dummy sentence'
        preprocess_sentence = sbiobert.preprocess(dummy_sentence)
        embedding = sbiobert.embed(preprocess_sentence)

        assert isinstance(embedding, np.ndarray)
        torch_model.assert_called_once()
        auto_tokenizer.from_pretrained().tokenize.assert_called_once()
        auto_tokenizer.from_pretrained().convert_tokens_to_ids.assert_called_once()

        n_articles = pd.read_csv(metadata_path)['sha'].notna().sum()
        n_sentences = n_articles * test_parameters['n_sections_per_article'] * test_parameters[
            'n_sentences_per_section']

        indices = np.arange(n_sentences)
        final_embeddings, retrieved_indices = compute_database_embeddings(fake_sqlalchemy_engine,
                                                                          sbiobert,
                                                                          indices)

        assert final_embeddings.shape == (n_sentences, 768)
        assert np.all(indices == retrieved_indices)

    def test_bsv_embedding(self, monkeypatch, tmpdir, fake_sqlalchemy_engine, test_parameters, metadata_path):
        sent2vec_module = Mock()
        bsv_model = Mock(spec=sent2vec.Sent2vecModel)
        bsv_model.embed_sentences.return_value = np.ones([1, 700])
        sent2vec_module.Sent2vecModel.return_value = bsv_model

        monkeypatch.setattr('bbsearch.embedding_models.sent2vec', sent2vec_module)

        new_file_path = Path(str(tmpdir)) / 'test.txt'
        new_file_path.touch()
        with pytest.raises(FileNotFoundError):
            BSV(checkpoint_model_path=Path(''))
        bsv = BSV(Path(new_file_path))
        dummy_sentence = 'This is a dummy sentence/test.'
        preprocess_truth = 'dummy sentence test'

        preprocess_sentence = bsv.preprocess(dummy_sentence)
        assert isinstance(preprocess_sentence, str)
        assert preprocess_sentence == preprocess_truth

        embedding = bsv.embed(preprocess_sentence)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (700,)
        bsv_model.embed_sentences.assert_called_once()

        n_articles = pd.read_csv(metadata_path)['sha'].notna().sum()
        n_sentences = n_articles * test_parameters['n_sections_per_article'] * test_parameters[
            'n_sentences_per_section']

        indices = np.arange(n_sentences)
        final_embeddings, retrieved_indices = compute_database_embeddings(fake_sqlalchemy_engine,
                                                                          bsv,
                                                                          indices)

        assert final_embeddings.shape == (n_sentences, 700)
        assert np.all(indices == retrieved_indices)

    def test_sbert_embedding(self, monkeypatch, fake_sqlalchemy_engine, metadata_path, test_parameters):
        sentence_transormer_class = Mock()
        sbert_model = Mock(spec=SentenceTransformer)
        sbert_model.encode.return_value = np.ones([1, 768])  # Need to check the dimensions
        sentence_transormer_class.return_value = sbert_model

        monkeypatch.setattr('bbsearch.embedding_models.SentenceTransformer', sentence_transormer_class)

        dummy_sentence = 'This is a dummy sentence/test.'
        sbert = SBERT()

        preprocessed_sentence = sbert.preprocess(dummy_sentence)
        assert preprocessed_sentence == dummy_sentence

        embedding = sbert.embed(preprocessed_sentence)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
        sbert_model.encode.assert_called_once()

        n_articles = pd.read_csv(metadata_path)['sha'].notna().sum()
        n_sentences = n_articles * test_parameters['n_sections_per_article'] * test_parameters[
            'n_sentences_per_section']

        indices = np.arange(n_sentences)
        final_embeddings, retrieved_indices = compute_database_embeddings(fake_sqlalchemy_engine,
                                                                          sbert,
                                                                          indices)

        assert final_embeddings.shape == (n_sentences, 768)
        assert np.all(indices == retrieved_indices)

    def test_use_embedding(self, monkeypatch, fake_sqlalchemy_engine, metadata_path, test_parameters):
        hub_module = Mock()
        use_model = Mock()
        hub_module.load.return_value = use_model
        use_model.return_value = tf.ones((1, 512))

        monkeypatch.setattr('bbsearch.embedding_models.hub', hub_module)

        dummy_sentence = 'This is a dummy sentence/test.'
        use = USE()

        preprocessed_sentence = use.preprocess(dummy_sentence)
        assert preprocessed_sentence == dummy_sentence

        embedding = use.embed(preprocessed_sentence)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (512,)
        use_model.assert_called_once()

        n_articles = pd.read_csv(metadata_path)['sha'].notna().sum()
        n_sentences = n_articles * test_parameters['n_sections_per_article'] * test_parameters[
            'n_sentences_per_section']

        indices = np.arange(n_sentences)
        final_embeddings, retrieved_indices = compute_database_embeddings(fake_sqlalchemy_engine,
                                                                          use,
                                                                          indices)

        assert final_embeddings.shape == (n_sentences, 512)
        assert np.all(indices == retrieved_indices)
