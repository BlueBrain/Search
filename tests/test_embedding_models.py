from pathlib import Path
import pytest
from unittest.mock import Mock, MagicMock

import numpy as np
import torch

from bbsearch.embedding_models import EmbeddingModel, SBioBERT, BSV


class TestEmbeddingModels:

    def test_abstractclass(self):
        with pytest.raises(TypeError):
            EmbeddingModel()

    def test_sbiobert_embedding(self, monkeypatch):

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
        embedding = sbiobert.encode(preprocess_sentence)

        assert isinstance(embedding, np.ndarray)
        torch_model.assert_called_once()
        auto_tokenizer.from_pretrained().tokenize.assert_called_once()
        auto_tokenizer.from_pretrained().convert_tokens_to_ids.assert_called_once()

    def test_bsv_embedding(self, monkeypatch):

        sent2vec = Mock()
        bsv_model = Mock(spec=sent2vec.Sent2vecModel)
        bsv_model.return_value = np.ones([1, 512])
        sent2vec.Sent2vecModel().load_model.return_value = bsv_model

        monkeypatch.setattr('bbsearch.embedding_models.sent2vec', sent2vec)
        bsv = BSV(assets_path=Path(''))
        dummy_sentence = 'This is a dummy sentence/test.'
        preprocess_truth = 'dummy sentence test'

        preprocess_sentence = bsv.preprocess(dummy_sentence)
        assert isinstance(preprocess_sentence, str)
        assert preprocess_sentence == preprocess_truth

        embedding = bsv.encode(preprocess_sentence)
        assert isinstance(embedding, np.ndarray)
        bsv_model.assert_called_once()
