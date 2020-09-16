"""Build Semantic Text Similarity (STS) datasets to evaluate sentence embedding models."""

from collections import namedtuple
from pathlib import Path
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as nnf
from bbsearch.embedding_models import BSV, EmbeddingModel
from bbsearch.utils import H5

Sentence = namedtuple('Sentence', 'id, text')
Pair = namedtuple('Pair', 'left, right, similarity, target')


def pair_sentences(model: EmbeddingModel, embeddings: torch.Tensor, engine, sampling: Callable,
                   pairing: Callable, n: int, groups: int, **kwargs) -> List[Pair]:
    targets = list(range(groups)) * (n // groups)
    sampled = sampling(n)
    pairs = []
    for left, target in zip(sampled, targets):
        similarities = compute_similarities(left, model, embeddings)
        index, similarity = pairing(similarities, groups, target, **kwargs)
        key = index + 1
        right = retrieve_sentence(key, engine)
        pair = Pair(left, right, similarity, target)
        pairs.append(pair)
    return pairs


# Proposed strategies:
#   1. random
#   2. annotated
#   3. k-means
#   4. LDA
# with automatic cherry picking (length, ...).
def sampling_mock(n: int) -> List[Sentence]:
    text = "Adiponectin inhibits hepatic glucose production and enhances glucose uptake in muscle."
    return [Sentence(i, text) for i in range(1, n + 1)]


# Proposed strategies:
#   1. random
#   2. quartiles
#   3. power law
# with automatic cherry picking (length, ...).
def pairing_powerlaw(similarities: torch.Tensor, groups: int, target: int, **kwargs
                     ) -> Tuple[int, int]:
    values, indexes = similarities.sort(descending=True)
    rank = ((groups - target) * kwargs['step']) ** kwargs['power']
    value = values[1:][rank]
    index = indexes[1:][rank]
    return index.item(), value.item()


def compute_similarities(sentence: Sentence, model: EmbeddingModel, embeddings: torch.Tensor
                         ) -> torch.Tensor:
    preprocessed = model.preprocess(sentence.text)
    embedding = model.embed(preprocessed)
    tensor = torch.from_numpy(embedding).to(dtype=torch.float32)
    norm = torch.norm(tensor).item()
    norm = 1 if norm == 0 else norm
    tensor /= norm
    return nnf.linear(tensor, embeddings)


def load_model_bsv() -> BSV:
    name = 'BioSentVec_PubMed_MIMICIII-bigram_d700.bin'
    path = Path(f'/raid/sync/proj115/bbs_data/trained_models/{name}')
    return BSV(path)


def load_embeddings_bsv(source: str) -> torch.Tensor:
    path = Path(f'/raid/sync/proj115/bbs_data/{source}/embeddings/embeddings_bsv_full.h5')
    embeddings = H5.load(path, 'BSV')[1:]
    tensor = torch.from_numpy(embeddings)
    norm = torch.norm(tensor, dim=1, keepdim=True)
    norm[norm == 0] = 1
    tensor /= norm
    return tensor


def retrieve_sentence(key: int, engine) -> Sentence:
    statement = f"SELECT text from sentences WHERE sentence_id = {key}"
    result = engine.execute(statement)
    row = result.fetchone()
    return Sentence(key, row['text'])
