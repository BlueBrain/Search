"""Build Semantic Text Similarity (STS) datasets to evaluate sentence embedding models."""

from collections import namedtuple
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch
from sqlalchemy.engine import Engine, RowProxy
from torch import Tensor

from bbsearch.embedding_models import BSV, EmbeddingModel
from bbsearch.utils import H5

Sampling = namedtuple('Sampling', 'index')
Pairing = namedtuple('Pairing', 'index, similarity')
Sentence = namedtuple('Sentence', 'id, text')
Pair = namedtuple('Pair', 'left, right, similarity, target')


# Sampling strategies

# Proposed:
#   1. random
#   2. annotated
#   3. k-means
#   4. LDA
# with automatic cherry picking (length, ...).

def sampling_random(n: int, **kwargs) -> List[Sampling]:
    # limit: int, seed: int
    rng = np.random.default_rng(kwargs['seed'])
    sampled = rng.integers(1, kwargs['limit'], size=n, endpoint=True)
    return list(map(Sampling, sampled))


# Pairing strategies

# Proposed:
#   1. random
#   2. quartiles
#   3. power law
# with automatic cherry picking (length, ...).

def pairing_powerlaw(similarities: Tensor, groups: int, target: int, **kwargs) -> Pairing:
    # step: int, power: int
    values, indexes = similarities.sort(descending=True)
    rank = ((groups - target) * kwargs['step']) ** kwargs['power']
    value = values[1:][rank]
    index = indexes[1:][rank]
    return Pairing(index.item(), value.item())


def pair_sentences(n: int, groups: int, sampling: Callable, sparams: dict, pairing: Callable,
                   pparams: dict, model: EmbeddingModel, embeddings: Tensor, engine) -> List[Pair]:
    targets = list(range(groups)) * (n // groups)
    samples = sampling(n, **sparams)
    pairs = []
    for sampled, target in zip(samples, targets):
        left = retrieve_sentence(sampled.index + 1, engine)
        similarities = compute_similarities(left, model, embeddings)
        paired = pairing(similarities, groups, target, **pparams)
        right = retrieve_sentence(paired.index + 1, engine)
        pair = Pair(left, right, paired.similarity, target)
        pairs.append(pair)
    return pairs


def compute_similarities(sentence: Sentence, model: EmbeddingModel, embeddings: Tensor) -> Tensor:
    preprocessed = model.preprocess(sentence.text)
    embedding = model.embed(preprocessed)
    tensor = torch.from_numpy(embedding).to(dtype=torch.float32)
    norm = torch.norm(tensor).item()
    norm = 1 if norm == 0 else norm
    tensor /= norm
    return torch.nn.functional.linear(tensor, embeddings)


def load_model_bsv() -> BSV:
    name = 'BioSentVec_PubMed_MIMICIII-bigram_d700.bin'
    path = Path(f'/raid/sync/proj115/bbs_data/trained_models/{name}')
    return BSV(path)


def load_embeddings_bsv(version: str) -> torch.Tensor:
    path = Path(f'/raid/sync/proj115/bbs_data/{version}/embeddings/embeddings_bsv_full.h5')
    name = 'BSV'
    embeddings = H5.load(path, name)[1:]
    tensor = torch.from_numpy(embeddings)
    norm = torch.norm(tensor, dim=1, keepdim=True)
    norm[norm == 0] = 1
    tensor /= norm
    return tensor


def retrieve_sentence(key: int, engine: Engine) -> Sentence:
    statement = f"SELECT text from sentences WHERE sentence_id = {key}"
    row = execute_sql(statement, engine)
    return Sentence(key, row['text'])


def sentences_count(engine: Engine) -> int:
    statement = "SELECT MAX(sentence_id) AS max from sentences"
    row = execute_sql(statement, engine)
    return row['max']


def execute_sql(statement: str, engine: Engine) -> RowProxy:
    result = engine.execute(statement)
    return result.fetchone()
