import logging

logger = logging.getLogger(__name__)

from sentence_transformers import SentenceTransformer
from bluesearch.embedding_models import SentTransformer

def embed_locally(model_name, client):


    model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

    model.add_module