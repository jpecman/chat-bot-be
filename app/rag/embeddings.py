from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from app.config import OllamaSettings


def create_embeddings(settings: OllamaSettings) -> Embeddings:
    return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.base_url)
