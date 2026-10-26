from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from app.config import OllamaSettings


class InstructedEmbeddings(Embeddings):
    """Adds a task instruction to queries; documents are embedded unchanged.

    Instruction-aware models such as Qwen3-Embedding expect it on the query side only.
    """

    def __init__(self, inner: Embeddings, instruction: str) -> None:
        self._inner = inner
        self._instruction = instruction

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(f"Instruct: {self._instruction}\nQuery:{text}")


def create_embeddings(settings: OllamaSettings) -> Embeddings:
    embeddings = OllamaEmbeddings(model=settings.embedding_model, base_url=settings.base_url)
    if settings.query_instruction:
        return InstructedEmbeddings(embeddings, settings.query_instruction)
    return embeddings
