from functools import lru_cache

from app.config import get_settings
from app.memory.in_memory import InMemoryChatHistoryStore
from app.rag.chain import create_rag_chain
from app.rag.embeddings import create_embeddings
from app.rag.llm import create_chat_model
from app.rag.retriever import create_retriever
from app.services.chat_service import ChatService


@lru_cache
def get_chat_service() -> ChatService:
    settings = get_settings()
    embeddings = create_embeddings(settings.ollama)
    retriever = create_retriever(settings.qdrant, embeddings, settings.retriever_k)
    llm = create_chat_model(settings.azure_openai)
    return ChatService(create_rag_chain(retriever, llm), InMemoryChatHistoryStore())
