from functools import lru_cache

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureOpenAISettings(BaseModel):
    endpoint: str
    api_key: SecretStr
    api_version: str
    chat_deployment: str
    embedding_deployment: str
    embedding_dimensions: int | None = None


class QdrantSettings(BaseModel):
    url: str
    api_key: SecretStr | None = None
    collection: str


class OllamaSettings(BaseModel):
    base_url: str = "http://localhost:11434"
    embedding_model: str


class Settings(BaseSettings):
    """Application settings loaded from environment variables / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "Book Chatbot API"
    cors_origins: list[str] = ["http://localhost:5173"]
    retriever_k: int = 5

    azure_openai: AzureOpenAISettings
    ollama: OllamaSettings
    qdrant: QdrantSettings


@lru_cache
def get_settings() -> Settings:
    return Settings()
