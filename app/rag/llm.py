from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI

from app.config import AzureOpenAISettings


def create_chat_model(settings: AzureOpenAISettings) -> BaseChatModel:
    return AzureChatOpenAI(
        azure_endpoint=settings.endpoint,
        api_key=settings.api_key,
        api_version=settings.api_version,
        azure_deployment=settings.chat_deployment,
        temperature=0.2,
        streaming=True,
    )
