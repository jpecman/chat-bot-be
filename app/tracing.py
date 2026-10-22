from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from app.config import LangfuseSettings


def create_langfuse_handler(settings: LangfuseSettings) -> CallbackHandler:
    # Registers the client in the SDK's per-public-key registry; the handler looks it up by key.
    Langfuse(
        public_key=settings.public_key,
        secret_key=settings.secret_key.get_secret_value(),
        base_url=settings.base_url,
    )
    return CallbackHandler(public_key=settings.public_key)
