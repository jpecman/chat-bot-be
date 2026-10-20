from langchain_core.runnables import Runnable


class ChatService:
    def __init__(self, chain: Runnable[str, str]) -> None:
        self._chain = chain

    async def ask(self, session_id: str, message: str) -> str:
        return await self._chain.ainvoke(message)
