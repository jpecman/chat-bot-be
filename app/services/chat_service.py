from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from app.memory.base import ChatHistoryStore
from app.rag.chain import RagResult

MAX_HISTORY_MESSAGES = 10


class ChatService:
    def __init__(
        self, chain: Runnable[dict[str, Any], RagResult], history: ChatHistoryStore
    ) -> None:
        self._chain = chain
        self._history = history

    async def ask(self, session_id: str, message: str) -> str:
        history = await self._history.get(session_id)
        result = await self._chain.ainvoke(
            {"question": message, "history": history[-MAX_HISTORY_MESSAGES:]},
            config={"run_name": "chat", "metadata": {"langfuse_session_id": session_id}},
        )
        answer = result["answer"]

        await self._history.append(session_id, [HumanMessage(message), AIMessage(answer)])

        return answer

    async def clear(self, session_id: str) -> None:
        await self._history.clear(session_id)
