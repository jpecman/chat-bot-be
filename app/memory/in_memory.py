from langchain_core.messages import BaseMessage


class InMemoryChatHistoryStore:
    def __init__(self) -> None:
        self._sessions: dict[str, list[BaseMessage]] = {}

    async def get(self, session_id: str) -> list[BaseMessage]:
        return list(self._sessions.get(session_id, []))

    async def append(self, session_id: str, messages: list[BaseMessage]) -> None:
        self._sessions.setdefault(session_id, []).extend(messages)

    async def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
