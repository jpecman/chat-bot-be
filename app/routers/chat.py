from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.dependencies import get_chat_service
from app.schemas import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(
    request: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    answer = await service.ask(request.session_id, request.message)
    return ChatResponse(answer=answer)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_session(
    session_id: str,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> None:
    await service.clear(session_id)
