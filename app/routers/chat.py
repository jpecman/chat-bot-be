from fastapi import APIRouter

from app.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])

@router.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    return ChatResponse(answer=f"You said: {request.message}")