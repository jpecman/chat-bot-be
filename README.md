# chat-bot-be

> A learning project.

FastAPI backend for a RAG chatbot over books. Answers questions using book chunks retrieved from a vector store and an LLM.

## Stack

- FastAPI (async) + uvicorn
- LangChain with pluggable providers. Currently Azure OpenAI (`gpt-4.1-mini`), Ollama embeddings (`qwen3-embedding:4b`) and Qdrant.
- Optional Langfuse tracing, enabled when `LANGFUSE__*` is set in `.env` (self-hosted stack in `../chat-bot-infra/langfuse`)
- Nix flake + uv for the Python environment

## Run

```sh
direnv allow                    # or: nix develop
cp .env.example .env            # fill in values
uvicorn app.main:app --reload   # http://localhost:8000/docs (add --port 8001 if 8000 is taken)
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Ask a question within a chat session |
| `DELETE` | `/sessions/{session_id}` | Clear a chat session's history |
