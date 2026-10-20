# chat-bot-be

> A learning project.

FastAPI backend for a RAG chatbot over books. Answers questions using book chunks retrieved from a vector store and an LLM.

## Stack

- FastAPI (async) + uvicorn
- LangChain with pluggable providers. Currently Azure OpenAI (`gpt-4.1-mini` + embeddings) and Qdrant.
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
| `POST` | `/chat` | Ask a question (work in progress) |
