# chat-bot-be

FastAPI backend for a RAG chatbot over books.

## Requirements

- [Nix](https://nixos.org/) with flakes enabled. The dev shell provides Python 3.12 and [uv](https://docs.astral.sh/uv/).

## Setup

```sh
git add flake.nix          # flakes only see files tracked by git
nix develop                # enters the shell, runs `uv sync`, activates .venv
cp .env.example .env       # then fill in values
```

### direnv (optional)

With [direnv](https://direnv.net/) and [nix-direnv](https://github.com/nix-community/nix-direnv) installed, the dev shell loads automatically when you `cd` into the project:

```sh
direnv allow
```

The shell is cached and rebuilt only when `flake.nix`, `flake.lock`, `pyproject.toml` or `uv.lock` change.

## Run

```sh
uvicorn app.main:app --reload
```

- API docs (Swagger): http://localhost:8000/docs
- Health check: `GET /health`

## Project layout

```
app/
  main.py          # app factory, CORS, router registration
  config.py        # settings from .env (pydantic-settings)
  routers/
    health.py      # GET /health
```

## Adding dependencies

```sh
uv add <package>
```
