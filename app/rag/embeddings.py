"""
Generates text embeddings through the local Ollama HTTP API.

Second (and last, for this phase) GenAI call in the project -- everything
around it (CSV parsing, batching, Postgres/pgvector storage) is plain
Python.
"""
from __future__ import annotations

import requests

from app.config import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL
from app.vision.ollama_vision import OllamaConnectionError, OllamaTimeoutError

REQUEST_TIMEOUT_SECONDS = 60


def embed_text(text: str, model: str = OLLAMA_EMBEDDING_MODEL) -> list[float]:
    """Return the embedding vector for a single piece of text."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": model, "input": text},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ConnectionError as exc:
        raise OllamaConnectionError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL}. Is `ollama serve` running?"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaTimeoutError(
            f"Ollama did not respond within {REQUEST_TIMEOUT_SECONDS} seconds."
        ) from exc

    response.raise_for_status()
    return response.json()["embeddings"][0]
