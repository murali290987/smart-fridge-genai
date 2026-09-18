"""
Central configuration for the Smart Fridge POC.

This intentionally has no dependency on python-dotenv (not in requirements.txt).
It's a few lines of plain Python that fill in os.environ from a .env file if one
exists, without overriding variables the shell/OS has already set.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(PROJECT_ROOT / ".env")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "qwen3-vl:8b")
VISION_CONFIDENCE_THRESHOLD = float(os.environ.get("VISION_CONFIDENCE_THRESHOLD", "0.60"))
MAX_IMAGE_SIZE_MB = float(os.environ.get("MAX_IMAGE_SIZE_MB", "10"))
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres@localhost:5432/smart_fridge")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "768"))


def _optional_bool(name: str) -> bool | None:
    value = os.environ.get(name, "").strip()
    return value.lower() in ("1", "true", "yes") if value else None


def _optional_int(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


# All unset (empty/absent) by default -- no filtering, same behavior as before.
RECIPE_VEGETARIAN_ONLY = _optional_bool("RECIPE_VEGETARIAN_ONLY")
RECIPE_CUISINE_FILTER = os.environ.get("RECIPE_CUISINE_FILTER", "").strip() or None
RECIPE_MAX_COOK_TIME_MINUTES = _optional_int("RECIPE_MAX_COOK_TIME_MINUTES")
