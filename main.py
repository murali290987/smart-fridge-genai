#!/usr/bin/env python3
"""
Smart Fridge GenAI POC -- Phase 1

Pipeline:
    fridge.jpg -> Python -> Ollama -> Qwen3-VL 8B -> structured JSON -> console

Run:
    python main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from app.config import OLLAMA_BASE_URL, OLLAMA_VISION_MODEL
from app.vision.image_processor import ImageValidationError, validate_image
from app.vision.ollama_vision import (
    InvalidModelResponseError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    check_ollama_available,
    detect_ingredients,
)

IMAGE_PATH = Path(__file__).resolve().parent / "uploads" / "fridge.jpg"


def main() -> int:
    print(f"Smart Fridge GenAI POC -- model={OLLAMA_VISION_MODEL} via {OLLAMA_BASE_URL}")
    print(f"Image: {IMAGE_PATH}\n")

    try:
        image_path = validate_image(IMAGE_PATH)
    except ImageValidationError as exc:
        print(f"Image validation failed: {exc}", file=sys.stderr)
        return 1

    try:
        check_ollama_available()
    except (OllamaConnectionError, OllamaModelNotFoundError) as exc:
        print(f"Ollama check failed: {exc}", file=sys.stderr)
        return 1

    print("Sending image to Qwen3-VL... (can take a while on CPU, please wait)")
    try:
        result = detect_ingredients(image_path)
    except OllamaConnectionError as exc:
        print(f"Could not reach Ollama: {exc}", file=sys.stderr)
        return 1
    except OllamaTimeoutError as exc:
        print(f"Ollama timed out: {exc}", file=sys.stderr)
        return 1
    except OllamaModelNotFoundError as exc:
        print(f"Model not found: {exc}", file=sys.stderr)
        return 1
    except InvalidModelResponseError as exc:
        print(f"Model returned an invalid response: {exc}", file=sys.stderr)
        return 1

    print("\nDetected ingredients:\n")
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
