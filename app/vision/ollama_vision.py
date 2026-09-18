"""
Calls Qwen3-VL 8B through the local Ollama HTTP API.

This is the only GenAI part of the pipeline: everything else in this
project (image validation, JSON parsing, schema validation) is plain
deterministic Python around this one model call.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import requests
from pydantic import ValidationError

from app.config import OLLAMA_BASE_URL, OLLAMA_VISION_MODEL
from app.models.schemas import IngredientDetectionResponse

REQUEST_TIMEOUT_SECONDS = 180
HEALTH_CHECK_TIMEOUT_SECONDS = 5
# The image itself can consume 3000+ tokens of the prompt. Ollama's default
# num_ctx (4096) leaves too little room for the model to finish the JSON on
# a fully stocked fridge, and its default num_predict cap (~512) truncates
# mid-object even sooner. Raise both explicitly.
CONTEXT_WINDOW_TOKENS = 8192
MAX_RESPONSE_TOKENS = 2048
# Tested raising this to 0.3 to see if it reduced filler-item hallucination
# on cluttered images (e.g. "green juice", "green sauce", "green powder") --
# it didn't fix the pattern, just changed its wording, and cost run-to-run
# consistency. Low confidence scores + the needs_confirmation threshold
# are what actually catch these, not temperature.
TEMPERATURE = 0.1

VISION_PROMPT = """You are a food inventory detection system.

Analyze the provided refrigerator or food image.

Your task is to identify ONLY the food ingredients that are visibly present in the image.

For each identified ingredient, return:

- name
- estimated_quantity
- unit
- confidence
- needs_confirmation

Rules:

1. Identify only ingredients that are visibly present in the image.
2. Do not invent, assume, or infer ingredients that cannot be visually identified.
3. Use simple, common food ingredient names such as:
   potato, tomato, onion, carrot, cucumber, capsicum, chicken, egg, milk, bread, spinach, etc.
4. Count clearly visible individual items when practical.
5. When counting items of the same type, look carefully for ones that overlap, touch, or rest against
   each other (e.g. two oranges leaning together, apples stacked in a bowl). Count each distinct item
   separately even when partially hidden behind another item of the same type -- do not collapse two
   overlapping items into a count of one.
6. If the quantity cannot reasonably be determined from the image, use:
   "unknown"
7. Quantity estimates are approximate and must NOT be treated as exact.
8. Do not estimate exact weight in grams or kilograms unless the weight is explicitly visible in the image.
9. Do not determine freshness, expiry date, quality, or food safety from the image.
10. If an ingredient is partially visible but can reasonably be identified, include it with an appropriate confidence score.
11. If an ingredient is too unclear to identify reliably, do not include it.
12. If the same ingredient appears multiple times, combine them into one entry when practical.
13. Confidence represents how confident you are that the identified ingredient is actually present.
14. Confidence must be a number between 0.0 and 1.0.
15. Set "needs_confirmation" to true when confidence is below 0.60.
16. Set "needs_confirmation" to false when confidence is 0.60 or higher.
17. Do not use the example values below as actual image results. Analyze the provided image independently.
18. Return valid JSON only.
19. Do not include Markdown code fences.
20. Do not include explanations, comments, or additional text outside the JSON.

Return exactly this JSON structure:

{
  "ingredients": [
    {
      "name": "<ingredient name>",
      "estimated_quantity": "<quantity or unknown>",
      "unit": "<unit or unknown>",
      "confidence": 0.00,
      "needs_confirmation": false
    }
  ]
}

If no food ingredients can be reliably identified, return:

{
  "ingredients": []
}
"""


class OllamaConnectionError(Exception):
    """Ollama is not reachable at OLLAMA_BASE_URL."""


class OllamaModelNotFoundError(Exception):
    """The configured model has not been pulled into Ollama."""


class OllamaTimeoutError(Exception):
    """Ollama did not respond within REQUEST_TIMEOUT_SECONDS."""


class InvalidModelResponseError(Exception):
    """The model's output was not parseable/valid JSON matching our schema."""


def check_ollama_available(base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_VISION_MODEL) -> None:
    """Raise a clear, actionable error if Ollama is down or the model is missing."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=HEALTH_CHECK_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise OllamaConnectionError(
            f"Could not reach Ollama at {base_url}.\n"
            "Start it with `ollama serve` (or open the Ollama desktop app), then try again."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise OllamaConnectionError(f"Ollama health check at {base_url} failed: {exc}") from exc

    available_models = [m.get("name", "") for m in response.json().get("models", [])]
    model_base = model.split(":")[0]
    if not any(m == model or m.startswith(f"{model_base}:") for m in available_models):
        raise OllamaModelNotFoundError(
            f"Model '{model}' was not found in Ollama.\n"
            f"Pull it first with:\n    ollama pull {model}\n"
            f"Models currently available: {available_models or 'none'}"
        )


def _encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def detect_ingredients(image_path: Path) -> IngredientDetectionResponse:
    """
    Send the fridge image + vision prompt to Qwen3-VL via Ollama and return
    a validated IngredientDetectionResponse.

    Raises OllamaConnectionError, OllamaTimeoutError, OllamaModelNotFoundError,
    or InvalidModelResponseError on failure.
    """
    encoded_image = _encode_image_base64(image_path)

    payload = {
        "model": OLLAMA_VISION_MODEL,
        "prompt": VISION_PROMPT,
        "images": [encoded_image],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_RESPONSE_TOKENS,
            "num_ctx": CONTEXT_WINDOW_TOKENS,
        },
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
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

    if response.status_code == 404:
        raise OllamaModelNotFoundError(
            f"Model '{OLLAMA_VISION_MODEL}' was not found. Pull it with:\n"
            f"    ollama pull {OLLAMA_VISION_MODEL}"
        )
    if response.status_code != 200:
        raise OllamaConnectionError(f"Ollama returned HTTP {response.status_code}: {response.text[:300]}")

    try:
        body = response.json()
        raw_text = body["response"]
    except (KeyError, json.JSONDecodeError) as exc:
        raise InvalidModelResponseError(f"Unexpected response shape from Ollama: {exc}") from exc

    if not raw_text.strip():
        # qwen3-vl:8b in Ollama uses a "thinking" renderer/parser. Combined
        # with format="json" (grammar-constrained decoding), the model's
        # entire structured answer sometimes gets classified into the
        # "thinking" channel instead of "response", leaving "response"
        # empty even though a complete, valid JSON answer was produced.
        raw_text = body.get("thinking") or ""

    cleaned_text = _strip_markdown_fences(raw_text)

    try:
        parsed_json = json.loads(cleaned_text)
    except json.JSONDecodeError as exc:
        raise InvalidModelResponseError(
            f"Model did not return valid JSON: {exc}\n--- raw model output ---\n{cleaned_text[:500]}"
        ) from exc

    try:
        return IngredientDetectionResponse.model_validate(parsed_json)
    except ValidationError as exc:
        raise InvalidModelResponseError(f"Model JSON failed schema validation:\n{exc}") from exc
