#!/usr/bin/env python3
"""
Smart Fridge GenAI POC -- Phase 2

Pipeline:
    fridge.jpg -> Python -> Ollama -> Qwen3-VL 8B -> structured JSON
        -> Pydantic validation -> PostgreSQL inventory -> console

Also reports the current meal period (breakfast/lunch/snacks/dinner/other)
from local system time -- deterministic Python, not an LLM call.

Run:
    python main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from app.config import OLLAMA_BASE_URL, OLLAMA_VISION_MODEL
from app.inventory.inventory_service import (
    InventoryDatabaseError,
    ensure_schema,
    fetch_all,
    get_connection,
    save_ingredients,
)
from app.meal_time.detector import detect_meal_period
from app.rag.recipe_retriever import recommend_recipes
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

    try:
        conn = get_connection()
    except InventoryDatabaseError as exc:
        print(f"\nInventory not saved -- {exc}", file=sys.stderr)
        return 1

    try:
        ensure_schema(conn)
        saved = save_ingredients(conn, result.ingredients)
        print(f"\nSaved {saved} ingredient(s) to inventory (source=vision).")

        inventory = fetch_all(conn)
        print(f"\nFull inventory ({len(inventory)} row(s), persisted across runs):\n")
        for row in inventory:
            flag = " [NEEDS CONFIRMATION]" if row.needs_confirmation else ""
            print(f"  #{row.id} {row.name}: {row.estimated_quantity} {row.unit} "
                  f"(conf={row.confidence}, source={row.source}){flag}")

        meal_info = detect_meal_period()
        print(f"\nCurrent meal period: {meal_info.model_dump_json()}")

        recommendations = recommend_recipes(conn)
        if recommendations:
            print(f"\nRecipe suggestions for {meal_info.meal_period.value} "
                  f"based on current inventory (nearest by embedding distance):\n")
            for rank, rec in enumerate(recommendations, start=1):
                print(f"  {rank}. {rec.name} (distance={rec.distance:.4f})")
                for line in rec.instructions.strip().splitlines():
                    print(f"     {line}")
                print()
        else:
            print("\nNo recipe suggestions -- run scripts/build_recipe_index.py first.")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
