#!/usr/bin/env python3
"""
Loads data/recipes/IndianFoodDatasetCSV.csv, embeds each recipe with a
local Ollama embedding model, and stores it in the same `recipes` table
used by scripts/build_recipe_index.py.

Unlike that script, this one indexes the FULL dataset rather than a
random subset -- at ~1000 recipes per 17s (measured locally with
nomic-embed-text), ~6,871 recipes takes roughly 2 minutes, so there's no
real need to subset.

This dataset has real cuisine/course/diet/servings/cook-time metadata,
which the first dataset lacked (those columns exist on `recipes` but were
always NULL for it) -- see _infer_vegetarian() below for how "vegetarian"
is derived, and where it's deliberately left unknown (NULL) rather than
guessed.

Safe to re-run: skips entirely if rows from this dataset (external_id
prefixed "indian-") already exist.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT
from app.inventory.inventory_service import get_connection
from app.rag.embeddings import embed_text
from app.rag.recipe_store import count_with_external_id_prefix, ensure_schema, insert_recipe

CSV_PATH = PROJECT_ROOT / "data" / "recipes" / "IndianFoodDatasetCSV.csv"
EXTERNAL_ID_PREFIX = "indian-"
COMMIT_EVERY = 25

# Only mapped when the Diet label unambiguously implies it. Labels like
# "Diabetic Friendly", "Gluten Free", or "Sugar Free Diet" say nothing
# about meat content, so those are left as NULL (unknown) rather than
# guessed -- see README's Known limitations.
_VEGETARIAN_DIETS = {
    "vegetarian", "high protein vegetarian", "eggetarian", "vegan",
    "no onion no garlic (sattvic)",
}
_NON_VEGETARIAN_DIETS = {"non vegeterian", "high protein non vegetarian"}


def _infer_vegetarian(diet: str) -> bool | None:
    key = diet.strip().lower()
    if key in _VEGETARIAN_DIETS:
        return True
    if key in _NON_VEGETARIAN_DIETS:
        return False
    return None


def _parse_int(value: str) -> int | None:
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        return None


def _parse_ingredients(raw: str) -> list[str]:
    # This dataset stores ingredients as a plain comma-separated string,
    # not a Python-list literal like the first dataset.
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_candidate_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [
            row for row in reader
            if row.get("TranslatedRecipeName", "").strip()
            and row.get("TranslatedInstructions", "").strip()
            and row.get("TranslatedIngredients", "").strip()
        ]


def main() -> int:
    conn = get_connection()
    try:
        ensure_schema(conn)

        existing = count_with_external_id_prefix(conn, EXTERNAL_ID_PREFIX)
        if existing > 0:
            print(f"{existing} row(s) from this dataset already indexed -- skipping. "
                  f"Delete rows with external_id LIKE 'indian-%%' first to rebuild.")
            return 0

        print(f"Reading {CSV_PATH} ...")
        rows = load_candidate_rows(CSV_PATH)
        print(f"{len(rows)} recipes have a name, ingredients, and instructions.")

        t0 = time.time()
        for i, row in enumerate(rows, start=1):
            ingredients = _parse_ingredients(row["TranslatedIngredients"])
            name = row["TranslatedRecipeName"].strip()
            searchable_text = f"{name}. Ingredients: {', '.join(ingredients)}"

            embedding = embed_text(searchable_text)
            insert_recipe(
                conn,
                external_id=f"{EXTERNAL_ID_PREFIX}{row['Srno']}",
                name=name,
                image_name=None,
                ingredients=ingredients,
                instructions=row["TranslatedInstructions"].strip(),
                embedding=embedding,
                cuisine=row.get("Cuisine", "").strip() or None,
                meal_type=row.get("Course", "").strip() or None,
                servings=_parse_int(row.get("Servings", "")),
                cooking_time_minutes=_parse_int(row.get("CookTimeInMins", "")),
                vegetarian=_infer_vegetarian(row.get("Diet", "")),
            )

            if i % COMMIT_EVERY == 0 or i == len(rows):
                conn.commit()
                elapsed = time.time() - t0
                print(f"  {i}/{len(rows)} indexed ({elapsed:.1f}s elapsed)")

        print(f"Done. {count_with_external_id_prefix(conn, EXTERNAL_ID_PREFIX)} Indian recipes now indexed.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
