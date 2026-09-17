#!/usr/bin/env python3
"""
Loads a subset of data/recipes/13k-recipes.csv, embeds each recipe with a
local Ollama embedding model, and stores it in the `recipes` table
(pgvector). Source: https://github.com/josephrmartinez/recipe-dataset

Safe to re-run: if the table is already populated, it exits without
re-embedding (delete rows first, e.g. via a fresh `TRUNCATE recipes;`, to
rebuild). Indexing all 13,501 recipes would take much longer locally, so
this takes a reproducible random subset instead.
"""
from __future__ import annotations

import ast
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT
from app.inventory.inventory_service import get_connection
from app.rag.embeddings import embed_text
from app.rag.recipe_store import count, ensure_schema, insert_recipe

CSV_PATH = PROJECT_ROOT / "data" / "recipes" / "13k-recipes.csv"
SUBSET_SIZE = 1000
RANDOM_SEED = 42
COMMIT_EVERY = 25


def _parse_ingredients(raw: str) -> list[str]:
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (ValueError, SyntaxError):
        pass
    return [raw]


def load_candidate_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            row for row in reader
            if row.get("Title", "").strip() and row.get("Instructions", "").strip()
        ]


def main() -> int:
    conn = get_connection()
    try:
        ensure_schema(conn)

        existing = count(conn)
        if existing > 0:
            print(f"recipes table already has {existing} row(s) -- skipping. "
                  f"Truncate the table first if you want to rebuild the index.")
            return 0

        print(f"Reading {CSV_PATH} ...")
        candidates = load_candidate_rows(CSV_PATH)
        print(f"{len(candidates)} recipes have both a title and instructions.")

        random.seed(RANDOM_SEED)
        subset = random.sample(candidates, min(SUBSET_SIZE, len(candidates)))
        print(f"Embedding a random subset of {len(subset)} recipes (seed={RANDOM_SEED})...")

        t0 = time.time()
        for i, row in enumerate(subset, start=1):
            ingredients = _parse_ingredients(row.get("Cleaned_Ingredients") or row.get("Ingredients", ""))
            searchable_text = f"{row['Title']}. Ingredients: {', '.join(ingredients)}"

            embedding = embed_text(searchable_text)
            insert_recipe(
                conn,
                external_id=row.get("", ""),
                name=row["Title"],
                image_name=row.get("Image_Name", ""),
                ingredients=ingredients,
                instructions=row["Instructions"],
                embedding=embedding,
            )

            if i % COMMIT_EVERY == 0 or i == len(subset):
                conn.commit()
                elapsed = time.time() - t0
                print(f"  {i}/{len(subset)} indexed ({elapsed:.1f}s elapsed)")

        print(f"Done. {count(conn)} recipes now indexed in pgvector.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
