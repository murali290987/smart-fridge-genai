"""
Combines current inventory + meal period into a recipe search query.

This is the actual "Recipe RAG" step: retrieval only (embed a text query,
find nearest recipes by vector similarity) -- no LLM call, no generated
recommendation text. Ranking/explaining results with an LLM is a further
step, not implemented here.
"""
from __future__ import annotations

from app.inventory.inventory_service import fetch_distinct_names
from app.meal_time.detector import detect_meal_period
from app.models.schemas import RecipeRecommendation
from app.rag.embeddings import embed_text
from app.rag.recipe_store import count, find_similar

INSTRUCTIONS_PREVIEW_LENGTH = 400


def build_query_text(ingredient_names: list[str], meal_period: str) -> str:
    return f"{' '.join(ingredient_names)} {meal_period}".strip()


def recommend_recipes(conn, limit: int = 5) -> list[RecipeRecommendation]:
    """
    Returns up to `limit` recipes whose embedding is closest to a query
    built from current inventory + the current meal period.

    Returns an empty list if there's no inventory yet or the recipe index
    hasn't been built (callers should check for that and message the user
    accordingly rather than treating it as an error).
    """
    if count(conn) == 0:
        return []

    ingredient_names = fetch_distinct_names(conn)
    if not ingredient_names:
        return []

    meal_period = detect_meal_period().meal_period.value
    query_text = build_query_text(ingredient_names, meal_period)
    query_embedding = embed_text(query_text)

    rows = find_similar(conn, query_embedding, limit=limit)
    return [
        RecipeRecommendation(
            id=row["id"],
            name=row["name"],
            distance=row["distance"],
            instructions_preview=row["instructions"][:INSTRUCTIONS_PREVIEW_LENGTH].strip() + "...",
        )
        for row in rows
    ]
