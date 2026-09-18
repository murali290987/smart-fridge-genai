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


def build_query_text(ingredient_names: list[str], meal_period: str) -> str:
    return f"{' '.join(ingredient_names)} {meal_period}".strip()


def recommend_recipes(conn, limit: int = 5, *, vegetarian: bool | None = None,
                       cuisine: str | None = None,
                       max_cook_time_minutes: int | None = None) -> list[RecipeRecommendation]:
    """
    Returns up to `limit` recipes whose embedding is closest to a query
    built from current inventory + the current meal period.

    The optional filters (vegetarian/cuisine/max_cook_time_minutes) only
    affect rows that have that metadata -- most recipes from the first
    dataset don't (see README), so they're excluded by vegetarian= or
    max_cook_time_minutes= filters but not by leaving them unset.

    Returns an empty list if there's no inventory yet, the recipe index
    hasn't been built, or the filters excluded every match -- callers
    should check for that and message the user accordingly rather than
    treating it as an error.
    """
    if count(conn) == 0:
        return []

    ingredient_names = fetch_distinct_names(conn)
    if not ingredient_names:
        return []

    meal_period = detect_meal_period().meal_period.value
    query_text = build_query_text(ingredient_names, meal_period)
    query_embedding = embed_text(query_text)

    rows = find_similar(
        conn, query_embedding, limit=limit,
        vegetarian=vegetarian, cuisine=cuisine, max_cook_time_minutes=max_cook_time_minutes,
    )
    return [
        RecipeRecommendation(
            id=row["id"],
            name=row["name"],
            distance=row["distance"],
            instructions=row["instructions"],
        )
        for row in rows
    ]
