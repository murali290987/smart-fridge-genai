"""
Stores recipes + their embeddings in PostgreSQL/pgvector.

Plain SQL through psycopg2 (with pgvector's psycopg2 adapter registered so
Python lists convert to the `vector` column type) -- no ORM.
"""
from __future__ import annotations

import json

import psycopg2.extras
from pgvector.psycopg2 import register_vector

from app.config import EMBEDDING_DIMENSIONS

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS recipes (
    id SERIAL PRIMARY KEY,
    external_id TEXT,
    name TEXT NOT NULL,
    image_name TEXT,
    ingredients JSONB NOT NULL,
    instructions TEXT NOT NULL,
    cuisine TEXT,
    meal_type TEXT,
    servings INTEGER,
    cooking_time_minutes INTEGER,
    vegetarian BOOLEAN,
    embedding VECTOR({EMBEDDING_DIMENSIONS}) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS recipes_embedding_idx
    ON recipes USING hnsw (embedding vector_cosine_ops);
"""


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(_CREATE_TABLE_SQL)
        cur.execute(_CREATE_INDEX_SQL)
    conn.commit()
    register_vector(conn)


def count(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM recipes;")
        return cur.fetchone()[0]


def count_with_external_id_prefix(conn, prefix: str) -> int:
    """Used by indexing scripts to check 'has this dataset already been loaded?'"""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM recipes WHERE external_id LIKE %s;", (f"{prefix}%",))
        return cur.fetchone()[0]


def insert_recipe(conn, *, external_id: str, name: str, image_name: str | None,
                   ingredients: list[str], instructions: str, embedding: list[float],
                   cuisine: str | None = None, meal_type: str | None = None,
                   servings: int | None = None, cooking_time_minutes: int | None = None,
                   vegetarian: bool | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO recipes (
                external_id, name, image_name, ingredients, instructions, embedding,
                cuisine, meal_type, servings, cooking_time_minutes, vegetarian
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (external_id, name, image_name, json.dumps(ingredients), instructions, embedding,
             cuisine, meal_type, servings, cooking_time_minutes, vegetarian),
        )


def find_similar(conn, query_embedding: list[float], limit: int = 5, *,
                  vegetarian: bool | None = None, cuisine: str | None = None,
                  max_cook_time_minutes: int | None = None):
    """Nearest recipes to query_embedding by cosine distance (smaller = closer).

    The explicit ::vector casts matter: without a known target column to
    infer the type from, Postgres can't resolve `<=>` against a bare
    parameter and raises "operator does not exist: vector <=> numeric[]".

    Optional filters only apply to rows that actually have that metadata
    (most rows from the first recipe dataset don't -- see README) and are
    combined with AND. A recipe with a NULL cooking_time_minutes is never
    excluded by max_cook_time_minutes, since NULL means "unknown", not
    "zero minutes".
    """
    where_clauses = []
    params: dict = {"qvec": query_embedding, "limit": limit}

    if vegetarian is not None:
        where_clauses.append("vegetarian = %(vegetarian)s")
        params["vegetarian"] = vegetarian
    if cuisine:
        where_clauses.append("cuisine ILIKE %(cuisine)s")
        params["cuisine"] = f"%{cuisine}%"
    if max_cook_time_minutes is not None:
        where_clauses.append("(cooking_time_minutes IS NULL OR cooking_time_minutes <= %(max_cook_time)s)")
        params["max_cook_time"] = max_cook_time_minutes

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if where_clauses:
            # The HNSW index does an approximate search and only explores a
            # small candidate window before Postgres applies the WHERE
            # filter -- if none of that window matches the filter, the
            # query can return zero rows even when many matches exist
            # elsewhere in the table (confirmed via EXPLAIN: "Filter: (NOT
            # vegetarian)" applied after the index scan). At this table's
            # size (a few thousand rows), forcing an exact scan for
            # filtered queries is cheap and guarantees correct results.
            # SET LOCAL only affects the current transaction.
            cur.execute("SET LOCAL enable_indexscan = off;")
        cur.execute(
            f"""
            SELECT id, name, instructions, embedding <=> %(qvec)s::vector AS distance
            FROM recipes
            {where_sql}
            ORDER BY embedding <=> %(qvec)s::vector
            LIMIT %(limit)s
            """,
            params,
        )
        return cur.fetchall()
