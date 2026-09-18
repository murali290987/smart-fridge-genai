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


def find_similar(conn, query_embedding: list[float], limit: int = 5):
    """Nearest recipes to query_embedding by cosine distance (smaller = closer).

    The explicit ::vector casts matter: without a known target column to
    infer the type from, Postgres can't resolve `<=>` against a bare
    parameter and raises "operator does not exist: vector <=> numeric[]".
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, name, instructions, embedding <=> %s::vector AS distance
            FROM recipes
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, limit),
        )
        return cur.fetchall()
