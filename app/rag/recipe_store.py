"""
Stores recipes + their embeddings in PostgreSQL/pgvector.

Plain SQL through psycopg2 (with pgvector's psycopg2 adapter registered so
Python lists convert to the `vector` column type) -- no ORM.
"""
from __future__ import annotations

import json

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


def insert_recipe(conn, *, external_id: str, name: str, image_name: str,
                   ingredients: list[str], instructions: str, embedding: list[float]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO recipes (external_id, name, image_name, ingredients, instructions, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (external_id, name, image_name, json.dumps(ingredients), instructions, embedding),
        )
