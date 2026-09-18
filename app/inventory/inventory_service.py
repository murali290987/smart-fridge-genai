"""
Persists detected ingredients to PostgreSQL as refrigerator inventory.

Plain SQL through psycopg2 -- no ORM. Names are normalized (descriptive
words stripped) before matching, and re-detecting an ingredient updates
its existing row (quantity, confidence, updated_at) instead of inserting
a duplicate.
"""
from __future__ import annotations

import psycopg2
import psycopg2.extras

from app.config import DATABASE_URL
from app.models.schemas import Ingredient, InventoryRecord

# Stripped before matching so e.g. "red apple" and "apple" merge into one
# inventory row. Deliberately narrow: color words are included because
# that's the case actually seen in testing (the model alternates between
# "apple" and "red apple" for the same fruit run to run), but this will
# also merge genuinely distinct items that happen to share a base word --
# e.g. "green onion" and "onion" become the same row, which loses a real
# distinction. Edit this set if a specific case matters to you.
_DESCRIPTIVE_WORDS = {
    "red", "green", "yellow", "purple", "white", "black", "brown",
    "ripe", "unripe", "overripe", "fresh", "raw", "whole",
    "large", "small", "medium", "big", "little",
    "organic", "baby", "young",
}


def normalize_ingredient_name(name: str) -> str:
    """Strip descriptive words for inventory matching/storage (see _DESCRIPTIVE_WORDS)."""
    words = [w for w in name.strip().split() if w.lower() not in _DESCRIPTIVE_WORDS]
    normalized = " ".join(words).strip()
    return normalized or name.strip()  # never reduce a name to nothing


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS inventory (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    estimated_quantity TEXT NOT NULL,
    unit TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    needs_confirmation BOOLEAN NOT NULL DEFAULT FALSE,
    source TEXT NOT NULL DEFAULT 'vision',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_CREATE_UNIQUE_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS inventory_name_unique_idx ON inventory (LOWER(name));
"""


class InventoryDatabaseError(Exception):
    """Could not connect to or query the inventory database."""


def get_connection():
    try:
        return psycopg2.connect(DATABASE_URL)
    except psycopg2.OperationalError as exc:
        raise InventoryDatabaseError(
            f"Could not connect to the inventory database at {DATABASE_URL}.\n"
            "Start PostgreSQL first, e.g.:\n"
            "    scripts/start_db.sh"
        ) from exc


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(_CREATE_TABLE_SQL)
        cur.execute(_CREATE_UNIQUE_INDEX_SQL)
    conn.commit()


def save_ingredients(conn, ingredients: list[Ingredient], source: str = "vision") -> int:
    """
    Upsert each ingredient into inventory by case-insensitive name: an
    existing row is updated in place (quantity/confidence/updated_at), a
    new name is inserted. Returns the number of ingredients processed.
    """
    if not ingredients:
        return 0

    # Normalize first, then dedup: two entries that normalize to the same
    # name (e.g. "apple" and "red apple") would otherwise violate ON
    # CONFLICT's "cannot affect row a second time in one command" rule --
    # keep the last occurrence.
    deduped: dict[str, tuple[str, Ingredient]] = {}
    for ing in ingredients:
        normalized_name = normalize_ingredient_name(ing.name)
        deduped[normalized_name.lower()] = (normalized_name, ing)

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO inventory (name, estimated_quantity, unit, confidence, needs_confirmation, source)
            VALUES %s
            ON CONFLICT (LOWER(name)) DO UPDATE SET
                name = EXCLUDED.name,
                estimated_quantity = EXCLUDED.estimated_quantity,
                unit = EXCLUDED.unit,
                confidence = EXCLUDED.confidence,
                needs_confirmation = EXCLUDED.needs_confirmation,
                source = EXCLUDED.source,
                updated_at = now()
            """,
            [
                (normalized_name, ing.estimated_quantity, ing.unit, ing.confidence, ing.needs_confirmation, source)
                for normalized_name, ing in deduped.values()
            ],
        )
    conn.commit()
    return len(deduped)


def fetch_all(conn) -> list[InventoryRecord]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM inventory ORDER BY created_at ASC, id ASC;")
        rows = cur.fetchall()
    return [InventoryRecord.model_validate(dict(row)) for row in rows]


def fetch_distinct_names(conn) -> list[str]:
    """Distinct ingredient names currently in inventory, for building a recipe query."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT name FROM inventory ORDER BY name;")
        return [row[0] for row in cur.fetchall()]


def clear_all(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM inventory;")
        deleted = cur.rowcount
    conn.commit()
    return deleted
