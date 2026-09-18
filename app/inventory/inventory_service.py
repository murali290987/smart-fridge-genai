"""
Persists detected ingredients to PostgreSQL as refrigerator inventory.

Plain SQL through psycopg2 -- no ORM. Re-detecting an ingredient that's
already in inventory (case-insensitive name match) updates that row
(quantity, confidence, updated_at) instead of inserting a duplicate. This
does not merge near-duplicates like "apple" and "red apple" -- that's the
model's own naming inconsistency, not something a DB-level key can fix.
"""
from __future__ import annotations

import psycopg2
import psycopg2.extras

from app.config import DATABASE_URL
from app.models.schemas import Ingredient, InventoryRecord

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

    # Two entries with the same name (case-insensitive) in one response
    # would otherwise violate ON CONFLICT's "cannot affect row a second
    # time in one command" rule -- keep the last occurrence.
    deduped: dict[str, Ingredient] = {ing.name.strip().lower(): ing for ing in ingredients}

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
                (ing.name, ing.estimated_quantity, ing.unit, ing.confidence, ing.needs_confirmation, source)
                for ing in deduped.values()
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
