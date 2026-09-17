"""
Persists detected ingredients to PostgreSQL as refrigerator inventory.

Plain SQL through psycopg2 -- no ORM. Each detection run appends new rows
(source="vision"); it does not merge/upsert against existing inventory.
Deduplicating repeated detections across runs is a natural next step, not
implemented here to keep this phase's scope small.
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
    conn.commit()


def save_ingredients(conn, ingredients: list[Ingredient], source: str = "vision") -> int:
    """Insert each ingredient as a new inventory row. Returns rows inserted."""
    if not ingredients:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO inventory (name, estimated_quantity, unit, confidence, needs_confirmation, source)
            VALUES %s
            """,
            [
                (ing.name, ing.estimated_quantity, ing.unit, ing.confidence, ing.needs_confirmation, source)
                for ing in ingredients
            ],
        )
    conn.commit()
    return len(ingredients)


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
