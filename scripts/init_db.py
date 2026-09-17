#!/usr/bin/env python3
"""One-time (idempotent) database setup: creates all tables this project needs."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inventory.inventory_service import ensure_schema as ensure_inventory_schema
from app.inventory.inventory_service import get_connection
from app.rag.recipe_store import ensure_schema as ensure_recipe_schema


def main() -> int:
    conn = get_connection()
    try:
        ensure_inventory_schema(conn)
        print("inventory table ready.")
        ensure_recipe_schema(conn)
        print("recipes table + pgvector index ready.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
