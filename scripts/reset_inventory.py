#!/usr/bin/env python3
"""Clears all rows from the inventory table. Run manually when testing."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inventory.inventory_service import clear_all, ensure_schema, get_connection


def main() -> int:
    conn = get_connection()
    try:
        ensure_schema(conn)
        deleted = clear_all(conn)
    finally:
        conn.close()
    print(f"Cleared {deleted} row(s) from inventory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
