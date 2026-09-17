#!/usr/bin/env bash
# Stops the local PostgreSQL 17 server used by this project.
set -euo pipefail

PG_BIN="/Applications/Postgres.app/Contents/Versions/17/bin"
DATA_DIR="$HOME/postgres-data/smartfridge-pg17"

"$PG_BIN/pg_ctl" -D "$DATA_DIR" stop
