#!/usr/bin/env bash
# Starts the local PostgreSQL 17 server (Postgres.app binaries) used by this
# project. This data directory was created outside of Postgres.app's GUI, so
# it must be started/stopped from the command line, not by clicking the app.
set -euo pipefail

PG_BIN="/Applications/Postgres.app/Contents/Versions/17/bin"
DATA_DIR="$HOME/postgres-data/smartfridge-pg17"
LOG_FILE="$HOME/postgres-data/smartfridge-pg17.log"

"$PG_BIN/pg_ctl" -D "$DATA_DIR" -l "$LOG_FILE" -o "-p 5432" start
"$PG_BIN/pg_isready" -p 5432
