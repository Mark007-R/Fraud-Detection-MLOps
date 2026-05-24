#!/usr/bin/env bash
# Bootstraps extra databases on top of the default POSTGRES_DB.
# Reads POSTGRES_MULTIPLE_DATABASES (comma-separated) and CREATE DATABASE
# any name not already present. Used by docker-compose to place the MLflow
# tracking + registry on the same Postgres instance as the telemetry store
# without sacrificing per-database isolation.

set -euo pipefail

if [[ -z "${POSTGRES_MULTIPLE_DATABASES:-}" ]]; then
    echo "[postgres-init] POSTGRES_MULTIPLE_DATABASES unset; nothing to do."
    exit 0
fi

IFS=',' read -ra extra_dbs <<< "${POSTGRES_MULTIPLE_DATABASES}"
for db in "${extra_dbs[@]}"; do
    db="$(echo "$db" | tr -d '[:space:]')"
    if [[ -z "$db" || "$db" == "${POSTGRES_DB:-}" ]]; then
        continue
    fi
    echo "[postgres-init] Creating database '${db}' (owner='${POSTGRES_USER}')"
    psql -v ON_ERROR_STOP=1 --username "${POSTGRES_USER}" <<-EOSQL
        CREATE DATABASE "${db}" OWNER "${POSTGRES_USER}";
EOSQL
done
