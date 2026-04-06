from __future__ import annotations

from pathlib import Path

import duckdb

from insider.config import RuntimeConfig


def connect(runtime: RuntimeConfig, working_root: Path) -> duckdb.DuckDBPyConnection:
    temp_directory = runtime.temp_directory
    if not temp_directory.is_absolute():
        temp_directory = working_root / temp_directory
    temp_directory.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    connection.execute(f"SET threads = {runtime.threads}")
    connection.execute(f"SET memory_limit = '{runtime.memory_limit}'")
    connection.execute(f"SET temp_directory = '{_sql_escape(temp_directory)}'")
    connection.execute(f"SET preserve_insertion_order = {str(runtime.preserve_insertion_order).lower()}")
    connection.execute("SET enable_progress_bar = false")
    return connection


def _sql_escape(path: Path | str) -> str:
    return str(path).replace("'", "''")
