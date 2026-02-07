"""Virtual Table Interface — query .qcr files with SQL.

Uses **DuckDB** as the in-process SQL engine.  Since ``.qcr`` files are
ZSTD-compressed Parquet under the hood, DuckDB can read them directly
via ``read_parquet()`` — no loading, no copying, no database server.

Usage (programmatic)::

    from qcr.vtable import query_qcr, register_qcr, QcrQueryResult

    # One-shot query
    result = query_qcr("SELECT * FROM 'data.qcr' WHERE volume > 1000000")

    # Register as a named table, then run multiple queries
    con = register_qcr("data.qcr", table_name="aapl")
    result = execute_sql(con, "SELECT * FROM aapl WHERE close > 150 LIMIT 10")

Usage (CLI)::

    qcr sql "SELECT * FROM 'data.qcr' WHERE volume > 1000000"
    qcr sql "SELECT avg(close), max(volume) FROM 'data.qcr'" --format csv
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import polars as pl
import pyarrow.parquet as pq

from qcr.schema import QCR_METADATA_KEY

# ---------------------------------------------------------------------------
# Optional DuckDB import
# ---------------------------------------------------------------------------

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    """Supported output formats for SQL query results."""

    TABLE = "table"
    CSV = "csv"
    JSON = "json"


@dataclass(frozen=True)
class QcrQueryResult:
    """Result of a SQL query against one or more ``.qcr`` files.

    Attributes:
        columns: Ordered list of column names.
        rows: List of tuples, one per result row.
        row_count: Number of result rows.
        column_types: DuckDB type names for each column.
    """

    columns: list[str]
    rows: list[tuple]
    row_count: int
    column_types: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class VTableError(Exception):
    """Raised when the virtual-table engine encounters an error."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_duckdb() -> None:
    """Raise a helpful error if DuckDB is not installed."""
    if duckdb is None:
        raise VTableError(
            "DuckDB is required for SQL queries. "
            "Install it with: pip install quantcrate[sql]"
        )


def _validate_qcr_file(path: Union[str, Path]) -> Path:
    """Verify that *path* points to a valid ``.qcr`` file.

    Checks:
        1. File exists.
        2. File has ``.qcr`` extension.
        3. File contains the ``qcr_metadata`` key in its Parquet schema.

    Returns the resolved ``Path``.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise VTableError(f"File not found: {p}")
    if p.suffix.lower() != ".qcr":
        raise VTableError(
            f"Expected a .qcr file, got '{p.suffix}'. "
            "Use read_parquet() directly for plain Parquet files."
        )
    # Quick metadata check — reads only the footer (a few KB).
    try:
        pf = pq.ParquetFile(str(p), memory_map=True)
        meta = pf.schema_arrow.metadata or {}
        if QCR_METADATA_KEY not in meta:
            raise VTableError(
                f"'{p.name}' is not a valid QuantCrate file — "
                "missing qcr_metadata in Arrow schema."
            )
    except Exception as exc:
        if isinstance(exc, VTableError):
            raise
        raise VTableError(f"Cannot read '{p.name}': {exc}") from exc

    return p


# ---------------------------------------------------------------------------
# SQL rewriting — DuckDB does not recognise the .qcr extension, so we
# transparently wrap any FROM 'path.qcr' with read_parquet('path.qcr').
# ---------------------------------------------------------------------------

_QCR_FROM_PATTERN = re.compile(
    r"""
    (?<=FROM\s)          # preceded by FROM + whitespace (look-behind)
    '([^']+\.qcr)'      # single-quoted path ending in .qcr
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _rewrite_qcr_paths(sql: str) -> str:
    """Rewrite ``FROM 'file.qcr'`` → ``FROM read_parquet('file.qcr')``.

    DuckDB auto-detects ``.parquet`` files but not ``.qcr``.  This helper
    transparently wraps every ``.qcr`` file reference so DuckDB can read it.
    """
    return _QCR_FROM_PATTERN.sub(r"read_parquet('\1')", sql)


def _result_from_connection(result: "duckdb.DuckDBPyConnection") -> QcrQueryResult:
    """Convert a DuckDB execute() result to a ``QcrQueryResult``.

    ``con.execute(sql)`` returns the *connection* itself (``DuckDBPyConnection``),
    which exposes ``.description`` (PEP 249) and ``.fetchall()``.
    """
    description = result.description or []
    columns = [desc[0] for desc in description]
    types = [desc[1] for desc in description]
    rows = result.fetchall()
    return QcrQueryResult(
        columns=columns,
        rows=rows,
        row_count=len(rows),
        column_types=[str(t) for t in types],
    )


# ---------------------------------------------------------------------------
# Public API — one-shot query
# ---------------------------------------------------------------------------

def query_qcr(
    sql: str,
    *,
    params: Optional[list] = None,
) -> QcrQueryResult:
    """Execute a SQL query that references ``.qcr`` files by path.

    DuckDB automatically resolves file paths inside ``FROM`` clauses,
    so you can write::

        SELECT * FROM 'path/to/data.qcr' WHERE volume > 1000000

    Internally, ``.qcr`` paths are wrapped with ``read_parquet()`` so
    DuckDB can recognise the file format.

    Args:
        sql: A SQL query string.  File paths in ``FROM`` must be quoted.
        params: Optional positional parameters for prepared statements.

    Returns:
        A ``QcrQueryResult`` with columns, rows, and metadata.

    Raises:
        VTableError: If DuckDB is missing, the file is invalid, or the
            query fails.
    """
    _ensure_duckdb()

    rewritten = _rewrite_qcr_paths(sql)

    try:
        con = duckdb.connect(":memory:")
        if params:
            result = con.execute(rewritten, params)
        else:
            result = con.execute(rewritten)
        return _result_from_connection(result)
    except duckdb.Error as exc:
        raise VTableError(f"SQL error: {exc}") from exc
    except Exception as exc:
        raise VTableError(f"Query failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API — register as a named virtual table
# ---------------------------------------------------------------------------

def register_qcr(
    path: Union[str, Path],
    *,
    table_name: Optional[str] = None,
    connection: Optional["duckdb.DuckDBPyConnection"] = None,
) -> "duckdb.DuckDBPyConnection":
    """Register a ``.qcr`` file as a named virtual table in DuckDB.

    After registration the file can be queried by table name::

        con = register_qcr("AAPL.qcr", table_name="aapl")
        result = execute_sql(con, "SELECT * FROM aapl WHERE close > 150")

    Args:
        path: Path to a ``.qcr`` file.
        table_name: SQL table name.  Defaults to the file stem
            (e.g. ``"AAPL"`` for ``AAPL.qcr``).
        connection: An existing DuckDB connection.  If ``None`` a fresh
            in-memory connection is created.

    Returns:
        The DuckDB connection (same object if *connection* was provided).
    """
    _ensure_duckdb()

    resolved = _validate_qcr_file(path)
    name = table_name or resolved.stem

    con = connection or duckdb.connect(":memory:")
    try:
        # Use forward slashes for DuckDB path compatibility on Windows.
        safe_path = str(resolved).replace("\\", "/")
        con.execute(
            f'CREATE OR REPLACE VIEW "{name}" AS '
            f"SELECT * FROM read_parquet('{safe_path}')"
        )
    except duckdb.Error as exc:
        raise VTableError(
            f"Failed to register '{resolved.name}' as table '{name}': {exc}"
        ) from exc

    return con


def execute_sql(
    connection: "duckdb.DuckDBPyConnection",
    sql: str,
    *,
    params: Optional[list] = None,
) -> QcrQueryResult:
    """Run a SQL query on a connection that has registered ``.qcr`` tables.

    Args:
        connection: A DuckDB connection (from ``register_qcr``).
        sql: SQL query string.
        params: Optional positional parameters.

    Returns:
        A ``QcrQueryResult``.
    """
    _ensure_duckdb()

    try:
        if params:
            result = connection.execute(sql, params)
        else:
            result = connection.execute(sql)
        return _result_from_connection(result)
    except duckdb.Error as exc:
        raise VTableError(f"SQL error: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API — query returning Polars DataFrame
# ---------------------------------------------------------------------------

def query_qcr_df(
    sql: str,
    *,
    params: Optional[list] = None,
) -> pl.DataFrame:
    """Execute a SQL query and return the result as a Polars DataFrame.

    This is a convenience wrapper around ``query_qcr`` that converts
    the result directly to a Polars DataFrame for downstream analysis.

    Args:
        sql: A SQL query string.
        params: Optional positional parameters.

    Returns:
        A Polars DataFrame with the query results.
    """
    _ensure_duckdb()

    rewritten = _rewrite_qcr_paths(sql)

    try:
        con = duckdb.connect(":memory:")
        if params:
            result = con.execute(rewritten, params)
        else:
            result = con.execute(rewritten)
        # DuckDB → Arrow → Polars (zero-copy where possible)
        arrow_table = result.fetch_arrow_table()
        return pl.from_arrow(arrow_table)
    except duckdb.Error as exc:
        raise VTableError(f"SQL error: {exc}") from exc
    except Exception as exc:
        raise VTableError(f"Query failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API — read QCR metadata via SQL
# ---------------------------------------------------------------------------

def describe_qcr(path: Union[str, Path]) -> dict:
    """Return a summary dict of a ``.qcr`` file: metadata + row/column counts.

    Combines the QuantCrate metadata with DuckDB's schema introspection
    for a complete picture without loading the full dataset.

    Args:
        path: Path to a ``.qcr`` file.

    Returns:
        A dict with ``metadata`` (the QCR identity/audit), ``columns``
        (name → type mapping), and ``row_count``.
    """
    _ensure_duckdb()

    resolved = _validate_qcr_file(path)

    # Read QCR metadata from the Parquet footer.
    pf = pq.ParquetFile(str(resolved), memory_map=True)
    raw_meta = (pf.schema_arrow.metadata or {}).get(QCR_METADATA_KEY)
    qcr_meta = json.loads(raw_meta) if raw_meta else {}

    # Use DuckDB to get column types and row count.
    safe_path = str(resolved).replace("\\", "/")
    con = duckdb.connect(":memory:")
    try:
        desc = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{safe_path}')"
        ).fetchall()
        count = con.execute(
            f"SELECT count(*) FROM read_parquet('{safe_path}')"
        ).fetchone()[0]
    except duckdb.Error as exc:
        raise VTableError(f"Cannot describe '{resolved.name}': {exc}") from exc

    columns = {row[0]: row[1] for row in desc}

    return {
        "file": str(resolved),
        "row_count": count,
        "columns": columns,
        "metadata": qcr_meta,
    }


# ---------------------------------------------------------------------------
# Formatting helpers (used by the CLI)
# ---------------------------------------------------------------------------

def format_result(
    result: QcrQueryResult,
    fmt: OutputFormat = OutputFormat.TABLE,
) -> str:
    """Format a ``QcrQueryResult`` as a string for terminal output.

    Args:
        result: The query result.
        fmt: Output format (table, csv, or json).

    Returns:
        A formatted string ready for printing.
    """
    if fmt == OutputFormat.CSV:
        return _format_csv(result)
    elif fmt == OutputFormat.JSON:
        return _format_json(result)
    else:
        return _format_table(result)


def _format_table(result: QcrQueryResult) -> str:
    """Format as a pretty ASCII table."""
    if not result.columns:
        return "(empty result)"

    # Stringify all values.
    str_rows = [[str(v) for v in row] for row in result.rows]

    # Calculate column widths.
    widths = [len(c) for c in result.columns]
    for row in str_rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    # Build header.
    header = " | ".join(c.ljust(w) for c, w in zip(result.columns, widths))
    separator = "-+-".join("-" * w for w in widths)

    lines = [header, separator]
    for row in str_rows:
        lines.append(" | ".join(v.ljust(w) for v, w in zip(row, widths)))

    lines.append(f"\n({result.row_count} row{'s' if result.row_count != 1 else ''})")
    return "\n".join(lines)


def _format_csv(result: QcrQueryResult) -> str:
    """Format as CSV (always uses ``\\n`` line endings, even on Windows)."""
    import csv
    import io

    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(result.columns)
    for row in result.rows:
        writer.writerow(row)
    return buf.getvalue().rstrip("\n")


def _format_json(result: QcrQueryResult) -> str:
    """Format as JSON (array of objects)."""
    records = [
        dict(zip(result.columns, [_json_safe(v) for v in row]))
        for row in result.rows
    ]
    return json.dumps(records, indent=2, default=str)


def _json_safe(value):
    """Convert a value to a JSON-serializable type."""
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    return str(value)
