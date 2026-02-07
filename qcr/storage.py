"""Storage engine for the .qcr format.

Provides `save_qcr` and `load_qcr` — the two fundamental I/O operations.
All data is persisted as Parquet (ZSTD-compressed) with QuantCrate metadata
embedded in the Arrow schema metadata under the `qcr_metadata` key.

**Encoding strategy (Phase 8):**

Each column uses a Parquet encoding chosen for its data type:

- ``timestamp``  → ``DELTA_BINARY_PACKED``  — timestamps are monotonically
  increasing integers; delta encoding stores only the tiny diffs.
- ``open/high/low/close`` (float32) → ``BYTE_STREAM_SPLIT`` — splits the
  IEEE-754 bytes into four homogeneous streams so ZSTD compresses them
  dramatically better.
- ``volume`` (uint64) → ``DELTA_BINARY_PACKED`` — integer volumes with
  gradual changes benefit from delta encoding.

Additional write-time optimisations:

- ``data_page_version="2.0"``  — enables page-level statistics & encoding
  fallback metadata.
- ``write_page_index=True``  — gathers column min/max into a page index
  for efficient range scans (e.g. ``WHERE timestamp BETWEEN …``).
- ``data_page_size=65536`` (64 KB) — smaller pages give finer granularity
  for predicate push-down on sorted timestamp data.
- ``sorting_columns=[timestamp ASC]`` — declares the physical sort order
  so readers can skip entire row groups.
- ``compression_level=9`` (ZSTD) — higher level for archival financial
  data; slower writes, smaller files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from qcr.schema import (
    PAYLOAD_SCHEMA,
    QCR_METADATA_KEY,
    AuditTrail,
    FullMetadata,
    QcrMetadata,
)

# Type alias for anything we accept as a DataFrame.
DataFrameLike = Union[pd.DataFrame, pl.DataFrame]


# ---------------------------------------------------------------------------
# Encoding / write-option constants
# ---------------------------------------------------------------------------

# Per-column Parquet encodings — chosen for the data characteristics of
# each column in the canonical OHLCV payload.
COLUMN_ENCODINGS: dict[str, str] = {
    "timestamp": "DELTA_BINARY_PACKED",
    "open":      "BYTE_STREAM_SPLIT",
    "high":      "BYTE_STREAM_SPLIT",
    "low":       "BYTE_STREAM_SPLIT",
    "close":     "BYTE_STREAM_SPLIT",
    "volume":    "DELTA_BINARY_PACKED",
}

# ZSTD compression level (1–22).  Level 9 is a good trade-off between
# compression ratio and write speed for archival financial data.
ZSTD_COMPRESSION_LEVEL: int = 9

# Target data-page size in bytes.  64 KB pages give finer granularity for
# predicate push-down on sorted timestamp data than the default 1 MB.
DATA_PAGE_SIZE: int = 65_536  # 64 KB

# Sorting declaration — tells readers the file is physically sorted by
# timestamp ascending, enabling row-group skipping.
SORTING_COLUMNS: tuple[pq.SortingColumn, ...] = (
    pq.SortingColumn(0, descending=False, nulls_first=False),
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_arrow_table(df: DataFrameLike) -> pa.Table:
    """Convert a Pandas or Polars DataFrame to a PyArrow Table.

    The resulting table is cast to the canonical PAYLOAD_SCHEMA so that
    column names, types, and nullability are always consistent.
    """
    if isinstance(df, pl.DataFrame):
        arrow_table = df.to_arrow()
    elif isinstance(df, pd.DataFrame):
        arrow_table = pa.Table.from_pandas(df, preserve_index=False)
    else:
        raise TypeError(f"Expected a Pandas or Polars DataFrame, got {type(df).__name__}")

    return _cast_to_payload_schema(arrow_table)


def _cast_to_payload_schema(table: pa.Table) -> pa.Table:
    """Cast an Arrow table to the canonical PAYLOAD_SCHEMA.

    Ensures every column exists, has the right type, and is in the right order.
    Raises a clear error if a required column is missing.
    """
    expected_columns = {field.name for field in PAYLOAD_SCHEMA}
    actual_columns = set(table.column_names)

    missing = expected_columns - actual_columns
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    # Cast each column to the expected type, preserving order from the schema.
    cast_arrays: list[pa.Array] = []
    for field in PAYLOAD_SCHEMA:
        column = table.column(field.name)
        if column.type != field.type:
            column = column.cast(field.type)
        cast_arrays.append(column)

    return pa.table(
        {field.name: arr for field, arr in zip(PAYLOAD_SCHEMA, cast_arrays)},
        schema=PAYLOAD_SCHEMA,
    )


def _embed_metadata(schema: pa.Schema, metadata: FullMetadata) -> pa.Schema:
    """Return a new Arrow schema with QCR metadata injected.

    Existing Arrow metadata (e.g. pandas metadata) is preserved alongside
    the QuantCrate metadata.
    """
    existing = schema.metadata or {}
    merged = {
        **existing,
        QCR_METADATA_KEY: metadata.model_dump_json().encode("utf-8"),
    }
    return schema.with_metadata(merged)


def _extract_metadata(schema: pa.Schema) -> FullMetadata:
    """Read and parse the QuantCrate metadata from an Arrow schema.

    Raises ValueError if the metadata key is missing or the JSON is invalid.
    """
    raw = (schema.metadata or {}).get(QCR_METADATA_KEY)
    if raw is None:
        raise ValueError(
            "File does not contain QuantCrate metadata. "
            "Is this a valid .qcr file?"
        )
    return FullMetadata.model_validate_json(raw)


def _write_table(table: pa.Table, path: str) -> None:
    """Write an Arrow table to disk with all QCR encoding optimisations.

    This is the single write path used by both ``save_qcr`` and
    ``save_sealed_qcr``.  Centralising it guarantees that every ``.qcr``
    file benefits from the same encoding and compression settings.
    """
    pq.write_table(
        table,
        path,
        # --- Compression -----------------------------------------------
        compression="zstd",
        compression_level=ZSTD_COMPRESSION_LEVEL,
        # --- Per-column encoding ---------------------------------------
        #     use_dictionary must be False when column_encoding is set.
        use_dictionary=False,
        column_encoding=COLUMN_ENCODINGS,
        # --- Page-level optimisations ----------------------------------
        data_page_version="2.0",
        data_page_size=DATA_PAGE_SIZE,
        write_page_index=True,
        write_statistics=True,
        # --- Row-group metadata ----------------------------------------
        sorting_columns=SORTING_COLUMNS,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_qcr(
    df: DataFrameLike,
    metadata: QcrMetadata,
    path: Union[str, Path],
) -> Path:
    """Save a DataFrame as a .qcr file.

    Steps:
        1. Validate and convert the DataFrame to an Arrow Table.
        2. Validate and embed the metadata into the Arrow schema.
        3. Write to disk as Parquet with ZSTD compression.

    Args:
        df: A Pandas or Polars DataFrame with OHLCV columns.
        metadata: A validated QcrMetadata identity header.
        path: Destination file path (will be created / overwritten).

    Returns:
        The resolved Path of the written file.

    Raises:
        TypeError: If `df` is not a supported DataFrame type.
        ValueError: If required columns are missing or metadata is invalid.
    """
    path = Path(path)

    # 1. DataFrame → Arrow (typed + validated)
    table = _to_arrow_table(df)

    # 2. Wrap identity metadata (no audit trail yet — that comes from the Auditor)
    full_metadata = FullMetadata(identity=metadata)
    schema_with_meta = _embed_metadata(table.schema, full_metadata)
    table = table.replace_schema_metadata(schema_with_meta.metadata)

    # 3. Write to disk (with per-column encoding + optimised compression)
    _write_table(table, str(path))

    return path.resolve()


def load_qcr(
    path: Union[str, Path],
) -> tuple[pl.DataFrame, FullMetadata]:
    """Load a .qcr file, returning the DataFrame and its metadata.

    The file is memory-mapped for near-instant reads — only the columns
    and row groups you actually access are read from disk.

    Args:
        path: Path to an existing .qcr file.

    Returns:
        A tuple of (Polars DataFrame, FullMetadata).

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file is missing QuantCrate metadata.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Memory-mapped read
    table = pq.read_table(str(path), memory_map=True)

    # Extract metadata before converting to Polars (Polars drops Arrow metadata)
    full_metadata = _extract_metadata(table.schema)

    df = pl.from_arrow(table)

    return df, full_metadata


def save_sealed_qcr(
    df: DataFrameLike,
    metadata: QcrMetadata,
    audit: "AuditTrail",
    path: Union[str, Path],
) -> Path:
    """Save a DataFrame as a *sealed* .qcr file (identity + audit trail).

    This is the "post-audit" counterpart to `save_qcr`.  The Auditor
    produces an `AuditTrail`; this function embeds it alongside the
    identity metadata before writing.

    Args:
        df: A Pandas or Polars DataFrame with OHLCV columns.
        metadata: A validated QcrMetadata identity header.
        audit: The AuditTrail produced by `run_audit().to_audit_trail()`.
        path: Destination file path (will be created / overwritten).

    Returns:
        The resolved Path of the written file.
    """
    path = Path(path)

    table = _to_arrow_table(df)

    full_metadata = FullMetadata(identity=metadata, audit=audit)
    schema_with_meta = _embed_metadata(table.schema, full_metadata)
    table = table.replace_schema_metadata(schema_with_meta.metadata)

    _write_table(table, str(path))

    return path.resolve()


def read_last_timestamp(path: Union[str, Path]) -> "datetime":
    """Read the last (most recent) timestamp from a .qcr file.

    This reads only the ``timestamp`` column (memory-mapped) and returns
    the maximum value.  It is used by the resumption feature to determine
    where a previous download left off.

    Args:
        path: Path to an existing .qcr file.

    Returns:
        The last ``datetime`` in the file (timezone-aware, UTC).

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file is missing QuantCrate metadata or has no rows.
    """
    from datetime import datetime as _dt

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Validate that this is a real .qcr file.
    _extract_metadata(pq.ParquetFile(str(path), memory_map=True).schema_arrow)

    # Read only the timestamp column for efficiency.
    table = pq.read_table(str(path), columns=["timestamp"], memory_map=True)
    if table.num_rows == 0:
        raise ValueError("The .qcr file contains no data rows.")

    ts_array = table.column("timestamp")
    # pc.max returns a scalar; .as_py() converts to Python datetime.
    import pyarrow.compute as pc

    last_ts = pc.max(ts_array).as_py()
    return last_ts


def read_qcr_metadata(path: Union[str, Path]) -> FullMetadata:
    """Read only the metadata from a .qcr file without loading data.

    This reads just the Parquet footer (a few KB), making it ideal for
    the `qcr info` command.

    Args:
        path: Path to an existing .qcr file.

    Returns:
        The FullMetadata (identity + optional audit trail).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    parquet_file = pq.ParquetFile(str(path), memory_map=True)
    schema = parquet_file.schema_arrow

    return _extract_metadata(schema)


def read_encoding_stats(path: Union[str, Path]) -> dict:
    """Read encoding and compression statistics from a .qcr file.

    Inspects the Parquet footer to extract per-column encoding,
    compression codec, file size, and row/row-group counts — without
    loading any data into memory.

    Args:
        path: Path to an existing .qcr file.

    Returns:
        A dict with keys:
            ``file_size_bytes``, ``num_rows``, ``num_row_groups``,
            ``compression``, ``columns`` (list of per-column dicts with
            ``name``, ``physical_type``, ``encodings``, ``compression``,
            ``total_compressed_bytes``, ``total_uncompressed_bytes``).

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    pf = pq.ParquetFile(str(path), memory_map=True)
    parquet_meta = pf.metadata

    # Aggregate per-column stats across all row groups.
    column_stats: dict[str, dict] = {}
    for rg_idx in range(parquet_meta.num_row_groups):
        rg = parquet_meta.row_group(rg_idx)
        for col_idx in range(rg.num_columns):
            col = rg.column(col_idx)
            name = col.path_in_schema
            if name not in column_stats:
                column_stats[name] = {
                    "name": name,
                    "physical_type": col.physical_type,
                    "encodings": set(),
                    "compression": col.compression,
                    "total_compressed_bytes": 0,
                    "total_uncompressed_bytes": 0,
                }
            entry = column_stats[name]
            # Encodings is a string like "DELTA_BINARY_PACKED, RLE"
            if col.encodings:
                for enc in col.encodings:
                    entry["encodings"].add(enc)
            entry["total_compressed_bytes"] += col.total_compressed_size
            entry["total_uncompressed_bytes"] += col.total_uncompressed_size

    # Convert sets to sorted lists for JSON-friendliness.
    columns = []
    for col_info in column_stats.values():
        col_info["encodings"] = sorted(col_info["encodings"])
        columns.append(col_info)

    return {
        "file_size_bytes": path.stat().st_size,
        "num_rows": parquet_meta.num_rows,
        "num_row_groups": parquet_meta.num_row_groups,
        "compression": columns[0]["compression"] if columns else "UNKNOWN",
        "columns": columns,
    }
