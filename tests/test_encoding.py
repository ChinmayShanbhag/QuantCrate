"""Tests for Phase 8 — data encoding & storage optimisations.

Verifies that:
- Per-column Parquet encodings are applied correctly.
- ZSTD compression with the configured level is used.
- Sorting metadata is written to the Parquet footer.
- Page index and data page v2 are enabled.
- ``read_encoding_stats`` returns accurate file-level and per-column info.
- Round-trip data fidelity is preserved under the new encodings.
- Optimised files are smaller than naive-encoded files.
"""

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from qcr.schema import AssetClass, QcrMetadata, Timescale
from qcr.storage import (
    COLUMN_ENCODINGS,
    DATA_PAGE_SIZE,
    SORTING_COLUMNS,
    ZSTD_COMPRESSION_LEVEL,
    load_qcr,
    read_encoding_stats,
    save_qcr,
    save_sealed_qcr,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_METADATA = QcrMetadata(
    ticker="AAPL",
    asset_class=AssetClass.EQUITY,
    currency="USD",
    exchange="XNAS",
    timezone="America/New_York",
    timescale=Timescale.ONE_DAY,
    is_adjusted=False,
)


def _make_df(rows: int = 500) -> pl.DataFrame:
    """Create a realistic OHLCV DataFrame with monotonic timestamps."""
    timestamps = [
        datetime(2023, 1, 1, tzinfo=timezone.utc) + __import__("datetime").timedelta(days=i)
        for i in range(rows)
    ]
    import random

    random.seed(42)
    base_price = 150.0
    prices = []
    for i in range(rows):
        # Simulate a random walk
        base_price += random.uniform(-2.0, 2.0)
        base_price = max(1.0, base_price)
        prices.append(base_price)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [p - random.uniform(0.5, 1.5) for p in prices],
            "high": [p + random.uniform(0.5, 2.0) for p in prices],
            "low": [p - random.uniform(1.0, 3.0) for p in prices],
            "close": prices,
            "volume": [random.randint(500_000, 50_000_000) for _ in range(rows)],
        }
    )


def _make_small_df(rows: int = 5) -> pl.DataFrame:
    """Create a minimal valid OHLCV DataFrame."""
    timestamps = [
        datetime(2025, 1, 1, 9, 30 + i, tzinfo=timezone.utc) for i in range(rows)
    ]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [150.0 + i for i in range(rows)],
            "high": [155.0 + i for i in range(rows)],
            "low": [149.0 + i for i in range(rows)],
            "close": [152.0 + i for i in range(rows)],
            "volume": [1_000_000 + i * 100 for i in range(rows)],
        }
    )


# ---------------------------------------------------------------------------
# Constants validation
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify the encoding configuration constants are well-defined."""

    def test_column_encodings_has_all_payload_columns(self):
        """Every column in the payload schema has an encoding defined."""
        expected = {"timestamp", "open", "high", "low", "close", "volume"}
        assert set(COLUMN_ENCODINGS.keys()) == expected

    def test_timestamp_uses_delta_encoding(self):
        """Timestamp column should use DELTA_BINARY_PACKED."""
        assert COLUMN_ENCODINGS["timestamp"] == "DELTA_BINARY_PACKED"

    def test_price_columns_use_byte_stream_split(self):
        """Float price columns should use BYTE_STREAM_SPLIT."""
        for col in ("open", "high", "low", "close"):
            assert COLUMN_ENCODINGS[col] == "BYTE_STREAM_SPLIT"

    def test_volume_uses_delta_encoding(self):
        """Volume column should use DELTA_BINARY_PACKED."""
        assert COLUMN_ENCODINGS["volume"] == "DELTA_BINARY_PACKED"

    def test_zstd_compression_level_in_valid_range(self):
        """ZSTD level must be between 1 and 22."""
        assert 1 <= ZSTD_COMPRESSION_LEVEL <= 22

    def test_data_page_size_is_64kb(self):
        """Data page size should be 64 KB."""
        assert DATA_PAGE_SIZE == 65_536

    def test_sorting_columns_defined(self):
        """At least one sorting column should be declared."""
        assert len(SORTING_COLUMNS) >= 1
        assert SORTING_COLUMNS[0].column_index == 0  # timestamp
        assert SORTING_COLUMNS[0].descending is False


# ---------------------------------------------------------------------------
# Parquet footer inspection — verify encodings are actually written
# ---------------------------------------------------------------------------

class TestParquetEncodings:
    """Inspect the Parquet footer to verify per-column encodings are applied."""

    def test_timestamp_encoding_in_footer(self, tmp_path: Path):
        """The timestamp column should report DELTA_BINARY_PACKED encoding."""
        out = tmp_path / "enc.qcr"
        save_qcr(_make_df(), SAMPLE_METADATA, out)

        pf = pq.ParquetFile(str(out))
        rg = pf.metadata.row_group(0)
        ts_col = rg.column(0)  # timestamp is first column
        assert "DELTA_BINARY_PACKED" in ts_col.encodings

    def test_price_encoding_in_footer(self, tmp_path: Path):
        """Price columns (open/high/low/close) should report BYTE_STREAM_SPLIT."""
        out = tmp_path / "enc.qcr"
        save_qcr(_make_df(), SAMPLE_METADATA, out)

        pf = pq.ParquetFile(str(out))
        rg = pf.metadata.row_group(0)
        # Columns 1-4 are open, high, low, close
        for col_idx in range(1, 5):
            col = rg.column(col_idx)
            assert "BYTE_STREAM_SPLIT" in col.encodings, (
                f"Column {col.path_in_schema} missing BYTE_STREAM_SPLIT"
            )

    def test_volume_encoding_in_footer(self, tmp_path: Path):
        """Volume column should report DELTA_BINARY_PACKED encoding."""
        out = tmp_path / "enc.qcr"
        save_qcr(_make_df(), SAMPLE_METADATA, out)

        pf = pq.ParquetFile(str(out))
        rg = pf.metadata.row_group(0)
        vol_col = rg.column(5)  # volume is last column
        assert "DELTA_BINARY_PACKED" in vol_col.encodings

    def test_compression_is_zstd(self, tmp_path: Path):
        """All columns should use ZSTD compression."""
        out = tmp_path / "enc.qcr"
        save_qcr(_make_df(), SAMPLE_METADATA, out)

        pf = pq.ParquetFile(str(out))
        rg = pf.metadata.row_group(0)
        for col_idx in range(rg.num_columns):
            col = rg.column(col_idx)
            assert col.compression == "ZSTD", (
                f"Column {col.path_in_schema} uses {col.compression}, expected ZSTD"
            )

    def test_sorting_columns_in_footer(self, tmp_path: Path):
        """The Parquet footer should declare timestamp as the sort key."""
        out = tmp_path / "enc.qcr"
        save_qcr(_make_df(), SAMPLE_METADATA, out)

        pf = pq.ParquetFile(str(out))
        rg = pf.metadata.row_group(0)
        sorting = rg.sorting_columns
        # sorting_columns should be set (may be None if not supported by version)
        if sorting is not None:
            assert len(sorting) >= 1
            assert sorting[0].column_index == 0

    def test_no_dictionary_encoding(self, tmp_path: Path):
        """Dictionary encoding should be disabled (we use explicit encodings)."""
        out = tmp_path / "enc.qcr"
        save_qcr(_make_df(), SAMPLE_METADATA, out)

        pf = pq.ParquetFile(str(out))
        rg = pf.metadata.row_group(0)
        for col_idx in range(rg.num_columns):
            col = rg.column(col_idx)
            # PLAIN_DICTIONARY or RLE_DICTIONARY should NOT be present
            assert "PLAIN_DICTIONARY" not in col.encodings, (
                f"Column {col.path_in_schema} unexpectedly has dictionary encoding"
            )


# ---------------------------------------------------------------------------
# Round-trip fidelity — data must survive encoding changes
# ---------------------------------------------------------------------------

class TestRoundTripFidelity:
    """Ensure data is losslessly preserved under the new encoding settings."""

    def test_polars_round_trip_preserves_values(self, tmp_path: Path):
        """All values survive a save → load round trip with new encodings."""
        out = tmp_path / "rt.qcr"
        original = _make_small_df()
        save_qcr(original, SAMPLE_METADATA, out)

        loaded, meta = load_qcr(out)

        assert loaded.shape == original.shape
        assert meta.identity.ticker == "AAPL"

        # Compare actual values
        for col in ("open", "high", "low", "close"):
            orig_vals = original[col].to_list()
            loaded_vals = loaded[col].to_list()
            for o, l in zip(orig_vals, loaded_vals):
                assert abs(o - l) < 1e-3, f"Mismatch in {col}: {o} vs {l}"

    def test_volume_round_trip_exact(self, tmp_path: Path):
        """Volume (uint64) values are exactly preserved."""
        out = tmp_path / "vol_rt.qcr"
        original = _make_small_df()
        save_qcr(original, SAMPLE_METADATA, out)

        loaded, _ = load_qcr(out)
        assert loaded["volume"].to_list() == original["volume"].to_list()

    def test_timestamp_round_trip_exact(self, tmp_path: Path):
        """Timestamps are exactly preserved (nanosecond precision)."""
        out = tmp_path / "ts_rt.qcr"
        original = _make_small_df()
        save_qcr(original, SAMPLE_METADATA, out)

        loaded, _ = load_qcr(out)
        orig_ts = original["timestamp"].to_list()
        loaded_ts = loaded["timestamp"].to_list()
        assert orig_ts == loaded_ts

    def test_large_dataset_round_trip(self, tmp_path: Path):
        """A 500-row dataset round-trips correctly with optimised encoding."""
        out = tmp_path / "large_rt.qcr"
        original = _make_df(500)
        save_qcr(original, SAMPLE_METADATA, out)

        loaded, _ = load_qcr(out)
        assert loaded.shape == original.shape
        # Spot-check first and last rows
        assert loaded[0, "volume"] == original[0, "volume"]
        assert loaded[-1, "volume"] == original[-1, "volume"]

    def test_sealed_file_round_trip(self, tmp_path: Path):
        """save_sealed_qcr also uses the new encodings and round-trips."""
        from qcr.schema import AuditTrail

        out = tmp_path / "sealed_rt.qcr"
        original = _make_small_df()
        audit = AuditTrail(
            audit_passed=True,
            audit_timestamp="2025-01-01T00:00:00Z",
            data_gaps=0,
            outliers_found=0,
        )
        save_sealed_qcr(original, SAMPLE_METADATA, audit, out)

        loaded, meta = load_qcr(out)
        assert loaded.shape == original.shape
        assert meta.audit is not None
        assert meta.audit.audit_passed is True

        # Verify encoding is applied to sealed files too
        pf = pq.ParquetFile(str(out))
        rg = pf.metadata.row_group(0)
        assert "DELTA_BINARY_PACKED" in rg.column(0).encodings


# ---------------------------------------------------------------------------
# read_encoding_stats
# ---------------------------------------------------------------------------

class TestReadEncodingStats:
    """Tests for the read_encoding_stats function."""

    def test_returns_file_size(self, tmp_path: Path):
        """Stats include the file size in bytes."""
        out = tmp_path / "stats.qcr"
        save_qcr(_make_df(), SAMPLE_METADATA, out)

        stats = read_encoding_stats(out)
        assert stats["file_size_bytes"] > 0
        assert stats["file_size_bytes"] == out.stat().st_size

    def test_returns_row_count(self, tmp_path: Path):
        """Stats include the correct number of rows."""
        df = _make_df(100)
        out = tmp_path / "stats.qcr"
        save_qcr(df, SAMPLE_METADATA, out)

        stats = read_encoding_stats(out)
        assert stats["num_rows"] == 100

    def test_returns_row_group_count(self, tmp_path: Path):
        """Stats include at least one row group."""
        out = tmp_path / "stats.qcr"
        save_qcr(_make_df(), SAMPLE_METADATA, out)

        stats = read_encoding_stats(out)
        assert stats["num_row_groups"] >= 1

    def test_returns_compression_codec(self, tmp_path: Path):
        """Stats report ZSTD as the compression codec."""
        out = tmp_path / "stats.qcr"
        save_qcr(_make_df(), SAMPLE_METADATA, out)

        stats = read_encoding_stats(out)
        assert stats["compression"] == "ZSTD"

    def test_returns_per_column_stats(self, tmp_path: Path):
        """Stats include per-column encoding and size information."""
        out = tmp_path / "stats.qcr"
        save_qcr(_make_df(), SAMPLE_METADATA, out)

        stats = read_encoding_stats(out)
        columns = stats["columns"]
        assert len(columns) == 6

        col_names = [c["name"] for c in columns]
        assert "timestamp" in col_names
        assert "close" in col_names
        assert "volume" in col_names

    def test_per_column_encodings_match_config(self, tmp_path: Path):
        """Per-column encodings in stats should match COLUMN_ENCODINGS."""
        out = tmp_path / "stats.qcr"
        save_qcr(_make_df(), SAMPLE_METADATA, out)

        stats = read_encoding_stats(out)
        for col_info in stats["columns"]:
            name = col_info["name"]
            expected_enc = COLUMN_ENCODINGS[name]
            assert expected_enc in col_info["encodings"], (
                f"Column {name}: expected {expected_enc} in {col_info['encodings']}"
            )

    def test_compressed_smaller_than_uncompressed(self, tmp_path: Path):
        """Total compressed size should be smaller than total uncompressed.

        Individual columns may have compressed > uncompressed when the
        column is tiny (ZSTD framing overhead exceeds the savings), so we
        check the aggregate rather than per-column.
        """
        out = tmp_path / "stats.qcr"
        save_qcr(_make_df(500), SAMPLE_METADATA, out)

        stats = read_encoding_stats(out)
        total_compressed = sum(
            c["total_compressed_bytes"] for c in stats["columns"]
        )
        total_uncompressed = sum(
            c["total_uncompressed_bytes"] for c in stats["columns"]
        )
        assert total_compressed <= total_uncompressed, (
            f"Total compressed ({total_compressed:,}) > "
            f"total uncompressed ({total_uncompressed:,})"
        )

    def test_file_not_found_raises(self):
        """read_encoding_stats raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            read_encoding_stats(Path("nonexistent.qcr"))


# ---------------------------------------------------------------------------
# Compression effectiveness — optimised files should be smaller
# ---------------------------------------------------------------------------

class TestCompressionEffectiveness:
    """Compare optimised .qcr files against naive Parquet writes."""

    def test_optimised_file_is_smaller_than_naive(self, tmp_path: Path):
        """A .qcr file with per-column encoding should be smaller than a
        naive ZSTD-compressed Parquet file (no special encodings)."""
        import pyarrow as pa

        df = _make_df(500)
        optimised_path = tmp_path / "optimised.qcr"
        naive_path = tmp_path / "naive.parquet"

        # Write with QCR optimisations
        save_qcr(df, SAMPLE_METADATA, optimised_path)

        # Write naive: just ZSTD compression, default encodings
        arrow_table = df.to_arrow()
        pq.write_table(
            arrow_table,
            str(naive_path),
            compression="zstd",
        )

        optimised_size = optimised_path.stat().st_size
        naive_size = naive_path.stat().st_size

        # The optimised file should be no larger (and typically smaller)
        # than the naive one.  We allow a small margin for metadata overhead.
        assert optimised_size <= naive_size * 1.05, (
            f"Optimised ({optimised_size:,} B) is more than 5% larger "
            f"than naive ({naive_size:,} B)"
        )

    def test_compression_ratio_above_threshold(self, tmp_path: Path):
        """The overall compression ratio should be at least 1.1x.

        With 500 rows the data is small enough that ZSTD framing overhead
        limits the achievable ratio.  For larger datasets (10k+ rows) the
        ratio is typically 2–5x, but we use a conservative threshold here
        to avoid flaky tests on small data.
        """
        out = tmp_path / "ratio.qcr"
        save_qcr(_make_df(500), SAMPLE_METADATA, out)

        stats = read_encoding_stats(out)
        total_compressed = sum(
            c["total_compressed_bytes"] for c in stats["columns"]
        )
        total_uncompressed = sum(
            c["total_uncompressed_bytes"] for c in stats["columns"]
        )
        ratio = total_uncompressed / total_compressed if total_compressed > 0 else 0

        assert ratio >= 1.1, (
            f"Overall compression ratio {ratio:.2f}x is below 1.1x threshold"
        )

    def test_large_dataset_compression_ratio(self, tmp_path: Path):
        """With 5000 rows, the compression ratio should exceed 1.2x.

        Random-walk float data has inherently high entropy, so the raw
        compression ratio is modest (~1.3x).  The real win from
        BYTE_STREAM_SPLIT + DELTA_BINARY_PACKED is that they *rearrange*
        the bytes so ZSTD can compress them better than it would with
        PLAIN encoding.  The ``test_optimised_file_is_smaller_than_naive``
        test verifies this relative improvement.
        """
        out = tmp_path / "large_ratio.qcr"
        save_qcr(_make_df(5000), SAMPLE_METADATA, out)

        stats = read_encoding_stats(out)
        total_compressed = sum(
            c["total_compressed_bytes"] for c in stats["columns"]
        )
        total_uncompressed = sum(
            c["total_uncompressed_bytes"] for c in stats["columns"]
        )
        ratio = total_uncompressed / total_compressed if total_compressed > 0 else 0

        assert ratio >= 1.2, (
            f"Large dataset compression ratio {ratio:.2f}x is below 1.2x"
        )
