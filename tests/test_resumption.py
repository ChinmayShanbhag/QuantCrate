"""Tests for the Resumption feature — Phase 7.

Covers:
    - ``read_last_timestamp`` from ``storage.py``
    - ``_detect_resume_start`` and ``_merge_dataframes`` from ``ingest.py``
    - ``ingest_to_qcr`` with ``resume=True`` (end-to-end)
    - CLI ``qcr ingest --resume`` flag
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qcr.adapters import AdapterError, BaseAdapter
from qcr.auditor import run_audit
from qcr.ingest import IngestResult, _detect_resume_start, _merge_dataframes, ingest_to_qcr
from qcr.schema import AssetClass, QcrMetadata, Timescale
from qcr.storage import load_qcr, read_last_timestamp, save_sealed_qcr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(
    rows: int = 30,
    start_date: datetime | None = None,
    timescale: Timescale = Timescale.ONE_DAY,
    base_price: float = 100.0,
) -> pl.DataFrame:
    """Create a valid OHLCV DataFrame that passes all audit checks."""
    base = start_date or datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    if timescale == Timescale.ONE_DAY:
        delta = timedelta(days=1)
    elif timescale == Timescale.ONE_MINUTE:
        delta = timedelta(minutes=1)
    else:
        delta = timedelta(hours=1)

    timestamps = [base + delta * i for i in range(rows)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [float(base_price + i * 0.5) for i in range(rows)],
            "high": [float(base_price + 5 + i * 0.5) for i in range(rows)],
            "low": [float(base_price - 2 + i * 0.5) for i in range(rows)],
            "close": [float(base_price + 2 + i * 0.5) for i in range(rows)],
            "volume": [1_000_000 + i * 100 for i in range(rows)],
        },
        schema={
            "timestamp": pl.Datetime("ns", "UTC"),
            "open": pl.Float32,
            "high": pl.Float32,
            "low": pl.Float32,
            "close": pl.Float32,
            "volume": pl.UInt64,
        },
    )


def _seal_test_file(path: Path, df: pl.DataFrame) -> Path:
    """Seal a DataFrame into a .qcr file for testing."""
    metadata = QcrMetadata(
        ticker="TEST",
        asset_class=AssetClass.EQUITY,
        currency="USD",
        exchange="XNAS",
        timezone="UTC",
        timescale=Timescale.ONE_DAY,
    )
    audit_result = run_audit(df, Timescale.ONE_DAY)
    return save_sealed_qcr(df, metadata, audit_result.to_audit_trail(), path)


class FakeAdapter(BaseAdapter):
    """A concrete adapter for testing that returns pre-configured data."""

    def __init__(
        self,
        df: Optional[pl.DataFrame] = None,
        error: Optional[str] = None,
    ) -> None:
        self._df = df
        self._error = error
        self.last_start: Optional[str] = None
        self.last_end: Optional[str] = None
        self.call_count: int = 0

    @property
    def source_name(self) -> str:
        return "FakeSource"

    async def fetch_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str,
        timescale: Timescale = Timescale.ONE_DAY,
    ) -> pl.DataFrame:
        self.call_count += 1
        self.last_start = start
        self.last_end = end
        if self._error:
            raise AdapterError(self._error)
        if self._df is not None:
            return self._df
        return _make_ohlcv_df()


# ===========================================================================
# Tests for read_last_timestamp
# ===========================================================================

class TestReadLastTimestamp:
    """Tests for ``storage.read_last_timestamp``."""

    def test_returns_last_timestamp(self, tmp_path: Path) -> None:
        """Reads the maximum timestamp from a .qcr file."""
        df = _make_ohlcv_df(rows=10)
        path = _seal_test_file(tmp_path / "test.qcr", df)

        last_ts = read_last_timestamp(path)

        # start = Jan 1 09:30 UTC, + 9 days
        expected = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc) + timedelta(days=9)
        assert last_ts == expected

    def test_single_row_file(self, tmp_path: Path) -> None:
        """Works correctly with a single-row file."""
        df = _make_ohlcv_df(rows=1)
        path = _seal_test_file(tmp_path / "single.qcr", df)

        last_ts = read_last_timestamp(path)
        expected = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
        assert last_ts == expected

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            read_last_timestamp(tmp_path / "missing.qcr")

    def test_non_qcr_file_raises_value_error(self, tmp_path: Path) -> None:
        """Raises ValueError for files without qcr_metadata."""
        # Write a plain Parquet file without qcr_metadata.
        table = pa.table({"timestamp": [1, 2, 3]})
        plain_path = tmp_path / "plain.qcr"
        pq.write_table(table, str(plain_path))

        with pytest.raises(ValueError, match="QuantCrate metadata"):
            read_last_timestamp(plain_path)

    def test_returns_timezone_aware_datetime(self, tmp_path: Path) -> None:
        """The returned datetime is timezone-aware (UTC)."""
        df = _make_ohlcv_df(rows=5)
        path = _seal_test_file(tmp_path / "tz.qcr", df)

        last_ts = read_last_timestamp(path)
        assert last_ts.tzinfo is not None

    def test_many_rows_returns_max(self, tmp_path: Path) -> None:
        """With many rows, returns the maximum (last) timestamp."""
        df = _make_ohlcv_df(rows=100)
        path = _seal_test_file(tmp_path / "many.qcr", df)

        last_ts = read_last_timestamp(path)
        expected = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc) + timedelta(days=99)
        assert last_ts == expected


# ===========================================================================
# Tests for _detect_resume_start
# ===========================================================================

class TestDetectResumeStart:
    """Tests for ``ingest._detect_resume_start``."""

    def test_no_file_returns_none(self, tmp_path: Path) -> None:
        """If the file doesn't exist, returns (None, None)."""
        new_start, existing_df = _detect_resume_start(
            tmp_path / "missing.qcr", "2025-01-01"
        )
        assert new_start is None
        assert existing_df is None

    def test_existing_file_returns_next_day(self, tmp_path: Path) -> None:
        """Returns the day after the last timestamp in the existing file."""
        df = _make_ohlcv_df(rows=10)
        path = tmp_path / "existing.qcr"
        _seal_test_file(path, df)

        new_start, existing_df = _detect_resume_start(path, "2025-01-01")

        # Last timestamp is Jan 1 09:30 + 9 days = Jan 10 09:30.
        # Next day = Jan 11.
        assert new_start == "2025-01-11"
        assert existing_df is not None
        assert existing_df.shape[0] == 10

    def test_corrupt_file_returns_none(self, tmp_path: Path) -> None:
        """A corrupt file is treated as non-existent (returns None, None)."""
        corrupt = tmp_path / "corrupt.qcr"
        corrupt.write_bytes(b"not a parquet file")

        new_start, existing_df = _detect_resume_start(corrupt, "2025-01-01")
        assert new_start is None
        assert existing_df is None

    def test_existing_df_is_loadable(self, tmp_path: Path) -> None:
        """The returned existing_df has the correct columns and types."""
        df = _make_ohlcv_df(rows=5)
        path = tmp_path / "check.qcr"
        _seal_test_file(path, df)

        _, existing_df = _detect_resume_start(path, "2025-01-01")
        assert existing_df is not None
        assert set(existing_df.columns) >= {"timestamp", "open", "high", "low", "close", "volume"}


# ===========================================================================
# Tests for _merge_dataframes
# ===========================================================================

class TestMergeDataframes:
    """Tests for ``ingest._merge_dataframes``."""

    def test_merge_non_overlapping(self) -> None:
        """Two non-overlapping DataFrames are concatenated."""
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        df1 = _make_ohlcv_df(rows=5, start_date=base)
        df2 = _make_ohlcv_df(rows=5, start_date=base + timedelta(days=5))

        merged = _merge_dataframes(df1, df2)
        assert merged.shape[0] == 10
        # Should be sorted by timestamp.
        timestamps = merged["timestamp"].to_list()
        assert timestamps == sorted(timestamps)

    def test_merge_with_overlap_deduplicates(self) -> None:
        """Overlapping timestamps are deduplicated (keep=last means new wins)."""
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        df1 = _make_ohlcv_df(rows=5, start_date=base, base_price=100.0)
        # Overlaps on days 3-4, extends to day 7.
        df2 = _make_ohlcv_df(rows=5, start_date=base + timedelta(days=3), base_price=200.0)

        merged = _merge_dataframes(df1, df2)
        # 3 unique from df1 (days 0-2) + 5 unique from df2 (days 3-7) = 8 rows.
        assert merged.shape[0] == 8

    def test_merge_new_data_takes_precedence(self) -> None:
        """When timestamps collide, the new (second) DataFrame's values win."""
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        df1 = _make_ohlcv_df(rows=3, start_date=base, base_price=100.0)
        df2 = _make_ohlcv_df(rows=3, start_date=base, base_price=200.0)

        merged = _merge_dataframes(df1, df2)
        assert merged.shape[0] == 3
        # All rows should have the new (200-based) prices.
        assert merged["open"][0] == pytest.approx(200.0, abs=0.1)

    def test_merge_empty_existing(self) -> None:
        """Merging with an empty existing DataFrame returns the new data."""
        empty = pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("ns", "UTC"),
                "open": pl.Float32,
                "high": pl.Float32,
                "low": pl.Float32,
                "close": pl.Float32,
                "volume": pl.UInt64,
            },
        )
        df2 = _make_ohlcv_df(rows=5)
        merged = _merge_dataframes(empty, df2)
        assert merged.shape[0] == 5

    def test_merge_empty_new(self) -> None:
        """Merging with empty new data returns the existing data."""
        df1 = _make_ohlcv_df(rows=5)
        empty = pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("ns", "UTC"),
                "open": pl.Float32,
                "high": pl.Float32,
                "low": pl.Float32,
                "close": pl.Float32,
                "volume": pl.UInt64,
            },
        )
        merged = _merge_dataframes(df1, empty)
        assert merged.shape[0] == 5

    def test_merge_result_is_sorted(self) -> None:
        """The merged result is always sorted by timestamp."""
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        # Pass the later data first, earlier data second.
        df1 = _make_ohlcv_df(rows=3, start_date=base + timedelta(days=5))
        df2 = _make_ohlcv_df(rows=3, start_date=base)

        merged = _merge_dataframes(df1, df2)
        timestamps = merged["timestamp"].to_list()
        assert timestamps == sorted(timestamps)


# ===========================================================================
# Tests for ingest_to_qcr with resume=True
# ===========================================================================

class TestIngestResumption:
    """End-to-end tests for ``ingest_to_qcr(resume=True)``."""

    def test_fresh_download_without_resume(self, tmp_path: Path) -> None:
        """Without resume, a fresh download works as before."""
        out = tmp_path / "fresh.qcr"
        adapter = FakeAdapter()

        result = asyncio.run(
            ingest_to_qcr(
                adapter=adapter,
                ticker="TEST",
                start="2025-01-01",
                end="2025-02-01",
                output=out,
                resume=False,
            )
        )

        assert result.success is True
        assert result.resumed is False
        assert result.rows == 30

    def test_resume_no_existing_file(self, tmp_path: Path) -> None:
        """Resume with no existing file behaves like a fresh download."""
        out = tmp_path / "new.qcr"
        adapter = FakeAdapter()

        result = asyncio.run(
            ingest_to_qcr(
                adapter=adapter,
                ticker="TEST",
                start="2025-01-01",
                end="2025-02-01",
                output=out,
                resume=True,
            )
        )

        assert result.success is True
        assert result.resumed is False
        assert result.path is not None
        assert result.path.exists()

    def test_resume_fetches_only_missing_data(self, tmp_path: Path) -> None:
        """Resume adjusts the start date to after the last existing timestamp."""
        out = tmp_path / "partial.qcr"

        # Step 1: Create an existing file with 10 days of data (Jan 1 - Jan 10).
        existing_df = _make_ohlcv_df(rows=10)
        _seal_test_file(out, existing_df)

        # Step 2: Resume with new data for days 11-20.
        new_data = _make_ohlcv_df(
            rows=10,
            start_date=datetime(2025, 1, 11, 9, 30, tzinfo=timezone.utc),
            base_price=110.0,
        )
        adapter = FakeAdapter(df=new_data)

        result = asyncio.run(
            ingest_to_qcr(
                adapter=adapter,
                ticker="TEST",
                start="2025-01-01",
                end="2025-01-21",
                output=out,
                resume=True,
            )
        )

        assert result.success is True
        assert result.resumed is True
        assert result.rows == 20  # 10 existing + 10 new
        assert result.new_rows == 10

        # Verify the adapter was called with the adjusted start date.
        assert adapter.last_start == "2025-01-11"

    def test_resume_already_up_to_date(self, tmp_path: Path) -> None:
        """Resume when existing data already covers the full range returns early."""
        out = tmp_path / "complete.qcr"

        # Create a file with data through Jan 30.
        existing_df = _make_ohlcv_df(rows=30)
        _seal_test_file(out, existing_df)

        adapter = FakeAdapter()

        result = asyncio.run(
            ingest_to_qcr(
                adapter=adapter,
                ticker="TEST",
                start="2025-01-01",
                end="2025-01-20",  # Existing data goes to Jan 30.
                output=out,
                resume=True,
            )
        )

        assert result.success is True
        assert result.resumed is True
        assert result.new_rows == 0
        assert result.rows == 30
        # The adapter should NOT have been called since data is up to date.
        assert adapter.call_count == 0

    def test_resume_merges_and_deduplicates(self, tmp_path: Path) -> None:
        """Resumed data is merged with existing, duplicates removed."""
        out = tmp_path / "overlap.qcr"
        base = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)

        # Existing: 5 rows (Jan 1-5).
        existing_df = _make_ohlcv_df(rows=5, start_date=base)
        _seal_test_file(out, existing_df)

        # New data: 5 rows starting Jan 6 (no overlap, since resume adjusts start).
        new_data = _make_ohlcv_df(
            rows=5,
            start_date=base + timedelta(days=5),
            base_price=150.0,
        )
        adapter = FakeAdapter(df=new_data)

        result = asyncio.run(
            ingest_to_qcr(
                adapter=adapter,
                ticker="TEST",
                start="2025-01-01",
                end="2025-01-11",
                output=out,
                resume=True,
            )
        )

        assert result.success is True
        assert result.resumed is True
        assert result.rows == 10

    def test_resume_adapter_error_returns_failure(self, tmp_path: Path) -> None:
        """If the adapter fails during resume, the result indicates failure."""
        out = tmp_path / "fail_resume.qcr"

        # Create existing file.
        existing_df = _make_ohlcv_df(rows=5)
        _seal_test_file(out, existing_df)

        adapter = FakeAdapter(error="Network timeout")

        result = asyncio.run(
            ingest_to_qcr(
                adapter=adapter,
                ticker="TEST",
                start="2025-01-01",
                end="2025-02-01",
                output=out,
                resume=True,
            )
        )

        assert result.success is False
        assert result.resumed is True
        assert "Network timeout" in result.error

    def test_resume_empty_fetch_reseals_existing(self, tmp_path: Path) -> None:
        """If the adapter returns empty data during resume, existing data is re-sealed."""
        out = tmp_path / "reseal.qcr"

        existing_df = _make_ohlcv_df(rows=10)
        _seal_test_file(out, existing_df)

        empty_df = pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("ns", "UTC"),
                "open": pl.Float32,
                "high": pl.Float32,
                "low": pl.Float32,
                "close": pl.Float32,
                "volume": pl.UInt64,
            },
        )
        adapter = FakeAdapter(df=empty_df)

        result = asyncio.run(
            ingest_to_qcr(
                adapter=adapter,
                ticker="TEST",
                start="2025-01-01",
                end="2025-02-01",
                output=out,
                resume=True,
            )
        )

        assert result.success is True
        assert result.resumed is True
        assert result.rows == 10
        assert result.new_rows == 0

    def test_resume_preserves_data_integrity(self, tmp_path: Path) -> None:
        """After resume, the sealed file loads correctly with all data."""
        out = tmp_path / "integrity.qcr"
        base = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)

        # Phase 1: initial download (10 rows).
        initial_df = _make_ohlcv_df(rows=10, start_date=base)
        _seal_test_file(out, initial_df)

        # Phase 2: resume with 10 more rows.
        new_data = _make_ohlcv_df(
            rows=10,
            start_date=base + timedelta(days=10),
            base_price=110.0,
        )
        adapter = FakeAdapter(df=new_data)

        asyncio.run(
            ingest_to_qcr(
                adapter=adapter,
                ticker="TEST",
                start="2025-01-01",
                end="2025-01-21",
                output=out,
                resume=True,
            )
        )

        # Verify the file is loadable and has all data.
        df, meta = load_qcr(out)
        assert df.shape[0] == 20
        assert meta.identity.ticker == "TEST"
        assert meta.audit is not None
        assert meta.audit.audit_passed is True

        # Verify timestamps are contiguous and sorted.
        timestamps = df["timestamp"].to_list()
        assert timestamps == sorted(timestamps)
        assert len(set(timestamps)) == 20  # No duplicates.

    def test_resume_false_overwrites_existing(self, tmp_path: Path) -> None:
        """Without resume, an existing file is overwritten entirely."""
        out = tmp_path / "overwrite.qcr"

        # Create existing file with 10 rows.
        existing_df = _make_ohlcv_df(rows=10)
        _seal_test_file(out, existing_df)

        # Ingest without resume — should overwrite.
        new_data = _make_ohlcv_df(rows=5)
        adapter = FakeAdapter(df=new_data)

        result = asyncio.run(
            ingest_to_qcr(
                adapter=adapter,
                ticker="TEST",
                start="2025-01-01",
                end="2025-01-06",
                output=out,
                resume=False,
            )
        )

        assert result.success is True
        assert result.resumed is False
        assert result.rows == 5  # Only the new data, not merged.

    def test_multiple_resume_cycles(self, tmp_path: Path) -> None:
        """Simulates multiple resume cycles building up data incrementally."""
        out = tmp_path / "multi.qcr"
        base = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)

        # Cycle 1: fresh download (5 rows).
        adapter1 = FakeAdapter(df=_make_ohlcv_df(rows=5, start_date=base))
        r1 = asyncio.run(
            ingest_to_qcr(
                adapter=adapter1,
                ticker="TEST",
                start="2025-01-01",
                end="2025-02-01",
                output=out,
                resume=False,
            )
        )
        assert r1.success is True
        assert r1.rows == 5

        # Cycle 2: resume with 5 more rows.
        adapter2 = FakeAdapter(
            df=_make_ohlcv_df(rows=5, start_date=base + timedelta(days=5), base_price=110.0)
        )
        r2 = asyncio.run(
            ingest_to_qcr(
                adapter=adapter2,
                ticker="TEST",
                start="2025-01-01",
                end="2025-02-01",
                output=out,
                resume=True,
            )
        )
        assert r2.success is True
        assert r2.resumed is True
        assert r2.rows == 10
        assert r2.new_rows == 5

        # Cycle 3: resume with 5 more rows.
        adapter3 = FakeAdapter(
            df=_make_ohlcv_df(rows=5, start_date=base + timedelta(days=10), base_price=120.0)
        )
        r3 = asyncio.run(
            ingest_to_qcr(
                adapter=adapter3,
                ticker="TEST",
                start="2025-01-01",
                end="2025-02-01",
                output=out,
                resume=True,
            )
        )
        assert r3.success is True
        assert r3.resumed is True
        assert r3.rows == 15
        assert r3.new_rows == 5

        # Final verification.
        df, meta = load_qcr(out)
        assert df.shape[0] == 15
        timestamps = df["timestamp"].to_list()
        assert len(set(timestamps)) == 15


# ===========================================================================
# Tests for IngestResult dataclass
# ===========================================================================

class TestIngestResult:
    """Tests for the ``IngestResult`` dataclass fields."""

    def test_has_resume_fields(self) -> None:
        """IngestResult has the ``resumed`` and ``new_rows`` fields."""
        r = IngestResult(success=True, rows=100, resumed=True, new_rows=50)
        assert r.resumed is True
        assert r.new_rows == 50

    def test_defaults(self) -> None:
        """IngestResult defaults ``resumed`` to False and ``new_rows`` to 0."""
        r = IngestResult(success=True, rows=100)
        assert r.resumed is False
        assert r.new_rows == 0

    def test_frozen(self) -> None:
        """IngestResult is frozen (immutable)."""
        r = IngestResult(success=True, rows=100)
        with pytest.raises(AttributeError):
            r.rows = 200  # type: ignore[misc]


# ===========================================================================
# Tests for CLI --resume flag
# ===========================================================================

class TestCliResume:
    """Tests for the ``qcr ingest --resume`` CLI flag."""

    def test_resume_flag_in_help(self) -> None:
        """The --resume flag appears in the ingest help text."""
        from typer.testing import CliRunner
        from qcr.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "--resume" in result.output

    def test_resume_short_flag_in_help(self) -> None:
        """The -R short flag appears in the ingest help text."""
        from typer.testing import CliRunner
        from qcr.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "-R" in result.output
