"""Tests for the qcr.vtable module — Phase 6 Virtual Table Interface tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from qcr.auditor import run_audit
from qcr.schema import AssetClass, QcrMetadata, Timescale
from qcr.storage import save_sealed_qcr
from qcr.vtable import (
    OutputFormat,
    QcrQueryResult,
    VTableError,
    describe_qcr,
    execute_sql,
    format_result,
    query_qcr,
    query_qcr_df,
    register_qcr,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sealed_qcr(path: Path, ticker: str = "AAPL", rows: int = 20) -> Path:
    """Create a sealed .qcr file for testing."""
    base = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    timestamps = [base + timedelta(days=i) for i in range(rows)]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [float(100 + i) for i in range(rows)],
            "high": [float(110 + i) for i in range(rows)],
            "low": [float(95 + i) for i in range(rows)],
            "close": [float(105 + i) for i in range(rows)],
            "volume": [1_000_000 + i * 50_000 for i in range(rows)],
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
    meta = QcrMetadata(
        ticker=ticker,
        asset_class=AssetClass.EQUITY,
        currency="USD",
        exchange="XNAS",
        timezone="America/New_York",
        timescale=Timescale.ONE_DAY,
    )
    audit = run_audit(df, Timescale.ONE_DAY)
    qcr_path = path / f"{ticker}.qcr"
    save_sealed_qcr(df, meta, audit.to_audit_trail(), qcr_path)
    return qcr_path


# ---------------------------------------------------------------------------
# QcrQueryResult tests
# ---------------------------------------------------------------------------

class TestQcrQueryResult:
    """Tests for the QcrQueryResult dataclass."""

    def test_basic_creation(self):
        """QcrQueryResult can be created with required fields."""
        r = QcrQueryResult(columns=["a", "b"], rows=[(1, 2)], row_count=1)
        assert r.columns == ["a", "b"]
        assert r.rows == [(1, 2)]
        assert r.row_count == 1
        assert r.column_types == []

    def test_frozen(self):
        """QcrQueryResult is immutable."""
        r = QcrQueryResult(columns=[], rows=[], row_count=0)
        with pytest.raises(AttributeError):
            r.row_count = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# query_qcr — one-shot SQL queries
# ---------------------------------------------------------------------------

class TestQueryQcr:
    """Tests for the query_qcr one-shot function."""

    def test_select_all(self, tmp_path):
        """SELECT * returns all rows from a .qcr file."""
        qcr = _make_sealed_qcr(tmp_path)
        result = query_qcr(f"SELECT * FROM '{qcr}'")
        assert result.row_count == 20
        assert "timestamp" in result.columns
        assert "close" in result.columns

    def test_where_filter(self, tmp_path):
        """WHERE clause filters rows correctly."""
        qcr = _make_sealed_qcr(tmp_path)
        result = query_qcr(f"SELECT * FROM '{qcr}' WHERE close > 115")
        # close values are 105..124, so close > 115 means rows 11..19 → 9 rows
        assert result.row_count == 9

    def test_aggregation(self, tmp_path):
        """Aggregate functions (avg, sum, count) work."""
        qcr = _make_sealed_qcr(tmp_path)
        result = query_qcr(f"SELECT count(*) AS cnt, avg(close) AS avg_close FROM '{qcr}'")
        assert result.row_count == 1
        assert result.columns == ["cnt", "avg_close"]
        assert result.rows[0][0] == 20  # count

    def test_order_by(self, tmp_path):
        """ORDER BY sorts results."""
        qcr = _make_sealed_qcr(tmp_path)
        result = query_qcr(f"SELECT close FROM '{qcr}' ORDER BY close DESC LIMIT 3")
        assert result.row_count == 3
        closes = [row[0] for row in result.rows]
        assert closes == sorted(closes, reverse=True)

    def test_limit(self, tmp_path):
        """LIMIT restricts result count."""
        qcr = _make_sealed_qcr(tmp_path)
        result = query_qcr(f"SELECT * FROM '{qcr}' LIMIT 5")
        assert result.row_count == 5

    def test_column_selection(self, tmp_path):
        """Selecting specific columns returns only those columns."""
        qcr = _make_sealed_qcr(tmp_path)
        result = query_qcr(f"SELECT close, volume FROM '{qcr}' LIMIT 1")
        assert result.columns == ["close", "volume"]
        assert len(result.rows[0]) == 2

    def test_group_by(self, tmp_path):
        """GROUP BY works on .qcr data."""
        qcr = _make_sealed_qcr(tmp_path)
        result = query_qcr(
            f"SELECT count(*) AS cnt FROM '{qcr}' "
            f"GROUP BY (volume > 1500000)"
        )
        assert result.row_count >= 1

    def test_invalid_sql_raises(self, tmp_path):
        """Invalid SQL raises VTableError."""
        qcr = _make_sealed_qcr(tmp_path)
        with pytest.raises(VTableError, match="SQL error"):
            query_qcr(f"SELCT * FORM '{qcr}'")

    def test_missing_file_raises(self):
        """Referencing a nonexistent file raises an error."""
        with pytest.raises(VTableError):
            query_qcr("SELECT * FROM 'nonexistent.qcr'")

    def test_column_types_populated(self, tmp_path):
        """column_types field is populated in the result."""
        qcr = _make_sealed_qcr(tmp_path)
        result = query_qcr(f"SELECT close, volume FROM '{qcr}' LIMIT 1")
        assert len(result.column_types) == 2


# ---------------------------------------------------------------------------
# query_qcr_df — SQL to Polars DataFrame
# ---------------------------------------------------------------------------

class TestQueryQcrDf:
    """Tests for query_qcr_df returning Polars DataFrames."""

    def test_returns_polars_dataframe(self, tmp_path):
        """Result is a Polars DataFrame."""
        qcr = _make_sealed_qcr(tmp_path)
        df = query_qcr_df(f"SELECT * FROM '{qcr}'")
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 20

    def test_filtered_dataframe(self, tmp_path):
        """Filtered query returns correct DataFrame."""
        qcr = _make_sealed_qcr(tmp_path)
        df = query_qcr_df(f"SELECT close, volume FROM '{qcr}' WHERE volume > 1500000")
        assert isinstance(df, pl.DataFrame)
        assert "close" in df.columns
        assert "volume" in df.columns
        assert all(v > 1_500_000 for v in df["volume"].to_list())

    def test_aggregated_dataframe(self, tmp_path):
        """Aggregation query returns a single-row DataFrame."""
        qcr = _make_sealed_qcr(tmp_path)
        df = query_qcr_df(f"SELECT avg(close) AS avg_close FROM '{qcr}'")
        assert df.shape == (1, 1)
        assert "avg_close" in df.columns


# ---------------------------------------------------------------------------
# register_qcr — named virtual tables
# ---------------------------------------------------------------------------

class TestRegisterQcr:
    """Tests for register_qcr and execute_sql."""

    def test_register_and_query(self, tmp_path):
        """A registered .qcr file can be queried by name."""
        qcr = _make_sealed_qcr(tmp_path)
        con = register_qcr(qcr, table_name="stocks")
        result = execute_sql(con, "SELECT count(*) FROM stocks")
        assert result.rows[0][0] == 20

    def test_default_table_name(self, tmp_path):
        """Without table_name, the file stem is used."""
        qcr = _make_sealed_qcr(tmp_path, ticker="TSLA")
        con = register_qcr(qcr)
        result = execute_sql(con, "SELECT count(*) FROM TSLA")
        assert result.rows[0][0] == 20

    def test_multiple_registrations(self, tmp_path):
        """Multiple .qcr files can be registered on the same connection."""
        aapl = _make_sealed_qcr(tmp_path, ticker="AAPL")
        tsla_path = tmp_path / "sub"
        tsla_path.mkdir()
        tsla = _make_sealed_qcr(tsla_path, ticker="TSLA", rows=10)

        con = register_qcr(aapl, table_name="aapl")
        register_qcr(tsla, table_name="tsla", connection=con)

        r1 = execute_sql(con, "SELECT count(*) FROM aapl")
        r2 = execute_sql(con, "SELECT count(*) FROM tsla")
        assert r1.rows[0][0] == 20
        assert r2.rows[0][0] == 10

    def test_join_across_files(self, tmp_path):
        """Two registered .qcr files can be JOINed."""
        aapl = _make_sealed_qcr(tmp_path, ticker="AAPL")
        tsla_path = tmp_path / "sub"
        tsla_path.mkdir()
        tsla = _make_sealed_qcr(tsla_path, ticker="TSLA")

        con = register_qcr(aapl, table_name="aapl")
        register_qcr(tsla, table_name="tsla", connection=con)

        result = execute_sql(
            con,
            "SELECT a.close AS aapl_close, t.close AS tsla_close "
            "FROM aapl a JOIN tsla t ON a.timestamp = t.timestamp "
            "LIMIT 5"
        )
        assert result.row_count == 5
        assert "aapl_close" in result.columns
        assert "tsla_close" in result.columns

    def test_register_nonexistent_file(self, tmp_path):
        """Registering a nonexistent file raises VTableError."""
        with pytest.raises(VTableError, match="File not found"):
            register_qcr(tmp_path / "ghost.qcr")

    def test_register_non_qcr_file(self, tmp_path):
        """Registering a non-.qcr file raises VTableError."""
        txt = tmp_path / "data.txt"
        txt.write_text("hello")
        with pytest.raises(VTableError, match="Expected a .qcr file"):
            register_qcr(txt)


# ---------------------------------------------------------------------------
# describe_qcr
# ---------------------------------------------------------------------------

class TestDescribeQcr:
    """Tests for the describe_qcr function."""

    def test_basic_describe(self, tmp_path):
        """describe_qcr returns file info, columns, and metadata."""
        qcr = _make_sealed_qcr(tmp_path)
        info = describe_qcr(qcr)

        assert info["row_count"] == 20
        assert "timestamp" in info["columns"]
        assert "close" in info["columns"]
        assert "volume" in info["columns"]
        assert "identity" in info["metadata"]

    def test_metadata_contains_ticker(self, tmp_path):
        """The metadata dict includes the ticker."""
        qcr = _make_sealed_qcr(tmp_path, ticker="GOOG")
        info = describe_qcr(qcr)
        assert info["metadata"]["identity"]["ticker"] == "GOOG"

    def test_describe_nonexistent(self, tmp_path):
        """Describing a nonexistent file raises VTableError."""
        with pytest.raises(VTableError, match="File not found"):
            describe_qcr(tmp_path / "nope.qcr")


# ---------------------------------------------------------------------------
# format_result
# ---------------------------------------------------------------------------

class TestFormatResult:
    """Tests for the format_result output formatting."""

    def _sample_result(self) -> QcrQueryResult:
        return QcrQueryResult(
            columns=["ticker", "close", "volume"],
            rows=[("AAPL", 150.5, 1_000_000), ("TSLA", 200.3, 2_000_000)],
            row_count=2,
            column_types=["VARCHAR", "FLOAT", "BIGINT"],
        )

    def test_table_format(self):
        """Table format produces readable ASCII output."""
        result = self._sample_result()
        text = format_result(result, OutputFormat.TABLE)
        assert "ticker" in text
        assert "AAPL" in text
        assert "TSLA" in text
        assert "2 rows" in text

    def test_csv_format(self):
        """CSV format produces comma-separated output."""
        result = self._sample_result()
        text = format_result(result, OutputFormat.CSV)
        lines = text.strip().split("\n")
        assert lines[0] == "ticker,close,volume"
        assert "AAPL" in lines[1]

    def test_json_format(self):
        """JSON format produces valid JSON array."""
        import json

        result = self._sample_result()
        text = format_result(result, OutputFormat.JSON)
        data = json.loads(text)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["ticker"] == "AAPL"

    def test_empty_result_table(self):
        """Empty result produces a placeholder message."""
        result = QcrQueryResult(columns=[], rows=[], row_count=0)
        text = format_result(result, OutputFormat.TABLE)
        assert "empty" in text.lower()

    def test_single_row_grammar(self):
        """Single row uses '1 row' not '1 rows'."""
        result = QcrQueryResult(
            columns=["x"], rows=[(1,)], row_count=1,
        )
        text = format_result(result, OutputFormat.TABLE)
        assert "1 row)" in text


# ---------------------------------------------------------------------------
# CLI sql command tests
# ---------------------------------------------------------------------------

class TestCliSql:
    """Tests for the `qcr sql` CLI command."""

    def test_sql_command_exists(self):
        """The sql command is registered in the CLI app."""
        from typer.testing import CliRunner

        from qcr.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["sql", "--help"])
        assert result.exit_code == 0
        assert "Query .qcr files with SQL" in result.output

    def test_sql_select_all(self, tmp_path):
        """SQL SELECT * via the CLI returns data."""
        from typer.testing import CliRunner

        from qcr.cli import app

        qcr = _make_sealed_qcr(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, [
            "sql", f"SELECT count(*) AS cnt FROM '{qcr}'"
        ])
        assert result.exit_code == 0
        assert "20" in result.output

    def test_sql_csv_format(self, tmp_path):
        """--format csv outputs CSV."""
        from typer.testing import CliRunner

        from qcr.cli import app

        qcr = _make_sealed_qcr(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, [
            "sql", f"SELECT close, volume FROM '{qcr}' LIMIT 2",
            "--format", "csv",
        ])
        assert result.exit_code == 0
        assert "close,volume" in result.output

    def test_sql_json_format(self, tmp_path):
        """--format json outputs JSON."""
        import json

        from typer.testing import CliRunner

        from qcr.cli import app

        qcr = _make_sealed_qcr(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, [
            "sql", f"SELECT close FROM '{qcr}' LIMIT 1",
            "--format", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_sql_with_limit_flag(self, tmp_path):
        """--limit N appends LIMIT to the query."""
        from typer.testing import CliRunner

        from qcr.cli import app

        qcr = _make_sealed_qcr(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, [
            "sql", f"SELECT * FROM '{qcr}'",
            "--limit", "3",
        ])
        assert result.exit_code == 0
        assert "3 rows" in result.output

    def test_sql_register_flag(self, tmp_path):
        """--register FILE=NAME allows querying by table name."""
        from typer.testing import CliRunner

        from qcr.cli import app

        qcr = _make_sealed_qcr(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, [
            "sql", "SELECT count(*) AS cnt FROM stocks",
            "--register", f"{qcr}=stocks",
        ])
        assert result.exit_code == 0
        assert "20" in result.output

    def test_sql_describe_flag(self, tmp_path):
        """--describe shows file schema and metadata."""
        from typer.testing import CliRunner

        from qcr.cli import app

        qcr = _make_sealed_qcr(tmp_path)
        runner = CliRunner()
        # Pass empty string as the query argument when using --describe.
        result = runner.invoke(app, [
            "sql", "",
            "--describe", str(qcr),
        ])
        assert result.exit_code == 0
        assert "Rows" in result.output or "20" in result.output

    def test_sql_invalid_format(self, tmp_path):
        """An invalid --format exits with code 1."""
        from typer.testing import CliRunner

        from qcr.cli import app

        qcr = _make_sealed_qcr(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, [
            "sql", f"SELECT * FROM '{qcr}'",
            "--format", "xml",
        ])
        assert result.exit_code == 1

    def test_sql_bad_query(self, tmp_path):
        """A malformed SQL query exits with code 1."""
        from typer.testing import CliRunner

        from qcr.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "sql", "SELCT * FORM nowhere",
        ])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    """Tests for .qcr file validation in the vtable module."""

    def test_validate_non_qcr_extension(self, tmp_path):
        """Files without .qcr extension are rejected by register_qcr."""
        fake = tmp_path / "data.parquet"
        fake.write_bytes(b"PAR1")
        with pytest.raises(VTableError, match="Expected a .qcr file"):
            register_qcr(fake)

    def test_validate_missing_metadata(self, tmp_path):
        """A plain Parquet file without qcr_metadata is rejected."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Write a plain Parquet file with .qcr extension.
        table = pa.table({"x": [1, 2, 3]})
        fake_qcr = tmp_path / "fake.qcr"
        pq.write_table(table, str(fake_qcr))

        with pytest.raises(VTableError, match="not a valid QuantCrate"):
            register_qcr(fake_qcr)
