"""Tests for the qcr.adapters and qcr.ingest modules — Phase 5 data adapter tests."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from qcr.adapters import AdapterError, BaseAdapter, YahooFinanceAdapter
from qcr.ingest import IngestResult, ingest_to_qcr
from qcr.schema import AssetClass, Timescale
from qcr.storage import load_qcr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(rows: int = 30, timescale: Timescale = Timescale.ONE_DAY) -> pl.DataFrame:
    """Create a valid OHLCV DataFrame that passes all audit checks."""
    base = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
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
            "open": [float(100 + i * 0.5) for i in range(rows)],
            "high": [float(105 + i * 0.5) for i in range(rows)],
            "low": [float(98 + i * 0.5) for i in range(rows)],
            "close": [float(102 + i * 0.5) for i in range(rows)],
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


class FakeAdapter(BaseAdapter):
    """A concrete adapter for testing that returns pre-configured data."""

    def __init__(
        self,
        df: Optional[pl.DataFrame] = None,
        error: Optional[str] = None,
    ) -> None:
        self._df = df
        self._error = error

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
        if self._error:
            raise AdapterError(self._error)
        if self._df is not None:
            return self._df
        return _make_ohlcv_df()


# ---------------------------------------------------------------------------
# BaseAdapter interface tests
# ---------------------------------------------------------------------------

class TestBaseAdapter:
    """Tests for the BaseAdapter abstract class."""

    def test_cannot_instantiate_directly(self):
        """BaseAdapter is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseAdapter()  # type: ignore[abstract]

    def test_subclass_must_implement_fetch(self):
        """A subclass that doesn't implement fetch_ohlcv raises TypeError."""

        class Incomplete(BaseAdapter):
            @property
            def source_name(self) -> str:
                return "Incomplete"

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_must_implement_source_name(self):
        """A subclass that doesn't implement source_name raises TypeError."""

        class Incomplete(BaseAdapter):
            async def fetch_ohlcv(
                self, ticker: str, start: str, end: str,
                timescale: Timescale = Timescale.ONE_DAY,
            ) -> pl.DataFrame:
                return pl.DataFrame()

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        """A fully implemented subclass can be instantiated."""
        adapter = FakeAdapter()
        assert adapter.source_name == "FakeSource"

    def test_fake_adapter_returns_data(self):
        """FakeAdapter.fetch_ohlcv returns the expected DataFrame."""
        adapter = FakeAdapter()
        df = asyncio.run(adapter.fetch_ohlcv("TEST", "2025-01-01", "2025-02-01"))
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 30

    def test_fake_adapter_raises_on_error(self):
        """FakeAdapter raises AdapterError when configured to fail."""
        adapter = FakeAdapter(error="Network timeout")
        with pytest.raises(AdapterError, match="Network timeout"):
            asyncio.run(adapter.fetch_ohlcv("TEST", "2025-01-01", "2025-02-01"))


# ---------------------------------------------------------------------------
# YahooFinanceAdapter tests (mocked — no network calls)
# ---------------------------------------------------------------------------

class TestYahooFinanceAdapter:
    """Tests for the YahooFinanceAdapter (yfinance is mocked)."""

    def test_source_name(self):
        """source_name returns 'YahooFinance'."""
        adapter = YahooFinanceAdapter()
        assert adapter.source_name == "YahooFinance"

    def test_unsupported_timescale_raises(self):
        """Timescales not supported by yfinance raise AdapterError."""
        adapter = YahooFinanceAdapter()
        with pytest.raises(AdapterError, match="not supported"):
            asyncio.run(
                adapter.fetch_ohlcv("AAPL", "2025-01-01", "2025-02-01", Timescale.TICK)
            )

    def test_unsupported_1s_timescale_raises(self):
        """1-second timescale is not supported by yfinance."""
        adapter = YahooFinanceAdapter()
        with pytest.raises(AdapterError, match="not supported"):
            asyncio.run(
                adapter.fetch_ohlcv("AAPL", "2025-01-01", "2025-02-01", Timescale.ONE_SECOND)
            )

    def test_download_called_correctly(self):
        """The yfinance.download function is called with the right args."""
        import pandas as pd

        # Create a fake pandas DataFrame that yfinance would return.
        dates = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
        fake_pdf = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "High": [105.0, 106.0, 107.0, 108.0, 109.0],
                "Low": [98.0, 99.0, 100.0, 101.0, 102.0],
                "Close": [102.0, 103.0, 104.0, 105.0, 106.0],
                "Volume": [1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000],
            },
            index=dates,
        )
        fake_pdf.index.name = "Date"

        with patch("qcr.adapters.yf") as mock_yf:
            mock_yf.download.return_value = fake_pdf
            adapter = YahooFinanceAdapter()
            df = asyncio.run(
                adapter.fetch_ohlcv("AAPL", "2025-01-01", "2025-01-06")
            )

        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 5
        assert set(df.columns) == {"timestamp", "open", "high", "low", "close", "volume"}
        assert df["open"].dtype == pl.Float32
        assert df["volume"].dtype == pl.UInt64

    def test_empty_response_raises(self):
        """An empty response from yfinance raises AdapterError."""
        import pandas as pd

        with patch("qcr.adapters.yf") as mock_yf:
            mock_yf.download.return_value = pd.DataFrame()
            adapter = YahooFinanceAdapter()
            with pytest.raises(AdapterError, match="no data"):
                asyncio.run(
                    adapter.fetch_ohlcv("INVALID", "2025-01-01", "2025-02-01")
                )

    def test_network_error_raises_adapter_error(self):
        """A network exception is wrapped in AdapterError."""
        with patch("qcr.adapters.yf") as mock_yf:
            mock_yf.download.side_effect = ConnectionError("DNS resolution failed")
            adapter = YahooFinanceAdapter()
            with pytest.raises(AdapterError, match="Failed to fetch"):
                asyncio.run(
                    adapter.fetch_ohlcv("AAPL", "2025-01-01", "2025-02-01")
                )

    def test_auto_adjust_passed_through(self):
        """auto_adjust flag is forwarded to yfinance.download."""
        import pandas as pd

        dates = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
        fake_pdf = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [105.0, 106.0, 107.0],
                "Low": [98.0, 99.0, 100.0],
                "Close": [102.0, 103.0, 104.0],
                "Volume": [1_000_000, 1_100_000, 1_200_000],
            },
            index=dates,
        )
        fake_pdf.index.name = "Date"

        with patch("qcr.adapters.yf") as mock_yf:
            mock_yf.download.return_value = fake_pdf
            adapter = YahooFinanceAdapter(auto_adjust=False)
            asyncio.run(adapter.fetch_ohlcv("AAPL", "2025-01-01", "2025-01-04"))

            call_kwargs = mock_yf.download.call_args
            assert call_kwargs[1]["auto_adjust"] is False


# ---------------------------------------------------------------------------
# ingest_to_qcr tests
# ---------------------------------------------------------------------------

class TestIngestToQcr:
    """Tests for the ingest_to_qcr async orchestrator."""

    def test_successful_ingest(self, tmp_path):
        """A clean fetch → audit → seal pipeline produces a .qcr file."""
        out = tmp_path / "test.qcr"
        adapter = FakeAdapter()

        result = asyncio.run(
            ingest_to_qcr(
                adapter=adapter,
                ticker="TEST",
                start="2025-01-01",
                end="2025-02-01",
                output=out,
            )
        )

        assert result.success is True
        assert result.path is not None
        assert result.path.exists()
        assert result.rows == 30
        assert result.audit is not None
        assert result.audit.passed is True

    def test_sealed_file_is_loadable(self, tmp_path):
        """The produced .qcr file can be loaded with load_qcr."""
        out = tmp_path / "loadable.qcr"
        adapter = FakeAdapter()

        asyncio.run(
            ingest_to_qcr(
                adapter=adapter,
                ticker="AAPL",
                start="2025-01-01",
                end="2025-02-01",
                output=out,
            )
        )

        df, meta = load_qcr(out)
        assert meta.identity.ticker == "AAPL"
        assert df.shape[0] == 30

    def test_metadata_is_correct(self, tmp_path):
        """All metadata fields are correctly embedded."""
        out = tmp_path / "meta.qcr"
        adapter = FakeAdapter()

        asyncio.run(
            ingest_to_qcr(
                adapter=adapter,
                ticker="BTC",
                start="2025-01-01",
                end="2025-02-01",
                asset_class=AssetClass.CRYPTO,
                currency="BTC",
                exchange="BINANCE",
                timezone="UTC",
                timescale=Timescale.ONE_DAY,
                output=out,
            )
        )

        _, meta = load_qcr(out)
        assert meta.identity.ticker == "BTC"
        assert meta.identity.asset_class == AssetClass.CRYPTO
        assert meta.identity.currency == "BTC"
        assert meta.identity.exchange == "BINANCE"

    def test_audit_trail_is_present(self, tmp_path):
        """The sealed file contains an audit trail."""
        out = tmp_path / "audited.qcr"
        adapter = FakeAdapter()

        asyncio.run(
            ingest_to_qcr(
                adapter=adapter,
                ticker="AAPL",
                start="2025-01-01",
                end="2025-02-01",
                output=out,
            )
        )

        _, meta = load_qcr(out)
        assert meta.audit is not None
        assert meta.audit.audit_passed is True

    def test_adapter_error_returns_failure(self, tmp_path):
        """An adapter error produces a failed IngestResult."""
        adapter = FakeAdapter(error="Connection refused")

        result = asyncio.run(
            ingest_to_qcr(
                adapter=adapter,
                ticker="FAIL",
                start="2025-01-01",
                end="2025-02-01",
                output=tmp_path / "fail.qcr",
            )
        )

        assert result.success is False
        assert "Connection refused" in result.error
        assert result.path is None

    def test_empty_data_returns_failure(self, tmp_path):
        """An empty DataFrame from the adapter produces a failed result."""
        empty_df = pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
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
        adapter = FakeAdapter(df=empty_df)

        result = asyncio.run(
            ingest_to_qcr(
                adapter=adapter,
                ticker="EMPTY",
                start="2025-01-01",
                end="2025-02-01",
                output=tmp_path / "empty.qcr",
            )
        )

        assert result.success is False
        assert "empty" in result.error.lower()

    def test_bad_data_fails_audit(self, tmp_path):
        """Data with logical errors fails the audit and is not sealed."""
        # Create data where high < low (audit error).
        base = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
        bad_df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(days=i) for i in range(10)],
                "open": [100.0] * 10,
                "high": [90.0] * 10,   # high < open → error
                "low": [95.0] * 10,
                "close": [102.0] * 10,
                "volume": [1_000_000] * 10,
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
        adapter = FakeAdapter(df=bad_df)

        result = asyncio.run(
            ingest_to_qcr(
                adapter=adapter,
                ticker="BAD",
                start="2025-01-01",
                end="2025-02-01",
                output=tmp_path / "bad.qcr",
            )
        )

        assert result.success is False
        assert result.audit is not None
        assert result.audit.passed is False
        assert "Audit failed" in result.error

    def test_default_output_path(self):
        """Without an explicit output, the file is named <ticker>.qcr."""
        adapter = FakeAdapter()

        result = asyncio.run(
            ingest_to_qcr(
                adapter=adapter,
                ticker="GOOG",
                start="2025-01-01",
                end="2025-02-01",
            )
        )

        assert result.success is True
        assert result.path is not None
        assert result.path.name == "GOOG.qcr"

        # Clean up.
        if result.path.exists():
            result.path.unlink()

    def test_ingest_result_dataclass(self):
        """IngestResult is a frozen dataclass with expected fields."""
        r = IngestResult(success=True, rows=100)
        assert r.success is True
        assert r.rows == 100
        assert r.path is None
        assert r.audit is None
        assert r.error is None


# ---------------------------------------------------------------------------
# CLI ingest command tests
# ---------------------------------------------------------------------------

class TestCliIngest:
    """Tests for the `qcr ingest` CLI command (adapter is mocked)."""

    def test_ingest_command_exists(self):
        """The ingest command is registered in the CLI app."""
        from typer.testing import CliRunner

        from qcr.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "Fetch OHLCV data" in result.output

    def test_ingest_invalid_source(self, tmp_path):
        """An unknown source adapter exits with code 1."""
        from typer.testing import CliRunner

        from qcr.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "ingest", "AAPL",
            "--start", "2025-01-01",
            "--end", "2025-02-01",
            "--source", "bloomberg",
            "--output", str(tmp_path / "test.qcr"),
        ])
        assert result.exit_code == 1

    def test_ingest_invalid_timescale(self, tmp_path):
        """An invalid timescale exits with code 1."""
        from typer.testing import CliRunner

        from qcr.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "ingest", "AAPL",
            "--start", "2025-01-01",
            "--end", "2025-02-01",
            "--timescale", "99z",
            "--output", str(tmp_path / "test.qcr"),
        ])
        assert result.exit_code == 1

    def test_ingest_invalid_asset_class(self, tmp_path):
        """An invalid asset class exits with code 1."""
        from typer.testing import CliRunner

        from qcr.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "ingest", "AAPL",
            "--start", "2025-01-01",
            "--end", "2025-02-01",
            "--asset-class", "NotReal",
            "--output", str(tmp_path / "test.qcr"),
        ])
        assert result.exit_code == 1
