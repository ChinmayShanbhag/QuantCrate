"""Tests for the qcr.cli module — Phase 3 CLI tests."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from qcr.cli import app
from qcr.schema import AssetClass, FullMetadata, QcrMetadata, Timescale
from qcr.storage import load_qcr, save_qcr

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_clean_csv(path: Path, rows: int = 20) -> Path:
    """Write a valid OHLCV CSV file and return the path."""
    base = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i) for i in range(rows)]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [float(100 + i) for i in range(rows)],
            "high": [float(105 + i) for i in range(rows)],
            "low": [float(98 + i) for i in range(rows)],
            "close": [float(102 + i) for i in range(rows)],
            "volume": [1_000_000 + i * 100 for i in range(rows)],
        }
    )
    csv_path = path / "data.csv"
    df.write_csv(csv_path)
    return csv_path


def _write_bad_csv(path: Path) -> Path:
    """Write a CSV that will fail the audit (high < low)."""
    base = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i) for i in range(10)]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * 10,
            "high": [90.0] * 10,   # Bad: high < open, high < close
            "low": [95.0] * 10,
            "close": [102.0] * 10,
            "volume": [1_000_000] * 10,
        }
    )
    csv_path = path / "bad_data.csv"
    df.write_csv(csv_path)
    return csv_path


def _write_aliased_csv(path: Path, rows: int = 10) -> Path:
    """Write a CSV with common header aliases (Date, Vol, etc.)."""
    base = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i) for i in range(rows)]
    df = pl.DataFrame(
        {
            "Date": timestamps,
            "Open": [float(100 + i) for i in range(rows)],
            "High": [float(105 + i) for i in range(rows)],
            "Low": [float(98 + i) for i in range(rows)],
            "Close": [float(102 + i) for i in range(rows)],
            "Vol": [1_000_000 + i * 100 for i in range(rows)],
        }
    )
    csv_path = path / "aliased.csv"
    df.write_csv(csv_path)
    return csv_path


def _make_sealed_qcr(path: Path) -> Path:
    """Create a sealed .qcr file for info tests."""
    from qcr.auditor import run_audit
    from qcr.storage import save_sealed_qcr

    base = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i) for i in range(10)]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [float(100 + i) for i in range(10)],
            "high": [float(105 + i) for i in range(10)],
            "low": [float(98 + i) for i in range(10)],
            "close": [float(102 + i) for i in range(10)],
            "volume": [1_000_000 + i * 100 for i in range(10)],
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
        ticker="AAPL",
        asset_class=AssetClass.EQUITY,
        currency="USD",
        exchange="XNAS",
        timezone="America/New_York",
        timescale=Timescale.ONE_MINUTE,
    )
    result = run_audit(df, Timescale.ONE_MINUTE)
    qcr_path = path / "sealed.qcr"
    save_sealed_qcr(df, meta, result.to_audit_trail(), qcr_path)
    return qcr_path


# ---------------------------------------------------------------------------
# qcr pack
# ---------------------------------------------------------------------------

class TestPack:
    """Tests for the `qcr pack` command."""

    def test_pack_creates_qcr_file(self, tmp_path):
        """A valid CSV is packed into a .qcr file."""
        csv = _write_clean_csv(tmp_path)
        out = tmp_path / "test.qcr"

        result = runner.invoke(app, [
            "pack", str(csv),
            "--ticker", "AAPL",
            "--asset-class", "Equity",
            "--exchange", "XNAS",
            "--timescale", "1m",
            "--output", str(out),
        ])

        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_packed_file_is_loadable(self, tmp_path):
        """The output .qcr file can be loaded with load_qcr."""
        csv = _write_clean_csv(tmp_path)
        out = tmp_path / "loadable.qcr"

        runner.invoke(app, [
            "pack", str(csv),
            "--ticker", "TSLA",
            "--asset-class", "Equity",
            "--exchange", "XNAS",
            "--timescale", "1m",
            "--output", str(out),
        ])

        df, meta = load_qcr(out)
        assert meta.identity.ticker == "TSLA"
        assert df.shape[0] == 20

    def test_packed_file_has_audit_trail(self, tmp_path):
        """The sealed file contains an audit trail."""
        csv = _write_clean_csv(tmp_path)
        out = tmp_path / "audited.qcr"

        runner.invoke(app, [
            "pack", str(csv),
            "--ticker", "AAPL",
            "--asset-class", "Equity",
            "--exchange", "XNAS",
            "--timescale", "1m",
            "--output", str(out),
        ])

        _, meta = load_qcr(out)
        assert meta.audit is not None
        assert meta.audit.audit_passed is True

    def test_pack_fails_on_bad_data(self, tmp_path):
        """A CSV that fails the audit exits with code 1."""
        csv = _write_bad_csv(tmp_path)
        out = tmp_path / "bad.qcr"

        result = runner.invoke(app, [
            "pack", str(csv),
            "--ticker", "BAD",
            "--asset-class", "Equity",
            "--exchange", "XNAS",
            "--timescale", "1m",
            "--output", str(out),
        ])

        assert result.exit_code == 1
        assert not out.exists()

    def test_pack_invalid_asset_class(self, tmp_path):
        """An invalid asset class exits with code 1."""
        csv = _write_clean_csv(tmp_path)

        result = runner.invoke(app, [
            "pack", str(csv),
            "--ticker", "X",
            "--asset-class", "NotReal",
            "--exchange", "XNAS",
            "--timescale", "1m",
        ])

        assert result.exit_code == 1

    def test_pack_invalid_timescale(self, tmp_path):
        """An invalid timescale exits with code 1."""
        csv = _write_clean_csv(tmp_path)

        result = runner.invoke(app, [
            "pack", str(csv),
            "--ticker", "X",
            "--asset-class", "Equity",
            "--exchange", "XNAS",
            "--timescale", "99z",
        ])

        assert result.exit_code == 1

    def test_pack_default_output_name(self, tmp_path, monkeypatch):
        """Without --output, the file is named <ticker>.qcr in cwd."""
        csv = _write_clean_csv(tmp_path)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, [
            "pack", str(csv),
            "--ticker", "GOOG",
            "--asset-class", "Equity",
            "--exchange", "XNAS",
            "--timescale", "1m",
        ])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "GOOG.qcr").exists()

    def test_pack_handles_aliased_headers(self, tmp_path):
        """CSVs with 'Date', 'Vol' etc. are handled via alias mapping."""
        csv = _write_aliased_csv(tmp_path)
        out = tmp_path / "aliased.qcr"

        result = runner.invoke(app, [
            "pack", str(csv),
            "--ticker", "AAPL",
            "--asset-class", "Equity",
            "--exchange", "XNAS",
            "--timescale", "1m",
            "--output", str(out),
        ])

        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_pack_preserves_metadata_fields(self, tmp_path):
        """All CLI flags are correctly embedded in the .qcr metadata."""
        csv = _write_clean_csv(tmp_path)
        out = tmp_path / "meta.qcr"

        runner.invoke(app, [
            "pack", str(csv),
            "--ticker", "BTC",
            "--asset-class", "Crypto",
            "--currency", "BTC",
            "--exchange", "BINANCE",
            "--timezone", "UTC",
            "--timescale", "1h",
            "--output", str(out),
        ])

        _, meta = load_qcr(out)
        assert meta.identity.ticker == "BTC"
        assert meta.identity.asset_class == AssetClass.CRYPTO
        assert meta.identity.currency == "BTC"
        assert meta.identity.exchange == "BINANCE"
        assert meta.identity.timescale == Timescale.ONE_HOUR


# ---------------------------------------------------------------------------
# qcr info
# ---------------------------------------------------------------------------

class TestInfo:
    """Tests for the `qcr info` command."""

    def test_info_shows_ticker(self, tmp_path):
        """Info command displays the ticker."""
        qcr = _make_sealed_qcr(tmp_path)

        result = runner.invoke(app, ["info", str(qcr)])

        assert result.exit_code == 0
        assert "AAPL" in result.output

    def test_info_shows_asset_class(self, tmp_path):
        """Info command displays the asset class."""
        qcr = _make_sealed_qcr(tmp_path)

        result = runner.invoke(app, ["info", str(qcr)])

        assert "Equity" in result.output

    def test_info_shows_audit_status(self, tmp_path):
        """Info command displays audit pass/fail status."""
        qcr = _make_sealed_qcr(tmp_path)

        result = runner.invoke(app, ["info", str(qcr)])

        assert "PASSED" in result.output

    def test_info_unsealed_file(self, tmp_path):
        """Info on an unsealed file shows the 'no audit trail' warning."""
        meta = QcrMetadata(
            ticker="MSFT",
            asset_class=AssetClass.EQUITY,
            currency="USD",
            exchange="XNAS",
            timezone="America/New_York",
            timescale=Timescale.ONE_DAY,
        )
        base = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
        df = pl.DataFrame(
            {
                "timestamp": [base + timedelta(days=i) for i in range(5)],
                "open": [100.0 + i for i in range(5)],
                "high": [105.0 + i for i in range(5)],
                "low": [98.0 + i for i in range(5)],
                "close": [102.0 + i for i in range(5)],
                "volume": [1_000_000] * 5,
            }
        )
        qcr_path = tmp_path / "unsealed.qcr"
        save_qcr(df, meta, qcr_path)

        result = runner.invoke(app, ["info", str(qcr_path)])

        assert result.exit_code == 0
        assert "MSFT" in result.output
        assert "No audit trail" in result.output

    def test_info_shows_all_identity_fields(self, tmp_path):
        """Info command displays all identity metadata fields."""
        qcr = _make_sealed_qcr(tmp_path)

        result = runner.invoke(app, ["info", str(qcr)])

        assert "USD" in result.output
        assert "XNAS" in result.output
        assert "1m" in result.output
