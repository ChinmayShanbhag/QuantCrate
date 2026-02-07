"""Tests for the qcr.storage module — Phase 1 round-trip tests."""

from datetime import datetime, timezone

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from qcr.schema import AssetClass, FullMetadata, QcrMetadata, Timescale
from qcr.storage import load_qcr, read_qcr_metadata, save_qcr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_METADATA = QcrMetadata(
    ticker="AAPL",
    asset_class=AssetClass.EQUITY,
    currency="USD",
    exchange="XNAS",
    timezone="America/New_York",
    timescale=Timescale.ONE_MINUTE,
    is_adjusted=False,
)


def _make_polars_df(rows: int = 5) -> pl.DataFrame:
    """Create a minimal valid Polars OHLCV DataFrame."""
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


def _make_pandas_df(rows: int = 5) -> pd.DataFrame:
    """Create a minimal valid Pandas OHLCV DataFrame."""
    timestamps = pd.date_range(
        "2025-01-01 09:30", periods=rows, freq="min", tz="UTC"
    )
    return pd.DataFrame(
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
# save_qcr tests
# ---------------------------------------------------------------------------

class TestSaveQcr:
    """Tests for the save_qcr function."""

    def test_save_creates_file(self, tmp_path):
        """Saving a valid DataFrame creates a .qcr file on disk."""
        out = tmp_path / "test.qcr"
        result = save_qcr(_make_polars_df(), SAMPLE_METADATA, out)
        assert result.exists()

    def test_save_accepts_polars(self, tmp_path):
        """save_qcr accepts a Polars DataFrame without error."""
        out = tmp_path / "polars.qcr"
        save_qcr(_make_polars_df(), SAMPLE_METADATA, out)
        assert out.exists()

    def test_save_accepts_pandas(self, tmp_path):
        """save_qcr accepts a Pandas DataFrame without error."""
        out = tmp_path / "pandas.qcr"
        save_qcr(_make_pandas_df(), SAMPLE_METADATA, out)
        assert out.exists()

    def test_save_rejects_missing_columns(self, tmp_path):
        """save_qcr raises ValueError when required columns are missing."""
        bad_df = pl.DataFrame({"timestamp": [1], "open": [1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            save_qcr(bad_df, SAMPLE_METADATA, tmp_path / "bad.qcr")

    def test_save_rejects_invalid_type(self, tmp_path):
        """save_qcr raises TypeError for unsupported input types."""
        with pytest.raises(TypeError, match="Expected a Pandas or Polars"):
            save_qcr({"not": "a dataframe"}, SAMPLE_METADATA, tmp_path / "bad.qcr")


# ---------------------------------------------------------------------------
# load_qcr tests
# ---------------------------------------------------------------------------

class TestLoadQcr:
    """Tests for the load_qcr function."""

    def test_round_trip_polars(self, tmp_path):
        """Data survives a save → load round trip (Polars input)."""
        out = tmp_path / "rt.qcr"
        original = _make_polars_df()
        save_qcr(original, SAMPLE_METADATA, out)

        loaded_df, loaded_meta = load_qcr(out)

        assert loaded_df.shape == original.shape
        assert loaded_meta.identity.ticker == "AAPL"
        assert loaded_meta.identity.asset_class == AssetClass.EQUITY

    def test_round_trip_pandas(self, tmp_path):
        """Data survives a save → load round trip (Pandas input)."""
        out = tmp_path / "rt_pd.qcr"
        original = _make_pandas_df()
        save_qcr(original, SAMPLE_METADATA, out)

        loaded_df, loaded_meta = load_qcr(out)

        assert loaded_df.shape[0] == len(original)
        assert loaded_meta.identity.currency == "USD"

    def test_load_returns_polars_dataframe(self, tmp_path):
        """load_qcr always returns a Polars DataFrame."""
        out = tmp_path / "type.qcr"
        save_qcr(_make_pandas_df(), SAMPLE_METADATA, out)

        loaded_df, _ = load_qcr(out)
        assert isinstance(loaded_df, pl.DataFrame)

    def test_load_preserves_column_types(self, tmp_path):
        """Loaded DataFrame has the correct Arrow-backed column types."""
        out = tmp_path / "types.qcr"
        save_qcr(_make_polars_df(), SAMPLE_METADATA, out)

        loaded_df, _ = load_qcr(out)

        assert loaded_df["open"].dtype == pl.Float32
        assert loaded_df["volume"].dtype == pl.UInt64

    def test_load_file_not_found(self):
        """load_qcr raises FileNotFoundError for a missing path."""
        with pytest.raises(FileNotFoundError):
            load_qcr("nonexistent.qcr")

    def test_metadata_identity_fields(self, tmp_path):
        """All identity metadata fields survive the round trip."""
        out = tmp_path / "meta.qcr"
        save_qcr(_make_polars_df(), SAMPLE_METADATA, out)

        _, meta = load_qcr(out)
        identity = meta.identity

        assert identity.ticker == "AAPL"
        assert identity.asset_class == AssetClass.EQUITY
        assert identity.currency == "USD"
        assert identity.exchange == "XNAS"
        assert identity.timezone == "America/New_York"
        assert identity.timescale == Timescale.ONE_MINUTE
        assert identity.is_adjusted is False
        assert identity.version == 1

    def test_audit_trail_is_none_before_sealing(self, tmp_path):
        """Before the Auditor runs, the audit trail should be None."""
        out = tmp_path / "no_audit.qcr"
        save_qcr(_make_polars_df(), SAMPLE_METADATA, out)

        _, meta = load_qcr(out)
        assert meta.audit is None


# ---------------------------------------------------------------------------
# read_qcr_metadata tests
# ---------------------------------------------------------------------------

class TestReadQcrMetadata:
    """Tests for the read_qcr_metadata (metadata-only read) function."""

    def test_reads_metadata_without_loading_data(self, tmp_path):
        """read_qcr_metadata returns metadata from the Parquet footer."""
        out = tmp_path / "info.qcr"
        save_qcr(_make_polars_df(), SAMPLE_METADATA, out)

        meta = read_qcr_metadata(out)

        assert isinstance(meta, FullMetadata)
        assert meta.identity.ticker == "AAPL"

    def test_metadata_only_file_not_found(self):
        """read_qcr_metadata raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            read_qcr_metadata("ghost.qcr")
