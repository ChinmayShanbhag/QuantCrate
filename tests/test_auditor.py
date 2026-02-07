"""Tests for the qcr.auditor module — Phase 2 financial linter tests."""

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from qcr.auditor import (
    AuditResult,
    Severity,
    check_chronology,
    check_logical_consistency,
    detect_outliers,
    identify_gaps,
    run_audit,
)
from qcr.schema import Timescale


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clean_df(rows: int = 10) -> pl.DataFrame:
    """Create a valid OHLCV DataFrame that passes all audit checks."""
    base = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i) for i in range(rows)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [float(100 + i) for i in range(rows)],
            "high": [float(105 + i) for i in range(rows)],
            "low": [float(98 + i) for i in range(rows)],
            "close": [float(102 + i) for i in range(rows)],
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


# ---------------------------------------------------------------------------
# check_logical_consistency
# ---------------------------------------------------------------------------

class TestLogicalConsistency:
    """Tests for OHLC price relationship validation."""

    def test_clean_data_passes(self):
        """A well-formed DataFrame produces no issues."""
        issues = check_logical_consistency(_make_clean_df())
        assert issues == []

    def test_high_less_than_open(self):
        """Detects rows where high < open."""
        df = _make_clean_df()
        # Force high below open in row 3.
        df = df.with_columns(
            pl.when(pl.arange(0, pl.len()) == 3)
            .then(pl.lit(90.0).cast(pl.Float32))
            .otherwise(pl.col("high"))
            .alias("high")
        )
        issues = check_logical_consistency(df)
        messages = [i.message for i in issues]
        assert any("high < open" in m for m in messages)

    def test_high_less_than_close(self):
        """Detects rows where high < close."""
        df = _make_clean_df()
        df = df.with_columns(
            pl.when(pl.arange(0, pl.len()) == 2)
            .then(pl.lit(90.0).cast(pl.Float32))
            .otherwise(pl.col("high"))
            .alias("high")
        )
        issues = check_logical_consistency(df)
        messages = [i.message for i in issues]
        assert any("high < close" in m for m in messages)

    def test_high_less_than_low(self):
        """Detects rows where high < low."""
        df = _make_clean_df()
        df = df.with_columns(
            pl.when(pl.arange(0, pl.len()) == 1)
            .then(pl.lit(90.0).cast(pl.Float32))
            .otherwise(pl.col("high"))
            .alias("high")
        )
        issues = check_logical_consistency(df)
        messages = [i.message for i in issues]
        assert any("high < low" in m for m in messages)

    def test_low_greater_than_open(self):
        """Detects rows where low > open."""
        df = _make_clean_df()
        df = df.with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(pl.lit(200.0).cast(pl.Float32))
            .otherwise(pl.col("low"))
            .alias("low")
        )
        issues = check_logical_consistency(df)
        messages = [i.message for i in issues]
        assert any("low > open" in m for m in messages)

    def test_low_greater_than_close(self):
        """Detects rows where low > close."""
        df = _make_clean_df()
        df = df.with_columns(
            pl.when(pl.arange(0, pl.len()) == 4)
            .then(pl.lit(200.0).cast(pl.Float32))
            .otherwise(pl.col("low"))
            .alias("low")
        )
        issues = check_logical_consistency(df)
        messages = [i.message for i in issues]
        assert any("low > close" in m for m in messages)

    def test_negative_price(self):
        """Detects rows where a price column is <= 0."""
        df = _make_clean_df()
        df = df.with_columns(
            pl.when(pl.arange(0, pl.len()) == 5)
            .then(pl.lit(-1.0).cast(pl.Float32))
            .otherwise(pl.col("open"))
            .alias("open")
        )
        issues = check_logical_consistency(df)
        error_issues = [i for i in issues if i.check == "positive_price"]
        assert len(error_issues) > 0

    def test_zero_price(self):
        """Detects rows where a price column is exactly 0."""
        df = _make_clean_df()
        df = df.with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(pl.lit(0.0).cast(pl.Float32))
            .otherwise(pl.col("close"))
            .alias("close")
        )
        issues = check_logical_consistency(df)
        error_issues = [i for i in issues if i.check == "positive_price"]
        assert len(error_issues) > 0

    def test_all_issues_are_errors(self):
        """Logical consistency violations are always error-severity."""
        df = _make_clean_df()
        df = df.with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(pl.lit(-5.0).cast(pl.Float32))
            .otherwise(pl.col("open"))
            .alias("open")
        )
        issues = check_logical_consistency(df)
        assert all(i.severity == Severity.ERROR for i in issues)


# ---------------------------------------------------------------------------
# check_chronology
# ---------------------------------------------------------------------------

class TestChronology:
    """Tests for timestamp ordering validation."""

    def test_clean_data_passes(self):
        """Strictly increasing timestamps produce no issues."""
        issues = check_chronology(_make_clean_df())
        assert issues == []

    def test_duplicate_timestamps(self):
        """Detects duplicate timestamps."""
        df = _make_clean_df()
        # Make row 3 have the same timestamp as row 2.
        ts = df["timestamp"].to_list()
        ts[3] = ts[2]
        df = df.with_columns(pl.Series("timestamp", ts))

        issues = check_chronology(df)
        assert len(issues) == 1
        assert "not strictly increasing" in issues[0].message

    def test_backward_timestamps(self):
        """Detects timestamps that go backward."""
        df = _make_clean_df()
        ts = df["timestamp"].to_list()
        ts[5], ts[6] = ts[6], ts[5]  # Swap to create backward jump.
        df = df.with_columns(pl.Series("timestamp", ts))

        issues = check_chronology(df)
        assert len(issues) == 1
        assert issues[0].severity == Severity.ERROR

    def test_single_row_passes(self):
        """A single-row DataFrame has no chronology issues."""
        df = _make_clean_df(rows=1)
        issues = check_chronology(df)
        assert issues == []


# ---------------------------------------------------------------------------
# detect_outliers
# ---------------------------------------------------------------------------

class TestOutlierDetection:
    """Tests for the z-score outlier detector."""

    def test_clean_data_no_outliers(self):
        """Smoothly trending data produces no outlier warnings."""
        issues = detect_outliers(_make_clean_df(rows=50))
        assert issues == []

    def test_spike_detected(self):
        """A massive price spike is flagged as an outlier."""
        df = _make_clean_df(rows=100)
        # Inject a huge spike at row 50 — large enough that even with
        # std inflation the z-score exceeds the 5-sigma threshold.
        close_vals = df["close"].to_list()
        close_vals[50] = 99_999.0
        df = df.with_columns(pl.Series("close", close_vals, dtype=pl.Float32))

        issues = detect_outliers(df)
        assert len(issues) > 0
        assert issues[0].severity == Severity.WARNING
        assert 50 in issues[0].row_indices

    def test_too_few_rows_skipped(self):
        """DataFrames with < 3 rows are skipped (not enough data)."""
        df = _make_clean_df(rows=2)
        issues = detect_outliers(df)
        assert issues == []


# ---------------------------------------------------------------------------
# identify_gaps
# ---------------------------------------------------------------------------

class TestGapDetection:
    """Tests for missing candle detection."""

    def test_clean_data_no_gaps(self):
        """Consecutive 1-minute candles produce no gaps."""
        issues = identify_gaps(_make_clean_df(), Timescale.ONE_MINUTE)
        assert issues == []

    def test_missing_candle_detected(self):
        """A missing candle in a 1-minute series is flagged."""
        df = _make_clean_df(rows=10)
        # Remove row 5 to create a gap.
        indices = list(range(10))
        indices.remove(5)
        df = df.with_row_index("__tmp").filter(pl.col("__tmp").is_in(indices)).drop("__tmp")

        issues = identify_gaps(df, Timescale.ONE_MINUTE)
        assert len(issues) == 1
        assert "1 gaps" in issues[0].message

    def test_tick_data_skipped(self):
        """Tick data has no expected interval — gaps are not checked."""
        issues = identify_gaps(_make_clean_df(), Timescale.TICK)
        assert issues == []

    def test_gaps_are_warnings(self):
        """Gap issues are warning-severity (not errors)."""
        df = _make_clean_df(rows=10)
        indices = list(range(10))
        indices.remove(3)
        df = df.with_row_index("__tmp").filter(pl.col("__tmp").is_in(indices)).drop("__tmp")

        issues = identify_gaps(df, Timescale.ONE_MINUTE)
        assert all(i.severity == Severity.WARNING for i in issues)


# ---------------------------------------------------------------------------
# run_audit (orchestrator)
# ---------------------------------------------------------------------------

class TestRunAudit:
    """Tests for the full audit orchestrator."""

    def test_clean_data_passes(self):
        """Clean data passes the full audit."""
        result = run_audit(_make_clean_df(), Timescale.ONE_MINUTE)
        assert result.passed is True
        assert result.issues == []
        assert result.data_gaps == 0
        assert result.outliers_found == 0

    def test_logical_error_fails_audit(self):
        """A logical consistency error causes the audit to fail."""
        df = _make_clean_df()
        df = df.with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(pl.lit(-1.0).cast(pl.Float32))
            .otherwise(pl.col("open"))
            .alias("open")
        )
        result = run_audit(df, Timescale.ONE_MINUTE)
        assert result.passed is False

    def test_warnings_do_not_fail_audit(self):
        """Gaps and outliers (warnings) do not block the audit from passing."""
        df = _make_clean_df(rows=10)
        # Remove a row to create a gap.
        indices = list(range(10))
        indices.remove(5)
        df = df.with_row_index("__tmp").filter(pl.col("__tmp").is_in(indices)).drop("__tmp")

        result = run_audit(df, Timescale.ONE_MINUTE)
        assert result.passed is True
        assert result.data_gaps > 0

    def test_result_has_timestamp(self):
        """The audit result contains a valid ISO 8601 timestamp."""
        result = run_audit(_make_clean_df(), Timescale.ONE_MINUTE)
        assert "T" in result.timestamp  # basic ISO 8601 check

    def test_to_audit_trail_conversion(self):
        """AuditResult converts cleanly to an AuditTrail schema model."""
        result = run_audit(_make_clean_df(), Timescale.ONE_MINUTE)
        trail = result.to_audit_trail()

        assert trail.audit_passed is True
        assert trail.data_gaps == 0
        assert trail.outliers_found == 0
        assert trail.auditor_version is not None

    def test_chronology_error_fails_audit(self):
        """A chronology violation causes the audit to fail."""
        df = _make_clean_df()
        ts = df["timestamp"].to_list()
        ts[3] = ts[2]
        df = df.with_columns(pl.Series("timestamp", ts))

        result = run_audit(df, Timescale.ONE_MINUTE)
        assert result.passed is False
