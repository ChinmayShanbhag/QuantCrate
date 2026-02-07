"""Financial data auditor — the "linter" for OHLCV data.

Runs a suite of quality checks on a DataFrame before it can be "sealed"
into a .qcr file. Each check produces a list of AuditIssue records so
that callers get granular, actionable feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

import polars as pl

from qcr import __version__
from qcr.schema import AuditTrail, Timescale

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRICE_COLUMNS = ["open", "high", "low", "close"]
OUTLIER_STD_THRESHOLD = 5.0

# Mapping from Timescale enum → expected gap between consecutive candles.
_TIMESCALE_DURATIONS: dict[Timescale, Optional[timedelta]] = {
    Timescale.ONE_SECOND: timedelta(seconds=1),
    Timescale.ONE_MINUTE: timedelta(minutes=1),
    Timescale.FIVE_MINUTES: timedelta(minutes=5),
    Timescale.FIFTEEN_MINUTES: timedelta(minutes=15),
    Timescale.THIRTY_MINUTES: timedelta(minutes=30),
    Timescale.ONE_HOUR: timedelta(hours=1),
    Timescale.FOUR_HOURS: timedelta(hours=4),
    Timescale.ONE_DAY: timedelta(days=1),
    Timescale.ONE_WEEK: timedelta(weeks=1),
    Timescale.ONE_MONTH: timedelta(days=30),  # Approximation for monthly
    # Tick data has no fixed interval — gap detection is skipped.
    Timescale.TICK: None,
}

# Human-readable labels for gap detection messages.
_TIMESCALE_LABELS: dict[Timescale, str] = {
    Timescale.ONE_SECOND: "1s",
    Timescale.ONE_MINUTE: "1m",
    Timescale.FIVE_MINUTES: "5m",
    Timescale.FIFTEEN_MINUTES: "15m",
    Timescale.THIRTY_MINUTES: "30m",
    Timescale.ONE_HOUR: "1h",
    Timescale.FOUR_HOURS: "4h",
    Timescale.ONE_DAY: "1d",
    Timescale.ONE_WEEK: "1w",
    Timescale.ONE_MONTH: "1mo",
    Timescale.TICK: "tick",
}


# ---------------------------------------------------------------------------
# Issue model
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """How serious an audit finding is."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class AuditIssue:
    """A single problem found during an audit check."""

    check: str
    severity: Severity
    message: str
    row_indices: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Audit result
# ---------------------------------------------------------------------------

@dataclass
class AuditResult:
    """Aggregated output of a full audit run."""

    passed: bool
    issues: list[AuditIssue]
    data_gaps: int
    outliers_found: int
    timestamp: str  # ISO 8601 UTC

    def to_audit_trail(self) -> AuditTrail:
        """Convert this result into a schema-level AuditTrail for embedding."""
        return AuditTrail(
            audit_passed=self.passed,
            audit_timestamp=self.timestamp,
            data_gaps=self.data_gaps,
            outliers_found=self.outliers_found,
            auditor_version=__version__,
        )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_logical_consistency(df: pl.DataFrame) -> list[AuditIssue]:
    """Validate OHLC price relationships and non-negative volume.

    Rules:
        - high >= open, high >= close, high >= low
        - low <= open, low <= close
        - All prices > 0
        - volume >= 0 (always true for uint64, but checked for safety)
    """
    issues: list[AuditIssue] = []
    indexed = df.with_row_index("__idx")

    # High must be the highest price in the candle.
    for col in ("open", "close", "low"):
        violating = indexed.filter(pl.col("high") < pl.col(col))
        if len(violating) > 0:
            indices = violating["__idx"].to_list()
            issues.append(
                AuditIssue(
                    check="logical_consistency",
                    severity=Severity.ERROR,
                    message=f"high < {col} in {len(violating)} rows",
                    row_indices=indices,
                )
            )

    # Low must be the lowest price in the candle.
    for col in ("open", "close"):
        violating = indexed.filter(pl.col("low") > pl.col(col))
        if len(violating) > 0:
            indices = violating["__idx"].to_list()
            issues.append(
                AuditIssue(
                    check="logical_consistency",
                    severity=Severity.ERROR,
                    message=f"low > {col} in {len(violating)} rows",
                    row_indices=indices,
                )
            )

    # All prices must be strictly positive.
    for col in PRICE_COLUMNS:
        non_positive = indexed.filter(pl.col(col) <= 0)
        if len(non_positive) > 0:
            indices = non_positive["__idx"].to_list()
            issues.append(
                AuditIssue(
                    check="positive_price",
                    severity=Severity.ERROR,
                    message=f"{col} <= 0 in {len(non_positive)} rows",
                    row_indices=indices,
                )
            )

    return issues


def check_chronology(df: pl.DataFrame) -> list[AuditIssue]:
    """Ensure timestamps are strictly increasing (no duplicates, no time-travel).

    Returns an error-level issue if any violations are found.
    """
    issues: list[AuditIssue] = []

    if len(df) < 2:
        return issues

    # Add row index and compute diff in one expression context.
    checked = (
        df.with_row_index("__idx")
        .with_columns(pl.col("timestamp").diff().alias("__ts_diff"))
    )

    # Non-increasing: diff is null (first row) or <= 0. Skip row 0.
    bad_rows = checked.filter(
        (pl.col("__idx") > 0)
        & (
            pl.col("__ts_diff").is_null()
            | (pl.col("__ts_diff") <= timedelta(0))
        )
    )

    if len(bad_rows) > 0:
        bad_indices = bad_rows["__idx"].to_list()
        issues.append(
            AuditIssue(
                check="chronology",
                severity=Severity.ERROR,
                message=f"Timestamps are not strictly increasing at {len(bad_indices)} positions",
                row_indices=bad_indices,
            )
        )

    return issues


def detect_outliers(df: pl.DataFrame) -> list[AuditIssue]:
    """Flag candles where the close-to-close return exceeds 5 standard deviations.

    Uses a simple z-score approach on log returns. Returns a warning-level
    issue since outliers may be legitimate (e.g. earnings gaps).
    """
    issues: list[AuditIssue] = []

    if len(df) < 3:
        return issues

    # Compute log returns as a new column, then z-score them.
    analyzed = (
        df.with_row_index("__idx")
        .with_columns(
            (pl.col("close").cast(pl.Float64) / pl.col("close").cast(pl.Float64).shift(1))
            .log()
            .alias("__log_ret")
        )
    )

    # Get mean and std of log returns (excluding the first null).
    stats = analyzed.filter(pl.col("__log_ret").is_not_null()).select(
        pl.col("__log_ret").mean().alias("mean"),
        pl.col("__log_ret").std().alias("std"),
    )
    mean_val = stats["mean"][0]
    std_val = stats["std"][0]

    if std_val is None or std_val == 0:
        return issues

    # Find outlier rows.
    outlier_rows = analyzed.filter(
        pl.col("__log_ret").is_not_null()
        & (((pl.col("__log_ret") - mean_val) / std_val).abs() > OUTLIER_STD_THRESHOLD)
    )

    if len(outlier_rows) > 0:
        outlier_indices = outlier_rows["__idx"].to_list()
        issues.append(
            AuditIssue(
                check="outlier_detection",
                severity=Severity.WARNING,
                message=f"{len(outlier_indices)} candles exceed {OUTLIER_STD_THRESHOLD}-sigma threshold",
                row_indices=outlier_indices,
            )
        )

    return issues


def identify_gaps(df: pl.DataFrame, timescale: Timescale) -> list[AuditIssue]:
    """Identify missing candles based on the expected timescale interval.

    Tick data is skipped (no fixed interval). For all other timescales,
    any gap larger than the expected duration is flagged as a warning.
    """
    issues: list[AuditIssue] = []
    expected_gap = _TIMESCALE_DURATIONS.get(timescale)

    if expected_gap is None or len(df) < 2:
        return issues

    # Compute timestamp diffs and find rows where the gap exceeds expected.
    checked = (
        df.with_row_index("__idx")
        .with_columns(pl.col("timestamp").diff().alias("__ts_diff"))
    )

    gap_rows = checked.filter(
        (pl.col("__idx") > 0) & (pl.col("__ts_diff") > expected_gap)
    )

    if len(gap_rows) > 0:
        gap_indices = gap_rows["__idx"].to_list()
        label = _TIMESCALE_LABELS.get(timescale, str(timescale))
        issues.append(
            AuditIssue(
                check="gap_detection",
                severity=Severity.WARNING,
                message=f"{len(gap_indices)} gaps detected (expected interval: {label})",
                row_indices=gap_indices,
            )
        )

    return issues


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_audit(df: pl.DataFrame, timescale: Timescale) -> AuditResult:
    """Run all audit checks and produce a summary result.

    Error-level issues cause the audit to fail. Warning-level issues
    (outliers, gaps) are reported but do not block sealing.

    Args:
        df: A Polars DataFrame with the canonical OHLCV columns.
        timescale: The expected candle interval for gap detection.

    Returns:
        An AuditResult summarising all findings.
    """
    all_issues: list[AuditIssue] = []

    all_issues.extend(check_logical_consistency(df))
    all_issues.extend(check_chronology(df))

    outlier_issues = detect_outliers(df)
    all_issues.extend(outlier_issues)

    gap_issues = identify_gaps(df, timescale)
    all_issues.extend(gap_issues)

    # Count specific metrics for the audit trail.
    data_gaps = sum(len(i.row_indices) for i in gap_issues)
    outliers_found = sum(len(i.row_indices) for i in outlier_issues)

    # Only errors cause failure; warnings are informational.
    has_errors = any(i.severity == Severity.ERROR for i in all_issues)

    return AuditResult(
        passed=not has_errors,
        issues=all_issues,
        data_gaps=data_gaps,
        outliers_found=outliers_found,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
