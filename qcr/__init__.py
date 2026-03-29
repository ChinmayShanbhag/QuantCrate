"""QuantCrate — a smart binary container for financial time-series data."""

__version__ = "0.1.1"

# Core storage functions
from qcr.storage import (
    COLUMN_ENCODINGS,
    DATA_PAGE_SIZE,
    SORTING_COLUMNS,
    ZSTD_COMPRESSION_LEVEL,
    load_qcr,
    read_encoding_stats,
    read_last_timestamp,
    read_qcr_metadata,
    save_qcr,
    save_sealed_qcr,
)

# Schema models
from qcr.schema import (
    ActionType,
    AssetClass,
    AuditTrail,
    CorporateAction,
    FullMetadata,
    QcrMetadata,
    Timescale,
)

# Auditor
from qcr.auditor import (
    AuditIssue,
    AuditResult,
    Severity,
    check_chronology,
    check_logical_consistency,
    detect_outliers,
    identify_gaps,
    run_audit,
)

# Corporate actions
from qcr.adjust import (
    adjust_ohlcv,
    apply_dividends,
    apply_splits,
)

__all__ = [
    # Version
    "__version__",
    # Storage
    "COLUMN_ENCODINGS",
    "DATA_PAGE_SIZE",
    "SORTING_COLUMNS",
    "ZSTD_COMPRESSION_LEVEL",
    "load_qcr",
    "read_encoding_stats",
    "read_last_timestamp",
    "read_qcr_metadata",
    "save_qcr",
    "save_sealed_qcr",
    # Schema
    "ActionType",
    "AssetClass",
    "AuditTrail",
    "CorporateAction",
    "FullMetadata",
    "QcrMetadata",
    "Timescale",
    # Auditor
    "AuditIssue",
    "AuditResult",
    "Severity",
    "check_chronology",
    "check_logical_consistency",
    "detect_outliers",
    "identify_gaps",
    "run_audit",
    # Adjustments
    "adjust_ohlcv",
    "apply_dividends",
    "apply_splits",
]

