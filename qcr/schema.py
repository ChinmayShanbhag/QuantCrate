"""Pydantic models and Arrow schema definitions for the .qcr format."""

from datetime import datetime
from enum import Enum
from typing import Optional

import pyarrow as pa
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enum definitions (mirrors SCHEMA_SPEC.md §4)
# ---------------------------------------------------------------------------

class AssetClass(str, Enum):
    """Supported asset classes."""

    EQUITY = "Equity"
    CRYPTO = "Crypto"
    FX = "FX"
    FUTURES = "Futures"
    OPTIONS = "Options"
    INDEX = "Index"


class Timescale(str, Enum):
    """Supported candle intervals."""

    TICK = "tick"
    ONE_SECOND = "1s"
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"


# ---------------------------------------------------------------------------
# Metadata model (SCHEMA_SPEC.md §1)
# ---------------------------------------------------------------------------

class QcrMetadata(BaseModel):
    """Identity header embedded in every .qcr file.

    All fields are validated on construction so that invalid metadata
    is caught *before* it touches the Arrow schema.
    """

    ticker: str = Field(..., min_length=1, description="Asset identifier (e.g. AAPL)")
    asset_class: AssetClass = Field(..., description="Equity, Crypto, FX, etc.")
    currency: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="ISO 4217 currency code",
    )
    exchange: str = Field(..., min_length=1, description="ISO 10383 MIC")
    timezone: str = Field(..., min_length=1, description="IANA timezone string")
    timescale: Timescale = Field(..., description="Candle interval")
    is_adjusted: bool = Field(default=False, description="True if prices are split/dividend-adjusted")
    version: int = Field(default=1, ge=1, description="Schema version for forward compatibility")


# ---------------------------------------------------------------------------
# Audit trail model (SCHEMA_SPEC.md §3)
# ---------------------------------------------------------------------------

class AuditTrail(BaseModel):
    """Quality-check results written by the Auditor before sealing."""

    audit_passed: bool
    audit_timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    data_gaps: int = Field(default=0, ge=0)
    outliers_found: int = Field(default=0, ge=0)
    auditor_version: str = Field(default="0.1.0")


# ---------------------------------------------------------------------------
# Corporate actions model (SCHEMA_SPEC.md §6)
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Types of corporate actions that affect price/volume data."""

    SPLIT = "split"
    DIVIDEND = "dividend"


class CorporateAction(BaseModel):
    """A single corporate action (split or dividend).

    Actions are stored chronologically in the metadata and applied
    via backward adjustment by the `adjust` module.
    """

    action_type: ActionType = Field(..., description="split or dividend")
    effective_date: datetime = Field(
        ..., description="UTC datetime when the action takes effect (ex-date)"
    )
    value: float = Field(
        ...,
        gt=0,
        description=(
            "For splits: the ratio (e.g. 4.0 for a 4-for-1 split). "
            "For dividends: the per-share cash amount."
        ),
    )
    description: str = Field(
        default="",
        description="Optional human-readable note (e.g. '4-for-1 forward split')",
    )


# ---------------------------------------------------------------------------
# Arrow payload schema (SCHEMA_SPEC.md §2)
# ---------------------------------------------------------------------------

PAYLOAD_SCHEMA = pa.schema(
    [
        pa.field("timestamp", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("open", pa.float32(), nullable=False),
        pa.field("high", pa.float32(), nullable=False),
        pa.field("low", pa.float32(), nullable=False),
        pa.field("close", pa.float32(), nullable=False),
        pa.field("volume", pa.uint64(), nullable=False),
    ]
)

# The key under which QCR metadata is stored inside the Arrow schema metadata.
QCR_METADATA_KEY = b"qcr_metadata"


# ---------------------------------------------------------------------------
# Full metadata model (identity + optional audit trail)
# ---------------------------------------------------------------------------

class FullMetadata(BaseModel):
    """Combined metadata stored in the Arrow schema.

    The audit trail is optional — it only exists after the Auditor seals the file.
    Corporate actions are optional — they only exist if splits/dividends have been recorded.
    """

    identity: QcrMetadata
    audit: Optional[AuditTrail] = None
    corporate_actions: list[CorporateAction] = Field(default_factory=list)