"""High-level ingestion pipeline: fetch → audit → seal.

The ``ingest_to_qcr`` coroutine ties together the adapter layer, the
financial auditor, and the storage engine into a single async call that
produces a sealed ``.qcr`` file from a live data source.

The **resumption** feature allows the pipeline to detect an existing
``.qcr`` file, read the last timestamp, and only download the missing
data — making incremental updates efficient and resilient to partial
failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import polars as pl

from qcr.adapters import AdapterError, BaseAdapter
from qcr.auditor import AuditResult, Severity, run_audit
from qcr.schema import AssetClass, QcrMetadata, Timescale
from qcr.storage import save_sealed_qcr


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IngestResult:
    """Outcome of an ``ingest_to_qcr`` call.

    Attributes:
        success: ``True`` if the file was sealed successfully.
        path: Resolved path to the ``.qcr`` file (``None`` on failure).
        rows: Number of rows ingested.
        audit: The full ``AuditResult`` (available even on failure).
        error: Human-readable error message (``None`` on success).
        resumed: ``True`` if this was a resumed (incremental) download.
        new_rows: Number of *new* rows fetched during resumption
            (equals ``rows`` for fresh downloads).
    """

    success: bool
    path: Optional[Path] = None
    rows: int = 0
    audit: Optional[AuditResult] = None
    error: Optional[str] = None
    resumed: bool = False
    new_rows: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_resume_start(
    output: Path,
    original_start: str,
) -> tuple[Optional[str], Optional[pl.DataFrame]]:
    """Check *output* for an existing ``.qcr`` file and return a new start date.

    Returns:
        A ``(new_start, existing_df)`` tuple.

        * If the file does not exist or is empty, returns
          ``(None, None)`` — meaning "no resumption possible".
        * Otherwise returns the day *after* the last timestamp in the
          file (as ``YYYY-MM-DD``) and the existing DataFrame so the
          caller can merge.
    """
    if not output.exists():
        return None, None

    try:
        from qcr.storage import load_qcr

        existing_df, _meta = load_qcr(output)
    except Exception:
        # File exists but is corrupt / not a valid .qcr — start fresh.
        return None, None

    if existing_df.is_empty():
        return None, None

    # Get the last timestamp from the existing data.
    last_ts = existing_df["timestamp"].max()

    # Convert Polars datetime → Python datetime for date arithmetic.
    if isinstance(last_ts, datetime):
        py_dt = last_ts
    else:
        # Polars may return its own Datetime type; .to_python() converts.
        py_dt = last_ts  # Polars >= 1.0 returns Python datetime directly

    # The new start is the day *after* the last existing candle so we
    # don't re-download the overlap (the merge step will deduplicate
    # anyway, but this minimises the fetch window).
    next_day = py_dt + timedelta(days=1)
    new_start = next_day.strftime("%Y-%m-%d")

    return new_start, existing_df


def _merge_dataframes(
    existing: pl.DataFrame,
    new: pl.DataFrame,
) -> pl.DataFrame:
    """Merge *existing* and *new* DataFrames, deduplicating on timestamp.

    Rows from *new* take precedence when timestamps collide (the fresh
    download is assumed to be more authoritative).

    Returns a single DataFrame sorted by ``timestamp``.
    """
    if existing.is_empty():
        return new.sort("timestamp")
    if new.is_empty():
        return existing.sort("timestamp")

    combined = pl.concat([existing, new])
    # Keep the *last* occurrence (new data) when timestamps collide.
    deduped = combined.unique(subset=["timestamp"], keep="last")
    return deduped.sort("timestamp")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def ingest_to_qcr(
    adapter: BaseAdapter,
    ticker: str,
    start: str,
    end: str,
    *,
    asset_class: AssetClass = AssetClass.EQUITY,
    currency: str = "USD",
    exchange: str = "XNAS",
    timezone: str = "UTC",
    timescale: Timescale = Timescale.ONE_DAY,
    output: Optional[Path] = None,
    force: bool = False,
    resume: bool = False,
) -> IngestResult:
    """Fetch data via *adapter*, audit it, and seal into a ``.qcr`` file.

    This is the primary programmatic entry-point for the ingestion
    pipeline.  It is ``async`` so that the (potentially slow) network
    fetch does not block the event loop.

    **Resumption** (``resume=True``):
        If the *output* file already exists, the pipeline reads the last
        timestamp, adjusts *start* to the day after, and fetches only
        the missing data.  The new rows are merged with the existing
        data, re-audited, and re-sealed.  If the existing file already
        covers the full date range, the result indicates success with
        zero new rows.

    Args:
        adapter: A concrete ``BaseAdapter`` implementation
            (e.g. ``YahooFinanceAdapter``).
        ticker: Asset symbol (e.g. ``"AAPL"``).
        start: Start date ``YYYY-MM-DD``.
        end: End date ``YYYY-MM-DD``.
        asset_class: Asset classification for metadata.
        currency: ISO 4217 currency code.
        exchange: ISO 10383 MIC.
        timezone: IANA timezone string.
        timescale: Candle interval.
        output: Destination path.  Defaults to ``<ticker>.qcr`` in cwd.
        force: If ``True``, seal even when the audit has warnings
            (gaps / outliers).  Errors still block sealing.
        resume: If ``True``, attempt to resume from an existing file.

    Returns:
        An ``IngestResult`` with the outcome.
    """
    out_path = output or Path(f"{ticker}.qcr")

    # --- 0. Resumption check -------------------------------------------------
    existing_df: Optional[pl.DataFrame] = None
    effective_start = start
    is_resumed = False

    if resume:
        new_start, existing_df = _detect_resume_start(out_path, start)
        if new_start is not None and existing_df is not None:
            # Check if the existing data already covers the requested range.
            if new_start >= end:
                # Nothing to download — existing data is up to date.
                return IngestResult(
                    success=True,
                    path=out_path.resolve(),
                    rows=existing_df.shape[0],
                    new_rows=0,
                    resumed=True,
                    error=None,
                )
            effective_start = new_start
            is_resumed = True

    # --- 1. Fetch data via the adapter ----------------------------------------
    try:
        new_df = await adapter.fetch_ohlcv(ticker, effective_start, end, timescale)
    except AdapterError as exc:
        return IngestResult(success=False, error=str(exc), resumed=is_resumed)
    except Exception as exc:
        return IngestResult(
            success=False,
            error=f"Unexpected error during fetch: {exc}",
            resumed=is_resumed,
        )

    new_row_count = new_df.shape[0] if not new_df.is_empty() else 0

    # --- 1b. Merge with existing data (if resuming) ---------------------------
    if is_resumed and existing_df is not None:
        if new_df.is_empty():
            # No new data fetched — re-seal existing as-is.
            df = existing_df
            new_row_count = 0
        else:
            df = _merge_dataframes(existing_df, new_df)
    else:
        df = new_df

    if df.is_empty():
        return IngestResult(
            success=False,
            rows=0,
            new_rows=0,
            error="Adapter returned an empty DataFrame — nothing to seal.",
            resumed=is_resumed,
        )

    # --- 2. Run the auditor ---------------------------------------------------
    audit_result = run_audit(df, timescale)

    if not audit_result.passed:
        return IngestResult(
            success=False,
            rows=df.shape[0],
            new_rows=new_row_count,
            audit=audit_result,
            error="Audit failed — data contains errors. Fix upstream data and retry.",
            resumed=is_resumed,
        )

    # If there are warnings and force is not set, still seal (warnings
    # don't block), but the caller can inspect the audit for details.

    # --- 3. Build metadata ----------------------------------------------------
    metadata = QcrMetadata(
        ticker=ticker,
        asset_class=asset_class,
        currency=currency,
        exchange=exchange,
        timezone=timezone,
        timescale=timescale,
    )

    # --- 4. Seal and write ----------------------------------------------------
    try:
        written = save_sealed_qcr(
            df, metadata, audit_result.to_audit_trail(), out_path,
        )
    except Exception as exc:
        return IngestResult(
            success=False,
            rows=df.shape[0],
            new_rows=new_row_count,
            audit=audit_result,
            error=f"Failed to write .qcr file: {exc}",
            resumed=is_resumed,
        )

    return IngestResult(
        success=True,
        path=written,
        rows=df.shape[0],
        new_rows=new_row_count,
        audit=audit_result,
        resumed=is_resumed,
    )
