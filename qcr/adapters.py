"""Data-source adapters for fetching financial time-series data.

Provides a ``BaseAdapter`` abstract class and concrete implementations
(e.g. ``YahooFinanceAdapter``) that normalise raw vendor data into the
canonical QuantCrate OHLCV schema so it can be audited and sealed.

Every adapter exposes an **async** ``fetch_ohlcv`` method.  Synchronous
vendor SDKs are wrapped with ``asyncio.to_thread`` so that the event loop
is never blocked — a design choice that makes future real-time / streaming
support straightforward.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import polars as pl

from qcr.schema import AssetClass, Timescale

try:
    import yfinance as yf
except ImportError:  # yfinance is an optional dependency
    yf = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Timescale → yfinance interval mapping
# ---------------------------------------------------------------------------

_YF_INTERVAL_MAP: dict[Timescale, str] = {
    Timescale.ONE_MINUTE: "1m",
    Timescale.FIVE_MINUTES: "5m",
    Timescale.FIFTEEN_MINUTES: "15m",
    Timescale.THIRTY_MINUTES: "30m",
    Timescale.ONE_HOUR: "1h",
    Timescale.ONE_DAY: "1d",
    Timescale.ONE_WEEK: "1wk",
    Timescale.ONE_MONTH: "1mo",
}


# ---------------------------------------------------------------------------
# Abstract base adapter
# ---------------------------------------------------------------------------

class BaseAdapter(ABC):
    """Abstract interface that every data-source adapter must implement.

    Subclasses override ``fetch_ohlcv`` to pull raw data from a specific
    vendor and return a **Polars** DataFrame with the canonical columns:

        timestamp (Datetime[ns, UTC]), open (Float32), high (Float32),
        low (Float32), close (Float32), volume (UInt64)

    The returned DataFrame is ready to be fed directly into the Auditor
    and then into ``save_qcr`` / ``save_sealed_qcr``.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name of the data source (e.g. ``'YahooFinance'``)."""

    @abstractmethod
    async def fetch_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str,
        timescale: Timescale = Timescale.ONE_DAY,
    ) -> pl.DataFrame:
        """Fetch OHLCV data for *ticker* between *start* and *end*.

        Args:
            ticker: Asset symbol (e.g. ``"AAPL"``).
            start: Start date as ``YYYY-MM-DD``.
            end: End date as ``YYYY-MM-DD``.
            timescale: Candle interval.

        Returns:
            A Polars DataFrame with the canonical OHLCV columns, sorted by
            timestamp ascending.

        Raises:
            AdapterError: If the fetch fails or returns no data.
        """


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class AdapterError(Exception):
    """Raised when an adapter cannot fulfil a data request."""


# ---------------------------------------------------------------------------
# Yahoo Finance adapter
# ---------------------------------------------------------------------------

class YahooFinanceAdapter(BaseAdapter):
    """Fetch historical OHLCV data from Yahoo Finance via ``yfinance``.

    This adapter wraps the synchronous ``yfinance.download`` call inside
    ``asyncio.to_thread`` so it plays nicely with async orchestration.

    Attributes:
        auto_adjust: If ``True`` (default), Yahoo's adjusted prices are
            used.  Set to ``False`` to get raw/unadjusted prices.
    """

    def __init__(self, auto_adjust: bool = True) -> None:
        self.auto_adjust = auto_adjust

    @property
    def source_name(self) -> str:  # noqa: D102
        return "YahooFinance"

    async def fetch_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str,
        timescale: Timescale = Timescale.ONE_DAY,
    ) -> pl.DataFrame:
        """Download OHLCV data from Yahoo Finance.

        Args:
            ticker: Yahoo Finance ticker (e.g. ``"AAPL"``, ``"BTC-USD"``).
            start: Start date ``YYYY-MM-DD``.
            end: End date ``YYYY-MM-DD``.
            timescale: Candle interval (mapped to yfinance intervals).

        Returns:
            Canonical Polars DataFrame.

        Raises:
            AdapterError: On network failure, invalid ticker, or empty result.
        """
        yf_interval = _YF_INTERVAL_MAP.get(timescale)
        if yf_interval is None:
            raise AdapterError(
                f"Timescale '{timescale.value}' is not supported by Yahoo Finance. "
                f"Supported: {', '.join(ts.value for ts in _YF_INTERVAL_MAP)}"
            )

        # Run the blocking yfinance call in a thread so we don't stall the
        # event loop — critical for future real-time / multi-ticker support.
        try:
            raw_df = await asyncio.to_thread(
                self._download_sync, ticker, start, end, yf_interval
            )
        except Exception as exc:
            raise AdapterError(
                f"Failed to fetch {ticker} from Yahoo Finance: {exc}"
            ) from exc

        if raw_df is None or raw_df.is_empty():
            raise AdapterError(
                f"Yahoo Finance returned no data for {ticker} "
                f"({start} → {end}, interval={yf_interval})"
            )

        return raw_df

    # -- synchronous helper (runs inside asyncio.to_thread) -----------------

    def _download_sync(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
    ) -> pl.DataFrame:
        """Perform the actual yfinance download and normalise the result.

        Returns a Polars DataFrame with the canonical OHLCV columns.
        """
        if yf is None:
            raise AdapterError(
                "yfinance is not installed. "
                "Install it with: pip install quantcrate[adapters]"
            )

        # yfinance returns a pandas DataFrame with a DatetimeIndex.
        pdf = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=self.auto_adjust,
            progress=False,
        )

        if pdf is None or pdf.empty:
            return pl.DataFrame()

        # yfinance >= 0.2.31 may return multi-level columns for single
        # tickers — flatten them.
        if hasattr(pdf.columns, "levels") and pdf.columns.nlevels > 1:
            pdf.columns = pdf.columns.get_level_values(0)

        # Reset the DatetimeIndex so "Date" becomes a regular column.
        pdf = pdf.reset_index()

        # Normalise column names to lowercase.
        pdf.columns = [c.strip().lower() for c in pdf.columns]

        # Rename common yfinance aliases.
        rename_map: dict[str, str] = {
            "date": "timestamp",
            "datetime": "timestamp",
            "adj close": "close",  # if auto_adjust is False, prefer adj close
        }
        for old, new in rename_map.items():
            if old in pdf.columns and new not in pdf.columns:
                pdf = pdf.rename(columns={old: new})

        # Convert to Polars and cast to canonical types.
        df = pl.from_pandas(pdf)

        # Keep only the required columns (drop extras like "adj close").
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        available = set(df.columns) & required
        missing = required - available
        if missing:
            raise AdapterError(
                f"Yahoo Finance response is missing columns: {sorted(missing)}"
            )
        df = df.select(sorted(required))

        df = df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("ns", "UTC")),
            pl.col("open").cast(pl.Float32),
            pl.col("high").cast(pl.Float32),
            pl.col("low").cast(pl.Float32),
            pl.col("close").cast(pl.Float32),
            pl.col("volume").cast(pl.UInt64),
        )

        return df.sort("timestamp")
