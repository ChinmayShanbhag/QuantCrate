"""Dynamic corporate-action adjustments for OHLCV data.

Provides vectorized, backward-looking adjustments for stock splits and
cash dividends.  Every function returns a **new** DataFrame — the original
raw data is never mutated.

Backward adjustment means we modify prices *before* the action's effective
date so that the entire series is comparable on a post-action basis:

    Split  → divide pre-split prices by the ratio, multiply pre-split volume.
    Dividend → subtract the per-share amount from pre-ex-date prices.
"""

from __future__ import annotations

import polars as pl

from qcr.schema import ActionType, CorporateAction

# Columns that represent prices (adjusted by both splits and dividends).
_PRICE_COLUMNS = ("open", "high", "low", "close")


# ---------------------------------------------------------------------------
# Individual adjustments
# ---------------------------------------------------------------------------

def apply_splits(
    df: pl.DataFrame,
    actions: list[CorporateAction],
) -> pl.DataFrame:
    """Apply backward split adjustments to an OHLCV DataFrame.

    For each split action the prices *before* the effective date are divided
    by the split ratio, and volumes are multiplied by it.  Multiple splits
    are applied cumulatively in chronological order.

    Args:
        df: A Polars DataFrame with ``timestamp``, OHLC (float32), and
            ``volume`` (uint64) columns.
        actions: One or more ``CorporateAction`` objects with
            ``action_type == ActionType.SPLIT``.

    Returns:
        A new DataFrame with adjusted prices and volumes.

    Raises:
        ValueError: If any action is not a split.
    """
    splits = _filter_and_validate(actions, ActionType.SPLIT)
    if not splits:
        return df

    # Sort chronologically so cumulative adjustments are correct.
    splits.sort(key=lambda a: a.effective_date)

    result = df.clone()
    for action in splits:
        effective_ts = action.effective_date
        ratio = action.value
        mask = pl.col("timestamp") < effective_ts

        # Divide prices by the ratio for rows before the effective date.
        price_exprs = [
            pl.when(mask)
            .then(pl.col(c) / ratio)
            .otherwise(pl.col(c))
            .cast(pl.Float32)
            .alias(c)
            for c in _PRICE_COLUMNS
        ]

        # Multiply volume by the ratio (more shares outstanding post-split).
        volume_expr = (
            pl.when(mask)
            .then((pl.col("volume").cast(pl.Float64) * ratio).cast(pl.UInt64))
            .otherwise(pl.col("volume"))
            .alias("volume")
        )

        result = result.with_columns([*price_exprs, volume_expr])

    return result


def apply_dividends(
    df: pl.DataFrame,
    actions: list[CorporateAction],
) -> pl.DataFrame:
    """Apply backward dividend adjustments to an OHLCV DataFrame.

    For each dividend the prices *before* the ex-date are reduced by the
    per-share cash amount.  This keeps the post-dividend price series
    continuous (no artificial drop on the ex-date).

    Args:
        df: A Polars DataFrame with ``timestamp`` and OHLC (float32) columns.
        actions: One or more ``CorporateAction`` objects with
            ``action_type == ActionType.DIVIDEND``.

    Returns:
        A new DataFrame with adjusted prices.  Volume is unchanged.

    Raises:
        ValueError: If any action is not a dividend.
    """
    dividends = _filter_and_validate(actions, ActionType.DIVIDEND)
    if not dividends:
        return df

    dividends.sort(key=lambda a: a.effective_date)

    result = df.clone()
    for action in dividends:
        effective_ts = action.effective_date
        amount = action.value
        mask = pl.col("timestamp") < effective_ts

        price_exprs = [
            pl.when(mask)
            .then(pl.col(c) - amount)
            .otherwise(pl.col(c))
            .cast(pl.Float32)
            .alias(c)
            for c in _PRICE_COLUMNS
        ]

        result = result.with_columns(price_exprs)

    return result


def adjust_ohlcv(
    df: pl.DataFrame,
    actions: list[CorporateAction],
) -> pl.DataFrame:
    """Apply all corporate actions (splits then dividends) in one pass.

    This is a convenience wrapper that partitions the actions by type,
    applies splits first (they affect volume), then dividends.

    Args:
        df: A Polars DataFrame with canonical OHLCV columns.
        actions: Mixed list of split and dividend actions.

    Returns:
        A fully adjusted DataFrame.
    """
    if not actions:
        return df

    splits = [a for a in actions if a.action_type == ActionType.SPLIT]
    dividends = [a for a in actions if a.action_type == ActionType.DIVIDEND]

    result = df
    if splits:
        result = apply_splits(result, splits)
    if dividends:
        result = apply_dividends(result, dividends)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_and_validate(
    actions: list[CorporateAction],
    expected_type: ActionType,
) -> list[CorporateAction]:
    """Filter actions to the expected type and reject any mismatches.

    Raises:
        ValueError: If an action has the wrong type.
    """
    filtered: list[CorporateAction] = []
    for action in actions:
        if action.action_type != expected_type:
            raise ValueError(
                f"Expected {expected_type.value} action, got {action.action_type.value} "
                f"(effective {action.effective_date.isoformat()})"
            )
        filtered.append(action)
    return filtered
