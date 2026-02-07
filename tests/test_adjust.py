"""Tests for the qcr.adjust module — Phase 4 dynamic adjustment tests."""

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from qcr.adjust import adjust_ohlcv, apply_dividends, apply_splits
from qcr.schema import ActionType, CorporateAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_df(rows: int = 20) -> pl.DataFrame:
    """Create a stable daily OHLCV DataFrame for adjustment tests.

    Prices are flat at 100.0 so that adjustments are easy to verify.
    """
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = [base + timedelta(days=i) for i in range(rows)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * rows,
            "high": [105.0] * rows,
            "low": [95.0] * rows,
            "close": [100.0] * rows,
            "volume": [1_000_000] * rows,
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


def _make_split_action(
    day_offset: int = 10,
    ratio: float = 2.0,
) -> CorporateAction:
    """Create a split action at a given day offset from 2025-01-01."""
    return CorporateAction(
        action_type=ActionType.SPLIT,
        effective_date=datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_offset),
        value=ratio,
        description=f"{ratio}-for-1 split",
    )


def _make_dividend_action(
    day_offset: int = 10,
    amount: float = 2.0,
) -> CorporateAction:
    """Create a dividend action at a given day offset from 2025-01-01."""
    return CorporateAction(
        action_type=ActionType.DIVIDEND,
        effective_date=datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_offset),
        value=amount,
        description=f"${amount} cash dividend",
    )


# ---------------------------------------------------------------------------
# apply_splits
# ---------------------------------------------------------------------------

class TestApplySplits:
    """Tests for backward split adjustments."""

    def test_no_actions_returns_unchanged(self):
        """An empty actions list returns the original DataFrame."""
        df = _make_daily_df()
        result = apply_splits(df, [])
        assert result.equals(df)

    def test_two_for_one_split_adjusts_pre_split_prices(self):
        """A 2-for-1 split halves pre-split prices."""
        df = _make_daily_df(rows=20)
        action = _make_split_action(day_offset=10, ratio=2.0)
        result = apply_splits(df, [action])

        # Rows 0-9 (before split) should have prices halved.
        pre_split = result.filter(pl.col("timestamp") < action.effective_date)
        assert pre_split["open"].to_list() == pytest.approx([50.0] * 10)
        assert pre_split["high"].to_list() == pytest.approx([52.5] * 10)
        assert pre_split["low"].to_list() == pytest.approx([47.5] * 10)
        assert pre_split["close"].to_list() == pytest.approx([50.0] * 10)

    def test_two_for_one_split_doubles_pre_split_volume(self):
        """A 2-for-1 split doubles pre-split volume."""
        df = _make_daily_df(rows=20)
        action = _make_split_action(day_offset=10, ratio=2.0)
        result = apply_splits(df, [action])

        pre_split = result.filter(pl.col("timestamp") < action.effective_date)
        assert all(v == 2_000_000 for v in pre_split["volume"].to_list())

    def test_post_split_rows_unchanged(self):
        """Rows on or after the effective date are not modified."""
        df = _make_daily_df(rows=20)
        action = _make_split_action(day_offset=10, ratio=2.0)
        result = apply_splits(df, [action])

        post_split = result.filter(pl.col("timestamp") >= action.effective_date)
        assert post_split["open"].to_list() == pytest.approx([100.0] * 10)
        assert post_split["volume"].to_list() == [1_000_000] * 10

    def test_four_for_one_split(self):
        """A 4-for-1 split divides pre-split prices by 4."""
        df = _make_daily_df(rows=20)
        action = _make_split_action(day_offset=10, ratio=4.0)
        result = apply_splits(df, [action])

        pre_split = result.filter(pl.col("timestamp") < action.effective_date)
        assert pre_split["close"].to_list() == pytest.approx([25.0] * 10)

    def test_cumulative_splits(self):
        """Multiple splits are applied cumulatively in chronological order."""
        df = _make_daily_df(rows=20)
        split_1 = _make_split_action(day_offset=10, ratio=2.0)  # Day 10
        split_2 = _make_split_action(day_offset=15, ratio=2.0)  # Day 15

        result = apply_splits(df, [split_2, split_1])  # Passed out of order

        # Rows 0-9: affected by BOTH splits → 100 / 2 / 2 = 25
        earliest = result.filter(pl.col("timestamp") < split_1.effective_date)
        assert earliest["close"].to_list() == pytest.approx([25.0] * 10)

        # Rows 10-14: affected by split_2 only → 100 / 2 = 50
        middle = result.filter(
            (pl.col("timestamp") >= split_1.effective_date)
            & (pl.col("timestamp") < split_2.effective_date)
        )
        assert middle["close"].to_list() == pytest.approx([50.0] * 5)

        # Rows 15-19: not affected → 100
        latest = result.filter(pl.col("timestamp") >= split_2.effective_date)
        assert latest["close"].to_list() == pytest.approx([100.0] * 5)

    def test_preserves_dtypes(self):
        """Output columns retain float32 prices and uint64 volume."""
        df = _make_daily_df()
        action = _make_split_action()
        result = apply_splits(df, [action])

        assert result["open"].dtype == pl.Float32
        assert result["close"].dtype == pl.Float32
        assert result["volume"].dtype == pl.UInt64

    def test_original_df_not_mutated(self):
        """The original DataFrame is not modified in place."""
        df = _make_daily_df()
        original_close = df["close"].to_list()
        _ = apply_splits(df, [_make_split_action()])
        assert df["close"].to_list() == original_close

    def test_wrong_action_type_raises(self):
        """Passing a dividend action to apply_splits raises ValueError."""
        df = _make_daily_df()
        dividend = _make_dividend_action()
        with pytest.raises(ValueError, match="Expected split action"):
            apply_splits(df, [dividend])


# ---------------------------------------------------------------------------
# apply_dividends
# ---------------------------------------------------------------------------

class TestApplyDividends:
    """Tests for backward dividend adjustments."""

    def test_no_actions_returns_unchanged(self):
        """An empty actions list returns the original DataFrame."""
        df = _make_daily_df()
        result = apply_dividends(df, [])
        assert result.equals(df)

    def test_cash_dividend_reduces_pre_exdate_prices(self):
        """A $2 dividend reduces pre-ex-date prices by $2."""
        df = _make_daily_df(rows=20)
        action = _make_dividend_action(day_offset=10, amount=2.0)
        result = apply_dividends(df, [action])

        pre_ex = result.filter(pl.col("timestamp") < action.effective_date)
        assert pre_ex["open"].to_list() == pytest.approx([98.0] * 10)
        assert pre_ex["high"].to_list() == pytest.approx([103.0] * 10)
        assert pre_ex["low"].to_list() == pytest.approx([93.0] * 10)
        assert pre_ex["close"].to_list() == pytest.approx([98.0] * 10)

    def test_post_exdate_prices_unchanged(self):
        """Rows on or after the ex-date are not modified."""
        df = _make_daily_df(rows=20)
        action = _make_dividend_action(day_offset=10, amount=2.0)
        result = apply_dividends(df, [action])

        post_ex = result.filter(pl.col("timestamp") >= action.effective_date)
        assert post_ex["close"].to_list() == pytest.approx([100.0] * 10)

    def test_volume_unchanged_by_dividend(self):
        """Dividends do not affect volume."""
        df = _make_daily_df(rows=20)
        action = _make_dividend_action(day_offset=10, amount=2.0)
        result = apply_dividends(df, [action])

        assert result["volume"].to_list() == [1_000_000] * 20

    def test_cumulative_dividends(self):
        """Multiple dividends are applied cumulatively."""
        df = _make_daily_df(rows=20)
        div_1 = _make_dividend_action(day_offset=10, amount=1.0)
        div_2 = _make_dividend_action(day_offset=15, amount=1.5)

        result = apply_dividends(df, [div_2, div_1])  # Out of order

        # Rows 0-9: affected by both → 100 - 1.0 - 1.5 = 97.5
        earliest = result.filter(pl.col("timestamp") < div_1.effective_date)
        assert earliest["close"].to_list() == pytest.approx([97.5] * 10)

        # Rows 10-14: affected by div_2 only → 100 - 1.5 = 98.5
        middle = result.filter(
            (pl.col("timestamp") >= div_1.effective_date)
            & (pl.col("timestamp") < div_2.effective_date)
        )
        assert middle["close"].to_list() == pytest.approx([98.5] * 5)

        # Rows 15-19: not affected → 100
        latest = result.filter(pl.col("timestamp") >= div_2.effective_date)
        assert latest["close"].to_list() == pytest.approx([100.0] * 5)

    def test_preserves_dtypes(self):
        """Output columns retain float32 prices."""
        df = _make_daily_df()
        action = _make_dividend_action()
        result = apply_dividends(df, [action])

        assert result["close"].dtype == pl.Float32
        assert result["volume"].dtype == pl.UInt64

    def test_original_df_not_mutated(self):
        """The original DataFrame is not modified in place."""
        df = _make_daily_df()
        original_close = df["close"].to_list()
        _ = apply_dividends(df, [_make_dividend_action()])
        assert df["close"].to_list() == original_close

    def test_wrong_action_type_raises(self):
        """Passing a split action to apply_dividends raises ValueError."""
        df = _make_daily_df()
        split = _make_split_action()
        with pytest.raises(ValueError, match="Expected dividend action"):
            apply_dividends(df, [split])


# ---------------------------------------------------------------------------
# adjust_ohlcv (orchestrator)
# ---------------------------------------------------------------------------

class TestAdjustOhlcv:
    """Tests for the combined adjustment orchestrator."""

    def test_empty_actions_returns_unchanged(self):
        """No actions returns the original DataFrame."""
        df = _make_daily_df()
        result = adjust_ohlcv(df, [])
        assert result.equals(df)

    def test_splits_and_dividends_combined(self):
        """A split and a dividend are both applied correctly."""
        df = _make_daily_df(rows=20)
        split = _make_split_action(day_offset=10, ratio=2.0)
        dividend = _make_dividend_action(day_offset=15, amount=1.0)

        result = adjust_ohlcv(df, [split, dividend])

        # Rows 0-9: split (100/2=50) then dividend (50-1=49)
        earliest = result.filter(pl.col("timestamp") < split.effective_date)
        assert earliest["close"].to_list() == pytest.approx([49.0] * 10)

        # Rows 10-14: no split, but dividend (100-1=99)
        middle = result.filter(
            (pl.col("timestamp") >= split.effective_date)
            & (pl.col("timestamp") < dividend.effective_date)
        )
        assert middle["close"].to_list() == pytest.approx([99.0] * 5)

        # Rows 15-19: nothing → 100
        latest = result.filter(pl.col("timestamp") >= dividend.effective_date)
        assert latest["close"].to_list() == pytest.approx([100.0] * 5)

    def test_only_splits_when_no_dividends(self):
        """Works correctly with only split actions."""
        df = _make_daily_df(rows=20)
        split = _make_split_action(day_offset=10, ratio=4.0)

        result = adjust_ohlcv(df, [split])
        pre = result.filter(pl.col("timestamp") < split.effective_date)
        assert pre["close"].to_list() == pytest.approx([25.0] * 10)

    def test_only_dividends_when_no_splits(self):
        """Works correctly with only dividend actions."""
        df = _make_daily_df(rows=20)
        div = _make_dividend_action(day_offset=10, amount=3.0)

        result = adjust_ohlcv(df, [div])
        pre = result.filter(pl.col("timestamp") < div.effective_date)
        assert pre["close"].to_list() == pytest.approx([97.0] * 10)

    def test_volume_adjusted_by_splits_not_dividends(self):
        """Volume is only affected by splits, not dividends."""
        df = _make_daily_df(rows=20)
        split = _make_split_action(day_offset=10, ratio=2.0)
        dividend = _make_dividend_action(day_offset=15, amount=1.0)

        result = adjust_ohlcv(df, [split, dividend])

        pre_split = result.filter(pl.col("timestamp") < split.effective_date)
        assert all(v == 2_000_000 for v in pre_split["volume"].to_list())

        post_all = result.filter(pl.col("timestamp") >= dividend.effective_date)
        assert all(v == 1_000_000 for v in post_all["volume"].to_list())


# ---------------------------------------------------------------------------
# CorporateAction model validation
# ---------------------------------------------------------------------------

class TestCorporateActionModel:
    """Tests for the CorporateAction Pydantic model."""

    def test_valid_split(self):
        """A valid split action is created without errors."""
        action = _make_split_action()
        assert action.action_type == ActionType.SPLIT
        assert action.value == 2.0

    def test_valid_dividend(self):
        """A valid dividend action is created without errors."""
        action = _make_dividend_action()
        assert action.action_type == ActionType.DIVIDEND
        assert action.value == 2.0

    def test_zero_value_rejected(self):
        """A value of 0 is rejected (must be > 0)."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            CorporateAction(
                action_type=ActionType.SPLIT,
                effective_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
                value=0.0,
            )

    def test_negative_value_rejected(self):
        """A negative value is rejected."""
        with pytest.raises(Exception):
            CorporateAction(
                action_type=ActionType.DIVIDEND,
                effective_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
                value=-1.0,
            )
