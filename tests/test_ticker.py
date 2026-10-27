"""Tests for Ticker price adjustment helpers."""

import pandas as pd
import pytest

from borsapy.ticker import _compute_adj_close


class TestComputeAdjClose:
    """Backward dividend adjustment yielding a yfinance-style Adj Close."""

    def test_empty_close_returns_empty(self):
        close = pd.Series([], dtype=float, name="Close")
        divs = pd.DataFrame({"Amount": []}, index=pd.DatetimeIndex([]))
        result = _compute_adj_close(close, divs)
        assert result.empty

    def test_no_dividends_returns_close_unchanged(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=idx)
        divs = pd.DataFrame(columns=["Amount"])
        result = _compute_adj_close(close, divs)
        pd.testing.assert_series_equal(result, close)

    def test_missing_amount_column_returns_close_unchanged(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        close = pd.Series([100.0, 101.0, 102.0], index=idx)
        divs = pd.DataFrame(
            {"OtherCol": [1.0]},
            index=pd.DatetimeIndex(["2024-01-02"]),
        )
        result = _compute_adj_close(close, divs)
        pd.testing.assert_series_equal(result, close)

    def test_single_dividend_scales_prior_prices(self):
        # Close=100 on ex-date with $1 dividend → factor = 99/100 = 0.99
        # All rows BEFORE ex-date multiplied by 0.99; ex-date and after unchanged.
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        close = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx)
        divs = pd.DataFrame(
            {"Amount": [1.0]},
            index=pd.DatetimeIndex(["2024-01-03"]),  # 3rd row
        )
        result = _compute_adj_close(close, divs)
        assert result.iloc[0] == pytest.approx(99.0)
        assert result.iloc[1] == pytest.approx(99.0)
        assert result.iloc[2] == pytest.approx(100.0)  # ex-date unchanged
        assert result.iloc[3] == pytest.approx(100.0)

    def test_multiple_dividends_compound(self):
        # Two dividends → factors multiply on rows BEFORE each ex-date
        # Ex1: 2024-01-02, close=100, div=1 → factor_1 = 0.99 applied to row 0
        # Ex2: 2024-01-04, close=100, div=2 → factor_2 = 0.98 applied to rows 0,1,2
        # Row 0 (2024-01-01): 100 * 0.99 * 0.98 = 97.02  (before both)
        # Row 1 (2024-01-02): 100 * 0.98 = 98.00  (ex-date 1, before ex-date 2)
        # Row 2 (2024-01-03): 100 * 0.98 = 98.00  (after ex-date 1, before ex-date 2)
        # Row 3 (2024-01-04): 100.0  (ex-date 2 itself, nothing after)
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        close = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx)
        divs = pd.DataFrame(
            {"Amount": [1.0, 2.0]},
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-04"]),
        )
        result = _compute_adj_close(close, divs)
        assert result.iloc[0] == pytest.approx(100.0 * 0.99 * 0.98)
        assert result.iloc[1] == pytest.approx(100.0 * 0.98)
        assert result.iloc[2] == pytest.approx(100.0 * 0.98)
        assert result.iloc[3] == pytest.approx(100.0)

    def test_dividend_outside_close_range_is_skipped(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        close = pd.Series([100.0, 100.0, 100.0], index=idx)
        divs = pd.DataFrame(
            {"Amount": [5.0]},
            index=pd.DatetimeIndex(["2025-06-01"]),  # way outside
        )
        result = _compute_adj_close(close, divs)
        pd.testing.assert_series_equal(result, close)

    def test_zero_or_negative_amount_skipped(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        close = pd.Series([100.0, 100.0, 100.0], index=idx)
        divs = pd.DataFrame(
            {"Amount": [0.0, -1.0]},
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
        )
        result = _compute_adj_close(close, divs)
        pd.testing.assert_series_equal(result, close)

    def test_nonzero_close_guarded_against_zero(self):
        # If close on ex-date is 0, adjustment would divide by zero — skip it.
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        close = pd.Series([100.0, 0.0, 100.0], index=idx)
        divs = pd.DataFrame(
            {"Amount": [1.0]},
            index=pd.DatetimeIndex(["2024-01-02"]),
        )
        result = _compute_adj_close(close, divs)
        # No adjustment applied because close_on_ex <= 0
        pd.testing.assert_series_equal(result, close)

    def test_timezone_aware_close_matches_naive_dividend(self):
        # Index is tz-aware (Europe/Istanbul); dividend index is tz-naive.
        # They should still match by calendar date.
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="Europe/Istanbul")
        close = pd.Series([100.0, 100.0, 100.0], index=idx)
        divs = pd.DataFrame(
            {"Amount": [1.0]},
            index=pd.DatetimeIndex(["2024-01-02"]),  # tz-naive
        )
        result = _compute_adj_close(close, divs)
        assert result.iloc[0] == pytest.approx(99.0)
        assert result.iloc[1] == pytest.approx(100.0)

    def test_returns_copy_when_unchanged(self):
        # Caller may mutate result; it shouldn't alias the input.
        idx = pd.date_range("2024-01-01", periods=2, freq="D")
        close = pd.Series([100.0, 100.0], index=idx)
        divs = pd.DataFrame(columns=["Amount"])
        result = _compute_adj_close(close, divs)
        result.iloc[0] = 999.0
        assert close.iloc[0] == 100.0


def _ohlcv_frame(dates: list[str], close: float) -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        [pd.Timestamp(f"{d} 09:00", tz="Europe/Istanbul") for d in dates]
    )
    return pd.DataFrame(
        {
            "Open": [close] * len(dates),
            "High": [close] * len(dates),
            "Low": [close] * len(dates),
            "Close": [close] * len(dates),
            "Volume": [3_000_000.0] * len(dates),
        },
        index=idx,
    )


def _splits_frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in rows])
    return pd.DataFrame(
        {
            "Capital": [0.0] * len(rows),
            "RightsIssue": [0.0] * len(rows),
            "BonusFromCapital": [pct for _, pct in rows],
            "BonusFromDividend": [0.0] * len(rows),
        },
        index=idx,
    )


class TestUnadjustPrices:
    """history(adjust=False) must reverse only adjustments TradingView applied.

    TradingView applies a capital-increase adjustment once the market
    re-prices the event; an event dated after the last bar has not been
    applied, so reversing it would fabricate scaled prices (live incident
    2026-10-27: MERCN 2026-11-02 bedelsiz inflated the whole 1mo window by 3).
    """

    @staticmethod
    def _ticker(monkeypatch, frame: pd.DataFrame, splits) -> "object":
        from borsapy.ticker import Ticker

        class FakeProvider:
            def get_history(self, **kwargs):
                return frame

        ticker = Ticker.__new__(Ticker)  # avoid provider/network on init
        ticker._symbol = "MERCN"
        ticker._tradingview = FakeProvider()
        if isinstance(splits, Exception):
            monkeypatch.setattr(
                Ticker, "splits", property(lambda self: (_ for _ in ()).throw(splits))
            )
        else:
            monkeypatch.setattr(Ticker, "splits", property(lambda self: splits))
        return ticker

    def test_future_bonus_split_leaves_prices_unchanged(self, monkeypatch):
        dates = [f"2026-10-{d:02d}" for d in (20, 21, 22, 23, 26, 27)]
        frame = _ohlcv_frame(dates, 8.81)
        splits = _splits_frame([("2026-11-02", 200.0)])
        ticker = self._ticker(monkeypatch, frame, splits)
        result = ticker.history(period="1mo", interval="1d", adjust=False)
        pd.testing.assert_frame_equal(result, frame)

    def test_past_bonus_split_scales_bars_before_event(self, monkeypatch):
        dates = [f"2023-09-{d:02d}" for d in (26, 27, 28)] + [
            f"2023-10-{d:02d}" for d in (2, 3)
        ]
        frame = _ohlcv_frame(dates, 15.0)
        splits = _splits_frame([("2023-09-29", 200.0)])
        ticker = self._ticker(monkeypatch, frame, splits)
        result = ticker.history(period="1mo", interval="1d", adjust=False)
        assert result["Close"].iloc[0] == pytest.approx(45.0)  # before event: ×3
        assert result["Close"].iloc[2] == pytest.approx(45.0)
        assert result["Close"].iloc[3] == pytest.approx(15.0)  # event day onward
        assert result["Volume"].iloc[0] == pytest.approx(1_000_000.0)  # ÷3

    def test_split_after_last_bar_on_same_day_still_applies(self, monkeypatch):
        # Event midnight equals the last bar's day: bars strictly before that
        # day were adjusted by TradingView and must still be reversed.
        dates = ["2023-09-27", "2023-09-28", "2023-09-29"]
        frame = _ohlcv_frame(dates, 15.0)
        splits = _splits_frame([("2023-09-29", 200.0)])
        ticker = self._ticker(monkeypatch, frame, splits)
        result = ticker.history(period="1mo", interval="1d", adjust=False)
        assert result["Close"].iloc[0] == pytest.approx(45.0)
        assert result["Close"].iloc[1] == pytest.approx(45.0)
        assert result["Close"].iloc[2] == pytest.approx(15.0)

    def test_empty_splits_leaves_prices_unchanged(self, monkeypatch):
        frame = _ohlcv_frame(["2026-10-26", "2026-10-27"], 8.81)
        splits = _splits_frame([])
        ticker = self._ticker(monkeypatch, frame, splits)
        result = ticker.history(period="1mo", interval="1d", adjust=False)
        pd.testing.assert_frame_equal(result, frame)

    def test_splits_lookup_error_leaves_prices_unchanged(self, monkeypatch):
        frame = _ohlcv_frame(["2026-10-26", "2026-10-27"], 8.81)
        ticker = self._ticker(monkeypatch, frame, RuntimeError("isyatirim down"))
        result = ticker.history(period="1mo", interval="1d", adjust=False)
        pd.testing.assert_frame_equal(result, frame)
