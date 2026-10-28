"""Synthetic tests for the cumulative TTM bridge and standalone-quarter lapping.

Turkish quarterly flow statements are year-to-date cumulative, so neither
``sum(last 4 columns)`` (the old bug) nor naive diffs across year boundaries are
valid.  These tests pin the canonical bridge and the lapping helper with
synthetic series only — no network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from borsapy._providers.isyatirim import IsYatirimProvider
from borsapy.exceptions import APIError
from borsapy.ticker import Ticker


def _cum_df(values: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {label: [value] for label, value in values.items()},
        index=pd.Index(["Revenue"], name="Item"),
    )


# =============================================================================
# C4 — canonical TTM bridge
# =============================================================================


class TestTtmBridge:
    def test_bridge_synthetic_series(self):
        cum = _cum_df(
            {
                "2026Q2": 100.0,
                "2026Q1": 60.0,
                "2025Q4": 150.0,
                "2025Q3": 90.0,
                "2025Q2": 55.0,
            }
        )
        ttm = Ticker._ttm_bridge(cum, "2026Q2", "2025Q4", "2025Q2")
        assert ttm.loc["Revenue", "TTM"] == 195.0  # 100 + 150 - 55

    def test_calculate_ttm_uses_newest_column(self):
        cum = _cum_df(
            {
                "2026Q2": 100.0,
                "2026Q1": 60.0,
                "2025Q4": 150.0,
                "2025Q3": 90.0,
                "2025Q2": 55.0,
            }
        )
        ticker = Ticker.__new__(Ticker)
        ttm = ticker._calculate_ttm(cum)
        assert ttm.loc["Revenue", "TTM"] == 195.0

    def test_q4_target_equals_full_year(self):
        cum = _cum_df({"2025Q4": 150.0, "2024Q4": 120.0, "2025Q3": 90.0})
        ttm = Ticker._ttm_bridge(cum, "2025Q4", "2024Q4", "2024Q4")
        assert ttm.loc["Revenue", "TTM"] == 150.0

    def test_missing_prior_period_raises_and_names_it(self):
        cum = _cum_df({"2026Q2": 100.0, "2026Q1": 60.0, "2025Q4": 150.0, "2025Q3": 90.0})
        with pytest.raises(APIError, match="2025Q2"):
            Ticker._ttm_bridge(cum, "2026Q2", "2025Q4", "2025Q2")

    def test_missing_prior_fy_raises_and_names_it(self):
        cum = _cum_df({"2026Q2": 100.0, "2026Q1": 60.0, "2025Q2": 55.0})
        with pytest.raises(APIError, match="2025Q4"):
            Ticker._ttm_bridge(cum, "2026Q2", "2025Q4", "2025Q2")

    def test_empty_data_raises(self):
        with pytest.raises(APIError):
            Ticker._ttm_bridge(pd.DataFrame(), "2026Q2", "2025Q4", "2025Q2")

    def test_ttm_methods_widen_fetch_to_eight_periods(self):
        ticker = Ticker.__new__(Ticker)
        calls: list[dict] = []

        def fake_get_income_stmt(**kwargs):
            calls.append(kwargs)
            return _cum_df(
                {
                    "2026Q2": 100.0,
                    "2026Q1": 60.0,
                    "2025Q4": 150.0,
                    "2025Q3": 90.0,
                    "2025Q2": 55.0,
                }
            )

        ticker.get_income_stmt = fake_get_income_stmt  # type: ignore[method-assign]
        result = ticker.get_ttm_income_stmt()
        assert result.loc["Revenue", "TTM"] == 195.0
        assert calls[0]["quarterly"] is True
        assert calls[0]["last_n"] >= 5

    def test_no_balance_sheet_ttm_api(self):
        assert not hasattr(Ticker, "get_ttm_balance_sheet")
        assert not hasattr(Ticker, "ttm_balance_sheet")


# =============================================================================
# C5 — standalone quarter derivation (lapping)
# =============================================================================


class TestStandaloneQuarters:
    def test_synthetic_series(self):
        cum = pd.DataFrame(
            {"2026Q4": [60.0], "2026Q3": [39.0], "2026Q2": [25.0], "2026Q1": [10.0]},
            index=pd.Index(["Revenue"], name="Item"),
        )
        result = IsYatirimProvider.derive_standalone_quarters(cum)
        assert list(result.columns) == ["2026Q4", "2026Q3", "2026Q2", "2026Q1"]
        assert result.loc["Revenue", "2026Q1"] == 10.0
        assert result.loc["Revenue", "2026Q2"] == 15.0
        assert result.loc["Revenue", "2026Q3"] == 14.0
        assert result.loc["Revenue", "2026Q4"] == 21.0

    def test_year_boundary_reset(self):
        cum = pd.DataFrame(
            {"2026Q1": [12.0], "2025Q4": [50.0], "2025Q3": [30.0]},
            index=pd.Index(["Revenue"], name="Item"),
        )
        result = IsYatirimProvider.derive_standalone_quarters(cum)
        assert result.loc["Revenue", "2026Q1"] == 12.0
        assert result.loc["Revenue", "2025Q4"] == 20.0
        assert pd.isna(result.loc["Revenue", "2025Q3"])

    def test_nan_propagation(self):
        cum = pd.DataFrame(
            {"2026Q2": [25.0], "2026Q1": [float("nan")]},
            index=pd.Index(["Revenue"], name="Item"),
        )
        result = IsYatirimProvider.derive_standalone_quarters(cum)
        assert pd.isna(result.loc["Revenue", "2026Q1"])
        assert pd.isna(result.loc["Revenue", "2026Q2"])

    def test_negative_values_are_preserved(self):
        cum = pd.DataFrame(
            {"2026Q2": [5.0], "2026Q1": [10.0]},
            index=pd.Index(["Revenue"], name="Item"),
        )
        result = IsYatirimProvider.derive_standalone_quarters(cum)
        assert result.loc["Revenue", "2026Q2"] == -5.0

    def test_duplicate_row_labels_preserved(self):
        cum = pd.DataFrame(
            {"2026Q2": [25.0, 40.0], "2026Q1": [10.0, 10.0]},
            index=pd.Index(["Same", "Same"], name="Item"),
        )
        result = IsYatirimProvider.derive_standalone_quarters(cum)
        assert list(result.index) == ["Same", "Same"]
        assert result["2026Q2"].tolist() == [15.0, 30.0]

    def test_empty_input(self):
        assert IsYatirimProvider.derive_standalone_quarters(pd.DataFrame()).empty
