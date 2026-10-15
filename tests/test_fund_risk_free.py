"""Test Fund.risk_metrics against a None risk-free rate source.

``borsapy.bond.risk_free_rate`` may return ``None`` when the underlying
source is unavailable. ``Fund.risk_metrics`` must fall back to ``rf=30.0``
instead of raising ``TypeError`` on ``None * 100``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from borsapy.fund import Fund


def test_risk_metrics_falls_back_when_risk_free_rate_none(monkeypatch):
    monkeypatch.setattr("borsapy.bond.risk_free_rate", lambda: None)

    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    prices = [100.0 + (i % 7) for i in range(30)]
    df = pd.DataFrame({"Price": prices}, index=dates)

    # Minimal offline Fund: bypass TEFAS construction and stub history().
    fund = object.__new__(Fund)
    fund.history = lambda period="1y", start=None, end=None: df

    metrics = fund.risk_metrics(period="1y")

    assert metrics["risk_free_rate"] == 30.0
    assert isinstance(metrics["sharpe_ratio"], float)
    assert np.isfinite(metrics["sharpe_ratio"])
