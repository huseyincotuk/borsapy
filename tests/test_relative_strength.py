"""W3i testleri: endeks göreli güç (RS vs XU100).

Testler **çıktı düzeyinde** ve **offline**'dır; fixture'lar elle yazılmıştır.
Float katmanı → karşılaştırmalar ``pytest.approx`` (rel=1e-9).

El hesabı örneği (sabit hisse, düşen endeks):
  hisse 10,10,10,10 · endeks 100,95,90,85 → oran 0,1 / 0,105263 / 0,111111 / 0,117647
  normalize → 100 / 105,263 / 111,111 / 117,647 ; change_pct = +17,6471
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import borsapy as bp
from borsapy.relative_strength import (
    RSEvent,
    relative_strength_events,
    relative_strength_series,
)
from borsapy.ticker import Ticker

_IDX = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])


def _s(values, index=_IDX):
    return pd.Series(values, index=index, dtype="float64")


# ---------------------------------------------------------------------------
# REQ-001 — oran, trend, normalizasyon
# ---------------------------------------------------------------------------


def test_flat_stock_vs_falling_index_rises_with_guclu_trend():
    """Sabit hisse + düşen endeks → RS yükselir, trend='guclu'."""
    stock = _s([10.0, 10.0, 10.0, 10.0])
    index = _s([100.0, 95.0, 90.0, 85.0])

    rs, meta = relative_strength_series(stock, index, normalize=False)

    assert rs.tolist() == pytest.approx(
        [10 / 100, 10 / 95, 10 / 90, 10 / 85], rel=1e-9
    )
    assert meta["trend"] == "guclu"
    assert meta["trend_slope"] > 0
    assert meta["align"] == "intersection"
    assert meta["n_common"] == 4
    assert meta["n_dropped_stock"] == 0
    assert meta["n_dropped_index"] == 0
    assert meta["series_sha256"] and len(meta["series_sha256"]) == 64


def test_flat_stock_vs_rising_index_falls_with_zayif_trend():
    """Sabit hisse + yükselen endeks → RS düşer, trend='zayif'."""
    stock = _s([10.0, 10.0, 10.0, 10.0])
    index = _s([100.0, 105.0, 110.0, 115.0])

    rs, meta = relative_strength_series(stock, index, normalize=False)

    assert rs.iloc[0] > rs.iloc[-1]
    assert meta["trend"] == "zayif"
    assert meta["trend_slope"] < 0
    assert meta["change_pct"] < 0


def test_normalize_true_first_value_is_100():
    """normalize=True, base=0 → ilk değer tam 100,0."""
    stock = _s([10.0, 11.0, 12.0, 13.0])
    index = _s([100.0, 102.0, 104.0, 106.0])

    rs, meta = relative_strength_series(stock, index, normalize=True, base=0)

    assert rs.iloc[0] == pytest.approx(100.0, abs=1e-12)
    assert meta["base_value"] == pytest.approx(10.0 / 100.0, rel=1e-12)
    assert meta["base_date"] == "2026-01-05"
    assert meta["normalized"] is True
    # El hesabı: bar1 = (11/102)/(10/100)*100
    assert rs.iloc[1] == pytest.approx((11 / 102) / (10 / 100) * 100, rel=1e-9)


def test_normalize_true_custom_base_position():
    """base=2 → o gün 100'e ölçeklenir; öncesi 100'den farklı olabilir."""
    stock = _s([10.0, 11.0, 12.0, 13.0])
    index = _s([100.0, 102.0, 104.0, 106.0])

    rs, meta = relative_strength_series(stock, index, normalize=True, base=2)

    assert rs.iloc[2] == pytest.approx(100.0, abs=1e-12)
    assert meta["base_date"] == "2026-01-07"
    assert rs.iloc[0] < 100.0


def test_normalize_false_returns_raw_ratio():
    stock = _s([10.0, 11.0, 12.0, 13.0])
    index = _s([100.0, 102.0, 104.0, 106.0])

    rs, meta = relative_strength_series(stock, index, normalize=False)

    assert rs.iloc[0] == pytest.approx(0.1, rel=1e-12)
    assert meta["normalized"] is False
    assert rs.name == "RS"


def test_change_pct_hand_calculation():
    """change_pct = (son/baz − 1) × 100, el hesabıyla."""
    stock = _s([10.0, 10.0, 10.0, 10.0])
    index = _s([100.0, 95.0, 90.0, 85.0])

    _, meta = relative_strength_series(stock, index, normalize=False)

    expected = ((10 / 85) / (10 / 100) - 1.0) * 100.0
    assert meta["change_pct"] == pytest.approx(expected, rel=1e-9)
    assert meta["change_pct"] == pytest.approx(17.6471, abs=1e-4)
    assert meta["last_value"] == pytest.approx(10 / 85, rel=1e-12)


def test_series_sha256_is_deterministic():
    stock = _s([10.0, 11.0, 12.0, 13.0])
    index = _s([100.0, 102.0, 104.0, 106.0])

    _, m1 = relative_strength_series(stock, index)
    _, m2 = relative_strength_series(stock, index)

    assert m1["series_sha256"] == m2["series_sha256"]


def test_dataframe_input_uses_close_column():
    idx = _IDX
    stock_df = pd.DataFrame({"Close": [10.0, 10.0, 10.0, 10.0], "Volume": 1.0}, index=idx)
    index_df = pd.DataFrame({"Close": [100.0, 95.0, 90.0, 85.0]}, index=idx)

    rs, meta = relative_strength_series(stock_df, index_df, normalize=False)

    assert meta["n_common"] == 4
    assert rs.iloc[0] == pytest.approx(0.1, rel=1e-12)


def test_dataframe_without_close_raises():
    stock_df = pd.DataFrame({"Price": [1.0, 2.0]}, index=_IDX[:2])
    with pytest.raises(ValueError, match="no 'Close' column"):
        relative_strength_series(stock_df, _s([100.0, 100.0], index=_IDX[:2]))


# ---------------------------------------------------------------------------
# Hizalama
# ---------------------------------------------------------------------------


def test_alignment_intersection_counts_dropped_bars():
    """Farklı gün kümeleri → kesişim; düşen günler künyede sayılır."""
    stock = _s([10.0, 10.0, 10.0, 10.0], index=_IDX)  # 05,06,07,08
    index = _s([100.0, 95.0, 90.0, 85.0], index=pd.to_datetime(
        ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-09"]
    ))

    rs, meta = relative_strength_series(stock, index, normalize=False)

    assert meta["n_common"] == 3
    assert meta["n_dropped_stock"] == 1   # 01-08 sadece histede
    assert meta["n_dropped_index"] == 1   # 01-09 sadece endekste
    assert list(rs.index) == list(pd.to_datetime(
        ["2026-01-05", "2026-01-06", "2026-01-07"]
    ))
    # Eksik gün NaN olarak ÜRETİLMEZ, atlanır
    assert not rs.isna().any()


def test_empty_intersection_raises():
    stock = _s([10.0], index=pd.to_datetime(["2026-02-01"]))
    index = _s([100.0], index=pd.to_datetime(["2026-03-01"]))
    with pytest.raises(ValueError, match="do not share any timestamp"):
        relative_strength_series(stock, index)


def test_zero_index_close_dropped_with_warning():
    """Endeks kapanışı 0 → o bar orandan atlanır, uyarı künyede."""
    idx = _IDX[:3]
    stock = _s([10.0, 10.0, 10.0], index=idx)
    index = _s([100.0, 0.0, 90.0], index=idx)

    rs, meta = relative_strength_series(stock, index, normalize=False)

    assert meta["n_common"] == 2
    assert any("zero/NaN index" in w for w in meta["warnings"])
    assert np.isfinite(rs.to_numpy()).all()


def test_lookback_below_two_raises():
    idx = _IDX[:2]
    stock = _s([10.0, 10.0], index=idx)
    index = _s([100.0, 100.0], index=idx)
    with pytest.raises(ValueError, match="lookback must be >= 2"):
        relative_strength_series(stock, index, lookback=1)


def test_base_out_of_range_raises():
    idx = _IDX[:2]
    stock = _s([10.0, 10.0], index=idx)
    index = _s([100.0, 100.0], index=idx)
    with pytest.raises(ValueError, match="out of range"):
        relative_strength_series(stock, index, base=99)


def test_mixed_timezone_falls_back_to_calendar_date():
    """Bir taraf tz-aware, diğeri naive → gün bazında hizala + uyarı."""
    stock = pd.Series(
        [10.0, 10.0],
        index=pd.DatetimeIndex(
            ["2026-01-05 09:00", "2026-01-06 09:00"], tz="Europe/Istanbul"
        ),
    )
    index = _s([100.0, 100.0], index=pd.DatetimeIndex(["2026-01-05", "2026-01-06"]))  # tz-naive

    rs, meta = relative_strength_series(stock, index, normalize=False)

    assert meta["n_common"] == 2
    assert any("timezone mismatch" in w for w in meta["warnings"])


# ---------------------------------------------------------------------------
# REQ-003 — RSEvent olayları
# ---------------------------------------------------------------------------


def test_new_high_event_matches_rolling_max():
    """yeni_yuksek: değer gerçekten lookback maks'ı olmalı."""
    rs = _s([100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            index=pd.date_range("2026-01-01", periods=6))

    events = relative_strength_events(rs, lookback=3)

    highs = [e for e in events if e.kind == "yeni_yuksek"]
    assert highs, "monoton artan seride yeni_yuksek beklenir"
    for e in highs:
        window = rs.loc[: e.index].iloc[-3:]
        assert e.value == pytest.approx(window.max(), rel=1e-12)
        assert e.kalibre_edilmedi is True


def test_new_low_event_matches_rolling_min():
    rs = _s([105.0, 104.0, 103.0, 102.0, 101.0, 100.0],
            index=pd.date_range("2026-01-01", periods=6))

    events = relative_strength_events(rs, lookback=3)

    lows = [e for e in events if e.kind == "yeni_dusuk"]
    assert lows
    for e in lows:
        window = rs.loc[: e.index].iloc[-3:]
        assert e.value == pytest.approx(window.min(), rel=1e-12)


def test_sicrama_absent_below_threshold():
    """z eşiğinin altında küçük dalgalanma → 'sicrama' olayı YOK."""
    # ±%0,2 salınım: kayan std'ye göre z ~1 civarı; eşiği 5'e çekip garanti et
    values = [100.0 + (0.2 if i % 2 else -0.2) for i in range(30)]
    rs = _s(values, index=pd.date_range("2026-01-01", periods=30))

    events = relative_strength_events(rs, lookback=10, z_threshold=5.0)

    assert all(e.kind != "sicrama" for e in events)


def test_sicrama_present_above_threshold():
    """Tek büyük sıçrama → z yüksek → 'sicrama' üretilir.

    Not: aynı barda 'yeni_yuksek' de sağlanabilir — hiçbir koşul bastırılmaz
    (dürüstlük); bu test yalnız sicrama'nın VAR olduğunu doğrular.
    """
    values = [100.0] * 15 + [100.0 + i * 0.05 for i in range(10)]
    values[20] = 130.0  # ani sıçrama
    rs = _s(values, index=pd.date_range("2026-01-01", periods=len(values)))

    events = relative_strength_events(rs, lookback=10, z_threshold=1.0)

    assert any(e.kind == "sicrama" for e in events)


def test_breakout_event_when_exceeding_previous_extreme():
    """kirilma: önceki pencere tepesini aşan bar yakalanır."""
    values = [100.0, 101.0, 102.0, 101.5, 101.0, 101.5, 103.0]
    rs = _s(values, index=pd.date_range("2026-01-01", periods=len(values)))

    events = relative_strength_events(rs, lookback=3)

    # 103,0 önceki 3 barın tepesini (102,0) aşar
    kinds = [e.kind for e in events]
    assert "kirilma" in kinds or "yeni_yuksek" in kinds
    last = [e for e in events if str(e.index.date()) == "2026-01-07"]
    assert last, "son bar bir olay üretmeli"


def test_no_events_when_no_condition_holds():
    """Koşul sağlanmazsa olay üretilmez (uydurma yok)."""
    rs = _s([100.0], index=pd.date_range("2026-01-01", periods=1))
    assert relative_strength_events(rs, lookback=5) == ()


def test_events_validation():
    rs = _s([100.0, 101.0], index=pd.date_range("2026-01-01", periods=2))
    with pytest.raises(ValueError, match="lookback must be >= 2"):
        relative_strength_events(rs, lookback=1)
    with pytest.raises(ValueError, match="z_threshold"):
        relative_strength_events(rs, z_threshold=0)
    with pytest.raises(ValueError, match="z_threshold"):
        relative_strength_events(rs, z_threshold=float("inf"))


def test_events_empty_series_returns_empty_tuple():
    assert relative_strength_events(pd.Series(dtype="float64")) == ()


def test_rs_event_is_frozen():
    import dataclasses

    e = RSEvent(index=pd.Timestamp("2026-01-01"), kind="yeni_yuksek", value=1.0, z_score=0.0)
    assert e.kalibre_edilmedi is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.value = 2.0  # frozen dataclass


# ---------------------------------------------------------------------------
# REQ-002 — Ticker.relative_strength (ağ stub'lanır)
# ---------------------------------------------------------------------------


def _stub_history_adjusted(self, period="1mo", interval="1d", **kwargs):
    idx = _IDX
    frame = pd.DataFrame(
        {
            "Open": [10.0, 10.0, 10.0, 10.0],
            "High": [10.0, 10.0, 10.0, 10.0],
            "Low": [10.0, 10.0, 10.0, 10.0],
            "Close": [10.0, 10.0, 10.0, 10.0],
            "Volume": [1.0, 1.0, 1.0, 1.0],
            "raw_close": [10.0, 10.0, 10.0, 10.0],
        },
        index=idx,
    )
    return frame, {"row_count": 4}


def test_ticker_relative_strength_stamps_adjusted_basis(monkeypatch):
    """Ticker.relative_strength künyesinde stock_adjusted=True + price_basis."""
    from borsapy.index import Index

    def stub_index_history(self, period="1mo", interval="1d", start=None, end=None):
        return pd.DataFrame({"Close": [100.0, 95.0, 90.0, 85.0]}, index=_IDX)

    monkeypatch.setattr(Ticker, "history_adjusted", _stub_history_adjusted)
    monkeypatch.setattr(Index, "history", stub_index_history)

    rs, meta = Ticker("BIMAS").relative_strength(index="XU100", period="2y")

    assert meta["stock_adjusted"] is True
    assert meta["price_basis"] == "adjusted_stock / raw_index"
    assert meta["index_symbol"] == "XU100"
    assert "raw close" in meta["index_source"]
    assert meta["trend"] == "guclu"
    assert meta["n_common"] == 4
    assert meta["change_pct"] == pytest.approx(17.6471, abs=1e-4)


def test_ticker_relative_strength_raises_when_index_empty(monkeypatch):
    from borsapy.index import Index

    monkeypatch.setattr(Ticker, "history_adjusted", _stub_history_adjusted)
    monkeypatch.setattr(
        Index, "history", lambda self, **kw: pd.DataFrame(columns=["Close"])
    )

    with pytest.raises(ValueError, match="single-sided series"):
        Ticker("BIMAS").relative_strength(index="XU100")


def test_ticker_relative_strength_raises_when_index_fetch_fails(monkeypatch):
    from borsapy.index import Index

    def boom(self, **kw):
        raise RuntimeError("provider down")

    monkeypatch.setattr(Ticker, "history_adjusted", _stub_history_adjusted)
    monkeypatch.setattr(Index, "history", boom)

    with pytest.raises(ValueError, match="could not fetch index"):
        Ticker("BIMAS").relative_strength(index="XU100")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_exports_reachable():
    assert bp.relative_strength_series is relative_strength_series
    assert bp.relative_strength_events is relative_strength_events
    assert bp.RSEvent is RSEvent
    assert hasattr(bp.Ticker, "relative_strength")
