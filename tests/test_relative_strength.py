"""W3i testleri: endeks göreli güç (RS vs XU100).

Testler **çıktı düzeyinde** ve **offline**'dır; fixture'lar elle yazılmıştır.
Float katmanı → karşılaştırmalar ``pytest.approx`` (rel=1e-9).

El hesabı örneği (sabit hisse, düşen endeks):
  hisse 10,10,10,10 · endeks 100,95,90,85 → oran 0,1 / 0,105263 / 0,111111 / 0,117647
  normalize → 100 / 105,263 / 111,111 / 117,647 ; change_pct = +17,6471
"""

from __future__ import annotations

import dataclasses
from datetime import time as dt_time

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


# ---------------------------------------------------------------------------
# REQ-004 — gün içi tekrarlanabilirlik: as_of kesimi + oluşmakta bar
#   (W3i yaması: trend etiketi açık seans barı yüzünden dönüyordu)
# ---------------------------------------------------------------------------

_TZ = "Europe/Istanbul"


def _today_bars(n: int = 5) -> pd.DatetimeIndex:
    """Bugünle biten, yerel saat 09:00 etiketli (tz-aware) barlar."""
    today = pd.Timestamp.now(tz=_TZ).normalize()
    days = [today - pd.Timedelta(days=n - 1 - i) for i in range(n)]
    return pd.DatetimeIndex([d + pd.Timedelta(hours=9) for d in days])


def _today_series(values, n=5):
    return pd.Series(values, index=_today_bars(n), dtype="float64")


def _local_now(hour: int, minute: int = 0) -> pd.Timestamp:
    today = pd.Timestamp.now(tz=_TZ).normalize()
    return today + pd.Timedelta(hours=hour, minutes=minute)


_OLD = pd.to_datetime(
    ["2019-03-04", "2019-03-05", "2019-03-06", "2019-03-07", "2019-03-08"]
)


def test_as_of_cutoff_matches_manual_truncation():
    """as_of → yalnız o ana kadarki barlar; sonuç elle kesmeyle BİREBİR."""
    stock = _s([10.0, 10.0, 10.0, 10.0, 10.0], _OLD)
    index = _s([100.0, 95.0, 90.0, 85.0, 80.0], _OLD)

    rs_cut, meta = relative_strength_series(stock, index, normalize=False, as_of="2019-03-06")
    rs_manual, _ = relative_strength_series(
        stock.iloc[:3], index.iloc[:3], normalize=False
    )

    assert meta["n_dropped_after_as_of"] == 2
    assert meta["bars_used"] == 3
    assert rs_cut.tolist() == pytest.approx(rs_manual.tolist(), rel=1e-12)
    assert meta["last_bar_date"] == "2019-03-06"


def test_as_of_with_explicit_time_is_honored():
    """Saatli as_of tam o an sınırdır (bar diliminde) — o günün barı DÜŞER."""
    idx = pd.DatetimeIndex(
        [pd.Timestamp(f"2019-03-0{d} 09:00", tz=_TZ) for d in (4, 5, 6, 7)]
    )
    stock = pd.Series([10.0, 10.0, 10.0, 10.0], index=idx, dtype="float64")
    index = pd.Series([100.0, 95.0, 90.0, 85.0], index=idx, dtype="float64")

    rs, meta = relative_strength_series(
        stock, index, normalize=False, as_of=pd.Timestamp("2019-03-06 08:59")
    )

    assert meta["n_dropped_after_as_of"] == 2
    assert len(rs) == 2
    assert meta["last_bar_date"] == "2019-03-05"


def test_as_of_makes_result_reproducible():
    """Aynı as_of iki koşuda aynı sha256 → gün içi sonuç SABİT."""
    stock = _today_series([10.0, 10.0, 10.0, 10.0, 10.0])
    index = _today_series([100.0, 95.0, 90.0, 85.0, 80.0])
    cutoff = str(_today_bars(5)[-2].date())  # dün: bugünün barı dışarıda

    _, m1 = relative_strength_series(stock, index, as_of=cutoff, lookback=2)
    _, m2 = relative_strength_series(stock, index, as_of=cutoff, lookback=2)

    assert m1["series_sha256"] == m2["series_sha256"]
    assert m1["bars_used"] == 4
    assert m1["last_bar_date"] == cutoff


def test_historical_last_bar_is_not_partial():
    """Geçmiş barlarda son bar KAPANMIŞ — session_close gerekmez."""
    stock = _s([10.0, 10.0, 10.0, 10.0], _OLD[:4])
    index = _s([100.0, 95.0, 90.0, 85.0], _OLD[:4])

    rs, meta = relative_strength_series(stock, index)

    assert meta["last_bar_partial"] is False
    assert meta["dropped_forming_bar"] is False
    assert meta["session_close"] is None
    assert meta["warnings"] == []
    assert len(rs) == 4


def test_last_bar_partial_false_when_session_closed():
    """Kapanış saatinden sonra son bar kapanmış sayılır → atılmaz."""
    stock = _today_series([10.0, 10.0, 10.0, 10.0, 10.0])
    index = _today_series([100.0, 95.0, 90.0, 85.0, 80.0])

    rs, meta = relative_strength_series(
        stock,
        index,
        normalize=False,
        closed_bars_only=True,
        session_close=dt_time(18, 0),
        now=_local_now(18, 30),
    )

    assert meta["last_bar_partial"] is False
    assert meta["dropped_forming_bar"] is False
    assert len(rs) == 5
    assert meta["last_bar_date"] == str(_today_bars(5)[-1].date())


def test_closed_bars_only_drops_forming_bar_and_matches_truncation():
    """Seans AÇIKKEN son bar atılır; sonuç elle kesmeyle BİREBİR (çıktı düzeyi)."""
    stock = _today_series([10.0, 10.0, 10.0, 10.0, 10.0])
    index = _today_series([100.0, 95.0, 90.0, 85.0, 80.0])

    rs_closed, meta = relative_strength_series(
        stock,
        index,
        normalize=False,
        closed_bars_only=True,
        session_close=dt_time(18, 0),
        now=_local_now(12, 0),
    )
    rs_manual, _ = relative_strength_series(
        stock.iloc[:-1], index.iloc[:-1], normalize=False
    )

    assert meta["last_bar_partial"] is True
    assert meta["dropped_forming_bar"] is True
    assert meta["bars_used"] == 4
    assert rs_closed.tolist() == pytest.approx(rs_manual.tolist(), rel=1e-12)


def test_closed_bars_only_without_session_close_raises():
    """Son bar bugüne ait + kapanış saati yok → TAHMİN YASAK (ValueError)."""
    stock = _today_series([10.0, 10.0, 10.0, 10.0], n=4)
    index = _today_series([100.0, 95.0, 90.0, 85.0], n=4)

    with pytest.raises(ValueError, match="session_close"):
        relative_strength_series(stock, index, closed_bars_only=True)


def test_today_bar_without_session_close_reports_none_and_warns():
    """Kapanış saati verilmezse durum VERİ YOK (None) + uyarı — sessiz sayı basılmaz."""
    stock = _today_series([10.0, 10.0, 10.0, 10.0], n=4)
    index = _today_series([100.0, 95.0, 90.0, 85.0], n=4)

    rs, meta = relative_strength_series(stock, index)

    assert meta["last_bar_partial"] is None
    assert meta["dropped_forming_bar"] is False
    assert len(rs) == 4  # varsayılan davranış DEĞİŞMEDİ
    assert any("forming" in w for w in meta["warnings"])


def test_forming_bar_can_flip_trend_sign():
    """BLOKLAYICI BULGU regresyonu: oluşmakta bar trend etiketini DÖNDÜRÜR.

    BIMAS/EKGYO'nun gerçek verideki davranışı: kapanmış seride eğim negatif
    (``zayif``), gün içi sıçrama dahil edilince eğim pozitif (``guclu``).
    """
    index = _today_series([100.0] * 5)
    stock = _today_series([100.0, 99.0, 98.0, 97.0, 105.0])  # oran: 1 / 0,99 / 0,98 / 0,97 / 1,05

    _, open_meta = relative_strength_series(stock, index, normalize=False, lookback=4)
    _, closed_meta = relative_strength_series(
        stock,
        index,
        normalize=False,
        lookback=4,
        closed_bars_only=True,
        session_close=dt_time(18, 0),
        now=_local_now(12, 0),
    )

    assert open_meta["trend"] == "guclu"      # açık bar dahil → yanıltıcı
    assert closed_meta["trend"] == "zayif"    # kapanmış barlar → gerçek
    assert open_meta["change_pct"] > 0
    assert closed_meta["change_pct"] < 0


def test_meta_reports_horizons():
    """Ufuklar künyede: trend 20 bar, change_pct tüm kullanılan barlar."""
    stock = _s([10.0, 10.0, 10.0, 10.0, 10.0], _OLD)
    index = _s([100.0, 95.0, 90.0, 85.0, 80.0], _OLD)

    _, meta = relative_strength_series(stock, index, lookback=20)

    assert meta["trend_horizon_bars"] == 5
    assert meta["change_pct_horizon_bars"] == 5
    assert meta["bars_used"] == 5


def test_ticker_relative_strength_passes_through_cutoff(monkeypatch):
    """Ticker.relative_strength yamayı uçtan uca taşır (as_of → 2 bar)."""
    from borsapy.index import Index

    def stub_index_history(self, period="1mo", interval="1d", start=None, end=None):
        return pd.DataFrame({"Close": [100.0, 95.0, 90.0, 85.0]}, index=_IDX)

    monkeypatch.setattr(Ticker, "history_adjusted", _stub_history_adjusted)
    monkeypatch.setattr(Index, "history", stub_index_history)

    rs, meta = Ticker("BIMAS").relative_strength(
        index="XU100", period="2y", as_of="2026-01-06", lookback=2
    )

    assert meta["bars_used"] == 2
    assert meta["n_dropped_after_as_of"] == 2
    assert meta["as_of"] == "2026-01-06"
    assert meta["stock_adjusted"] is True
    assert len(rs) == 2
