"""W3g testleri: tam düzeltilmiş fiyat serisi (split × temettü).

Testler **çıktı düzeyindedir** ve **offline**'dır; fixture'lar elle yazılmıştır.
Float katmanı olduğu için karşılaştırmalar ``pytest.approx`` (rel=1e-9).

El hesabı örnekleri:
  - saf fonksiyon: factor = adj/close; OHLC × factor; Volume'a DOKUNULMAZ
    (girdi neyse o çıkar; lot/split düzeltmesi çağırandan gelir)
  - BIMAS-benzeri: nominal 813,00 → 414,00 (oran 0,5092); baz seride
    yarıya inmiş olduğundan düzeltilmiş oran ≈ 1,0000
  - TUPRS-benzeri: nominal −%6,71; factor = (C−D)/C = 0,92807 → düzeltilmiş
    getiri ≈ +%0,52
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import borsapy as bp
from borsapy.ticker import (
    SPLIT_DETECT_RETURN_THRESHOLD,
    Ticker,
    build_adjusted_frame,
    to_raw_price,
)

_IDX = pd.to_datetime(
    ["2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15"]
)


def _raw(rows, index=None):
    """OHLCV frame; ``rows`` = (Open, High, Low, Close, Volume)."""
    if index is None:
        index = pd.date_range("2026-05-11", periods=len(rows), freq="D")
    return pd.DataFrame(
        rows,
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=index,
        dtype="float64",
    )


# ---------------------------------------------------------------------------
# REQ-001 — saf fonksiyon: çarpan uygulaması
# ---------------------------------------------------------------------------


def test_pure_function_multiplies_ohlc_and_passes_volume_through():
    """OHLC = raw × factor (satır bazlı); Volume'a DOKUNULMAZ (pass-through)."""
    raw = _raw(
        [
            (10.0, 11.0, 9.0, 10.0, 100.0),
            (12.0, 13.0, 11.0, 12.0, 200.0),
        ]
    )
    adj_close = pd.Series([9.0, 12.0], index=raw.index)  # factor 0.9 / 1.0

    frame, meta = build_adjusted_frame(raw, adj_close)

    # El hesabı: bar0 factor = 9/10 = 0.9
    assert frame["Open"].iloc[0] == pytest.approx(9.0, rel=1e-9)
    assert frame["High"].iloc[0] == pytest.approx(9.9, rel=1e-9)
    assert frame["Low"].iloc[0] == pytest.approx(8.1, rel=1e-9)
    assert frame["Close"].iloc[0] == pytest.approx(9.0, rel=1e-9)
    # bar1 factor = 12/12 = 1.0 → değişmez
    assert frame["Close"].iloc[1] == pytest.approx(12.0, rel=1e-9)

    # Volume pass-through: saf fonksiyon girdisini değiştirmeden geçirir
    pd.testing.assert_series_equal(
        frame["Volume"], raw["Volume"], check_names=False, check_exact=True
    )
    assert meta["volume_policy"].startswith("split_lot_adjusted")
    assert meta["factor_min"] == pytest.approx(0.9, rel=1e-12)
    assert meta["factor_max"] == pytest.approx(1.0, rel=1e-12)


def test_output_columns_and_order_are_fixed():
    raw = _raw([(10.0, 11.0, 9.0, 10.0, 100.0)])
    frame, _ = build_adjusted_frame(raw, pd.Series([10.0], index=raw.index))
    assert list(frame.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "raw_close",
    ]


def test_factor_one_leaves_prices_identical():
    """factor == 1 (temettü yok — PETKM tipi) → düzeltilmiş == adjust=True."""
    raw = _raw(
        [
            (10.0, 11.0, 9.0, 10.0, 100.0),
            (10.5, 11.5, 10.0, 10.5, 150.0),
            (11.0, 12.0, 10.5, 11.0, 120.0),
        ]
    )
    adj_close = raw["Close"].copy()  # Adj Close == Close → factor 1

    frame, meta = build_adjusted_frame(raw, adj_close)

    for col in ["Open", "High", "Low", "Close"]:
        pd.testing.assert_series_equal(
            frame[col], raw[col], check_names=False, check_exact=True
        )
    assert meta["factor_min"] == pytest.approx(1.0, rel=1e-12)
    assert meta["factor_max"] == pytest.approx(1.0, rel=1e-12)
    # Dürüstlük: etkisiz temettü katmanı uyarısı görünür
    assert any("factor == 1" in w for w in meta["warnings"])


def test_bonus_issue_day_ratio_is_one_in_adjusted_series():
    """BIMAS 14.05.2026 tipi: baz seri zaten yarıya inmiş → oran ≈ 1,0.

    Nominal seride oran 414,00/813,00 ≈ 0,5092 (sahte −%49 düşüş).
    """
    nominal = pd.Series([400.0, 406.5, 420.0, 414.0, 410.0], index=_IDX)
    base = _raw(
        [(c, c, c, c, 1000.0) for c in [400.0, 406.5, 420.0, 414.0, 410.0]]
    )

    frame, meta = build_adjusted_frame(base, base["Close"], nominal_close=nominal)

    ratio = frame["Close"].iloc[3] / frame["Close"].iloc[2]
    assert ratio == pytest.approx(1.0, abs=0.05)  # REQ-002.1 toleransı
    assert ratio > 0.95
    # Nominal seride sahte düşüş yok burada; ayrı test aşağıda


def test_bonus_issue_detected_from_nominal_series():
    """BIMAS gerçek ölçümü: nominal 813,00 → 414,00 → split tespit edilir."""
    idx = pd.to_datetime(["2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15"])
    base_close = [806.0, 813.0, 414.0, 417.0]  # baz: 13.05'te 813 (split öncesi)
    nominal = pd.Series(base_close, index=idx)
    raw = _raw(
        [(c, c, c, c, 1000.0) for c in base_close], index=idx
    )

    frame, meta = build_adjusted_frame(raw, raw["Close"], nominal_close=nominal)

    events = meta["split_days_detected"]["events"]
    assert len(events) == 1
    assert events[0]["date"] == "2026-05-14"
    assert events[0]["return_pct"] == pytest.approx(-49.077, abs=0.01)
    assert events[0]["implied_split_factor"] == pytest.approx(2.0, abs=0.05)
    # Stock Splits kolonuna güvenilmediği künyede yazılı
    assert "Stock Splits" in meta["split_days_detected"]["note"]
    assert meta["split_days_detected"]["threshold_pct"] == pytest.approx(-15.0)


def test_dividend_day_return_above_minus_two_percent():
    """TUPRS 27.09.2024 tipi: ham −%6,71 → düzeltilmiş > −%2."""
    amount = 6.71
    c0, c1 = 100.0, 93.29  # ham seri: −%6,71
    adj0 = c0 * (c1 - amount) / c1  # 92.8071
    raw = _raw([(c0, c0, c0, c0, 500.0), (c1, c1, c1, c1, 600.0)])
    adj_close = pd.Series([adj0, c1], index=raw.index)

    frame, meta = build_adjusted_frame(raw, adj_close)

    got = frame["Close"].iloc[1] / frame["Close"].iloc[0] - 1.0
    assert got == pytest.approx(0.0052, abs=1e-3)
    assert got > -0.02  # REQ-002.2
    # Ham seride bu hareket −%6,71 idi (kanıt: karşılaştırma)
    assert (c1 / c0 - 1.0) == pytest.approx(-0.0671, abs=1e-9)
    # Temettü günü künyede türetim olarak işaretli
    days = meta["dividend_days"]
    assert len(days) == 1
    assert days[0]["date"] == "2026-05-12"
    assert days[0]["amount_implied"] == pytest.approx(amount, abs=1e-6)
    assert "TÜRETİM" in days[0]["amount_basis"]


def test_no_false_split_in_adjusted_series():
    """Düzeltilmiş seride getiri < −%15 gün YOK (5 sembol taramasının birimi)."""
    idx = pd.to_datetime(pd.date_range("2025-01-01", periods=20, freq="B"))
    close = np.linspace(100.0, 120.0, 20)
    nominal = pd.Series(close, index=idx)
    # Nominal seride sahte %50 düşüş (bedelsiz), baz seride düz
    nominal.iloc[10:] = nominal.iloc[10:] / 2.0
    base_close = np.linspace(100.0, 120.0, 20)
    raw = _raw(
        [(c, c, c, c, 1000.0) for c in base_close], index=idx
    )

    frame, meta = build_adjusted_frame(raw, raw["Close"], nominal_close=nominal)

    returns = frame["Close"].pct_change(fill_method=None).dropna()
    assert (returns < SPLIT_DETECT_RETURN_THRESHOLD).sum() == 0
    assert len(meta["split_days_detected"]["events"]) == 1  # nominalde var


# ---------------------------------------------------------------------------
# raw_close + to_raw_price
# ---------------------------------------------------------------------------


def test_raw_close_and_to_raw_price_are_consistent():
    """raw_close gerçek (grafik) fiyatı taşır; to_raw_price ile tutarlı."""
    raw = _raw(
        [
            (10.0, 11.0, 9.0, 10.0, 100.0),
            (11.0, 12.0, 10.0, 11.0, 110.0),
        ]
    )
    adj_close = pd.Series([9.0, 11.0], index=raw.index)  # factor 0.9 / 1.0

    frame, _ = build_adjusted_frame(raw, adj_close)

    # raw_close = split-düzeltilmiş, temettü çarpanı uygulanmamış fiyat
    pd.testing.assert_series_equal(
        frame["raw_close"], raw["Close"], check_names=False, check_exact=True
    )
    # Düzeltilmiş seviyeyi geri çevir
    assert to_raw_price(frame["Close"].iloc[0], 0.9) == pytest.approx(10.0, rel=1e-12)
    assert to_raw_price(9.0, 0.9) == pytest.approx(10.0, rel=1e-12)


@pytest.mark.parametrize("bad", [0, -0.5, float("nan"), float("inf")])
def test_to_raw_price_rejects_bad_factor(bad):
    with pytest.raises(ValueError):
        to_raw_price(9.0, bad)


# ---------------------------------------------------------------------------
# Künye
# ---------------------------------------------------------------------------


def test_meta_has_all_required_fields():
    raw = _raw(
        [
            (10.0, 11.0, 9.0, 10.0, 100.0),
            (11.0, 12.0, 10.0, 11.0, 110.0),
        ]
    )
    _, meta = build_adjusted_frame(raw, pd.Series([10.0, 11.0], index=raw.index))

    for key in [
        "source",
        "split_adjusted",
        "dividend_adjusted",
        "factor_min",
        "factor_max",
        "dividend_days",
        "split_days_detected",
        "volume_policy",
        "row_count",
        "first",
        "last",
        "frame_sha256",
        "warnings",
    ]:
        assert key in meta, f"künye alanı eksik: {key}"

    assert meta["source"] == "tradingview (split-adjusted) × dividend factor"
    assert meta["split_adjusted"] is True
    assert meta["dividend_adjusted"] is True
    assert meta["row_count"] == 2
    assert meta["first"] == "2026-05-11"
    assert meta["last"] == "2026-05-12"
    assert len(meta["frame_sha256"]) == 64


def test_frame_sha256_is_deterministic():
    """İki koşum aynı digest'i verir (mühür determinizmi)."""
    raw = _raw(
        [
            (10.0, 11.0, 9.0, 10.0, 100.0),
            (11.0, 12.0, 10.0, 11.0, 110.0),
            (12.0, 13.0, 11.0, 12.0, 120.0),
        ]
    )
    adj = pd.Series([9.0, 11.0, 11.4], index=raw.index)

    first, meta1 = build_adjusted_frame(raw, adj)
    second, meta2 = build_adjusted_frame(raw, adj)

    assert meta1["frame_sha256"] == meta2["frame_sha256"]
    pd.testing.assert_frame_equal(first, second)


def test_nan_factor_row_stays_nan_without_inf():
    """factor NaN ise satır NaN; inf asla dönmez (uydurma yok)."""
    raw = _raw(
        [
            (10.0, 11.0, 9.0, 10.0, 100.0),
            (11.0, 12.0, 10.0, 11.0, 110.0),
        ]
    )
    adj_close = pd.Series([np.nan, 11.0], index=raw.index)

    frame, meta = build_adjusted_frame(raw, adj_close)

    assert frame["Close"].iloc[0] != frame["Close"].iloc[0]  # NaN
    for col in ["Open", "High", "Low", "Close"]:
        values = frame[col].to_numpy(dtype="float64")
        assert not np.isinf(values[~np.isnan(values)]).any()
    assert any("NaN factor" in w for w in meta["warnings"])


def test_empty_frame_returns_empty_with_meta_row_count_zero():
    raw = _raw([]).iloc[0:0]
    frame, meta = build_adjusted_frame(raw, pd.Series(dtype="float64"))

    assert len(frame) == 0
    assert list(frame.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "raw_close",
    ]
    assert meta["row_count"] == 0
    assert meta["first"] is None and meta["last"] is None
    assert meta["frame_sha256"] is not None


def test_missing_adj_close_raises_visible_error():
    """Adj Close yok → görünür ValueError (sessiz tek-katman YASAK)."""
    raw = _raw([(10.0, 11.0, 9.0, 10.0, 100.0)])
    with pytest.raises(ValueError, match="Adj Close"):
        build_adjusted_frame(raw, None)
    with pytest.raises(ValueError, match="Adj Close"):
        build_adjusted_frame(raw, pd.Series(dtype="float64"))


def test_missing_close_column_raises():
    raw = _raw([(10.0, 11.0, 9.0, 10.0, 100.0)]).drop(columns=["Close"])
    with pytest.raises(ValueError, match="Close"):
        build_adjusted_frame(raw, pd.Series([1.0]))


def test_pure_function_volume_is_a_verbatim_pass_through():
    """Saf fonksiyon Volume'u DEĞİŞTİRMEZ (girdi = çıktı)."""
    raw = _raw(
        [
            (10.0, 11.0, 9.0, 10.0, 1234.0),
            (11.0, 12.0, 10.0, 11.0, 98765.0),
            (12.0, 13.0, 11.0, 12.0, 42.0),
        ]
    )
    adj = pd.Series([9.0, 11.0, 11.4], index=raw.index)
    frame, _ = build_adjusted_frame(raw, adj)

    pd.testing.assert_series_equal(
        frame["Volume"], raw["Volume"], check_names=False, check_exact=True
    )


# ---------------------------------------------------------------------------
# Ticker.history_adjusted — yönlendirme + kwargs koruması (ağ yok)
# ---------------------------------------------------------------------------


def test_history_adjusted_rejects_managed_kwargs_before_network():
    """adjust/auto_adjust/actions verilirse hata (sessiz yok sayma yok)."""
    stock = Ticker("THYAO")
    for kw in [{"adjust": True}, {"auto_adjust": True}, {"actions": True}]:
        with pytest.raises(ValueError, match="manages"):
            stock.history_adjusted(period="1mo", **kw)


def test_history_adjusted_end_to_end_with_stubbed_history(monkeypatch):
    """Ağ yok: sahte history() üç farklı kwarg kümesini ayrı döndürür."""
    idx = pd.to_datetime(["2026-05-11", "2026-05-12", "2026-05-13"])
    nominal = _raw(
        [
            (813.0, 813.0, 813.0, 813.0, 100.0),
            (414.0, 414.0, 414.0, 414.0, 200.0),
            (417.0, 417.0, 417.0, 417.0, 150.0),
        ],
        index=idx,
    )
    base = _raw(
        [
            (406.5, 406.5, 406.5, 406.5, 100.0),
            (414.0, 414.0, 414.0, 414.0, 200.0),
            (417.0, 417.0, 417.0, 417.0, 150.0),
        ],
        index=idx,
    )

    def fake_history(**kwargs):
        if kwargs.get("auto_adjust"):
            out = base.copy()
            out["Adj Close"] = base["Close"]
            return out
        if kwargs.get("adjust") is False:
            return nominal.copy()
        return base.copy()

    stock = Ticker("BIMAS")
    monkeypatch.setattr(stock, "history", fake_history)

    frame, meta = stock.history_adjusted(period="1mo")

    assert list(frame.columns)[-1] == "raw_close"
    assert meta["source"] == "tradingview (split-adjusted) × dividend factor"
    ratio = frame["Close"].iloc[1] / frame["Close"].iloc[0]
    assert ratio == pytest.approx(1.0185, abs=0.01)  # BIMAS 14.05 ölçümü
    events = meta["split_days_detected"]["events"]
    assert len(events) == 1 and events[0]["date"] == "2026-05-12"


def test_history_adjusted_empty_history_returns_empty_meta(monkeypatch):
    stock = Ticker("XU100")
    empty = pd.DataFrame(
        columns=["Open", "High", "Low", "Close", "Volume"]
    )
    monkeypatch.setattr(stock, "history", lambda **kwargs: empty)

    frame, meta = stock.history_adjusted()
    assert frame.empty
    assert meta["row_count"] == 0


def test_exports_reachable():
    assert bp.build_adjusted_frame is build_adjusted_frame
    assert bp.to_raw_price is to_raw_price
    assert bp.Ticker.history_adjusted is not None


# ---------------------------------------------------------------------------
# Volume semantiği — split/lot düzeltmesi (dürüst künye pini)
# ---------------------------------------------------------------------------


def test_volume_policy_documents_split_lot_adjustment() -> None:
    """Künye, Volume'un split/lot düzeltilmiş olduğunu AÇIKÇA söyler.

    ``history_adjusted`` tabanı ``history(adjust=True)``tir: %100 bedelsizde
    lot sayısı ikiye katlandığı için o serinin Volume'u eski günlerde YARI
    bazda kalır. Ham (``adjust=False``) hacim kullanılsaydı bedelsiz gününden
    sonra yapay bir sıçrama oluşur ve göreli hacim (rel-volume) yanlış
    hesaplanırdı — 078a klimaks tespitinde sahte SC üretirdi.
    """
    raw = _raw(
        [
            (10.0, 11.0, 9.0, 10.0, 1234.0),
            (11.0, 12.0, 10.0, 11.0, 98765.0),
        ]
    )
    adj = pd.Series([10.0, 11.0], index=raw.index)
    _, meta = build_adjusted_frame(raw, adj)

    policy = meta["volume_policy"]
    assert policy.startswith("split_lot_adjusted"), policy
    assert "dividend-independent" in policy
    assert "not TL turnover" in policy
    # eski yanıltıcı ifade geri gelmemeli
    assert "raw (not adjusted)" not in policy
