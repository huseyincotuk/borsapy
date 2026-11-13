"""W3d testleri: Anchored VWAP (AVWAP).

Testler **çıktı düzeyindedir**; pinler elle hesaplanmıştır.
Float katmanı olduğu için karşılaştırmalar ``pytest.approx`` (rel=1e-9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from borsapy.technical import (
    TechnicalAnalyzer,
    add_indicators,
    calculate_avwap,
    calculate_avwap_bands,
    calculate_vwap,
)


def _frame(rows, days=None):
    """OHLCV frame; ``rows`` = (High, Low, Close, Volume)."""
    n = len(rows)
    days = days or [f"2026-11-0{i + 1}" for i in range(n)]
    return pd.DataFrame(
        rows,
        columns=["High", "Low", "Close", "Volume"],
        index=pd.to_datetime(days),
    )


#: 5 bars with hand-computable typical prices.
#: TP = (H + L + C) / 3
#:  bar0: (10+8+9)/3  = 9.0        vol 100
#:  bar1: (12+10+11)/3 = 11.0      vol 200
#:  bar2: (14+12+13)/3 = 13.0      vol 300
#:  bar3: (16+14+15)/3 = 15.0      vol 100
#:  bar4: (18+16+17)/3 = 17.0      vol 400
_ROWS = [
    (10.0, 8.0, 9.0, 100.0),
    (12.0, 10.0, 11.0, 200.0),
    (14.0, 12.0, 13.0, 300.0),
    (16.0, 14.0, 15.0, 100.0),
    (18.0, 16.0, 17.0, 400.0),
]


# ---------------------------------------------------------------------------
# REQ-001 — calculate_avwap, output-level pins
# ---------------------------------------------------------------------------


def test_avwap_from_int_anchor_matches_hand_calculation():
    """Çapa=index 2 → 3 bar ortalaması elle hesapla birebir."""
    df = _frame(_ROWS)
    result = calculate_avwap(df, anchor=2)

    # bar2: (13*300)/(300) = 13.0
    assert result.iloc[2] == pytest.approx(13.0, rel=1e-9)
    # bar3: (13*300 + 15*100)/400 = 5400/400 = 13.5
    assert result.iloc[3] == pytest.approx(13.5, rel=1e-9)
    # bar4: (13*300 + 15*100 + 17*400)/800 = (3900+1500+6800)/800 = 12200/800
    assert result.iloc[4] == pytest.approx(15.25, rel=1e-9)
    assert result.name == "AVWAP"


def test_avwap_bars_before_anchor_are_nan():
    """Çapa öncesi barlar NaN — uydurma değer yok."""
    df = _frame(_ROWS)
    result = calculate_avwap(df, anchor=2)

    assert result.iloc[0] != result.iloc[0]  # NaN
    assert result.iloc[1] != result.iloc[1]
    assert pd.isna(result.iloc[0]) and pd.isna(result.iloc[1])
    assert not pd.isna(result.iloc[2])


def test_avwap_anchor_zero_equals_cumulative_vwap():
    """Çapa=0 → kümülatif VWAP ile aynı olmalı (tek fark: isim)."""
    df = _frame(_ROWS)
    avwap = calculate_avwap(df, anchor=0)
    vwap = calculate_vwap(df)

    pd.testing.assert_series_equal(
        avwap.rename("VWAP"), vwap, check_names=True, rtol=1e-12
    )


def test_avwap_date_anchor_starts_at_first_bar_of_that_day():
    """Tarih çapası → o tarihin ilk barı."""
    df = _frame(_ROWS)
    result = calculate_avwap(df, anchor="2026-11-03")

    assert pd.isna(result.iloc[0]) and pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(13.0, rel=1e-9)


def test_avwap_timestamp_anchor_accepted():
    df = _frame(_ROWS)
    result = calculate_avwap(df, anchor=pd.Timestamp("2026-11-04"))
    assert pd.isna(result.iloc[2])
    assert result.iloc[3] == pytest.approx(15.0, rel=1e-9)


def test_avwap_multiple_bars_same_day_uses_first():
    """Aynı gün iki bar varsa o günün İLK barı çapa olur."""
    df = _frame(
        _ROWS,
        days=["2026-11-01", "2026-11-02", "2026-11-02", "2026-11-03", "2026-11-04"],
    )
    result = calculate_avwap(df, anchor="2026-11-02")
    # İlk 11-02 barı index 1 → 0 NaN, 1 dolu
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(11.0, rel=1e-9)


def test_avwap_negative_int_anchor_raises():
    df = _frame(_ROWS)
    with pytest.raises(ValueError):
        calculate_avwap(df, anchor=-1)


def test_avwap_out_of_range_int_anchor_raises():
    df = _frame(_ROWS)
    with pytest.raises(ValueError):
        calculate_avwap(df, anchor=5)


def test_avwap_date_anchor_on_non_datetime_index_raises():
    df = pd.DataFrame(
        _ROWS, columns=["High", "Low", "Close", "Volume"], index=[0, 1, 2, 3, 4]
    )
    with pytest.raises(ValueError):
        calculate_avwap(df, anchor="2026-11-03")


def test_avwap_missing_date_raises():
    df = _frame(_ROWS)
    with pytest.raises(ValueError):
        calculate_avwap(df, anchor="2026-12-25")


def test_avwap_zero_volume_bar_yields_nan_then_recovers():
    """Kümülatif hacim 0 olan barlar NaN; hacim gelince toparlanır (inf yok)."""
    rows = [
        (10.0, 8.0, 9.0, 0.0),   # hacimsiz -> kümülatif hacim 0
        (12.0, 10.0, 11.0, 0.0),  # hâlâ 0 -> NaN
        (14.0, 12.0, 13.0, 300.0),
    ]
    df = _frame(rows)
    result = calculate_avwap(df, anchor=0)

    assert pd.isna(result.iloc[0])  # kümülatif hacim 0 -> bölme yok
    assert pd.isna(result.iloc[1])
    # bar2: (9*0 + 11*0 + 13*300)/300 = 13.0
    assert result.iloc[2] == pytest.approx(13.0, rel=1e-9)
    assert np.isfinite(result.iloc[2])


def test_avwap_zero_volume_bar_keeps_previous_average():
    """Ara barda hacim 0 ise kümülatif hacim 0'a DÜŞMEZ: ortalama korunur."""
    rows = [
        (10.0, 8.0, 9.0, 100.0),
        (12.0, 10.0, 11.0, 0.0),  # hacimsiz ara bar
        (14.0, 12.0, 13.0, 300.0),
    ]
    df = _frame(rows)
    result = calculate_avwap(df, anchor=0)

    assert result.iloc[0] == pytest.approx(9.0, rel=1e-9)
    # bar1: (9*100 + 11*0)/100 = 9.0 — ortalama korunur (NaN değil)
    assert result.iloc[1] == pytest.approx(9.0, rel=1e-9)
    # bar2: (900 + 0 + 3900)/400 = 12.0
    assert result.iloc[2] == pytest.approx(12.0, rel=1e-9)


def test_avwap_never_returns_inf():
    df = _frame([(10.0, 8.0, 9.0, 0.0), (12.0, 10.0, 11.0, 0.0)])
    result = calculate_avwap(df, anchor=0)
    assert result.isna().all()
    assert not np.isinf(result.to_numpy(dtype="float64")).any()


def test_avwap_missing_column_returns_nan_series_with_warning():
    """Eksik sütun → nan serisi + görünür uyarı (mevcut vwap deseni)."""
    df = _frame(_ROWS).drop(columns=["Volume"])
    with pytest.warns(RuntimeWarning):
        result = calculate_avwap(df, anchor=0)
    assert result.isna().all()
    assert result.name == "AVWAP"


def test_avwap_empty_frame_returns_empty_series():
    df = pd.DataFrame(columns=["High", "Low", "Close", "Volume"])
    result = calculate_avwap(df, anchor=0)
    assert len(result) == 0
    assert result.name == "AVWAP"


def test_avwap_single_row_with_anchor_zero():
    df = _frame([(10.0, 8.0, 9.0, 100.0)])
    result = calculate_avwap(df, anchor=0)
    assert len(result) == 1
    assert result.iloc[0] == pytest.approx(9.0, rel=1e-9)


def test_avwap_custom_column_uses_that_price():
    """`column="Open"` → tipik fiyat (H+L+Open)/3."""
    df = _frame(_ROWS).assign(Open=[9.0, 11.0, 13.0, 15.0, 17.0])
    # bar2: (14+12+13)/3 = 13 — aynı çıkar (Open==Close burada); farklı olsun:
    df.loc[df.index[2], "Open"] = 20.0
    result = calculate_avwap(df, anchor=2, column="Open")
    assert result.iloc[2] == pytest.approx((14.0 + 12.0 + 20.0) / 3, rel=1e-9)


# ---------------------------------------------------------------------------
# REQ-002 — analyzer + add_indicators + export
# ---------------------------------------------------------------------------


def test_technical_analyzer_avwap_wraps_function():
    df = _frame(_ROWS)
    ta = TechnicalAnalyzer(df)
    result = ta.avwap(anchor=2)
    assert result.iloc[4] == pytest.approx(15.25, rel=1e-9)
    assert result.name == "AVWAP"


def test_add_indicators_avwap_without_anchor_raises():
    """Çapasız avwap → ValueError (varsayılan çapa UYDURULMAZ)."""
    df = _frame(_ROWS)
    with pytest.raises(ValueError, match="avwap requires an anchor"):
        add_indicators(df, indicators=["avwap"])


def test_add_indicators_avwap_with_anchor_output():
    df = _frame(_ROWS)
    out = add_indicators(df, indicators=["avwap"], avwap_anchor=2)
    assert "AVWAP" in out.columns
    assert out["AVWAP"].iloc[4] == pytest.approx(15.25, rel=1e-9)
    assert pd.isna(out["AVWAP"].iloc[0])


def test_export_is_reachable():
    import borsapy as bp

    assert bp.calculate_avwap is calculate_avwap
    assert bp.calculate_avwap_bands is calculate_avwap_bands


# ---------------------------------------------------------------------------
# REQ-003 — multi-anchor helper
# ---------------------------------------------------------------------------


def test_avwap_bands_column_names_and_count():
    df = _frame(_ROWS)
    frame = calculate_avwap_bands(df, [(0, "start"), (2, "breakout")])

    assert list(frame.columns) == ["avwap_start", "avwap_breakout"]
    assert len(frame) == len(df)
    # avwap_start çapa 0 → ilk bardan dolu
    assert frame["avwap_start"].iloc[0] == pytest.approx(9.0, rel=1e-9)
    # avwap_breakout çapa 2 → ilk iki bar NaN
    assert pd.isna(frame["avwap_breakout"].iloc[0])
    assert frame["avwap_breakout"].iloc[2] == pytest.approx(13.0, rel=1e-9)


def test_avwap_bands_bare_anchors_use_str_labels():
    df = _frame(_ROWS)
    frame = calculate_avwap_bands(df, [0])
    assert list(frame.columns) == ["avwap_0"]


def test_avwap_bands_empty_list_yields_empty_frame_with_warning():
    df = _frame(_ROWS)
    with pytest.warns(RuntimeWarning):
        frame = calculate_avwap_bands(df, [])
    assert frame.empty
    assert list(frame.columns) == []


# ---------------------------------------------------------------------------
# Regresyon — calculate_vwap byte-identical
# ---------------------------------------------------------------------------


def test_calculate_vwap_output_is_byte_identical():
    """Regresyon: mevcut calculate_vwap çıktısı değişmedi (elle hesap pini)."""
    df = _frame(_ROWS)
    result = calculate_vwap(df)

    # Elle hesap: kümülatif TP*V / kümülatif V
    expected = [
        9.0,
        (9.0 * 100 + 11.0 * 200) / 300,
        (9.0 * 100 + 11.0 * 200 + 13.0 * 300) / 600,
        (9.0 * 100 + 11.0 * 200 + 13.0 * 300 + 15.0 * 100) / 700,
        (9.0 * 100 + 11.0 * 200 + 13.0 * 300 + 15.0 * 100 + 17.0 * 400) / 1100,
    ]
    assert result.name == "VWAP"
    assert result.to_list() == pytest.approx(expected, rel=1e-12)


def test_calculate_vwap_missing_columns_still_returns_nan_series():
    """Regresyon: eksik sütunda vwap hâlâ sessiz nan serisi döner (uyarı YOK)."""
    df = _frame(_ROWS).drop(columns=["Volume"])
    result = calculate_vwap(df)
    assert result.isna().all()
    assert result.name == "VWAP"
