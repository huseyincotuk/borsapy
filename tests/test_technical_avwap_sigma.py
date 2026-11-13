"""W3e testleri: AVWAP ±σ bantları (çapa-bazlı oynaklık zarfı).

Testler **çıktı düzeyindedir**; pinler elle hesaplanmıştır.
Float katmanı olduğu için karşılaştırmalar ``pytest.approx`` (rel=1e-9).

El hesabı (çapa=2, _ROWS fixture'ı):
  TP  = 9, 11, 13, 15, 17
  AVWAP: bar2=13.0, bar3=13.5, bar4=15.25
  sapma: bar2=0, bar3=1.5, bar4=1.75
  σ (ddof=0, kümülatif):
    bar2: std[0]              = 0
    bar3: std[0, 1.5]         = 0.75
    bar4: std[0, 1.5, 1.75]   = sqrt(43/72) ≈ 0.7728105127
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from borsapy.technical import (
    TechnicalAnalyzer,
    calculate_avwap,
    calculate_avwap_sigma_bands,
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


#: 5 bars — W3d fixture'ı ile aynı (TP = 9, 11, 13, 15, 17).
_ROWS = [
    (10.0, 8.0, 9.0, 100.0),
    (12.0, 10.0, 11.0, 200.0),
    (14.0, 12.0, 13.0, 300.0),
    (16.0, 14.0, 15.0, 100.0),
    (18.0, 16.0, 17.0, 400.0),
]

#: El hesabı sabitleri (çapa=2).
_SIGMA_BAR3 = 0.75
_SIGMA_BAR4 = math.sqrt(43.0 / 72.0)  # ≈ 0.7728105127


# ---------------------------------------------------------------------------
# REQ-001 — calculate_avwap_sigma_bands, output-level pins
# ---------------------------------------------------------------------------


def test_sigma_bands_match_hand_calculation():
    """5 barlık seri, çapa=2 → avwap/upper/lower el-hesabıyla birebir."""
    df = _frame(_ROWS)
    result = calculate_avwap_sigma_bands(df, anchor=2)

    assert list(result.columns) == ["avwap", "upper", "lower", "sigma"]

    # bar2: avwap=13.0, σ=0 → upper=lower=13.0
    assert result["avwap"].iloc[2] == pytest.approx(13.0, rel=1e-9)
    assert result["upper"].iloc[2] == pytest.approx(13.0, rel=1e-9)
    assert result["lower"].iloc[2] == pytest.approx(13.0, rel=1e-9)

    # bar3: avwap=13.5, σ=0.75
    assert result["avwap"].iloc[3] == pytest.approx(13.5, rel=1e-9)
    assert result["upper"].iloc[3] == pytest.approx(13.5 + _SIGMA_BAR3, rel=1e-9)
    assert result["lower"].iloc[3] == pytest.approx(13.5 - _SIGMA_BAR3, rel=1e-9)

    # bar4: avwap=15.25, σ=sqrt(43/72)
    assert result["avwap"].iloc[4] == pytest.approx(15.25, rel=1e-9)
    assert result["upper"].iloc[4] == pytest.approx(15.25 + _SIGMA_BAR4, rel=1e-9)
    assert result["lower"].iloc[4] == pytest.approx(15.25 - _SIGMA_BAR4, rel=1e-9)


def test_sigma_column_is_constant_provenance_stamp():
    """`sigma` sütunu = kullanılan çarpan (künye); her satırda sabit."""
    df = _frame(_ROWS)
    result = calculate_avwap_sigma_bands(df, anchor=2, sigma=1.0)
    assert result["sigma"].to_list() == [1.0] * len(df)

    result2 = calculate_avwap_sigma_bands(df, anchor=2, sigma=2.0)
    assert result2["sigma"].to_list() == [2.0] * len(df)


def test_bars_before_anchor_are_nan():
    """Çapa öncesi barlar: avwap, upper, lower → NaN (uydurma yok)."""
    df = _frame(_ROWS)
    result = calculate_avwap_sigma_bands(df, anchor=2)

    for col in ["avwap", "upper", "lower"]:
        assert pd.isna(result[col].iloc[0])
        assert pd.isna(result[col].iloc[1])
    assert not pd.isna(result["avwap"].iloc[2])


def test_sigma_2_doubles_band_width():
    """sigma=2.0 → bant genişliği tam 2 kat (aynı σ_t)."""
    df = _frame(_ROWS)
    one = calculate_avwap_sigma_bands(df, anchor=2, sigma=1.0)
    two = calculate_avwap_sigma_bands(df, anchor=2, sigma=2.0)

    width_one = one["upper"] - one["lower"]
    width_two = two["upper"] - two["lower"]
    assert width_two.iloc[3] == pytest.approx(2.0 * width_one.iloc[3], rel=1e-9)
    assert width_two.iloc[4] == pytest.approx(2.0 * width_one.iloc[4], rel=1e-9)
    # El hesabı teyidi: bar3 → 1.5 ve 3.0
    assert width_one.iloc[3] == pytest.approx(1.5, rel=1e-9)
    assert width_two.iloc[3] == pytest.approx(3.0, rel=1e-9)


def test_avwap_column_equals_calculate_avwap_output():
    """avwap sütunu = calculate_avwap çıktısı (tek hesap yolu), birebir."""
    df = _frame(_ROWS)
    result = calculate_avwap_sigma_bands(df, anchor=2)
    expected = calculate_avwap(df, anchor=2)

    pd.testing.assert_series_equal(
        result["avwap"],
        expected,
        check_names=False,
        check_exact=True,
    )


def test_single_bar_anchor_last_sigma_zero():
    """Tek bar (çapa = son bar, ddof=0): σ=0 → upper=lower=avwap (NaN değil)."""
    df = _frame(_ROWS)
    result = calculate_avwap_sigma_bands(df, anchor=4)

    # AVWAP = 17.0 (tek bar), sapma = 0, σ = 0
    assert result["avwap"].iloc[4] == pytest.approx(17.0, rel=1e-9)
    assert result["upper"].iloc[4] == pytest.approx(17.0, rel=1e-9)
    assert result["lower"].iloc[4] == pytest.approx(17.0, rel=1e-9)
    assert not pd.isna(result["upper"].iloc[4])
    assert not pd.isna(result["lower"].iloc[4])


def test_zero_volume_bands_cascade_nan_no_inf():
    """Hacim 0 → sapma NaN → bantlar o bar ve sonrası NaN (zincirleme).

    avwap sütunu calculate_avwap'tan geldiği için W3d gibi toparlanır
    (tek hesap yolu); zincirleme NaN yalnız bant hesabındadır.
    """
    rows = [
        (10.0, 8.0, 9.0, 0.0),    # kümülatif hacim 0 → AVWAP NaN
        (12.0, 10.0, 11.0, 0.0),  # hâlâ 0 → NaN
        (14.0, 12.0, 13.0, 300.0),  # AVWAP toparlanır (W3d paritesi)
    ]
    df = _frame(rows)
    result = calculate_avwap_sigma_bands(df, anchor=0)

    # avwap sütunu W3d ile birebir: bar2 = 13.0
    assert result["avwap"].iloc[2] == pytest.approx(13.0, rel=1e-9)
    # Bantlar: bar0'daki NaN sapma kümeyi bozdu → o bar ve sonrası NaN
    for col in ["upper", "lower"]:
        assert result[col].isna().all()
    # inf asla
    for col in ["avwap", "upper", "lower"]:
        values = result[col].to_numpy(dtype="float64")
        assert not np.isinf(values[~np.isnan(values)]).any()


@pytest.mark.parametrize("bad_sigma", [0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_sigma_raises(bad_sigma):
    """sigma=0 / negatif / nan / inf → ValueError."""
    df = _frame(_ROWS)
    with pytest.raises(ValueError, match="sigma"):
        calculate_avwap_sigma_bands(df, anchor=2, sigma=bad_sigma)


def test_negative_ddof_raises():
    df = _frame(_ROWS)
    with pytest.raises(ValueError, match="ddof"):
        calculate_avwap_sigma_bands(df, anchor=2, ddof=-1)


def test_ddof1_single_observation_nan_then_recovers():
    """ddof=1 (n=1): σ tanımsız → çapa barı NaN, sonrası toparlanır."""
    df = _frame(_ROWS)
    result = calculate_avwap_sigma_bands(df, anchor=2, ddof=1)

    # bar2: tek gözlem, ddof=1 → σ NaN
    assert pd.isna(result["upper"].iloc[2])
    assert pd.isna(result["lower"].iloc[2])
    # bar3: std[0, 1.5] ddof=1 = sqrt(1.125)
    sigma_ddof1 = math.sqrt(1.125)
    assert result["upper"].iloc[3] == pytest.approx(13.5 + sigma_ddof1, rel=1e-9)
    assert result["lower"].iloc[3] == pytest.approx(13.5 - sigma_ddof1, rel=1e-9)
    # bar4 dolu (toparlanma)
    assert not pd.isna(result["upper"].iloc[4])


def test_missing_columns_nan_frame_with_warning():
    """Eksik sütun → NaN frame + RuntimeWarning (W3d deseni)."""
    df = _frame(_ROWS).drop(columns=["Volume"])
    with pytest.warns(RuntimeWarning):
        result = calculate_avwap_sigma_bands(df, anchor=2)

    assert list(result.columns) == ["avwap", "upper", "lower", "sigma"]
    for col in result.columns:
        assert result[col].isna().all()


def test_empty_frame_returns_empty_frame():
    """Boş df → boş frame (hata değil)."""
    df = pd.DataFrame(columns=["High", "Low", "Close", "Volume"])
    result = calculate_avwap_sigma_bands(df, anchor=0)

    assert len(result) == 0
    assert list(result.columns) == ["avwap", "upper", "lower", "sigma"]


def test_invalid_anchor_raises():
    """Çapa hataları calculate_avwap ile aynı şekilde yükselir."""
    df = _frame(_ROWS)
    with pytest.raises(ValueError):
        calculate_avwap_sigma_bands(df, anchor=-1)
    with pytest.raises(ValueError):
        calculate_avwap_sigma_bands(df, anchor=5)


# ---------------------------------------------------------------------------
# REQ-002 — analyzer yüzeyi + export
# ---------------------------------------------------------------------------


def test_technical_analyzer_wrapper_matches_function():
    df = _frame(_ROWS)
    ta = TechnicalAnalyzer(df)
    result = ta.avwap_sigma_bands(anchor=2)

    pd.testing.assert_frame_equal(
        result, calculate_avwap_sigma_bands(df, anchor=2)
    )
    assert result["upper"].iloc[3] == pytest.approx(13.5 + _SIGMA_BAR3, rel=1e-9)


def test_analyzer_wrapper_accepts_sigma_and_ddof():
    df = _frame(_ROWS)
    ta = TechnicalAnalyzer(df)
    result = ta.avwap_sigma_bands(anchor=2, sigma=2.0, ddof=1)

    sigma_ddof1 = math.sqrt(1.125)
    assert result["sigma"].iloc[3] == 2.0
    assert result["upper"].iloc[3] == pytest.approx(
        13.5 + 2.0 * sigma_ddof1, rel=1e-9
    )


def test_export_is_reachable():
    import borsapy as bp
    from borsapy import technical

    assert bp.calculate_avwap_sigma_bands is calculate_avwap_sigma_bands
    assert technical.calculate_avwap_sigma_bands is calculate_avwap_sigma_bands
    # W3d export'u bozulmadı
    assert bp.calculate_avwap is calculate_avwap


# ---------------------------------------------------------------------------
# Regresyon — calculate_avwap byte-identical (W3d değişmez)
# ---------------------------------------------------------------------------


def test_calculate_avwap_regression_pin():
    """Regresyon: calculate_avwap çıktısı W3d pinleriyle aynı (dokunulmadı)."""
    df = _frame(_ROWS)
    result = calculate_avwap(df, anchor=2)

    assert result.name == "AVWAP"
    assert pd.isna(result.iloc[0]) and pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(13.0, rel=1e-12)
    assert result.iloc[3] == pytest.approx(13.5, rel=1e-12)
    assert result.iloc[4] == pytest.approx(15.25, rel=1e-12)


def test_calculate_avwap_signature_unchanged():
    """Regresyon: calculate_avwap hâlâ tek zorunlu konumsal parametre + kwonly."""
    import inspect

    signature = inspect.signature(calculate_avwap)
    params = list(signature.parameters.values())
    assert [p.name for p in params] == ["df", "anchor", "column"]
    assert params[1].kind is inspect.Parameter.KEYWORD_ONLY
    assert params[2].default == "Close"
