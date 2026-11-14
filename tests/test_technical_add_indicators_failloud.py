"""W3f testleri: add_indicators fail-loud (bilinmeyen ad sessizce yutulmaz).

Testler **çıktı düzeyindedir**. REQ-004 tablosunun tamamı:
bilinmeyen ad / yönlendirme / atomiklik / eksik veri / strip / None
byte-identical / 19 adlı regresyon matrisi / boş liste.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from borsapy.technical import (
    _SUPPORTED_INDICATORS,
    add_indicators,
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_obv,
    calculate_rsi,
    calculate_sma,
    calculate_stochastic,
    calculate_supertrend,
    calculate_vwap,
)


def _ohlcv(n: int = 60) -> pd.DataFrame:
    """Deterministik OHLCV frame (seed sabit)."""
    np.random.seed(7)
    close = 100.0 + np.cumsum(np.random.randn(n))
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(100_000, 500_000, n).astype(float)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=pd.date_range("2026-09-01", periods=n, freq="D"),
    )


#: Regresyon matrisi: (ad, kwargs, beklenen yeni sütunlar).
#: Sabitle eş örtüşme ayrı testle pinlenir.
_MATRIX: list[tuple[str, dict, list[str]]] = [
    ("sma", {}, ["SMA_20"]),
    ("ema", {}, ["EMA_12"]),
    ("rsi", {}, ["RSI_14"]),
    ("macd", {}, ["MACD", "MACD_Signal", "MACD_Hist"]),
    ("bollinger", {}, ["BB_Upper", "BB_Middle", "BB_Lower"]),
    ("atr", {}, ["ATR_14"]),
    ("stochastic", {}, ["Stoch_K", "Stoch_D"]),
    ("obv", {}, ["OBV"]),
    ("vwap", {}, ["VWAP"]),
    ("avwap", {"avwap_anchor": 0}, ["AVWAP"]),
    ("adx", {}, ["ADX_14"]),
    ("supertrend", {}, ["Supertrend", "Supertrend_Direction"]),
    ("hhv", {}, ["HHV_14"]),
    ("llv", {}, ["LLV_14"]),
    ("mom", {}, ["MOM_10"]),
    ("roc", {}, ["ROC_10"]),
    ("wma", {}, ["WMA_20"]),
    ("dema", {}, ["DEMA_20"]),
    ("tema", {}, ["TEMA_20"]),
]


# ---------------------------------------------------------------------------
# REQ-001 — bilinmeyen ad → ValueError (fail-loud)
# ---------------------------------------------------------------------------


def test_unknown_indicator_raises_with_supported_list():
    """'yokboyle' → ValueError; mesajda desteklenen liste ve örnek adlar."""
    df = _ohlcv()
    with pytest.raises(ValueError, match="unknown indicator 'yokboyle'"):
        add_indicators(df, ["yokboyle"])
    with pytest.raises(ValueError, match="supported"):
        add_indicators(df, ["yokboyle"])
    # Mesaj desteklenen adları sıralar (örnek iki ad ayrı ayrı görünür).
    with pytest.raises(ValueError, match="'sma'"):
        add_indicators(df, ["yokboyle"])
    with pytest.raises(ValueError, match="'tema'"):
        add_indicators(df, ["yokboyle"])


def test_typo_variant_raises_not_swallowed():
    """'rsr' (yanlış yazım) → ValueError (sessiz yutma YOK)."""
    df = _ohlcv()
    with pytest.raises(ValueError, match="unknown indicator 'rsr'"):
        add_indicators(df, ["rsr"])


def test_atomic_no_output_when_any_name_invalid():
    """['rsi', 'yokboyle'] → ValueError; hiçbir sütun eklenmez (atomiklik).

    Doğrulama TÜM adlar için hesaptan ÖNCE yapılır: geçerli adlar listede
    önce gelse bile fonksiyon kısmi sonuç üretemez (istisna fırlar) ve
    orijinal frame bozulmaz.
    """
    df = _ohlcv()
    original_columns = list(df.columns)

    with pytest.raises(ValueError, match="unknown indicator 'yokboyle'"):
        add_indicators(df, ["rsi", "macd", "yokboyle", "sma"])

    # Orijinal frame'e hiçbir sütun eklenmedi.
    assert list(df.columns) == original_columns
    assert "RSI_14" not in df.columns


def test_atomic_invalid_first_name_raises():
    """Geçersiz ad başta → hata; sıra fark etmez."""
    df = _ohlcv()
    with pytest.raises(ValueError):
        add_indicators(df, ["yokboyle", "rsi"])


def test_supported_but_missing_data_raises_with_column_names():
    """'atr' (High/Low yok) → ValueError; mesajda eksik sütun adları."""
    df = _ohlcv().drop(columns=["High", "Low"])
    with pytest.raises(ValueError, match="requires columns"):
        add_indicators(df, ["atr"])
    with pytest.raises(ValueError, match="'High'"):
        add_indicators(df, ["atr"])
    with pytest.raises(ValueError, match="'Low'"):
        add_indicators(df, ["atr"])


def test_obv_without_volume_raises():
    """'obv' Volume yok → ValueError (Volume adı mesajda)."""
    df = _ohlcv().drop(columns=["Volume"])
    with pytest.raises(ValueError, match="'Volume'"):
        add_indicators(df, ["obv"])


def test_vwap_without_volume_raises():
    """'vwap' Volume yok → ValueError (önceden sessizce düşerdi)."""
    df = _ohlcv().drop(columns=["Volume"])
    with pytest.raises(ValueError, match="missing from frame"):
        add_indicators(df, ["vwap"])


def test_none_path_with_missing_data_never_raises():
    """İstisna (bilinçli): indicators=None → eksik veride hata YOK.

    Varsayılan liste koşullu filtrelemeyle sütun üretir (mevcut davranış
    korunur — bu yolda fail-loud uygulamak regresyon olurdu).
    """
    df = _ohlcv().drop(columns=["Volume"])
    result = add_indicators(df)  # hata fırlamaz
    assert "SMA_20" in result.columns
    assert "ATR_14" in result.columns  # HLC var → eklenir
    assert "OBV" not in result.columns  # Volume yok → filtrelenir
    assert "VWAP" not in result.columns


# ---------------------------------------------------------------------------
# REQ-002 — büyük/küçük harf + boşluk toleransı
# ---------------------------------------------------------------------------


def test_whitespace_and_case_tolerated():
    """" RSI " → çalışır; RSI_14 eklenir (strip + lower)."""
    df = _ohlcv()
    result = add_indicators(df, [" RSI ", "SMA"])
    assert "RSI_14" in result.columns
    assert "SMA_20" in result.columns
    assert len(result.columns) == len(df.columns) + 2


# ---------------------------------------------------------------------------
# REQ-003 — avwap_sigma / avwap_bands özel yönlendirme
# ---------------------------------------------------------------------------


def test_avwap_sigma_redirects_to_direct_function():
    """'avwap_sigma' → ValueError + calculate_avwap_sigma_bands yönlendirmesi."""
    df = _ohlcv()
    with pytest.raises(ValueError, match="avwap_sigma"):
        add_indicators(df, ["avwap_sigma"])
    with pytest.raises(ValueError, match="calculate_avwap_sigma_bands"):
        add_indicators(df, ["avwap_sigma"])


def test_avwap_bands_redirects_to_direct_function():
    """'avwap_bands' → ValueError + calculate_avwap_bands yönlendirmesi."""
    df = _ohlcv()
    with pytest.raises(ValueError, match="avwap_bands"):
        add_indicators(df, ["avwap_bands"])
    with pytest.raises(ValueError, match="calculate_avwap_bands"):
        add_indicators(df, ["avwap_bands"])


def test_multi_column_names_are_not_supported_constants():
    """avwap_sigma/avwap_bands desteklenen ad DEĞİL (W3e kararı korunur)."""
    assert "avwap_sigma" not in _SUPPORTED_INDICATORS
    assert "avwap_bands" not in _SUPPORTED_INDICATORS
    assert len(_SUPPORTED_INDICATORS) == 19


# ---------------------------------------------------------------------------
# indicators=None — BYTE-IDENTICAL kanıtı
# ---------------------------------------------------------------------------


def test_indicators_none_default_path_byte_identical():
    """None → varsayılan liste, koşullu filtreleme; çıktı birebir pinli.

    Beklenen frame elle kurulur (df.copy() + her calculate_* çıktısı) ve
    assert_frame_equal ile karşılaştırılır — sütun SETİ, SIRASI ve DEĞERLERİ
    (dtype dahil) değişmedi.
    """
    df = _ohlcv()
    result = add_indicators(df)

    expected = df.copy()
    expected["SMA_20"] = calculate_sma(df, 20)
    expected["EMA_12"] = calculate_ema(df, 12)
    expected["RSI_14"] = calculate_rsi(df, 14)
    macd_df = calculate_macd(df)
    expected["MACD"] = macd_df["MACD"]
    expected["MACD_Signal"] = macd_df["Signal"]
    expected["MACD_Hist"] = macd_df["Histogram"]
    bb_df = calculate_bollinger_bands(df, 20)
    expected["BB_Upper"] = bb_df["BB_Upper"]
    expected["BB_Middle"] = bb_df["BB_Middle"]
    expected["BB_Lower"] = bb_df["BB_Lower"]
    expected["ATR_14"] = calculate_atr(df, 14)
    stoch_df = calculate_stochastic(df)
    expected["Stoch_K"] = stoch_df["Stoch_K"]
    expected["Stoch_D"] = stoch_df["Stoch_D"]
    expected["ADX_14"] = calculate_adx(df, 14)
    st_df = calculate_supertrend(df, 10, 3.0)
    expected["Supertrend"] = st_df["Supertrend"]
    expected["Supertrend_Direction"] = st_df["Supertrend_Direction"]
    expected["OBV"] = calculate_obv(df)
    expected["VWAP"] = calculate_vwap(df)

    pd.testing.assert_frame_equal(result, expected)


def test_indicators_none_without_hlc_filters_conditionally():
    """None + HLC yok → ATR/Stoch/ADX/Supertrend/VWAP filtrelenir (korunur)."""
    df = _ohlcv().drop(columns=["High", "Low"])
    result = add_indicators(df)
    assert "SMA_20" in result.columns
    assert "OBV" in result.columns  # Volume var
    assert "ATR_14" not in result.columns
    assert "ADX_14" not in result.columns
    assert "VWAP" not in result.columns
    assert "Supertrend" not in result.columns


# ---------------------------------------------------------------------------
# REQ-004 — regresyon matrisi (19 ad) + boş liste
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "kwargs", "expected_columns"),
    _MATRIX,
    ids=[item[0] for item in _MATRIX],
)
def test_supported_indicator_regression_matrix(name, kwargs, expected_columns):
    """Mevcut tüm gösterge adları teker teker çalışır (regresyon matrisi)."""
    df = _ohlcv()
    result = add_indicators(df, [name], **kwargs)
    for col in expected_columns:
        assert col in result.columns, f"{name}: {col} eksik"
    # Tam eklenen sütun sayısı — fazlası da eksik de yok (sessiz yutma
    # kanıtı: istenen ad tam olarak istenen sütunları üretir).
    assert len(result.columns) == len(df.columns) + len(expected_columns)
    # Orijinal frame mutasyona uğramadı.
    assert len(df.columns) == 5


def test_regression_matrix_covers_supported_constants():
    """Matris, _SUPPORTED_INDICATORS sabitiyle birebir örtüşür (tamamlanma).

    Yeni ad sabite eklenirse matris de güncellenmelidir — bu test bunu
    zorlar; aksi halde sabit ile test senkronizasyonu kırılır.
    """
    matrix_names = {item[0] for item in _MATRIX}
    assert matrix_names == set(_SUPPORTED_INDICATORS)


def test_empty_list_is_noop():
    """Boş liste [] → hata değil, boş değişiklik (meşru)."""
    df = _ohlcv()
    result = add_indicators(df, [])
    pd.testing.assert_frame_equal(result, df)
    assert list(result.columns) == list(df.columns)
