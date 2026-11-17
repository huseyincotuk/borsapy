"""W3h testleri: add_indicators ``**kwargs`` fail-loud.

W3f gösterge ADLARINI fail-loud yaptı; W3h PARAMETRELERİ yapar.
Testler **çıktı düzeyinde** ve **offline**'dır.

Kritik: ``indicators=None`` (varsayılan "hepsi" yolu) **byte-identical**
kalır — orada parametre doğrulaması UYGULANMAZ (REQ-002 istisnası).
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pandas as pd
import pytest

from borsapy import technical
from borsapy.technical import (
    _SUPPORTED_KWARGS,
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
    np.random.seed(11)
    close = 100.0 + np.cumsum(np.random.randn(n))
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(n) * 0.3,
            "High": close + np.abs(np.random.randn(n)),
            "Low": close - np.abs(np.random.randn(n)),
            "Close": close,
            "Volume": np.random.randint(100_000, 500_000, n).astype(float),
        },
        index=pd.date_range("2026-01-01", periods=n, freq="D"),
    )


#: Regresyon matrisi: (parametre, değer, gösterge, beklenen yeni sütun).
_KWARG_MATRIX: list[tuple[str, object, str, str]] = [
    ("sma_period", 5, "sma", "SMA_5"),
    ("ema_period", 7, "ema", "EMA_7"),
    ("rsi_period", 9, "rsi", "RSI_9"),
    ("bb_period", 10, "bollinger", "BB_Upper"),
    ("atr_period", 5, "atr", "ATR_5"),
    ("adx_period", 6, "adx", "ADX_6"),
    ("supertrend_period", 7, "supertrend", "Supertrend"),
    ("supertrend_multiplier", 2.5, "supertrend", "Supertrend"),
    ("hhv_period", 9, "hhv", "HHV_9"),
    ("llv_period", 8, "llv", "LLV_8"),
    ("mom_period", 4, "mom", "MOM_4"),
    ("roc_period", 6, "roc", "ROC_6"),
    ("wma_period", 12, "wma", "WMA_12"),
    ("dema_period", 11, "dema", "DEMA_11"),
    ("tema_period", 13, "tema", "TEMA_13"),
]


# ---------------------------------------------------------------------------
# REQ-001 — bilinmeyen parametre
# ---------------------------------------------------------------------------


def test_typo_param_suggests_correct_name():
    """'sma_perod' → ValueError + 'did you mean 'sma_period'?'."""
    df = _ohlcv()
    with pytest.raises(ValueError, match="unknown parameter 'sma_perod'"):
        add_indicators(df, ["sma"], sma_perod=5)
    with pytest.raises(ValueError, match=r"did you mean 'sma_period'\?"):
        add_indicators(df, ["sma"], sma_perod=5)


def test_completely_unknown_param_lists_supported():
    """'yokboyle' → ValueError; mesajda desteklenen parametre listesi."""
    df = _ohlcv()
    with pytest.raises(ValueError, match="unknown parameter 'yokboyle'"):
        add_indicators(df, ["sma"], yokboyle=1)
    # Liste örnekleri mesajda görünür
    with pytest.raises(ValueError, match="'sma_period'"):
        add_indicators(df, ["sma"], yokboyle=1)
    with pytest.raises(ValueError, match="'avwap_column'"):
        add_indicators(df, ["sma"], yokboyle=1)


def test_typo_with_no_close_match_has_no_hint():
    """Uzak ad → öneri yok, ama yine hata (sessiz yutma YOK)."""
    df = _ohlcv()
    with pytest.raises(ValueError, match="unknown parameter") as exc:
        add_indicators(df, ["sma"], zzzzzzzz=1)
    assert "did you mean" not in str(exc.value)


# ---------------------------------------------------------------------------
# REQ-002 — ilgisiz gösterge parametresi
# ---------------------------------------------------------------------------


def test_irrelevant_param_raises():
    """'rsi_period' + ['sma'] → ValueError (ilgisiz parametre)."""
    df = _ohlcv()
    with pytest.raises(ValueError, match="does not apply to indicator 'sma'"):
        add_indicators(df, ["sma"], rsi_period=999)
    with pytest.raises(ValueError, match="'rsi_period'"):
        add_indicators(df, ["sma"], rsi_period=999)


def test_relevant_param_is_accepted():
    """'rsi_period' + ['rsi'] → çalışır, RSI_999 üretilir."""
    df = _ohlcv()
    result = add_indicators(df, ["rsi"], rsi_period=999)
    assert "RSI_999" in result.columns


def test_param_applies_when_any_requested_indicator_matches():
    """['sma','rsi'] + rsi_period → geçerli (liste içinde rsi var)."""
    df = _ohlcv()
    result = add_indicators(df, ["sma", "rsi"], rsi_period=7, sma_period=5)
    assert "RSI_7" in result.columns
    assert "SMA_5" in result.columns


def test_avwap_params_apply_only_to_avwap():
    df = _ohlcv()
    with pytest.raises(ValueError, match="'avwap_anchor' does not apply"):
        add_indicators(df, ["sma"], avwap_anchor=0)
    result = add_indicators(df, ["avwap"], avwap_anchor=0, avwap_column="Open")
    assert "AVWAP" in result.columns


# ---------------------------------------------------------------------------
# REQ-003 — değer doğrulaması (tür/aralık)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, -100, True, False, "abc", 5.0, None])
def test_invalid_period_value_raises_early(bad):
    """*_period → int >= 1; bool/str/float/None reddedilir."""
    df = _ohlcv()
    with pytest.raises(ValueError, match="parameter 'sma_period'"):
        add_indicators(df, ["sma"], sma_period=bad)


def test_numpy_integer_period_accepted():
    """np.int64 kabul edilir (pandas dönemleri sık np.integer)."""
    df = _ohlcv()
    result = add_indicators(df, ["sma"], sma_period=np.int64(5))
    assert "SMA_5" in result.columns


@pytest.mark.parametrize("bad", [0, -2.5, float("nan"), float("inf"), -float("inf"), True, "x"])
def test_invalid_multiplier_value_raises(bad):
    """*_multiplier → sonlu float > 0."""
    df = _ohlcv()
    with pytest.raises(ValueError, match="parameter 'supertrend_multiplier'"):
        add_indicators(df, ["supertrend"], supertrend_multiplier=bad)


def test_validation_happens_before_computation():
    """Geçersiz değer hesaptan ÖNCE yakalanır (pandas'a inmeden)."""
    df = _ohlcv()
    # Eski davranışta buradan pandas 'window must be an integer' gelirdi.
    with pytest.raises(ValueError, match="must be an integer >= 1"):
        add_indicators(df, ["sma"], sma_period="abc")


# ---------------------------------------------------------------------------
# Atomiklik
# ---------------------------------------------------------------------------


def test_atomic_no_columns_when_kwarg_invalid():
    """['sma','rsi'] + geçersiz kwarg → ValueError, hiçbir sütun eklenmez."""
    df = _ohlcv()
    original = list(df.columns)

    with pytest.raises(ValueError):
        add_indicators(df, ["sma", "rsi"], sma_period=0)
    with pytest.raises(ValueError):
        add_indicators(df, ["sma", "rsi"], yokboyle=1)
    with pytest.raises(ValueError):
        add_indicators(df, ["sma", "rsi"], ema_period=5)

    assert list(df.columns) == original
    assert "SMA_20" not in df.columns
    assert "RSI_14" not in df.columns


# ---------------------------------------------------------------------------
# indicators=None — BYTE-IDENTICAL
# ---------------------------------------------------------------------------


def test_none_path_byte_identical_default_output():
    """None → varsayılan liste; çıktı elle kurulan beklentiyle birebir."""
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


def test_none_path_does_not_validate_kwargs():
    """None yolu: bilinmeyen/ilgisiz/geçersiz parametre HATA VERMEZ.

    "hepsi" yolu byte-identical kalır — W3f/W3h doğrulaması burada
    uygulanmaz (REQ-002 istisnası). Değerler yine de kwargs.get ile
    kullanılır (eski davranış).
    """
    df = _ohlcv()

    # Bilinmeyen anahtar sessizce yok sayılır (eski davranış).
    out = add_indicators(df, yokboyle=1)
    assert "SMA_20" in out.columns

    # İlgisiz parametre de hata vermez.
    out2 = add_indicators(df, rsi_period=999)
    assert "RSI_999" in out2.columns  # default listede rsi var → uygulanır
    assert "SMA_20" in out2.columns

    # Geçersiz değer: doğrulama yok → eski davranış (pandas hatası olabilir);
    # burada geçerli bir değerle yolun bozulmadığını pinliyoruz.
    out3 = add_indicators(df, sma_period=5)
    assert "SMA_5" in out3.columns


# ---------------------------------------------------------------------------
# Senkron: _SUPPORTED_KWARGS ↔ kwargs.get(...) (17 anahtar)
# ---------------------------------------------------------------------------


def test_supported_kwargs_sync_with_source():
    """Sabit, gövdedeki ``kwargs.get("...")`` çağrılarıyla BİREBİR senkron."""
    source = textwrap.dedent(inspect.getsource(technical.add_indicators))
    tree = ast.parse(source)

    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "kwargs"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            found.add(node.args[0].value)

    assert len(found) == 17, f"beklenen 17 kwargs.get; bulunan {len(found)}: {sorted(found)}"
    assert found == set(_SUPPORTED_KWARGS)


def test_supported_kwargs_owners_are_supported_indicators():
    """Her parametrenin sahibi _SUPPORTED_INDICATORS içinde olmalı."""
    for name, owners in _SUPPORTED_KWARGS.items():
        assert owners, f"{name}: sahip listesi boş"
        for owner in owners:
            assert owner in technical._SUPPORTED_INDICATORS, (
                f"{name}: bilinmeyen sahip gösterge {owner}"
            )


# ---------------------------------------------------------------------------
# Regresyon matrisi — mevcut tüm geçerli parametreler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("param", "value", "indicator", "expected_column"),
    _KWARG_MATRIX,
    ids=[item[0] for item in _KWARG_MATRIX],
)
def test_valid_param_regression_matrix(param, value, indicator, expected_column):
    """Her geçerli parametre ait olduğu göstergeyle çalışır; değer UYGULANIR."""
    df = _ohlcv()
    result = add_indicators(df, [indicator], **{param: value})
    assert expected_column in result.columns, f"{param} uygulanmadı"
    # Orijinal frame mutasyona uğramadı
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_avwap_param_matrix():
    df = _ohlcv()
    result = add_indicators(df, ["avwap"], avwap_anchor=0, avwap_column="Open")
    assert "AVWAP" in result.columns
    assert not result["AVWAP"].isna().all()


def test_period_value_actually_honored_not_default():
    """sma_period=5 → SMA_5 (SMA_20 DEĞİL) — sessiz varsayılan yok."""
    df = _ohlcv()
    result = add_indicators(df, ["sma"], sma_period=5)
    assert "SMA_5" in result.columns
    assert "SMA_20" not in result.columns
