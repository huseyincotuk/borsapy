"""W3a-veri: BIST fiyat adımları, tavan/taban ve limitli seans bayrağı testleri.

Altın pinler **iki bağımsız kaynaktan** gelir, bu modülün çıktısından değil:

* Fiyat adımı tablosu — BIST duyuru **14571 (28.08.2023)**, Tablo 1.
  Ek dosya ``https://borsaistanbul.com/files/duyuru-19412.pdf``
  (sha256 ``cb0a1e0091d79918…``). Kanıt künyesi:
  ``gecmis baglam/w3a_veri_launch_2026_11_12/KAYNAK_KANITI.md``.
* Limit seans senaryoları — **sentetik** (kurgusal), çünkü gerçek bir sembolün
  doğrulanmış tavan günü launch öncesi **sağlanamadı (VERİ YOK)**. Sentetik
  satırlar gerçek sembol iddiası taşımaz.

Testler **çıktı düzeyindedir** (075l dersi 2): her pin dönen nesnenin/çerçevenin
gözlenebilir çıktısını doğrular, yardımcı fonksiyon dönüşünü değil.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from borsapy._providers.bist_limits import (
    LIMIT_BAND_PCT_DEFAULT,
    LIMIT_EQUALITY_TOLERANCE_PCT,
    PRICE_TICK_TABLE,
    PRICE_TICK_TABLE_SOURCE,
    adtv,
    adtv_diagnostics,
    compute_price_limits,
    flag_limit_sessions,
    round_to_tick,
    tick_for_price,
)

#: Tablo 1 — (fiyat, beklenen adım). Kaynak: duyuru 14571 (28.08.2023).
TICK_PINS = (
    ("0.01", "0.01"),
    ("19.99", "0.01"),
    ("20", "0.02"),
    ("49.99", "0.02"),
    ("50", "0.05"),
    ("99.99", "0.05"),
    ("100", "0.10"),
    ("249.99", "0.10"),
    ("250", "0.25"),
    ("499.99", "0.25"),
    ("500", "0.50"),
    ("999.99", "0.50"),
    ("1000", "1.00"),
    ("2499.99", "1.00"),
    ("2500", "2.50"),
    ("10000", "2.50"),
)


# ---------------------------------------------------------------------------
# REQ-001 — fiyat adımı tablosu
# ---------------------------------------------------------------------------


def test_source_metadata_is_stamped():
    """The table must never travel without its provenance."""
    assert PRICE_TICK_TABLE_SOURCE["url"] == (
        "https://borsaistanbul.com/files/duyuru-19412.pdf"
    )
    assert PRICE_TICK_TABLE_SOURCE["published"] == "2023-08-28"
    assert PRICE_TICK_TABLE_SOURCE["sha256_prefix"] == "cb0a1e0091d79918"
    assert "Tablo 1" in PRICE_TICK_TABLE_SOURCE["table"]
    assert len(PRICE_TICK_TABLE) == 8


@pytest.mark.parametrize(("price", "expected"), TICK_PINS)
def test_tick_band_pins(price, expected):
    assert tick_for_price(price) == Decimal(expected)


def test_tick_for_price_rejects_below_first_band():
    with pytest.raises(ValueError):
        tick_for_price("0.001")


def test_tick_for_price_rejects_non_numeric():
    with pytest.raises(ValueError):
        tick_for_price("abc")


@pytest.mark.parametrize(
    ("price", "expected"),
    (
        ("10.004", "10.00"),
        ("10.006", "10.01"),
        ("20.01", "20.02"),
        ("20.03", "20.04"),
        ("100.06", "100.10"),
        ("250.10", "250.00"),
        ("250.13", "250.25"),
        ("1000.40", "1000.00"),
        ("2500.00", "2500.00"),
        ("3500.00", "3500.00"),
    ),
)
def test_round_to_tick_outputs(price, expected):
    """Output-level: the rounded value itself, not an intermediate helper."""
    result = round_to_tick(price)
    assert isinstance(result, Decimal)
    assert result == Decimal(expected)


def test_round_to_tick_is_decimal_not_float():
    """Decimal pin: float artefakti görünmez, iki hane korunur."""
    result = round_to_tick("10.1")
    assert type(result) is Decimal
    assert str(result) == "10.10"
    # A float round-trip would surface 10.0999999999999996447...
    assert Decimal(str(float(result))) == Decimal("10.1")


def test_round_to_tick_rejects_non_positive():
    with pytest.raises(ValueError):
        round_to_tick("0")
    with pytest.raises(ValueError):
        round_to_tick("-5")


# ---------------------------------------------------------------------------
# REQ-002 — tavan/taban
# ---------------------------------------------------------------------------


def test_compute_price_limits_default_band_output():
    limits = compute_price_limits("100")
    assert limits.tavan == Decimal("110.00")
    assert limits.taban == Decimal("90.00")
    assert isinstance(limits.tavan, Decimal) and isinstance(limits.taban, Decimal)
    assert limits.band_pct == LIMIT_BAND_PCT_DEFAULT
    assert limits.band_source == "default"
    assert "AOF" in limits.reference_source  # documented approximation stamp
    assert limits.equality_tolerance_pct == LIMIT_EQUALITY_TOLERANCE_PCT


def test_compute_price_limits_tick_aligned_on_odd_reference():
    """Reference that is not itself tick-aligned still yields aligned limits."""
    limits = compute_price_limits("33.33")
    assert limits.tavan == Decimal("36.66")  # 36.663 -> nearest 0.02 step
    assert limits.taban == Decimal("30.00")  # 29.997 -> nearest 0.02 step
    assert limits.tavan == round_to_tick(limits.tavan)


def test_compute_price_limits_custom_band_is_stamped():
    limits = compute_price_limits("100", band_pct=Decimal("0.05"))
    assert limits.tavan == Decimal("105.00")
    assert limits.taban == Decimal("95.00")
    assert limits.band_source == "caller_supplied"
    # VBTS narrowed bands have no source in this module -> honest VERİ YOK note.
    assert any("VERI YOK" in note for note in limits.notes)


def test_compute_price_limits_rejects_bad_inputs():
    with pytest.raises(ValueError):
        compute_price_limits("0")
    with pytest.raises(ValueError):
        compute_price_limits("100", band_pct="1.5")


# ---------------------------------------------------------------------------
# REQ-003 — limitli seans bayrağı
# ---------------------------------------------------------------------------


def _frame(rows, start="2026-01-05"):
    idx = pd.date_range(start, periods=len(rows), freq="D")
    return pd.DataFrame(
        rows,
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=idx,
    )


def test_flag_limit_sessions_kesin_tavan_output():
    """Single-priced session at the ceiling -> limitli, yon=tavan, guven=KESIN."""
    df = _frame(
        [
            (100, 100, 100, 100, 1_000),  # normal gün, referans üretir
            (110, 110, 110, 110, 5_000),  # 100*1.10 = 110 -> tek fiyat tavan
        ]
    )
    out = flag_limit_sessions(df)

    for column in ("referans", "tavan", "taban", "limitli", "yon", "gerekce",
                   "guven", "hacimsiz"):
        assert column in out.columns

    second = out.iloc[1]
    assert bool(second["limitli"]) is True
    assert second["yon"] == "tavan"
    assert second["guven"] == "KESIN"
    assert second["tavan"] == Decimal("110.00")
    assert "tek fiyat" in second["gerekce"]

    first = out.iloc[0]
    assert bool(first["limitli"]) is False
    assert first["guven"] is None or pd.isna(first["guven"])
    assert "VERİ YOK" in first["gerekce"]


def test_flag_limit_sessions_kesin_taban_output():
    df = _frame(
        [
            (100, 100, 100, 100, 1_000),
            (90, 90, 90, 90, 2_000),  # 100*0.90 = 90 -> tek fiyat taban
        ]
    )
    out = flag_limit_sessions(df)
    second = out.iloc[1]
    assert bool(second["limitli"]) is True
    assert second["yon"] == "taban"
    assert second["guven"] == "KESIN"


def test_flag_limit_sessions_supheli_when_not_single_priced():
    """Touched and closed at the ceiling but traded through the range."""
    df = _frame(
        [
            (100, 100, 100, 100, 1_000),
            (105, 110, 104, 110, 50_000),  # high=close=tavan, low<tavan
        ]
    )
    out = flag_limit_sessions(df)
    second = out.iloc[1]
    assert bool(second["limitli"]) is True
    assert second["yon"] == "tavan"
    # SUPHELI must never be promoted to KESIN (ERRATUM v2 D2).
    assert second["guven"] == "SUPHELI"


def test_flag_limit_sessions_normal_day_is_not_flagged():
    df = _frame(
        [
            (100, 100, 100, 100, 1_000),
            (101, 105, 99, 103, 40_000),
        ]
    )
    out = flag_limit_sessions(df)
    second = out.iloc[1]
    assert bool(second["limitli"]) is False
    assert second["gerekce"] == "limit temasi yok"


def test_flag_limit_sessions_zero_volume_marked_separately():
    df = _frame(
        [
            (100, 100, 100, 100, 0),      # hacimsiz
            (110, 110, 110, 110, 0),      # tek fiyat tavan + hacimsiz
        ]
    )
    out = flag_limit_sessions(df)
    assert bool(out.iloc[0]["hacimsiz"]) is True
    assert bool(out.iloc[1]["hacimsiz"]) is True
    assert bool(out.iloc[1]["limitli"]) is True
    assert out.iloc[1]["guven"] == "KESIN"
    assert "hacimsiz" in out.iloc[1]["gerekce"]


def test_flag_limit_sessions_nan_volume_is_hacimsiz():
    df = _frame([(100, 100, 100, 100, 1_000), (101, 102, 100, 101, None)])
    out = flag_limit_sessions(df)
    assert bool(out.iloc[1]["hacimsiz"]) is True


def test_flag_limit_sessions_empty_frame_does_not_crash():
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    out = flag_limit_sessions(empty)
    assert out.empty
    assert "limitli" in out.columns


def test_flag_limit_sessions_single_row_has_no_reference():
    df = _frame([(100, 100, 100, 100, 1_000)])
    out = flag_limit_sessions(df)
    assert len(out) == 1
    assert bool(out.iloc[0]["limitli"]) is False
    assert "VERİ YOK" in out.iloc[0]["gerekce"]


def test_flag_limit_sessions_requires_ohlc_columns():
    df = pd.DataFrame({"Close": [1.0, 2.0], "Volume": [10, 20]})
    with pytest.raises(ValueError):
        flag_limit_sessions(df)


def test_flag_limit_sessions_preserves_input_untouched():
    df = _frame([(100, 100, 100, 100, 1_000), (110, 110, 110, 110, 0)])
    before = df.copy()
    flag_limit_sessions(df)
    pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------------------
# REQ-004 — ADTV
# ---------------------------------------------------------------------------


def test_adtv_output_values_and_min_periods():
    df = _frame(
        [
            (100, 100, 100, 100, 1_000),
            (100, 100, 100, 100, 1_000),
            (100, 100, 100, 100, 2_000),
        ]
    )
    series = adtv(df, window=2)

    assert len(series) == 3
    assert pd.isna(series.iloc[0])  # min_periods=window -> partial window NaN
    assert series.iloc[1] == pytest.approx(100_000.0)
    assert series.iloc[2] == pytest.approx(150_000.0)


def test_adtv_typical_price_uses_hlc_average():
    df = _frame([(100, 120, 90, 90, 100)])  # typical = (120+90+90)/3 = 100
    series = adtv(df, window=1, min_periods=1)
    assert series.iloc[0] == pytest.approx(10_000.0)


def test_adtv_diagnostics_counts():
    df = _frame(
        [
            (100, 100, 100, 100, 0),
            (100, 100, 100, 100, None),
            (100, 100, 100, 100, 500),
        ]
    )
    diag = adtv_diagnostics(df)
    assert diag["rows"] == 3
    assert diag["volume_nan"] == 1
    assert diag["volume_zero"] == 2  # explicit zero + NaN-as-zero bucket
    assert diag["price_nan"] == 0


def test_adtv_rejects_bad_window():
    df = _frame([(100, 100, 100, 100, 1_000)])
    with pytest.raises(ValueError):
        adtv(df, window=0)
