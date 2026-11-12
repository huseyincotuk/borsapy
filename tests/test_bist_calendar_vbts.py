"""W3a-ek testleri: AOF kaynağı, BIST işlem takvimi, VBTS bant guard.

Altın pinler **kaynaklardan** gelir, modülün çıktısından değil:

* Tatil tablosu — BIST `pay-piyasasi-2026-yili-tatil-tablosu.pdf`
  (sha256 ``e45fa97e2f67d85b…``), duyuru 15282. Kanıt:
  ``gecmis baglam/w3a_ek_launch_2026_11_12/kanit/``.
* AOF alan eşlemesi — İş Yatırım ``Data.aspx/HisseTekil`` yanıtı
  (``HGDG_AOF``); ölçülen kanıt dosyaları aynı klasörde
  (THYAO ``cfafa642bff99138…``, GARAN ``3ca8969f9aa301da…``).

Testler **çıktı düzeyindedir** (075l dersi 2): dönen nesne/çerçeve doğrulanır,
yardımcı fonksiyon dönüşü yeterli sayılmaz.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from borsapy._providers.bist_calendar import (
    BIST_HOLIDAY_SOURCE,
    CalendarCoverageError,
    available_years,
    holiday_rows,
    is_half_day,
    is_trading_day,
    previous_trading_day,
    trading_days,
)
from borsapy._providers.vbts_guard import (
    BandSourceUnverifiedError,
    assert_band_source_safe,
    vbts_symbols_from_status,
)

# ---------------------------------------------------------------------------
# REQ-002 — BIST işlem takvimi
# ---------------------------------------------------------------------------


def test_calendar_source_is_stamped():
    assert BIST_HOLIDAY_SOURCE["pdf"].endswith("pay-piyasasi-2026-yili-tatil-tablosu.pdf")
    assert BIST_HOLIDAY_SOURCE["pdf_sha256_prefix"] == "e45fa97e2f67d85b"
    assert BIST_HOLIDAY_SOURCE["coverage_years"] == (2026,)
    assert "15282" in BIST_HOLIDAY_SOURCE["announcement"]


def test_coverage_years_only_sourced_year():
    assert available_years() == (2026,)


@pytest.mark.parametrize(
    "day",
    (
        date(2026, 1, 1),   # Yılbaşı
        date(2026, 3, 20),  # Ramazan Bayramı
        date(2026, 4, 23),  # Ulusal Egemenlik
        date(2026, 5, 1),   # Emek ve Dayanışma
        date(2026, 5, 19),  # Gençlik ve Spor
        date(2026, 5, 27),  # Kurban Bayramı
        date(2026, 7, 15),  # Demokrasi ve Millî Birlik
        date(2026, 10, 29),  # Cumhuriyet Bayramı
    ),
)
def test_full_holidays_are_not_trading_days(day):
    """Output-level pin: the official table's closed days are NOT trading days."""
    assert is_trading_day(day) is False


@pytest.mark.parametrize(
    "day",
    (
        date(2026, 1, 2),    # Yılbaşından sonraki Cuma
        date(2026, 3, 23),   # Ramazan sonrası Pazartesi
        date(2026, 10, 30),  # Cumhuriyet Bayramı sonrası Cuma
        date(2026, 11, 2),   # Pazartesi
    ),
)
def test_regular_weekdays_are_trading_days(day):
    assert is_trading_day(day) is True


def test_weekends_are_not_trading_days():
    assert is_trading_day(date(2026, 11, 7)) is False  # Cumartesi
    assert is_trading_day(date(2026, 11, 8)) is False  # Pazar


@pytest.mark.parametrize(
    "day",
    (date(2026, 3, 19), date(2026, 5, 26), date(2026, 10, 28)),
)
def test_half_days_are_trading_but_flagged(day):
    """Arife günleri işlem günüdür (seans var) ama yarım gündür."""
    assert is_trading_day(day) is True
    assert is_half_day(day) is True


def test_non_half_day_is_not_flagged():
    assert is_half_day(date(2026, 11, 2)) is False
    assert is_half_day(date(2026, 10, 29)) is False


def test_previous_trading_day_skips_holiday_block():
    """29 Ekim 2026 (Perşembe, tatil) -> önceki işlem günü 28 Ekim (yarım gün)."""
    assert previous_trading_day(date(2026, 10, 29)) == date(2026, 10, 28)
    # 2 Kasım Pazartesi -> 30 Ekim Cuma
    assert previous_trading_day(date(2026, 11, 2)) == date(2026, 10, 30)


def test_previous_trading_day_include_self():
    assert previous_trading_day(date(2026, 11, 3), include_self=True) == date(2026, 11, 3)
    assert previous_trading_day(date(2026, 11, 3)) == date(2026, 11, 2)


def test_trading_days_range_output():
    """29 Ekim (tatil) + 30 Ekim (Cuma) + 31/1 (hafta sonu) + 2 Kasım (Pzt)."""
    days = trading_days(date(2026, 10, 29), date(2026, 11, 2))
    assert days == [date(2026, 10, 30), date(2026, 11, 2)]


def test_trading_days_rejects_reversed_range():
    with pytest.raises(ValueError):
        trading_days(date(2026, 11, 5), date(2026, 11, 1))


@pytest.mark.parametrize(
    "func",
    (is_trading_day, is_half_day, previous_trading_day),
)
def test_out_of_coverage_year_raises(func):
    """2027 kaynakta YOK -> tahmin yok, hata (VERİ YOK)."""
    with pytest.raises(CalendarCoverageError):
        func(date(2027, 1, 4))


def test_trading_days_out_of_coverage_raises():
    with pytest.raises(CalendarCoverageError):
        trading_days(date(2026, 12, 30), date(2027, 1, 5))


def test_holiday_rows_counts_match_source():
    """Çıktı düzeyi: 14 kapalı + 3 yarım gün = 17 satır."""
    rows = holiday_rows(2026)
    assert len(rows) == 17
    kinds = [row.kind for row in rows]
    assert kinds.count("closed") == 14
    assert kinds.count("half_day") == 3
    assert [row.day for row in rows if row.kind == "half_day"] == [
        date(2026, 3, 19),
        date(2026, 5, 26),
        date(2026, 10, 28),
    ]


def test_holiday_rows_out_of_coverage_raises():
    with pytest.raises(CalendarCoverageError):
        holiday_rows(2030)


# ---------------------------------------------------------------------------
# REQ-003 — VBTS bant guard
# ---------------------------------------------------------------------------

_VBTS_STATUS = {
    "symbols": {
        "GARAN": {"vbts_var": True, "within_window": True},
        "TERA": {"vbts_var": True, "within_window": False},
    }
}


def test_vbts_symbols_from_status_honours_window():
    symbols = vbts_symbols_from_status(_VBTS_STATUS)
    assert symbols == frozenset({"GARAN"})  # TERA pencere dışı


def test_vbts_guard_rejects_default_band_for_vbts_symbol():
    """Senaryo 1: VBTS'li sembol + default bant -> hata."""
    with pytest.raises(BandSourceUnverifiedError) as excinfo:
        assert_band_source_safe("GARAN", "default", _VBTS_STATUS)
    assert excinfo.value.symbol == "GARAN"
    assert excinfo.value.band_source == "default"


def test_vbts_guard_allows_caller_supplied_band():
    """Senaryo 2: VBTS'li + caller_supplied -> geçer."""
    assert assert_band_source_safe("GARAN", "caller_supplied", _VBTS_STATUS) is None
    assert assert_band_source_safe("GARAN", "vbts_explicit", _VBTS_STATUS) is None


def test_vbts_guard_rejects_empty_list():
    """Senaryo 3: liste boş -> hata (sessiz 'VBTS yok' varsayımı yasak)."""
    with pytest.raises(BandSourceUnverifiedError):
        assert_band_source_safe("THYAO", "default", [])


def test_vbts_guard_rejects_missing_list():
    with pytest.raises(BandSourceUnverifiedError):
        assert_band_source_safe("THYAO", "default", None)


def test_vbts_guard_allows_non_vbts_symbol():
    """Senaryo 4: VBTS'siz sembol -> geçer (liste dolu)."""
    assert assert_band_source_safe("THYAO", "default", _VBTS_STATUS) is None


def test_vbts_guard_accepts_plain_iterable():
    """Düz yineleyici de kabul edilir (KAP çıktısı zorunlu değil)."""
    with pytest.raises(BandSourceUnverifiedError):
        assert_band_source_safe("GARAN", "default", ["GARAN"])
    assert assert_band_source_safe("GARAN", "caller_supplied", ["GARAN"]) is None


def test_vbts_guard_accepts_lowercase_symbol():
    with pytest.raises(BandSourceUnverifiedError):
        assert_band_source_safe("garan", "default", _VBTS_STATUS)


def test_vbts_guard_rejects_empty_symbol():
    with pytest.raises(ValueError):
        assert_band_source_safe("", "default", _VBTS_STATUS)


def test_vbts_guard_integrates_with_price_limits():
    """Çıktı düzeyi: compute_price_limits'in band_source'u doğrudan kullanılır."""
    from borsapy._providers.bist_limits import compute_price_limits

    limits = compute_price_limits("100")
    assert limits.band_source == "default"
    # Varsayılan bant VBTS'li sembolde guard'a takılır.
    with pytest.raises(BandSourceUnverifiedError):
        assert_band_source_safe("GARAN", limits.band_source, _VBTS_STATUS)

    custom = compute_price_limits("100", band_pct=Decimal("0.05"))
    assert custom.band_source == "caller_supplied"
    assert assert_band_source_safe("GARAN", custom.band_source, _VBTS_STATUS) is None
