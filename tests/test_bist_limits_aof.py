"""W3b testleri: AOF + takvim → compute_price_limits bağlantısı.

Testler **çıktı düzeyindedir**: dönen `PriceLimits`/DataFrame doğrulanır,
yardımcı fonksiyon dönüşü tek başına yeterli sayılmaz (075l dersi 2).

AOF değeri testlerde **kayıtlı kanıttan** gelir (İş Yatırım
``HGDG_AOF``): ``tests/fixtures/isyatirim/aof_thyao_2026_11.json``
sha256 ``cfafa642bff99138…``. THYAO 02-11-2026: AOF 252,992 / kapanış 261,00.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from borsapy._providers import bist_calendar
from borsapy._providers.bist_limits import (
    AOFUnavailableError,
    PriceLimits,
    compute_price_limits,
    compute_price_limits_from_aof,
    flag_limit_sessions,
)
from borsapy._providers.vbts_guard import BandSourceUnverifiedError

FIXTURE = Path(__file__).parent / "fixtures" / "isyatirim" / "aof_thyao_2026_11.json"


def _aof_frame(day: date, aof: str = "252.992", close: str = "261.0") -> pd.DataFrame:
    """Build a ``get_daily_aof``-shaped frame for one day."""
    frame = pd.DataFrame(
        {
            "aof": [Decimal(aof)],
            "close": [Decimal(close)],
            "volume": [Decimal("12373925452.0")],
        },
        index=[day],
    )
    return frame


def _fetcher(frame: pd.DataFrame):
    """Return a fetcher stub matching `get_daily_aof`'s signature."""

    def _fetch(symbol, start=None, end=None):  # noqa: ARG001
        return frame, {"rows_returned": len(frame)}

    return _fetch


# ---------------------------------------------------------------------------
# Fixture / pin integrity
# ---------------------------------------------------------------------------


def test_aof_fixture_pin_matches_source():
    """Kanıt fixture'ı kaynağın birebir kopyası (sha256 pini)."""
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert digest.startswith("cfafa642bff99138")
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    first = payload["value"][0]
    assert first["HGDG_TARIH"] == "02-11-2026"
    assert Decimal(str(first["HGDG_AOF"])) == Decimal("252.992")
    assert Decimal(str(first["HGDG_KAPANIS"])) == Decimal("261.0")


# ---------------------------------------------------------------------------
# REQ-001 — compute_price_limits_from_aof
# ---------------------------------------------------------------------------


def test_limits_from_aof_output_matches_hand_calculation():
    """Çıktı düzeyi: AOF 252,992 → tavan 278,25 / taban 227,70 (el hesabı)."""
    limits = compute_price_limits_from_aof(
        "THYAO",
        as_of=date(2026, 11, 3),
        aof_fetcher=_fetcher(_aof_frame(date(2026, 11, 2))),
    )

    assert isinstance(limits, PriceLimits)
    assert limits.reference == Decimal("252.992")
    assert limits.tavan == Decimal("278.25")  # 278,2912 -> 0,25 adım
    assert limits.taban == Decimal("227.70")  # 227,6928 -> 0,10 adım (bant değişimi)
    assert isinstance(limits.tavan, Decimal) and isinstance(limits.taban, Decimal)
    assert limits.band_pct == Decimal("0.10")
    assert limits.band_source == "default"


def test_limits_from_aof_stamps_reference_source_and_date():
    limits = compute_price_limits_from_aof(
        "THYAO",
        as_of=date(2026, 11, 3),
        aof_fetcher=_fetcher(_aof_frame(date(2026, 11, 2))),
    )
    assert limits.reference_source == "aof"
    assert limits.reference_date == date(2026, 11, 2)
    assert limits.as_of_normalized_from is None
    assert any("AOF" in note for note in limits.notes)
    view = limits.as_dict()
    assert view["reference_source"] == "aof"
    assert view["reference_date"] == date(2026, 11, 2)


def test_reference_is_previous_trading_day_after_holiday():
    """29 Ekim tatil; 30 Ekim Cuma'nın referansı 28 Ekim (yarım gün) olmalı."""
    captured: dict = {}

    def _fetch(symbol, start=None, end=None):
        captured["start"] = start
        captured["end"] = end
        return _aof_frame(start), {}

    compute_price_limits_from_aof(
        "THYAO", as_of=date(2026, 10, 30), aof_fetcher=_fetch
    )
    assert captured["start"] == date(2026, 10, 28)
    assert captured["end"] == date(2026, 10, 28)


def test_limits_from_aof_delegates_to_single_formula():
    """Tek hesap yolu: aynı AOF ile `compute_price_limits` birebir aynı sonucu verir."""
    aof = Decimal("252.992")
    direct = compute_price_limits(aof)
    via_aof = compute_price_limits_from_aof(
        "THYAO",
        as_of=date(2026, 11, 3),
        aof_fetcher=_fetcher(_aof_frame(date(2026, 11, 2))),
    )
    assert (direct.tavan, direct.taban, direct.reference) == (
        via_aof.tavan,
        via_aof.taban,
        via_aof.reference,
    )


def test_missing_aof_raises_and_never_falls_back_to_close():
    """AOF satırı yok → AOFUnavailableError; kapanışa düşmez."""
    empty = pd.DataFrame(columns=["aof", "close", "volume"])
    with pytest.raises(AOFUnavailableError) as excinfo:
        compute_price_limits_from_aof(
            "THYAO", as_of=date(2026, 11, 3), aof_fetcher=_fetcher(empty)
        )
    assert excinfo.value.symbol == "THYAO"
    assert excinfo.value.reference_date == date(2026, 11, 2)


def test_null_aof_value_raises():
    """Satır var ama AOF None → yine hata (uydurma yok)."""
    frame = pd.DataFrame(
        {"aof": [None], "close": [Decimal("261.0")]}, index=[date(2026, 11, 2)]
    )
    with pytest.raises(AOFUnavailableError):
        compute_price_limits_from_aof(
            "THYAO", as_of=date(2026, 11, 3), aof_fetcher=_fetcher(frame)
        )


def test_weekend_as_of_is_normalized_and_stamped():
    """7 Kasım Cumartesi → 6 Kasım Cuma; referans 5 Kasım; damga set."""
    limits = compute_price_limits_from_aof(
        "THYAO",
        as_of=date(2026, 11, 7),
        aof_fetcher=_fetcher(_aof_frame(date(2026, 11, 5))),
    )
    assert limits.as_of_normalized_from == date(2026, 11, 7)
    assert limits.reference_date == date(2026, 11, 5)
    assert any("islem gunu degil" in note for note in limits.notes)


def test_holiday_as_of_is_normalized():
    """29 Ekim tatil → 28 Ekim (yarım gün) çapası; referans 27 Ekim."""
    limits = compute_price_limits_from_aof(
        "THYAO",
        as_of=date(2026, 10, 29),
        aof_fetcher=_fetcher(_aof_frame(date(2026, 10, 27))),
    )
    assert limits.as_of_normalized_from == date(2026, 10, 29)
    assert limits.reference_date == date(2026, 10, 27)


def test_calendar_coverage_error_propagates():
    """Takvim kapsamı dışı yıl → CalendarCoverageError yükselir (tahmin yok)."""
    with pytest.raises(bist_calendar.CalendarCoverageError):
        compute_price_limits_from_aof(
            "THYAO", as_of=date(2027, 1, 5), aof_fetcher=_fetcher(_aof_frame(date(2027, 1, 4)))
        )


def test_empty_symbol_rejected():
    with pytest.raises(ValueError):
        compute_price_limits_from_aof("", as_of=date(2026, 11, 3))


def test_custom_band_is_stamped_caller_supplied():
    limits = compute_price_limits_from_aof(
        "THYAO",
        as_of=date(2026, 11, 3),
        band_pct=Decimal("0.05"),
        aof_fetcher=_fetcher(_aof_frame(date(2026, 11, 2))),
    )
    assert limits.band_source == "caller_supplied"
    assert limits.band_pct == Decimal("0.05")


# ---------------------------------------------------------------------------
# REQ-003 — VBTS guard integration (opt-in)
# ---------------------------------------------------------------------------


def test_vbts_guard_blocks_default_band_in_call_path():
    """VBTS'li sembol + varsayılan bant → çağrı yolunda hata."""
    with pytest.raises(BandSourceUnverifiedError):
        compute_price_limits_from_aof(
            "GARAN",
            as_of=date(2026, 11, 3),
            vbts_symbols={"symbols": {"GARAN": {"vbts_var": True, "within_window": True}}},
            aof_fetcher=_fetcher(_aof_frame(date(2026, 11, 2))),
        )


def test_vbts_guard_allows_explicit_band():
    limits = compute_price_limits_from_aof(
        "GARAN",
        as_of=date(2026, 11, 3),
        band_pct=Decimal("0.05"),
        vbts_symbols={"symbols": {"GARAN": {"vbts_var": True, "within_window": True}}},
        aof_fetcher=_fetcher(_aof_frame(date(2026, 11, 2))),
    )
    assert limits.band_source == "caller_supplied"


def test_vbts_guard_ignored_when_not_requested():
    """Liste verilmezse davranış değişmez (opt-in)."""
    limits = compute_price_limits_from_aof(
        "GARAN",
        as_of=date(2026, 11, 3),
        aof_fetcher=_fetcher(_aof_frame(date(2026, 11, 2))),
    )
    assert limits.band_source == "default"


# ---------------------------------------------------------------------------
# REQ-002 — flag_limit_sessions(calendar=...) : GAP denetimi
# ---------------------------------------------------------------------------


def _frame(days, rows):
    return pd.DataFrame(
        rows,
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=pd.to_datetime(days),
    )


def test_flag_limit_sessions_without_calendar_is_unchanged():
    """Regresyon: calendar=None → mevcut davranış birebir."""
    df = _frame(
        ["2026-11-02", "2026-11-03"],
        [(100, 100, 100, 100, 1_000), (110, 110, 110, 110, 0)],
    )
    baseline = flag_limit_sessions(df)
    with_none = flag_limit_sessions(df, calendar=None)

    pd.testing.assert_frame_equal(baseline, with_none)
    assert list(baseline.columns) == list(with_none.columns)
    assert bool(baseline.iloc[1]["limitli"]) is True
    assert baseline.iloc[1]["guven"] == "KESIN"


def test_flag_limit_sessions_detects_calendar_gap_and_downgrades():
    """Seride tatil atlanmış (GAP) → SUPHELI + gerekçe damgası (KESIN'e yükselmez)."""
    # 28 Ekim -> 30 Ekim: arada 29 Ekim tatil, ama takvime göre 30 Ekim'in
    # önceki işlem günü 28 Ekim'dir; seride 28 ve 30 var -> GAP YOK.
    # GAP için: 27 Ekim -> 30 Ekim (28 Ekim atlanmış).
    df = _frame(
        ["2026-10-27", "2026-10-30"],
        [(100, 100, 100, 100, 1_000), (110, 110, 110, 110, 0)],
    )
    out = flag_limit_sessions(df, calendar=bist_calendar)

    row = out.iloc[1]
    assert bool(row["limitli"]) is True
    assert "onceki_seans_takvim_dogrulanmadi" in row["gerekce"]
    assert row["guven"] == "SUPHELI"


def test_flag_limit_sessions_no_gap_keeps_kesin():
    """Komşu seans gerçekten takvimdeki önceki günse KESIN korunur."""
    df = _frame(
        ["2026-10-28", "2026-10-30"],
        [(100, 100, 100, 100, 1_000), (110, 110, 110, 110, 0)],
    )
    out = flag_limit_sessions(df, calendar=bist_calendar)
    row = out.iloc[1]
    assert row["guven"] == "KESIN"
    assert "takvim_dogrulanmadi" not in row["gerekce"]


def test_flag_limit_sessions_calendar_without_api_raises():
    with pytest.raises(ValueError):
        flag_limit_sessions(
            _frame(["2026-11-02"], [(100, 100, 100, 100, 1)]), calendar=object()
        )


# ---------------------------------------------------------------------------
# Regresyon — compute_price_limits çıktısı değişmedi
# ---------------------------------------------------------------------------


def test_compute_price_limits_output_unchanged():
    """Mevcut yol byte-identical: W3b alanları yalnız set edildiğinde görünür."""
    limits = compute_price_limits("100")
    view = limits.as_dict()
    assert "reference_date" not in view
    assert "as_of_normalized_from" not in view
    assert limits.reference_date is None
    assert limits.reference_source == "prev_close (AOF yok: VERI YOK)"
    assert (limits.tavan, limits.taban) == (Decimal("110.00"), Decimal("90.00"))
