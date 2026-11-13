"""W3c testleri: `flag_limit_sessions(aof_provider=...)` — seri-içi AOF referansı.

Testler **çıktı düzeyindedir**: dönen DataFrame doğrulanır.

AOF pinleri kayıtlı kanıttan gelir (İş Yatırım ``HGDG_AOF``,
``tests/fixtures/isyatirim/aof_thyao_2026_11.json`` sha ``cfafa642bff99138…``).
"""

from __future__ import annotations

import hashlib
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from borsapy._providers import bist_calendar
from borsapy._providers.bist_limits import (
    AOFSeriesProvider,
    flag_limit_sessions,
)

FIXTURE = Path(__file__).parent / "fixtures" / "isyatirim" / "aof_thyao_2026_11.json"


def _frame(days, rows):
    """OHLCV frame with a DatetimeIndex (daily sessions)."""
    return pd.DataFrame(
        rows,
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=pd.to_datetime(days),
    )


class _StubProvider:
    """Reference-AOF provider: ``aof_for(day)`` = previous session's AOF."""

    def __init__(self, by_day: dict[date, Decimal | None]):
        self.by_day = by_day
        self.calls: list[date] = []

    def aof_for(self, day: date) -> Decimal | None:
        from borsapy._providers.bist_calendar import previous_trading_day

        self.calls.append(day)
        return self.by_day.get(previous_trading_day(day))


# ---------------------------------------------------------------------------
# REQ-001.1 / contract: default path is byte-identical
# ---------------------------------------------------------------------------


def test_default_path_is_byte_identical_and_has_no_new_column():
    """`aof_provider=None` → eski çıktı birebir; `referans_kaynagi` EKLENMEZ."""
    df = _frame(
        ["2026-11-02", "2026-11-03", "2026-11-04"],
        [
            (100, 100, 100, 100, 1_000),
            (110, 110, 110, 110, 5_000),
            (121, 121, 121, 121, 300),
        ],
    )
    baseline = flag_limit_sessions(df)
    explicit_none = flag_limit_sessions(df, aof_provider=None)

    pd.testing.assert_frame_equal(baseline, explicit_none)
    assert "referans_kaynagi" not in baseline.columns
    # Historical behaviour: reference = previous close.
    assert baseline.iloc[1]["referans"] == Decimal("100")
    assert baseline.iloc[1]["guven"] == "KESIN"


# ---------------------------------------------------------------------------
# REQ-001.2/5 — AOF reference drives tavan/taban
# ---------------------------------------------------------------------------


def test_aof_reference_changes_limits_and_stamps_source():
    """AOF 100 → tavan 110 (kapanış referansı 100 ile aynı gün) ama sütun farklı."""
    df = _frame(
        ["2026-11-02", "2026-11-03"],
        [(100, 100, 100, 100, 1_000), (110, 110, 110, 110, 0)],
    )
    provider = _StubProvider({date(2026, 11, 2): Decimal("100")})
    out = flag_limit_sessions(df, aof_provider=provider)

    assert "referans_kaynagi" in out.columns
    row = out.iloc[1]
    assert row["referans"] == Decimal("100")
    assert isinstance(row["referans"], Decimal)
    assert row["tavan"] == Decimal("110.00")
    assert row["taban"] == Decimal("90.00")
    assert row["limitli"] is True or bool(row["limitli"]) is True
    assert row["yon"] == "tavan"
    assert row["guven"] == "KESIN"
    assert row["referans_kaynagi"] == "aof"


def test_aof_reference_shifts_limits_versus_close():
    """Ölçülen kayma sınıfı: AOF 252,992 ↔ kapanış 261,00 → tavan farklı."""
    df = _frame(
        ["2026-11-02", "2026-11-03"],
        [(261, 261, 261, 261, 1_000), (270, 280, 265, 275, 9_000)],
    )
    close_based = flag_limit_sessions(df)
    aof_based = flag_limit_sessions(
        df, aof_provider=_StubProvider({date(2026, 11, 2): Decimal("252.992")})
    )

    assert close_based.iloc[1]["tavan"] == Decimal("287.00")
    assert aof_based.iloc[1]["tavan"] == Decimal("278.25")
    assert aof_based.iloc[1]["tavan"] != close_based.iloc[1]["tavan"]
    assert aof_based.iloc[1]["referans_kaynagi"] == "aof"
    # Varsayılan yol byte-identical: `referans_kaynagi` sütunu YOK.
    assert "referans_kaynagi" not in close_based.columns


# ---------------------------------------------------------------------------
# REQ-001.3 — AOF missing → VERİ YOK, no fallback to close
# ---------------------------------------------------------------------------


def test_missing_aof_on_one_day_yields_veri_yok_without_fallback():
    """Bir günün AOF'u yok → o satır VERİ YOK; kapanışa DÜŞMEZ."""
    df = _frame(
        ["2026-11-02", "2026-11-03", "2026-11-04"],
        [
            (100, 100, 100, 100, 1_000),
            (110, 110, 110, 110, 5_000),
            (121, 121, 121, 121, 300),
        ],
    )
    # 03.11 için referans (02.11) var; 04.11 için referans (03.11) YOK.
    provider = _StubProvider({date(2026, 11, 2): Decimal("100")})
    out = flag_limit_sessions(df, aof_provider=provider)

    middle = out.iloc[1]
    assert middle["referans_kaynagi"] == "aof"
    assert middle["limitli"] is True or bool(middle["limitli"]) is True

    last = out.iloc[2]
    assert bool(last["limitli"]) is False
    assert last["guven"] is None or pd.isna(last["guven"])
    assert "VERİ YOK" in last["gerekce"]
    assert "AOF" in last["gerekce"]
    assert last["tavan"] is None and last["taban"] is None


def test_missing_aof_for_whole_series_yields_veri_yok_everywhere():
    """Tüm seri AOF'suz → her satır VERİ YOK; hiçbir yerde kapanış referansı yok."""
    df = _frame(
        ["2026-11-02", "2026-11-03", "2026-11-04"],
        [
            (100, 100, 100, 100, 1_000),
            (110, 110, 110, 110, 5_000),
            (121, 121, 121, 121, 300),
        ],
    )
    out = flag_limit_sessions(df, aof_provider=_StubProvider({}))

    assert out["limitli"].tolist() == [False, False, False]
    assert out["tavan"].isna().all()
    assert out["taban"].isna().all()
    assert all("VERİ YOK: AOF referansı yok" in reason for reason in out["gerekce"])
    # Hiçbir satır prev_close referansına düşmedi.
    assert set(out["referans_kaynagi"].dropna()) <= {"aof"}
    assert "prev_close" not in out["referans_kaynagi"].dropna().tolist()


# ---------------------------------------------------------------------------
# REQ-001.x — classification still works on AOF limits
# ---------------------------------------------------------------------------


def test_kesin_and_supheli_classification_with_aof_reference():
    """AOF referansıyla KESIN (tek fiyat) ve SUPHELI (dokunup kapanış) ayrımı."""
    df = _frame(
        ["2026-11-02", "2026-11-03", "2026-11-04"],
        [
            (100, 100, 100, 100, 1_000),
            (200, 200, 200, 200, 7_000),  # tek fiyat: tavan 110 DEĞİL -> dokunma yok
            (110, 110, 109, 110, 8_000),  # high=close=tavan(110), low<110 -> SUPHELI
        ],
    )
    provider = _StubProvider(
        {date(2026, 11, 2): Decimal("100"), date(2026, 11, 3): Decimal("100")}
    )
    out = flag_limit_sessions(df, aof_provider=provider)

    # 03.11: tavan 110 ama High=200 → limit teması yok
    assert bool(out.iloc[1]["limitli"]) is False
    # 04.11: referans AOF 100 -> tavan 110; High=Close=110, Low=109 -> SUPHELI
    row = out.iloc[2]
    assert bool(row["limitli"]) is True
    assert row["yon"] == "tavan"
    assert row["guven"] == "SUPHELI"


def test_provider_must_expose_aof_for():
    df = _frame(["2026-11-02"], [(100, 100, 100, 100, 1)])
    with pytest.raises(ValueError):
        flag_limit_sessions(df, aof_provider=object())


# ---------------------------------------------------------------------------
# AOFSeriesProvider — single bulk request, calendar-aware reference day
# ---------------------------------------------------------------------------


def test_aof_series_provider_issues_exactly_one_request():
    """REQ-001: tek toplu istek (N değil); istek sayısı pinli."""
    payload = __import__("json").loads(FIXTURE.read_text(encoding="utf-8"))
    rows = payload["value"]
    frame = pd.DataFrame(
        [
            {
                "aof": None if r["HGDG_AOF"] is None else Decimal(str(r["HGDG_AOF"])),
                "close": None if r["HGDG_KAPANIS"] is None
                else Decimal(str(r["HGDG_KAPANIS"])),
                "volume": None,
            }
            for r in rows
        ],
        index=[
            date(
                int(r["HGDG_TARIH"][6:10]),
                int(r["HGDG_TARIH"][3:5]),
                int(r["HGDG_TARIH"][0:2]),
            )
            for r in rows
        ],
    )
    calls: list[tuple] = []

    def _fetcher(symbol, start=None, end=None):
        calls.append((symbol, start, end))
        return frame, {"rows_returned": len(frame)}

    provider = AOFSeriesProvider(
        "THYAO", date(2026, 11, 1), date(2026, 11, 12), aof_fetcher=_fetcher
    )

    assert len(calls) == 1  # tek toplu istek
    assert provider.request_count == 1
    assert calls[0][0] == "THYAO"
    assert len(provider) == 8  # fixture'daki 8 satır


def test_aof_series_provider_resolves_previous_trading_day():
    """`aof_for` referans günü = önceki işlem günü (takvim-farkındalıklı)."""
    frame = pd.DataFrame(
        {"aof": [Decimal("252.992")], "close": [None], "volume": [None]},
        index=[date(2026, 11, 2)],
    )
    provider = AOFSeriesProvider(
        "THYAO",
        date(2026, 11, 1),
        date(2026, 11, 12),
        aof_fetcher=lambda s, a=None, b=None: (frame, {}),
    )

    # 03.11'in referansı 02.11 → AOF var
    assert provider.aof_for(date(2026, 11, 3)) == Decimal("252.992")
    # 02.11'in referansı 30.10 (Cuma) → kapsamda satır yok
    assert provider.aof_for(date(2026, 11, 2)) is None


def test_aof_series_provider_after_holiday_reference_day():
    """Tatil sonrası referans günü takvimden gelir (29 Ekim tatil → 28 Ekim)."""
    frame = pd.DataFrame(
        {"aof": [Decimal("100")], "close": [None], "volume": [None]},
        index=[date(2026, 10, 28)],
    )
    provider = AOFSeriesProvider(
        "THYAO",
        date(2026, 10, 20),
        date(2026, 11, 2),
        aof_fetcher=lambda s, a=None, b=None: (frame, {}),
    )
    # 30 Ekim'in referansı 28 Ekim (29 Ekim tatil) → AOF bulunur
    assert provider.aof_for(date(2026, 10, 30)) == Decimal("100")
    assert provider.reference_day(date(2026, 10, 30)) == date(2026, 10, 28)


def test_aof_series_provider_rejects_bad_input():
    with pytest.raises(ValueError):
        AOFSeriesProvider("", date(2026, 11, 1), date(2026, 11, 2))
    with pytest.raises(ValueError):
        AOFSeriesProvider("X", date(2026, 11, 5), date(2026, 11, 1))


def test_aof_series_provider_ignores_null_aof_rows():
    """`aof=None` satırı haritaya girmez (VERİ YOK, uydurma yok)."""
    frame = pd.DataFrame(
        {"aof": [None, Decimal("50")], "close": [None, None], "volume": [None, None]},
        index=[date(2026, 11, 2), date(2026, 11, 3)],
    )
    provider = AOFSeriesProvider(
        "X",
        date(2026, 11, 1),
        date(2026, 11, 5),
        aof_fetcher=lambda s, a=None, b=None: (frame, {}),
    )
    assert len(provider) == 1
    assert provider.aof_for(date(2026, 11, 4)) == Decimal("50")  # ref = 03.11


# ---------------------------------------------------------------------------
# Regresyon: calendar + aof_provider birlikte
# ---------------------------------------------------------------------------


def test_calendar_and_aof_provider_together():
    df = _frame(
        ["2026-11-02", "2026-11-03"],
        [(100, 100, 100, 100, 1_000), (110, 110, 110, 110, 0)],
    )
    out = flag_limit_sessions(
        df,
        calendar=bist_calendar,
        aof_provider=_StubProvider({date(2026, 11, 2): Decimal("100")}),
    )
    assert "referans_kaynagi" in out.columns
    assert out.iloc[1]["referans_kaynagi"] == "aof"
    assert "takvim_dogrulanmadi" not in out.iloc[1]["gerekce"]


def test_fixture_sha_pin_unchanged():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest().startswith(
        "cfafa642bff99138"
    )
