"""BIST Pay Piyasası işlem takvimi (resmî tatil tablosuna dayalı).

Kaynak künyesi
--------------
* Sayfa: ``https://www.borsaistanbul.com/resmi-tatil-gunleri``
* Duyuru: «2026 Yılı Resmi Tatil Günleri Nedeniyle Borsa Piyasa ve Pazarlarında
  Uygulanacak İşlem, Tescil ve Takas Esasları Hk.» (duyuru 15282,
  yayım 04.12.2025).
* Tablo: **``/files/pay-piyasasi-2026-yili-tatil-tablosu.pdf``**
  (122.997 B, sha256 ``e45fa97e2f67d85b…``), sayfa 1, «EK-3 BORSA İSTANBUL A.Ş.
  PAY PİYASASI 2026 YILI TATİL TABLOSU».
* Kanıt: ``gecmis baglam/w3a_ek_launch_2026_11_12/kanit/``
  (``pay_piyasasi_2026_tatil.pdf``, ``2026_tatil_kesin.json``,
  ``tatil_satirlari.json`` satır-satır koordinat dökümü).

Kapsam sınırı (dürüst)
----------------------
Kaynak **yalnız 2026** yılını kapsar. 2027 ve sonrası için tatil verisi
**YOKTUR**; bu modül o yıllar için **tahmin üretmez** ve
:class:`CalendarCoverageError` yükseltir.

Yarım günler (arife) **işlem günüdür** (seans yapılır) ama kısadır; bu modül
``is_half_day`` ile ayrıca işaretler. Olağanüstü kapanışlar (deprem, sel vb.)
kaynakta yoksa **kapsam dışıdır**.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

__all__ = [
    "HolidayRow",
    "holiday_rows",
    "BIST_HOLIDAY_SOURCE",
    "CalendarCoverageError",
    "available_years",
    "is_half_day",
    "is_trading_day",
    "previous_trading_day",
    "trading_days",
]

#: Tatil tablosunun makine-okunur künyesi.
BIST_HOLIDAY_SOURCE = {
    "market": "BIST Pay Piyasası",
    "page": "https://www.borsaistanbul.com/resmi-tatil-gunleri",
    "announcement": (
        "2026 Yılı Resmi Tatil Günleri Nedeniyle Borsa Piyasa ve Pazarlarında "
        "Uygulanacak İşlem, Tescil ve Takas Esasları Hk. (duyuru 15282, "
        "yayım 04.12.2025)"
    ),
    "pdf": "https://www.borsaistanbul.com/files/pay-piyasasi-2026-yili-tatil-tablosu.pdf",
    "pdf_sha256_prefix": "e45fa97e2f67d85b",
    "pdf_bytes": 122997,
    "table": "EK-3 BORSA İSTANBUL A.Ş. PAY PİYASASI 2026 YILI TATİL TABLOSU",
    "retrieved": "2026-11-12",
    "coverage_years": (2026,),
    "coverage_note": (
        "Kaynak yalnız 2026 yılını kapsar; diğer yıllar VERİ YOK "
        "(tahmin üretilmez)."
    ),
}

#: 2026 — ``seans yapılmayacaktır`` satırları (resmî tablodan birebir).
_CLOSED_2026: tuple[str, ...] = (
    "2026-01-01",  # Yılbaşı (Perşembe)
    "2026-03-20",  # Ramazan Bayramı 1. gün (Cuma)
    "2026-03-21",  # Ramazan Bayramı (Cumartesi)
    "2026-03-22",  # Ramazan Bayramı (Pazar)
    "2026-04-23",  # Ulusal Egemenlik ve Çocuk Bayramı (Perşembe)
    "2026-05-01",  # Emek ve Dayanışma Günü (Cuma)
    "2026-05-19",  # Atatürk'ü Anma, Gençlik ve Spor Bayramı (Salı)
    "2026-05-27",  # Kurban Bayramı (Çarşamba)
    "2026-05-28",  # Kurban Bayramı (Perşembe)
    "2026-05-29",  # Kurban Bayramı (Cuma)
    "2026-05-30",  # Kurban Bayramı (Cumartesi)
    "2026-07-15",  # Demokrasi ve Millî Birlik Günü (Çarşamba)
    "2026-08-30",  # Zafer Bayramı (Pazar)
    "2026-10-29",  # Cumhuriyet Bayramı (Perşembe)
)

#: 2026 — ``yarım gün seans yapılacaktır`` satırları (arife günleri).
_HALF_2026: tuple[str, ...] = (
    "2026-03-19",  # Ramazan Bayramı arifesi (Perşembe)
    "2026-05-26",  # Kurban Bayramı arifesi (Salı)
    "2026-10-28",  # Cumhuriyet Bayramı arifesi (Çarşamba)
)

_HOLIDAY_TABLE: dict[int, dict[str, frozenset[date]]] = {
    2026: {
        "closed": frozenset(date.fromisoformat(d) for d in _CLOSED_2026),
        "half_day": frozenset(date.fromisoformat(d) for d in _HALF_2026),
    }
}


class CalendarCoverageError(ValueError):
    """Raised when a date falls outside the sourced calendar coverage.

    The exchange calendar has no estimated entries: a year absent from the
    official table is **VERİ YOK**, not "probably a weekday".
    """


def available_years() -> tuple[int, ...]:
    """Years covered by the sourced calendar (ascending)."""
    return tuple(sorted(_HOLIDAY_TABLE))


def _table_for(day: date) -> dict[str, frozenset[date]]:
    """Return the sourced table for ``day``'s year or fail closed."""
    table = _HOLIDAY_TABLE.get(day.year)
    if table is None:
        raise CalendarCoverageError(
            f"BIST tatil tablosu {day.year} yilini kapsamiyor "
            f"(kapsanan yillar: {available_years()}); VERI YOK - tahmin "
            "uretilmez"
        )
    return table


def is_trading_day(day: date) -> bool:
    """Return whether ``day`` is a BIST Pay Piyasası trading day.

    A trading day is a weekday that is not a full-holiday entry. Half days
    (arife) **are** trading days; use :func:`is_half_day` for that detail.

    Args:
        day: Date to test.

    Returns:
        ``True`` for trading days (including half days), else ``False``.

    Raises:
        CalendarCoverageError: If ``day``'s year is not covered by the source.
    """
    if not isinstance(day, date):
        raise ValueError(f"day must be a date, got {type(day).__name__}")
    table = _table_for(day)
    if day.weekday() >= 5:  # Cumartesi/Pazar
        return False
    return day not in table["closed"]


def is_half_day(day: date) -> bool:
    """Return whether ``day`` is a documented half trading session (arife).

    Args:
        day: Date to test.

    Returns:
        ``True`` for arife days listed in the official table.

    Raises:
        CalendarCoverageError: If ``day``'s year is not covered by the source.
    """
    if not isinstance(day, date):
        raise ValueError(f"day must be a date, got {type(day).__name__}")
    return day in _table_for(day)["half_day"]


def previous_trading_day(day: date, *, include_self: bool = False) -> date:
    """Return the trading day before ``day`` (strictly earlier by default).

    Args:
        day: Reference date.
        include_self: When ``True`` and ``day`` is itself a trading day, return
            ``day``.

    Returns:
        The nearest trading day at or before ``day`` (see ``include_self``).

    Raises:
        CalendarCoverageError: If the search leaves the sourced coverage.
    """
    if not isinstance(day, date):
        raise ValueError(f"day must be a date, got {type(day).__name__}")
    _table_for(day)  # fail fast on unsupported years

    cursor = day if include_self else day - timedelta(days=1)
    for _ in range(400):  # güvenlik tavanı; kapsam dışına çıkarsa hata
        if is_trading_day(cursor):
            return cursor
        cursor -= timedelta(days=1)
    raise CalendarCoverageError(
        f"no trading day found before {day} within coverage {available_years()}"
    )


def trading_days(start: date, end: date) -> list[date]:
    """Return every BIST trading day in ``[start, end]`` (inclusive).

    Args:
        start: First day of the window.
        end: Last day of the window.

    Returns:
        Ascending list of trading days (half days included).

    Raises:
        ValueError: If ``start`` is after ``end``.
        CalendarCoverageError: If any part of the window is outside coverage.
    """
    if start > end:
        raise ValueError(f"start {start} is after end {end}")

    days: list[date] = []
    cursor = start
    while cursor <= end:
        if is_trading_day(cursor):
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


@dataclass(frozen=True)
class HolidayRow:
    """One sourced holiday entry (kept for evidence/printing)."""

    day: date
    kind: str  # "closed" | "half_day"

    def as_dict(self) -> dict:
        """Plain-dict view."""
        return {"date": self.day.isoformat(), "kind": self.kind}


def holiday_rows(year: int = 2026) -> tuple[HolidayRow, ...]:
    """Return the sourced holiday rows for ``year`` (evidence view).

    Args:
        year: Covered year.

    Returns:
        Rows sorted by date, ``kind`` being ``closed`` or ``half_day``.

    Raises:
        CalendarCoverageError: If ``year`` is not covered.
    """
    table = _HOLIDAY_TABLE.get(year)
    if table is None:
        raise CalendarCoverageError(
            f"BIST tatil tablosu {year} yilini kapsamiyor "
            f"(kapsanan yillar: {available_years()})"
        )
    rows = [HolidayRow(d, "closed") for d in table["closed"]]
    rows += [HolidayRow(d, "half_day") for d in table["half_day"]]
    return tuple(sorted(rows, key=lambda row: (row.day, row.kind)))
