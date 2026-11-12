"""BIST günlük AOF (ağırlıklı ortalama fiyat) verisi — İş Yatırım kaynağı.

Kaynak künyesi
--------------
* Sağlayıcı: **İş Yatırım** (BIST'in kendi veri sağlayıcısı).
* Uç nokta: ``https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil``
  (``?hisse=<SEMBOL>&startdate=DD-MM-YYYY&enddate=DD-MM-YYYY``)
* Alan eşlemesi (yanıt ``value[]`` nesneleri):

  ================  ==========================================
  Kaynak alanı      Anlamı
  ================  ==========================================
  ``HGDG_TARIH``    işlem günü (``DD-MM-YYYY``)
  ``HGDG_AOF``      **günlük ağırlıklı ortalama fiyat (TL)**
  ``HGDG_KAPANIS``  kapanış (TL)
  ``HGDG_HACIM``    işlem hacmi (TL)
  ================  ==========================================

* Ölçülen kanıt (2026-11-12, kanıt klasörü
  ``gecmis baglam/w3a_ek_launch_2026_11_12/kanit/``):
  ``aof_THYAO_20261101_20261112.json`` (sha256 ``cfafa642bff99138…``, 8 satır) ve
  ``aof_GARAN_…json`` (sha256 ``3ca8969f9aa301da…``, 9 satır).
  THYAO 02-11-2026: ``HGDG_KAPANIS`` = 261,00 · **``HGDG_AOF`` = 252,992**
  → kapanış ≠ AOF, yani fiyat limiti referansı için AOF **gereklidir** (R1).

Neden ayrı modül: ``bist_limits.py`` W3a dalgasında land edildi ve bu dalga
onu **değiştirmez** (AOF'u ``compute_price_limits``'a bağlamak ayrı iştir).
Bu modül yalnız veri sağlar; hiçbir hesap yapmaz.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

import pandas as pd

from borsapy._providers.base import BaseProvider
from borsapy.exceptions import APIError

__all__ = [
    "AOF_SOURCE",
    "AOFRow",
    "AOFDiagnostics",
    "get_daily_aof",
]

#: Kaynağın makine-okunur künyesi (sayı tek başına basılmaz).
AOF_SOURCE = {
    "provider": "İş Yatırım (BIST veri sağlayıcısı)",
    "endpoint": (
        "https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/"
        "Common/Data.aspx/HisseTekil"
    ),
    "fields": {
        "HGDG_TARIH": "işlem günü (DD-MM-YYYY)",
        "HGDG_AOF": "günlük ağırlıklı ortalama fiyat (TL)",
        "HGDG_KAPANIS": "kapanış (TL)",
        "HGDG_HACIM": "işlem hacmi (TL)",
    },
    "evidence": {
        "THYAO": {
            "file": "aof_THYAO_20261101_20261112.json",
            "sha256_prefix": "cfafa642bff99138",
            "rows": 8,
        },
        "GARAN": {
            "file": "aof_GARAN_20261101_20261112.json",
            "sha256_prefix": "3ca8969f9aa301da",
            "rows": 9,
        },
    },
    "retrieved": "2026-11-12",
    "note": (
        "THYAO 02-11-2026 kapanış=261.00, AOF=252.992 — kapanış != AOF; "
        "limit referansı için AOF gereklidir."
    ),
}


@dataclass(frozen=True)
class AOFRow:
    """One trading day's AOF record (all values ``Decimal`` where numeric)."""

    trade_date: date
    aof: Decimal | None
    close: Decimal | None
    volume: Decimal | None
    raw: dict

    def as_dict(self) -> dict:
        """Plain-dict view with ISO date (Decimal values preserved)."""
        return {
            "date": self.trade_date.isoformat(),
            "aof": self.aof,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass(frozen=True)
class AOFDiagnostics:
    """Honest counts about what the source did *not* provide.

    Missing days are **absent rows**, never fabricated rows; these counters
    make the absence visible.
    """

    requested_start: date
    requested_end: date
    rows_returned: int
    aof_missing: int
    close_missing: int
    volume_missing: int
    calendar_days: int

    def as_dict(self) -> dict:
        """Plain-dict view."""
        return {
            "requested_start": self.requested_start.isoformat(),
            "requested_end": self.requested_end.isoformat(),
            "calendar_days": self.calendar_days,
            "rows_returned": self.rows_returned,
            "aof_missing": self.aof_missing,
            "close_missing": self.close_missing,
            "volume_missing": self.volume_missing,
            "veri_yok_note": (
                "kaynakta olmayan gün satır ÜRETİLMEZ; sayılar burada raporlanır"
            ),
        }


def _to_decimal(value: object) -> Decimal | None:
    """Source number → ``Decimal`` (no float rounding), ``None`` when absent."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    if isinstance(value, str):
        text = value.strip().replace(" ", "")
        if not text:
            return None
        try:
            if "," in text:
                # Türk biçimi: '.' binlik, ',' ondalık.
                return Decimal(text.replace(".", "").replace(",", "."))
            if text.count(".") > 1:
                # Yalnız binlik ayraclı tam sayı (ör. 1.234.567).
                return Decimal(text.replace(".", ""))
            # Tek nokta -> ondalık ayraç (JSON sayılarıyla tutarlı).
            return Decimal(text)
        except InvalidOperation:
            return None
    return None


def _to_date(value: object) -> date | None:
    """``DD-MM-YYYY`` (source format) → :class:`datetime.date`."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    for fmt in ("%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


class IsYatirimAOFProvider(BaseProvider):
    """Fetch BIST daily AOF from İş Yatırım (read-only, one request per call)."""

    AOF_URL = AOF_SOURCE["endpoint"]
    TIMEOUT_S = 40

    def fetch(
        self, symbol: str, start: date | None = None, end: date | None = None
    ) -> tuple[list[AOFRow], AOFDiagnostics]:
        """Return the raw AOF rows plus honest diagnostics for a range.

        Args:
            symbol: BIST ticker (e.g. ``"THYAO"``).
            start: First requested day (defaults to ``end`` − 30 days).
            end: Last requested day (defaults to today).

        Returns:
            ``(rows, diagnostics)``. Rows carry the source values only.

        Raises:
            APIError: On transport error or a non-``ok`` source response.
        """
        ticker = symbol.strip().upper()
        if not ticker:
            raise ValueError("symbol must be a non-empty string")

        end_date = end or date.today()
        start_date = start or end_date.replace(
            year=end_date.year, month=end_date.month, day=1
        )
        if start_date > end_date:
            raise ValueError(f"start {start_date} is after end {end_date}")

        params = {
            "hisse": ticker,
            "startdate": start_date.strftime("%d-%m-%Y"),
            "enddate": end_date.strftime("%d-%m-%Y"),
        }

        try:
            response = self._client.get(
                self.AOF_URL,
                params=params,
                timeout=self.TIMEOUT_S,
                headers={
                    "Accept": "application/json",
                    "Referer": (
                        "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/"
                        "Sayfalar/Tarihsel-Fiyat-Bilgileri.aspx"
                    ),
                },
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # transport/parse
            raise APIError(f"AOF fetch failed for {ticker}: {exc}") from exc

        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise APIError(
                f"AOF source returned ok!={payload.get('ok') if isinstance(payload, dict) else payload!r}"
            )

        rows: list[AOFRow] = []
        for item in payload.get("value") or []:
            if not isinstance(item, dict):
                continue
            trade_date = _to_date(item.get("HGDG_TARIH"))
            if trade_date is None:
                continue
            rows.append(
                AOFRow(
                    trade_date=trade_date,
                    aof=_to_decimal(item.get("HGDG_AOF")),
                    close=_to_decimal(item.get("HGDG_KAPANIS")),
                    volume=_to_decimal(item.get("HGDG_HACIM")),
                    raw=dict(item),
                )
            )

        rows.sort(key=lambda row: row.trade_date)
        diagnostics = AOFDiagnostics(
            requested_start=start_date,
            requested_end=end_date,
            rows_returned=len(rows),
            aof_missing=sum(1 for row in rows if row.aof is None),
            close_missing=sum(1 for row in rows if row.close is None),
            volume_missing=sum(1 for row in rows if row.volume is None),
            calendar_days=(end_date - start_date).days + 1,
        )
        return rows, diagnostics


def get_daily_aof(
    symbol: str,
    start: date | None = None,
    end: date | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return daily AOF as a DataFrame with the source credit attached.

    Args:
        symbol: BIST ticker (e.g. ``"THYAO"``).
        start: First day requested (defaults to the 1st of ``end``'s month).
        end: Last day requested (defaults to today).

    Returns:
        ``(frame, diagnostics)`` where ``frame`` has ``date`` (index),
        ``aof``/``close``/``volume`` as ``Decimal`` and ``frame.attrs``
        carries :data:`AOF_SOURCE`. Missing days are **absent rows**.

    Raises:
        APIError: If the source is unreachable or answers ``ok != true``.
    """
    provider = IsYatirimAOFProvider()
    rows, diagnostics = provider.fetch(symbol, start, end)

    frame = pd.DataFrame(
        [
            {
                "date": row.trade_date,
                "aof": row.aof,
                "close": row.close,
                "volume": row.volume,
            }
            for row in rows
        ],
        columns=["date", "aof", "close", "volume"],
    )
    if not frame.empty:
        frame = frame.set_index("date")

    frame.attrs["source"] = dict(AOF_SOURCE)
    frame.attrs["diagnostics"] = diagnostics.as_dict()
    return frame, diagnostics.as_dict()
