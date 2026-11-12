"""AOF (günlük ağırlıklı ortalama fiyat) testleri — kayıtlı yanıtla, ağ yok.

Fixture'lar **kaynak yanıtının birebir kopyasıdır** (İş Yatırım
``Data.aspx/HisseTekil``), sha256 pinlidir; beklenen değerler bu yanıttan
bağımsız olarak okunmuştur (2026-11-12 çekimi).

Kanıt: ``gecmis baglam/w3a_ek_launch_2026_11_12/kanit/``
(``aof_THYAO_…json`` sha ``cfafa642bff99138…``, ``aof_GARAN_…json`` sha
``3ca8969f9aa301da…``).
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from borsapy._providers.isyatirim_aof import (
    AOF_SOURCE,
    IsYatirimAOFProvider,
    _to_date,
    _to_decimal,
    get_daily_aof,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "isyatirim"
THYAO_FIXTURE = FIXTURE_DIR / "aof_thyao_2026_11.json"
GARAN_FIXTURE = FIXTURE_DIR / "aof_garan_2026_11.json"

THYAO_SHA = "cfafa642bff99138"
GARAN_SHA = "3ca8969f9aa301da"


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


class _Resp:
    def __init__(self, text: str, status: int = 200):
        self.text = text
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return json.loads(self.text)


class _Client:
    def __init__(self, text: str, status: int = 200, exc: Exception | None = None):
        self._text = text
        self._status = status
        self._exc = exc
        self.urls: list[str] = []
        self.params: list[dict] = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        self.params.append(kwargs.get("params") or {})
        if self._exc is not None:
            raise self._exc
        return _Resp(self._text, self._status)


def _provider(fixture: Path) -> tuple[IsYatirimAOFProvider, _Client]:
    provider = IsYatirimAOFProvider()
    client = _Client(fixture.read_text(encoding="utf-8"))
    provider._client = client
    return provider, client


# ---------------------------------------------------------------------------
# Fixture integrity + source metadata
# ---------------------------------------------------------------------------


def test_aof_fixture_checksums():
    assert _sha16(THYAO_FIXTURE) == THYAO_SHA
    assert _sha16(GARAN_FIXTURE) == GARAN_SHA


def test_aof_source_metadata_is_stamped():
    assert AOF_SOURCE["endpoint"].endswith("/Data.aspx/HisseTekil")
    assert AOF_SOURCE["fields"]["HGDG_AOF"] == "günlük ağırlıklı ortalama fiyat (TL)"
    assert AOF_SOURCE["evidence"]["THYAO"]["sha256_prefix"] == THYAO_SHA
    assert "252.992" in AOF_SOURCE["note"]


# ---------------------------------------------------------------------------
# Output-level: rows + Decimal pins
# ---------------------------------------------------------------------------


def test_fetch_returns_rows_and_decimal_pins():
    """Çıktı düzeyi: kayıtlı yanıt -> AOFRow listesi, Decimal değerler."""
    provider, client = _provider(THYAO_FIXTURE)
    rows, diagnostics = provider.fetch(
        "THYAO", date(2026, 11, 1), date(2026, 11, 12)
    )

    assert len(rows) == 8
    assert len(client.urls) == 1  # tek istek
    assert diagnostics.rows_returned == 8
    assert diagnostics.aof_missing == 0

    first = rows[0]
    assert first.trade_date == date(2026, 11, 2)
    # Pin: kaynak yanıtındaki değerlerin birebir Decimal karşılığı.
    assert first.aof == Decimal("252.992")
    assert first.close == Decimal("261.0")
    assert isinstance(first.aof, Decimal) and isinstance(first.close, Decimal)
    # Kapanış != AOF (R1 gerekçesi).
    assert first.aof != first.close


def test_fetch_is_sorted_ascending():
    provider, _ = _provider(GARAN_FIXTURE)
    rows, _diag = provider.fetch("GARAN", date(2026, 11, 1), date(2026, 11, 12))
    dates = [row.trade_date for row in rows]
    assert dates == sorted(dates)


def test_fetch_passes_symbol_and_dates_to_source():
    provider, client = _provider(THYAO_FIXTURE)
    provider.fetch("thyao", date(2026, 11, 1), date(2026, 11, 12))
    params = client.params[0]
    assert params["hisse"] == "THYAO"
    assert params["startdate"] == "01-11-2026"
    assert params["enddate"] == "12-11-2026"


def test_get_daily_aof_frame_output():
    """Çıktı düzeyi: DataFrame + künye + diagnostics."""
    provider, _ = _provider(THYAO_FIXTURE)
    import borsapy._providers.isyatirim_aof as mod

    original = mod.IsYatirimAOFProvider
    mod.IsYatirimAOFProvider = lambda: provider  # type: ignore[assignment]
    try:
        frame, diagnostics = get_daily_aof(
            "THYAO", date(2026, 11, 1), date(2026, 11, 12)
        )
    finally:
        mod.IsYatirimAOFProvider = original

    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["aof", "close", "volume"]
    assert len(frame) == 8
    assert frame.index[0] == date(2026, 11, 2)
    assert frame.iloc[0]["aof"] == Decimal("252.992")
    assert frame.attrs["source"]["fields"]["HGDG_AOF"].startswith("günlük")
    assert diagnostics["rows_returned"] == 8
    assert diagnostics["veri_yok_note"].startswith("kaynakta olmayan gün")


def test_diagnostics_counts_missing_values():
    """Eksik değer satır ÜRETMEZ, diagnostics'te sayılır."""
    payload = {
        "ok": True,
        "value": [
            {"HGDG_TARIH": "02-11-2026", "HGDG_AOF": None,
             "HGDG_KAPANIS": 261.0, "HGDG_HACIM": 1000},
            {"HGDG_TARIH": "03-11-2026", "HGDG_AOF": 264.51,
             "HGDG_KAPANIS": None, "HGDG_HACIM": None},
        ],
    }
    provider = IsYatirimAOFProvider()
    provider._client = _Client(json.dumps(payload))
    rows, diag = provider.fetch("X", date(2026, 11, 1), date(2026, 11, 12))

    assert len(rows) == 2
    assert diag.aof_missing == 1
    assert diag.close_missing == 1
    assert diag.volume_missing == 1
    assert rows[0].aof is None  # uydurma yok


def test_source_not_ok_raises():
    from borsapy.exceptions import APIError

    provider = IsYatirimAOFProvider()
    provider._client = _Client(json.dumps({"ok": False, "value": []}))
    with pytest.raises(APIError):
        provider.fetch("X", date(2026, 11, 1), date(2026, 11, 12))


def test_transport_error_raises():
    from borsapy.exceptions import APIError

    provider = IsYatirimAOFProvider()
    provider._client = _Client("", exc=RuntimeError("network down"))
    with pytest.raises(APIError):
        provider.fetch("X", date(2026, 11, 1), date(2026, 11, 12))


def test_start_after_end_raises():
    provider = IsYatirimAOFProvider()
    provider._client = _Client("{}")
    with pytest.raises(ValueError):
        provider.fetch("X", date(2026, 11, 12), date(2026, 11, 1))


# ---------------------------------------------------------------------------
# Parsing helpers (targeted, not the only assertion)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        (261.0, Decimal("261.0")),
        ("252.992", Decimal("252.992")),  # tek nokta -> ondalık ayraç
        ("1.234,56", Decimal("1234.56")),  # virgül -> Türk biçimi
        ("1.234.567", Decimal("1234567")),  # çok nokta -> binlik
        (None, None),
        ("", None),
    ),
)
def test_to_decimal_conversions(raw, expected):
    assert _to_decimal(raw) == expected


def test_to_date_source_format():
    assert _to_date("02-11-2026") == date(2026, 11, 2)
    assert _to_date("2026-11-02") == date(2026, 11, 2)
    assert _to_date("bozuk") is None
