"""VBTS bant güvenliği — çağıranı yanlış bant varsayımından korur.

Arka plan
---------
VBTS (Volatilite Bazlı Tedbir Sistemi) kapsamındaki paylarda fiyat bandı
**daraltılır**. Borsapy'de daraltılmış bant **oranı yoktur** (kaynak yok →
VERİ YOK, uydurma yasak). Bu yüzden bir çağıran VBTS'li sembol için
varsayılan ±%10 bandı kullanırsa **yanlış tavan/taban** hesaplar.

Bu modül hesap yapmaz; yalnız **kapı** koyar: sembol VBTS listesindeyse ve
çağıran "bu bant doğrulanmış" demiyorsa hata yükseltir.

VBTS listesi
------------
``KAPProvider.get_vbts_status()`` (075c'de land edildi) **yeniden yazılmaz**;
bu modül yalnız onun çıktısını tüketir. Liste yoksa/erişilemiyorsa bu bir
**hata**dır — sessiz geçiş yoktur.

Kaynak künyesi: ``gecmis baglam/vbts_dkb_arastirma_2026_11_09/`` (mühürlü).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "BandSourceUnverifiedError",
    "VBTS_ACKNOWLEDGED_BAND_SOURCES",
    "assert_band_source_safe",
    "vbts_symbols_from_status",
]

#: Çağıranın "daraltılmış bandı biliyorum" dediğini gösteren kaynak etiketleri.
#: ``bist_limits.compute_price_limits`` bu etiketi ``band_source`` alanına
#: yazar (varsayılan bant ``"default"``dır ve burada **güvenli değildir**).
VBTS_ACKNOWLEDGED_BAND_SOURCES: frozenset[str] = frozenset(
    {"caller_supplied", "vbts_explicit"}
)


class BandSourceUnverifiedError(ValueError):
    """Raised when a VBTS symbol would be priced with an unverified band.

    Attributes:
        symbol: The offending ticker.
        band_source: The band source label the caller supplied.
    """

    def __init__(self, symbol: str, band_source: str, reason: str) -> None:
        self.symbol = symbol
        self.band_source = band_source
        self.reason = reason
        super().__init__(
            f"VBTS guard: {symbol} icin bant kaynagi dogrulanmadi "
            f"(band_source={band_source!r}); {reason}"
        )


def vbts_symbols_from_status(status: Mapping[str, Any] | None) -> frozenset[str]:
    """Extract the VBTS symbol set from ``KAPProvider.get_vbts_status()``.

    Args:
        status: The dict returned by ``get_vbts_status`` (or ``None``).

    Returns:
        Frozenset of upper-case tickers that currently carry a VBTS measure
        (``within_window`` is honoured when present).

    Raises:
        BandSourceUnverifiedError: If ``status`` is missing or has no usable
            ``symbols`` mapping — an unavailable list must never be read as
            "no VBTS symbols".
    """
    if not isinstance(status, Mapping):
        raise BandSourceUnverifiedError(
            "<unknown>",
            "<none>",
            "VBTS listesi yok/erisilemedi (sessiz gecis yasak)",
        )

    symbols = status.get("symbols")
    if not isinstance(symbols, Mapping):
        raise BandSourceUnverifiedError(
            "<unknown>",
            "<none>",
            "VBTS listesi 'symbols' alanini icermiyor (VERI YOK)",
        )

    selected: set[str] = set()
    for raw_symbol, entry in symbols.items():
        if not isinstance(raw_symbol, str) or not raw_symbol.strip():
            continue
        ticker = raw_symbol.strip().upper()
        if isinstance(entry, Mapping):
            if entry.get("vbts_var") is False:
                continue
            if entry.get("within_window") is False:
                continue
        selected.add(ticker)
    return frozenset(selected)


def assert_band_source_safe(
    symbol: str,
    band_source: str,
    vbts_symbols: Iterable[str] | Mapping[str, Any] | None,
) -> None:
    """Fail closed when a VBTS symbol would use an unverified price band.

    Args:
        symbol: Ticker being priced (e.g. ``"THYAO"``).
        band_source: Band provenance label, typically
            ``PriceLimits.band_source`` (``"default"`` when the ±%10 default
            was used).
        vbts_symbols: Either an already-extracted iterable of tickers, or the
            ``KAPProvider.get_vbts_status()`` mapping (from which symbols are
            extracted). ``None`` is treated as "list unavailable" → error.

    Returns:
        ``None`` when the caller may proceed.

    Raises:
        BandSourceUnverifiedError: If ``symbol`` is in the VBTS set and
            ``band_source`` is not one of
            :data:`VBTS_ACKNOWLEDGED_BAND_SOURCES`; or if the VBTS list is
            unavailable.
    """
    ticker = (symbol or "").strip().upper()
    if not ticker:
        raise ValueError("symbol must be a non-empty string")

    if isinstance(vbts_symbols, Mapping):
        current = vbts_symbols
    elif vbts_symbols is None:
        current = None
    else:
        current = {
            "symbols": {
                str(item).strip().upper(): {"vbts_var": True}
                for item in vbts_symbols
                if str(item).strip()
            }
        }

    affected = vbts_symbols_from_status(current)  # raises when unavailable

    if not affected:
        # Boş liste "VBTS'li sembol yok" diye OKUNAMAZ: liste gerçekten boş mu,
        # yoksa çekilemedi mi ayırt edilemez -> fail-closed (spec REQ-003).
        raise BandSourceUnverifiedError(
            ticker,
            band_source or "<empty>",
            "VBTS listesi bos/erişilemedi; 'VBTS yok' varsayimi yapılamaz "
            "(sessiz gecis yasak)",
        )

    source = (band_source or "").strip()
    if ticker in affected and source not in VBTS_ACKNOWLEDGED_BAND_SOURCES:
        raise BandSourceUnverifiedError(
            ticker,
            source or "<empty>",
            "sembol VBTS kapsaminda; daraltilmis bant orani kaynakta YOK - "
            "cagiran bilincli karar vermeli (band_source='caller_supplied' "
            "veya 'vbts_explicit')",
        )
