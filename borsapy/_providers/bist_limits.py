"""BIST Pay Piyasası fiyat adımları, tavan/taban ve limitli seans bayrağı.

Bu modül **Wyckoff faz makinesinden (W3) önce** gereken veri katmanını sağlar:
limitli seanslar (tavan/taban kilitli günler) hacim-fiyat analizini bozar, bu
yüzden işaretlenmelidir (ERRATUM v2 A7).

Kaynak künyeleri
----------------
* **Fiyat adımı tablosu** — BIST duyuru *«Pay Piyasasında Fiyat Adımı ve
  Kotasyon Yayılma Aralıkları Düzenlemeleri…»*, **28.08.2023**, Tablo 1.
  Ek dosya: ``https://borsaistanbul.com/files/duyuru-19412.pdf``
  (651.061 B, sha256 ``cb0a1e0091d79918…``). Kanıt:
  ``gecmis baglam/w3a_veri_launch_2026_11_12/KAYNAK_KANITI.md``.
* **AOF (ağırlıklı ortalama fiyat)** — ``Ticker.history()`` OHLCV döndürür,
  AOF alanı **YOK**. Bu yüzden limit referansı için belgelenmiş yaklaşım
  **önceki kapanış** kullanılır ve her sonuç ``reference_source`` damgası taşır.
* **VBTS daraltılmış bant** — hisse bazlı daraltılmış bant oranları için
  kaynak **YOK**; ``band_source="default"`` damgalanır, uydurma yapılmaz.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

import pandas as pd

__all__ = [
    "PRICE_TICK_TABLE",
    "PRICE_TICK_TABLE_SOURCE",
    "LIMIT_BAND_PCT_DEFAULT",
    "LIMIT_EQUALITY_TOLERANCE_PCT",
    "LIMIT_VOLUME_RATIO_V0",
    "PriceLimits",
    "AOFUnavailableError",
    "adtv",
    "adtv_diagnostics",
    "compute_price_limits",
    "compute_price_limits_from_aof",
    "flag_limit_sessions",
    "round_to_tick",
    "tick_for_price",
]

# ---------------------------------------------------------------------------
# Fiyat adımı tablosu — KAYNAKLI VERİ (sabit; hesaplanmaz)
# ---------------------------------------------------------------------------
#: Tablonun künyesi (kod içinde taşınır; sayı tek başına basılmaz).
PRICE_TICK_TABLE_SOURCE = {
    "source": "Borsa İstanbul A.Ş. duyuru 14571",
    "title": (
        "Pay Piyasasında Fiyat Adımı ve Kotasyon Yayılma Aralıkları "
        "Düzenlemeleri ile Borsa Yatırım Fonlarında Açılış Seansı "
        "Uygulanmasına Yönelik Değişiklikler"
    ),
    "published": "2023-08-28",
    "url": "https://borsaistanbul.com/files/duyuru-19412.pdf",
    "sha256_prefix": "cb0a1e0091d79918",
    "table": "Tablo 1 — Paylar, Yeni Pay Alma Hakları, Gayrimenkul "
    "Sertifikaları, Sahipliğe Dayalı Kira Sertifikaları Fiyat Adımları",
    "retrieved": "2026-11-12",
}

#: ``(alt_sınır_dahil, üst_sınır_dahil | None, adım)`` — Tablo 1, aynen.
#: Üst sınır ``None`` = açık uçlu ("2.500 ve üzeri").
PRICE_TICK_TABLE: tuple[tuple[Decimal, Decimal | None, Decimal], ...] = (
    (Decimal("0.01"), Decimal("19.99"), Decimal("0.01")),
    (Decimal("20"), Decimal("49.99"), Decimal("0.02")),
    (Decimal("50"), Decimal("99.99"), Decimal("0.05")),
    (Decimal("100"), Decimal("249.99"), Decimal("0.10")),
    (Decimal("250"), Decimal("499.99"), Decimal("0.25")),
    (Decimal("500"), Decimal("999.99"), Decimal("0.50")),
    (Decimal("1000"), Decimal("2499.99"), Decimal("1.00")),
    (Decimal("2500"), None, Decimal("2.50")),
)

#: Yüzde bant genişliği — **PARAMETRE + VARSAYILAN damgası** (K0-a c).
#: BIST pay piyasası standart bant ±%10'dur; bu bir seçim değil varsayılandır,
#: çağıran daraltılmış bant (VBTS) için kendi değerini geçebilir.
LIMIT_BAND_PCT_DEFAULT = Decimal("0.10")

#: P5 (ERRATUM v2 §G): tavan/taban **eşitlik toleransı** = ±%0,1.
#: Bant genişliğinden AYRI parametredir.
LIMIT_EQUALITY_TOLERANCE_PCT = Decimal("0.001")

#: Hacim eşiği — **KALİBRE EDİLMEDİ (v0 parametre)**.
#: Ampirik kalibrasyon yapılmadan "mühürlü" sayılamaz; yalnız SUPHELİ
#: sınıfında kullanılır ve karar zincirine girmez (ERRATUM v2 D2/D8).
LIMIT_VOLUME_RATIO_V0 = 0.5

#: Yuvarlama modu — kaynak tablo yuvarlama YÖNÜ belirtmiyor; en yakın adım
#: (yarım yukarı) uygulanır ve sonuç ``rounding`` alanında damgalanır.
_ROUNDING_MODE = "nearest_tick(ROUND_HALF_UP)"

#: Bant geçişinde yakınsama için azami tur (adım değişimi en fazla birkaç kez).
_MAX_ROUNDS = 4


def _as_decimal(value: object, *, field_name: str) -> Decimal:
    """``Decimal``'e güvenli çevrim (float yuvarlaması kullanılmaz)."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number, got bool")
    if isinstance(value, numbers.Integral):
        return Decimal(int(value))
    if isinstance(value, numbers.Real):
        # str() üzerinden: float ikili gösterimini Decimal'e taşımayız.
        return Decimal(str(float(value)))
    if isinstance(value, str):
        try:
            return Decimal(value.strip().replace(",", "."))
        except InvalidOperation as exc:
            raise ValueError(f"{field_name} is not a number: {value!r}") from exc
    raise ValueError(f"{field_name} is not a number: {value!r}")


def tick_for_price(price: object) -> Decimal:
    """Return the BIST tick size of the band that contains ``price``.

    Args:
        price: Price in TL (>= 0.01).

    Returns:
        Tick size as :class:`~decimal.Decimal`.

    Raises:
        ValueError: If ``price`` is not numeric or is below the first band.
    """
    value = _as_decimal(price, field_name="price")
    if value < PRICE_TICK_TABLE[0][0]:
        raise ValueError(
            f"price {value} is below the first BIST band "
            f"({PRICE_TICK_TABLE[0][0]})"
        )
    for lower, upper, tick in PRICE_TICK_TABLE:
        if value >= lower and (upper is None or value <= upper):
            return tick
    # Açık uçlu son satır her zaman eşleşir; buraya düşmek tablo hatasıdır.
    raise ValueError(f"no BIST tick band matched price {value}")


def round_to_tick(price: object) -> Decimal:
    """Round ``price`` to the nearest valid BIST tick (``Decimal`` only).

    The tick depends on the price level, so a rounding step can move the value
    into a neighbouring band. The resolution is therefore iterative: round with
    the band of the current value, re-resolve the band, repeat at most
    ``_MAX_ROUNDS`` times. Converged value is returned.

    Args:
        price: Price in TL.

    Returns:
        Tick-aligned :class:`~decimal.Decimal`.

    Raises:
        ValueError: If ``price`` is not numeric or is not positive.
    """
    value = _as_decimal(price, field_name="price")
    if value <= 0:
        raise ValueError(f"price must be positive, got {value}")

    current = value
    for _ in range(_MAX_ROUNDS):
        tick = tick_for_price(current)
        rounded = (current / tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick
        if rounded == current:
            return rounded
        current = rounded
    return current


@dataclass(frozen=True)
class PriceLimits:
    """Tavan/taban for one reference price, with full provenance stamps."""

    reference: Decimal
    band_pct: Decimal
    tavan: Decimal
    taban: Decimal
    reference_source: str
    band_source: str
    rounding: str = _ROUNDING_MODE
    equality_tolerance_pct: Decimal = LIMIT_EQUALITY_TOLERANCE_PCT
    notes: tuple[str, ...] = field(default_factory=tuple)
    #: W3b: AOF'dan türetilen okumalarda referansın ait olduğu işlem günü.
    #: Varsayılan ``None`` — mevcut çağrıların çıktısı değişmez.
    reference_date: date | None = None
    #: W3b: ``as_of`` işlem günü değilse (hafta sonu/tatil) girdi burada.
    as_of_normalized_from: date | None = None

    def as_dict(self) -> dict[str, object]:
        """Plain-dict view (Decimal values kept as Decimal).

        The two ``W3b`` keys appear **only when set**, so plain
        :func:`compute_price_limits` output is byte-identical to before.
        """
        view: dict[str, object] = {
            "reference": self.reference,
            "band_pct": self.band_pct,
            "tavan": self.tavan,
            "taban": self.taban,
            "reference_source": self.reference_source,
            "band_source": self.band_source,
            "rounding": self.rounding,
            "equality_tolerance_pct": self.equality_tolerance_pct,
            "notes": list(self.notes),
        }
        if self.reference_date is not None:
            view["reference_date"] = self.reference_date
        if self.as_of_normalized_from is not None:
            view["as_of_normalized_from"] = self.as_of_normalized_from
        return view


class AOFSeriesProvider:
    """Daily AOF series for a symbol, fetched with **one** bulk request.

    Contract expected by :func:`flag_limit_sessions` when ``aof_provider`` is
    supplied: ``aof_for(day) -> Decimal | None``, where the returned value is
    the **reference** AOF for ``day`` (BIST rule: the previous trading
    session's weighted average price). ``None`` means VERİ YOK.

    The previous-session resolution is handled **inside** this provider (the
    series-level caller only knows the row's own date), using the sourced BIST
    calendar. The whole range is downloaded once — never N requests.

    Attributes:
        symbol: Upper-case ticker.
        request_count: HTTP requests issued so far (1 after a successful bulk
            fetch); pinned by tests.
    """

    def __init__(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        aof_fetcher: object | None = None,
    ) -> None:
        """Fetch the AOF series for ``symbol`` once and build the day map.

        Args:
            symbol: BIST ticker.
            start: First session of the requested window.
            end: Last session of the requested window.
            aof_fetcher: Test seam; ``(symbol, start, end) -> (frame, diag)``.
                Defaults to ``isyatirim_aof.get_daily_aof``.

        Raises:
            ValueError: Empty symbol or ``start`` after ``end``.
        """
        ticker = (symbol or "").strip().upper()
        if not ticker:
            raise ValueError("symbol must be a non-empty string")
        if start > end:
            raise ValueError(f"start {start} is after end {end}")

        self.symbol = ticker
        self.start = start
        self.end = end
        self.request_count = 0

        fetcher = aof_fetcher
        if fetcher is None:
            from borsapy._providers.isyatirim_aof import get_daily_aof

            fetcher = get_daily_aof

        frame, _diagnostics = fetcher(ticker, start, end)
        self.request_count = 1

        #: ``{trading_day: AOF}`` — only days the source actually returned.
        self._aof_by_day: dict[date, Decimal] = {}
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            for label, value in frame["aof"].items():
                if value is None or _is_missing(value):
                    continue
                day = _as_date(label)
                if day is None:
                    continue
                self._aof_by_day[day] = (
                    value
                    if isinstance(value, Decimal)
                    else _as_decimal(value, field_name="aof")
                )

    def reference_day(self, day: date) -> date:
        """Return the AOF reference session for ``day`` (previous trading day).

        Args:
            day: The session being priced.

        Returns:
            The trading session strictly before ``day``.

        Raises:
            CalendarCoverageError: If the calendar cannot resolve the day.
        """
        from borsapy._providers.bist_calendar import previous_trading_day

        return previous_trading_day(day)

    def aof_for(self, day: date) -> Decimal | None:
        """Return the reference AOF for the session ``day`` (or ``None``).

        Args:
            day: The session being priced.

        Returns:
            The previous session's AOF as :class:`~decimal.Decimal`, or
            ``None`` when the source has no row for that session (VERİ YOK).

        Raises:
            CalendarCoverageError: If the calendar cannot resolve ``day``.
        """
        return self._aof_by_day.get(self.reference_day(day))

    def __len__(self) -> int:
        """Number of days with an AOF value."""
        return len(self._aof_by_day)


class AOFUnavailableError(ValueError):
    """Raised when the AOF for the reference session is not available.

    Falling back to the previous close would silently shift the price limits
    (measured: THYAO 02.11.2026 close 261.00 vs AOF 252.992 = 3.1%), so the
    caller must decide explicitly instead.
    """

    def __init__(self, symbol: str, reference_date: date, reason: str) -> None:
        self.symbol = symbol
        self.reference_date = reference_date
        self.reason = reason
        super().__init__(
            f"AOF yok: {symbol} / {reference_date.isoformat()} — {reason}; "
            "kapanisa sessiz dusme YASAK (cagiran bilincli karar vermeli)"
        )


def compute_price_limits(
    reference: object,
    band_pct: object = LIMIT_BAND_PCT_DEFAULT,
    *,
    equality_tolerance_pct: object = LIMIT_EQUALITY_TOLERANCE_PCT,
) -> PriceLimits:
    """Compute BIST tavan/taban around a reference price.

    Args:
        reference: Reference price in TL. For BIST the official reference is the
            previous session's weighted average price (AOF); the data layer has
            no AOF, so callers pass the previous close and the returned object
            carries ``reference_source`` accordingly.
        band_pct: Band width as a fraction (``0.10`` = ±%10). **PARAMETRE +
            VARSAYILAN damgası** — a narrowed (VBTS) band must be passed
            explicitly.
        equality_tolerance_pct: Equality tolerance used later by
            :func:`flag_limit_sessions` (P5, ±%0.1). Not a band parameter.

    Returns:
        :class:`PriceLimits` with tick-aligned ``tavan``/``taban``.

    Raises:
        ValueError: If ``reference`` is not positive or ``band_pct`` is invalid.
    """
    ref = _as_decimal(reference, field_name="reference")
    if ref <= 0:
        raise ValueError(f"reference must be positive, got {ref}")

    band = _as_decimal(band_pct, field_name="band_pct")
    if band <= 0 or band >= 1:
        raise ValueError(f"band_pct must be in (0, 1), got {band}")

    tolerance = _as_decimal(equality_tolerance_pct, field_name="equality_tolerance_pct")

    upped = ref * (Decimal(1) + band)
    downed = ref * (Decimal(1) - band)
    tavan = round_to_tick(upped)
    taban = round_to_tick(downed)

    notes: list[str] = []
    band_source = "default"
    if band != LIMIT_BAND_PCT_DEFAULT:
        band_source = "caller_supplied"
        notes.append(
            "bant_genisligi cagiran tarafindan verildi; VBTS daraltilmis bant "
            "kaynagi modulde YOK (VERI YOK)"
        )

    return PriceLimits(
        reference=ref,
        band_pct=band,
        tavan=tavan,
        taban=taban,
        reference_source="prev_close (AOF yok: VERI YOK)",
        band_source=band_source,
        rounding=_ROUNDING_MODE,
        equality_tolerance_pct=tolerance,
        notes=tuple(notes),
    )


def compute_price_limits_from_aof(
    symbol: str,
    *,
    as_of: date,
    band_pct: object = LIMIT_BAND_PCT_DEFAULT,
    band_source: str | None = None,
    vbts_symbols: object | None = None,
    aof_fetcher: object | None = None,
) -> PriceLimits:
    """Compute BIST tavan/taban using the **AOF** of the previous session.

    BIST's official reference is the previous session's weighted average price
    (AOF), not its close. Measured gap (THYAO 02.11.2026): close 261.00 vs AOF
    252.992 → using the close shifts the limits by ~3%.

    Flow:

    1. ``as_of`` is normalised to a trading day: if it is a weekend/holiday the
       nearest **earlier** trading day is used and the input is stamped in
       ``as_of_normalized_from`` (never a silent shift).
    2. The reference session = ``previous_trading_day(anchor)`` — strictly
       before the anchor, from the sourced BIST calendar.
    3. Reference value = that session's AOF (``Decimal``). A missing AOF row
       raises :class:`AOFUnavailableError`; falling back to the close is
       **forbidden** (no silent shift).
    4. Tavan/taban are computed by delegating to :func:`compute_price_limits`
       (single calculation path; no second formula).

    Args:
        symbol: BIST ticker (e.g. ``"THYAO"``).
        as_of: The trading day whose limits are wanted (not the reference day).
        band_pct: Band width; ``PARAMETRE+VARSAYILAN`` (±%10).
        band_source: Optional explicit band provenance. When omitted it is
            derived (``"default"`` for the ±%10 default, ``"caller_supplied"``
            otherwise) so the VBTS guard can see it.
        vbts_symbols: Optional VBTS list (or ``KAPProvider.get_vbts_status()``
            mapping). When given, :func:`vbts_guard.assert_band_source_safe` is
            called — a VBTS symbol with an unverified band raises.
        aof_fetcher: Test seam. Callable ``(symbol, start, end) -> (frame,
            diagnostics)``; defaults to
            :func:`borsapy._providers.isyatirim_aof.get_daily_aof`.

    Returns:
        :class:`PriceLimits` with ``reference_source="aof"`` and
        ``reference_date`` set to the reference session.

    Raises:
        AOFUnavailableError: The reference session has no AOF value.
        CalendarCoverageError: ``as_of`` or the reference session is outside the
            sourced calendar coverage.
        BandSourceUnverifiedError: VBTS symbol with an unverified band.
        ValueError: Empty ``symbol`` or invalid ``band_pct``.
    """
    ticker = (symbol or "").strip().upper()
    if not ticker:
        raise ValueError("symbol must be a non-empty string")
    if not isinstance(as_of, date):
        raise ValueError(f"as_of must be a date, got {type(as_of).__name__}")

    # Local imports keep this module dependency-light and immune to import
    # ordering in ``borsapy/__init__`` (no cycle exists, this is defensive).
    from borsapy._providers.bist_calendar import (
        is_trading_day,
        previous_trading_day,
    )

    # --- 1) normalise as_of (weekend/holiday) --------------------------
    as_of_input = as_of
    anchor = as_of
    if not is_trading_day(anchor):
        anchor = previous_trading_day(anchor, include_self=True)

    # --- 2) reference session = previous trading day -------------------
    reference_day = previous_trading_day(anchor)

    # --- 3) AOF of the reference session -------------------------------
    fetcher = aof_fetcher
    if fetcher is None:
        from borsapy._providers.isyatirim_aof import get_daily_aof

        fetcher = get_daily_aof

    frame, _diagnostics = fetcher(ticker, reference_day, reference_day)

    aof_value = _aof_for_day(frame, reference_day)
    if aof_value is None:
        raise AOFUnavailableError(
            ticker,
            reference_day,
            "kaynakta o gun icin HGDG_AOF satiri/degeri yok "
            "(get_daily_aof bos veya aof=None dondu)",
        )

    # --- 3b) band provenance (VBTS guard needs it) ---------------------
    band_decimal = _as_decimal(band_pct, field_name="band_pct")
    source_label = band_source
    if source_label is None:
        source_label = (
            "default"
            if band_decimal == LIMIT_BAND_PCT_DEFAULT
            else "caller_supplied"
        )

    if vbts_symbols is not None:
        from borsapy._providers.vbts_guard import assert_band_source_safe

        assert_band_source_safe(ticker, source_label, vbts_symbols)

    # --- 4) delegate the calculation (single formula) ------------------
    limits = compute_price_limits(aof_value, band_decimal)

    notes = list(limits.notes)
    notes.append(
        f"referans = AOF ({reference_day.isoformat()}); kapanis DEGIL "
        "(BIST kurali)"
    )
    if anchor != as_of_input:
        notes.append(
            f"as_of {as_of_input.isoformat()} islem gunu degil; "
            f"{anchor.isoformat()} gunune normalize edildi"
        )

    return PriceLimits(
        reference=limits.reference,
        band_pct=limits.band_pct,
        tavan=limits.tavan,
        taban=limits.taban,
        reference_source="aof",
        band_source=source_label,
        rounding=limits.rounding,
        equality_tolerance_pct=limits.equality_tolerance_pct,
        notes=tuple(notes),
        reference_date=reference_day,
        as_of_normalized_from=(as_of_input if anchor != as_of_input else None),
    )


def _aof_for_day(frame: object, day: date) -> Decimal | None:
    """Extract the ``aof`` cell for ``day`` from a ``get_daily_aof`` frame.

    ``None`` when the day is absent or its value is missing — the caller turns
    that into :class:`AOFUnavailableError`.
    """
    if frame is None:
        return None

    if isinstance(frame, pd.DataFrame):
        if frame.empty:
            return None
        if day in frame.index:
            value = frame.loc[day, "aof"]
            if isinstance(value, pd.Series):  # duplicate index safety
                value = value.iloc[0]
            if _is_missing(value) or value is None:
                return None
            return value if isinstance(value, Decimal) else _as_decimal(
                value, field_name="aof"
            )
        return None

    # Mapping fallback (``{date: Decimal|None}``), useful for callers/tests.
    if isinstance(frame, dict):
        value = frame.get(day)
        if value is None or _is_missing(value):
            return None
        return value if isinstance(value, Decimal) else _as_decimal(
            value, field_name="aof"
        )

    raise ValueError(
        f"unsupported AOF payload type: {type(frame).__name__}"
    )


def _column(df: pd.DataFrame, *names: str) -> pd.Series | None:
    """Return the first present column among ``names`` (case-insensitive)."""
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for name in names:
        key = name.strip().lower()
        if key in lowered:
            return df[lowered[key]]
    return None


def _is_missing(value: object) -> bool:
    """True for NaN/None/NaT-like values without importing numpy explicitly."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def flag_limit_sessions(
    df: pd.DataFrame,
    *,
    band_pct: object = LIMIT_BAND_PCT_DEFAULT,
    equality_tolerance_pct: object = LIMIT_EQUALITY_TOLERANCE_PCT,
    volume_ratio_threshold: float = LIMIT_VOLUME_RATIO_V0,
    calendar: object | None = None,
    aof_provider: object | None = None,
) -> pd.DataFrame:
    """Flag limit-locked sessions (tavan/taban) in a daily OHLCV frame.

    Conditions (documented, applied in this order):

    1. **Referans yok** — previous close missing/NaN/<=0 → ``limitli=False``,
       ``gerekce="VERİ YOK: referans fiyat yok"``, ``guven=None``.
    2. **KESIN** — the whole session printed a single price at the limit:
       ``High == Low == tavan`` (or ``taban``) within the equality tolerance.
       This is a geometric fact; it uses **no** volume threshold.
    3. **SUPHELI** — the limit was touched and the close finished at it
       (``High == tavan`` and ``Close == tavan``) but the session was not
       single-priced. Reported so analysts can see it; **SUPHELI never enters
       a decision chain** (ERRATUM v2 D2).
    4. **SUPHELI (hacim)** — close at the limit with volume below
       ``volume_ratio_threshold ×`` trailing median volume. The threshold is a
       **v0 parameter (KALİBRE EDİLMEDİ)**, so this can only ever be SUPHELI.
    5. **Hacimsiz gün** — ``Volume`` is 0/NaN. Reported in a dedicated column;
       a single-priced limit session with no volume is still KESIN (a locked
       session genuinely cannot trade) but the reason string says so.

    Args:
        df: Daily OHLCV frame (``Open``/``High``/``Low``/``Close``/``Volume``,
            case-insensitive). Index is preserved.
        band_pct: Band width passed to :func:`compute_price_limits`.
        equality_tolerance_pct: Equality tolerance (P5, ±%0.1 of the limit).
        volume_ratio_threshold: v0 volume ratio (calibration pending).
        calendar: Optional trading-calendar module (W3b). ``None`` (default)
            keeps the historical behaviour **byte-identical**. When supplied,
            every consecutive session pair is checked against the calendar: if
            the previous row is not the calendar's previous trading day, the
            row is stamped ``onceki_seans_takvim_dogrulanmadi`` in ``gerekce``
            and its ``guven`` is capped at ``SUPHELI`` (never KESIN). The
            calendar object must expose ``previous_trading_day(date)``.
        aof_provider: Optional AOF series provider (W3c). ``None`` (default)
            keeps the historical behaviour **byte-identical** — no new column
            appears. When supplied it must expose ``aof_for(day) -> Decimal |
            None`` returning the **reference** AOF for that session (already
            resolved to the previous trading day by the provider). Then the
            limits are computed from that AOF and a ``referans_kaynagi`` column
            is added (``"aof"`` / ``"prev_close"`` / ``None``). A missing AOF
            yields ``limitli=False`` + ``VERİ YOK`` reason + ``guven=None``;
            falling back to the close is **forbidden** (ERRATUM v2 §H.3).

    Returns:
        A copy of ``df`` with added columns ``tavan``, ``taban``, ``limitli``,
        ``yon`` (``tavan``/``taban``/``None``), ``gerekce``, ``guven``
        (``KESIN``/``SUPHELI``/``None``), ``hacimsiz`` and ``referans``.

    Raises:
        ValueError: If a required OHLCV column is missing, or ``calendar`` is
            given without a ``previous_trading_day`` attribute.
    """
    calendar_prev = None
    if calendar is not None:
        calendar_prev = getattr(calendar, "previous_trading_day", None)
        if calendar_prev is None:
            raise ValueError(
                "calendar must expose previous_trading_day(date); got "
                f"{type(calendar).__name__}"
            )

    aof_lookup = None
    if aof_provider is not None:
        aof_lookup = getattr(aof_provider, "aof_for", None)
        if aof_lookup is None:
            raise ValueError(
                "aof_provider must expose aof_for(date); got "
                f"{type(aof_provider).__name__}"
            )
    required = {
        "high": _column(df, "High", "high"),
        "low": _column(df, "Low", "low"),
        "close": _column(df, "Close", "close"),
    }
    missing = [name for name, series in required.items() if series is None]
    if missing:
        raise ValueError(f"flag_limit_sessions requires columns: {missing}")

    volume = _column(df, "Volume", "volume")
    out = df.copy()
    n = len(out)

    tavans: list[Decimal | None] = []
    tabans: list[Decimal | None] = []
    referanslar: list[Decimal | None] = []
    limitli: list[bool] = []
    yon: list[str | None] = []
    gerekce: list[str] = []
    guven: list[str | None] = []
    hacimsiz: list[bool] = []
    referans_kaynagi: list[str | None] = []

    tolerance_pct = _as_decimal(equality_tolerance_pct, field_name="equality_tolerance_pct")
    # v0 hacim eşiği için iz süren medyan (ilk satırlarda pencere dolarsa).
    volume_median = (
        volume.rolling(20, min_periods=5).median() if volume is not None else None
    )

    prev_close: Decimal | None = None
    for i in range(n):
        high = _column_value(required["high"], i)
        low = _column_value(required["low"], i)
        close = _column_value(required["close"], i)
        vol = _column_value(volume, i) if volume is not None else None

        vol_missing = vol is None or _is_missing(vol)
        is_hacimsiz = bool(vol_missing) or (not vol_missing and float(vol) == 0.0)
        hacimsiz.append(is_hacimsiz)

        # --- W3c: AOF-based reference (opt-in) -------------------------
        if aof_lookup is not None:
            row_day = _as_date(out.index[i])
            aof_value: Decimal | None = None
            aof_error: Exception | None = None
            if row_day is not None:
                try:
                    raw_aof = aof_lookup(row_day)
                except Exception as exc:  # takvim/kapsam hatası -> görünür
                    raw_aof = None
                    aof_error = exc
                if raw_aof is not None and not _is_missing(raw_aof):
                    aof_value = (
                        raw_aof
                        if isinstance(raw_aof, Decimal)
                        else _as_decimal(raw_aof, field_name="aof")
                    )

            if aof_value is None or aof_value <= 0:
                # VERİ YOK yolu: kapanışa DÜŞME YOK (ERRATUM v2 §H.3).
                referanslar.append(aof_value)
                tavans.append(None)
                tabans.append(None)
                limitli.append(False)
                yon.append(None)
                guven.append(None)
                referans_kaynagi.append("aof" if aof_value else None)
                day_text = row_day.isoformat() if row_day is not None else "?"
                if aof_error is not None:
                    gerekce.append(
                        f"VERİ YOK: AOF referansı yok ({day_text}) "
                        f"[takvim: {type(aof_error).__name__}]"
                    )
                else:
                    gerekce.append(f"VERİ YOK: AOF referansı yok ({day_text})")
                if not _is_missing(close):
                    prev_close = _as_decimal(close, field_name="close")
                continue

            if any(_is_missing(value) for value in (high, low, close)):
                referanslar.append(aof_value)
                tavans.append(None)
                tabans.append(None)
                limitli.append(False)
                yon.append(None)
                guven.append(None)
                referans_kaynagi.append("aof")
                gerekce.append("VERİ YOK: OHLC eksik")
                prev_close = aof_value
                continue

            limits = compute_price_limits(
                aof_value,
                band_pct,
                equality_tolerance_pct=tolerance_pct,
            )
            referanslar.append(aof_value)
            tavans.append(limits.tavan)
            tabans.append(limits.taban)
            referans_kaynagi.append("aof")
            prev_close = aof_value
            # Paylaşılan sınıflandırma bloğu aşağıda çalışır.
        elif prev_close is None or prev_close <= 0 or any(
            _is_missing(value) for value in (high, low, close)
        ):
            referanslar.append(prev_close)
            tavans.append(None)
            tabans.append(None)
            limitli.append(False)
            yon.append(None)
            guven.append(None)
            referans_kaynagi.append(None)
            gerekce.append("VERİ YOK: referans fiyat yok")
            if not _is_missing(close):
                prev_close = _as_decimal(close, field_name="close")
            continue

        if aof_lookup is None:
            limits = compute_price_limits(
                prev_close,
                band_pct,
                equality_tolerance_pct=tolerance_pct,
            )
            referanslar.append(prev_close)
            tavans.append(limits.tavan)
            tabans.append(limits.taban)
            referans_kaynagi.append("prev_close")
        else:
            # AOF yolu yukarıda limits/tavan/taban/referansı zaten yazdı.
            limits = compute_price_limits(
                referanslar[-1],
                band_pct,
                equality_tolerance_pct=tolerance_pct,
            )

        tol_up = limits.tavan * tolerance_pct
        tol_down = limits.taban * tolerance_pct
        high_d = _as_decimal(high, field_name="high")
        low_d = _as_decimal(low, field_name="low")
        close_d = _as_decimal(close, field_name="close")

        locked_up = (
            abs(high_d - limits.tavan) <= tol_up and abs(low_d - limits.tavan) <= tol_up
        )
        locked_down = (
            abs(low_d - limits.taban) <= tol_down
            and abs(high_d - limits.taban) <= tol_down
        )
        touched_up = abs(high_d - limits.tavan) <= tol_up
        touched_down = abs(low_d - limits.taban) <= tol_down
        closed_up = abs(close_d - limits.tavan) <= tol_up
        closed_down = abs(close_d - limits.taban) <= tol_down

        flag = False
        direction: str | None = None
        reason = ""
        confidence: str | None = None

        if locked_up or locked_down:
            flag = True
            direction = "tavan" if locked_up else "taban"
            confidence = "KESIN"
            if is_hacimsiz:
                reason = (
                    f"tek fiyat seansi ({direction}) + hacimsiz gun (Volume=0/NaN)"
                )
            else:
                reason = (
                    f"tek fiyat seansi: High=Low={direction} "
                    f"(ref={prev_close})"
                )
        elif (touched_up and closed_up) or (touched_down and closed_down):
            flag = True
            direction = "tavan" if touched_up and closed_up else "taban"
            confidence = "SUPHELI"
            reason = f"kapanis {direction}da, tek fiyat degil"
            if (
                volume_median is not None
                and not is_hacimsiz
                and not _is_missing(volume_median.iloc[i])
                and float(volume_median.iloc[i]) > 0
                and float(vol) < volume_ratio_threshold * float(volume_median.iloc[i])
            ):
                reason += (
                    f"; hacim esigin altinda (v0 oran={volume_ratio_threshold}, "
                    "KALIBRE EDILMEDI)"
                )
        elif is_hacimsiz:
            reason = "hacimsiz gun; limit temasi yok"
        else:
            reason = "limit temasi yok"

        # --- W3b: calendar-aware previous-session check (opt-in) ---------
        if calendar_prev is not None:
            row_day = _as_date(out.index[i])
            prev_day = _as_date(out.index[i - 1]) if i > 0 else None
            if row_day is not None and prev_day is not None:
                try:
                    expected = calendar_prev(row_day)
                except Exception as exc:  # coverage/takvim hatası -> görünür
                    expected = None
                    reason = (
                        f"{reason}; onceki_seans_takvim_dogrulanmadi "
                        f"(takvim: {type(exc).__name__})"
                    )
                if expected is not None and expected != prev_day:
                    reason = (
                        f"{reason}; onceki_seans_takvim_dogrulanmadi "
                        f"(beklenen={expected.isoformat()}, "
                        f"seride={prev_day.isoformat()})"
                    )
                    # GAP: KESIN asla korunmaz, SUPHELI'ye sabitlenir.
                    if confidence == "KESIN":
                        confidence = "SUPHELI"

        limitli.append(flag)
        yon.append(direction)
        gerekce.append(reason)
        guven.append(confidence)
        prev_close = close_d

    out["referans"] = referanslar
    out["tavan"] = tavans
    out["taban"] = tabans
    out["limitli"] = limitli
    out["yon"] = yon
    out["gerekce"] = gerekce
    out["guven"] = guven
    out["hacimsiz"] = hacimsiz
    if aof_lookup is not None:
        # W3c: yeni sütun YALNIZ opt-in yolda eklenir (byte-identical korunur).
        out["referans_kaynagi"] = referans_kaynagi
    return out


def _as_date(value: object) -> date | None:
    """Coerce an index label to :class:`datetime.date` (``None`` when not a date)."""
    if isinstance(value, date) and not isinstance(value, type(None)):
        # ``datetime`` is a ``date`` subclass; keep only the calendar day.
        return date(value.year, value.month, value.day)
    try:
        converted = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(converted):
        return None
    return date(converted.year, converted.month, converted.day)


def _column_value(series: pd.Series | None, position: int) -> object:
    """Positional cell value (``None`` when the frame has no volume column)."""
    if series is None:
        return None
    return series.iloc[position]


def adtv(
    df: pd.DataFrame,
    window: int = 20,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    """Average daily traded value in TL (ADTV).

    ``ADTV = mean(window)`` of ``Volume × typical_price``, where the typical
    price is ``(High + Low + Close) / 3``. This is a documented definition, not
    an exchange-published series.

    Args:
        df: Daily OHLCV frame (case-insensitive column names).
        window: Rolling window length in sessions (default 20).
        min_periods: Minimum observations required; defaults to ``window`` so a
            partially filled window yields ``NaN`` instead of a thin average.

    Returns:
        Series of ADTV in TL, aligned to ``df.index``.

    Raises:
        ValueError: If a required column is missing or ``window`` < 1.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    high = _column(df, "High", "high")
    low = _column(df, "Low", "low")
    close = _column(df, "Close", "close")
    volume = _column(df, "Volume", "volume")
    if any(series is None for series in (high, low, close, volume)):
        raise ValueError("adtv requires High, Low, Close and Volume columns")

    typical = (high.astype(float) + low.astype(float) + close.astype(float)) / 3.0
    traded_value = volume.astype(float) * typical
    effective_min = window if min_periods is None else min_periods
    return traded_value.rolling(window, min_periods=effective_min).mean()


def adtv_diagnostics(df: pd.DataFrame) -> dict[str, int]:
    """Count suspicious sessions used by :func:`adtv` (NaN/zero buckets).

    Args:
        df: Daily OHLCV frame.

    Returns:
        ``{"rows", "volume_nan", "volume_zero", "price_nan"}`` counts.
    """
    volume = _column(df, "Volume", "volume")
    high = _column(df, "High", "high")
    low = _column(df, "Low", "low")
    close = _column(df, "Close", "close")
    if any(series is None for series in (volume, high, low, close)):
        raise ValueError("adtv_diagnostics requires High, Low, Close and Volume")

    volume_nan = int(volume.isna().sum())
    volume_zero = int((volume.fillna(0) == 0).sum())
    price_nan = int(
        (high.isna() | low.isna() | close.isna()).sum()
    )
    return {
        "rows": int(len(df)),
        "volume_nan": volume_nan,
        "volume_zero": volume_zero,
        "price_nan": price_nan,
    }
