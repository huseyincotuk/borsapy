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
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

import pandas as pd

__all__ = [
    "PRICE_TICK_TABLE",
    "PRICE_TICK_TABLE_SOURCE",
    "LIMIT_BAND_PCT_DEFAULT",
    "LIMIT_EQUALITY_TOLERANCE_PCT",
    "LIMIT_VOLUME_RATIO_V0",
    "PriceLimits",
    "adtv",
    "adtv_diagnostics",
    "compute_price_limits",
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

    def as_dict(self) -> dict[str, object]:
        """Plain-dict view (Decimal values kept as Decimal)."""
        return {
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

    Returns:
        A copy of ``df`` with added columns ``tavan``, ``taban``, ``limitli``,
        ``yon`` (``tavan``/``taban``/``None``), ``gerekce``, ``guven``
        (``KESIN``/``SUPHELI``/``None``), ``hacimsiz`` and ``referans``.

    Raises:
        ValueError: If a required OHLCV column is missing.
    """
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

        if prev_close is None or prev_close <= 0 or any(
            _is_missing(value) for value in (high, low, close)
        ):
            referanslar.append(prev_close)
            tavans.append(None)
            tabans.append(None)
            limitli.append(False)
            yon.append(None)
            guven.append(None)
            gerekce.append("VERİ YOK: referans fiyat yok")
            if not _is_missing(close):
                prev_close = _as_decimal(close, field_name="close")
            continue

        limits = compute_price_limits(
            prev_close,
            band_pct,
            equality_tolerance_pct=tolerance_pct,
        )
        referanslar.append(prev_close)
        tavans.append(limits.tavan)
        tabans.append(limits.taban)

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
    return out


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
