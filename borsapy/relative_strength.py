"""Endeks göreli güç (Relative Strength, RS) vs bir endeks.

Wyckoff Rehberi §13: "Hisse/BIST 100 oranını çizin. Oran yükseliyorsa hisse
endeksten güçlüdür." Bu modül o oranı **düzeltilmiş** hisse fiyatıyla üretir ve
künyesiyle birlikte döner.

Not: Float katmanı — gösterge fiyatı, muhasebe değeri değil.

Düzeltme eki §6 hatırlatması: "Hisse/BIST 100 oranının yükselmesi endekse göre
güç anlamına gelebilir; mutlak fiyat artışı ya da reel getiri garantisi
değildir." Bu modül yalnız oranı ve eşik-tabanlı olayları üretir; yorum
(bist-research) katmanındadır.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import time as dt_time
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "RSEvent",
    "relative_strength_series",
    "relative_strength_events",
]

#: Göreli güç trendi için varsayılan geri-bakış barı. **PARAMETRE — KALİBRE
#: EDİLMEDİ** (EK-B.2).
DEFAULT_RS_LOOKBACK: int = 20

#: RS "sıçrama" olayı için varsayılan z eşiği. **PARAMETRE — KALİBRE
#: EDİLMEDİ** (EK-B.2).
DEFAULT_RS_Z_THRESHOLD: float = 1.0


# ---------------------------------------------------------------------------
# Bar kapanışı ("oluşmakta olan bar") yardımcıları
# ---------------------------------------------------------------------------
#
# Borsapy'de BIST **seans kapanış saati** kaynağı YOKTUR (``bist_calendar``
# yalnız tatil/yarım gün bilir). Bu yüzden kapanış saati **uydurulmaz**:
# çağıran taraf ``session_close`` ile verir, verilmezse son barın durumu
# ``None`` (VERİ YOK) olarak raporlanır.


def _series_tz(index: Any) -> Any:
    """Serinin bar dilimini (tz) döndür; naive ise ``None``."""
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        return index.tz
    return None


def _label_to_bar_tz(label: Any, bar_tz: Any) -> pd.Timestamp:
    """İşlenmiş seri etiketini (UTC-naive) bar dilimine geri çevir."""
    ts = pd.Timestamp(label)
    if bar_tz is None:
        return ts
    if ts.tzinfo is None:
        return ts.tz_localize("UTC").tz_convert(bar_tz)
    return ts.tz_convert(bar_tz)


def _bar_tz_to_label(ts: pd.Timestamp, bar_tz: Any) -> pd.Timestamp:
    """Bar dilimindeki anı seri index çerçevesine (UTC-naive) çevir."""
    if bar_tz is None:
        return ts
    return ts.tz_convert("UTC").tz_localize(None)


def _as_of_ceiling(as_of: Any, bar_tz: Any) -> pd.Timestamp:
    """``as_of`` kesim sınırını seri index çerçevesinde hesapla (dahil).

    Saat bilgisi olmayan bir ``as_of`` (``date`` veya gece yarısı) **o günün
    tamamını** kapsar; saat verilmişse tam o an (bar diliminde) sınırdır.
    """
    ts = pd.Timestamp(as_of)
    date_only = (ts.hour, ts.minute, ts.second, ts.microsecond) == (0, 0, 0, 0)
    if ts.tzinfo is None and bar_tz is not None:
        ts = ts.tz_localize(bar_tz)
    if date_only:
        ts = ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return _bar_tz_to_label(ts, bar_tz)


def _now_in_bar_tz(now: Any, bar_tz: Any) -> pd.Timestamp:
    """``now`` (enjekte edilebilir saat) değerini bar dilimine çevir."""
    if now is None:
        return pd.Timestamp.now(tz=bar_tz) if bar_tz is not None else pd.Timestamp.now()
    ts = pd.Timestamp(now)
    if bar_tz is None:
        if ts.tzinfo is None:
            return ts
        return ts.tz_convert("UTC").tz_localize(None)
    if ts.tzinfo is None:
        return ts.tz_localize(bar_tz)
    return ts.tz_convert(bar_tz)


# ---------------------------------------------------------------------------
# Hizalama yardımcıları
# ---------------------------------------------------------------------------


def _as_close_series(data: pd.DataFrame | pd.Series, name: str) -> pd.Series:
    """DataFrame ise ``Close`` sütununu, Series ise kendisini float döndür.

    Args:
        data: Ham veri.
        name: Hata mesajında kullanılacak taraf adı (``stock`` / ``index``).

    Returns:
        ``float64`` kapanış serisi.

    Raises:
        ValueError: DataFrame'de ``Close`` yoksa veya girdi Series/DataFrame
            değilse.
    """
    if isinstance(data, pd.DataFrame):
        if "Close" not in data.columns:
            raise ValueError(
                f"{name} frame has no 'Close' column; got {list(data.columns)}"
            )
        series = data["Close"]
    elif isinstance(data, pd.Series):
        series = data
    else:
        raise ValueError(
            f"{name} must be a DataFrame or Series; got {type(data).__name__}"
        )
    return series.astype("float64")


def _coerce_datetime_index(series: pd.Series) -> pd.Series:
    """DatetimeIndex'i tz'den bağımsız, karşılaştırılabilir hale getir.

    Her iki seri de tz-aware ise UTC'ye (naive) çevrilir — aynı **an** aynı
    etikete düşer. Naive seriler olduğu gibi bırakılır (çağıran taraf karışık
    durumu tarih bazında çözer).
    """
    if isinstance(series.index, pd.DatetimeIndex) and series.index.tz is not None:
        converted = series.index.tz_convert("UTC").tz_localize(None)
        out = series.copy()
        out.index = converted
        return out
    return series


def _sha256_series(series: pd.Series) -> str:
    """Serinin deterministik sha256'sı (mühür için).

    Ad + index etiketleri + float64 değer baytları hash'lenir; aynı girdi iki
    koşumda aynı digest'i verir.
    """
    values = np.asarray(series.to_numpy(dtype="float64"), dtype="float64")
    h = hashlib.sha256()
    h.update(str(series.name).encode("utf-8"))
    h.update("|".join(str(i) for i in series.index).encode("utf-8"))
    h.update(str(values.shape).encode("utf-8"))
    h.update(np.ascontiguousarray(values).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# REQ-001 — göreli güç serisi
# ---------------------------------------------------------------------------


def relative_strength_series(
    stock: pd.DataFrame | pd.Series,
    index: pd.DataFrame | pd.Series,
    *,
    normalize: bool = True,
    base: int = 0,
    lookback: int = DEFAULT_RS_LOOKBACK,
    as_of: Any = None,
    closed_bars_only: bool = False,
    session_close: dt_time | None = None,
    now: Any = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Hisse/endeks oranı (göreli güç) serisini künyesiyle üret.

    **Tek hesap yolu:** ``rs_t = stock_close_t / index_close_t`` — satır bazlı,
    index hizalı. Hisse tarafı çağıran tarafta **düzeltilmiş** olmalıdır
    (:meth:`borsapy.Ticker.relative_strength` bunu garanti eder, künyede
    ``stock_adjusted=True`` damgasıyla); düzeltilmemiş seriyle RS temettü/bedelsiz
    günlerinde sahte zayıflık gösterir.

    **Hizalama:** iki serinin **kesişimi** kullanılır. Yalnız bir tarafta olan
    günler **atlanır** (``NaN`` üretilmez) ve künyede sayılır
    (``n_dropped_stock`` / ``n_dropped_index``). Boş kesişim ``ValueError``'dır.

    **tz:** iki seri de tz-aware ise aynı ana göre eşleştirilir (UTC). tz'ler
    karışıksa (biri aware, biri naive) takvim günü bazında hizalanır ve
    künyeye uyarı eklenir.

    Not: Float katmanı (gösterge fiyatı; muhasebe değeri değil).

    Args:
        stock: Hisse kapanışları (DataFrame ise ``Close``).
        index: Endeks kapanışları (DataFrame ise ``Close``).
        normalize: ``True`` (varsayılan) ise oran ``base`` gününe göre 100'e
            ölçeklenir (100 = endekse eşit performans). ``False`` ise ham oran.
        base: Normalizasyon bazının **0-tabanlı konumu** (kesişim serisi
            içinde). Varsayılan 0 = ilk geçerli gün.
        lookback: Trendin eğim penceresi (bar). **PARAMETRE — KALİBRE
            EDİLMEDİ.**

        as_of: Kesim (dahil). Verilirse yalnız bu ana kadarki barlar kullanılır
            ve atlananlar ``n_dropped_after_as_of`` ile sayılır. Saat
            verilmemiş bir ``as_of`` (ör. ``date(2026, 11, 14)``) **o günün
            tamamını** kapsar. Kesim gün içi tekrarlanabilirliği sağlar:
            aynı ``as_of`` iki koşuda aynı seriyi verir.
        closed_bars_only: ``True`` ise **oluşmakta olan (kapanmamış) son bar
            atılır**. Barın kapanıp kapanmadığı ancak ``session_close``
            biliniyorsa söylenebilir: son bar bugüne ait ve ``session_close``
            verilmemişse **``ValueError``** (tahmin yasak, VERİ YOK).
        session_close: BIST seans kapanış saati (bar diliminde yorumlanır).
            **PARAMETRE** — borsapy'de kaynaklı kapanış saati YOKTUR, çağıran
            verir. Verilmezse son barın durumu ``last_bar_partial=None``
            (VERİ YOK) olarak raporlanır.
        now: "Şimdi" (test enjeksiyonu). ``None`` ise gerçek saat kullanılır.
            Bar dilimine çevrilir; naive verilirse bar diliminde kabul edilir.

    Returns:
        ``(rs, meta)``. ``rs``: index'i kesişim olan float seri (adı
        ``RS``); ``meta``: aşağıdaki alanlarla künye sözlüğü —

        ``source``, ``align``, ``n_common``, ``n_dropped_stock``,
        ``n_dropped_index``, ``n_dropped_after_as_of``, ``bars_used``,
        ``base_date``, ``base_value``, ``last_value``, ``change_pct``,
        ``trend`` (``guclu``/``zayif``/``yatay``), ``trend_slope``,
        ``trend_rule``, ``lookback``, ``trend_horizon_bars``,
        ``change_pct_horizon_bars``, ``as_of``, ``closed_bars_only``,
        ``session_close``, ``last_bar_date``, ``last_bar_partial``
        (``True``/``False``/``None``=VERİ YOK), ``dropped_forming_bar``,
        ``normalized``, ``series_sha256``, ``warnings``.

    Raises:
        ValueError: ``Close`` eksik, kesişim boş, ``lookback < 2``,
            ``base`` aralık dışı, baz değeri geçersiz (``<= 0`` / ``NaN``),
            ``as_of`` sonrası < 2 bar kalması veya ``closed_bars_only=True``
            iken son barın kapanmış olduğunun bilinememesi (VERİ YOK).
    """
    stock_close = _as_close_series(stock, "stock")
    index_close = _as_close_series(index, "index")
    warnings: list[str] = []

    both_aware = (
        isinstance(stock_close.index, pd.DatetimeIndex)
        and stock_close.index.tz is not None
        and isinstance(index_close.index, pd.DatetimeIndex)
        and index_close.index.tz is not None
    )
    any_aware = (
        isinstance(stock_close.index, pd.DatetimeIndex)
        and stock_close.index.tz is not None
    ) or (
        isinstance(index_close.index, pd.DatetimeIndex)
        and index_close.index.tz is not None
    )
    mixed_tz = any_aware and not both_aware

    stock_s = _coerce_datetime_index(stock_close)
    index_s = _coerce_datetime_index(index_close)

    if mixed_tz:
        # Bir taraf tz-aware, diğeri naive: aynı an karşılaştırılamaz →
        # takvim günü bazında hizala (dürüst geri düşüş, uyarıyla).
        stock_s = stock_s.copy()
        index_s = index_s.copy()
        stock_s.index = pd.DatetimeIndex(stock_s.index).normalize()
        index_s.index = pd.DatetimeIndex(index_s.index).normalize()
        warnings.append(
            "timezone mismatch between stock and index series; aligned on "
            "calendar date (intraday bars may collapse)"
        )

    common = stock_s.index.intersection(index_s.index)
    n_dropped_stock = int(len(stock_s.index) - len(common))
    n_dropped_index = int(len(index_s.index) - len(common))

    if len(common) == 0:
        raise ValueError(
            "relative_strength_series: stock and index series do not share any "
            f"timestamp (stock={len(stock_s.index)} bars, "
            f"index={len(index_s.index)} bars) — a single-sided series is not "
            "a relative strength"
        )

    stk = stock_s.reindex(common)
    idx = index_s.reindex(common)

    # Payda 0 / NaN → oran tanımsız; bu satırlar atlanır (uydurma yok).
    invalid = idx.isna() | (idx == 0) | stk.isna()
    if invalid.any():
        warnings.append(
            f"{int(invalid.sum())} common bar(s) had zero/NaN index or NaN "
            "stock close and were dropped from the ratio"
        )
        stk = stk[~invalid]
        idx = idx[~invalid]
        if len(stk) == 0:
            raise ValueError(
                "relative_strength_series: no valid bar after removing "
                "zero/NaN index or stock closes"
            )

    ratio = (stk / idx).astype("float64")
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()

    if len(ratio) == 0:
        raise ValueError("relative_strength_series: ratio is empty after cleaning")

    if lookback < 2:
        raise ValueError(f"lookback must be >= 2; got {lookback}")

    # --- bar dilimi: kesim/kapanış karşılaştırmaları için ---
    bar_tz = _series_tz(stock_close.index) or _series_tz(index_close.index)

    # --- as_of kesimi (dahil) — gün içi tekrarlanabilirlik ---
    n_dropped_after_as_of = 0
    if as_of is not None:
        ceiling = _as_of_ceiling(as_of, bar_tz)
        keep = ratio.index <= ceiling
        n_dropped_after_as_of = int((~keep).sum())
        if n_dropped_after_as_of:
            ratio = ratio[keep]
            warnings.append(
                f"{n_dropped_after_as_of} bar(s) after as_of={as_of!r} were "
                "dropped (cutoff is inclusive)"
            )
        if len(ratio) < 2:
            raise ValueError(
                f"relative_strength_series: as_of={as_of!r} leaves "
                f"{len(ratio)} bar(s); at least 2 are required"
            )

    # --- son bar oluşmakta mı? (kapanış saati UYDURULMAZ) ---
    last_bar_date = _label_to_bar_tz(ratio.index[-1], bar_tz)
    now_local = _now_in_bar_tz(now, bar_tz)
    last_bar_partial: bool | None
    if last_bar_date.date() != now_local.date():
        # Son bar önceki seansa ait → kapanmış (saat bilgisi gerekmez).
        last_bar_partial = False
    elif session_close is None:
        last_bar_partial = None
        warnings.append(
            "last bar belongs to today and no session_close was supplied — it may "
            "be a forming (unclosed) bar: trend/change_pct can change intraday "
            "(VERİ YOK: kapanış saati bilinmiyor); pass as_of= or "
            "closed_bars_only=True with session_close="
        )
    else:
        last_bar_partial = now_local.time() < session_close

    dropped_forming_bar = False
    if closed_bars_only:
        if last_bar_partial is None:
            raise ValueError(
                "closed_bars_only=True but the last bar belongs to today and "
                "session_close is unknown — refusing to guess whether the bar is "
                "closed (VERİ YOK); pass session_close= or as_of="
            )
        if last_bar_partial:
            ratio = ratio.iloc[:-1]
            dropped_forming_bar = True
            warnings.append(
                "forming (unclosed) last bar dropped per closed_bars_only=True"
            )
            if len(ratio) < 2:
                raise ValueError(
                    "relative_strength_series: dropping the forming bar leaves "
                    f"{len(ratio)} bar(s); at least 2 are required"
                )

    # --- normalizasyon (baz günü 100) ---
    if normalize:
        if base < 0 or base >= len(ratio):
            raise ValueError(
                f"base position {base} is out of range for {len(ratio)} common bars"
            )
        base_value = float(ratio.iloc[base])
        if not np.isfinite(base_value) or base_value <= 0:
            raise ValueError(
                f"base value at position {base} is invalid ({base_value}); "
                "cannot normalize"
            )
        out = ratio / base_value * 100.0
    else:
        base = 0 if base is None else base
        if base < 0 or base >= len(ratio):
            raise ValueError(
                f"base position {base} is out of range for {len(ratio)} common bars"
            )
        base_value = float(ratio.iloc[base])
        out = ratio.copy()

    out.name = "RS"

    # --- trend: son `lookback` barın eğim işareti (EK-B.3) ---
    window = out.iloc[-lookback:] if len(out) >= lookback else out
    slope = float("nan")
    trend = "yatay"
    if len(window) >= 2:
        x = np.arange(len(window), dtype="float64")
        y = window.to_numpy(dtype="float64")
        if np.isfinite(y).all():
            slope = float(np.polyfit(x, y, 1)[0])
            if slope > 0:
                trend = "guclu"
            elif slope < 0:
                trend = "zayif"
            else:
                trend = "yatay"
    if not np.isfinite(slope):
        warnings.append(
            "trend undetermined (non-finite values in slope window); set to 'yatay'"
        )

    # Değişim HER ZAMAN ham oran bazında raporlanır (normalizasyon lineer
    # ölçektir; yüzde değişim değişmez). Böylece base_value / last_value /
    # change_pct aynı bazdadır — karışık baz yok.
    ratio_base_value = float(ratio.iloc[base])
    ratio_last_value = float(ratio.iloc[-1])
    last_value = ratio_last_value
    change_pct = (
        (ratio_last_value / ratio_base_value - 1.0) * 100.0
        if ratio_base_value
        else float("nan")
    )

    meta: dict[str, Any] = {
        "source": "stock_close / index_close (ratio, index-aligned)",
        "align": "intersection",
        "n_common": int(len(out)),
        "n_dropped_stock": n_dropped_stock,
        "n_dropped_index": n_dropped_index,
        "n_dropped_after_as_of": int(n_dropped_after_as_of),
        "bars_used": int(len(out)),
        "base_date": str(pd.Timestamp(out.index[base]).date()),
        "base_value": ratio_base_value,
        "last_value": last_value,
        "value_basis": "raw ratio (normalization is a linear scale; change_pct is scale-invariant)",
        "change_pct": change_pct,
        "trend": trend,
        "trend_slope": slope,
        "trend_rule": (
            "sign of OLS slope over the last `lookback` bars (no deadband; "
            "PARAMETRE — KALİBRE EDİLMEDİ)"
        ),
        "lookback": int(lookback),
        "trend_horizon_bars": int(min(lookback, len(out))),
        "change_pct_horizon_bars": int(len(out)),
        "as_of": None if as_of is None else str(as_of),
        "closed_bars_only": bool(closed_bars_only),
        "session_close": (
            None if session_close is None else session_close.strftime("%H:%M:%S")
        ),
        "last_bar_date": str(_label_to_bar_tz(out.index[-1], bar_tz).date()),
        "last_bar_partial": last_bar_partial,
        "dropped_forming_bar": bool(dropped_forming_bar),
        "normalized": bool(normalize),
        "series_sha256": _sha256_series(out),
        "warnings": warnings,
    }
    return out, meta


# ---------------------------------------------------------------------------
# REQ-003 — Wyckoff bağlamı: RS olayları
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RSEvent:
    """Göreli güç serisinde tek bir yapısal olay (donmuş, kalibre edilmemiş).

    Attributes:
        index: Olayın gerçekleştiği bar (``pd.Timestamp``).
        kind: ``"yeni_yuksek"`` | ``"yeni_dusuk"`` | ``"sicrama"`` |
            ``"kirilma"``.
        value: O bardaki RS değeri.
        z_score: ``(value − kayan ortalama) / kayan std`` (``sicrama`` için
            eşik anlamlıdır; diğerlerinde de raporlanır).
        evidence: Olayı doğuran sayısal kanıt (kural adı + değer).
        kalibre_edilmedi: Eşiklerin kalibre edilmediği damgası (her zaman
            ``True``).
    """

    index: pd.Timestamp
    kind: str
    value: float
    z_score: float
    evidence: tuple[tuple[str, float], ...] = ()
    kalibre_edilmedi: bool = field(default=True)


def relative_strength_events(
    rs: pd.Series,
    *,
    lookback: int = DEFAULT_RS_LOOKBACK,
    z_threshold: float = DEFAULT_RS_Z_THRESHOLD,
) -> tuple[RSEvent, ...]:
    """RS serisinden Wyckoff bağlamı olaylarını üret (uydurma yok).

    Dört kural (hepsi **PARAMETRE — KALİBRE EDİLMEDİ**):

    * ``yeni_yuksek``: değer, o bar dahil son ``lookback`` barın en yükseği.
    * ``yeni_dusuk``: değer, o bar dahil son ``lookback`` barın en düşüğü.
    * ``sicrama``: ``|z_score| >= z_threshold``; ``z`` = değerin o bar dahil
      son ``lookback`` barın kayan ortalamasından sapması / kayan std.
    * ``kirilma``: değer, **önceki** ``lookback`` barın (o bar hariç) tepesini
      (yukarı) veya dibini (aşağı) aşar.

    Bir barda birden çok koşul sağlanabilir; **hiçbiri bastırılmaz** — her
    sağlanan koşul ayrı :class:`RSEvent` üretir (dürüstlük: ``sicrama``
    susturulmaz). Koşul sağlanmayan bar için olay **üretilmez**.

    Args:
        rs: RS serisi (ör. :func:`relative_strength_series` çıktısı).
        lookback: Pencere (bar). **PARAMETRE — KALİBRE EDİLMEDİ.**
        z_threshold: ``sicrama`` eşiği. **PARAMETRE — KALİBRE EDİLMEDİ.**

    Returns:
        Bar sırasına göre sıralı, değişmez (immutable) olay demeti. Hiçbir
        koşul sağlanmazsa boş demet.

    Raises:
        ValueError: ``lookback < 2`` veya ``z_threshold`` sonlu değil/``<= 0``.
    """
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2; got {lookback}")
    if not np.isfinite(z_threshold) or z_threshold <= 0:
        raise ValueError(
            f"z_threshold must be a finite value > 0; got {z_threshold!r}"
        )

    series = rs.astype("float64")
    if series.empty:
        return ()

    roll_max = series.rolling(window=lookback, min_periods=lookback).max()
    roll_min = series.rolling(window=lookback, min_periods=lookback).min()
    roll_mean = series.rolling(window=lookback, min_periods=lookback).mean()
    roll_std = series.rolling(window=lookback, min_periods=lookback).std(ddof=0)
    z = (series - roll_mean) / roll_std.replace(0.0, np.nan)

    # Önceki pencere (o bar hariç) → kırılma karşılaştırması.
    prev_max = series.shift(1).rolling(window=lookback, min_periods=lookback).max()
    prev_min = series.shift(1).rolling(window=lookback, min_periods=lookback).min()

    events: list[RSEvent] = []
    for ts, value in series.items():
        if not np.isfinite(value):
            continue
        z_val = float(z.loc[ts]) if pd.notna(z.loc[ts]) else float("nan")
        evidence: list[tuple[str, float]] = [
            ("lookback", float(lookback)),
            ("z_threshold", float(z_threshold)),
        ]

        rmax = roll_max.loc[ts]
        rmin = roll_min.loc[ts]
        pmax = prev_max.loc[ts]
        pmin = prev_min.loc[ts]

        if pd.notna(rmax) and value >= float(rmax) - 1e-12:
            events.append(
                RSEvent(
                    index=pd.Timestamp(ts),
                    kind="yeni_yuksek",
                    value=float(value),
                    z_score=z_val,
                    evidence=tuple(evidence + [("rolling_max", float(rmax))]),
                )
            )
        if pd.notna(rmin) and value <= float(rmin) + 1e-12:
            events.append(
                RSEvent(
                    index=pd.Timestamp(ts),
                    kind="yeni_dusuk",
                    value=float(value),
                    z_score=z_val,
                    evidence=tuple(evidence + [("rolling_min", float(rmin))]),
                )
            )
        if (pd.notna(pmax) and value > float(pmax) + 1e-12) or (
            pd.notna(pmin) and value < float(pmin) - 1e-12
        ):
            above = pd.notna(pmax) and value > float(pmax) + 1e-12
            events.append(
                RSEvent(
                    index=pd.Timestamp(ts),
                    kind="kirilma",
                    value=float(value),
                    z_score=z_val,
                    evidence=tuple(
                        evidence
                        + [
                            ("direction", 1.0 if above else -1.0),
                            ("prev_extreme", float(pmax if above else pmin)),
                        ]
                    ),
                )
            )
        if pd.notna(z_val) and abs(z_val) >= z_threshold:
            events.append(
                RSEvent(
                    index=pd.Timestamp(ts),
                    kind="sicrama",
                    value=float(value),
                    z_score=z_val,
                    evidence=tuple(
                        evidence + [("rolling_mean", float(roll_mean.loc[ts]))]
                    ),
                )
            )

    return tuple(events)
