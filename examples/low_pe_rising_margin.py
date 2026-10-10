"""
Düşük F/K ve Yükselen Kar Marjı Tarama Örneği
==============================================

Bu örnek, borsapy kullanarak aşağıdaki kriterlere uyan hisseleri bulur:
1. Fiyat/Kazanç (F/K) oranı 8'in altında
2. Son 3 çeyrek net kar marjı yükselme eğiliminde

Kullanım:
    python examples/low_pe_rising_margin.py

    # Sadece XU030 endeksinde ara
    python -c "from examples.low_pe_rising_margin import screen_low_pe_rising_margin; screen_low_pe_rising_margin(index='XU030')"

Gereksinimler:
    pip install borsapy pandas

Not:
    - Bankalar ve finansal kuruluşlar için mali tablo verisi farklı formatta
      olduğundan (UFRS) bu taramaya dahil edilmezler.
    - Tarama yaklaşık 60 hisse için 2-3 dakika sürebilir.
"""

import pandas as pd

import borsapy as bp

# Banka ve finans sektörü hisseleri (UFRS formatı kullanırlar)
BANK_SYMBOLS = {
    "AKBNK", "GARAN", "ISCTR", "VAKBN", "YKBNK", "HALKB", "SKBNK",
    "TSKB", "ALBRK", "QNBFB", "ICBCT", "KLNMA", "TEKFK", "SEKFK",
    "TURSG", "ANSGR", "AKGRT", "ANHYT", "AGESA", "ISFIN", "GARFA",
    "VAKFA", "ULUFA", "LIDFA", "GLCVY",
}


def calculate_net_margin(income_stmt: pd.DataFrame) -> pd.Series:
    """
    Gelir tablosundan net kar marjını hesapla.

    Net Kar Marjı = (Net Kar / Satış Gelirleri) * 100

    Args:
        income_stmt: Çeyreklik gelir tablosu DataFrame'i
                    (satır isimleri index'te, sütunlar çeyrekler)

    Returns:
        Net kar marjı serisi (%)
    """
    # Türkçe satır isimleri (İş Yatırım API)
    revenue_keywords = ["Satış Gelirleri", "Hasılat", "Net Satışlar"]
    net_income_keywords = [
        "Ana Ortaklık Payları",  # THYAO gibi şirketler
        "SÜRDÜRÜLEN FAALİYETLER DÖNEM KARI",
        "Dönem Net Kar",
        "Net Dönem Karı",
    ]

    # Index'te arama yap (satır isimleri index'te)
    index_list = income_stmt.index.tolist()

    revenue_idx = None
    net_income_idx = None

    # Satış gelirlerini bul
    for keyword in revenue_keywords:
        for idx in index_list:
            if keyword.lower() in str(idx).lower():
                revenue_idx = idx
                break
        if revenue_idx:
            break

    # Net karı bul
    for keyword in net_income_keywords:
        for idx in index_list:
            if keyword.lower() in str(idx).lower():
                net_income_idx = idx
                break
        if net_income_idx:
            break

    if revenue_idx is None or net_income_idx is None:
        return pd.Series(dtype=float)

    # Çeyrek sütunlarını bul (örn: 2024Q3, 2024Q2, ...)
    quarter_cols = [col for col in income_stmt.columns if "Q" in str(col)]

    margins = {}
    for col in quarter_cols:
        try:
            revenue = float(income_stmt.loc[revenue_idx, col])
            net_income = float(income_stmt.loc[net_income_idx, col])
            if revenue != 0 and pd.notna(revenue) and pd.notna(net_income):
                margins[col] = (net_income / revenue) * 100
        except (ValueError, TypeError, KeyError):
            continue

    return pd.Series(margins)


def is_margin_increasing(margins: pd.Series, last_n: int = 3) -> bool:
    """
    Son n çeyrekte kar marjının yükselme eğiliminde olup olmadığını kontrol et.

    Args:
        margins: Kar marjı serisi (en yeni → en eski sıralı)
        last_n: Kontrol edilecek çeyrek sayısı

    Returns:
        True ise kar marjı yükseliyor
    """
    if len(margins) < last_n:
        return False

    # Son n çeyreği al (kronolojik sırala: eski → yeni)
    recent = margins.head(last_n).sort_index()

    # Her çeyrek bir öncekinden büyük mü?
    values = recent.values
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            return False

    return True


def screen_low_pe_rising_margin(
    pe_max: float = 8.0,
    quarters: int = 3,
    index: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Düşük F/K ve yükselen kar marjına sahip hisseleri tara.

    Args:
        pe_max: Maksimum F/K oranı (varsayılan: 8)
        quarters: Kontrol edilecek çeyrek sayısı (varsayılan: 3)
        index: Endeks filtresi (örn: "XU030", "XU100")
        verbose: Detaylı çıktı göster

    Returns:
        Kriterlere uyan hisseler DataFrame'i
    """
    if verbose:
        print("📊 Tarama kriterleri:")
        print(f"   - F/K < {pe_max}")
        print(f"   - Son {quarters} çeyrekte net kar marjı yükseliyor")
        if index:
            print(f"   - Endeks: {index}")
        print()

    # 1. Adım: Düşük F/K'lı hisseleri bul
    if verbose:
        print("🔍 Düşük F/K'lı hisseler aranıyor...")

    screener = bp.Screener()
    screener.add_filter("pe", min=0, max=pe_max)  # Negatif F/K hariç

    if index:
        screener.set_index(index)

    low_pe_stocks = screener.run()

    if low_pe_stocks.empty:
        if verbose:
            print(f"❌ F/K < {pe_max} olan hisse bulunamadı.")
        return pd.DataFrame()

    if verbose:
        print(f"✅ {len(low_pe_stocks)} hisse bulundu (F/K < {pe_max})")
        print()

    # 2. Adım: Her hisse için kar marjı eğilimini kontrol et
    results = []

    if verbose:
        print("📈 Kar marjı eğilimleri kontrol ediliyor...")
        print("-" * 60)

    skipped_banks = 0
    skipped_no_data = 0
    skipped_no_trend = 0

    for _, row in low_pe_stocks.iterrows():
        symbol = row["symbol"]
        name = row.get("name", "")
        # PE sütunu "pe" veya "criteria_28" olarak gelebilir (İş Yatırım API)
        pe = row.get("pe") or row.get("criteria_28") or row.get("pe_ratio")

        # Banka ve finans sektörünü atla (farklı mali tablo formatı)
        if symbol in BANK_SYMBOLS:
            skipped_banks += 1
            continue

        try:
            ticker = bp.Ticker(symbol)
            income_stmt = ticker.get_income_stmt(quarterly=True)

            if income_stmt.empty:
                skipped_no_data += 1
                continue

            margins = calculate_net_margin(income_stmt)

            if margins.empty:
                skipped_no_data += 1
                continue

            # Son n çeyrekte yükseliyor mu?
            if is_margin_increasing(margins, last_n=quarters):
                # Son 3 çeyreğin marjlarını al
                recent_margins = margins.head(quarters).sort_index()
                margin_values = recent_margins.values
                margin_quarters = recent_margins.index.tolist()

                # PE değeri zaten yukarıda alındı
                pe_val = pe

                results.append(
                    {
                        "symbol": symbol,
                        "name": name,
                        "pe": pe_val,
                        "margin_q1": margin_values[0] if len(margin_values) > 0 else None,
                        "margin_q2": margin_values[1] if len(margin_values) > 1 else None,
                        "margin_q3": margin_values[2] if len(margin_values) > 2 else None,
                        "quarters": " → ".join(margin_quarters),
                    }
                )

                if verbose:
                    margin_str = " → ".join([f"{m:.1f}%" for m in margin_values])
                    pe_str = f"{pe_val:.1f}" if pe_val is not None else "N/A"
                    print(f"✅ {symbol:8} F/K: {pe_str:>5} | Marj: {margin_str}")
            else:
                skipped_no_trend += 1

        except Exception:
            skipped_no_data += 1
            continue

    if verbose and (skipped_banks or skipped_no_data or skipped_no_trend):
        print()
        print(f"   ℹ️  Atlanan: {skipped_banks} banka/finans, {skipped_no_data} veri yok, {skipped_no_trend} trend yok")

    if verbose:
        print("-" * 60)
        print()

    if not results:
        if verbose:
            print("❌ Kriterlere uyan hisse bulunamadı.")
        return pd.DataFrame()

    # DataFrame oluştur
    df = pd.DataFrame(results)

    if verbose:
        print(f"🎯 Toplam {len(df)} hisse kriterlere uyuyor:")
        print()
        print(df.to_string(index=False))

    return df


def main():
    """Ana fonksiyon."""
    print("=" * 60)
    print("borsapy - Düşük F/K ve Yükselen Kar Marjı Taraması")
    print("=" * 60)
    print()

    # Taramayı çalıştır
    results = screen_low_pe_rising_margin(
        pe_max=8.0,
        quarters=3,
        index=None,  # Tüm BIST
        verbose=True,
    )

    if not results.empty:
        # CSV'ye kaydet
        output_file = "low_pe_rising_margin_results.csv"
        results.to_csv(output_file, index=False)
        print()
        print(f"📁 Sonuçlar '{output_file}' dosyasına kaydedildi.")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
