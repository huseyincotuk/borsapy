"""
Nakit Zengini Şirketler
=======================

Borcundan fazla nakdi olan (net nakit pozitif) şirketleri bulur.

Net Nakit = Nakit ve Nakit Benzerleri - Toplam Finansal Borç

Nakit zengini şirketler:
- Ekonomik krizlere dayanıklı
- Temettü ödeme kapasitesi yüksek
- Büyüme fırsatlarını değerlendirebilir
- Hisse geri alımı yapabilir

Kullanım:
    python examples/cash_rich_companies.py
"""

import pandas as pd

import borsapy as bp


def analyze_cash_position(symbol: str) -> dict | None:
    """
    Nakit pozisyonu analizi.

    Returns:
        Nakit metrikleri veya None
    """
    try:
        ticker = bp.Ticker(symbol)
        balance_sheet = ticker.balance_sheet
        info = ticker.info

        if balance_sheet.empty:
            return None

        # Değişkenleri başlat
        cash = None
        financial_debt = None
        total_assets = None
        total_equity = None

        # Bilanço kalemlerini bul
        for idx in balance_sheet.index:
            idx_lower = str(idx).lower()

            # Nakit ve nakit benzerleri
            if 'nakit ve nakit benzerleri' in idx_lower and cash is None:
                cash = balance_sheet.loc[idx].iloc[0]

            # Finansal borçlar (kısa + uzun vadeli)
            if 'finansal borç' in idx_lower:
                val = balance_sheet.loc[idx].iloc[0]
                if pd.notna(val):
                    if financial_debt is None:
                        financial_debt = val
                    else:
                        financial_debt += val

            # Toplam varlıklar
            if 'toplam varlıklar' in idx_lower:
                total_assets = balance_sheet.loc[idx].iloc[0]

            # Özkaynaklar
            if 'özkaynaklar' in idx_lower and 'ana ortaklık' not in idx_lower:
                if total_equity is None:
                    total_equity = balance_sheet.loc[idx].iloc[0]

        if cash is None:
            return None

        # Varsayılan değerler
        if financial_debt is None:
            financial_debt = 0

        # Net nakit pozisyonu
        net_cash = cash - financial_debt

        # Piyasa değeri
        market_cap = info.get('market_cap') or info.get('marketCap', 0)
        last_price = info.get('last') or info.get('regularMarketPrice', 0)

        # Oranlar
        cash_to_assets = (cash / total_assets * 100) if total_assets and total_assets > 0 else None
        net_cash_to_mcap = (net_cash / market_cap * 100) if market_cap and market_cap > 0 else None
        cash_to_debt = (cash / financial_debt) if financial_debt and financial_debt > 0 else float('inf')

        return {
            'symbol': symbol,
            'cash': cash,
            'financial_debt': financial_debt,
            'net_cash': net_cash,
            'market_cap': market_cap,
            'cash_to_assets_pct': round(cash_to_assets, 1) if cash_to_assets else None,
            'net_cash_to_mcap_pct': round(net_cash_to_mcap, 1) if net_cash_to_mcap else None,
            'cash_to_debt_ratio': round(cash_to_debt, 2) if cash_to_debt != float('inf') else None,
            'last_price': round(last_price, 2) if last_price else None,
        }

    except Exception:
        return None


def format_number(num, suffix=''):
    """Büyük sayıları formatla (milyar, milyon)."""
    if num is None:
        return "N/A"
    if abs(num) >= 1e9:
        return f"{num/1e9:.1f}B{suffix}"
    if abs(num) >= 1e6:
        return f"{num/1e6:.0f}M{suffix}"
    return f"{num:.0f}{suffix}"


def scan_cash_rich(
    index: str = "XU100",
    min_net_cash_ratio: float = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Nakit zengini şirket taraması."""

    if verbose:
        print("📊 Nakit Zengini Şirketler Tarama")
        print(f"   - Endeks: {index}")
        print("   - Kriter: Net Nakit > 0 (Nakit > Borç)")
        print()

    idx = bp.Index(index)
    symbols = idx.component_symbols

    # Bankaları hariç tut
    bank_symbols = {"AKBNK", "GARAN", "ISCTR", "VAKBN", "YKBNK", "HALKB", "SKBNK", "TSKB", "ALBRK"}
    symbols = [s for s in symbols if s not in bank_symbols]

    if verbose:
        print(f"🔍 {len(symbols)} hisse analiz ediliyor (bankalar hariç)...")
        print("-" * 85)

    results = []

    for i, symbol in enumerate(symbols):
        if verbose:
            print(f"\r   İşleniyor: {i+1}/{len(symbols)} - {symbol:8}", end="", flush=True)

        metrics = analyze_cash_position(symbol)

        if metrics is None:
            continue

        # Net nakit pozitif olanlar
        if metrics['net_cash'] > min_net_cash_ratio:
            results.append(metrics)

    if verbose:
        print()
        print("-" * 85)
        print()

    if not results:
        if verbose:
            print("❌ Kriterlere uyan hisse bulunamadı.")
        return pd.DataFrame()

    # Net nakit / piyasa değerine göre sırala
    df = pd.DataFrame(results)
    df = df.sort_values('net_cash', ascending=False).reset_index(drop=True)

    if verbose:
        print(f"💰 {len(df)} Nakit Zengini Şirket Bulundu (Net Nakit > 0):")
        print()
        print(f"{'Sembol':<8} {'Nakit':>12} {'Borç':>12} {'Net Nakit':>12} {'Nakit/Varlık':>13} {'Net/PD':>10}")
        print("-" * 75)

        for _, row in df.head(20).iterrows():
            cash_str = format_number(row['cash'])
            debt_str = format_number(row['financial_debt'])
            net_str = format_number(row['net_cash'])
            c2a_str = f"%{row['cash_to_assets_pct']:.1f}" if row['cash_to_assets_pct'] else "N/A"
            nc2m_str = f"%{row['net_cash_to_mcap_pct']:.1f}" if row['net_cash_to_mcap_pct'] else "N/A"

            print(f"{row['symbol']:<8} {cash_str:>12} {debt_str:>12} {net_str:>12} {c2a_str:>13} {nc2m_str:>10}")

        print()
        print("💡 Net Nakit = Nakit - Finansal Borç")
        print("💡 Net/PD: Net nakit / Piyasa değeri (yüksek = ucuz değerleme)")

        # En yüksek Net/PD oranına sahip 5 hisse
        top_value = df.dropna(subset=['net_cash_to_mcap_pct']).nlargest(5, 'net_cash_to_mcap_pct')
        if not top_value.empty:
            print()
            print("🏆 En Yüksek Net Nakit / Piyasa Değeri:")
            for _, row in top_value.iterrows():
                print(f"   {row['symbol']}: Net nakit piyasa değerinin %{row['net_cash_to_mcap_pct']:.1f}'i")

    return df


def main():
    print("=" * 85)
    print("borsapy - Nakit Zengini Şirketler")
    print("=" * 85)
    print()

    results = scan_cash_rich(
        index="XU100",
        verbose=True,
    )

    if not results.empty:
        results.to_csv("cash_rich_results.csv", index=False)
        print()
        print("📁 Sonuçlar 'cash_rich_results.csv' dosyasına kaydedildi.")

    print()
    print("=" * 85)


if __name__ == "__main__":
    main()
