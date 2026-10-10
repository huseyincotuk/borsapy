"""
Büyüme Hissesi Tarama
=====================

Son 4 çeyrekte tutarlı gelir ve kar büyümesi gösteren şirketleri bulur.

Kriterler:
- Her çeyrekte bir önceki yılın aynı çeyreğine göre (YoY) gelir artışı
- Her çeyrekte YoY net kar artışı
- Pozitif büyüme trendi

Kullanım:
    python examples/growth_stock_scanner.py
"""

import pandas as pd

import borsapy as bp


def analyze_growth(symbol: str, min_quarters: int = 4) -> dict | None:
    """
    Çeyreklik büyüme analizi yap.

    Returns:
        Büyüme metrikleri veya None
    """
    try:
        ticker = bp.Ticker(symbol)
        income_stmt = ticker.get_income_stmt(quarterly=True)

        if income_stmt.empty:
            return None

        # Çeyrek sütunlarını bul ve sırala
        quarter_cols = [col for col in income_stmt.columns if 'Q' in str(col)]
        quarter_cols = sorted(quarter_cols, reverse=True)  # En yeni önce

        if len(quarter_cols) < min_quarters + 4:  # YoY için +4 çeyrek gerekli
            return None

        # Gelir ve net kar satırlarını bul
        revenue_idx = None
        net_income_idx = None

        for idx in income_stmt.index:
            idx_lower = str(idx).lower()
            if 'satış gelirleri' in idx_lower:
                revenue_idx = idx
            if 'ana ortaklık payları' in idx_lower:
                net_income_idx = idx

        if revenue_idx is None or net_income_idx is None:
            return None

        # YoY büyüme hesapla
        revenue_growth = []
        profit_growth = []

        for i in range(min_quarters):
            current_q = quarter_cols[i]
            # 4 çeyrek önceki (aynı dönem geçen yıl)
            year_ago_idx = i + 4

            if year_ago_idx >= len(quarter_cols):
                break

            year_ago_q = quarter_cols[year_ago_idx]

            # Gelir büyümesi
            rev_current = income_stmt.loc[revenue_idx, current_q]
            rev_year_ago = income_stmt.loc[revenue_idx, year_ago_q]

            if pd.notna(rev_current) and pd.notna(rev_year_ago) and rev_year_ago > 0:
                rev_growth = (rev_current - rev_year_ago) / rev_year_ago * 100
                revenue_growth.append({
                    'quarter': current_q,
                    'growth': round(rev_growth, 1)
                })

            # Kar büyümesi
            profit_current = income_stmt.loc[net_income_idx, current_q]
            profit_year_ago = income_stmt.loc[net_income_idx, year_ago_q]

            if pd.notna(profit_current) and pd.notna(profit_year_ago) and profit_year_ago > 0:
                pft_growth = (profit_current - profit_year_ago) / profit_year_ago * 100
                profit_growth.append({
                    'quarter': current_q,
                    'growth': round(pft_growth, 1)
                })

        if len(revenue_growth) < min_quarters or len(profit_growth) < min_quarters:
            return None

        # Tutarlı büyüme kontrolü
        positive_rev_growth = sum(1 for g in revenue_growth if g['growth'] > 0)
        positive_profit_growth = sum(1 for g in profit_growth if g['growth'] > 0)

        avg_rev_growth = sum(g['growth'] for g in revenue_growth) / len(revenue_growth)
        avg_profit_growth = sum(g['growth'] for g in profit_growth) / len(profit_growth)

        return {
            'symbol': symbol,
            'revenue_growth_quarters': positive_rev_growth,
            'profit_growth_quarters': positive_profit_growth,
            'avg_revenue_growth': round(avg_rev_growth, 1),
            'avg_profit_growth': round(avg_profit_growth, 1),
            'latest_rev_growth': revenue_growth[0]['growth'],
            'latest_profit_growth': profit_growth[0]['growth'],
            'revenue_details': revenue_growth,
            'profit_details': profit_growth,
        }

    except Exception:
        return None


def scan_growth_stocks(
    index: str = "XU100",
    min_growth_quarters: int = 3,
    min_avg_growth: float = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Büyüme hissesi taraması."""

    if verbose:
        print("📊 Büyüme Hissesi Tarama")
        print(f"   - Endeks: {index}")
        print(f"   - Min pozitif büyüme çeyreği: {min_growth_quarters}/4")
        print(f"   - Min ortalama büyüme: %{min_avg_growth}")
        print()

    idx = bp.Index(index)
    symbols = idx.component_symbols

    # Banka sembollerini filtrele
    bank_symbols = {"AKBNK", "GARAN", "ISCTR", "VAKBN", "YKBNK", "HALKB", "SKBNK", "TSKB"}
    symbols = [s for s in symbols if s not in bank_symbols]

    if verbose:
        print(f"🔍 {len(symbols)} hisse analiz ediliyor (bankalar hariç)...")
        print("-" * 80)

    results = []

    for i, symbol in enumerate(symbols):
        if verbose:
            print(f"\r   İşleniyor: {i+1}/{len(symbols)} - {symbol:8}", end="", flush=True)

        growth = analyze_growth(symbol)

        if growth is None:
            continue

        # Filtreler
        passes_rev = growth['revenue_growth_quarters'] >= min_growth_quarters
        passes_profit = growth['profit_growth_quarters'] >= min_growth_quarters
        passes_avg = (growth['avg_revenue_growth'] >= min_avg_growth or
                      growth['avg_profit_growth'] >= min_avg_growth)

        if passes_rev and passes_profit and passes_avg:
            # Büyüme skoru
            growth['growth_score'] = round(
                (growth['avg_revenue_growth'] + growth['avg_profit_growth']) / 2, 1
            )
            results.append(growth)

    if verbose:
        print()
        print("-" * 80)
        print()

    if not results:
        if verbose:
            print("❌ Kriterlere uyan hisse bulunamadı.")
        return pd.DataFrame()

    # Skora göre sırala
    df = pd.DataFrame(results)
    df = df.sort_values('growth_score', ascending=False).reset_index(drop=True)

    if verbose:
        print(f"🚀 {len(df)} Büyüme Hissesi Bulundu:")
        print()
        print(f"{'Sembol':<8} {'Gelir +':>8} {'Kar +':>8} {'Ort.Gelir':>12} {'Ort.Kar':>12} {'Skor':>8}")
        print("-" * 65)

        for _, row in df.head(15).iterrows():
            print(f"{row['symbol']:<8} "
                  f"{row['revenue_growth_quarters']}/4{' ':>4} "
                  f"{row['profit_growth_quarters']}/4{' ':>4} "
                  f"%{row['avg_revenue_growth']:>10.1f} "
                  f"%{row['avg_profit_growth']:>10.1f} "
                  f"{row['growth_score']:>7.1f}")

        print()
        print("💡 Gelir+/Kar+: Son 4 çeyrekte YoY pozitif büyüme sayısı")
        print("💡 Ort.Gelir/Kar: 4 çeyreklik ortalama YoY büyüme oranı")

    # Detay sütunlarını kaldır (export için)
    df = df.drop(columns=['revenue_details', 'profit_details'], errors='ignore')

    return df


def main():
    print("=" * 80)
    print("borsapy - Büyüme Hissesi Tarama")
    print("=" * 80)
    print()

    results = scan_growth_stocks(
        index="XU100",
        min_growth_quarters=3,
        min_avg_growth=10,
        verbose=True,
    )

    if not results.empty:
        results.to_csv("growth_stocks_results.csv", index=False)
        print()
        print("📁 Sonuçlar 'growth_stocks_results.csv' dosyasına kaydedildi.")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
