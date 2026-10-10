"""
Çoklu Varlık Portföy Takibi
===========================

Hisse + Altın + Döviz + Kripto içeren bir portföyün
değerini ve performansını takip eder.

borsapy'nin Portfolio sınıfını kullanır.

Kullanım:
    python examples/multi_asset_portfolio.py
"""


import borsapy as bp


def create_sample_portfolio() -> bp.Portfolio:
    """Örnek çoklu varlık portföyü oluştur."""

    portfolio = bp.Portfolio()

    # Hisseler
    portfolio.add("THYAO", shares=100, cost=250.0)
    portfolio.add("ASELS", shares=50, cost=45.0)
    portfolio.add("BIMAS", shares=30, cost=180.0)

    # Altın (FX olarak)
    portfolio.add("gram-altin", shares=10, cost=2800.0, asset_type="fx")

    # Döviz
    portfolio.add("USD", shares=1000, cost=32.0, asset_type="fx")
    portfolio.add("EUR", shares=500, cost=35.0, asset_type="fx")

    # Kripto
    portfolio.add("BTCTRY", shares=0.01, cost=2_000_000.0)
    portfolio.add("ETHTRY", shares=0.5, cost=120_000.0)

    # Fon
    portfolio.add("YAY", shares=1000, cost=2.5, asset_type="fund")

    return portfolio


def analyze_portfolio(portfolio: bp.Portfolio, verbose: bool = True) -> dict:
    """Portföy analizi yap."""

    if verbose:
        print("📊 PORTFÖY DURUMU")
        print("=" * 80)
        print()

    # Holdings
    holdings = portfolio.holdings

    if verbose:
        print("📦 VARLIKLAR:")
        print("-" * 80)
        print(f"{'Sembol':<12} {'Tip':<8} {'Adet':>12} {'Maliyet':>12} {'Değer':>12} {'K/Z':>12} {'Ağırlık':>8}")
        print("-" * 80)

        for _, row in holdings.iterrows():
            pnl = row['value'] - row['cost']
            pnl_str = f"+{pnl:,.0f}" if pnl >= 0 else f"{pnl:,.0f}"
            pnl_color = "✅" if pnl >= 0 else "❌"

            print(f"{row['symbol']:<12} {row['asset_type']:<8} "
                  f"{row['shares']:>12,.4f} {row['cost']:>12,.0f} "
                  f"{row['value']:>12,.0f} {pnl_color}{pnl_str:>10} "
                  f"%{row['weight']:>6.1f}")

        print("-" * 80)
        print(f"{'TOPLAM':<12} {'':<8} {'':<12} "
              f"{holdings['cost'].sum():>12,.0f} "
              f"{holdings['value'].sum():>12,.0f} "
              f"{holdings['pnl'].sum():>12,.0f}")

    # Varlık tipi dağılımı
    type_breakdown = holdings.groupby('asset_type').agg({
        'value': 'sum',
        'weight': 'sum'
    }).round(2)

    if verbose:
        print()
        print("📈 VARLIK TİPİ DAĞILIMI:")
        print("-" * 40)
        for asset_type, row in type_breakdown.iterrows():
            bar = "█" * int(row['weight'] / 5)
            print(f"   {asset_type:<10} %{row['weight']:>5.1f} {bar}")

    # Risk metrikleri
    if verbose:
        print()
        print("📉 RİSK METRİKLERİ (1 Yıllık):")
        print("-" * 40)

    try:
        risk = portfolio.risk_metrics(period="1y")

        if verbose:
            print(f"   Yıllık Getiri:    %{risk.get('annualized_return', 0):.1f}")
            print(f"   Volatilite:       %{risk.get('annualized_volatility', 0):.1f}")
            print(f"   Sharpe Oranı:     {risk.get('sharpe_ratio', 0):.2f}")
            print(f"   Sortino Oranı:    {risk.get('sortino_ratio', 0):.2f}")
            print(f"   Max Drawdown:     %{risk.get('max_drawdown', 0):.1f}")
    except Exception as e:
        risk = {}
        if verbose:
            print(f"   Risk metrikleri hesaplanamadı: {e}")

    # Özet
    total_value = portfolio.value
    total_cost = holdings['cost'].sum()
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

    summary = {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct,
        'holdings_count': len(holdings),
        'risk_metrics': risk,
        'type_breakdown': type_breakdown.to_dict(),
    }

    if verbose:
        print()
        print("=" * 80)
        print(f"💰 TOPLAM PORTFÖY DEĞERİ: {total_value:,.0f} TL")
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        print(f"{pnl_emoji}  TOPLAM K/Z: {total_pnl:+,.0f} TL (%{total_pnl_pct:+.1f})")
        print("=" * 80)

    return summary


def main():
    print("=" * 80)
    print("borsapy - Çoklu Varlık Portföy Takibi")
    print("=" * 80)
    print()

    # Portföy oluştur
    print("🔧 Örnek portföy oluşturuluyor...")
    print("   - 3 Hisse (THYAO, ASELS, BIMAS)")
    print("   - 1 Altın (10 gram)")
    print("   - 2 Döviz (USD, EUR)")
    print("   - 2 Kripto (BTC, ETH)")
    print("   - 1 Fon (YAY)")
    print()

    portfolio = create_sample_portfolio()

    # Analiz
    analyze_portfolio(portfolio, verbose=True)

    # Holdings'i kaydet
    holdings = portfolio.holdings
    holdings.to_csv("multi_asset_portfolio.csv", index=False)

    print()
    print("📁 Portföy 'multi_asset_portfolio.csv' dosyasına kaydedildi.")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
