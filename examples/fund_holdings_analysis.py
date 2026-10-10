"""
Fon Portföy Analizi
===================

Fon varlık dağılımını ve performansını analiz eder.

borsapy'nin Fund sınıfı ile allocation özelliğini kullanır.

Kullanım:
    python examples/fund_holdings_analysis.py
"""

import pandas as pd

import borsapy as bp


def analyze_fund_allocation(fund_code: str, verbose: bool = True) -> dict:
    """Fon varlık dağılımını analiz et."""

    if verbose:
        print(f"📊 {fund_code} Fon Analizi")
        print("=" * 70)
        print()

    try:
        fund = bp.Fund(fund_code)
        info = fund.info

        if verbose:
            print("📋 FON BİLGİLERİ:")
            print(f"   Ad: {info.get('name', 'N/A')}")
            print(f"   Tip: {info.get('fund_type', 'N/A')}")
            print(f"   Fiyat: {info.get('price', 0):.4f} TL")
            print(f"   Fon Büyüklüğü: {info.get('fund_size', 0):,.0f} TL")
            print(f"   Yatırımcı Sayısı: {info.get('investor_count', 0):,}")
            print(f"   Risk Değeri: {info.get('risk_value', 'N/A')}/7")
            print()

        # Getiriler
        if verbose:
            print("📈 GETİRİLER:")
            print("-" * 50)
            print(f"   Günlük:    %{info.get('daily_return', 0) or 0:>8.2f}")
            print(f"   1 Aylık:   %{info.get('return_1m', 0) or 0:>8.2f}")
            print(f"   3 Aylık:   %{info.get('return_3m', 0) or 0:>8.2f}")
            print(f"   6 Aylık:   %{info.get('return_6m', 0) or 0:>8.2f}")
            print(f"   YTD:       %{info.get('return_ytd', 0) or 0:>8.2f}")
            print(f"   1 Yıllık:  %{info.get('return_1y', 0) or 0:>8.2f}")
            print(f"   3 Yıllık:  %{info.get('return_3y', 0) or 0:>8.2f}")
            print()

        # Risk metrikleri
        try:
            risk = fund.risk_metrics(period="1y")
            if verbose:
                print("📉 RİSK METRİKLERİ (1 Yıllık):")
                print("-" * 50)
                print(f"   Yıllık Getiri:     %{risk.get('annualized_return', 0):.2f}")
                print(f"   Yıllık Volatilite: %{risk.get('annualized_volatility', 0):.2f}")
                print(f"   Sharpe Oranı:      {risk.get('sharpe_ratio', 0):.2f}")
                print(f"   Sortino Oranı:     {risk.get('sortino_ratio', 0):.2f}")
                print(f"   Max Drawdown:      %{risk.get('max_drawdown', 0):.2f}")
                print()
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Risk metrikleri hesaplanamadı: {e}")
                print()

        # Varlık dağılımı (allocation)
        allocation = info.get('allocation', [])
        if allocation and verbose:
            print("📦 VARLIK DAĞILIMI:")
            print("-" * 50)
            print(f"{'Varlık Tipi':<30} {'Ağırlık':>15}")
            print("-" * 50)

            for item in allocation:
                asset_name = item.get('asset_name', item.get('asset_type', 'N/A'))
                weight = item.get('weight', 0)
                bar = "█" * int(weight / 5)
                print(f"{asset_name:<30} %{weight:>12.2f} {bar}")

            print("-" * 50)
            total_weight = sum(item.get('weight', 0) for item in allocation)
            print(f"{'TOPLAM':<30} %{total_weight:>12.2f}")

        result = {
            'fund_code': fund_code,
            'fund_name': info.get('name'),
            'fund_type': info.get('fund_type'),
            'price': info.get('price'),
            'fund_size': info.get('fund_size'),
            'return_1y': info.get('return_1y'),
            'risk_value': info.get('risk_value'),
            'allocation': allocation,
        }

        return result

    except Exception as e:
        if verbose:
            print(f"❌ Hata: {e}")
        return {'fund_code': fund_code, 'error': str(e)}


def compare_fund_allocations(fund_codes: list[str], verbose: bool = True) -> pd.DataFrame:
    """Birden fazla fonun varlık dağılımını karşılaştır."""

    if verbose:
        print("📊 FON VARLIK DAĞILIMI KARŞILAŞTIRMASI")
        print("=" * 70)
        print()

    results = []

    for code in fund_codes:
        try:
            fund = bp.Fund(code)
            info = fund.info

            results.append({
                'fund_code': code,
                'name': info.get('name', code)[:30],
                'fund_type': info.get('fund_type', 'N/A'),
                'return_1y': info.get('return_1y', 0),
                'return_ytd': info.get('return_ytd', 0),
                'fund_size': info.get('fund_size', 0),
                'risk_value': info.get('risk_value', 0),
            })

            if verbose:
                print(f"✅ {code}: {info.get('name', 'N/A')[:40]}")

        except Exception as e:
            if verbose:
                print(f"❌ {code}: {e}")

    if not results:
        if verbose:
            print("❌ Fon verisi alınamadı.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values('return_1y', ascending=False).reset_index(drop=True)

    if verbose:
        print()
        print("📋 KARŞILAŞTIRMA TABLOSU:")
        print("-" * 80)
        print(f"{'Kod':<8} {'Ad':<32} {'1Y':>10} {'YTD':>10} {'Risk':>8}")
        print("-" * 80)

        for _, row in df.iterrows():
            return_1y = row['return_1y'] if row['return_1y'] is not None else 0
            return_ytd = row['return_ytd'] if row['return_ytd'] is not None else 0
            risk_value = row['risk_value'] if row['risk_value'] is not None else 0
            print(f"{row['fund_code']:<8} {row['name']:<32} "
                  f"%{return_1y:>8.1f} %{return_ytd:>8.1f} "
                  f"{risk_value:>7}/7")

    return df


def main():
    print("=" * 70)
    print("borsapy - Fon Portföy Analizi")
    print("=" * 70)
    print()

    # Örnek fon analizi - Teknoloji fonu
    analyze_fund_allocation("YAY", verbose=True)

    print()
    print("=" * 70)
    print()

    # Birden fazla fon karşılaştırması
    funds_to_compare = ["YAY", "TTE", "AFO", "AAK", "GAF"]

    print(f"📊 Karşılaştırılacak fonlar: {', '.join(funds_to_compare)}")
    print()

    comparison = compare_fund_allocations(funds_to_compare, verbose=True)

    if not comparison.empty:
        comparison.to_csv("fund_comparison_results.csv", index=False)
        print()
        print("📁 Sonuçlar 'fund_comparison_results.csv' dosyasına kaydedildi.")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
