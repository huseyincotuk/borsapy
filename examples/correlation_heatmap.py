"""
Korelasyon Matrisi ve Isı Haritası
==================================

Hisseler arasındaki korelasyonları hesaplar ve görselleştirir.

Korelasyon:
- +1: Mükemmel pozitif (birlikte hareket)
- 0: Korelasyon yok
- -1: Mükemmel negatif (ters hareket)

Portföy çeşitlendirmesi için düşük/negatif korelasyonlu
hisseler tercih edilmelidir.

Kullanım:
    python examples/correlation_heatmap.py
"""

import numpy as np
import pandas as pd

import borsapy as bp


def get_correlation_matrix(
    symbols: list[str],
    period: str = "1y",
) -> tuple[pd.DataFrame | None, dict]:
    """
    Hisseler arası korelasyon matrisini hesapla.

    Returns:
        (korelasyon matrisi, hisse bilgileri)
    """
    prices = {}
    info_dict = {}

    for symbol in symbols:
        try:
            ticker = bp.Ticker(symbol)
            df = ticker.history(period=period)

            if df.empty or len(df) < 50:
                continue

            prices[symbol] = df['Close']

            # Sektör bilgisi
            info = ticker.info
            info_dict[symbol] = {
                'name': info.get('name', symbol),
                'sector': info.get('sector', 'Bilinmiyor'),
            }

        except Exception:
            continue

    if len(prices) < 2:
        return None, {}

    # DataFrame oluştur
    prices_df = pd.DataFrame(prices)
    prices_df = prices_df.dropna()

    # Getiri hesapla
    returns_df = prices_df.pct_change().dropna()

    # Korelasyon matrisi
    corr_matrix = returns_df.corr()

    return corr_matrix, info_dict


def print_text_heatmap(corr_matrix: pd.DataFrame, title: str = "Korelasyon Matrisi"):
    """Terminal'de basit ısı haritası göster."""

    def get_color_code(val: float) -> str:
        """Korelasyon değerine göre renk kodu."""
        if val >= 0.7:
            return "🟥"  # Yüksek pozitif
        elif val >= 0.4:
            return "🟧"  # Orta pozitif
        elif val >= 0.1:
            return "🟨"  # Düşük pozitif
        elif val >= -0.1:
            return "⬜"  # Nötr
        elif val >= -0.4:
            return "🟦"  # Düşük negatif
        else:
            return "🟪"  # Yüksek negatif

    symbols = corr_matrix.columns.tolist()

    # Başlık
    print(f"\n{title}")
    print("=" * (12 + len(symbols) * 8))

    # Header
    header = " " * 10
    for sym in symbols:
        header += f"{sym:>7} "
    print(header)

    # Satırlar
    for sym1 in symbols:
        row = f"{sym1:<8} "
        for sym2 in symbols:
            val = corr_matrix.loc[sym1, sym2]
            color = get_color_code(val)
            row += f"{color}{val:>5.2f} "
        print(row)

    # Legend
    print()
    print("Legend: 🟥>0.7  🟧>0.4  🟨>0.1  ⬜±0.1  🟦<-0.1  🟪<-0.4")


def find_diversification_pairs(corr_matrix: pd.DataFrame, threshold: float = 0.3) -> list:
    """Düşük korelasyonlu çiftleri bul."""
    pairs = []
    symbols = corr_matrix.columns.tolist()

    for i, sym1 in enumerate(symbols):
        for sym2 in symbols[i+1:]:
            corr = corr_matrix.loc[sym1, sym2]
            if corr < threshold:
                pairs.append({
                    'pair': f"{sym1}-{sym2}",
                    'correlation': round(corr, 3),
                })

    return sorted(pairs, key=lambda x: x['correlation'])


def analyze_correlations(
    symbols: list[str] | None = None,
    index: str = "XU030",
    period: str = "1y",
    verbose: bool = True,
) -> pd.DataFrame:
    """Korelasyon analizi yap."""

    # Semboller belirtilmemişse endeksten al
    if symbols is None:
        idx = bp.Index(index)
        symbols = idx.component_symbols[:15]  # İlk 15

    if verbose:
        print("📊 Korelasyon Analizi")
        print(f"   - Hisse sayısı: {len(symbols)}")
        print(f"   - Dönem: {period}")
        print()
        print("🔍 Veriler alınıyor...")

    corr_matrix, info_dict = get_correlation_matrix(symbols, period)

    if corr_matrix is None:
        if verbose:
            print("❌ Yeterli veri alınamadı.")
        return pd.DataFrame()

    if verbose:
        print(f"✅ {len(corr_matrix)} hisse analiz edildi")

        # Isı haritası
        print_text_heatmap(corr_matrix)

        # İstatistikler
        print()
        print("=" * 60)
        print("📈 KORELASYON İSTATİSTİKLERİ")
        print("=" * 60)

        # Üst üçgen değerleri (diagonal hariç)
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        all_corrs = upper_tri.stack().values

        print(f"\nOrtalama Korelasyon: {np.mean(all_corrs):.3f}")
        print(f"Medyan Korelasyon:   {np.median(all_corrs):.3f}")
        print(f"Min Korelasyon:      {np.min(all_corrs):.3f}")
        print(f"Max Korelasyon:      {np.max(all_corrs):.3f}")

        # En düşük korelasyonlu çiftler
        pairs = find_diversification_pairs(corr_matrix, threshold=0.4)

        print()
        print("🎯 ÇEŞİTLENDİRME İÇİN İDEAL ÇİFTLER (Düşük Korelasyon):")
        if pairs:
            for p in pairs[:10]:
                print(f"   {p['pair']:<15} Korelasyon: {p['correlation']:>6.3f}")
        else:
            print("   Düşük korelasyonlu çift bulunamadı.")

        # En yüksek korelasyonlu çiftler (dikkat!)
        high_corr_pairs = []
        for i, sym1 in enumerate(corr_matrix.columns):
            for sym2 in corr_matrix.columns[i+1:]:
                corr = corr_matrix.loc[sym1, sym2]
                if corr > 0.8:
                    high_corr_pairs.append({
                        'pair': f"{sym1}-{sym2}",
                        'correlation': round(corr, 3),
                    })

        if high_corr_pairs:
            print()
            print("⚠️  YÜKSEK KORELASYONLU ÇİFTLER (Dikkat - Çeşitlendirme yok):")
            for p in sorted(high_corr_pairs, key=lambda x: -x['correlation'])[:5]:
                print(f"   {p['pair']:<15} Korelasyon: {p['correlation']:>6.3f}")

    return corr_matrix


def main():
    print("=" * 60)
    print("borsapy - Korelasyon Analizi")
    print("=" * 60)
    print()

    # Farklı sektörlerden hisseler
    diverse_portfolio = [
        "THYAO",   # Havacılık
        "TUPRS",   # Rafineri
        "BIMAS",   # Perakende
        "ASELS",   # Savunma
        "AKBNK",   # Banka
        "TCELL",   # Telekom
        "EREGL",   # Metal
        "FROTO",   # Otomotiv
        "MGROS",   # Market
        "PGSUS",   # Havacılık
    ]

    corr_matrix = analyze_correlations(
        symbols=diverse_portfolio,
        period="1y",
        verbose=True,
    )

    if not corr_matrix.empty:
        corr_matrix.to_csv("correlation_matrix.csv")
        print()
        print("📁 Korelasyon matrisi 'correlation_matrix.csv' dosyasına kaydedildi.")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
