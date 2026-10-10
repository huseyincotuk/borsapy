"""
MACD Histogram Tersine Dönüş Tarayıcı
=====================================

MACD histogramın tersine döndüğü (momentum değişimi) hisseleri bulur.

Strateji:
- Histogram negatiften pozitife geçiş = bullish reversal
- Histogram pozitiften negatife geçiş = bearish reversal

Kullanım:
    python examples/macd_histogram_reversal.py
"""

import borsapy as bp


def find_macd_reversals(index_name: str = "XU030", verbose: bool = True) -> dict:
    """MACD histogram tersine dönüşlerini bul."""

    if verbose:
        print("📊 MACD Histogram Tersine Dönüş Tarayıcı")
        print("=" * 60)
        print()

    # Endeks bileşenlerini al
    index = bp.Index(index_name)
    symbols = index.component_symbols

    if verbose:
        print(f"🔍 {index_name} endeksindeki {len(symbols)} hisse taranıyor...")
        print()

    bullish_reversals = []
    bearish_reversals = []

    for symbol in symbols:
        try:
            stock = bp.Ticker(symbol)
            ta = stock.technicals(period="3mo")

            # MACD hesapla
            macd_df = ta.macd()
            if macd_df is None or len(macd_df) < 3:
                continue

            # Son 3 histogram değeri
            hist = macd_df['Histogram'].iloc[-3:].values

            # Bullish reversal: negatiften pozitife
            if hist[-2] < 0 and hist[-1] > 0:
                bullish_reversals.append({
                    'symbol': symbol,
                    'prev_hist': hist[-2],
                    'curr_hist': hist[-1],
                    'macd': macd_df['MACD'].iloc[-1],
                    'signal': macd_df['Signal'].iloc[-1],
                })

            # Bearish reversal: pozitiften negatife
            elif hist[-2] > 0 and hist[-1] < 0:
                bearish_reversals.append({
                    'symbol': symbol,
                    'prev_hist': hist[-2],
                    'curr_hist': hist[-1],
                    'macd': macd_df['MACD'].iloc[-1],
                    'signal': macd_df['Signal'].iloc[-1],
                })

        except Exception as e:
            if verbose:
                print(f"   ⚠️ {symbol}: {e}")

    if verbose:
        # Bullish reversals
        print("📈 BULLISH REVERSALS (Negatiften Pozitife)")
        print("-" * 60)
        if bullish_reversals:
            print(f"{'Sembol':<10} {'Önceki Hist':>12} {'Güncel Hist':>12} {'MACD':>10}")
            print("-" * 60)
            for r in bullish_reversals:
                print(f"{r['symbol']:<10} {r['prev_hist']:>12.4f} {r['curr_hist']:>12.4f} {r['macd']:>10.4f}")
        else:
            print("   Bullish reversal bulunamadı.")

        print()
        print("📉 BEARISH REVERSALS (Pozitiften Negatife)")
        print("-" * 60)
        if bearish_reversals:
            print(f"{'Sembol':<10} {'Önceki Hist':>12} {'Güncel Hist':>12} {'MACD':>10}")
            print("-" * 60)
            for r in bearish_reversals:
                print(f"{r['symbol']:<10} {r['prev_hist']:>12.4f} {r['curr_hist']:>12.4f} {r['macd']:>10.4f}")
        else:
            print("   Bearish reversal bulunamadı.")

        print()
        print(f"📊 Özet: {len(bullish_reversals)} bullish, {len(bearish_reversals)} bearish reversal")

    return {
        'bullish': bullish_reversals,
        'bearish': bearish_reversals,
    }


if __name__ == "__main__":
    results = find_macd_reversals("XU030")
