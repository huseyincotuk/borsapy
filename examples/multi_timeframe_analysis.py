"""
Çoklu Zaman Dilimi Analizi
==========================

Bir hisseyi farklı zaman dilimlerinde analiz eder.
Tüm zaman dilimlerinde aynı yönde sinyal = güçlü sinyal.

Kullanım:
    python examples/multi_timeframe_analysis.py
"""

import borsapy as bp


def analyze_multi_timeframe(symbol: str, verbose: bool = True) -> dict:
    """Çoklu zaman dilimi analizi yap."""

    if verbose:
        print(f"📊 Çoklu Zaman Dilimi Analizi: {symbol}")
        print("=" * 70)
        print()

    stock = bp.Ticker(symbol)

    # Farklı period'larla analiz
    timeframes = [
        ('1mo', 'Kısa Vade (1 Ay)'),
        ('3mo', 'Orta Vade (3 Ay)'),
        ('6mo', 'Uzun Vade (6 Ay)'),
        ('1y', 'Çok Uzun Vade (1 Yıl)'),
    ]

    results = {}

    for period, label in timeframes:
        try:
            ta = stock.technicals(period=period)

            # Göstergeleri hesapla
            rsi = ta.rsi().iloc[-1] if ta.rsi() is not None else None
            macd_df = ta.macd()
            macd = macd_df['MACD'].iloc[-1] if macd_df is not None else None
            signal = macd_df['Signal'].iloc[-1] if macd_df is not None else None

            # SMA trend
            sma_20 = ta.sma(20).iloc[-1] if ta.sma(20) is not None else None
            sma_50 = ta.sma(50).iloc[-1] if ta.sma(50) is not None and len(ta.sma(50)) > 0 else None

            # Mevcut fiyat
            df = ta._df
            current_price = df['Close'].iloc[-1]

            # Trend belirleme
            trend = "NEUTRAL"
            if sma_20 and sma_50:
                if current_price > sma_20 > sma_50:
                    trend = "BULLISH"
                elif current_price < sma_20 < sma_50:
                    trend = "BEARISH"

            # RSI durumu
            rsi_status = "NEUTRAL"
            if rsi:
                if rsi > 70:
                    rsi_status = "OVERBOUGHT"
                elif rsi < 30:
                    rsi_status = "OVERSOLD"

            # MACD durumu
            macd_status = "NEUTRAL"
            if macd and signal:
                macd_status = "BULLISH" if macd > signal else "BEARISH"

            results[period] = {
                'label': label,
                'price': current_price,
                'rsi': rsi,
                'rsi_status': rsi_status,
                'macd': macd,
                'macd_status': macd_status,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'trend': trend,
            }

        except Exception as e:
            if verbose:
                print(f"   ⚠️ {period}: {e}")
            results[period] = {'label': label, 'error': str(e)}

    if verbose:
        print(f"{'Zaman Dilimi':<25} {'Trend':<10} {'RSI':<12} {'MACD':<10} {'Fiyat':>10}")
        print("-" * 70)

        for _period, data in results.items():
            if 'error' in data:
                print(f"{data['label']:<25} ERROR")
                continue

            trend_emoji = "📈" if data['trend'] == "BULLISH" else "📉" if data['trend'] == "BEARISH" else "➡️"
            rsi_str = f"{data['rsi']:.1f}" if data['rsi'] else "N/A"
            macd_emoji = "📈" if data['macd_status'] == "BULLISH" else "📉" if data['macd_status'] == "BEARISH" else "➡️"

            print(f"{data['label']:<25} {trend_emoji} {data['trend']:<8} "
                  f"{rsi_str:<12} {macd_emoji} {data['macd_status']:<8} {data['price']:>10.2f}")

        # Genel değerlendirme
        print()
        print("=" * 70)
        print("📊 GENEL DEĞERLENDİRME:")

        bullish_count = sum(1 for d in results.values() if d.get('trend') == 'BULLISH')
        bearish_count = sum(1 for d in results.values() if d.get('trend') == 'BEARISH')

        if bullish_count >= 3:
            print("   ✅ GÜÇLÜ ALIM SİNYALİ - Çoğu zaman diliminde yükseliş trendi")
        elif bearish_count >= 3:
            print("   ❌ GÜÇLÜ SATIM SİNYALİ - Çoğu zaman diliminde düşüş trendi")
        else:
            print("   ⚠️ KARIŞIK SİNYALLER - Zaman dilimleri arasında uyumsuzluk var")

    return results


def scan_aligned_stocks(index_name: str = "XU030", verbose: bool = True) -> dict:
    """Tüm zaman dilimlerinde aynı yönde sinyal veren hisseleri bul."""

    if verbose:
        print("🔍 Zaman Dilimi Uyumlu Hisse Tarayıcı")
        print("=" * 70)
        print()

    index = bp.Index(index_name)
    symbols = index.component_symbols[:10]  # İlk 10 hisse (hız için)

    bullish_aligned = []
    bearish_aligned = []

    for symbol in symbols:
        try:
            result = analyze_multi_timeframe(symbol, verbose=False)

            trends = [d.get('trend') for d in result.values() if 'trend' in d]

            if all(t == 'BULLISH' for t in trends) and len(trends) >= 3:
                bullish_aligned.append(symbol)
            elif all(t == 'BEARISH' for t in trends) and len(trends) >= 3:
                bearish_aligned.append(symbol)

        except Exception:
            pass

    if verbose:
        print(f"📈 TÜM ZAMANDİLİMLERİNDE BULLISH: {bullish_aligned or 'Yok'}")
        print(f"📉 TÜM ZAMANDİLİMLERİNDE BEARISH: {bearish_aligned or 'Yok'}")

    return {
        'bullish_aligned': bullish_aligned,
        'bearish_aligned': bearish_aligned,
    }


if __name__ == "__main__":
    # Tek hisse analizi
    analyze_multi_timeframe("THYAO")

    print("\n" + "=" * 70 + "\n")

    # Uyumlu hisse taraması
    scan_aligned_stocks("XU030")
