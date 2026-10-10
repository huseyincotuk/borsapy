"""
ATR Volatilite Tarayıcı
=======================

Average True Range (ATR) ile volatilite analizi yapar.

Yüksek ATR = Yüksek volatilite (swing trading için uygun)
Düşük ATR = Düşük volatilite (breakout beklenebilir)

Kullanım:
    python examples/atr_volatility_scanner.py
"""

import borsapy as bp


def scan_volatility(index_name: str = "XU030", verbose: bool = True) -> dict:
    """ATR bazlı volatilite taraması yap."""

    if verbose:
        print("📊 ATR Volatilite Tarayıcı")
        print("=" * 60)
        print()

    # Endeks bileşenlerini al
    index = bp.Index(index_name)
    symbols = index.component_symbols

    if verbose:
        print(f"🔍 {index_name} endeksindeki {len(symbols)} hisse taranıyor...")
        print()

    volatility_data = []

    for symbol in symbols:
        try:
            stock = bp.Ticker(symbol)
            df = stock.history(period="3mo")

            if df is None or len(df) < 20:
                continue

            # ATR hesapla
            atr = stock.atr()
            current_price = df['Close'].iloc[-1]

            # ATR yüzdesi (fiyata göre normalize)
            atr_pct = (atr / current_price) * 100

            # Son 20 günlük ortalama hacim
            avg_volume = df['Volume'].tail(20).mean()

            # Bollinger genişliği (volatilite göstergesi)
            bb = stock.bollinger_bands()
            bb_width = ((bb['upper'] - bb['lower']) / bb['middle']) * 100 if bb else 0

            volatility_data.append({
                'symbol': symbol,
                'price': current_price,
                'atr': atr,
                'atr_pct': atr_pct,
                'bb_width': bb_width,
                'avg_volume': avg_volume,
            })

        except Exception as e:
            if verbose:
                print(f"   ⚠️ {symbol}: {e}")

    # ATR yüzdesine göre sırala
    volatility_data.sort(key=lambda x: x['atr_pct'], reverse=True)

    if verbose:
        print("📈 EN YÜKSEK VOLATİLİTE (Swing Trading İçin)")
        print("-" * 70)
        print(f"{'Sembol':<10} {'Fiyat':>10} {'ATR':>10} {'ATR %':>8} {'BB Width':>10} {'Ort.Hacim':>12}")
        print("-" * 70)

        for v in volatility_data[:10]:
            print(f"{v['symbol']:<10} {v['price']:>10.2f} {v['atr']:>10.2f} "
                  f"%{v['atr_pct']:>7.2f} %{v['bb_width']:>9.2f} {v['avg_volume']:>12,.0f}")

        print()
        print("📉 EN DÜŞÜK VOLATİLİTE (Breakout Bekleyenler)")
        print("-" * 70)
        print(f"{'Sembol':<10} {'Fiyat':>10} {'ATR':>10} {'ATR %':>8} {'BB Width':>10} {'Ort.Hacim':>12}")
        print("-" * 70)

        for v in volatility_data[-10:]:
            print(f"{v['symbol']:<10} {v['price']:>10.2f} {v['atr']:>10.2f} "
                  f"%{v['atr_pct']:>7.2f} %{v['bb_width']:>9.2f} {v['avg_volume']:>12,.0f}")

        print()
        avg_atr_pct = sum(v['atr_pct'] for v in volatility_data) / len(volatility_data)
        print(f"📊 Ortalama ATR %: {avg_atr_pct:.2f}%")

    return {
        'high_volatility': volatility_data[:10],
        'low_volatility': volatility_data[-10:],
        'all': volatility_data,
    }


if __name__ == "__main__":
    results = scan_volatility("XU030")
