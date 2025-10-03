import pandas as pd
import numpy as np
from backtester import BackTester

def process_data(data):
    close = data["close"]
    high = data["high"]
    low = data["low"]

    # SMA & EMA
    data["SMA_20"] = close.rolling(window=20, min_periods=1).mean()
    data["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    # Bollinger Bands
    sma = data["SMA_20"]
    std = close.rolling(window=20, min_periods=1).std()
    data["BB_upper"] = sma + 2 * std
    data["BB_lower"] = sma - 2 * std
    data["BB_width"] = data["BB_upper"] - data["BB_lower"]
    data["BB_pct"] = data["BB_width"] / close  # relative width

    # Donchian Channels
    data["Donchian_high"] = high.rolling(window=20).max()
    data["Donchian_low"] = low.rolling(window=20).min()

    # ATR
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["ATR"] = tr.rolling(window=14).mean()

    # Keltner Channels
    data["KC_middle"] = data["EMA_20"]
    data["KC_upper"] = data["KC_middle"] + 2 * data["ATR"]
    data["KC_lower"] = data["KC_middle"] - 2 * data["ATR"]

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))

    # ADX
    plus_dm = (high - high.shift()).clip(lower=0)
    minus_dm = (low.shift() - low).clip(lower=0)
    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr_all = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr_all.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    data["ADX"] = dx.ewm(alpha=1/14).mean()

    # Fibonacci Levels
    period = 60
    data["Fib_high"] = high.rolling(window=period).max()
    data["Fib_low"] = low.rolling(window=period).min()
    diff = data["Fib_high"] - data["Fib_low"]
    for level in [0.236, 0.382, 0.5, 0.618, 0.786, 1.618]:
        data[f"Fib_{level}"] = data["Fib_high"] - diff * level

    return data


def strat(data):
    cooldown_days=3
    data["signals"] = 0
    data["trade_type"] = "HOLD"
    position = 0
    trailing_stop = 0
    take_profit = 0
    entry_price = 0
    atr_mult = 2
    tp_mult = 3
    atr_mean = data["ATR"].rolling(window=14).mean()
    last_trade_idx = -cooldown_days

    for i in range(60, len(data)):
        price = data.loc[i, "close"]
        atr_val = data.loc[i, "ATR"]
        rsi = data.loc[i, "RSI"]
        macd = data.loc[i, "MACD"]
        macd_signal = data.loc[i, "MACD_signal"]
        adx = data.loc[i, "ADX"]
        bb_pct = data.loc[i, "BB_pct"]

        # ✅ Skip choppy zones
        if adx < 20 or bb_pct < 0.02:
            continue

        trending = adx > 30
        high_volatility = atr_val > atr_mean.iloc[i]

        fib50 = data.loc[i, "Fib_0.5"]
        fib618 = data.loc[i, "Fib_0.618"]
        donchian_high = data.loc[i, "Donchian_high"]
        donchian_low = data.loc[i, "Donchian_low"]
        kc_upper = data.loc[i, "KC_upper"]
        kc_lower = data.loc[i, "KC_lower"]

        if position == 0 and (i - last_trade_idx >= cooldown_days):
            if trending:
                score_fib = int(abs(price - fib50) < 0.005 * price or abs(price - fib618) < 0.005 * price)
                score_fib += int(macd > macd_signal)
                score_fib += int(rsi < 60)

                if score_fib >= 3:
                    data.loc[i, "signals"] = 1
                    data.loc[i, "trade_type"] = "LONG"
                    position = 1
                    entry_price = price
                    trailing_stop = price - atr_mult * atr_val
                    take_profit = price + tp_mult * atr_val
                    last_trade_idx = i
                    continue

                score_donchian_long = int(price > donchian_high) + int(macd > macd_signal) + int(high_volatility) + int(rsi < 70)
                if score_donchian_long >= 3:
                    data.loc[i, "signals"] = 1
                    data.loc[i, "trade_type"] = "LONG"
                    position = 1
                    entry_price = price
                    trailing_stop = price - atr_mult * atr_val
                    take_profit = price + tp_mult * atr_val
                    last_trade_idx = i
                    continue

                score_donchian_short = int(price < donchian_low) + int(macd < macd_signal) + int(high_volatility) + int(rsi > 30)
                if score_donchian_short >= 3:
                    data.loc[i, "signals"] = -1
                    data.loc[i, "trade_type"] = "SHORT"
                    position = -1
                    entry_price = price
                    trailing_stop = price + atr_mult * atr_val
                    take_profit = price - tp_mult * atr_val
                    last_trade_idx = i
                    continue

            else:
                score_kc_long = int(price > kc_upper) + int(macd > macd_signal) + int(high_volatility) + int(rsi < 70)
                if score_kc_long >= 3:
                    data.loc[i, "signals"] = 1
                    data.loc[i, "trade_type"] = "LONG"
                    position = 1
                    entry_price = price
                    trailing_stop = price - atr_mult * atr_val
                    take_profit = price + tp_mult * atr_val
                    last_trade_idx = i
                    continue

                score_kc_short = int(price < kc_lower) + int(macd < macd_signal) + int(high_volatility) + int(rsi > 30)
                if score_kc_short >= 3:
                    data.loc[i, "signals"] = -1
                    data.loc[i, "trade_type"] = "SHORT"
                    position = -1
                    entry_price = price
                    trailing_stop = price + atr_mult * atr_val
                    take_profit = price - tp_mult * atr_val
                    last_trade_idx = i
                    continue

        elif position == 1:
            if price < trailing_stop or price > take_profit:
                data.loc[i, "signals"] = -1
                data.loc[i, "trade_type"] = "CLOSE"
                position = 0
            else:
                trailing_stop = max(trailing_stop, price - atr_mult * atr_val)

        elif position == -1:
            if price > trailing_stop or price < take_profit:
                data.loc[i, "signals"] = 1
                data.loc[i, "trade_type"] = "CLOSE"
                position = 0
            else:
                trailing_stop = min(trailing_stop, price + atr_mult * atr_val)

    return data


def main():
    data = pd.read_csv("BTC_2019_2023_1d.csv")
    processed_data = process_data(data)
    result_data = strat(processed_data)
    result_data.to_csv("final_data.csv", index=False)

    bt = BackTester("BTC", signal_data_path="final_data.csv", master_file_path="final_data.csv", compound_flag=1)
    bt.get_trades(1000)

    for trade in bt.trades:
        print(trade, trade.pnl())

    stats = bt.get_statistics()
    for key, val in stats.items():
        print(key, ":", val)

    print("Checking for lookahead bias...")
    lookahead_bias = False
    for i in range(len(result_data)):
        if result_data.loc[i, 'signals'] != 0:
            temp_data = data.iloc[:i+1].copy()
            temp_data = process_data(temp_data)
            temp_data = strat(temp_data)
            if temp_data.loc[i, 'signals'] != result_data.loc[i, 'signals']:
                print(f"Lookahead bias detected at index {i}")
                lookahead_bias = True

    if not lookahead_bias:
        print("No lookahead bias detected.")

    bt.make_trade_graph()
    bt.make_pnl_graph()


if __name__ == "__main__":
    main()
