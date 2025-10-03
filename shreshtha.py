import pandas as pd
import numpy as np
from backtester import BackTester

def calculate_indicators(data):
    # ATR
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(window=14).mean()

    # RSI
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = data['close'].ewm(span=12, adjust=False).mean()
    ema26 = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # SMA
    data['SMA_20'] = data['close'].rolling(window=20).mean()

    return data

def strat(data):
    data['trade_type'] = "HOLD"
    data['signals'] = 0
    position = 0
    trailing_stop = 0
    trailing_stop_multiplier = 2
    num_wrong = 0

    for i in range(14, len(data)):
        vol_spike = np.mean(data.loc[i - 5:i, 'volume']) + 1.5 * np.std(data.loc[i - 5:i, 'volume'])

        if position == 0:
            if data.loc[i, 'volume'] > vol_spike:
                if data.loc[i, 'close'] > data.loc[i, 'open']:
                    data.loc[i, 'signals'] = 1
                    position = 1
                    data.loc[i, 'trade_type'] = "LONG"
                    trailing_stop = data.loc[i, 'close'] - (data.loc[i, "ATR"] * trailing_stop_multiplier)
                elif data.loc[i, 'close'] < data.loc[i, 'open']:
                    data.loc[i, 'signals'] = -1
                    position = -1
                    data.loc[i, 'trade_type'] = "SHORT"
                    trailing_stop = data.loc[i, 'close'] + (data.loc[i, "ATR"] * trailing_stop_multiplier)

        elif position == 1:
            trend_rev = data.loc[i, 'volume'] >= vol_spike and data.loc[i, 'close'] < data.loc[i, 'open']
            if data.loc[i, 'close'] <= data.loc[i - 1, 'close']:
                num_wrong += 1
            else:
                num_wrong = 0

            if trend_rev:
                data.loc[i, 'signals'] = -2
                position = -1
                trailing_stop = data.loc[i, 'close'] + (data.loc[i, "ATR"] * trailing_stop_multiplier)
                num_wrong = 0
                data.loc[i, 'trade_type'] = "REVERSE_LONG_TO_SHORT"
            elif num_wrong == 3:
                data.loc[i, 'signals'] = -1
                position = 0
                num_wrong = 0
                data.loc[i, 'trade_type'] = "CLOSE"
            elif data.loc[i, "close"] < trailing_stop:
                data.loc[i, 'signals'] = -1
                position = 0
                data.loc[i, 'trade_type'] = 'CLOSE'
            else:
                trailing_stop = max(trailing_stop, data.loc[i, "close"] - (data.loc[i, "ATR"] * trailing_stop_multiplier))

        elif position == -1:
            trend_rev = data.loc[i, 'volume'] >= vol_spike and data.loc[i, 'close'] > data.loc[i, 'open']
            if data.loc[i, 'close'] >= data.loc[i - 1, 'close']:
                num_wrong += 1
            else:
                num_wrong = 0

            if trend_rev:
                data.loc[i, 'signals'] = 2
                position = 1
                trailing_stop = data.loc[i, 'close'] - (data.loc[i, "ATR"] * trailing_stop_multiplier)
                num_wrong = 0
                data.loc[i, 'trade_type'] = "REVERSE_SHORT_TO_LONG"
            elif num_wrong == 3:
                data.loc[i, 'signals'] = 1
                position = 0
                num_wrong = 0
                data.loc[i, 'trade_type'] = "CLOSE"
            elif data.loc[i, "close"] > trailing_stop:
                data.loc[i, 'signals'] = 1
                position = 0
                data.loc[i, 'trade_type'] = 'CLOSE'
            else:
                trailing_stop = min(trailing_stop, data.loc[i, "close"] + (data.loc[i, "ATR"] * trailing_stop_multiplier))
    return data

def main():
    data = pd.read_csv("BTC_2019_2023_1d.csv")
    processed_data = calculate_indicators(data)
    result_data = strat(processed_data)
    result_data.to_csv("final_data.csv", index=False)

    bt = BackTester("BTC", signal_data_path="final_data.csv", master_file_path="final_data.csv", compound_flag=1)
    bt.get_trades(1000)

    for trade in bt.trades:
        print(trade)
        print(trade.pnl())

    stats = bt.get_statistics()
    for key, val in stats.items():
        print(key, ":", val)

    print("Checking for lookahead bias...")
    lookahead_bias = False
    for i in range(len(result_data)):
        if result_data.loc[i, 'signals'] != 0:
            temp_data = data.iloc[:i + 1].copy()
            temp_data = calculate_indicators(temp_data)
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
