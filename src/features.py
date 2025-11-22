import pandas as pd

def add_return_and_target(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["Return"] = data["Close"].pct_change()
    data["Tomorrow_Return"] = data["Return"].shift(-1)
    data["Target"] = (data["Tomorrow_Return"] > 0).astype(int)
    return data

def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # Moving averages
    data["MA5"] = data["Close"].rolling(window=5).mean()
    data["MA10"] = data["Close"].rolling(window=10).mean()
    data["MA20"] = data["Close"].rolling(window=20).mean()

    # Volatility
    data["Volatility_10"] = data["Return"].rolling(window=10).std()

    # RSI
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    # Bollinger Bands
    close_array = data["Close"].to_numpy().ravel()
    close = pd.Series(close_array, index=data.index)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    data["BB_Middle"] = ma20
    data["BB_Upper"] = ma20 + 2 * std20
    data["BB_Lower"] = ma20 - 2 * std20
    data["BB_Percent"] = (close - data["BB_Lower"]) / (data["BB_Upper"] - data["BB_Lower"])

    # Lagged returns
    data["Lag1"] = data["Return"].shift(1)
    data["Lag2"] = data["Return"].shift(2)
    data["Lag3"] = data["Return"].shift(3)

    # Final clean
    data = data.dropna()
    return data

def get_feature_columns():
    return [
        "Return", "Lag1", "Lag2", "Lag3",
        "MA5", "MA10", "MA20",
        "Volatility_10",
        "RSI",
        "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Percent",
        "Volume",
    ]
