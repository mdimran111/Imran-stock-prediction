
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

STOCKS = {
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "TCS": "TCS.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "BHARTIARTL": "BHARTIARTL.NS"
}

def fetch_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", progress=False)
    df.dropna(inplace=True)
    return df

def create_features(df):
    df["Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()

    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift()),
            abs(df["Low"] - df["Close"].shift())
        )
    )
    df["ATR"] = df["TR"].rolling(14).mean()

    df["Target"] = np.where(df["Return"].shift(-1) > 0, 1, 0)
    df.dropna(inplace=True)
    return df
def predict_signal(df):
    if df is None or df.empty or len(df) < 30:
        return "HOLD 🟡", 0.0, None, None, None

    features = ["MA5", "MA10", "RSI"]
    X = df[features].iloc[:-1]
    y = (df["Close"].shift(-1) > df["Close"]).astype(int).iloc[:-1]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    latest = df.iloc[-1]
    latest_features = latest[features].values.reshape(1, -1)

    pred = model.predict(latest_features)[0]
    prob = model.predict_proba(latest_features)[0][pred]

    close = float(latest["Close"])
    ma5 = float(latest["MA5"])
    ma10 = float(latest["MA10"])
    atr = float(latest["ATR"])

    entry = round(close, 2)

    if pred == 1 and close > ma5 and ma5 > ma10:
        signal = "BUY 🟢"
        sl = round(entry - 1.5 * atr, 2)
        tp = round(entry + 3 * atr, 2)
    elif pred == 0 and close < ma5 and ma5 < ma10:
        signal = "SELL 🔴"
        sl = round(entry + 1.5 * atr, 2)
        tp = round(entry - 3 * atr, 2)
    else:
        signal = "HOLD 🟡"
        sl = None
        tp = None

    return signal, round(prob * 100, 2), entry, sl, tp
