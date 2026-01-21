
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
    X = df[["Return", "MA5", "MA10"]]
    y = df["Target"]

    model = LogisticRegression()
    model.fit(X[:-1], y[:-1])

    pred = model.predict(X.tail(1))[0]
    prob = model.predict_proba(X.tail(1))[0][pred]

    latest = df.iloc[-1]
    entry = round(latest["Close"], 2)
    atr = latest["ATR"]

    if pred == 1 and latest["Close"] > latest["MA5"] > latest["MA10"]:
        signal = "BUY 🟢"
        sl = round(entry - 1.5 * atr, 2)
        tp = round(entry + 3 * atr, 2)
    elif pred == 0 and latest["Close"] < latest["MA5"] < latest["MA10"]:
        signal = "SELL 🔴"
        sl = round(entry + 1.5 * atr, 2)
        tp = round(entry - 3 * atr, 2)
    else:
        signal = "HOLD 🟡"
        sl = None
        tp = None

    return signal, prob, entry, sl, tp
