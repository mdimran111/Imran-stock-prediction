
import streamlit as st
import pandas as pd
from utils import STOCKS, fetch_data, create_features, predict_signal

st.set_page_config(page_title="Indian Stock Prediction App", layout="wide")

st.title("📈 Indian Stock Market Prediction App")
st.caption("Yahoo Finance | ML | BUY / SELL | SL & Target")

results = []

for stock, ticker in STOCKS.items():
    df = fetch_data(ticker)
    df = create_features(df)

    signal, prob, entry, sl, tp = predict_signal(df)

    results.append({
        "Stock": stock,
        "Signal": signal,
        "Confidence (%)": round(prob * 100, 2),
        "Entry Price": entry,
        "Stop Loss": sl,
        "Target": tp
    })

df_results = pd.DataFrame(results)

st.subheader("📊 Today's Trade Signals")
st.dataframe(df_results, use_container_width=True)

st.warning("⚠️ Educational purpose only. Not investment advice.")
