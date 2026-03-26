import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from sklearn.ensemble import RandomForestClassifier
import os

st.set_page_config(layout="wide")
st.title("🚀 Crypto AI Signals PRO (No Binance)")

# =========================
# 🛡️ Safe Request (Rate Limit Protection)
# =========================
last_request_time = 0
REQUEST_DELAY = 0.8

def safe_request(url):
    global last_request_time
    try:
        now = time.time()
        if now - last_request_time < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY)

        last_request_time = time.time()
        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            return None

        return r.json()
    except:
        return None


# =========================
# 📡 KuCoin
# =========================
def get_kucoin(symbol="BTC-USDT"):
    url = f"https://api.kucoin.com/api/v1/market/candles?type=1min&symbol={symbol}"
    data = safe_request(url)

    if not data or "data" not in data:
        return []

    candles = []

    try:
        for c in data["data"]:
            candles.append({
                "timestamp": c[0],
                "price": float(c[2]),
                "volume": float(c[5]),
                "source": "kucoin"
            })
    except:
        pass

    return candles


# =========================
# 📡 Bybit
# =========================
def get_bybit(symbol="BTCUSDT"):
    url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval=1"
    data = safe_request(url)

    if not data or "result" not in data:
        return []

    candles = []

    try:
        for c in data["result"]["list"]:
            candles.append({
                "timestamp": c[0],
                "price": float(c[4]),
                "volume": float(c[5]),
                "source": "bybit"
            })
    except:
        pass

    return candles


# =========================
# 📡 OKX
# =========================
def get_okx(symbol="BTC-USDT"):
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar=1m"
    data = safe_request(url)

    if not data or "data" not in data:
        return []

    candles = []

    try:
        for c in data["data"]:
            candles.append({
                "timestamp": c[0],
                "price": float(c[4]),
                "volume": float(c[5]),
                "source": "okx"
            })
    except:
        pass

    return candles


# =========================
# 🔥 Merge All Data
# =========================
def get_all_data(symbol):
    data = []

    for func in [get_kucoin, get_bybit, get_okx]:
        try:
            d = func(symbol)
            if d:
                data.extend(d)
        except:
            continue

    if not data:
        return []

    df = pd.DataFrame(data)

    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp")

    return df.to_dict("records")


# =========================
# 🧠 Feature Engineering
# =========================
def create_features(df):
    df = pd.DataFrame(df)

    df["return"] = df["price"].pct_change()
    df["vol_change"] = df["volume"].pct_change()

    df["ma5"] = df["price"].rolling(5).mean()
    df["ma10"] = df["price"].rolling(10).mean()

    df["trend"] = df["ma5"] - df["ma10"]
    df["vol_spike"] = df["volume"] / df["volume"].rolling(10).mean()

    df = df.dropna()
    return df


# =========================
# 🟢 Data Status
# =========================
def get_data_status(df):
    if len(df) >= 40:
        return "🟢 GOOD"
    elif len(df) >= 20:
        return "🟡 MEDIUM"
    else:
        return "🔴 LOW"


# =========================
# 📊 Signal Engine
# =========================
def get_entry_exit_signal(df):
    last = df.iloc[-1]

    score = 0

    if last["trend"] > 0:
        score += 2

    if last["vol_spike"] > 1.5:
        score += 2

    if last["return"] > 0:
        score += 1

    if score >= 5:
        return "🟢 BUY STRONG"

    if score >= 3:
        return "🟢 BUY"

    if last["trend"] < 0 and last["vol_spike"] > 1.5:
        return "🔴 EXIT"

    return "⚪ HOLD"


# =========================
# 🚀 UI INPUT
# =========================
symbols = st.text_input("Enter symbols (comma separated)", "BTCUSDT,ETHUSDT,DOTUSDT")

if st.button("Run AI Scan"):

    coins = symbols.split(",")

    rows = []

    for symbol in coins:

        raw = get_all_data(symbol.strip())

        if len(raw) < 10:
            continue

        df = create_features(raw)

        if len(df) == 0:
            continue

        signal = get_entry_exit_signal(df)
        last = df.iloc[-1]

        rows.append({
            "Coin": symbol.strip(),
            "Price": round(last["price"], 4),
            "Trend": round(last["trend"], 4),
            "Volx": round(last["vol_spike"], 2),
            "Return %": round(last["return"] * 100, 2),
            "Data Status": get_data_status(df),
            "Signal": signal
        })

    result_df = pd.DataFrame(rows)

    # =========================
    # 🎨 Color Styling
    # =========================
    def color_status(val):
        if "GOOD" in val:
            return "background-color: #2ecc71; color: black;"
        elif "MEDIUM" in val:
            return "background-color: #f1c40f; color: black;"
        else:
            return "background-color: #e74c3c; color: white;"

    styled_df = result_df.style.applymap(
        color_status,
        subset=["Data Status"]
    )

    st.subheader("📊 Trading Signals Dashboard")
    st.dataframe(styled_df, use_container_width=True)
