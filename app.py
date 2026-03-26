import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Scanner AI PRO")

JSON_FILE = "data.json"

# =========================
# LOAD / SAVE JSON
# =========================
def load_data():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(JSON_FILE, "w") as f:
        json.dump(data, f, indent=4)

# =========================
# COINGECKO TOP COINS
# =========================
def get_top_coins(limit=20):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "volume_desc",
        "per_page": limit,
        "page": 1
    }
    data = requests.get(url, params=params).json()
    coins = []
    for c in data:
        coins.append(c["symbol"].upper() + "USDT")
    return coins

# =========================
# APIs (NO BINANCE)
# =========================
def get_kucoin(symbol):
    try:
        url = f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={symbol.replace('USDT','-USDT')}"
        r = requests.get(url).json()
        return float(r["data"]["price"])
    except:
        return None

def get_bybit(symbol):
    try:
        url = f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}"
        r = requests.get(url).json()
        return float(r["result"]["list"][0]["lastPrice"])
    except:
        return None

def get_okx(symbol):
    try:
        url = f"https://www.okx.com/api/v5/market/ticker?instId={symbol.replace('USDT','-USDT')}"
        r = requests.get(url).json()
        return float(r["data"][0]["last"])
    except:
        return None

# =========================
# FEATURES
# =========================
def compute_momentum(prices):
    if len(prices) < 4:
        return 0
    return prices[-1] - prices[-4]

def volx(prices):
    if len(prices) < 10:
        return 1
    return round(np.std(prices[-10:]) / np.mean(prices[-10:]), 2)

def momentum_icon(x):
    if x > 2:
        return "🟢 ↑↑"
    elif x > 0:
        return "🟢 ↑"
    elif x == 0:
        return "🟡 ●"
    elif x > -2:
        return "🔴 ↓"
    else:
        return "🔴 ↓↓"

def data_status(n):
    if n > 30:
        return "🟢 GOOD"
    elif n > 15:
        return "🟡 MEDIUM"
    return "🔴 LOW"

# =========================
# AI MODEL
# =========================
def train_model(dataset):
    X, y = [], []

    for d in dataset:
        if not all(k in d for k in ["momentum", "volx", "price", "label"]):
            continue

        X.append([d["momentum"], d["volx"], d["price"]])
        y.append(d["label"])

    if len(X) < 20:
        return None

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

def ai_score(model, features):
    if model is None:
        return round(np.random.uniform(40, 90), 1)

    prob = model.predict_proba([features])[0][1]
    return round(prob * 100, 1)

def signal(score):
    if score > 80:
        return "🟢 STRONG BUY"
    elif score > 60:
        return "🟢 BUY"
    elif score > 40:
        return "🟡 WAIT"
    return "🔴 SELL"

# =========================
# RUN
# =========================
if st.button("🚀 Run AI Scan"):

    symbols = get_top_coins(20)
    data = load_data()
    model = train_model(data)

    rows = []

    for sym in symbols:

        price = get_kucoin(sym) or get_bybit(sym) or get_okx(sym)
        if not price:
            continue

        # simulate history
        history = [price * (1 + np.random.normal(0, 0.01)) for _ in range(20)]
        history.append(price)

        mom = compute_momentum(history)
        vx = volx(history)

        score = ai_score(model, [mom, vx, price])
        sig = signal(score)

        rows.append({
            "Coin": sym,
            "Price": price,
            "Signal": sig,
            "Momentum": momentum_icon(mom),
            "AI Score": score,
            "VolX": vx,
            "Data Status": data_status(len(history))
        })

        data.append({
            "momentum": mom,
            "volx": vx,
            "price": price,
            "label": 1 if score > 70 else 0
        })

    save_data(data)

    df = pd.DataFrame(rows)

    # =========================
    # TABLE FORMAT
    # =========================
    df = df[[
        "Coin",
        "Price",
        "Signal",
        "Momentum",
        "AI Score",
        "VolX",
        "Data Status"
    ]]

    df["Price"] = df["Price"].apply(lambda x: f"{x:,.4f}")
    df["AI Score"] = df["AI Score"].apply(lambda x: f"{x}%")
    df["VolX"] = df["VolX"].apply(lambda x: f"{x}x")

    df = df.sort_values(by="AI Score", ascending=False)

    # =========================
    # COLORS
    # =========================
    def color_signal(val):
        if "STRONG BUY" in val:
            return "background-color: #006400; color: white"
        elif "BUY" in val:
            return "background-color: green; color: white"
        elif "WAIT" in val:
            return "background-color: orange; color: black"
        elif "SELL" in val:
            return "background-color: red; color: white"
        return ""

    def color_status(val):
        if "GOOD" in val:
            return "background-color: green; color: white"
        elif "MEDIUM" in val:
            return "background-color: orange; color: black"
        elif "LOW" in val:
            return "background-color: red; color: white"
        return ""

    styled_df = df.style.applymap(color_signal, subset=["Signal"]) \
                         .applymap(color_status, subset=["Data Status"])

    st.subheader("📊 Trading Dashboard")
    st.dataframe(styled_df, use_container_width=True)

    # =========================
    # STATS
    # =========================
    st.subheader("🧠 AI Learning Status")

    st.write("Dataset size:", len(data))
    st.write("Avg confidence:", round(df["AI Score"].str.replace("%","").astype(float).mean(), 2), "%")

    buy = len(df[df["Signal"].str.contains("BUY")])
    st.write("Active BUY signals:", buy)
