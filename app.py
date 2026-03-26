import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Scanner AI PRO MAX")

JSON_FILE = "data.json"

# =========================
# LOAD / SAVE DATA
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
# TOP COINS
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
        symbol = c.get("symbol")
        if symbol:
            coins.append(symbol.upper() + "USDT")

    return coins

# =========================
# PRICE (KuCoin)
# =========================
def get_price(symbol):
    try:
        url = f"https://api.kucoin.com/api/v1/market/orderbook/level1"
        params = {"symbol": symbol.replace("USDT", "-USDT")}
        r = requests.get(url, params=params).json()
        return float(r["data"]["price"])
    except:
        return None

# =========================
# OHLC HISTORY (REAL)
# =========================
def get_history(symbol):
    try:
        url = "https://api.kucoin.com/api/v1/market/candles"
        params = {
            "type": "1min",
            "symbol": symbol.replace("USDT", "-USDT")
        }

        r = requests.get(url, params=params).json()

        prices = [float(c[2]) for c in r["data"][:30]]  # close price
        return prices

    except:
        return []

# =========================
# FEATURES
# =========================
def compute_momentum(prices):
    if len(prices) < 5:
        return 0
    return prices[-1] - prices[-5]

def volx(prices):
    if len(prices) < 10:
        return 0
    return round(np.std(prices) / np.mean(prices), 4)

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
    if n > 25:
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
        if "momentum" in d and "volx" in d and "price" in d:
            X.append([d["momentum"], d["volx"], d["price"]])
            y.append(d["label"])

    if len(X) < 20:
        return None

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

# =========================
# AI SCORE (VOLX INCLUDED)
# =========================
def ai_score(model, features):
    mom, vx, price = features

    base = 0

    # momentum effect
    base += mom * 2

    # volatility effect
    if vx > 0.06:
        base += 20
    elif vx < 0.01:
        base -= 10

    if model is None:
        return max(0, min(100, 50 + base))

    prob = model.predict_proba([features])[0][1] * 100

    return max(0, min(100, prob + base))

# =========================
# SIGNALS
# =========================
def signal(score, vx):
    if vx > 0.06:
        if score > 70:
            return "🔥 VOLATILE BUY"
        return "⚠️ HIGH RISK"

    if score > 80:
        return "🟢 STRONG BUY"
    elif score > 60:
        return "🟢 BUY"
    elif score > 40:
        return "🟡 WAIT"
    return "🔴 SELL"

# =========================
# RUN SCANNER
# =========================
if st.button("🚀 Run AI Scan"):

    symbols = get_top_coins(20)
    data = load_data()
    model = train_model(data)

    rows = []

    for sym in symbols:

        price = get_price(sym)
        if not price:
            continue

        history = get_history(sym)
        if len(history) < 10:
            continue

        mom = compute_momentum(history)
        vx = volx(history)

        score = ai_score(model, [mom, vx, price])
        sig = signal(score, vx)

        rows.append({
            "Coin": sym,
            "Price": round(price, 4),
            "Signal": sig,
            "Momentum": momentum_icon(mom),
            "AI Score": round(score, 2),
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

    df = df.sort_values(by="AI Score", ascending=False)

    st.subheader("📊 AI Trading Dashboard")
    st.dataframe(df, use_container_width=True)

    st.subheader("🧠 AI Learning Status")
    st.write("Dataset size:", len(data))
    st.write("Active BUY signals:", len(df[df["Signal"].str.contains("BUY")]))
