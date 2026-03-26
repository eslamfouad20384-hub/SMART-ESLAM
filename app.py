import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# =========================
# CONFIG
# =========================
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
# EXCHANGE APIS (NO BINANCE)
# =========================

def get_kucoin(symbol="BTC-USDT"):
    try:
        url = f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={symbol}"
        r = requests.get(url, timeout=5).json()
        return float(r["data"]["price"])
    except:
        return None


def get_bybit(symbol="BTCUSDT"):
    try:
        url = f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}"
        r = requests.get(url, timeout=5).json()
        return float(r["result"]["list"][0]["lastPrice"])
    except:
        return None


def get_okx(symbol="BTC-USDT"):
    try:
        url = f"https://www.okx.com/api/v5/market/ticker?instId={symbol}"
        r = requests.get(url, timeout=5).json()
        return float(r["data"][0]["last"])
    except:
        return None

# =========================
# FEATURE ENGINEERING
# =========================

def compute_momentum(prices):
    if len(prices) < 4:
        return 0
    return prices[-1] - prices[-4]


def volx(prices):
    if len(prices) < 5:
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

# =========================
# DATA STATUS
# =========================

def data_status(n):
    if n > 30:
        return "🟢 GOOD"
    elif n > 15:
        return "🟡 MEDIUM"
    return "🔴 LOW"

# =========================
# SIMPLE AI MODEL
# =========================

def train_model(dataset):
    if len(dataset) < 20:
        return None

    X = []
    y = []

    for d in dataset:
        X.append([
            d["momentum"],
            d["volx"],
            d["price"]
        ])
        y.append(d["label"])

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model


def ai_score(model, features):
    if model is None:
        return round(np.random.uniform(40, 90), 1)

    prob = model.predict_proba([features])[0][1]
    return round(prob * 100, 1)

# =========================
# SIGNAL LOGIC (SIMPLE)
# =========================

def signal(score):
    if score > 80:
        return "🟢 STRONG BUY"
    elif score > 60:
        return "🟢 BUY"
    elif score > 40:
        return "🟡 WAIT"
    return "🔴 SELL"

# =========================
# MAIN SCANNER
# =========================

coins = {
    "BTC-USDT": "BTCUSDT",
    "ETH-USDT": "ETHUSDT",
    "SOL-USDT": "SOLUSDT",
    "XRP-USDT": "XRPUSDT"
}

data = load_data()

model = train_model(data)

rows = []

for symbol, bybit_symbol in coins.items():

    price = (
        get_kucoin(symbol) or
        get_bybit(bybit_symbol) or
        get_okx(symbol)
    )

    if not price:
        continue

    # simulate price history
    history = [price * (1 + np.random.normal(0, 0.01)) for _ in range(20)]
    history.append(price)

    mom = compute_momentum(history)
    vx = volx(history)

    score = ai_score(model, [mom, vx, price])
    sig = signal(score)

    rows.append({
        "Coin": symbol,
        "Price": round(price, 4),
        "Signal": sig,
        "Momentum": momentum_icon(mom),
        "AI Score": score,
        "VolX": vx,
        "Data Status": data_status(len(history))
    })

    # save for learning
    data.append({
        "momentum": mom,
        "volx": vx,
        "price": price,
        "label": 1 if score > 70 else 0
    })

save_data(data)

# =========================
# DASHBOARD TABLE
# =========================

df = pd.DataFrame(rows)

st.subheader("📊 Market Dashboard")
st.dataframe(df, use_container_width=True)

# =========================
# SUMMARY
# =========================

buy = len(df[df["Signal"].str.contains("BUY")])
sell = len(df[df["Signal"].str.contains("SELL")])

st.write("🟢 BUY Signals:", buy)
st.write("🔴 SELL Signals:", sell)
