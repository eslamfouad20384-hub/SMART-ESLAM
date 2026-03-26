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
# LOAD / SAVE
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
# TOP COINS (SAFE)
# =========================
def get_top_coins(limit=20):
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "volume_desc",
            "per_page": limit,
            "page": 1
        }

        r = requests.get(url, params=params, timeout=5)

        if r.status_code != 200:
            return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

        data = r.json()

        coins = []
        for c in data:
            if isinstance(c, dict) and c.get("symbol"):
                coins.append(c["symbol"].upper() + "USDT")

        return coins

    except:
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# =========================
# PRICE
# =========================
def get_price(symbol):
    try:
        url = f"https://api.kucoin.com/api/v1/market/orderbook/level1"
        params = {"symbol": symbol.replace("USDT", "-USDT")}
        r = requests.get(url, params=params, timeout=5).json()
        return float(r["data"]["price"])
    except:
        return None

# =========================
# OHLC HISTORY
# =========================
def get_history(symbol):
    try:
        url = "https://api.kucoin.com/api/v1/market/candles"
        params = {
            "type": "1min",
            "symbol": symbol.replace("USDT", "-USDT")
        }

        r = requests.get(url, params=params, timeout=5).json()

        return [float(c[2]) for c in r["data"][:30]]

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
    return "🔴 ↓↓"

# =========================
# MARKET STATE
# =========================
def market_state(prices):
    if len(prices) < 20:
        return "UNKNOWN", 0

    returns = np.diff(prices) / prices[:-1]

    vol = np.std(returns)
    stress_score = vol * 100

    if stress_score > 8:
        return "CRITICAL", stress_score
    elif stress_score > 4:
        return "HIGH", stress_score
    elif stress_score > 2:
        return "NORMAL", stress_score
    else:
        return "CALM", stress_score

def crash_detector(prices):
    if len(prices) < 10:
        return False

    drop = (prices[-1] - prices[-5]) / prices[-5]

    return drop < -0.03

def auto_pause(state):
    return state == "CRITICAL"

# =========================
# AI MODEL
# =========================
def train_model(data):
    X, y = [], []

    for d in data:
        if all(k in d for k in ["momentum", "volx", "price", "label"]):
            X.append([d["momentum"], d["volx"], d["price"]])
            y.append(d["label"])

    if len(X) < 20:
        return None

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

def ai_score(model, features):
    mom, vx, price = features

    base = mom * 2

    if vx > 0.06:
        base += 20
    elif vx < 0.01:
        base -= 10

    if model is None:
        return max(0, min(100, 50 + base))

    prob = model.predict_proba([features])[0][1] * 100

    return max(0, min(100, prob + base))

def signal(score, vx, state):
    if state in ["CRITICAL", "HIGH"]:
        if score > 85:
            return "⚠️ SAFE BUY ONLY"
        return "⛔ NO TRADE"

    if vx > 0.06:
        return "🔥 VOLATILE BUY"

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

        price = get_price(sym)
        if not price:
            continue

        history = get_history(sym)
        if len(history) < 10:
            continue

        mom = compute_momentum(history)
        vx = volx(history)

        state, stress_score = market_state(history)

        if auto_pause(state):
            st.error("⛔ MARKET PAUSED (CRITICAL STRESS)")
            break

        if crash_detector(history):
            st.warning(f"⚠️ CRASH WARNING: {sym}")

        score = ai_score(model, [mom, vx, price])
        sig = signal(score, vx, state)

        rows.append({
            "Coin": sym,
            "Price": round(price, 4),
            "Signal": sig,
            "Momentum": momentum_icon(mom),
            "AI Score": round(score, 2),
            "VolX": vx,
            "Market State": state,
            "Stress": round(stress_score, 2)
        })

        data.append({
            "momentum": mom,
            "volx": vx,
            "price": price,
            "label": 1 if score > 70 else 0
        })

    save_data(data)

    df = pd.DataFrame(rows)

    st.subheader("📊 AI Dashboard")
    st.dataframe(df, use_container_width=True)

    st.subheader("🌐 Market Stress Status")

    if len(rows) > 0:
        avg_state = rows[0]["Market State"]

        if avg_state == "CRITICAL":
            st.error("🔴 CRITICAL MARKET")
        elif avg_state == "HIGH":
            st.warning("🟠 HIGH VOLATILITY")
        elif avg_state == "NORMAL":
            st.info("🟡 NORMAL MARKET")
        else:
            st.success("🟢 CALM MARKET")

    st.subheader("🧠 AI Stats")
    st.write("Dataset size:", len(data))
    st.write("Active BUY signals:", len(df[df["Signal"].str.contains("BUY")]))
