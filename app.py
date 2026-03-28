import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import json

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Scanner AI PRO MAX (Optimized)")

# ==============================
# 🔁 Caching
# ==============================
@st.cache_data(ttl=600)
def get_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency":"usd","order":"volume_desc","per_page":20,"page":1}
    return requests.get(url, params=params, timeout=10).json()

@st.cache_data(ttl=600)
def fetch_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency":"usd","days":30}

    for _ in range(3):  # retry
        try:
            return requests.get(url, params=params, timeout=10).json()
        except:
            time.sleep(1)
    return {}

# ==============================
# Indicators
# ==============================
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs)).iloc[-1]

def calculate_bollinger(prices):
    s = pd.Series(prices)
    sma = s.rolling(20).mean()
    std = s.rolling(20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std

    if pd.isna(upper.iloc[-1]):
        return 0,0

    return upper.iloc[-1], lower.iloc[-1]

def calculate_atr(prices):
    return np.std(prices[-14:])

def ema(prices, span):
    return pd.Series(prices).ewm(span=span).mean().iloc[-1]

# ==============================
# Collector (بدون nonlocal)
# ==============================
def run_collector():
    coins = get_coins()
    results = []

    def work(c):
        try:
            d = fetch_data(c["id"])
            prices = [p[1] for p in d.get("prices",[])]
            vols = [v[1] for v in d.get("total_volumes",[])]

            candles = []
            for i in range(len(prices)):
                candles.append({
                    "price": float(prices[i]),
                    "volume": float(vols[i]) if i < len(vols) else 0
                })

            return {"coin": c["symbol"].upper(), "candles": candles}
        except:
            return None

    with ThreadPoolExecutor(max_workers=5) as ex:
        data = list(ex.map(work, coins))

    return [d for d in data if d]

# ==============================
# Load Data
# ==============================
data = run_collector()

rows = []

# ==============================
# Build Dataset
# ==============================
for coin_data in data:

    if not isinstance(coin_data, dict):
        continue

    candles = coin_data.get("candles", [])
    if len(candles) < 30:
        continue

    prices = np.array([c["price"] for c in candles])
    vols = np.array([c["volume"] for c in candles])

    for i in range(20, len(prices)-3):

        rsi = calculate_rsi(prices[i-15:i])
        drop = ((prices[i] - prices[:i].max()) / prices[:i].max())
        volx = vols[i] / (vols[i-10:i].mean() + 1e-9)
        change = ((prices[i] - prices[i-3]) / prices[i-3])

        upper, lower = calculate_bollinger(prices[i-20:i])
        atr = calculate_atr(prices[i-14:i])

        ema_fast = ema(prices[i-10:i], 5)
        ema_slow = ema(prices[i-20:i], 10)

        rows.append({
            "rsi": rsi,
            "drop": drop,
            "volx": volx,
            "change": change,
            "bb": prices[i] - lower,
            "ema_diff": ema_fast - ema_slow,
            "atr": atr,
            "target": 1 if prices[i+3] > prices[i] else 0
        })

df_ai = pd.DataFrame(rows)

# ==============================
# Train or Load Model
# ==============================
MODEL_FILE = "model.pkl"

if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        model, scaler = pickle.load(f)
else:
    X = df_ai.drop("target", axis=1)
    y = df_ai["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_scaled, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump((model, scaler), f)

# ==============================
# Predictions
# ==============================
latest = []

for coin_data in data:

    if not isinstance(coin_data, dict):
        continue

    candles = coin_data.get("candles", [])
    if len(candles) < 30:
        continue

    prices = np.array([c["price"] for c in candles])
    vols = np.array([c["volume"] for c in candles])

    rsi = calculate_rsi(prices[-15:])
    drop = ((prices[-1] - prices.max()) / prices.max())
    volx = vols[-1] / (vols[-10:].mean() + 1e-9)
    change = ((prices[-1] - prices[-3]) / prices[-3])

    upper, lower = calculate_bollinger(prices[-20:])
    atr = calculate_atr(prices[-14:])

    ema_fast = ema(prices[-10:], 5)
    ema_slow = ema(prices[-20:], 10)

    features = np.array([[rsi, drop, volx, change, prices[-1]-lower, ema_fast-ema_slow, atr]])
    features_scaled = scaler.transform(features)

    chance = model.predict_proba(features_scaled)[0][1] * 100

    if chance >= 70:
        signal = "🔥 Strong Buy"
    elif chance >= 60:
        signal = "🚀 Buy"
    elif chance >= 50:
        signal = "🟡 Watch"
    else:
        signal = "❌ No Trade"

    latest.append({
        "Coin": coin_data["coin"],
        "Price": round(prices[-1], 2),
        "Chance %": round(chance, 2),
        "Signal": signal
    })

df = pd.DataFrame(latest)

st.dataframe(df.sort_values("Chance %", ascending=False), use_container_width=True)
