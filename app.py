import streamlit as st
import pandas as pd
import requests
import numpy as np
import json
import os
import datetime
from concurrent.futures import ThreadPoolExecutor

# ==============================
# SETUP
# ==============================
st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto AI PRO FULL SYSTEM")

DATA_FILE = "data.json"

# ==============================
# SAFE JSON STORAGE
# ==============================
def load_data():
    if not os.path.exists(DATA_FILE):
        return []

    try:
        with open(DATA_FILE, "r") as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except:
        return []

def save_data(data):
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except:
        pass

def safe_append(record):
    data = load_data()

    if not isinstance(data, list):
        data = []

    data.append(record)
    save_data(data)

# ==============================
# MARKET DATA
# ==============================
@st.cache_data(ttl=300)
def get_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "volume_desc",
        "per_page": 50,
        "page": 1
    }
    return requests.get(url, params=params).json()

# ==============================
# INDICATORS
# ==============================
def rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)

    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:]) + 1e-9

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def ema(prices, span):
    return pd.Series(prices).ewm(span=span).mean().values[-1]

# ==============================
# ANALYZE COIN
# ==============================
def analyze_coin(coin):
    try:
        coin_id = coin["id"]

        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": 30}

        data = requests.get(url, params=params).json()
        prices = np.array([p[1] for p in data["prices"]])

        if len(prices) < 30:
            return None

        price = prices[-1]
        max_price = prices.max()

        drop = ((price - max_price) / max_price) * 100

        rsi_val = rsi(prices[-20:])
        ema20 = ema(prices, 20)
        ema50 = ema(prices, 50)

        ema_trend = 1 if ema20 > ema50 else 0

        # ==============================
        # SIMPLE STRATEGY
        # ==============================
        if rsi_val < 30 and drop < -20:
            signal = "STRONG BUY"
        elif rsi_val < 40:
            signal = "BUY"
        else:
            signal = "NO TRADE"

        # ==============================
        # SAVE TO JSON (SAFE)
        # ==============================
        safe_append({
            "time": str(datetime.datetime.now()),
            "coin": coin["symbol"],
            "price": float(price),
            "rsi": float(rsi_val),
            "drop": float(drop),
            "ema_trend": int(ema_trend),
            "signal": signal,
            "result": "PENDING"
        })

        return {
            "Coin": coin["symbol"].upper(),
            "Price": round(price, 6),
            "RSI": round(rsi_val, 2),
            "Drop %": round(drop, 2),
            "EMA Trend": ema_trend,
            "Signal": signal
        }

    except:
        return None

# ==============================
# SCAN MARKET
# ==============================
if st.button("🔍 Scan Market AI PRO"):

    coins = get_coins()
    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        data = list(executor.map(analyze_coin, coins))

    for r in data:
        if r:
            results.append(r)

    df = pd.DataFrame(results)

    st.success(f"🔥 Found {len(df)} signals")
    st.dataframe(df, use_container_width=True)
