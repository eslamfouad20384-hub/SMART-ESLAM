import streamlit as st
import pandas as pd
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Reversal Scanner PRO AI")

# ==============================
# إعدادات
# ==============================
MIN_VOLUME = 10_000_000

# ==============================
# جلب العملات
# ==============================
@st.cache_data(ttl=300)
def get_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "volume_desc",
        "per_page": 100,
        "page": 1
    }
    return requests.get(url, params=params).json()

# ==============================
# INDICATORS
# ==============================
def rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50

    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)

    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:]) + 1e-9

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def ema(prices, span=20):
    return pd.Series(prices).ewm(span=span).mean().values[-1]

def bollinger(prices, window=20):
    sma = np.mean(prices[-window:])
    std = np.std(prices[-window:])
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower

def atr(prices):
    diffs = np.abs(np.diff(prices))
    return np.mean(diffs[-14:]) if len(diffs) >= 14 else np.mean(diffs)

# ==============================
# تحليل العملة
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

        # ==============================
        # Drop %
        # ==============================
        drop = ((price - max_price) / max_price) * 100

        # ==============================
        # Indicators
        # ==============================
        rsi_val = rsi(prices[-20:])
        ema20 = ema(prices, 20)
        ema50 = ema(prices, 50)
        upper, lower = bollinger(prices)
        atr_val = atr(prices)

        # ==============================
        # Conditions
        # ==============================
        near_support = price <= lower
        trend_up = ema20 > ema50
        volatility_ok = atr_val > np.mean(prices) * 0.001

        # ==============================
        # Score Engine
        # ==============================
        score = 0

        if drop < -25:
            score += 2

        if rsi_val < 35:
            score += 2

        if near_support:
            score += 2

        if trend_up:
            score += 2

        if volatility_ok:
            score += 1

        # ==============================
        # Signal
        # ==============================
        if score >= 8:
            signal = "🚀 STRONG REVERSAL BUY"
        elif score >= 6:
            signal = "🔥 BUY"
        elif score >= 4:
            signal = "⏳ WAIT"
        else:
            signal = "❌ NO TRADE"

        return {
            "Coin": coin["symbol"].upper(),
            "Price": round(price, 6),
            "Drop %": round(drop, 2),
            "RSI": round(rsi_val, 2),
            "EMA20": round(ema20, 6),
            "EMA50": round(ema50, 6),
            "Support": round(lower, 6),
            "Resistance": round(upper, 6),
            "ATR": round(atr_val, 6),
            "Score": score,
            "Signal": signal
        }

    except:
        return None

# ==============================
# RUN SCANNER
# ==============================
if st.button("🔍 Scan السوق بالكامل"):

    coins = get_coins()
    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        data = list(executor.map(analyze_coin, coins))

    for r in data:
        if r:
            results.append(r)

    df = pd.DataFrame(results)

    if not df.empty:
        df = df.sort_values(by="Score", ascending=False)

        df = df[df["Score"] >= 5]

        st.success(f"🔥 تم العثور على {len(df)} فرصة قوية")

        st.dataframe(df, use_container_width=True)

    else:
        st.warning("❌ مفيش فرص حالياً")
