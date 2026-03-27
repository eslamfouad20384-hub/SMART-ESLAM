import streamlit as st
import pandas as pd
import requests
import numpy as np
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Scanner PRO + AI Trade Engine")

# ==============================
# إعدادات
# ==============================
JSON_FILE = "signals.json"
MODEL_FILE = "ai_model.pkl"
SCALER_FILE = "scaler.pkl"

# ==============================
# JSON
# ==============================
def load_json():
    if not os.path.exists(JSON_FILE):
        return []
    with open(JSON_FILE, "r") as f:
        return json.load(f)

def save_json(data):
    with open(JSON_FILE, "w") as f:
        json.dump(data, f, indent=4)

def save_signal(signal):
    data = load_json()

    today = datetime.now().strftime("%Y-%m-%d")
    for item in data:
        if item["Coin"] == signal["Coin"] and item["date"] == today:
            return

    signal["date"] = today
    signal["time"] = datetime.now().strftime("%H:%M")

    data.append(signal)
    save_json(data)

# ==============================
# API
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
# RSI احترافي (Wilder)
# ==============================
def calculate_rsi(prices, period=14):
    prices = np.array(prices)

    if len(prices) < period + 1:
        return 50

    deltas = np.diff(prices)

    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

    if avg_loss[-1] == 0:
        return 100

    rs = avg_gain[-1] / avg_loss[-1]
    return 100 - (100 / (1 + rs))

# ==============================
# AI MODEL
# ==============================
def train_model():
    X, y = [], []

    for _ in range(2000):
        drop = np.random.uniform(-80, 10)
        rsi = np.random.uniform(10, 90)
        vol = np.random.uniform(0.5, 3)
        support = np.random.uniform(0, 1)
        trend = np.random.randint(0, 2)

        label = 1 if (drop < -20 and rsi < 40 and vol > 1.2) else 0

        X.append([drop, rsi, vol, support, trend])
        y.append(label)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

def load_model():
    if not os.path.exists(MODEL_FILE):
        train_model()

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    return model, scaler

# ==============================
# تحليل العملة (Multi-Timeframe + Trade Engine)
# ==============================
def analyze_coin(coin):
    try:
        coin_id = coin["id"]

        # ================= DAILY =================
        daily = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": 30}
        ).json()

        # ================= HOURLY =================
        hourly = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": 1, "interval": "hourly"}
        ).json()

        if "prices" not in daily or "prices" not in hourly:
            return None

        daily_prices = np.array([p[1] for p in daily["prices"]])
        hour_prices = np.array([p[1] for p in hourly["prices"]])

        volumes = np.array([v[1] for v in daily["total_volumes"]])

        current_price = daily_prices[-1]

        # ================= Indicators =================
        max_price = daily_prices.max()
        drop_percent = ((current_price - max_price) / max_price) * 100

        rsi_now = calculate_rsi(daily_prices[-15:])

        avg_volume = volumes[:-1].mean()
        current_volume = volumes[-1]

        support_zone = np.percentile(daily_prices[-20:], 20)

        trend_condition = current_price > pd.Series(daily_prices).ewm(span=20).mean().values[-1]

        hour_trend = hour_prices[-1] > np.mean(hour_prices[-5:])

        # ================= AI =================
        model, scaler = load_model()

        features = np.array([[
            drop_percent,
            rsi_now,
            current_volume / avg_volume if avg_volume > 0 else 1,
            support_zone / current_price,
            1 if trend_condition else 0
        ]])

        ai_score = model.predict_proba(scaler.transform(features))[0][1] * 100

        # ================= TRADE ENGINE =================

        buy_condition = (
            drop_percent < -15 and
            rsi_now < 40 and
            hour_trend
        )

        sell_condition = (
            rsi_now > 70 or
            (drop_percent > -5 and not hour_trend)
        )

        if buy_condition:
            signal = "🟢 ENTRY BUY"
        elif sell_condition:
            signal = "🔴 EXIT / SELL"
        else:
            signal = "⏳ HOLD"

        result = {
            "Coin": coin["symbol"].upper(),
            "Price": round(current_price, 6),
            "Drop %": round(drop_percent, 2),
            "RSI": round(rsi_now, 2),
            "Vol x": round(current_volume / avg_volume, 2),
            "AI Score": round(ai_score, 2),
            "Signal": signal
        }

        if ai_score >= 70:
            save_signal(result)

        return result

    except Exception as e:
        print(e)
        return None

# ==============================
# SCAN
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
        df = df.sort_values(by="AI Score", ascending=False)
        st.success(f"🔥 تم العثور على {len(df)} عملة")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("❌ مفيش فرص")

# ==============================
# HISTORY
# ==============================
if st.checkbox("📁 الصفقات المحفوظة"):
    data = load_json()
    if data:
        st.dataframe(pd.DataFrame(data))
    else:
        st.info("لا يوجد بيانات")
