import streamlit as st
import pandas as pd
import requests
import numpy as np
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Reversal Scanner PRO + JSON")

# ==============================
# إعدادات
# ==============================
MIN_VOLUME = 10_000_000
RSI_PERIOD = 14
JSON_FILE = "signals.json"

# ==============================
# تحميل JSON
# ==============================
def load_json():
    if not os.path.exists(JSON_FILE):
        return []
    with open(JSON_FILE, "r") as f:
        return json.load(f)

def save_json(data):
    with open(JSON_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ==============================
# حفظ الإشارة
# ==============================
def save_signal(signal):
    data = load_json()

    # منع التكرار (نفس العملة في نفس اليوم)
    today = datetime.now().strftime("%Y-%m-%d")
    for item in data:
        if item["Coin"] == signal["Coin"] and item["date"] == today:
            return

    signal["date"] = today
    signal["time"] = datetime.now().strftime("%H:%M")

    data.append(signal)
    save_json(data)

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
# RSI
# ==============================
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)

    if len(gain) < period:
        return 50

    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ==============================
# تحليل العملة
# ==============================
def analyze_coin(coin):
    try:
        coin_id = coin["id"]

        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": 30}
        data = requests.get(url, params=params).json()

        if "prices" not in data:
            return None

        prices = np.array([p[1] for p in data["prices"]])
        volumes = np.array([v[1] for v in data["total_volumes"]])

        if len(prices) < 30:
            return None

        current_price = prices[-1]
        max_price = prices.max()

        drop_percent = ((current_price - max_price) / max_price) * 100

        # RSI
        rsi_now = calculate_rsi(prices[-15:])
        rsi_prev = calculate_rsi(prices[-16:-1])
        rsi_condition = rsi_now < 35 and rsi_now > rsi_prev

        # Volume
        avg_volume = volumes[:-1].mean()
        current_volume = volumes[-1]
        volume_condition = current_volume > avg_volume * 1.5

        # Support
        recent_prices = prices[-20:]
        support_zone = np.percentile(recent_prices, 20)
        near_support = current_price <= support_zone * 1.1

        # Trend
        ema_20 = pd.Series(prices).ewm(span=20).mean().values
        ema_condition = current_price > ema_20[-1]

        recent_lows = prices[-5:]
        higher_low = recent_lows[-1] > recent_lows.min()

        trend_condition = ema_condition or higher_low

        # Candle
        last = prices[-1]
        prev = prices[-2]
        prev2 = prices[-3]

        bullish_engulfing = (prev < prev2) and (last > prev) and (last > prev2)
        body = abs(last - prev)
        lower_shadow = abs(prev - prev2)
        hammer = lower_shadow > body * 2

        candle_signal = bullish_engulfing or hammer

        # Smart reversal
        strong_drop = drop_percent < -40
        bounce_started = current_price > prices[-3]
        smart_reversal = strong_drop and bounce_started

        # Score
        score = 0
        if drop_percent < -25: score += 2
        if rsi_condition: score += 2
        if volume_condition: score += 2
        if near_support: score += 2
        if trend_condition: score += 2
        if candle_signal: score += 2
        if smart_reversal: score += 2

        probability = min(score * 8, 95)

        if score >= 10:
            signal = "🚀 STRONG BUY"
        elif score >= 8:
            signal = "🔥 BUY"
        elif score >= 6:
            signal = "⏳ EARLY"
        else:
            signal = "❌ NO"

        result = {
            "Coin": coin["symbol"].upper(),
            "Price": round(current_price, 6),
            "Drop %": round(drop_percent, 2),
            "RSI": round(rsi_now, 2),
            "Vol x": round(current_volume / avg_volume, 2),
            "Support": round(support_zone, 6),
            "Score": score,
            "Chance %": probability,
            "Signal": signal
        }

        # ==============================
        # حفظ فقط الفرص القوية
        # ==============================
        if score >= 8:
            save_signal(result)

        return result

    except:
        return None

# ==============================
# تشغيل الفحص
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
        df = df[df["Score"] >= 6]

        st.success(f"🔥 تم العثور على {len(df)} فرصة قوية")
        st.dataframe(df, use_container_width=True)

    else:
        st.warning("❌ مفيش فرص حالياً")

# ==============================
# عرض البيانات المحفوظة
# ==============================
if st.checkbox("📁 عرض الصفقات المحفوظة"):
    data = load_json()
    if data:
        st.dataframe(pd.DataFrame(data))
    else:
        st.info("لسه مفيش بيانات محفوظة")
