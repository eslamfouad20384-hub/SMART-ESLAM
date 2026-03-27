import streamlit as st
import pandas as pd
import requests
import numpy as np
import json
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Scanner PRO MAX")

JSON_FILE = "signals.json"

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

# ==============================
# حفظ الإشارة
# ==============================
def save_signal(signal):
    data = load_json()
    now = datetime.now()

    for item in data:
        if item["Coin"] == signal["Coin"] and item["date"] == now.strftime("%Y-%m-%d"):
            return

    signal.update({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "status": "pending",
        "high_after": None,
        "low_after": None,
        "result": None
    })

    data.append(signal)
    save_json(data)

# ==============================
# تحديث الصفقات بعد 24 ساعة
# ==============================
def update_results():
    data = load_json()
    updated = False

    for item in data:
        if item["status"] == "done":
            continue

        signal_time = datetime.strptime(item["date"] + " " + item["time"], "%Y-%m-%d %H:%M")

        if datetime.now() - signal_time < timedelta(hours=24):
            continue

        try:
            url = f"https://api.coingecko.com/api/v3/coins/{item['Coin'].lower()}/market_chart"
            params = {"vs_currency": "usd", "days": 1}
            res = requests.get(url, params=params).json()

            prices = [p[1] for p in res["prices"]]

            high = max(prices)
            low = min(prices)

            entry = item["Price"]

            item["high_after"] = round(high, 6)
            item["low_after"] = round(low, 6)

            change = ((high - entry) / entry) * 100

            if change >= 5:
                item["result"] = "WIN"
            else:
                item["result"] = "LOSS"

            item["status"] = "done"
            updated = True

        except:
            continue

    if updated:
        save_json(data)

# ==============================
# Ranking
# ==============================
def get_ranking():
    data = load_json()
    df = pd.DataFrame(data)

    if df.empty or "result" not in df:
        return pd.DataFrame()

    df = df[df["status"] == "done"]

    ranking = df.groupby("Coin")["result"].apply(
        lambda x: (x == "WIN").sum() / len(x) * 100
    ).reset_index()

    ranking.columns = ["Coin", "Win Rate %"]
    return ranking.sort_values(by="Win Rate %", ascending=False)

# ==============================
# RSI
# ==============================
def calculate_rsi(prices):
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)

    if len(gain) < 14:
        return 50

    avg_gain = np.mean(gain[:14])
    avg_loss = np.mean(loss[:14])

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

        prices = np.array([p[1] for p in data["prices"]])
        volumes = np.array([v[1] for v in data["total_volumes"]])

        if len(prices) < 30:
            return None

        current_price = prices[-1]
        max_price = prices.max()
        drop_percent = ((current_price - max_price) / max_price) * 100

        rsi_now = calculate_rsi(prices[-15:])
        rsi_prev = calculate_rsi(prices[-16:-1])
        rsi_condition = rsi_now < 35 and rsi_now > rsi_prev

        avg_volume = volumes[:-1].mean()
        current_volume = volumes[-1]
        volume_condition = current_volume > avg_volume * 1.5

        support = np.percentile(prices[-20:], 20)
        near_support = current_price <= support * 1.1

        score = 0
        if drop_percent < -25: score += 2
        if rsi_condition: score += 2
        if volume_condition: score += 2
        if near_support: score += 2

        if score >= 8:
            st.toast(f"🔥 فرصة قوية: {coin['symbol'].upper()}")

        result = {
            "Coin": coin["symbol"].upper(),
            "Price": round(current_price, 6),
            "Drop %": round(drop_percent, 2),
            "RSI": round(rsi_now, 2),
            "Score": score
        }

        if score >= 8:
            save_signal(result)

        return result

    except:
        return None

# ==============================
# تشغيل
# ==============================
update_results()

if st.button("🔍 Scan السوق بالكامل"):

    coins = requests.get(
        "https://api.coingecko.com/api/v3/coins/markets",
        params={"vs_currency": "usd", "order": "volume_desc", "per_page": 100}
    ).json()

    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        data = list(executor.map(analyze_coin, coins))

    for r in data:
        if r:
            results.append(r)

    df = pd.DataFrame(results)

    if not df.empty:
        df = df.sort_values(by="Score", ascending=False)
        st.dataframe(df)

# ==============================
# عرض البيانات
# ==============================
if st.checkbox("📁 عرض الصفقات"):
    st.dataframe(pd.DataFrame(load_json()))

if st.checkbox("🏆 Ranking العملات"):
    st.dataframe(get_ranking())
