import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import os
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from github import Github, Auth
import json
from streamlit_autorefresh import st_autorefresh
import base64

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Scanner AI PRO MAX (Optimized)")

# 🔄 Auto refresh (UI فقط)
st_autorefresh(interval=180000, key="auto_refresh")

# ==============================
# 🔔 Telegram
# ==============================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=5)
    except:
        pass


# ==============================
# 🔊 Sound
# ==============================
def play_sound():
    try:
        with open("alert.mp3", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        st.markdown(f"""
        <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}">
        </audio>
        """, unsafe_allow_html=True)
    except:
        pass


# ==============================
# GitHub
# ==============================
GITHUB_TOKEN = st.secrets["GITHUB"]["TOKEN"]
REPO_NAME = st.secrets["GITHUB"]["REPO"]
BRANCH = st.secrets["GITHUB"]["BRANCH"]

FILE_PATH = "data.json"
ALERT_FILE = "alerts.json"

g = Github(auth=Auth.Token(GITHUB_TOKEN))
repo = g.get_repo(REPO_NAME)


def load_file(path):
    try:
        file = repo.get_contents(path, ref=BRANCH)
        return json.loads(file.decoded_content.decode("utf-8"))
    except:
        return []


def save_file(path, data):
    content = json.dumps(data, indent=4)
    try:
        file = repo.get_contents(path, ref=BRANCH)
        repo.update_file(file.path, "update", content, file.sha, branch=BRANCH)
    except:
        repo.create_file(path, "create", content, branch=BRANCH)


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


def calculate_macd(prices):
    series = pd.Series(prices)

    if len(series) < 26:
        return 0, 0

    ema_fast = series.ewm(span=12).mean()
    ema_slow = series.ewm(span=26).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9).mean()

    if len(macd) == 0:
        return 0, 0

    return macd.iloc[-1], signal.iloc[-1]


def calculate_bollinger(prices):
    s = pd.Series(prices)
    if len(s) < 20:
        return 0, 0

    sma = s.rolling(20).mean()
    std = s.rolling(20).std()
    return (sma + 2*std).iloc[-1], (sma - 2*std).iloc[-1]


# ==============================
# API
# ==============================
@st.cache_data(ttl=300)
def get_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    return requests.get(url, params={"vs_currency":"usd","order":"volume_desc","per_page":30}).json()


@st.cache_data(ttl=300)
def fetch_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    return requests.get(url, params={"vs_currency":"usd","days":30}).json()


# ==============================
# COLLECTOR (NO DUPLICATES)
# ==============================
def run_collector():
    coins = get_coins()
    raw = load_file(FILE_PATH)

    if isinstance(raw, dict):
        old_data = raw.get("data", [])
    else:
        old_data = raw

    for c in coins:
        try:
            symbol = c["symbol"].upper()
            d = fetch_data(c["id"])

            prices = [p[1] for p in d.get("prices", [])]
            vols = [v[1] for v in d.get("total_volumes", [])]

            candles = [
                {
                    "price": float(prices[i]),
                    "volume": float(vols[i]) if i < len(vols) else 0
                }
                for i in range(len(prices))
            ]

            found = False

            for row in old_data:
                if row["coin"] == symbol:
                    row["candles"] = candles
                    found = True
                    break

            if not found:
                old_data.append({
                    "coin": symbol,
                    "candles": candles
                })

        except:
            pass

    payload = {
        "last_update": time.time(),
        "data": old_data
    }

    save_file(FILE_PATH, payload)
    st.success("✅ Data Updated")


if st.button("🔄 Update Now"):
    run_collector()


# ==============================
# LOAD DATA
# ==============================
raw = load_file(FILE_PATH)

if isinstance(raw, dict):
    data = raw.get("data", [])
    last_update = raw.get("last_update", 0)
else:
    data = raw
    last_update = 0

alerts_sent = load_file(ALERT_FILE)

# ==============================
# AUTO UPDATE (15 min)
# ==============================
UPDATE_INTERVAL = 900  # 15 minutes
current_time = time.time()

if current_time - last_update > UPDATE_INTERVAL:
    st.info("⏳ Auto updating data...")
    run_collector()


# ==============================
# AI SCAN
# ==============================
rows = []

for coin_data in data:
    candles = coin_data.get("candles", [])
    if len(candles) < 30:
        continue

    prices = np.array([c["price"] for c in candles])
    vols = np.array([c["volume"] for c in candles])

    for i in range(25, len(prices)-3):

        rsi = calculate_rsi(prices[i-14:i])
        drop = ((prices[i] - np.max(prices[i-20:i])) / np.max(prices[i-20:i])) * 100
        volx = vols[i] / (np.mean(vols[i-10:i]) + 1e-9)
        change = ((prices[i] - prices[i-3]) / prices[i-3]) * 100

        macd_line, signal_line = calculate_macd(prices[i-26:i])
        upper, lower = calculate_bollinger(prices[i-20:i])

        rows.append({
            "rsi": rsi,
            "drop": drop,
            "volx": volx,
            "change": change,
            "macd": macd_line - signal_line,
            "bb": prices[i] - lower,
            "target": 1 if prices[i+3] > prices[i] else 0
        })

df_ai = pd.DataFrame(rows)


# ==============================
# MODEL
# ==============================
if len(df_ai) > 50:

    X = df_ai.drop("target", axis=1)
    y = df_ai["target"]

    model = RandomForestClassifier(n_estimators=150)
    model.fit(X, y)

    latest_rows = []

    for coin_data in data:
        coin = coin_data["coin"]
        candles = coin_data.get("candles", [])

        if len(candles) < 30:
            continue

        prices = np.array([c["price"] for c in candles])
        vols = np.array([c["volume"] for c in candles])

        rsi = calculate_rsi(prices[-14:])
        drop = ((prices[-1] - np.max(prices[-20:])) / np.max(prices[-20:])) * 100
        volx = vols[-1] / (np.mean(vols[-10:]) + 1e-9)
        change = ((prices[-1] - prices[-3]) / prices[-3]) * 100

        macd_line, signal_line = calculate_macd(prices[-26:])
        upper, lower = calculate_bollinger(prices[-20:])

        features = [[
            rsi, drop, volx, change,
            macd_line - signal_line,
            prices[-1] - lower
        ]]

        chance = model.predict_proba(features)[0][1] * 100

        signal = "❌ No Trade"
        if chance > 70:
            signal = "🔥 Strong Buy"
        elif chance > 55:
            signal = "🚀 Buy"

        key = f"{coin}_{signal}"

        if signal != "❌ No Trade" and key not in alerts_sent:
            msg = f"🚀 {signal}\n{coin}\nPrice: {prices[-1]:.4f}\nChance: {chance:.2f}%"
            send_telegram(msg)
            play_sound()
            alerts_sent.append(key)

    save_file(ALERT_FILE, alerts_sent)

    df = pd.DataFrame(latest_rows)
    st.dataframe(df.sort_values("Chance %", ascending=False), use_container_width=True)

else:
    st.warning("⚠️ Not enough data")
