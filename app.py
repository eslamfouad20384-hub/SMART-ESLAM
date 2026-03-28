import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import os
import json
import base64
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from github import Github, Auth
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Scanner AI PRO MAX (With Alerts)")

# =========================
# Auto refresh
# =========================
st_autorefresh(interval=180000, key="auto_refresh")

# =========================
# Telegram
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=5)
    except Exception as e:
        print("Telegram error:", e)

# =========================
# Sound
# =========================
def play_sound():
    try:
        with open("alert.mp3", "rb") as f:
            sound_bytes = f.read()
        b64 = base64.b64encode(sound_bytes).decode()

        st.markdown(f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """, unsafe_allow_html=True)
    except Exception as e:
        print("Sound error:", e)

# =========================
# GitHub
# =========================
GITHUB_TOKEN = st.secrets["GITHUB"]["TOKEN"]
REPO_NAME = st.secrets["GITHUB"]["REPO"]
BRANCH = st.secrets["GITHUB"]["BRANCH"]
FILE_PATH = "data.json"

g = Github(auth=Auth.Token(GITHUB_TOKEN))
repo = g.get_repo(REPO_NAME)

def load_github_data():
    try:
        file = repo.get_contents(FILE_PATH, ref=BRANCH)
        return json.loads(file.decoded_content.decode("utf-8"))
    except Exception as e:
        print("Load error (maybe file not exist):", e)
        return []

def save_github_data(data):
    if not isinstance(data, list):
        print("❌ Data is not list, skipping save")
        return

    content = json.dumps(data, indent=4)

    try:
        file = repo.get_contents(FILE_PATH, ref=BRANCH)
        repo.update_file(
            file.path,
            f"update {time.time()}",
            content,
            file.sha,
            branch=BRANCH
        )
        print("✅ File updated on GitHub")

    except Exception as e:
        print("Update failed:", e)

        try:
            repo.create_file(
                FILE_PATH,
                f"create {time.time()}",
                content,
                branch=BRANCH
            )
            print("🆕 File created on GitHub")

        except Exception as e2:
            print("❌ Create failed:", e2)

# =========================
# API
# =========================
def get_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "volume_desc", "per_page": 50, "page": 1}
    return requests.get(url).json()

def fetch_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": 30}
    return requests.get(url, params=params).json()

# =========================
# Collector
# =========================
def run_collector():
    st.info("⏳ Updating data...")

    coins = get_coins()
    data = load_github_data()

    for c in coins:
        try:
            symbol = c["symbol"].upper()
            d = fetch_data(c["id"])

            prices = [p[1] for p in d.get("prices", [])]
            vols = [v[1] for v in d.get("total_volumes", [])]

            if len(prices) == 0:
                continue

            candles = []
            for i in range(len(prices)):
                candles.append({
                    "timestamp": int(d["prices"][i][0]),
                    "price": float(prices[i]),
                    "volume": float(vols[i]) if i < len(vols) else 0
                })

            found = False
            for row in data:
                if row.get("coin") == symbol:
                    row["candles"] = candles
                    found = True
                    break

            if not found:
                data.append({"coin": symbol, "candles": candles})

        except Exception as e:
            print(f"Coin error {c.get('id')}:", e)

    save_github_data(data)
    st.success("✅ Done")

if st.button("🔄 Update"):
    run_collector()

# =========================
# Load
# =========================
data = load_github_data()
rows = []

# =========================
# Indicators
# =========================
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs)).iloc[-1]

def calculate_macd(prices):
    ema_fast = pd.Series(prices).ewm(span=12).mean()
    ema_slow = pd.Series(prices).ewm(span=26).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9).mean()
    return macd.iloc[-1], signal.iloc[-1]

def calculate_bollinger(prices):
    sma = pd.Series(prices).rolling(20).mean()
    std = pd.Series(prices).rolling(20).std()
    return sma + 2 * std, sma - 2 * std

def get_support_resistance(prices):
    return np.min(prices[-20:]), np.max(prices[-20:])

def detect_liquidity_sweep(prices, window=20):
    if len(prices) < window:
        return 0
    recent = prices[-window:]
    high = np.max(recent)
    low = np.min(recent)
    last = prices[-1]

    if last > high:
        return -1
    if last < low:
        return 1
    return 0

# =========================
# AI dataset
# =========================
for coin_data in data:
    candles = coin_data.get("candles", [])
    if len(candles) < 25:
        continue

    prices = np.array([c["price"] for c in candles])
    vols = np.array([c["volume"] for c in candles])

    for i in range(30, len(prices) - 3):
        try:
            rsi = calculate_rsi(prices[i-15:i])
            drop = ((prices[i] - prices[:i].max()) / prices[:i].max()) * 100
            volx = vols[i] / (vols[i-10:i].mean() + 1e-9)
            change = ((prices[i] - prices[i-3]) / prices[i-3]) * 100

            macd_line, signal_line = calculate_macd(prices[i-26:i])
            upper_bb, lower_bb = calculate_bollinger(prices[i-20:i])

            sma_short = pd.Series(prices[i-10:i]).mean()
            sma_long = pd.Series(prices[i-30:i]).mean()

            support, resistance = get_support_resistance(prices[i-20:i])
            sweep = detect_liquidity_sweep(prices[i-20:i])

            score = 0
            if rsi < 35: score += 3
            if drop < -20: score += 3
            if volx > 1.5: score += 2
            if change > 0: score += 2
            if macd_line > signal_line: score += 1

            rows.append({
                "rsi": rsi,
                "drop": drop,
                "volx": volx,
                "change": change,
                "macd_diff": macd_line - signal_line,
                "bb_lower_diff": prices[i] - lower_bb.iloc[-1],
                "sma_short": sma_short,
                "sma_long": sma_long,
                "score": score,
                "sweep": sweep,
                "target": 1 if prices[i+3] > prices[i] else 0
            })

        except Exception as e:
            print("Row error:", e)

df_ai = pd.DataFrame(rows)

# =========================
# MODEL
# =========================
if len(df_ai) > 50:
    X = df_ai[[
        "rsi","drop","volx","change",
        "macd_diff","bb_lower_diff",
        "sma_short","sma_long",
        "score","sweep"
    ]]
    y = df_ai["target"]

    model = RandomForestClassifier(n_estimators=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)

    latest_rows = []

    for coin_data in data:
        coin = coin_data["coin"]
        candles = coin_data.get("candles", [])

        if len(candles) < 30:
            continue

        prices = np.array([c["price"] for c in candles])
        vols = np.array([c["volume"] for c in candles])

        rsi = calculate_rsi(prices[-15:])
        drop = ((prices[-1] - prices.max()) / prices.max()) * 100
        volx = vols[-1] / (vols[-10:].mean() + 1e-9)
        change = ((prices[-1] - prices[-3]) / prices[-3]) * 100

        macd_line, signal_line = calculate_macd(prices[-26:])
        upper_bb, lower_bb = calculate_bollinger(prices[-20:])

        sma_short = pd.Series(prices[-10:]).mean()
        sma_long = pd.Series(prices[-30:]).mean()

        support, resistance = get_support_resistance(prices)
        sweep = detect_liquidity_sweep(prices[-20:])

        score = 0
        if rsi < 35: score += 3
        if drop < -20: score += 3
        if volx > 1.5: score += 2
        if change > 0: score += 2

        signal = "Hold"
        if score >= 8:
            signal = "Strong Buy"
        elif score >= 5:
            signal = "Buy"

        chance = model.predict_proba([[
            rsi, drop, volx, change,
            macd_line - signal_line,
            prices[-1] - lower_bb.iloc[-1],
            sma_short, sma_long,
            score, sweep
        ]])[0][1] * 100

        latest_rows.append({
            "Coin": coin,
            "Price": round(prices[-1], 2),
            "RSI": round(rsi, 2),
            "Score": score,
            "Chance %": round(chance, 2),
            "Signal": signal
        })

    df = pd.DataFrame(latest_rows)
    st.dataframe(df.sort_values("Chance %", ascending=False), use_container_width=True)

else:
    st.warning("⚠️ Not enough data")
