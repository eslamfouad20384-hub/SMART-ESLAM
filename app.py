import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from github import Github, Auth
import json

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Scanner AI PRO MAX (With Sweep + Data Status)")

# ==============================
# GitHub setup
# ==============================
GITHUB_TOKEN = st.secrets["GITHUB"]["TOKEN"]
REPO_NAME = st.secrets["GITHUB"]["REPO"]
BRANCH = st.secrets["GITHUB"]["BRANCH"]
FILE_PATH = "data.json"

g = Github(auth=Auth.Token(GITHUB_TOKEN))
repo = g.get_repo(REPO_NAME)

# ==============================
# Load / Save
# ==============================
def load_github_data():
    try:
        file = repo.get_contents(FILE_PATH, ref=BRANCH)
        return json.loads(file.decoded_content.decode("utf-8"))
    except:
        return []

def save_github_data(data):
    content = json.dumps(data, indent=4)
    try:
        file = repo.get_contents(FILE_PATH, ref=BRANCH)
        repo.update_file(file.path, "update", content, file.sha, branch=BRANCH)
    except:
        repo.create_file(FILE_PATH, "create", content, branch=BRANCH)

# ==============================
# Indicators
# ==============================
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period).mean()
    rs = avg_gain / avg_loss
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
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper.iloc[-1], lower.iloc[-1]

def get_support_resistance(prices):
    return np.min(prices[-20:]), np.max(prices[-20:])

# ==============================
# Liquidity Sweep Detection
# ==============================
def detect_liquidity_sweep(prices, window=20):
    if len(prices) < window:
        return 0

    recent = prices[-window:]
    high = np.max(recent)
    low = np.min(recent)

    last = prices[-1]
    prev = prices[-2]

    if last > high and last < prev:
        return -1
    if last < low and last > prev:
        return 1
    return 0

# ==============================
# Data Status
# ==============================
def get_data_status(candles):
    if len(candles) < 20:
        return "⚠️ بيانات غير كافية"
    elif len(candles) < 60:
        return "🟡 بيانات متوسطة"
    else:
        return "🟢 بيانات قوية"

# ==============================
# API
# ==============================
def get_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency":"usd","order":"volume_desc","per_page":50,"page":1}
    return requests.get(url, params=params).json()

def fetch_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency":"usd","days":30}
    return requests.get(url, params=params).json()

# ==============================
# Collector
# ==============================
def run_collector():
    st.info("⏳ Updating data...")
    coins = get_coins()
    data = load_github_data()
    updated = False

    for i in range(0, len(coins), 10):
        batch = coins[i:i+10]

        def work(c):
            nonlocal data, updated
            try:
                symbol = c["symbol"].upper()
                d = fetch_data(c["id"])

                prices = [p[1] for p in d.get("prices",[])]
                vols = [v[1] for v in d.get("total_volumes",[])]

                candles = []
                for i in range(len(prices)):
                    candles.append({
                        "timestamp": int(d["prices"][i][0]),
                        "price": float(prices[i]),
                        "volume": float(vols[i]) if i < len(vols) else 0
                    })

                for row in data:
                    if row["coin"] == symbol:
                        row["candles"] = candles
                        updated = True
                        return

                data.append({"coin":symbol,"candles":candles})
                updated = True

            except:
                pass

        with ThreadPoolExecutor(max_workers=5) as ex:
            ex.map(work, batch)

        time.sleep(2)

    if updated:
        save_github_data(data)

    st.success("✅ Done")

# ==============================
# Button
# ==============================
if st.button("🔄 Update"):
    run_collector()

# ==============================
# Load data
# ==============================
data = load_github_data()
rows = []

# ==============================
# AI + Sweep + Status
# ==============================
for coin_data in data:
    coin = coin_data.get("coin")
    candles = coin_data.get("candles", [])

    if len(candles) < 25:
        continue

    prices = np.array([c["price"] for c in candles])
    vols = np.array([c["volume"] for c in candles])

    for i in range(20, len(prices)-3):

        rsi = calculate_rsi(prices[i-15:i])
        drop = ((prices[i] - prices[:i].max()) / prices[:i].max()) * 100
        volx = vols[i] / (vols[i-10:i].mean() + 1e-9)
        change = ((prices[i] - prices[i-3]) / prices[i-3]) * 100

        macd_line, signal_line = calculate_macd(prices[i-26:i]) if i>=26 else (0,0)
        upper_bb, lower_bb = calculate_bollinger(prices[i-20:i]) if i>=20 else (0,0)

        sma_short = pd.Series(prices[i-10:i]).mean()
        sma_long = pd.Series(prices[i-30:i]).mean() if i>=30 else sma_short

        support, resistance = get_support_resistance(prices[i-20:i])

        sweep = detect_liquidity_sweep(prices[i-20:i])

        score = 0
        if rsi < 35: score += 3
        if drop < -20: score += 3
        if volx > 1.5: score += 2
        if change > 0: score += 2
        if macd_line > signal_line: score += 1
        if prices[i] < lower_bb: score += 1

        rows.append({
            "rsi": rsi,
            "drop": drop,
            "volx": volx,
            "change": change,
            "macd_diff": macd_line - signal_line,
            "bb_lower_diff": prices[i] - lower_bb,
            "sma_short": sma_short,
            "sma_long": sma_long,
            "score": score,
            "sweep": sweep,
            "target": 1 if prices[i+3] > prices[i] else 0
        })

df_ai = pd.DataFrame(rows)

# ==============================
# Train AI
# ==============================
if len(df_ai) > 50:

    X = df_ai[["rsi","drop","volx","change","macd_diff","bb_lower_diff","sma_short","sma_long","score","sweep"]]
    y = df_ai["target"]

    model = RandomForestClassifier(n_estimators=200)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    model.fit(X_train,y_train)

    latest_rows = []

    for coin_data in data:
        coin = coin_data["coin"]
        candles = coin_data.get("candles", [])

        if len(candles) < 25:
            continue

        prices = np.array([c["price"] for c in candles])
        vols = np.array([c["volume"] for c in candles])

        rsi = calculate_rsi(prices[-15:])
        drop = ((prices[-1] - prices.max()) / prices.max()) * 100
        volx = vols[-1] / (vols[-10:].mean() + 1e-9)
        change = ((prices[-1] - prices[-3]) / prices[-3]) * 100

        macd_line, signal_line = calculate_macd(prices[-26:]) if len(prices)>=26 else (0,0)
        upper_bb, lower_bb = calculate_bollinger(prices[-20:]) if len(prices)>=20 else (0,0)

        sma_short = pd.Series(prices[-10:]).mean()
        sma_long = pd.Series(prices[-30:]).mean() if len(prices)>=30 else sma_short

        support, resistance = get_support_resistance(prices)

        sweep = detect_liquidity_sweep(prices[-20:])

        score = 0
        if rsi < 35: score += 3
        if drop < -20: score += 3
        if volx > 1.5: score += 2
        if change > 0: score += 2
        if macd_line > signal_line: score += 1
        if prices[-1] < lower_bb: score += 1

        if sweep == 1:
            signal = "🔥 Bullish Sweep Buy"
        elif sweep == -1:
            signal = "⚠️ Bearish Sweep Fakeout"
        elif score >= 8:
            signal = "🔥 Strong Buy"
        elif score >= 5:
            signal = "🚀 Buy"
        elif score >= 3:
            signal = "🟠 Hold"
        else:
            signal = "❌ No Trade"

        chance = model.predict_proba([[rsi,drop,volx,change,macd_line-signal_line,prices[-1]-lower_bb,sma_short,sma_long,score,sweep]])[0][1]*100

        latest_rows.append({
            "Coin": coin,
            "Price": round(prices[-1],2),
            "RSI": round(rsi,2),
            "Drop %": round(drop,2),
            "Volume x": round(volx,2),
            "Support": round(support,2),
            "Resistance": round(resistance,2),
            "Score": score,
            "Chance %": round(chance,2),
            "Signal": signal,
            "Data Status": get_data_status(candles)
        })

    df = pd.DataFrame(latest_rows)
    st.dataframe(df.sort_values("Chance %", ascending=False), use_container_width=True)

else:
    st.warning("⚠️ Not enough data")
