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
st.title("🚀 Smart Crypto Scanner AI PRO MAX V3")

# ==============================
# GitHub
# ==============================
GITHUB_TOKEN = st.secrets["GITHUB"]["TOKEN"]
REPO_NAME = st.secrets["GITHUB"]["REPO"]
BRANCH = st.secrets["GITHUB"]["BRANCH"]
FILE_PATH = "data.json"

g = Github(auth=Auth.Token(GITHUB_TOKEN))
repo = g.get_repo(REPO_NAME)

def load_data():
    try:
        file = repo.get_contents(FILE_PATH, ref=BRANCH)
        return json.loads(file.decoded_content.decode())
    except:
        return []

def save_data(data):
    content = json.dumps(data, indent=2)
    try:
        file = repo.get_contents(FILE_PATH, ref=BRANCH)
        repo.update_file(file.path, "update", content, file.sha, branch=BRANCH)
    except:
        repo.create_file(FILE_PATH, "create", content, branch=BRANCH)

# ==============================
# Indicators (Fixed)
# ==============================
def rsi(prices, period=14):
    if len(prices) < period+1:
        return 50
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(period).mean().iloc[-1]
    avg_loss = pd.Series(loss).rolling(period).mean().iloc[-1]

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(prices):
    if len(prices) < 26:
        return 0, 0

    ema12 = pd.Series(prices).ewm(span=12).mean()
    ema26 = pd.Series(prices).ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9).mean()

    return macd_line.iloc[-1], signal.iloc[-1]

def bollinger(prices):
    if len(prices) < 20:
        return 0, 0

    sma = pd.Series(prices).rolling(20).mean()
    std = pd.Series(prices).rolling(20).std()

    upper = sma + 2*std
    lower = sma - 2*std

    return upper.iloc[-1], lower.iloc[-1]

def support_resistance(prices):
    if len(prices) < 20:
        return prices.min(), prices.max()
    return prices[-20:].min(), prices[-20:].max()

# ==============================
# Multi API (KuCoin + Bybit + OKX)
# ==============================
def kucoin(symbol):
    try:
        url = "https://api.kucoin.com/api/v1/market/candles"
        r = requests.get(url, params={"symbol": f"{symbol}-USDT", "type": "1min"}).json()
        data = r.get("data", [])[:20]

        return [{"t": int(i[0])*1000, "p": float(i[2]), "v": float(i[5])} for i in data]
    except:
        return []

def bybit(symbol):
    try:
        url = "https://api.bybit.com/v5/market/kline"
        r = requests.get(url, params={"symbol": f"{symbol}USDT", "interval": "1", "limit": 20}).json()
        data = r.get("result", {}).get("list", [])

        return [{"t": int(i[0]), "p": float(i[4]), "v": float(i[5])} for i in data]
    except:
        return []

def okx(symbol):
    try:
        url = "https://www.okx.com/api/v5/market/candles"
        r = requests.get(url, params={"instId": f"{symbol}-USDT", "bar": "1m", "limit": 20}).json()
        data = r.get("data", [])

        return [{"t": int(i[0]), "p": float(i[4]), "v": float(i[5])} for i in data]
    except:
        return []

def get_candles(symbol):
    all_data = []
    for f in [kucoin, bybit, okx]:
        all_data += f(symbol)

    uniq = {x["t"]: x for x in all_data}
    return sorted(uniq.values(), key=lambda x: x["t"])

# ==============================
# Pump Detector
# ==============================
def pump_detector(prices, vols):
    try:
        if len(prices) < 10:
            return 0, "NO", 0, 0

        price_change = ((prices[-1] - prices[-5]) / prices[-5]) * 100 if prices[-5] != 0 else 0
        volx = vols[-1] / (np.mean(vols[-10:]) + 1e-9)

        r = rsi(prices)
        m, s = macd(prices)

        score = 0
        if volx > 2: score += 4
        if price_change < 3: score += 2
        if r < 45: score += 2
        if m > s: score += 1

        if score >= 7:
            signal = "💣 Pump Soon"
        elif score >= 5:
            signal = "🔥 Accumulation"
        elif score >= 3:
            signal = "👀 Watch"
        else:
            signal = "NO"

        return score, signal, volx, price_change
    except:
        return 0, "NO", 0, 0

# ==============================
# Coins
# ==============================
def get_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    return requests.get(url, params={"vs_currency":"usd","order":"volume_desc","per_page":40}).json()

# ==============================
# Collector
# ==============================
def update():
    coins = get_coins()
    data = load_data()
    updated = False

    def job(c):
        nonlocal data, updated

        symbol = c["symbol"].upper()
        candles = get_candles(symbol)

        if len(candles) == 0:
            return

        for row in data:
            if row["coin"] == symbol:
                old = row.get("candles", [])
                merged = old + candles
                row["candles"] = list({x["t"]: x for x in merged}.values())
                updated = True
                return

        data.append({"coin": symbol, "candles": candles})
        updated = True

    with ThreadPoolExecutor(max_workers=5) as ex:
        ex.map(job, coins)

    if updated:
        save_data(data)

    st.success("✅ Updated")

if st.button("🔄 Update"):
    update()

# ==============================
# Load
# ==============================
data = load_data()

rows_ai = []
pump_rows = []

# ==============================
# AI Dataset
# ==============================
for c in data:
    candles = c["candles"]
    if len(candles) < 10:
        continue

    prices = np.array([x["p"] for x in candles])
    vols = np.array([x["v"] for x in candles])

    for i in range(15, len(prices)-3):
        r = rsi(prices[i-14:i])
        m, s = macd(prices[i-26:i])
        u, l = bollinger(prices[i-20:i])

        volx = vols[i] / (np.mean(vols[i-10:i]) + 1e-9)
        drop = (prices[i] - np.max(prices[:i])) / np.max(prices[:i]) * 100
        change = ((prices[i] - prices[i-3]) / prices[i-3]) * 100

        score = 0
        if r < 40: score += 2
        if volx > 1.5: score += 2
        if change > 0: score += 2
        if m > s: score += 1

        target = 1 if prices[i+3] > prices[i] else 0

        rows_ai.append([r, drop, volx, change, m-s, prices[i]-l, score, target])

df_ai = pd.DataFrame(rows_ai)

# ==============================
# Train AI
# ==============================
if len(df_ai) > 30:
    X = df_ai.iloc[:, :-1]
    y = df_ai.iloc[:, -1]

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)

    results = []

    for c in data:
        candles = c["candles"]
        if len(candles) < 10:
            continue

        prices = np.array([x["p"] for x in candles])
        vols = np.array([x["v"] for x in candles])

        r = rsi(prices)
        m, s = macd(prices)
        u, l = bollinger(prices)
        sup, res = support_resistance(prices)

        volx = vols[-1] / (np.mean(vols[-10:]) + 1e-9)
        drop = (prices[-1] - np.max(prices)) / np.max(prices) * 100

        pump_score, pump_signal, vx, pc = pump_detector(prices, vols)

        pred = model.predict_proba([[r, drop, volx, pc, m-s, prices[-1]-l, pump_score]])[0][1] * 100

        results.append({
            "Coin": c["coin"],
            "Price": round(prices[-1], 4),
            "RSI": round(r, 2),
            "VolX": round(volx, 2),
            "Pump": pump_signal,
            "PumpScore": pump_score,
            "Chance%": round(pred, 2),
            "Support": round(sup, 4),
            "Resistance": round(res, 4)
        })

    df = pd.DataFrame(results).sort_values("Chance%", ascending=False)
    st.dataframe(df, use_container_width=True)

else:
    st.warning("⚠️ Not enough data")
