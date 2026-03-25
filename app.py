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
st.title("🚀 Smart Crypto Scanner AI PRO MAX (Enhanced)")

# ==============================
# 🎨 Dark Mode Table
# ==============================
st.markdown("""
<style>
[data-testid="stDataFrame"] {
    background-color: #0e1117;
    color: white;
}
thead tr th {
    background-color: #111 !important;
    color: white !important;
}
tbody tr td {
    background-color: #0e1117 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

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
# GitHub functions
# ==============================
def load_github_data():
    try:
        contents = repo.get_contents(FILE_PATH, ref=BRANCH)
        return json.loads(contents.decoded_content.decode())
    except:
        return []

def save_github_data(data):
    content = json.dumps(data, indent=4)
    try:
        contents = repo.get_contents(FILE_PATH, ref=BRANCH)
        repo.update_file(contents.path, "update", content, contents.sha, branch=BRANCH)
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
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean()
    ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line.iloc[-1], signal_line.iloc[-1]

def calculate_bollinger(prices, period=20, mult=2):
    sma = pd.Series(prices).rolling(window=period).mean()
    std = pd.Series(prices).rolling(window=period).std()
    upper = sma + mult*std
    lower = sma - mult*std
    return upper.iloc[-1], lower.iloc[-1]

def calculate_atr(prices, high, low, period=14):
    df = pd.DataFrame({"close": prices, "high": high, "low": low})
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df[['high','low','prev_close']].apply(lambda x: max(x['high']-x['low'], abs(x['high']-x['prev_close']), abs(x['low']-x['prev_close'])), axis=1)
    atr = df['tr'].rolling(period).mean()
    return atr.iloc[-1] if not atr.empty else 0

def calculate_obv(prices, volumes):
    obv = [0]
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            obv.append(obv[-1] + volumes[i])
        elif prices[i] < prices[i-1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    return obv[-1]

def get_support_resistance(prices):
    support = np.min(prices[-20:]) if len(prices) >= 1 else 0
    resistance = np.max(prices[-20:]) if len(prices) >= 1 else 0
    return support, resistance

# ==============================
# APIs
# ==============================
def get_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency":"usd","order":"volume_desc","per_page":50,"page":1}
    return requests.get(url, params=params).json()

def fetch_data(coin_id, days=30):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency":"usd","days":days}
    return requests.get(url, params=params).json()

# ==============================
# Smart update
# ==============================
def should_update(data, coin):
    for row in data:
        if row["coin"] == coin:
            candles = row.get("candles", [])
            if not candles:
                return True
            last = candles[-1]["timestamp"] / 1000
            if time.time() - last < 600:
                return False
    return True

# ==============================
# Update collector
# ==============================
def run_collector():
    st.info("⏳ تحديث آمن...")
    coins = get_coins()
    data = load_github_data()
    updated = False

    for i in range(0, len(coins), 10):
        batch = coins[i:i+10]

        def work(c):
            nonlocal data, updated
            try:
                symbol = c["symbol"].upper()
                if not should_update(data, symbol):
                    return

                d = fetch_data(c["id"])
                prices = [p[1] for p in d.get("prices",[])]
                vols = [v[1] for v in d.get("total_volumes",[])]
                highs = [p[1]*1.01 for p in d.get("prices",[])]  # تقريبي للـ High
                lows = [p[1]*0.99 for p in d.get("prices",[])]    # تقريبي للـ Low

                candles = []
                for i in range(len(prices)):
                    candles.append({
                        "timestamp": int(d["prices"][i][0]),
                        "price": float(prices[i]),
                        "high": float(highs[i]),
                        "low": float(lows[i]),
                        "volume": float(vols[i]) if i<len(vols) else 0
                    })

                for row in data:
                    if row["coin"] == symbol:
                        row["candles"] = candles
