import streamlit as st
import pandas as pd
import requests
import numpy as np
import json
import os
import joblib
import datetime
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from github import Github

# ==============================
# SETUP
# ==============================
st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto AI PRO FULL SYSTEM")

DATA_FILE = "data.json"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# DATA HANDLING
# ==============================
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ==============================
# COINS API
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
# STRATEGY ENGINE
# ==============================
def choose_strategy(rsi_val, drop):
    if rsi_val < 30 and drop < -20:
        return "AGGRESSIVE"
    elif rsi_val < 40:
        return "SAFE"
    else:
        return "TREND"

# ==============================
# MODEL PER COIN
# ==============================
def model_path(symbol):
    return f"{MODEL_DIR}/{symbol}.pkl"

def train_model(symbol, df):
    if len(df) < 20:
        return

    X = df[["rsi", "drop", "ema_trend"]]
    y = df["result"]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    joblib.dump(model, model_path(symbol))

def predict(symbol, rsi_val, drop, ema_trend):
    path = model_path(symbol)

    if not os.path.exists(path):
        return "NO_MODEL"

    model = joblib.load(path)
    return model.predict([[rsi_val, drop, ema_trend]])[0]

# ==============================
# LOGGING
# ==============================
def log_trade(symbol, rsi_val, drop, ema_trend, signal, result):
    data = load_data()

    data.append({
        "time": str(datetime.datetime.now()),
        "coin": symbol,
        "rsi": rsi_val,
        "drop": drop,
        "ema_trend": ema_trend,
        "signal": signal,
        "result": result
    })

    save_data(data)

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

        strategy = choose_strategy(rsi_val, drop)

        signal = predict(coin["symbol"], rsi_val, drop, ema_trend)

        if signal == "NO_MODEL":
            signal = "WAIT"

        # log fake result placeholder (later for learning)
        log_trade(coin["symbol"], rsi_val, drop, ema_trend, signal, "PENDING")

        return {
            "Coin": coin["symbol"].upper(),
            "Price": round(price, 6),
            "Drop %": round(drop, 2),
            "RSI": round(rsi_val, 2),
            "EMA Trend": ema_trend,
            "Strategy": strategy,
            "AI Signal": signal
        }

    except:
        return None

# ==============================
# DAILY TRAINING
# ==============================
def auto_train():
    data = load_data()
    df = pd.DataFrame(data)

    if len(df) > 50:
        for coin in df["coin"].unique():
            coin_df = df[df["coin"] == coin]
            train_model(coin, coin_df)

# ==============================
# GITHUB UPLOAD (OPTIONAL)
# ==============================
def upload_to_github(token, repo_name):
    g = Github(token)
    repo = g.get_user().get_repo(repo_name)

    with open(DATA_FILE, "r") as f:
        content = f.read()

    try:
        contents = repo.get_contents(DATA_FILE)
        repo.update_file(DATA_FILE, "update data", content, contents.sha)
    except:
        repo.create_file(DATA_FILE, "init data", content)

# ==============================
# RUN
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

    # Auto training after scan
    auto_train()
