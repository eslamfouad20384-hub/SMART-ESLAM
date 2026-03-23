import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Scanner AI (ملف واحد)")

DB_NAME = "crypto.db"

# ==============================
# Database functions
# ==============================
def connect():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def create_tables():
    conn = connect()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS market_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        coin TEXT,
        timestamp INTEGER,
        price REAL,
        volume REAL,
        rsi REAL,
        score REAL
    )
    """)
    conn.commit()
    conn.close()

create_tables()

# ==============================
# RSI function
# ==============================
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ==============================
# Collector (fetch & store)
# ==============================
def get_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency":"usd","order":"volume_desc","per_page":50,"page":1}
    return requests.get(url, params=params).json()

def fetch_coin_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": 30}
    return requests.get(url, params=params).json()

def calculate_score(price, max_price, rsi, volume, avg_volume):
    score = 0
    drop = ((price - max_price)/max_price)*100
    if drop < -25: score+=2
    if rsi < 35: score+=2
    if volume > avg_volume*1.5: score+=2
    return score

def save_row(coin, timestamp, price, volume, rsi, score):
    conn = connect()
    c = conn.cursor()
    c.execute("""
    INSERT INTO market_data (coin, timestamp, price, volume, rsi, score)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (coin, timestamp, price, volume, rsi, score))
    conn.commit()
    conn.close()

def run_collector_once():
    conn = connect()
    df_check = pd.read_sql("SELECT * FROM market_data", conn)
    conn.close()
    if df_check.empty:
        st.info("⏳ لا توجد بيانات، البرنامج بيجيب البيانات لأول مرة…")
        coins = get_coins()
        results = []

        def analyze_coin(coin):
            try:
                coin_id = coin["id"]
                data = fetch_coin_data(coin_id)
                prices = np.array([p[1] for p in data["prices"]])
                volumes = np.array([v[1] for v in data["total_volumes"]])
                if len(prices) < 30: return None
                current_price = prices[-1]
                max_price = prices.max()
                rsi = calculate_rsi(prices[-15:])
                avg_volume = volumes[:-1].mean()
                current_volume = volumes[-1]
                score = calculate_score(current_price, max_price, rsi, current_volume, avg_volume)
                timestamp = int(data["prices"][-1][0])
                save_row(coin["symbol"].upper(), timestamp, current_price, current_volume, rsi, score)
                return coin["symbol"].upper()
            except:
                return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            updated = list(executor.map(analyze_coin, coins))
        st.success(f"✅ تم تحديث {len([u for u in updated if u])} عملة")

run_collector_once()

# ==============================
# Read DB & Train AI
# ==============================
conn = connect()
df = pd.read_sql("SELECT * FROM market_data", conn)
conn.close()

if df.empty:
    st.warning("❌ لا توجد بيانات بعد جمعها…")
else:
    # Prepare target
    df["target"] = (df["price"].shift(-3) > df["price"]).astype(int)
    df = df.dropna()
    X = df[["rsi","score"]]
    y = df["target"]
    # Train model
    model = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    # Predict latest
    latest = df.sort_values("timestamp").groupby("coin").tail(1)
    X_latest = latest[["rsi","score"]]
    probs = model.predict_proba(X_latest)[:,1]
    latest["Chance %"] = probs*100
    latest = latest.sort_values("Chance %", ascending=False)
    st.success(f"دقة الموديل: {round(acc*100,2)}%")
    st.dataframe(latest[["coin","price","rsi","score","Chance %"]], use_container_width=True)
