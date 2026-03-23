import streamlit as st
import pandas as pd
import numpy as np
import requests
import sqlite3
import os
from concurrent.futures import ThreadPoolExecutor

# ==============================
# إعدادات
# ==============================
DB_NAME = "crypto.db"
MIN_VOLUME = 10_000_000
RSI_PERIOD = 14

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Scanner PRO (Collector مدمج)")

# ==============================
# إنشاء قاعدة البيانات + الجدول
# ==============================
def create_tables():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS market_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        coin TEXT,
        price REAL,
        volume REAL,
        rsi REAL,
        score REAL,
        timestamp INTEGER
    )
    """)
    conn.commit()
    conn.close()

create_tables()

# ==============================
# دالة لحساب RSI
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
# جلب العملات من CoinGecko
# ==============================
def get_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency":"usd","order":"volume_desc","per_page":100,"page":1}
    return requests.get(url, params=params).json()

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
        if len(prices) < 30: return None
        current_price = prices[-1]
        max_price = prices.max()
        drop_percent = ((current_price - max_price)/max_price)*100
        rsi_now = calculate_rsi(prices[-15:])
        score = 0
        if drop_percent < -25: score+=2
        if rsi_now < 35: score+=2
        probability = min(score*8,95)
        return {
            "coin": coin["symbol"].upper(),
            "price": round(current_price,6),
            "rsi": round(rsi_now,2),
            "score": score,
            "Chance %": probability
        }
    except: return None

# ==============================
# دالة لملء قاعدة البيانات إذا فاضية
# ==============================
def fill_database_if_empty():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    df_check = pd.read_sql("SELECT * FROM market_data", conn)
    if df_check.empty:
        st.info("⏳ لا توجد بيانات، البرنامج بيجيب البيانات من الإنترنت لأول مرة…")
        coins = get_coins()
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            data = list(executor.map(analyze_coin, coins))
        for r in data:
            if r: results.append(r)
        # حفظ البيانات في SQLite
        for r in results:
            conn.execute("""
            INSERT INTO market_data (coin, price, rsi, score, timestamp) 
            VALUES (?, ?, ?, ?, strftime('%s','now'))
            """, (r["coin"], r["price"], r["rsi"], r["score"]))
        conn.commit()
    conn.close()

# ==============================
# تشغيل Collector مدمج
# ==============================
fill_database_if_empty()

# ==============================
# قراءة البيانات وعرضها
# ==============================
conn = sqlite3.connect(DB_NAME, check_same_thread=False)
df = pd.read_sql("SELECT * FROM market_data", conn)
conn.close()

if df.empty:
    st.warning("❌ لا توجد بيانات حالياً")
else:
    st.success(f"تم العثور على {len(df)} عملة")
    st.dataframe(df[["coin","price","rsi","score","Chance %"]], use_container_width=True)
