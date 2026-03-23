import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Scanner AI + Signals + Data Status")

# ==============================
# Data folder setup
# ==============================
DB_FOLDER = "data"
DB_PATH = os.path.join(DB_FOLDER, "crypto.db")
CSV_PATH = os.path.join(DB_FOLDER, "crypto_backup.csv")

# إنشاء فولدر البيانات بأمان
if not os.path.exists(DB_FOLDER):
    try:
        os.makedirs(DB_FOLDER)
    except Exception as e:
        st.warning(f"⚠️ لم يتم إنشاء فولدر البيانات: {e}")

# ==============================
# Database functions
# ==============================
def connect():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        return conn
    except Exception as e:
        st.error(f"❌ فشل الاتصال بقاعدة البيانات: {e}")
        raise

def create_tables():
    try:
        conn = connect()
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT,
            timestamp INTEGER,
            price REAL,
            drop_percent REAL,
            rsi REAL,
            volume REAL,
            volx REAL,
            support REAL,
            score REAL
        )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"❌ فشل إنشاء الجداول: {e}")

create_tables()

# ==============================
# RSI calculation
# ==============================
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    if avg_loss == 0: return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1+rs))

# ==============================
# Collector functions
# ==============================
def get_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency":"usd","order":"volume_desc","per_page":50,"page":1}
    return requests.get(url, params=params).json()

def fetch_coin_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": 30}
    return requests.get(url, params=params).json()

def calculate_score(price, drop, rsi, vol, volx):
    score = 0
    if drop < -25: score +=2
    if rsi < 35: score +=2
    if volx > 1.5: score +=2
    return score

def save_row(coin, timestamp, price, drop, rsi, volume, volx, support, score):
    conn = connect()
    c = conn.cursor()
    c.execute("""
    INSERT INTO market_data (coin, timestamp, price, drop_percent, rsi, volume, volx, support, score)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (coin, timestamp, price, drop, rsi, volume, volx, support, score))
    conn.commit()
    conn.close()

    # Backup to CSV
    row_df = pd.DataFrame([{
        "coin": coin,
        "timestamp": timestamp,
        "price": price,
        "drop_percent": drop,
        "rsi": rsi,
        "volume": volume,
        "volx": volx,
        "support": support,
        "score": score
    }])
    if not os.path.exists(CSV_PATH):
        row_df.to_csv(CSV_PATH, index=False)
    else:
        row_df.to_csv(CSV_PATH, mode='a', header=False, index=False)

def run_collector():
    st.info("⏳ جاري تحديث البيانات…")
    coins = get_coins()
    updated_count = 0

    def analyze_coin(coin):
        try:
            coin_id = coin["id"]
            data = fetch_coin_data(coin_id)
            prices = np.array([p[1] for p in data["prices"]])
            volumes = np.array([v[1] for v in data["total_volumes"]])
            if len(prices) < 30: return None
            current_price = prices[-1]
            max_price = prices.max()
            drop = ((current_price - max_price)/max_price)*100
            rsi = calculate_rsi(prices[-15:])
            avg_vol = volumes[:-1].mean()
            volx = volumes[-1]/avg_vol
            support = np.percentile(prices[-20:],20)
            score = calculate_score(current_price, drop, rsi, volumes[-1], volx)
            timestamp = int(data["prices"][-1][0])
            save_row(coin["symbol"].upper(), timestamp, current_price, drop, rsi, volumes[-1], volx, support, score)
            return coin["symbol"].upper()
        except:
            return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        updated = list(executor.map(analyze_coin, coins))
    updated_count = len([u for u in updated if u])
    st.success(f"✅ تم تحديث {updated_count} عملة")

if st.button("🔄 تحديث البيانات"):
    run_collector()

# ==============================
# Read DB & AI
# ==============================
conn = connect()
df = pd.read_sql("SELECT * FROM market_data", conn)
conn.close()

if df.empty:
    st.warning("❌ لا توجد بيانات بعد جمعها…")
else:
    # Train AI
    df["target"] = (df["price"].shift(-3) > df["price"]).astype(int)
    df_ai = df.dropna()
    X = df_ai[["rsi","score"]]
    y = df_ai["target"]
    model = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    model.fit(X_train,y_train)
    acc = model.score(X_test,y_test)

    # Latest rows + Chance %
    latest = df.sort_values("timestamp").groupby("coin").tail(1)
    X_latest = latest[["rsi","score"]]
    try:
        probs = model.predict_proba(X_latest)[:,1]
        latest["Chance %"] = probs*100
    except Exception:
        latest["Chance %"] = 0

    # Signal
    def get_signal(score):
        if score>=10: return "🚀 STRONG BUY"
        elif score>=8: return "🔥 BUY"
        elif score>=6: return "⏳ EARLY"
        elif score>=4: return "⏳ WAIT"
        else: return "❌ NO"
    latest["Signal"] = latest["score"].apply(get_signal)

    # Data Status
    counts = df.groupby("coin").size()
    def status_color(n):
        if n>=20: return "🟩 كافي"
        elif n>=10: return "🟨 متوسط"
        else: return "🟥 قليل"
    latest["Data Status"] = latest["coin"].apply(lambda c: status_color(counts.get(c,0)))

    # Display table
    latest = latest.sort_values("Chance %", ascending=False)
    st.success(f"دقة الموديل: {round(acc*100,2)}%")
    st.dataframe(latest[["coin","price","drop_percent","rsi","volx","support","score","Signal","Chance %","Data Status"]], use_container_width=True)

    # تفاصيل أي عملة
    coin_list = latest["coin"].unique().tolist()
    selected_coin = st.selectbox("اختار عملة للتفاصيل", coin_list)
    if selected_coin:
        st.subheader(f"📊 بيانات {selected_coin}")
        st.dataframe(df[df["coin"]==selected_coin].sort_values("timestamp", ascending=False), use_container_width=True)
