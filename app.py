import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from streamlit_autorefresh import st_autorefresh

# =========================
# CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("🚀 ULTRA PRO Crypto AI Scanner (FIXED)")

st_autorefresh(interval=15 * 60 * 1000, key="refresh")

# =========================
# TELEGRAM (optional)
# =========================
BOT_TOKEN = "PUT_TOKEN"
CHAT_ID = "PUT_CHAT_ID"

sent_signals = set()

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
    except:
        pass

# =========================
# DATA
# =========================
@st.cache_data(ttl=300)
def get_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "volume_desc",
        "per_page": 20,
        "page": 1,
        "sparkline": False
    }

    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        return pd.DataFrame()

    return pd.DataFrame(r.json())

# =========================
# INDICATORS
# =========================
def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def ema(series, span):
    return series.ewm(span=span).mean()

def macd(series):
    return ema(series, 12) - ema(series, 26)

def volume_x(vol, avg):
    return vol / (avg + 1e-9)

# =========================
# SUPPORT / RESISTANCE (FIXED)
# =========================
def support_resistance(df, n=10):
    support = df["current_price"].rolling(n).min()
    resistance = df["current_price"].rolling(n).max()
    return support, resistance

# =========================
# MODEL
# =========================
model = RandomForestClassifier(n_estimators=150)
scaler = StandardScaler()

def train(df):
    df = df.dropna()

    X = df[["rsi", "macd", "vol_x"]]
    y = (df["price_change"] > 0).astype(int)

    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

# =========================
# LOAD DATA
# =========================
data = get_coins()

if data.empty:
    st.error("No data")
    st.stop()

# =========================
# CLEAN + FEATURES
# =========================
data["price_change"] = data["price_change_percentage_24h"]

data["rsi"] = rsi(data["current_price"])
data["macd"] = macd(data["current_price"])
data["vol_x"] = volume_x(data["total_volume"], data["total_volume"].mean())

# =========================
# SUPPORT / RESISTANCE APPLY
# =========================
data["support"], data["resistance"] = support_resistance(data, 10)

# تنظيف
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# =========================
# AI TRAIN
# =========================
if len(data) > 5:
    train(data)

    X = scaler.transform(data[["rsi", "macd", "vol_x"]])
    data["chance"] = model.predict_proba(X)[:, 1] * 100
else:
    data["chance"] = 0

# =========================
# SIGNALS
# =========================
def signal(row):
    if row["chance"] > 70 and row["rsi"] < 60:
        return "🟢 BUY"
    elif row["chance"] < 40:
        return "🔴 WEAK"
    return "🟡 WAIT"

data["signal"] = data.apply(signal, axis=1)

# =========================
# TELEGRAM ALERTS (NO SPAM)
# =========================
for _, row in data.iterrows():
    key = f"{row['symbol']}-{row['signal']}"

    if row["signal"] == "🟢 BUY" and key not in sent_signals:
        send_telegram(
            f"🚀 ULTRA PRO SIGNAL\n"
            f"Coin: {row['symbol']}\n"
            f"Price: {row['current_price']}\n"
            f"Chance: {row['chance']:.2f}%\n"
            f"RSI: {row['rsi']:.2f}"
        )
        sent_signals.add(key)

# =========================
# UI
# =========================
st.dataframe(data[[
    "symbol",
    "current_price",
    "rsi",
    "macd",
    "vol_x",
    "chance",
    "signal",
    "support",
    "resistance"
]])
