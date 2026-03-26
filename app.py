import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto AI Pump Predictor PRO")

# =========================
# 🔥 حماية من الـ API Limit
# =========================
REQUEST_DELAY = 0.8  # مهم جداً
last_request_time = 0

def safe_request(url):
    global last_request_time
    now = time.time()
    if now - last_request_time < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY)

    last_request_time = time.time()
    try:
        r = requests.get(url, timeout=10)
        return r.json()
    except:
        return None


# =========================
# 📊 Normalize Data
# =========================
def normalize_candles(candles):
    clean = []
    for c in candles:
        try:
            clean.append({
                "timestamp": c.get("timestamp") or c.get("t"),
                "price": float(c.get("close") or c.get("c") or c.get("price") or c.get("last")),
                "volume": float(c.get("volume") or c.get("v") or 0)
            })
        except:
            continue

    clean = sorted(clean, key=lambda x: x["timestamp"])
    return clean


# =========================
# 📡 Example API (Binance fallback)
# =========================
def get_binance(symbol="BTCUSDT"):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=50"
    data = safe_request(url)
    if not data:
        return []

    candles = []
    for c in data:
        candles.append({
            "timestamp": c[0],
            "close": c[4],
            "volume": c[5]
        })
    return candles


# =========================
# 🧠 Feature Engineering
# =========================
def create_features(candles):
    df = pd.DataFrame(candles)

    df["return"] = df["price"].pct_change()
    df["vol_change"] = df["volume"].pct_change()

    df["ma5"] = df["price"].rolling(5).mean()
    df["ma10"] = df["price"].rolling(10).mean()

    df["vol_spike"] = df["volume"] / (df["volume"].rolling(10).mean())

    df = df.dropna()
    return df


# =========================
# 💣 Labeling (Training Data)
# =========================
def label_data(df):
    df["future_return"] = df["price"].shift(-5) / df["price"] - 1

    df["target"] = df["future_return"].apply(lambda x: 1 if x > 0.02 else 0)
    df = df.dropna()

    return df


# =========================
# 🤖 Train AI Model
# =========================
def train_model(df):
    features = ["return", "vol_change", "ma5", "ma10", "vol_spike"]

    X = df[features]
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    return model, scaler


# =========================
# 🔮 Prediction
# =========================
def predict(df, model, scaler):
    features = ["return", "vol_change", "ma5", "ma10", "vol_spike"]

    X = scaler.transform(df[features].iloc[-1:].values)
    prob = model.predict_proba(X)[0][1]

    if prob > 0.75:
        signal = "💣 STRONG PUMP (30-60 min)"
    elif prob > 0.6:
        signal = "🔥 Pump Possible"
    else:
        signal = "❌ No Signal"

    return prob, signal


# =========================
# 📦 Load / Train Model
# =========================
if os.path.exists("model.pkl"):
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
else:
    model = None
    scaler = None


# =========================
# 🚀 MAIN LOOP
# =========================
symbol = st.text_input("Enter Symbol", "BTCUSDT")

if st.button("Run AI Scan"):
    raw = get_binance(symbol)

    if len(raw) < 20:
        st.error("❌ Not enough data")
        st.stop()

    candles = normalize_candles(raw)

    df = create_features(candles)
    df = label_data(df)

    if model is None:
        st.info("🧠 Training AI Model...")
        model, scaler = train_model(df)
    else:
        st.info("🧠 Using existing model")

    prob, signal = predict(df, model, scaler)

    st.subheader("📊 Result")
    st.write("Pump Probability:", round(prob * 100, 2), "%")
    st.success(signal)
