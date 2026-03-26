import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.title("🚀 Crypto AI Pump Predictor PRO (No Binance)")

# =========================
# 🛡️ Safe API
# =========================
last_request_time = 0
REQUEST_DELAY = 0.8

def safe_request(url):
    global last_request_time
    try:
        if time.time() - last_request_time < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY)

        last_request_time = time.time()
        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            return None

        return r.json()
    except:
        return None


# =========================
# 📡 Sources (No Binance)
# =========================
def get_kucoin(symbol="BTC-USDT"):
    url = f"https://api.kucoin.com/api/v1/market/candles?type=1min&symbol={symbol}"
    data = safe_request(url)

    if not data or "data" not in data:
        return []

    out = []
    try:
        for c in data["data"]:
            out.append([float(c[2]), float(c[5])])
    except:
        pass

    return out


def get_bybit(symbol="BTCUSDT"):
    url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval=1"
    data = safe_request(url)

    if not data or "result" not in data:
        return []

    out = []
    try:
        for c in data["result"]["list"]:
            out.append([float(c[4]), float(c[5])])
    except:
        pass

    return out


def get_okx(symbol="BTC-USDT"):
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar=1m"
    data = safe_request(url)

    if not data or "data" not in data:
        return []

    out = []
    try:
        for c in data["data"]:
            out.append([float(c[4]), float(c[5])])
    except:
        pass

    return out


# =========================
# 🔥 Merge Data
# =========================
def get_all_data(symbol):
    data = []

    for f in [get_kucoin, get_bybit, get_okx]:
        try:
            d = f(symbol)
            if d:
                data.extend(d)
        except:
            continue

    if len(data) < 20:
        return None

    df = pd.DataFrame(data, columns=["price", "volume"])

    return df


# =========================
# 🧠 Feature Engineering
# =========================
def create_features(df):
    df = df.copy()

    df["return"] = df["price"].pct_change()
    df["ma5"] = df["price"].rolling(5).mean()
    df["ma10"] = df["price"].rolling(10).mean()

    df["trend"] = df["ma5"] - df["ma10"]
    df["vol_ma"] = df["volume"].rolling(10).mean()
    df["volx"] = df["volume"] / df["vol_ma"]

    df = df.dropna()

    return df


# =========================
# 💣 Labeling (Pump / No Pump)
# =========================
def make_labels(df):
    df = df.copy()

    future_return = df["price"].shift(-5) / df["price"] - 1

    df["label"] = (future_return > 0.003).astype(int)  # Pump threshold

    df = df.dropna()

    return df


# =========================
# 🧠 AI Model
# =========================
def train_model(df):
    features = ["return", "trend", "volx"]

    X = df[features]
    y = df["label"]

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        random_state=42
    )

    model.fit(X, y)

    return model


# =========================
# 📊 Signal Engine
# =========================
def get_signal(pred_prob):
    if pred_prob > 0.75:
        return "🟢 BUY STRONG"
    elif pred_prob > 0.55:
        return "🟢 BUY"
    elif pred_prob < 0.35:
        return "🔴 EXIT"
    else:
        return "⚪ HOLD"


# =========================
# 🟢 Data Status
# =========================
def status(n):
    if n > 60:
        return "🟢 GOOD"
    elif n > 30:
        return "🟡 MEDIUM"
    return "🔴 LOW"


# =========================
# 🚀 UI
# =========================
symbols = st.text_input("Enter symbols", "BTCUSDT,ETHUSDT")

if st.button("Run AI Scan"):

    results = []

    for sym in symbols.split(","):

        df = get_all_data(sym.strip())

        if df is None or len(df) < 50:
            continue

        df = create_features(df)
        df = make_labels(df)

        if len(df) < 30:
            continue

        model = train_model(df)

        last = df.iloc[-1]

        X_live = np.array([[last["return"], last["trend"], last["volx"]]])

        prob = model.predict_proba(X_live)[0][1]

        results.append({
            "Coin": sym.strip(),
            "Price": round(last["price"], 5),
            "Volx": round(last["volx"], 2),
            "Trend": round(last["trend"], 6),
            "Pump Probability": round(prob * 100, 2),
            "Data Status": status(len(df)),
            "Signal": get_signal(prob)
        })

    result_df = pd.DataFrame(results)

    # =========================
    # 🎨 UI SAFE TABLE
    # =========================
    if result_df.empty:
        st.warning("⚠️ مفيش بيانات كفاية")
    else:

        def color(val):
            val = str(val)
            if "GOOD" in val:
                return "background-color:#2ecc71;color:black"
            if "MEDIUM" in val:
                return "background-color:#f1c40f;color:black"
            if "LOW" in val:
                return "background-color:#e74c3c;color:white"
            return ""

        if "Data Status" not in result_df.columns:
            result_df["Data Status"] = "LOW"

        styled = result_df.style.applymap(color, subset=["Data Status"])

        st.subheader("📊 AI Pump Prediction Dashboard")
        st.dataframe(styled, use_container_width=True)
