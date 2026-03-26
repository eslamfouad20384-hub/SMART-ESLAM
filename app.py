import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
from github import Github, Auth
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(layout="wide")
st.title("🚀 Crypto LSTM AI PRO MAX (TradingView Style)")

# =========================
# 🔥 GitHub Dataset
# =========================
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
    content = json.dumps(data, indent=4)
    try:
        file = repo.get_contents(FILE_PATH, ref=BRANCH)
        repo.update_file(file.path, "update", content, file.sha, branch=BRANCH)
    except:
        repo.create_file(FILE_PATH, "create", content, branch=BRANCH)


# =========================
# 📡 CoinGecko API
# =========================
def get_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "volume_desc", "per_page": 30}
    return requests.get(url, params=params).json()


def get_history(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": 30}
    return requests.get(url, params=params).json()


# =========================
# 📊 Feature Builder
# =========================
def build_df(candles):
    df = pd.DataFrame(candles, columns=["price", "volume"])

    df["return"] = df["price"].pct_change()
    df["volx"] = df["volume"] / df["volume"].rolling(10).mean()
    df["trend"] = df["price"].rolling(5).mean() - df["price"].rolling(20).mean()

    df = df.dropna()
    return df


# =========================
# 💣 Whale Detection
# =========================
def detect_whale(df):
    if df["volx"].iloc[-1] > 2:
        return 1
    return 0


# =========================
# 🧠 LSTM Dataset
# =========================
def create_sequences(data, seq_len=20):
    X, y = [], []

    for i in range(len(data) - seq_len - 5):
        seq = data[i:i+seq_len]
        target = 1 if data[i+seq_len+5][0] > data[i+seq_len][0] else 0

        X.append(seq)
        y.append(target)

    return np.array(X), np.array(y)


# =========================
# 🧠 Build LSTM Model
# =========================
def build_model(input_shape):
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(32))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


# =========================
# 🚀 Main Run
# =========================
if st.button("🚀 Run LSTM AI Scan"):

    data = load_data()
    coins = get_coins()

    results = []

    for c in coins:

        try:
            hist = get_history(c["id"])
            prices = [p[1] for p in hist["prices"]]
            vols = [v[1] for v in hist["total_volumes"]]

            candles = list(zip(prices, vols))

            df = build_df(candles)

            if len(df) < 50:
                continue

            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df[["price", "volume", "return", "volx", "trend"]])

            X, y = create_sequences(scaled)

            if len(X) < 20:
                continue

            model = build_model((X.shape[1], X.shape[2]))

            model.fit(X, y, epochs=3, batch_size=16, verbose=0)

            last_seq = X[-1].reshape(1, X.shape[1], X.shape[2])
            pred = model.predict(last_seq, verbose=0)[0][0]

            whale = detect_whale(df)

            # =========================
            # 🎯 Signal Logic
            # =========================
            if pred > 0.75:
                signal = "🔥 STRONG BUY"
            elif pred > 0.55:
                signal = "🚀 BUY"
            elif whale == 1:
                signal = "🐋 WHALE ALERT"
            else:
                signal = "⚪ HOLD"

            results.append({
                "Coin": c["symbol"].upper(),
                "Price": round(prices[-1], 4),
                "Pump Probability": round(pred * 100, 2),
                "Volume Spike": df["volx"].iloc[-1],
                "Whale": whale,
                "Signal": signal
            })

        except:
            continue

    df_out = pd.DataFrame(results)

    # =========================
    # 📊 Dashboard
    # =========================
    if df_out.empty:
        st.warning("⚠️ مفيش بيانات كفاية")
    else:
        df_out = df_out.sort_values("Pump Probability", ascending=False)
        st.subheader("📊 LSTM AI Dashboard")
        st.dataframe(df_out, use_container_width=True)

        # =========================
        # 📈 Backtesting
        # =========================
        st.subheader("📈 Backtesting (Simple Simulation)")

        profit = 0
        trades = 0

        for _, row in df_out.iterrows():
            if "BUY" in row["Signal"]:
                profit += row["Pump Probability"] * 0.01
                trades += 1

        st.write("Total Trades:", trades)
        st.write("Estimated Profit %:", round(profit, 2))
