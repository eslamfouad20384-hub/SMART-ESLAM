import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from github import Github, Auth
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

st.set_page_config(layout="wide")
st.title("🚀 Self-Learning Crypto GRU AI (Adaptive System)")

# =========================
# 🔥 GitHub Storage
# =========================
GITHUB_TOKEN = st.secrets["GITHUB"]["TOKEN"]
REPO_NAME = st.secrets["GITHUB"]["REPO"]
BRANCH = st.secrets["GITHUB"]["BRANCH"]
FILE_PATH = "data.json"

g = Github(auth=Auth.Token(GITHUB_TOKEN))
repo = g.get_repo(REPO_NAME)


# =========================
# 📥 Load Dataset
# =========================
def load_data():
    try:
        file = repo.get_contents(FILE_PATH, ref=BRANCH)
        return json.loads(file.decoded_content.decode())
    except:
        return []


# =========================
# 📤 Save Dataset
# =========================
def save_data(data):
    content = json.dumps(data, indent=4)
    try:
        file = repo.get_contents(FILE_PATH, ref=BRANCH)
        repo.update_file(file.path, "update", content, file.sha, branch=BRANCH)
    except:
        repo.create_file(FILE_PATH, "create", content, branch=BRANCH)


# =========================
# 📡 Market Data
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
# 📊 Feature Engineering
# =========================
def build_df(candles):
    df = pd.DataFrame(candles, columns=["price", "volume"])

    df["return"] = df["price"].pct_change()
    df["volx"] = df["volume"] / df["volume"].rolling(10).mean()
    df["trend"] = df["price"].rolling(5).mean() - df["price"].rolling(20).mean()

    df = df.dropna()
    return df


# =========================
# 🐋 Whale Detection
# =========================
def detect_whale(df):
    return 1 if df["volx"].iloc[-1] > 2 else 0


# =========================
# 🧠 Sequences
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
# 🧠 GRU Model
# =========================
def build_model(input_shape):
    model = Sequential()
    model.add(GRU(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(32))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# =========================
# 📊 Build Training Data from JSON
# =========================
def build_training_from_json(json_data):
    df = pd.DataFrame(json_data)

    if len(df) < 10:
        return None, None

    # لازم يكون فيه نتائج سابقة
    df = df.dropna()

    features = df[["price", "volx", "Pump Probability"]].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    X, y = [], []

    for i in range(len(scaled) - 20):
        X.append(scaled[i:i+20])
        y.append(1 if df["profit"].iloc[i+20] > 0 else 0)

    return np.array(X), np.array(y)


# =========================
# 🚀 MAIN SYSTEM
# =========================
if st.button("🚀 Run Self-Learning AI"):

    json_data = load_data()
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
            model.fit(X, y, epochs=2, batch_size=16, verbose=0)

            last_seq = X[-1].reshape(1, X.shape[1], X.shape[2])
            pred = model.predict(last_seq, verbose=0)[0][0]

            whale = detect_whale(df)

            # =========================
            # 🎯 SIGNAL
            # =========================
            if pred > 0.75:
                signal = "🔥 STRONG BUY"
            elif pred > 0.55:
                signal = "🚀 BUY"
            elif whale:
                signal = "🐋 WHALE ALERT"
            else:
                signal = "⚪ HOLD"

            profit = float(pred - 0.5) * 2  # simulation score

            row = {
                "coin": c["symbol"],
                "price": prices[-1],
                "volx": float(df["volx"].iloc[-1]),
                "Pump Probability": float(pred),
                "signal": signal,
                "profit": profit
            }

            results.append(row)

        except:
            continue

    # =========================
    # 💾 SAVE LEARNING DATA
    # =========================
    json_data.extend(results)
    save_data(json_data)

    df_out = pd.DataFrame(results)

    if df_out.empty:
        st.warning("⚠️ مفيش بيانات كفاية")
    else:
        df_out = df_out.sort_values("Pump Probability", ascending=False)

        st.subheader("📊 AI Live Dashboard")
        st.dataframe(df_out, use_container_width=True)

        # =========================
        # 📈 Self Learning Score
        # =========================
        avg_profit = df_out["Pump Probability"].mean()

        st.subheader("🧠 AI Learning Status")
        st.write("Dataset size:", len(json_data))
        st.write("Avg confidence:", round(avg_profit * 100, 2), "%")
