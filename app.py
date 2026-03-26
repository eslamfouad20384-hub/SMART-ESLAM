import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from github import Github, Auth
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.title("🚀 Crypto AI PRO MAX (No Deep Learning Edition)")

# =========================
# 🔥 GitHub JSON Storage
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
        file = repo.get_contents(FILE_PATH)
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

    df["ma_fast"] = df["price"].rolling(5).mean()
    df["ma_slow"] = df["price"].rolling(20).mean()
    df["trend"] = df["ma_fast"] - df["ma_slow"]

    df["momentum"] = df["price"] - df["price"].shift(3)

    df = df.dropna()
    return df


# =========================
# 🐋 Whale Detection
# =========================
def detect_whale(df):
    return 1 if df["volx"].iloc[-1] > 2.2 else 0


# =========================
# 🧠 Feature Matrix
# =========================
def create_dataset(df):
    features = df[["price", "volx", "trend", "momentum", "return"]].values

    X, y = [], []

    for i in range(len(features) - 5):
        X.append(features[i])
        y.append(1 if df["price"].iloc[i+5] > df["price"].iloc[i] else 0)

    return np.array(X), np.array(y)


# =========================
# 🧠 Model
# =========================
def build_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )


# =========================
# 🚀 MAIN
# =========================
if st.button("🚀 Run AI Scan"):

    coins = get_coins()
    json_data = load_data()

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

            X, y = create_dataset(df)

            if len(X) < 20:
                continue

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            model = build_model()
            model.fit(X_scaled, y)

            last = X_scaled[-1].reshape(1, -1)
            pred = model.predict_proba(last)[0][1]

            whale = detect_whale(df)

            # =========================
            # 🎯 SIGNALS
            # =========================
            if pred > 0.75:
                signal = "🔥 STRONG BUY"
            elif pred > 0.55:
                signal = "🚀 BUY"
            elif whale:
                signal = "🐋 WHALE ALERT"
            else:
                signal = "⚪ HOLD"

            row = {
                "coin": c["symbol"].upper(),
                "price": prices[-1],
                "volx": float(df["volx"].iloc[-1]),
                "trend": float(df["trend"].iloc[-1]),
                "momentum": float(df["momentum"].iloc[-1]),
                "prob": float(pred),
                "signal": signal
            }

            results.append(row)

        except:
            continue

    # =========================
    # 💾 SELF LEARNING STORAGE
    # =========================
    json_data.extend(results)
    save_data(json_data)

    df_out = pd.DataFrame(results)

    if df_out.empty:
        st.warning("⚠️ مفيش بيانات كفاية")
    else:
        df_out = df_out.sort_values("prob", ascending=False)

        st.subheader("📊 AI Dashboard")
        st.dataframe(df_out, use_container_width=True)

        # =========================
        # 🧠 AI STATS
        # =========================
        st.subheader("🧠 AI Learning Status")

        st.write("Dataset size:", len(json_data))
        st.write("Avg confidence:", round(df_out["prob"].mean() * 100, 2), "%")

        buy_signals = len(df_out[df_out["signal"].str.contains("BUY")])
        st.write("Active BUY signals:", buy_signals)
