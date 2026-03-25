import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from github import Github, Auth
import json

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Scanner AI PRO MAX")

# ==============================
# 🎨 Dark Mode Table
# ==============================
st.markdown("""
<style>
[data-testid="stDataFrame"] {
    background-color: #0e1117;
    color: white;
}
thead tr th {
    background-color: #111 !important;
    color: white !important;
}
tbody tr td {
    background-color: #0e1117 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# GitHub setup
# ==============================
GITHUB_TOKEN = st.secrets["GITHUB"]["TOKEN"]
REPO_NAME = st.secrets["GITHUB"]["REPO"]
BRANCH = st.secrets["GITHUB"]["BRANCH"]
FILE_PATH = "data.json"

g = Github(auth=Auth.Token(GITHUB_TOKEN))
repo = g.get_repo(REPO_NAME)

# ==============================
# GitHub
# ==============================
def load_github_data():
    try:
        contents = repo.get_contents(FILE_PATH, ref=BRANCH)
        return json.loads(contents.decoded_content.decode())
    except:
        return []

def save_github_data(data):
    content = json.dumps(data, indent=4)
    try:
        contents = repo.get_contents(FILE_PATH, ref=BRANCH)
        repo.update_file(contents.path, "update", content, contents.sha, branch=BRANCH)
    except:
        repo.create_file(FILE_PATH, "create", content, branch=BRANCH)

# ==============================
# RSI (EMA improved)
# ==============================
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).ewm(alpha=1/period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# ==============================
# Support / Resistance
# ==============================
def get_support_resistance(prices):
    support = np.min(prices[-20:])
    resistance = np.max(prices[-20:])
    return support, resistance

# ==============================
# APIs
# ==============================
def get_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency":"usd","order":"volume_desc","per_page":50,"page":1}
    return requests.get(url, params=params).json()

def fetch_data(coin_id, days=30):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency":"usd","days":days}
    return requests.get(url, params=params).json()

# ==============================
# Smart update
# ==============================
def should_update(data, coin):
    for row in data:
        if row["coin"] == coin:
            candles = row.get("candles", [])
            if not candles:
                return True
            last = candles[-1]["timestamp"] / 1000
            if time.time() - last < 600:
                return False
    return True

# ==============================
# Update collector (optimized save)
# ==============================
def run_collector():
    st.info("⏳ تحديث آمن...")
    coins = get_coins()
    data = load_github_data()
    updated = False

    for i in range(0, len(coins), 10):
        batch = coins[i:i+10]

        def work(c):
            nonlocal data, updated
            try:
                symbol = c["symbol"].upper()
                if not should_update(data, symbol):
                    return

                d = fetch_data(c["id"])
                prices = [p[1] for p in d.get("prices",[])]
                vols = [v[1] for v in d.get("total_volumes",[])]

                candles = []
                for i in range(len(prices)):
                    candles.append({
                        "timestamp": int(d["prices"][i][0]),
                        "price": float(prices[i]),
                        "volume": float(vols[i]) if i<len(vols) else 0
                    })

                for row in data:
                    if row["coin"] == symbol:
                        row["candles"] = candles
                        updated = True
                        return

                data.append({"coin":symbol,"candles":candles})
                updated = True

            except Exception as e:
                print(e)

        with ThreadPoolExecutor(max_workers=5) as ex:
            ex.map(work, batch)

        time.sleep(2)

    if updated:
        save_github_data(data)

    st.success("✅ تم التحديث")

# ==============================
# زر التحديث
# ==============================
if st.button("🔄 تحديث"):
    run_collector()

# ==============================
# Load data
# ==============================
data = load_github_data()
rows = []

# ==============================
# Prepare AI dataset
# ==============================
for coin_data in data:
    candles = coin_data.get("candles", [])
    if len(candles) < 20:
        continue

    prices = np.array([c["price"] for c in candles])
    vols = np.array([c["volume"] for c in candles])

    for i in range(15, len(prices)-3):
        rsi = calculate_rsi(prices[i-15:i])
        drop = ((prices[i] - prices[:i].max()) / prices[:i].max()) * 100
        avg_vol = vols[i-10:i].mean()
        volx = vols[i] / avg_vol if avg_vol>0 else 1
        change = ((prices[i] - prices[i-3]) / prices[i-3]) * 100

        score = 0
        if rsi < 35: score += 3
        if drop < -20: score += 3
        if volx > 1.5: score += 2
        if change > 0: score += 2

        target = 1 if prices[i+3] > prices[i] else 0

        rows.append({
            "rsi": rsi,
            "drop": drop,
            "volx": volx,
            "change": change,
            "score": score,
            "target": target
        })

df_ai = pd.DataFrame(rows)

# ==============================
# Train AI
# ==============================
if len(df_ai) > 50:
    X = df_ai[["rsi","drop","volx","change","score"]]
    y = df_ai["target"]

    model = RandomForestClassifier(n_estimators=100)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    model.fit(X_train,y_train)

    latest_rows = []

    for coin_data in data:
        coin = coin_data["coin"]
        candles = coin_data.get("candles", [])

        if len(candles) < 20:
            continue

        prices = np.array([c["price"] for c in candles])
        vols = np.array([c["volume"] for c in candles])

        rsi = calculate_rsi(prices[-15:])
        drop = ((prices[-1] - prices.max()) / prices.max()) * 100
        avg_vol = vols[-10:].mean()
        volx = vols[-1] / avg_vol if avg_vol>0 else 1
        change = ((prices[-1] - prices[-3]) / prices[-3]) * 100

        support, resistance = get_support_resistance(prices)

        score = 0
        if rsi < 35: score += 3
        if drop < -20: score += 3
        if volx > 1.5: score += 2
        if change > 0: score += 2

        # Data Status
        if len(candles) >= 30:
            data_status = "🟢 Good"
        elif len(candles) >= 20:
            data_status = "🟡 Moderate"
        else:
            data_status = "🔴 Low"

        # Signal
        if score >= 7:
            signal = "🔥 Strong Buy"
        elif score >= 5:
            signal = "🚀 Buy"
        elif score >= 3:
            signal = "🟠 Hold"
        else:
            signal = "❌ No Trade"

        chance = model.predict_proba([[rsi,drop,volx,change,score]])[0][1]*100

        latest_rows.append({
            "Coin": coin,
            "Price (USD)": round(prices[-1],2),
            "Drop %": round(drop,2),
            "RSI": round(rsi,2),
            "Volume x": round(volx,2),
            "Support": round(support,2),
            "Resistance": round(resistance,2),
            "Score /10": score,
            "Chance %": round(chance,2),
            "🚨 Signal": signal,
            "📊 Data": data_status
        })

    df = pd.DataFrame(latest_rows)
    df = df.sort_values("Chance %", ascending=False)

    st.dataframe(df, use_container_width=True)

else:
    st.warning("⚠️ البيانات غير كافية للـ AI")
