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
st.title("🚀 Smart Crypto Scanner AI PRO")

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
# RSI
# ==============================
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    if len(gain) < period:
        return 50
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1+rs))

# ==============================
# CoinGecko
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
# Smart Update
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
# Save candles
# ==============================
def update_coin(data, coin, candles):
    for row in data:
        if row["coin"] == coin:
            all_c = {c["timestamp"]:c for c in row.get("candles",[])}
            for c in candles:
                all_c[c["timestamp"]] = c
            row["candles"] = sorted(all_c.values(), key=lambda x:x["timestamp"])
            save_github_data(data)
            return
    data.append({"coin":coin,"candles":candles})
    save_github_data(data)

# ==============================
# Collector
# ==============================
def run_collector():
    st.info("⏳ تحديث آمن...")
    coins = get_coins()
    data = load_github_data()

    for i in range(0, len(coins), 10):
        batch = coins[i:i+10]

        def work(c):
            try:
                symbol = c["symbol"].upper()
                if not should_update(data, symbol):
                    return None

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
                update_coin(data, symbol, candles)
                return symbol
            except:
                return None

        with ThreadPoolExecutor(max_workers=5) as ex:
            ex.map(work, batch)

        time.sleep(2)

    st.success("✅ تم التحديث بدون حظر")

# ==============================
# زر التحديث
# ==============================
if st.button("🔄 تحديث"):
    run_collector()

# ==============================
# AI + Analysis
# ==============================
data = load_github_data()
rows = []

for coin_data in data:
    coin = coin_data["coin"]
    candles = coin_data.get("candles", [])

    if len(candles) < 15:
        continue

    prices = np.array([c["price"] for c in candles])
    vols = np.array([c["volume"] for c in candles])

    for i in range(14, len(candles)-3):
        rsi = calculate_rsi(prices[i-14:i+1])
        drop = ((prices[i] - prices[:i+1].max()) / prices[:i+1].max()) * 100
        avg_vol = vols[i-10:i].mean()
        volx = vols[i] / avg_vol if avg_vol>0 else 1
        score = 0
        if drop < -25: score += 2
        if rsi < 35: score += 2
        if volx > 1.5: score += 2
        target = 1 if prices[i+3] > prices[i] else 0
        rows.append({"coin": coin,"rsi": rsi,"score": score,"target": target})

df_ai = pd.DataFrame(rows)
if len(df_ai) > 20:
    X = df_ai[["rsi","score"]]
    y = df_ai["target"]
    model = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    model.fit(X_train,y_train)

    latest_rows = []
    for coin_data in data:
        coin = coin_data["coin"]
        candles = coin_data.get("candles", [])
        if len(candles) < 15: continue
        prices = np.array([c["price"] for c in candles])
        vols = np.array([c["volume"] for c in candles])
        rsi = calculate_rsi(prices[-15:])
        drop = ((prices[-1] - prices.max()) / prices.max()) * 100
        avg_vol = vols[-10:].mean()
        volx = vols[-1] / avg_vol if avg_vol>0 else 1
        score = 0
        if drop < -25: score += 2
        if rsi < 35: score += 2
        if volx > 1.5: score += 2

        # ==============================
        # Data Status
        # ==============================
        if len(candles) >= 30: status = "🟢 Good"
        elif len(candles) >= 20: status = "🟡 Moderate"
        else: status = "🔴 Low"

        # ==============================
        # Recommendation
        # ==============================
        if score >= 6 or (score >= 4 and drop < -5):
            rec = "Strong Buy" if score >= 6 else "Buy"
        elif score >= 3: rec = "Hold"
        else: rec = "No"

        latest_rows.append({
            "Coin": coin,
            "Price (USD)": round(prices[-1],2),
            "Drop %": round(drop,2),
            "RSI": round(rsi,2),
            "Volume x": round(volx,2),
            "Score": score,
            "Chance %": round(model.predict_proba([[rsi,score]])[0][1]*100,2),
            "Recommendation": rec,
            "Data Status": status
        })

    latest_df = pd.DataFrame(latest_rows)
    latest_df = latest_df.sort_values("Chance %", ascending=False)

    # ==============================
    # مؤشر الخوف والطمع + حالة السوق العام
    # ==============================
    try:
        fg = requests.get("https://api.alternative.me/fng/").json()
        fg_value = fg["data"][0]["value"]
        fg_status = fg["data"][0]["value_classification"]
        if fg_value < 30: fg_emoji = "😨"
        elif fg_value < 70: fg_emoji = "😐"
        else: fg_emoji = "😎"
    except:
        fg_value, fg_status, fg_emoji = "N/A", "N/A", "❓"

    market_trend = "⏸️"  # افتراضي عرضي
    avg_score = latest_df["Score"].mean() if not latest_df.empty else 0
    if avg_score > 4: market_trend = "🚀"
    elif avg_score < 2: market_trend = "📉"

    st.markdown(f"### Fear & Greed: {fg_value} {fg_emoji} | Market Trend: {market_trend}")

    st.dataframe(latest_df, use_container_width=True)

else:
    st.warning("⚠️ البيانات غير كافية للـ AI")
