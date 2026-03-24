import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from github import Github, Auth
import json

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Scanner AI + Daily Candles + GitHub Sync")

# ==============================
# GitHub setup
# ==============================
GITHUB_TOKEN = st.secrets["GITHUB"]["TOKEN"]
REPO_NAME = st.secrets["GITHUB"]["REPO"]
BRANCH = st.secrets["GITHUB"]["BRANCH"]
FILE_PATH = "data.json"

auth = Auth.Token(GITHUB_TOKEN)
g = Github(auth=auth)
repo = g.get_repo(REPO_NAME)

# ==============================
# GitHub functions
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
        repo.update_file(contents.path, "Update data", content, contents.sha, branch=BRANCH)
    except:
        repo.create_file(FILE_PATH, "Create data", content, branch=BRANCH)

# ==============================
# RSI
# ==============================
def calculate_rsi(prices, period=14):
    if len(prices) < period:
        return 50
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1+rs))

# ==============================
# توحيد شكل البيانات
# ==============================
def normalize_coin(coin):
    try:
        return {
            "id": coin.get("id") or coin.get("symbol"),
            "symbol": coin.get("symbol", "").upper()
        }
    except:
        return None

# ==============================
# مصادر العملات
# ==============================
def get_coins_coingecko():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency":"usd","order":"volume_desc","per_page":50,"page":1}
    try:
        data = requests.get(url, params=params).json()
        return [normalize_coin(c) for c in data if c]
    except:
        return []

def get_coins_cryptocompare():
    url = "https://min-api.cryptocompare.com/data/top/mktcapfull"
    params = {"limit":50,"tsym":"USD"}
    try:
        data = requests.get(url, params=params).json()
        return [{"id":c["CoinInfo"]["Name"], "symbol":c["CoinInfo"]["Name"].upper()} for c in data.get("Data",[])]
    except:
        return []

def get_coins_binance():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        data = requests.get(url).json()
        return [{"id":c["symbol"], "symbol":c["symbol"]} for c in data if c["symbol"].endswith("USDT")]
    except:
        return []

# ==============================
# بيانات العملة
# ==============================
def fetch_coin_data_daily(coin_id):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": 30}
        return requests.get(url, params=params).json()
    except:
        return {"prices":[],"total_volumes":[]}

# ==============================
# سكورد
# ==============================
def calculate_score(price, drop, rsi, volx):
    score = 0
    if drop < -25: score += 2
    if rsi < 35: score += 2
    if volx > 1.5: score += 2
    return score

# ==============================
# تحديث GitHub
# ==============================
def update_github_row(data, coin, timestamp, price, drop, rsi, volume, volx, support, score):
    for row in data:
        if row["coin"] == coin:
            row.update({
                "timestamp": timestamp,
                "price": price,
                "drop_percent": drop,
                "rsi": rsi,
                "volume": volume,
                "volx": volx,
                "support": support,
                "score": score
            })
            save_github_data(data)
            return

    data.append({
        "coin": coin,
        "timestamp": timestamp,
        "price": price,
        "drop_percent": drop,
        "rsi": rsi,
        "volume": volume,
        "volx": volx,
        "support": support,
        "score": score
    })
    save_github_data(data)

# ==============================
# تشغيل الكولكتور
# ==============================
def run_collector():
    st.info("⏳ جاري تحديث البيانات…")

    coins = []
    coins += get_coins_coingecko()
    coins += get_coins_cryptocompare()
    coins += get_coins_binance()

    # إزالة المكرر
    seen = set()
    unique_coins = []
    for c in coins:
        if c and c["symbol"] not in seen:
            seen.add(c["symbol"])
            unique_coins.append(c)

    github_data = load_github_data()

    def analyze_coin(coin):
        try:
            coin_id = coin["id"]
            data = fetch_coin_data_daily(coin_id)

            prices = np.array([p[1] for p in data.get("prices",[])])
            volumes = np.array([v[1] for v in data.get("total_volumes",[])])

            if len(prices) == 0:
                return None

            current_price = prices[-1]
            max_price = prices.max()
            drop = ((current_price - max_price)/max_price)*100

            rsi = calculate_rsi(prices)
            avg_vol = volumes.mean() if len(volumes)>0 else 1
            volx = volumes[-1]/avg_vol if avg_vol>0 else 1

            support = np.percentile(prices,20)
            score = calculate_score(current_price, drop, rsi, volx)

            timestamp = int(time.time()*1000)

            update_github_row(
                github_data,
                coin["symbol"],
                timestamp,
                current_price,
                drop,
                rsi,
                volumes[-1] if len(volumes)>0 else 0,
                volx,
                support,
                score
            )

            return coin["symbol"]

        except:
            return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        updated = list(executor.map(analyze_coin, unique_coins))

    st.success(f"✅ تم تحديث {len([u for u in updated if u])} عملة")

# ==============================
# تشغيل تلقائي
# ==============================
if "last_run" not in st.session_state:
    st.session_state.last_run = 0

if time.time() - st.session_state.last_run > 600:
    run_collector()
    st.session_state.last_run = time.time()

if st.button("🔄 تحديث البيانات"):
    run_collector()

# ==============================
# عرض البيانات
# ==============================
github_data = load_github_data()
df = pd.DataFrame(github_data)

if not df.empty:
    df["target"] = (df["price"].shift(-3) > df["price"]).astype(int)
    df_ai = df.dropna()

    if len(df_ai) > 10:
        X = df_ai[["rsi","score"]]
        y = df_ai["target"]

        model = RandomForestClassifier()
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
        model.fit(X_train,y_train)

        acc = model.score(X_test,y_test)

        latest = df.sort_values("timestamp").groupby("coin").tail(1)
        X_latest = latest[["rsi","score"]]

        try:
            probs = model.predict_proba(X_latest)[:,1]
            latest["Chance %"] = probs*100
        except:
            latest["Chance %"] = 0

        def get_signal(score):
            if score>=10: return "🚀 STRONG BUY"
            elif score>=8: return "🔥 BUY"
            elif score>=6: return "⏳ EARLY"
            elif score>=4: return "⏳ WAIT"
            else: return "❌ NO"

        latest["Signal"] = latest["score"].apply(get_signal)

        latest = latest.sort_values("Chance %", ascending=False)

        st.success(f"دقة الموديل: {round(acc*100,2)}%")
        st.dataframe(latest, use_container_width=True)
