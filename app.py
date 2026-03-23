import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from github import Github
import json

st.set_page_config(layout="wide")
st.title("🚀 Smart Scanner AI")

# ==============================
# GitHub setup via Streamlit Secrets
# ==============================
GITHUB_TOKEN = st.secrets["GITHUB"]["TOKEN"]
REPO_NAME = st.secrets["GITHUB"]["REPO"]
BRANCH = st.secrets["GITHUB"]["BRANCH"]
FILE_PATH = "data.json"

g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)   # ✅ التعديل هنا

# ==============================
# Helper functions
# ==============================
def load_github_data():
    try:
        contents = repo.get_contents(FILE_PATH, ref=BRANCH)
        data = json.loads(contents.decoded_content.decode())
    except:
        data = []
    return data

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
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    if avg_loss == 0: return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1+rs))

# ==============================
# Data Collector
# ==============================
def get_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency":"usd","order":"volume_desc","per_page":50,"page":1}
    return requests.get(url, params=params).json()

def fetch_coin_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": 30}
    return requests.get(url, params=params).json()

def calculate_score(price, drop, rsi, volx):
    score = 0
    if drop < -25: score += 2
    if rsi < 35: score += 2
    if volx > 1.5: score += 2
    return score

def update_github_row(data, coin, timestamp, price, drop, rsi, volume, volx, support, score):
    found = False
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
            found = True
            break
    if not found:
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

def run_collector():
    st.info("⏳ جاري تحديث البيانات…")
    coins = get_coins()
    github_data = load_github_data()

    def analyze_coin(coin):
        try:
            coin_id = coin["id"]
            data = fetch_coin_data(coin_id)
            prices = np.array([p[1] for p in data.get("prices",[])])
            volumes = np.array([v[1] for v in data.get("total_volumes",[])])
            if len(prices) == 0: return None

            current_price = prices[-1]
            max_price = prices.max()
            drop = ((current_price - max_price)/max_price)*100
            rsi = calculate_rsi(prices[-min(15,len(prices)):])
            avg_vol = volumes[:-1].mean() if len(volumes)>1 else volumes[-1]
            volx = volumes[-1]/avg_vol if avg_vol>0 else 1
            support = np.percentile(prices[-min(20,len(prices)):],20)
            score = calculate_score(current_price, drop, rsi, volx)
            timestamp = int(time.time()*1000)

            update_github_row(
                github_data,
                coin["symbol"].upper(),
                timestamp,
                current_price,
                drop,
                rsi,
                volumes[-1] if len(volumes)>0 else 0,
                volx,
                support,
                score
            )

            return coin["symbol"].upper()
        except:
            return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        updated = list(executor.map(analyze_coin, coins))

    st.success(f"✅ تم تحديث {len([u for u in updated if u])} عملة")

if st.button("🔄 تحديث البيانات"):
    run_collector()

# ==============================
# Load & AI
# ==============================
github_data = load_github_data()
df = pd.DataFrame(github_data)

if not df.empty:
    df["target"] = (df["price"].shift(-3) > df["price"]).astype(int)
    df_ai = df.dropna()

    if not df_ai.empty:
        X = df_ai[["rsi","score"]]
        y = df_ai["target"]

        model = RandomForestClassifier()
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
        model.fit(X_train,y_train)
        acc = model.score(X_test,y_test)

        latest = df.sort_values("timestamp").groupby("coin").tail(1)
        probs = model.predict_proba(latest[["rsi","score"]])[:,1]
        latest["Chance %"] = probs*100

        def get_signal(score):
            if score>=10: return "🚀 STRONG BUY"
            elif score>=8: return "🔥 BUY"
            elif score>=6: return "⏳ EARLY"
            elif score>=4: return "⏳ WAIT"
            else: return "❌ NO"

        latest["Signal"] = latest["score"].apply(get_signal)

        counts = df.groupby("coin").size()
        latest["Data Status"] = latest["coin"].apply(
            lambda c: "🟩 كافي" if counts.get(c,0)>=20 else "🟨 متوسط" if counts.get(c,0)>=10 else "🟥 قليل"
        )

        latest = latest.sort_values("Chance %", ascending=False)

        st.success(f"دقة الموديل: {round(acc*100,2)}%")
        st.dataframe(latest[["coin","price","drop_percent","rsi","volx","support","score","Signal","Chance %","Data Status"]], use_container_width=True)
