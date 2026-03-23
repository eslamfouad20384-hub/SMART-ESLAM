import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from github import Github
import json

st.set_page_config(layout="wide")
st.title("🚀 Scanner AI")

# ==============================
# GitHub setup via Streamlit Secrets
# ==============================
# ملف .streamlit/secrets.toml محتوي:
# [GITHUB]
# TOKEN = "ghp_XXXXXXXXXXXX"
# REPO = "eslamfouad20384-hub/SMART-ESLAM"
# BRANCH = "main"

GITHUB_TOKEN = st.secrets["GITHUB"]["TOKEN"]
REPO_NAME = st.secrets["GITHUB"]["REPO"]
BRANCH = st.secrets["GITHUB"]["BRANCH"]
FILE_PATH = "data.json"

g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)

# ==============================
# Helper functions for GitHub
# ==============================
def load_github_data():
    try:
        contents = repo.get_contents(FILE_PATH, ref=BRANCH)
        data = json.loads(contents.decoded_content.decode())
    except:
        data = []  # file not exist yet
    return data

def save_github_data(data):
    content = json.dumps(data, indent=4)
    try:
        contents = repo.get_contents(FILE_PATH, ref=BRANCH)
        repo.update_file(contents.path, "Update data", content, contents.sha, branch=BRANCH)
    except:
        repo.create_file(FILE_PATH, "Create data", content, branch=BRANCH)

# ==============================
# RSI calculation
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
# Collector functions
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
    # تحديث العملة إذا موجودة مسبقًا
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
            timestamp = int(data["prices"][-1][0]) if len(data.get("prices",[]))>0 else int(time.time()*1000)
            update_github_row(github_data, coin["symbol"].upper(), timestamp, current_price, drop, rsi, volumes[-1] if len(volumes)>0 else 0, volx, support, score)
            return coin["symbol"].upper()
        except:
            return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        updated = list(executor.map(analyze_coin, coins))
    updated_count = len([u for u in updated if u])
    st.success(f"✅ تم تحديث {updated_count} عملة على GitHub")

if st.button("🔄 تحديث البيانات"):
    run_collector()

# ==============================
# Load data for display & AI
# ==============================
github_data = load_github_data()
df = pd.DataFrame(github_data)

# ==============================
# آخر تحديث لكل عملة
# ==============================
if not df.empty:
    st.subheader("🕒 آخر تحديث لكل عملة")
    last_update = df.groupby("coin")["timestamp"].max().reset_index()
    last_update["last_time"] = pd.to_datetime(last_update["timestamp"], unit='ms')
    st.dataframe(last_update.sort_values("last_time", ascending=False), use_container_width=True)

# ==============================
# AI + Signals
# ==============================
if not df.empty:
    # Train AI
    df["target"] = (df["price"].shift(-3) > df["price"]).astype(int)
    df_ai = df.dropna()
    X = df_ai[["rsi","score"]]
    y = df_ai["target"]
    model = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    model.fit(X_train,y_train)
    acc = model.score(X_test,y_test)

    # Latest rows + Chance %
    latest = df.sort_values("timestamp").groupby("coin").tail(1)
    X_latest = latest[["rsi","score"]]
    try:
        probs = model.predict_proba(X_latest)[:,1]
        latest["Chance %"] = probs*100
    except:
        latest["Chance %"] = 0

    # Signal
    def get_signal(score):
        if score>=10: return "🚀 STRONG BUY"
        elif score>=8: return "🔥 BUY"
        elif score>=6: return "⏳ EARLY"
        elif score>=4: return "⏳ WAIT"
        else: return "❌ NO"
    latest["Signal"] = latest["score"].apply(get_signal)

    # Data Status
    counts = df.groupby("coin").size()
    def status_color(n):
        if n>=20: return "🟩 كافي"
        elif n>=10: return "🟨 متوسط"
        else: return "🟥 قليل"
    latest["Data Status"] = latest["coin"].apply(lambda c: status_color(counts.get(c,0)))

    # Display table
    latest = latest.sort_values("Chance %", ascending=False)
    st.success(f"دقة الموديل: {round(acc*100,2)}%")
    st.dataframe(latest[["coin","price","drop_percent","rsi","volx","support","score","Signal","Chance %","Data Status"]], use_container_width=True)

    # تفاصيل أي عملة
    coin_list = latest["coin"].unique().tolist()
    selected_coin = st.selectbox("اختار عملة للتفاصيل", coin_list)
    if selected_coin:
        st.subheader(f"📊 بيانات {selected_coin}")
        st.dataframe(df[df["coin"]==selected_coin].sort_values("timestamp", ascending=False), use_container_width=True)
