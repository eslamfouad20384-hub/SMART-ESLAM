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
st.title("🚀 Smart Crypto Scanner AI + Daily Candles + GitHub Sync")

# ==============================
# GitHub setup via Streamlit Secrets
# ==============================
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
def get_coins_coingecko():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency":"usd","order":"volume_desc","per_page":50,"page":1}
    try:
        return requests.get(url, params=params).json()
    except:
        return []

def get_coins_cryptocompare():
    url = "https://min-api.cryptocompare.com/data/top/mktcapfull"
    params = {"limit":50,"tsym":"USD"}
    try:
        data = requests.get(url, params=params).json()
        return [{"id":coin["CoinInfo"]["Name"], "symbol":coin["CoinInfo"]["Name"].upper()} for coin in data.get("Data",[])]
    except:
        return []

def get_coins_binance():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        data = requests.get(url).json()
        return [{"id":coin["symbol"], "symbol":coin["symbol"]} for coin in data if coin["symbol"].endswith("USDT")]
    except:
        return []

def fetch_coin_data_daily(coin_id, source="coingecko"):
    if source=="coingecko":
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": 30, "interval":"daily"}
        return requests.get(url, params=params).json()
    return {"prices":[],"total_volumes":[]}

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
    coins = get_coins_coingecko() + get_coins_cryptocompare() + get_coins_binance()
    github_data = load_github_data()

    def analyze_coin(coin):
        try:
            coin_id = coin["id"]
            data = fetch_coin_data_daily(coin_id)
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
        except Exception as e:
            st.warning(f"⚠️ خطأ في تحديث {coin['symbol'].upper()}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        updated = list(executor.map(analyze_coin, coins))
    updated_count = len([u for u in updated if u])
    st.success(f"✅ تم تحديث {updated_count} عملة على GitHub")

# ==============================
# Auto run every 10 minutes
# ==============================
if "last_run" not in st.session_state:
    st.session_state.last_run = 0

if time.time() - st.session_state.last_run > 600:  # 600 ثانية = 10 دقايق
    run_collector()
    st.session_state.last_run = time.time()

if st.button("🔄 تحديث البيانات"):
    run_collector()

# ==============================
# Load data for display & AI
# ==============================
github_data = load_github_data()
df = pd.DataFrame(github_data)

if df.empty:
    st.warning("⚠️ مفيش بيانات لعرضها دلوقتي")
else:
    # ==============================
    # AI + إشارات + جدول نهائي
    # ==============================
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
            latest["Chance %"] = probs * 100
        except:
            latest["Chance %"] = 0

        # ==============================
        # إشارات التداول (4 حالات)
        # ==============================
        def get_signal(score):
            if score >= 6:
                return "🚀 شراء قوي"
            elif score >= 4:
                return "🔥 شراء"
            elif score >= 2:
                return "⏳ انتظار"
            else:
                return "❌ رفض"

        latest["Signal"] = latest["score"].apply(get_signal)

        # ==============================
        # حالة البيانات (🟩🟨🟥)
        # ==============================
        counts = df.groupby("coin").size()
        def status_color(n):
            if n >= 20:
                return "🟩 كافي"
            elif n >= 10:
                return "🟨 متوسط"
            else:
                return "🟥 قليل"
        latest["Data Status"] = latest["coin"].apply(lambda c: status_color(counts.get(c, 0)))

        # ==============================
        # ترتيب الجدول
        # ==============================
        latest = latest.sort_values("Chance %", ascending=False)
        st.success(f"دقة الموديل: {round(acc*100,2)}%")

        # ==============================
        # تجهيز وعرض الجدول النهائي
        # ==============================
        display_df = latest[[
            "coin","price","drop_percent","rsi","volx","support","score","Signal","Chance %","Data Status"
        ]].rename(columns={
            "coin": "العملة",
            "price": "السعر",
            "drop_percent": "% الهبوط",
            "rsi": "RSI",
            "volx": "Vol X",
            "support": "الدعم",
            "score": "Score",
            "Signal": "الإشارة",
            "Chance %": "احتمال الصعود %",
            "Data Status": "حالة البيانات"
        })

        def color_signal(val):
            if "شراء قوي" in val:
                return "background-color: green; color: white"
            elif "شراء" in val:
                return "background-color: darkgreen; color: white"
            elif "انتظار" in val:
                return "background-color: orange"
            else:
                return "background-color: red; color: white"

        def color_data_status(val):
            if "🟩" in val:
                return "background-color: green; color: white"
            elif "🟨" in val:
                return "background-color: orange"
            else:
                return "background-color: red; color: white"

        st.dataframe(
            display_df.style
            .applymap(color_signal, subset=["الإشارة"])
            .applymap(color_data_status, subset=["حالة البيانات"]),
            use_container_width=True
        )

    else:
        st.warning("⚠️ البيانات غير كافية لتشغيل AI")
