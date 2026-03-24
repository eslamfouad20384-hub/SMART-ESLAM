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
st.title("🚀 Smart Crypto Scanner AI PRO - Full Version")

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
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        st.error("⚠️ فشل جلب العملات من CoinGecko")
        return []

def fetch_data(coin_id, days=30):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency":"usd","days":days}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return {"prices":[],"total_volumes":[]}

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
            row["candles"] = sorted(all_c.values(), key=lambda x:x["timestamp"])[-90:]  # حفظ أكبر عدد شمعات
            save_github_data(data)
            return
    data.append({"coin":coin,"candles":candles[-90:]})
    save_github_data(data)

# ==============================
# Collector
# ==============================
def run_collector():
    st.info("⏳ تحديث آمن...")
    coins = get_coins()
    if not coins:
        return
    data = load_github_data()

    for i in range(0, len(coins), 10):
        batch = coins[i:i+10]

        def work(c):
            try:
                symbol = c["symbol"].upper()
                if not should_update(data, symbol):
                    return None

                d = fetch_data(c["id"], days=30)
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
# AI Analysis
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

        # ==============================
        # مؤشرات إضافية
        # ==============================
        rsi_prev = calculate_rsi(prices[i-15:i])
        rsi_cond = rsi < 35 and rsi > rsi_prev
        recent_prices = prices[i-20:i]
        support_zone = np.percentile(recent_prices, 20)
        near_support = prices[i] <= support_zone * 1.1
        ema_20 = pd.Series(prices[i-20:i]).ewm(span=20).mean().values
        ema_cond = prices[i] > ema_20[-1]
        higher_low = prices[i-5:i].min() < prices[i]
        last = prices[i]
        prev = prices[i-1]
        prev2 = prices[i-2]
        bullish_engulfing = (prev < prev2) and (last > prev) and (last > prev2)
        body = abs(last - prev)
        lower_shadow = abs(prev - prev2)
        hammer = lower_shadow > body * 2
        candle_signal = bullish_engulfing or hammer
        strong_drop = drop < -40
        bounce_started = last > prices[i-3]
        smart_reversal = strong_drop and bounce_started

        score = 0
        if drop < -25: score += 2
        if rsi_cond: score += 2
        if volx > 1.5: score += 2
        if near_support: score += 2
        if ema_cond or higher_low: score += 2
        if candle_signal: score += 2
        if smart_reversal: score += 2

        target = 1 if prices[i+3] > prices[i] else 0

        rows.append({
            "coin": coin,
            "rsi": rsi,
            "score": score,
            "target": target
        })

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
        if len(candles) < 15:
            continue
        prices = np.array([c["price"] for c in candles])
        vols = np.array([c["volume"] for c in candles])
        rsi = calculate_rsi(prices[-15:])
        drop = ((prices[-1] - prices.max()) / prices.max()) * 100
        avg_vol = vols[-10:].mean()
        volx = vols[-1] / avg_vol if avg_vol>0 else 1
        rsi_prev = calculate_rsi(prices[-16:-1])
        rsi_cond = rsi < 35 and rsi > rsi_prev
        recent_prices = prices[-20:]
        support_zone = np.percentile(recent_prices, 20)
        near_support = prices[-1] <= support_zone * 1.1
        ema_20 = pd.Series(prices[-20:]).ewm(span=20).mean().values
        ema_cond = prices[-1] > ema_20[-1]
        higher_low = prices[-5:].min() < prices[-1]
        last = prices[-1]
        prev = prices[-2]
        prev2 = prices[-3]
        bullish_engulfing = (prev < prev2) and (last > prev) and (last > prev2)
        body = abs(last - prev)
        lower_shadow = abs(prev - prev2)
        hammer = lower_shadow > body * 2
        candle_signal = bullish_engulfing or hammer
        strong_drop = drop < -40
        bounce_started = last > prices[-3]
        smart_reversal = strong_drop and bounce_started
        score = 0
        if drop < -25: score += 2
        if rsi_cond: score += 2
        if volx > 1.5: score += 2
        if near_support: score += 2
        if ema_cond or higher_low: score += 2
        if candle_signal: score += 2
        if smart_reversal: score += 2
        if len(candles) >= 30:
            status = "🟢 Good"
        elif len(candles) >= 20:
            status = "🟡 Moderate"
        else:
            status = "🔴 Low"
        if score >= 6 or (score >= 4 and drop < -5):
            rec = "Strong Buy" if score >= 6 else "Buy"
        elif score >= 3:
            rec = "Hold"
        else:
            rec = "No"
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
    def get_fear_greed_index():
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            r = requests.get(url).json()
            value = int(r["data"][0]["value"])
            if value < 40:
                emoji = "😨"
            elif value < 60:
                emoji = "😐"
            else:
                emoji = "😎"
            return value, emoji
        except:
            return None, ""

    def get_market_trend(latest_df):
        if latest_df.empty:
            return "❓"
        avg_score = latest_df["Score"].mean()
        if avg_score >= 7:
            return "🚀 صاعد"
        elif avg_score >= 4:
            return "⏸️ عرضي"
        else:
            return "📉 هابط"

    fear_value, fear_emoji = get_fear_greed_index()
    market_trend = get_market_trend(latest_df)

    st.markdown(f"### مؤشر الخوف والطمع: {fear_value} {fear_emoji}   |   حالة السوق العام: {market_trend}")
    st.dataframe(latest_df, use_container_width=True)

else:
    st.warning("⚠️ البيانات غير كافية للـ AI")
