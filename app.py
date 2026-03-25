import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from github import Github, Auth
import json

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Scanner AI PRO MAX (Final Safe Version)")

MODEL_FILE = "model.pkl"

# ==============================
# GitHub Setup
# ==============================
g = Github(auth=Auth.Token(st.secrets["GITHUB"]["TOKEN"]))
repo = g.get_repo(st.secrets["GITHUB"]["REPO"])
BRANCH = st.secrets["GITHUB"]["BRANCH"]
FILE_PATH = "data.json"

def load_data():
    try:
        contents = repo.get_contents(FILE_PATH, ref=BRANCH)
        return json.loads(contents.decoded_content.decode())
    except:
        return []

def save_data(data):
    content = json.dumps(data, indent=4)
    try:
        contents = repo.get_contents(FILE_PATH, ref=BRANCH)
        repo.update_file(contents.path, "update", content, contents.sha, branch=BRANCH)
    except:
        repo.create_file(FILE_PATH, "create", content, branch=BRANCH)

# ==============================
# Indicators
# ==============================
def rsi(prices):
    if len(prices) < 2: return 50
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = np.abs(np.minimum(delta, 0))
    rs = pd.Series(gain).mean() / (pd.Series(loss).mean() + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(prices):
    if len(prices) < 5: return 0, 0
    s = pd.Series(prices)
    m = s.ewm(span=12).mean() - s.ewm(span=26).mean()
    sig = m.ewm(span=9).mean()
    return m.iloc[-1], sig.iloc[-1]

def atr(prices, high, low):
    if len(prices) < 2: return 0
    df = pd.DataFrame({"c": prices, "h": high, "l": low})
    df["pc"] = df["c"].shift(1)
    df["tr"] = df.apply(lambda x: max(
        x["h"] - x["l"],
        abs(x["h"] - x["pc"]),
        abs(x["l"] - x["pc"])
    ), axis=1)
    return df["tr"].mean()

def obv(prices, vol):
    if len(prices) < 2: return 0
    o = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            o += vol[i]
        elif prices[i] < prices[i-1]:
            o -= vol[i]
    return o

def support_resistance(prices):
    if len(prices) < 1:
        return 0, 0
    return np.min(prices[-20:]), np.max(prices[-20:])

def volume_x(vol):
    if len(vol) < 5: return 1
    avg = np.mean(vol[-5:])
    return vol[-1] / avg if avg > 0 else 1

# ==============================
# Update Data
# ==============================
def update():
    st.info("⏳ Updating...")
    coins = requests.get(
        "https://api.coingecko.com/api/v3/coins/markets",
        params={"vs_currency": "usd", "per_page": 50}
    ).json()

    data = []

    for c in coins:
        try:
            d = requests.get(
                f"https://api.coingecko.com/api/v3/coins/{c['id']}/market_chart",
                params={"vs_currency": "usd", "days": 30}
            ).json()

            prices = [p[1] for p in d.get("prices", [])]
            vols = [v[1] for v in d.get("total_volumes", [])]

            candles = []
            for i in range(len(prices)):
                candles.append({
                    "price": prices[i],
                    "high": prices[i]*1.01,
                    "low": prices[i]*0.99,
                    "volume": vols[i] if i < len(vols) else 0
                })

            data.append({
                "coin": c["symbol"].upper(),
                "candles": candles
            })

        except:
            continue

    save_data(data)
    st.success("✅ Updated Successfully")

if st.button("🔄 تحديث البيانات"):
    update()

# ==============================
# Load Data
# ==============================
data = load_data()

# ==============================
# Build AI Dataset
# ==============================
rows_ai = []

for d in data:
    c = d.get("candles", [])
    if len(c) < 10: continue

    prices = np.array([x["price"] for x in c])
    vol = np.array([x["volume"] for x in c])
    high = np.array([x.get("high", x["price"]*1.01) for x in c])
    low = np.array([x.get("low", x["price"]*0.99) for x in c])

    for i in range(5, len(prices)-2):
        r = rsi(prices[:i])
        m, s = macd(prices[:i])
        a = atr(prices[:i], high[:i], low[:i])
        o = obv(prices[:i], vol[:i])
        target = 1 if prices[i+2] > prices[i] else 0
        rows_ai.append([r, m-s, a, o, target])

df_ai = pd.DataFrame(rows_ai, columns=["rsi","macd","atr","obv","target"])

# ==============================
# Train or Load Model
# ==============================
model = None
if len(df_ai) > 50:
    X = df_ai[["rsi","macd","atr","obv"]]
    y = df_ai["target"]
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE,"rb") as f: model = pickle.load(f)
    else:
        model = RandomForestClassifier(n_estimators=150)
        model.fit(X,y)
        with open(MODEL_FILE,"wb") as f: pickle.dump(model,f)

# ==============================
# Coloring function
# ==============================
def color_cells(val, col):
    try:
        if col=="Vol x":
            if val>1.5: return "background-color: green; color: white"
            elif val>1: return "background-color: orange; color: black"
            else: return "background-color: red; color: white"
        elif col=="ATR":
            if val>1: return "background-color: green; color: white"
            elif val>0.5: return "background-color: orange; color: black"
            else: return "background-color: red; color: white"
        elif col=="OBV":
            if val>0: return "background-color: green; color: white"
            elif val==0: return "background-color: orange; color: black"
            else: return "background-color: red; color: white"
    except: return ""
    return ""

# ==============================
# Build final table
# ==============================
rows = []

for d in data:
    c = d.get("candles", [])
    if len(c)<1: continue

    prices = np.array([x["price"] for x in c])
    vol = np.array([x["volume"] for x in c])
    high = np.array([x.get("high", x["price"]*1.01) for x in c])
    low = np.array([x.get("low", x["price"]*0.99) for x in c])

    r = rsi(prices)
    m, s = macd(prices)
    a = atr(prices, high, low)
    o = obv(prices, vol)
    volx = volume_x(vol)
    sup,res = support_resistance(prices)

    score = 0
    if r<35: score+=2
    if m>s: score+=2
    if a>0.5: score+=1
    if o>0: score+=1
    if volx>1.5: score+=2

    chance = 0
    if model and len(c)>=10:
        chance = model.predict_proba([[r,m-s,a,o]])[0][1]*100

    # Signal
    if score>=6: signal="🔥 Strong Buy"
    elif score>=4: signal="🚀 Buy"
    elif score>=2: signal="🟠 Hold"
    else: signal="❌ No Trade"

    # Data status
    if len(c)>=30: status="🟢 Good"
    elif len(c)>=20: status="🟡 Moderate"
    else: status="🔴 Low"

    rows.append({
        "Coin": d["coin"],
        "Price": round(prices[-1],2),
        "RSI": round(r,2),
        "Vol x": round(volx,2),
        "ATR": round(a,2),
        "OBV": round(o,2),
        "Support": round(sup,2),
        "Resistance": round(res,2),
        "Score": score,
        "Chance %": round(chance,2),
        "Signal": signal,
        "Data": status
    })

df = pd.DataFrame(rows)

# ==============================
# Safe sort
# ==============================
if "Chance %" in df.columns:
    df = df.sort_values("Chance %", ascending=False)

# ==============================
# Style
# ==============================
styled_df = df.style.apply(
    lambda x: [
        color_cells(x["Vol x"],"Vol x"),
        color_cells(x["ATR"],"ATR"),
        color_cells(x["OBV"],"OBV")
    ],
    axis=1
)

st.dataframe(styled_df,use_container_width=True)
