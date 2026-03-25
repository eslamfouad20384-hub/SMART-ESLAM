import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from github import Github, Auth
import json

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Scanner AI PRO MAX (AI Enabled)")

MODEL_FILE = "model.pkl"

# ==============================
# GitHub
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
# Indicators SAFE
# ==============================
def rsi(prices):
    if len(prices)<2: return 50
    delta=np.diff(prices)
    gain=np.maximum(delta,0)
    loss=np.abs(np.minimum(delta,0))
    rs = pd.Series(gain).mean()/(pd.Series(loss).mean()+1e-9)
    return 100-(100/(1+rs))

def macd(prices):
    if len(prices)<5: return 0,0
    s=pd.Series(prices)
    m=s.ewm(12).mean()-s.ewm(26).mean()
    sig=m.ewm(9).mean()
    return m.iloc[-1],sig.iloc[-1]

def atr(prices,high,low):
    if len(prices)<2: return 0
    df=pd.DataFrame({"c":prices,"h":high,"l":low})
    df["pc"]=df["c"].shift(1)
    df["tr"]=df.apply(lambda x:max(x["h"]-x["l"],abs(x["h"]-x["pc"]),abs(x["l"]-x["pc"])),axis=1)
    return df["tr"].mean()

def obv(prices,vol):
    if len(prices)<2: return 0
    o=0
    for i in range(1,len(prices)):
        if prices[i]>prices[i-1]: o+=vol[i]
        elif prices[i]<prices[i-1]: o-=vol[i]
    return o

# ==============================
# Update Data
# ==============================
def update():
    coins=requests.get("https://api.coingecko.com/api/v3/coins/markets",
                       params={"vs_currency":"usd","per_page":50}).json()
    data=[]
    for c in coins:
        try:
            d=requests.get(f"https://api.coingecko.com/api/v3/coins/{c['id']}/market_chart",
                           params={"vs_currency":"usd","days":30}).json()
            prices=[p[1] for p in d["prices"]]
            vols=[v[1] for v in d["total_volumes"]]

            candles=[]
            for i in range(len(prices)):
                candles.append({
                    "price":prices[i],
                    "high":prices[i]*1.01,
                    "low":prices[i]*0.99,
                    "volume":vols[i] if i<len(vols) else 0
                })

            data.append({"coin":c["symbol"].upper(),"candles":candles})
        except:
            pass

    save_data(data)
    st.success("Updated")

if st.button("🔄 تحديث"):
    update()

data=load_data()

# ==============================
# Build AI Dataset
# ==============================
rows=[]
for d in data:
    c=d["candles"]
    if len(c)<10: continue

    prices=np.array([x["price"] for x in c])
    vol=np.array([x["volume"] for x in c])
    high=np.array([x.get("high",x["price"]*1.01) for x in c])
    low=np.array([x.get("low",x["price"]*0.99) for x in c])

    for i in range(5,len(prices)-2):
        r=rsi(prices[:i])
        m,s=macd(prices[:i])
        a=atr(prices[:i],high[:i],low[:i])
        o=obv(prices[:i],vol[:i])

        target=1 if prices[i+2]>prices[i] else 0

        rows.append([r,m-s,a,o,target])

df_ai=pd.DataFrame(rows,columns=["rsi","macd","atr","obv","target"])

# ==============================
# Train or Load Model
# ==============================
model=None
if len(df_ai)>50:
    X=df_ai[["rsi","macd","atr","obv"]]
    y=df_ai["target"]

    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE,"rb") as f:
            model=pickle.load(f)
    else:
        model=RandomForestClassifier(n_estimators=150)
        model.fit(X,y)
        with open(MODEL_FILE,"wb") as f:
            pickle.dump(model,f)

# ==============================
# Display Table
# ==============================
rows=[]
for d in data:
    c=d["candles"]
    if len(c)<2: continue

    prices=np.array([x["price"] for x in c])
    vol=np.array([x["volume"] for x in c])
    high=np.array([x.get("high",x["price"]*1.01) for x in c])
    low=np.array([x.get("low",x["price"]*0.99) for x in c])

    r=rsi(prices)
    m,s=macd(prices)
    a=atr(prices,high,low)
    o=obv(prices,vol)

    score=0
    if r<35: score+=2
    if m>s: score+=2
    if a>0.5: score+=1
    if o>0: score+=1

    chance=0
    if model and len(c)>=10:
        chance=model.predict_proba([[r,m-s,a,o]])[0][1]*100

    status="🔴 Low" if len(c)<20 else "🟡 Mid" if len(c)<30 else "🟢 Good"

    rows.append({
        "Coin":d["coin"],
        "Price":round(prices[-1],2),
        "RSI":round(r,2),
        "ATR":round(a,2),
        "OBV":round(o,2),
        "Score":score,
        "Chance %":round(chance,2),
        "Data":status
    })

df=pd.DataFrame(rows).sort_values("Chance %",ascending=False)
st.dataframe(df,use_container_width=True)
