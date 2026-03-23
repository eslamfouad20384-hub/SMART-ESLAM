import streamlit as st
import pandas as pd
from database import connect
from ai_model import train_model

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Scanner AI")

conn = connect()

df = pd.read_sql("SELECT * FROM market_data", conn)

if df.empty:
    st.warning("لسه مفيش داتا... شغل collector الأول")
else:
    model, acc = train_model()

    latest = df.sort_values("timestamp").groupby("coin").tail(1)

    X = latest[["rsi", "score"]]

    probs = model.predict_proba(X)[:,1]

    latest["Chance %"] = probs * 100

    latest = latest.sort_values("Chance %", ascending=False)

    st.success(f"دقة الموديل: {round(acc*100,2)}%")

    st.dataframe(latest[["coin","price","rsi","score","Chance %"]], use_container_width=True)
