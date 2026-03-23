import streamlit as st
import pandas as pd
import os
import sqlite3

# ==============================
# إعدادات
# ==============================
DB_NAME = "crypto.db"

st.set_page_config(layout="wide")
st.title("🚀 Smart Crypto Scanner AI (نسخة آمنة)")

# ==============================
# التأكد من وجود قاعدة البيانات والجدول
# ==============================
def create_tables():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS market_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        coin TEXT,
        timestamp INTEGER,
        price REAL,
        volume REAL,
        rsi REAL,
        score REAL
    )
    """)
    conn.commit()
    conn.close()

if not os.path.exists(DB_NAME):
    create_tables()
else:
    # في حالة DB موجود بس ممكن الجدول مش موجود
    create_tables()

# ==============================
# قراءة البيانات
# ==============================
try:
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    df = pd.read_sql("SELECT * FROM market_data", conn)
    conn.close()
except Exception as e:
    st.error("❌ حصل خطأ أثناء قراءة قاعدة البيانات.")
    st.stop()

# ==============================
# التحقق من البيانات
# ==============================
if df.empty:
    st.warning("لسه مفيش داتا. شغل Collector الأول.")
else:
    # عرض آخر سعر لكل عملة
    latest = df.sort_values("timestamp").groupby("coin").tail(1)

    # إذا كنت شغال مع AI ممكن تضيف Prediction لاحقًا
    latest["Chance %"] = latest["score"] * 8  # مجرد تقدير مؤقت

    latest = latest.sort_values("Chance %", ascending=False)

    st.success(f"تم العثور على {len(latest)} عملة حديثة في قاعدة البيانات")
    st.dataframe(latest[["coin","price","rsi","score","Chance %"]], use_container_width=True)
