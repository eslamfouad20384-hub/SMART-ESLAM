import streamlit as st
import requests
import json
import base64
import time

# =========================
# 🔑 GitHub Settings
# =========================
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
REPO = "eslamfouad20384-hub/SMART-ESLAM"
FILE_PATH = "data.json"
BRANCH = "data"

# =========================
# 📥 Load from GitHub
# =========================
def load_json():
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    for branch in ["data", "main"]:
        url = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}?ref={branch}"
        r = requests.get(url, headers=headers)

        if r.status_code == 200:
            content = r.json()["content"]
            decoded = base64.b64decode(content).decode("utf-8")
            return json.loads(decoded)

    return []

# =========================
# 📤 Save to GitHub
# =========================
def save_json(data):
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    url_get = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}?ref={BRANCH}"
    r = requests.get(url_get, headers=headers)

    sha = r.json().get("sha") if r.status_code == 200 else None

    content = base64.b64encode(
        json.dumps(data, indent=4).encode("utf-8")
    ).decode("utf-8")

    payload = {
        "message": "update data.json",
        "content": content,
        "branch": BRANCH
    }

    if sha:
        payload["sha"] = sha

    url_put = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"
    res = requests.put(url_put, headers=headers, json=payload)

    return res.status_code in [200, 201]

# =========================
# 🧠 Generate ID
# =========================
def generate_id(symbol, direction, price):
    return f"{symbol}_{direction}_{price}"

# =========================
# 🔄 Add or Update Signal (No duplicates)
# =========================
def upsert_signal(data, new_signal):
    signal_id = generate_id(
        new_signal["symbol"],
        new_signal["direction"],
        new_signal["entry_price"]
    )

    for item in data:
        if item.get("id") == signal_id:
            item.update(new_signal)
            item["id"] = signal_id
            item["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            return data, "updated"

    new_signal["id"] = signal_id
    new_signal["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data.append(new_signal)

    return data, "added"

# =========================
# 🧠 Dynamic Market Update (NO Binance)
# =========================
def dynamic_update(signal, current_price):
    entry = signal["entry_price"]
    sl = signal.get("stop_loss")

    if signal["direction"] == "LONG":
        move = current_price - entry
    else:
        move = entry - current_price

    # 📈 confidence update
    if move > 0:
        signal["confidence"] = min(100, signal.get("confidence", 50) + 2)

        # trailing stop
        if sl:
            if signal["direction"] == "LONG":
                signal["stop_loss"] = max(sl, current_price * 0.99)
            else:
                signal["stop_loss"] = min(sl, current_price * 1.01)

    else:
        signal["confidence"] = max(0, signal.get("confidence", 50) - 3)

    # 🚨 auto close
    if signal.get("confidence", 50) < 30:
        signal["status"] = "CLOSED"

    signal["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    return signal

# =========================
# 🎨 Streamlit UI
# =========================
st.set_page_config(layout="wide")
st.title("🚀 Smart AI Trading System (GitHub + Dynamic)")

data = load_json()

st.subheader("📥 Current Data")
st.write(data)

# =========================
# ➕ Add Signal
# =========================
symbol = st.text_input("Symbol")
direction = st.selectbox("Direction", ["LONG", "SHORT"])
entry_price = st.number_input("Entry Price")
stop_loss = st.number_input("Stop Loss")
take_profit = st.number_input("Take Profit")

if st.button("➕ Add / Update Signal"):

    new_signal = {
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "confidence": 50,
        "status": "OPEN"
    }

    data, status = upsert_signal(data, new_signal)
    save_json(data)

    if status == "updated":
        st.info("🔄 تم تحديث الصفقة")
    else:
        st.success("➕ تم إضافة صفقة جديدة")

# =========================
# 🧠 Manual Dynamic Update
# =========================
st.subheader("🧠 Dynamic Update (Manual Price)")

for signal in data:
    if signal.get("status") == "OPEN":

        current_price = st.number_input(
            f"{signal['symbol']} current price",
            key=signal["id"]
        )

        signal = dynamic_update(signal, current_price)

if st.button("🔄 Save Updates"):
    save_json(data)
    st.success("تم تحديث كل الصفقات 🔥")
