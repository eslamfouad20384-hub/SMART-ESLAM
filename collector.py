import requests
import numpy as np
import time
from database import create_tables, connect
from indicators import calculate_rsi

create_tables()

def get_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "volume_desc",
        "per_page": 50,
        "page": 1
    }
    return requests.get(url, params=params).json()

def fetch_coin_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": 30}
    return requests.get(url, params=params).json()

def save_row(coin, timestamp, price, volume, rsi, score):
    conn = connect()
    c = conn.cursor()

    c.execute("""
    INSERT INTO market_data (coin, timestamp, price, volume, rsi, score)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (coin, timestamp, price, volume, rsi, score))

    conn.commit()
    conn.close()

def calculate_score(price, max_price, rsi, volume, avg_volume):
    score = 0

    drop = ((price - max_price) / max_price) * 100

    if drop < -25:
        score += 2
    if rsi < 35:
        score += 2
    if volume > avg_volume * 1.5:
        score += 2

    return score

while True:
    try:
        coins = get_coins()

        for coin in coins:
            coin_id = coin["id"]

            data = fetch_coin_data(coin_id)

            prices = np.array([p[1] for p in data["prices"]])
            volumes = np.array([v[1] for v in data["total_volumes"]])

            if len(prices) < 30:
                continue

            current_price = prices[-1]
            max_price = prices.max()

            rsi = calculate_rsi(prices[-15:])

            avg_volume = volumes[:-1].mean()
            current_volume = volumes[-1]

            score = calculate_score(current_price, max_price, rsi, current_volume, avg_volume)

            timestamp = int(data["prices"][-1][0])

            save_row(coin["symbol"], timestamp, current_price, current_volume, rsi, score)

            print(f"{coin['symbol']} updated ✅")

            time.sleep(1)  # حماية من البلوك

    except Exception as e:
        print("Error:", e)

    time.sleep(600)  # كل 10 دقايق
