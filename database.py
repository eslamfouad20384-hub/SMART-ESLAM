import sqlite3

DB_NAME = "crypto.db"

def connect():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def create_tables():
    conn = connect()
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
