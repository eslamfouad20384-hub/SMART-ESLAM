import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from database import connect

def train_model():
    conn = connect()
    df = pd.read_sql("SELECT * FROM market_data", conn)
    conn.close()

    df["target"] = (df["price"].shift(-3) > df["price"]).astype(int)

    df = df.dropna()

    X = df[["rsi", "score"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    return model, acc
