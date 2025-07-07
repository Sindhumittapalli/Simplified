# Simplified
utils.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def fetch_user_data(usernames):
    # Mock function or Tweepy API call
    return [{"username": u, "followers": 10, "following": 100, "tweets": 50} for u in usernames]

def extract_features(users):
    df = pd.DataFrame(users)
    df["ff_ratio"] = df["followers"] / (df["following"] + 1)
    df["label"] = [0, 1]  # 0: real, 1: fake (for testing)
    return df

def train_model(df):
    X = df.drop("label", axis=1)
    y = df["label"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model
