import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model():
    df = pd.read_csv("train.csv")
    x = df["text"]
    y = df["label"]

    tfidf = TfidfVectorizer(stop_words ="english", max_df = 0.7)
    x_tfidf = tfidf.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_tfidf, y, test_size = 0.2, random_state = 42)
    model = PassiveAggressiveClassifier()
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    print(f"accuracy: {acc:.2f}")
    joblib.dump(model, "model.pkl")
    joblib.dump(tfidf,"vectorized.pkl")
    
if __name__ == "__main__":
        train_model()
