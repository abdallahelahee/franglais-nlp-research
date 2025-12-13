import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocessing import preprocess_text
from embeddings import embed_texts

def train_classifier(csv_path):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Minimal preprocessing
    df["clean"] = df["text"].apply(preprocess_text)

    # Embed
    X = embed_texts(df["clean"].tolist())

    # Map labels
    label_map = {"en": 0, "fr": 1, "mix": 2}
    y = df["label"].map(label_map).values

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    preds = clf.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, preds, target_names=label_map.keys()))

    return clf, label_map
