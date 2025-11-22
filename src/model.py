import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(data: pd.DataFrame, feature_cols):
    X = data[feature_cols]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    baseline_pred = np.ones_like(y_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    report = classification_report(y_test, y_pred, digits=3)

    results = {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": acc,
        "baseline_accuracy": baseline_acc,
        "classification_report": report,
    }

    return results

def predict_tomorrow(model, latest_row, feature_cols):
    latest_features = latest_row[feature_cols].values.reshape(1, -1)
    pred = model.predict(latest_features)[0]
    return pred
