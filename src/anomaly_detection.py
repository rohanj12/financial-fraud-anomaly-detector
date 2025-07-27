import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

def train_isolation_forest(X_train):
    # Set contamination to the expected fraud rate (e.g., 0.0017 for creditcard.csv)
    model = IsolationForest(n_estimators=100, contamination=0.0017, random_state=42)
    model.fit(X_train)
    return model

def detect_anomalies(model, X):
    scores = model.decision_function(X)
    preds = model.predict(X)  # -1 = anomaly, 1 = normal
    return preds, scores


def evaluate_model(y_true, y_pred):
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))