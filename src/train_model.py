import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, log_loss, confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve

FEATURE_COLS = ["HOME_OFF_RTG_ROLL10",  "AWAY_OFF_RTG_ROLL10",
    "HOME_EFG_PCT_ROLL10",  "AWAY_EFG_PCT_ROLL10",
    "HOME_TOV_PCT_ROLL10",  "AWAY_TOV_PCT_ROLL10",
    "HOME_FT_PCT_ROLL10",  "AWAY_FT_PCT_ROLL10",
    "HOME_PTS_ROLL10",      "AWAY_PTS_ROLL10",
    "HOME_OREB_ROLL10",     "AWAY_OREB_ROLL10",
    "HOME_DREB_ROLL10",     "AWAY_DREB_ROLL10",
    "HOME_STL_ROLL10",      "AWAY_STL_ROLL10",
    "HOME_BLK_ROLL10",      "AWAY_BLK_ROLL10",
    "HOME_PLUS_MINUS_ROLL10", "AWAY_PLUS_MINUS_ROLL10",]

TRAIN_SEASONS = ["2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22"]
TEST_SEASONS = ["2022-23", "2023-24"]

def load_features(path: str = "data/games_features.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def split_data(df: pd.DataFrame):
    train = df[df["HOME_SEASON"].isin(TRAIN_SEASONS)]
    test = df[df["HOME_SEASON"].isin(TEST_SEASONS)]
    X_train = train[FEATURE_COLS]
    y_train = train["TARGET"]
    X_test = test[FEATURE_COLS]
    y_test = test["TARGET"]
    print(f"Train: {len(train)} games | Test: {len(test)} games")
    return X_train, y_train, X_test, y_test

def train(save: bool = True):
    df = load_features()
    season_col = [c for c in df.columns if "SEASON" in c]
    print("Season columns found:", season_col)
    X_train, y_train, X_test, y_test = split_data(df)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter = 1000, random_state = 42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Log Loss: {loss:.4f}")
    print(f"Baseline (always pick home): {y_test.mean():.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    if save:
        joblib.dump(model, "models/logreg_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        print("Saved model and scaler to models")

    plot_results(model, scaler, X_test, y_test, y_prob, FEATURE_COLS)
    return model, scaler

def plot_results(model, scaler, X_test, y_test, y_prob, feature_cols):
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    y_pred = model.predict(scaler.transform(X_test))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], xticklabels=["Away Win", "Home Win"], yticklabels= ["Away Win", "Home Win"])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")

    coefs = pd.Series(model.coef_[0], index=feature_cols).sort_values()
    coefs.plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_title("Feature Coefficients")
    axes[1].axvline(0, color="black", linewidth=0.8)

    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    axes[2].plot(prob_pred, prob_true, marker="o", label="Model")
    axes[2].plot([0,1], [0,1], linestyle="--", color="gray", label="Perfect")
    axes[2].set_title("Calibration Curve")
    axes[2].set_xlabel("Mean Predicted Probability")
    axes[2].set_ylabel("Fraction of Positives")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig("models/evalutaion.png", dpi=150)
    plt.show()
    print("Saved evalutaion plots to models/evaluation.png")

if __name__ == "__main__":
    train()
