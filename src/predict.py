import pandas as pd
import numpy as np
import joblib

from clean_data import load_raw, add_basic_features, add_rolling_features

FEATURE_COLS = [
    "HOME_OFF_RTG_ROLL10",  "AWAY_OFF_RTG_ROLL10",
    "HOME_EFG_PCT_ROLL10",  "AWAY_EFG_PCT_ROLL10",
    "HOME_TOV_PCT_ROLL10",  "AWAY_TOV_PCT_ROLL10",
    "HOME_FT_PCT_ROLL10",   "AWAY_FT_PCT_ROLL10",
    "HOME_PTS_ROLL10",      "AWAY_PTS_ROLL10",
    "HOME_OREB_ROLL10",     "AWAY_OREB_ROLL10",
    "HOME_DREB_ROLL10",     "AWAY_DREB_ROLL10",
    "HOME_STL_ROLL10",      "AWAY_STL_ROLL10",
    "HOME_BLK_ROLL10",      "AWAY_BLK_ROLL10",
    "HOME_PLUS_MINUS_ROLL10", "AWAY_PLUS_MINUS_ROLL10",
]


def get_team_rolling_stats(df: pd.DataFrame, team_abbr: str) -> pd.Series:
    team_df = df[df["TEAM_ABBREVIATION"] == team_abbr].sort_values("GAME_DATE")

    if team_df.empty:
        raise ValueError(f"Team '{team_abbr}' not found. Check the abbreviation.")

    latest = team_df.iloc[-1]  # most recent game row

    stats = {
        "OFF_RTG_ROLL10":    latest.get("OFF_RTG_ROLL10"),
        "EFG_PCT_ROLL10":    latest.get("EFG_PCT_ROLL10"),
        "TOV_PCT_ROLL10":    latest.get("TOV_PCT_ROLL10"),
        "FT_PCT_ROLL10":     latest.get("FT_PCT_ROLL10"),
        "PTS_ROLL10":        latest.get("PTS_ROLL10"),
        "OREB_ROLL10":       latest.get("OREB_ROLL10"),
        "DREB_ROLL10":       latest.get("DREB_ROLL10"),
        "STL_ROLL10":        latest.get("STL_ROLL10"),
        "BLK_ROLL10":        latest.get("BLK_ROLL10"),
        "PLUS_MINUS_ROLL10": latest.get("PLUS_MINUS_ROLL10"),
    }
    return pd.Series(stats)


def build_prediction_row(home_stats: pd.Series, away_stats: pd.Series) -> pd.DataFrame:
    row = {}
    for key in home_stats.index:
        row[f"HOME_{key}"] = home_stats[key]
        row[f"AWAY_{key}"] = away_stats[key]
    return pd.DataFrame([row])[FEATURE_COLS]


def predict_game(home_team: str, away_team: str):
    model  = joblib.load("models/logreg_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    df = load_raw()
    df = add_basic_features(df)
    df = add_rolling_features(df)
    home_stats = get_team_rolling_stats(df, home_team)
    away_stats = get_team_rolling_stats(df, away_team)
    X = build_prediction_row(home_stats, away_stats)
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0]
    away_prob = prob[0]
    home_prob = prob[1]

    winner = home_team if home_prob > away_prob else away_team

    print(f"\n{'='*40}")
    print(f"  {home_team} (home) vs {away_team} (away)")
    print(f"{'='*40}")
    print(f"  {home_team} win probability:  {home_prob:.1%}")
    print(f"  {away_team} win probability:  {away_prob:.1%}")
    print(f"{'='*40}")
    print(f"  Predicted winner: {winner}")
    print(f"{'='*40}\n")

    return home_prob, away_prob


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        # Run from terminal: python src/predict.py BOS GSW
        predict_game(sys.argv[1], sys.argv[2])
    else:
        # Default example
        predict_game("BOS", "GSW")