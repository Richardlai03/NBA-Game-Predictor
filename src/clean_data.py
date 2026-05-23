import pandas as pd
import numpy as np

def load_raw(path:str = "data/games_raw.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop = True)
    return df

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    fga = df["FGA"].replace(0, np.nan)
    # Possessions = FGA - ORB + TOV + (0.44 x FTA)
    poss = (df["FGA"] - df["OREB"] + df["TOV"] + (0.44 * df["FTA"]))
    poss = poss.replace(0, np.nan)
    df["POSS"]  = poss

    # Offensive rating: points per 100 possessions
    df["OFF_RTG"] = (df["PTS"] / df["POSS"]) * 100
    
    # Effective FG%
    df["EFG_PCT"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"]

    # Turnover Rate
    df["TOV_PCT"] = df["TOV"] / df["POSS"]

    # Home court: MATCHUP contains 'vs.' for home, '@' for away
    df["HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)

    # Win as binary
    df["WIN"] = df["WL"].apply(lambda x: 1 if x == "W" else 0)

    return df

# Compute rolling averages of key stats over last N games for each team
def add_rolling_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    feature_cols =  ["OFF_RTG", "EFG_PCT", "TOV_PCT", "FT_PCT",
                    "PTS", "OREB", "DREB", "STL", "BLK", "PLUS_MINUS"]
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)
    for col in feature_cols:
        df[f"{col}_ROLL{window}"] = (df.groupby("TEAM_ID")[col]
        .transform(lambda x: x.shift(1).rolling(window, min_periods = 3).mean())
        )
    return df

def build_matchup_row (df:pd.DataFrame) -> pd.DataFrame:
    home = df[df["HOME"] == 1].copy()
    away = df[df["HOME"] == 0].copy()
    home = home.add_prefix("HOME_")
    away = away.add_prefix("AWAY_")
    home = home.rename(columns={"HOME_GAME_ID": "GAME_ID", "HOME_GAME_DATE": "GAME_DATE"})
    away = away.rename(columns={"AWAY_GAME_ID": "GAME_ID"})
    merged = pd.merge(home, away, on="GAME_ID", how="inner")
    merged["TARGET"] = merged["HOME_WIN"]
    return merged

def build_features (save: bool = True) -> pd.DataFrame:
    df = load_raw()
    df = add_basic_features(df)
    df = add_rolling_features(df)
    df = build_matchup_row(df)

    roll_cols = [c for c in df.columns if "_ROLL in c"]
    df = df.dropna(subset=roll_cols).reset_index(drop=True)

    if save:
        df.to_csv("data/games_features.csv", index=False)
        print(f"Saved {len(df)} matchup rows to data/games_features.csv")

    return df

if __name__ == "__main__":
    build_features()

