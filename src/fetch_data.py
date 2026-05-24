import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19", 
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
    "2024-25", "2025-26"
]

def fetch_season(season: str) -> pd.DataFrame:
    print(f"Fetching {season}...")
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00" # NBA
    )
    df = finder.get_data_frames()[0]
    time.sleep(0.8)
    return df

def fetch_all_seasons() -> pd.DataFrame:
    all_seasons = []
    for season in SEASONS:
        df = fetch_season(season)
        df["SEASON "] = season
        all_seasons.append(df)
    
    combined = pd.concat(all_seasons, ignore_index=True)
    combined.to_csv("data/games_raw.csv", index=False)
    print(f"Saved {len(combined)} rows to data/games_raw.csv")
    return combined

if __name__ == "__main__":
    fetch_all_seasons()