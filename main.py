import argparse
from src.fetch_data import fetch_all_seasons
from src.clean_data import build_features
from src.train_model import train
from src.predict import predict_game

def main():
    parser = argparse.ArgumentParser(description="NBA Game Predictor")
    parser.add_argument("--pipeline", action="store_true",
                        help="Run full pipeline: fetch → clean → train")
    parser.add_argument("--predict", nargs=2, metavar=("HOME", "AWAY"),
                        help="Predict a game: --predict HOME AWAY (e.g. --predict SAS OKC)")
    args = parser.parse_args()

    if args.pipeline:
        print("=== Step 1: Fetching data ===")
        fetch_all_seasons()
        print("\n=== Step 2: Building features ===")
        build_features()
        print("\n=== Step 3: Training model ===")
        train()
        print("\nPipeline complete.")

    elif args.predict:
        home, away = args.predict
        predict_game(home.upper(), away.upper())

    else:
        parser.print_help()

if __name__ == "__main__":
    main()