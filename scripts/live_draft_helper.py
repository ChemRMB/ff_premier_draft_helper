import argparse, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
board = pd.read_csv(ROOT / "data/outputs/draft_board_flex_aware.csv")


def filter_taken(board, taken):
    taken_set = {t.strip().lower() for t in taken}
    mask = ~board["web_name"].str.lower().isin(taken_set)
    return board[mask].copy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--taken",
        type=str,
        default="",
        help="Comma-separated list of drafted player web_name values",
    )
    ap.add_argument("--topn", type=int, default=20)
    args = ap.parse_args()

    taken = [s for s in args.taken.split(",") if s.strip()]
    avail = filter_taken(board, taken).reset_index(drop=True)

    print("\n== Best available (overall) ==")
    print(
        avail[["web_name", "team_name", "pos", "proj_points", "VOR"]]
        .head(args.topn)
        .to_string(index=False)
    )

    for pos in ["F", "M", "D", "GK"]:
        sub = avail[avail["pos"] == pos].head(10)
        print(f"\n== Best available {pos} ==")
        print(
            sub[["web_name", "team_name", "proj_points", "VOR"]].to_string(index=False)
        )
