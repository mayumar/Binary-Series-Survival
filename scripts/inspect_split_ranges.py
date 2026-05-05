import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
from typing import Dict, List, Tuple

from data_loading.data_loader import load_data
from main import calculate_horizon


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_split(name: str, split: Tuple[list, list, list]) -> None:
    x_list, _, _ = split
    print(f"\n=== {name} ===")
    print(f"Runs: {len(x_list)}")

    if not x_list:
        return

    start_times = [df["time"].iloc[0] for df in x_list]
    end_times = [df["time"].iloc[-1] for df in x_list]

    print(f"Global start: {min(start_times)}")
    print(f"Global end:   {max(end_times)}")
    print(f"First run:    {start_times[0]} -> {end_times[0]}")
    print(f"Last run:     {start_times[-1]} -> {end_times[-1]}")

    print("\nPer-run ranges:")
    for i, df in enumerate(x_list):
        print(f"  run {i:02d}: {df['time'].iloc[0]} -> {df['time'].iloc[-1]} | len={len(df)}")


def collect_split_ranges(split: Tuple[list, list, list]) -> List[Dict[str, float]]:
    x_list, _, _ = split
    ranges: List[Dict[str, float]] = []

    for i, df in enumerate(x_list):
        ranges.append(
            {
                "run_idx": i,
                "start_time": float(df["time"].iloc[0]),
                "end_time": float(df["time"].iloc[-1]),
                "length": int(len(df)),
            }
        )

    return ranges


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect temporal train/val/test split ranges without running training."
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target variable (failure mode) to inspect.",
    )
    parser.add_argument(
        "--config",
        default="config/config.json",
        help="Path to configuration JSON.",
    )
    parser.add_argument(
        "--cv-mode",
        action="store_true",
        help="Use cross-validation split mode.",
    )
    parser.add_argument(
        "--fold-idx",
        type=int,
        default=0,
        help="Fold index when using --cv-mode.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds when using --cv-mode.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save split ranges as JSON.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    with open(config["paths"]["mtbfs"], "r", encoding="utf-8") as f:
        mtbfs = json.load(f)

    horizon = calculate_horizon(
        mtbfs[args.target],
        config["parameters"]["max_horizon"],
    )

    print(f"Target:  {args.target}")
    print(f"Horizon: {horizon}h")
    print(f"CV mode: {args.cv_mode}")
    if args.cv_mode:
        print(f"Fold:    {args.fold_idx + 1}/{args.n_folds}")

    train_data, val_data, test_data = load_data(
        config["paths"]["input_data"],
        args.target,
        horizon_h=horizon,
        cv_mode=args.cv_mode,
        fold_idx=args.fold_idx,
        n_folds=args.n_folds,
        # start_idx=1770321000
    )

    summarize_split("TRAIN", train_data)
    summarize_split("VAL", val_data)
    summarize_split("TEST", test_data)

    if args.output:
        payload = {
            "target": args.target,
            "horizon_h": horizon,
            "cv_mode": args.cv_mode,
            "fold_idx": args.fold_idx if args.cv_mode else None,
            "n_folds": args.n_folds if args.cv_mode else None,
            "splits": {
                "train": collect_split_ranges(train_data),
                "val": collect_split_ranges(val_data),
                "test": collect_split_ranges(test_data),
            },
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSplit ranges saved to: {args.output}")


if __name__ == "__main__":
    main()
