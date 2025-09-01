#!/usr/bin/env python3
"""
Aggregate comment-level probabilities to user-level text features and merge
into processed tables (train/dev_folds/test) with suffix *_text.csv.

Usage:
  python -m src.features.text_agg \
    --comment-probs data/processed/comment_probs.csv \
    --user-col username \
    --train data/processed/train.csv \
    --dev data/processed/dev_folds.csv \
    --test data/processed/test.csv \
    --outdir data/processed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--comment-probs", required=True, help="CSV with columns: <user-col>, p_sensitive"
    )
    ap.add_argument("--user-col", default="username")
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--dev", default="data/processed/dev_folds.csv")
    ap.add_argument("--test", default="data/processed/test.csv")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--tau", type=float, default=0.5, help="threshold for frac_sensitive")
    return ap.parse_args()


def aggregate(comment_probs: pd.DataFrame, user_col: str, tau: float) -> pd.DataFrame:
    gp = (
        comment_probs.groupby(user_col)["p_sensitive"]
        .agg(
            text_mean_prob="mean",
            text_max_prob="max",
            text_frac_sensitive=lambda s: (s >= tau).mean(),
            text_n_comments="count",
        )
        .reset_index()
    )
    return gp


def merge_save(base: pd.DataFrame, agg: pd.DataFrame, user_col: str, out_path: Path) -> None:
    if user_col not in base.columns:
        # attempt lower-case fallback
        cols = {c.lower(): c for c in base.columns}
        if user_col.lower() in cols:
            user_col = cols[user_col.lower()]
        else:
            raise ValueError(f"Base table missing user column: {user_col}")
    merged = base.merge(agg, on=user_col, how="left")
    merged.to_csv(out_path, index=False)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    probs = pd.read_csv(args.comment_probs)
    if args.user_col not in probs.columns or "p_sensitive" not in probs.columns:
        raise ValueError(f"comment-probs must contain columns: {args.user_col}, p_sensitive")

    agg = aggregate(probs, args.user_col, args.tau)
    print(f"[TextAgg] Users with text: {len(agg)}; features: {list(agg.columns)}")

    # Merge to train/dev/test
    train = pd.read_csv(args.train)
    dev = pd.read_csv(args.dev)
    test = pd.read_csv(args.test)

    merge_save(train, agg, args.user_col, outdir / "train_text.csv")
    merge_save(dev, agg, args.user_col, outdir / "dev_folds_text.csv")
    merge_save(test, agg, args.user_col, outdir / "test_text.csv")

    print(f"[TextAgg] Saved merged files to: {outdir}")


if __name__ == "__main__":
    main()
