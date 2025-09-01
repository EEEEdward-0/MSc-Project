#!/usr/bin/env python3
"""
prepare_data.py

Turn the enriched EDA snapshot into two CSVs:
  1) data/processed/all_users.csv  – keep everyone (incl. 0,0 lurkers)
  2) data/processed/train.csv      – filtered for model training (drop (0,0))

Also engineers a few stable features and creates a weak-supervision label.

Usage:
  python -m src.prepare_data \
    --input reports/figures/enriched_activity_snapshot.csv \
    --outdir data/processed \
    --act-hi 500 --post-hi 50 --cmt-hi 300
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Username detection + non-leaky features
# -----------------------------
YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[-_/\.](\d{1,2})[-_/\.]((?:19|20)?\d{2})(?!\d)", re.I)


def _valid_date(m: re.Match) -> bool:
    d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y < 100:
        y += 2000 if y < 30 else 1900
    try:
        dt.date(y, mth, d)
        return True
    except Exception:
        return False


def guess_username_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "username",
        "user",
        "author",
        "name",
        "account",
        "userid",
        "user_id",
        "author_name",
        "screen_name",
    ]
    cols = {c.lower(): c for c in df.columns}
    for key in candidates:
        if key in cols:
            return cols[key]
    # fallback: any column containing 'user' or 'author'
    for c in df.columns:
        lc = c.lower()
        if "user" in lc or "author" in lc:
            return c
    return None


def username_features(u: str) -> dict:
    if not isinstance(u, str):
        u = ""
    u_str = u.strip()
    lower = u_str.lower()
    # split tokens by _,.,-, camelCase boundary
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", u_str)
    cleaned = re.sub(r"[^A-Za-z0-9\-_\.]+", " ", spaced)
    tokens = [t for t in re.split(r"[\s_\-.]+", cleaned) if t]

    length = len(u_str)
    n_digits = sum(ch.isdigit() for ch in u_str)
    digit_frac = n_digits / length if length else 0.0
    has_year = bool(YEAR_RE.search(lower))
    has_date = any(_valid_date(m) for m in DATE_RE.finditer(lower))
    segs = len(tokens)

    score = 0.0
    score += 1.0 if has_year else 0.0
    score += 2.0 if has_date else 0.0
    score += 0.5 if digit_frac >= 0.3 else 0.0

    return {
        "username_len": length,
        "username_digit_frac": digit_frac,
        "username_has_year": int(has_year),
        "username_has_date": int(has_date),
        "username_seg_count": segs,
        "username_risk_score": score,
        "username_is_highrisk": int(score >= 2.0),
    }


def _infer_cols(df: pd.DataFrame) -> Tuple[str, str]:
    post_candidates = [c for c in df.columns if "post" in c.lower()]
    comment_candidates = [c for c in df.columns if "comment" in c.lower()]
    if not post_candidates or not comment_candidates:
        raise KeyError(
            "Could not infer 'posts' and 'comments' columns. Available columns: "
            + ", ".join(map(str, df.columns))
        )

    def pick(cands, preferred):
        for p in preferred:
            for c in cands:
                if c.lower() == p:
                    return c
        return cands[0]

    posts_col = pick(post_candidates, ["posts", "n_posts", "num_posts"])
    comments_col = pick(comment_candidates, ["comments", "n_comments", "num_comments"])
    return posts_col, comments_col


def build_features(df: pd.DataFrame, posts_col: str, comments_col: str) -> pd.DataFrame:
    df = df.copy()
    for c in (posts_col, comments_col):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df.loc[df[c] < 0, c] = 0

    if "total_activity" not in df.columns:
        df["total_activity"] = df[posts_col] + df[comments_col]

    # engineered features
    df["log_posts"] = np.log1p(df[posts_col])
    df["log_comments"] = np.log1p(df[comments_col])
    df["log_total_activity"] = np.log1p(df["total_activity"])
    df["post_comment_ratio"] = df[posts_col] / (df[comments_col] + 1)
    df["comment_post_ratio"] = df[comments_col] / (df[posts_col] + 1)
    df["is_empty"] = (df[posts_col] == 0) & (df[comments_col] == 0)
    return df


def weak_label(
    df: pd.DataFrame,
    act_hi: int = 500,
    post_hi: int = 50,
    cmt_hi: int = 300,
) -> pd.Series:
    """
    Binary labels: 1=HIGH, 0=LOW.
    HIGH if any threshold is met, or if any column containing 'pii' or
    'sensitive' is positive (best-effort scan).
    """
    hi = (
        (df["total_activity"] >= act_hi)
        | (df.filter(like="post").max(axis=1) >= post_hi)
        | (df.filter(like="comment").max(axis=1) >= cmt_hi)
    )
    pii_cols = [c for c in df.columns if "pii" in c.lower() or "sensitive" in c.lower()]
    if pii_cols:
        hi = hi | (df[pii_cols].fillna(0).max(axis=1) > 0)
    return hi.astype(int)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare train/all CSVs for modelling")
    p.add_argument(
        "--input",
        type=str,
        default="reports/figures/enriched_activity_snapshot.csv",
        help="Input CSV (from visualize_dist)",
    )
    p.add_argument("--outdir", type=str, default="data/processed", help="Output dir")
    p.add_argument("--act-hi", type=int, default=500, help="High-risk threshold for total_activity")
    p.add_argument("--post-hi", type=int, default=50, help="High-risk threshold for posts")
    p.add_argument("--cmt-hi", type=int, default=300, help="High-risk threshold for comments")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)
    if df.empty:
        raise ValueError("Input CSV is empty")

    posts_col, comments_col = _infer_cols(df)
    df = build_features(df, posts_col, comments_col)

    # --- unify username column and add username-derived features ---
    uname_col = guess_username_col(df)
    if uname_col is not None:
        # create normalized 'username' column
        if "username" not in df.columns:
            df["username"] = df[uname_col].astype(str).str.strip().str.lower()
        else:
            df["username"] = df["username"].astype(str).str.strip().str.lower()
        # stable user_id (md5 of username, empty string allowed)
        df["user_id"] = (
            df["username"].fillna("").map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())
        )
        # append username_* features
        uf = df["username"].apply(username_features).apply(pd.Series)
        df = pd.concat([df, uf], axis=1)
    else:
        # still provide a user_id fallback if possible (row index based)
        if "user_id" not in df.columns:
            df["user_id"] = pd.util.hash_pandas_object(df.index, index=True).astype(str)

    # 1) full snapshot (for analysis/visualisation, keep lurkers)
    all_users = df.copy()
    all_users.to_csv(outdir / "all_users.csv", index=False)

    # 2) training data (drop (0,0))
    train = df.loc[~df["is_empty"].astype(bool)].copy()
    train["label"] = weak_label(train, args.act_hi, args.post_hi, args.cmt_hi)
    train.to_csv(outdir / "train.csv", index=False)

    # console summary
    n_all = len(all_users)
    n_empty = int(all_users["is_empty"].sum())
    n_train = len(train)
    pct_empty = (n_empty / n_all * 100.0) if n_all else 0.0
    pos_rate = float(train["label"].mean()) if n_train else 0.0
    print(
        "\n[Prepare Data Summary]",
        f"\nInput: {in_path}",
        f"\nAll users: {n_all} | Empty (0,0): {n_empty} ({pct_empty:.1f}%)",
        f"\nTrain rows: {n_train} | Pos rate (HIGH label): {pos_rate:.3f}",
        f"\nOutputs: {outdir/'all_users.csv'}, {outdir/'train.csv'}\n",
    )


if __name__ == "__main__":
    main()
from pathlib import Path

import numpy as np

#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "data" / "processed" / "train.csv"
OUTDIR = ROOT / "data" / "processed"
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    if not IN.exists():
        raise FileNotFoundError(f"Missing {IN}")
    df = pd.read_csv(IN)
    if "label" not in df.columns:
        raise ValueError("train.csv 中缺少 label 列")

    # keep identifiers for analysis
    id_cols = [c for c in ["user_id", "username"] if c in df.columns]

    y = df["label"].astype(int)

    # 80/20 split
    dev, test = train_test_split(df, test_size=0.2, stratify=y, random_state=42)

    # save test with identifiers
    test.to_csv(OUTDIR / "test.csv", index=False)

    # 5-fold on dev
    dev = dev.copy()
    dev["fold"] = -1
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, val_idx) in enumerate(skf.split(dev, dev["label"].astype(int))):
        dev.loc[dev.iloc[val_idx].index, "fold"] = fold

    dev.to_csv(OUTDIR / "dev_folds.csv", index=False)
    print(f"[Split] dev: {len(dev)} rows (5 folds) | test: {len(test)} rows → {OUTDIR}")


if __name__ == "__main__":
    main()
