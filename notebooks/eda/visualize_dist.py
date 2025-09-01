#!/usr/bin/env python3
"""
visualize_dist.py

Exploratory data analysis (EDA) for user-level Reddit activity counts.
Generates summary statistics and distribution plots for `posts` and `comments`,
plus a few derived ratios, and saves them under `reports/figures/`.

Usage:
  python -m src.visualize_dist \
    --input data/processed/train.csv \
    --outdir reports/figures \
    --min-total 0

Notes:
- The script is robust to column names that contain the substrings
  "post" and "comment" (case-insensitive). E.g., it will pick from
  ["posts", "n_posts", "num_posts"], ["comments", "n_comments", ...].
- (posts == 0 & comments == 0) rows are flagged and reported. Use
  --min-total to drop very low-activity accounts if needed.
- No seaborn dependency; plain matplotlib as per project conventions.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------
# Helpers
# -----------------------------


def _infer_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cols = {c.lower(): c for c in df.columns}
    post_candidates = [c for c in df.columns if "post" in c.lower()]
    comment_candidates = [c for c in df.columns if "comment" in c.lower()]

    if not post_candidates or not comment_candidates:
        raise KeyError(
            "Could not infer 'posts' and 'comments' columns. "
            f"Available columns: {list(df.columns)}"
        )

    # prefer exact plural names if present
    def pick(cands, preferred):
        for p in preferred:
            for c in cands:
                if c.lower() == p:
                    return c
        # fallback to the first candidate
        return cands[0]

    posts_col = pick(post_candidates, ["posts", "n_posts", "num_posts"])
    comments_col = pick(comment_candidates, ["comments", "n_comments", "num_comments"])
    return posts_col, comments_col


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_int(x, default=0):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default


# Alias constants and helper for extracting counts flexibly
ALIAS_POSTS_NUM = ("posts", "n_posts", "num_posts", "post_count", "posts_count")
ALIAS_COMMENTS_NUM = ("comments", "n_comments", "num_comments", "comment_count", "comments_count")
ALIAS_POSTS_LIST = ("posts", "submissions", "links")
ALIAS_COMMENTS_LIST = ("comments", "replies")


def _extract_counts_from_mapping(m: dict) -> tuple[int, int]:
    # Numeric-style keys
    for k in ALIAS_POSTS_NUM:
        if k in m and not isinstance(m[k], (list, tuple)):
            p = _safe_int(m.get(k), 0)
            # try comments alongside
            for kc in ALIAS_COMMENTS_NUM:
                if kc in m and not isinstance(m[kc], (list, tuple)):
                    c = _safe_int(m.get(kc), 0)
                    return p, c
            return p, 0

    for k in ALIAS_COMMENTS_NUM:
        if k in m and not isinstance(m[k], (list, tuple)):
            c = _safe_int(m.get(k), 0)
            return 0, c

    # List-style keys
    p = 0
    c = 0
    for k in ALIAS_POSTS_LIST:
        if isinstance(m.get(k), (list, tuple)):
            p = max(p, len(m.get(k) or []))
    for k in ALIAS_COMMENTS_LIST:
        if isinstance(m.get(k), (list, tuple)):
            c = max(c, len(m.get(k) or []))

    if p or c:
        return p, c

    return 0, 0


def _count_from_obj(obj) -> tuple[int, int]:
    """Infer (posts, comments) counts from a JSON-like object.
    Supports several shapes:
    - {"posts": [...], "comments": [...]} (lists)
    - {"posts": 12, "comments": 34} (counts)
    - {"items": [...]} where items have a type/kind field ("post"/"comment")
    - A plain list of items with a type/kind field
    Returns (posts, comments).
    """
    if isinstance(obj, dict):
        # Some exports wrap the payload under `data`
        cand = obj.get("data") if isinstance(obj.get("data"), dict) else obj

        # Try direct extraction via alias keys / list fields
        p, c = _extract_counts_from_mapping(cand)
        if p or c:
            return p, c

        # items-style container
        items = cand.get("items")
        if isinstance(items, list):
            posts = comments = 0
            for it in items:
                t = str(it.get("type") or it.get("kind") or it.get("t") or "").lower()
                if "post" in t or t == "submission" or t == "link":
                    posts += 1
                elif "comment" in t:
                    comments += 1
            return posts, comments

        # single activity object – best-effort
        t = str(cand.get("type") or cand.get("kind") or "").lower()
        if "post" in t or t == "submission" or t == "link":
            return 1, 0
        if "comment" in t:
            return 0, 1
        return 0, 0
    elif isinstance(obj, list):
        posts = comments = 0
        for it in obj:
            if isinstance(it, dict):
                t = str(it.get("type") or it.get("kind") or "").lower()
                if "post" in t or t == "submission" or t == "link":
                    posts += 1
                elif "comment" in t:
                    comments += 1
        return posts, comments
    return 0, 0


def _username_from(obj, fallback: str) -> str:
    if isinstance(obj, dict):
        for k in ("username", "user", "author", "name"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return fallback


def load_from_json_path(in_path: Path) -> pd.DataFrame:
    """Load user-level counts from a JSON/JSONL file or a directory of JSON files.
    Heuristics:
      - If directory: iterate over *.json files, each is one user; use filename as fallback username.
      - If .jsonl: each line is a JSON record (one user or one container), use `_count_from_obj`.
      - If .json: single JSON that could be a dict (one user) or list (many). If many, aggregate by
        an available username/author when possible, otherwise treat as one container.
    Returns a DataFrame with columns: username, posts, comments.
    """
    records = []
    if in_path.is_dir():
        # Recursively gather JSON-like files
        fps = (
            list(in_path.rglob("*.json"))
            + list(in_path.rglob("*.jsonl"))
            + list(in_path.rglob("*.ndjson"))
        )
        if not fps:
            # Show a small preview of what exists to help the user debug paths
            try:
                sample = sorted([str(p) for p in in_path.iterdir()])[:20]
            except Exception:
                sample = []
            raise FileNotFoundError(
                "No JSON/JSONL files found under directory: "
                + str(in_path)
                + ("\nSample entries: " + "\n".join(sample) if sample else "")
            )
        for fp in sorted(fps):
            try:
                if fp.suffix.lower() == ".jsonl" or fp.suffix.lower() == ".ndjson":
                    with fp.open() as f:
                        for i, line in enumerate(f):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            username = _username_from(obj, f"{fp.stem}_{i}")
                            p, c = _count_from_obj(obj)
                            records.append({"username": username, "posts": p, "comments": c})
                else:  # .json
                    obj = json.loads(fp.read_text())
                    username = _username_from(obj, fp.stem)
                    p, c = _count_from_obj(obj)
                    records.append({"username": username, "posts": p, "comments": c})
            except Exception:
                # Skip unreadable files but continue
                continue
        return pd.DataFrame.from_records(records)

    # file input
    suffix = in_path.suffix.lower()
    if suffix == ".jsonl":
        with in_path.open() as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                username = _username_from(obj, f"row{i}")
                p, c = _count_from_obj(obj)
                records.append({"username": username, "posts": p, "comments": c})
        return pd.DataFrame.from_records(records)

    if suffix == ".json":
        try:
            obj = json.loads(in_path.read_text())
        except Exception:
            raise
        # If it's a list of user containers, try to aggregate per user
        if isinstance(obj, list):
            for i, o in enumerate(obj):
                username = _username_from(o, f"row{i}")
                p, c = _count_from_obj(o)
                records.append({"username": username, "posts": p, "comments": c})
            return pd.DataFrame.from_records(records)
        else:
            username = _username_from(obj, in_path.stem)
            p, c = _count_from_obj(obj)
            return pd.DataFrame([{"username": username, "posts": p, "comments": c}])

    raise FileNotFoundError(f"Unsupported JSON path: {in_path}")


# -----------------------------
# Core EDA
# -----------------------------


def eda(df: pd.DataFrame, posts_col: str, comments_col: str, outdir: Path) -> pd.DataFrame:
    """Compute basic summaries, derive ratio features, and create plots."""
    df = df.copy()

    # Ensure numeric and non-negative
    for c in [posts_col, comments_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df.loc[df[c] < 0, c] = 0

    # Derived features
    df["total_activity"] = df[posts_col] + df[comments_col]
    df["post_comment_ratio"] = df[posts_col] / (df[comments_col] + 1)
    df["comment_post_ratio"] = df[comments_col] / (df[posts_col] + 1)
    df["is_empty"] = (df[posts_col] == 0) & (df[comments_col] == 0)
    df["is_post_heavy"] = (df[posts_col] > 0) & (df[comments_col] == 0)
    df["is_comment_heavy"] = (df[comments_col] > 0) & (df[posts_col] == 0)

    # -----------------
    # Summary CSV
    # -----------------
    summary = {
        "n_rows": len(df),
        "n_empty_accounts": int(df["is_empty"].sum()),
        "pct_empty_accounts": float(df["is_empty"].mean() * 100),
        "median_posts": float(df[posts_col].median()),
        "median_comments": float(df[comments_col].median()),
        "p95_posts": float(np.percentile(df[posts_col], 95)),
        "p95_comments": float(np.percentile(df[comments_col], 95)),
        "max_posts": int(df[posts_col].max()),
        "max_comments": int(df[comments_col].max()),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(outdir / "eda_summary.csv", index=False)

    # Also export a small frequency table for activity buckets
    buckets = pd.cut(
        df["total_activity"],
        bins=[-1, 0, 1, 5, 10, 25, 50, 100, 250, 500, 1000, np.inf],
        labels=[
            "0",
            "1",
            "2–5",
            "6–10",
            "11–25",
            "26–50",
            "51–100",
            "101–250",
            "251–500",
            "501–1000",
            ">1000",
        ],
    )
    bucket_counts = buckets.value_counts().sort_index()
    bucket_counts.to_csv(outdir / "activity_buckets.csv", header=["count"])

    # Export empty accounts for inspection (if username present)
    cols = [c for c in ["username", posts_col, comments_col, "total_activity"] if c in df.columns]
    if cols:
        df.loc[df["is_empty"], cols].to_csv(outdir / "empty_accounts.csv", index=False)

    # -----------------
    # Plots
    # -----------------
    plt.figure()
    plt.hist(np.log1p(df[posts_col]), bins=50, alpha=0.8)
    plt.xlabel("log1p(posts)")
    plt.ylabel("Frequency")
    plt.title("Distribution of posts (log1p)")
    plt.tight_layout()
    plt.savefig(outdir / "hist_posts_log1p.png", dpi=200)
    plt.close()

    plt.figure()
    plt.hist(np.log1p(df[comments_col]), bins=50, alpha=0.8)
    plt.xlabel("log1p(comments)")
    plt.ylabel("Frequency")
    plt.title("Distribution of comments (log1p)")
    plt.tight_layout()
    plt.savefig(outdir / "hist_comments_log1p.png", dpi=200)
    plt.close()

    # Scatter posts vs comments (log-log)
    plt.figure()
    x = df[posts_col].astype(float)
    y = df[comments_col].astype(float)
    plt.scatter(x + 1e-6, y + 1e-6, s=8, alpha=0.3)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("posts (log scale)")
    plt.ylabel("comments (log scale)")
    plt.title("Posts vs Comments (log–log)")
    # add y=x reference line
    lims = [
        min((x + 1).min(), (y + 1).min()),
        max((x + 1).max(), (y + 1).max()),
    ]
    plt.plot(lims, lims, linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(outdir / "scatter_posts_vs_comments_loglog.png", dpi=200)
    plt.close()

    # Ratio histograms (log1p for stability)
    plt.figure()
    plt.hist(np.log1p(df["post_comment_ratio"]), bins=60, alpha=0.9)
    plt.xlabel("log1p(post/comment)")
    plt.ylabel("Frequency")
    plt.title("Distribution of post-to-comment ratio (log1p)")
    plt.tight_layout()
    plt.savefig(outdir / "hist_post_comment_ratio_log1p.png", dpi=200)
    plt.close()

    plt.figure()
    plt.hist(np.log1p(df["comment_post_ratio"]), bins=60, alpha=0.9)
    plt.xlabel("log1p(comment/post)")
    plt.ylabel("Frequency")
    plt.title("Distribution of comment-to-post ratio (log1p)")
    plt.tight_layout()
    plt.savefig(outdir / "hist_comment_post_ratio_log1p.png", dpi=200)
    plt.close()

    # Activity buckets bar chart
    plt.figure()
    bucket_counts.plot(kind="bar")
    plt.xlabel("Total activity (posts + comments)")
    plt.ylabel("Number of users")
    plt.title("User activity buckets")
    plt.tight_layout()
    plt.savefig(outdir / "bar_activity_buckets.png", dpi=200)
    plt.close()

    return df


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDA for Reddit user activity distributions")
    p.add_argument(
        "--input",
        type=str,
        default="data/processed/train.csv",
        help="Input path: CSV file OR JSON/JSONL file OR a directory of per-user JSON files",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="reports/figures",
        help="Directory to save figures & summaries",
    )
    p.add_argument(
        "--min-total", type=int, default=0, help="Drop users with (posts+comments) < min-total"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    outdir = Path(args.outdir)
    _mkdir(outdir)

    if not in_path.exists():
        raise FileNotFoundError(f"Input path not found: {in_path}")

    if in_path.suffix.lower() == ".csv":
        df = pd.read_csv(in_path)
    elif in_path.is_dir() or in_path.suffix.lower() in {".json", ".jsonl"}:
        df = load_from_json_path(in_path)
    else:
        raise FileNotFoundError(
            f"Unsupported input type for {in_path}. Use CSV, JSON/JSONL, or a directory of JSON files."
        )

    if df is None or df.empty:
        raise ValueError(
            f"Loaded zero records from {in_path}. If you provided a directory, make sure it contains *.json or *.jsonl files."
        )

    # If we loaded from JSON, the columns should be username, posts, comments
    posts_col, comments_col = _infer_cols(df)

    # Optional filtering by total activity
    if args.min_total > 0:
        df = df[(df[posts_col] + df[comments_col]) >= args.min_total].copy()

    # Run EDA and write enriched CSV (with derived columns) for downstream use
    enriched = eda(df, posts_col, comments_col, outdir)
    enriched.to_csv(outdir / "enriched_activity_snapshot.csv", index=False)

    # Print a concise console summary
    n = len(enriched)
    n_empty = int(enriched["is_empty"].sum())
    n_post_heavy = int(enriched["is_post_heavy"].sum())
    n_comment_heavy = int(enriched["is_comment_heavy"].sum())
    print(
        "\n[EDA Summary]",
        f"\nInput: {in_path}",
        f"\nRows: {n}",
        f"\nEmpty accounts (0,0): {n_empty}",
        f"\nPost-heavy (posts>0, comments=0): {n_post_heavy}",
        f"\nComment-heavy (comments>0, posts=0): {n_comment_heavy}",
        f"\nFigures & CSV saved to: {outdir}\n",
    )


if __name__ == "__main__":
    main()
