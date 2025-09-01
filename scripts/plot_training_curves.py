#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training curves comparison (validation metrics) for multiple models.

- Input: one or several log directories, each containing CSV logs like:
    reports/model_binary/train_log_fold0.csv
    reports/model_text/train_log_fold0.csv
    reports/model_blend/train_log_fold0.csv
  The file pattern is configurable.

- Robust to column name variants. It tries to find validation AUC / PR-AUC /
  logloss among common names (e.g., 'auc', 'val_auc', 'AUC', 'pr_auc', etc.).

- It aggregates across folds by epoch (mean ± std) and plots 3 panels:
  (1) Validation AUC  (↑ better)
  (2) Validation PR-AUC (↑ better)
  (3) Validation Logloss (↓ lower is better)

- Usage examples:
    python scripts/plot_training_compare.py \
        --logdir reports/model_binary:Binary \
        --logdir reports/model_text:Semantic/Text \
        --logdir reports/model_blend:Blend \
        --pattern "train_log_fold*.csv" \
        --out reports/plots/training_compare.png

    # If you don't pass labels after ":", folder name will be used.
"""

import argparse
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime


# ---- Column name mapping ------------------------------------------------------
# Try several aliases for each target metric to be robust to different logs.
ALIAS = {
    "epoch": ["epoch", "iter", "iteration"],
    "val_auc": ["val_auc", "auc", "AUC", "valid_auc", "eval_auc"],
    "val_prauc": ["val_pr_auc", "prauc", "pr_auc", "VAL_PRAUC", "valid_pr_auc"],
    "val_logloss": ["val_logloss", "logloss", "brier", "VAL_BRIER", "valid_logloss", "valid_loss"],
}


def find_first(df: pd.DataFrame, keys: List[str]) -> str:
    """Return the first matching column name in df.columns for aliases in keys."""
    cols_lower = {c.lower(): c for c in df.columns}
    for k in keys:
        if k.lower() in cols_lower:
            return cols_lower[k.lower()]
    return ""


def load_one_dir(log_dir: str, pattern: str) -> pd.DataFrame:
    """Load and concatenate logs in a directory; keep only (epoch, val metrics).
    Suffix per-fold metric columns to avoid name collisions during merge.
    """
    files = sorted(glob.glob(os.path.join(log_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No logs under {log_dir} matching {pattern}")

    frames: List[pd.DataFrame] = []
    for idx, fp in enumerate(files):
        df = pd.read_csv(fp)

        # Resolve columns (robust to variants)
        c_epoch = find_first(df, ALIAS["epoch"])
        c_auc  = find_first(df, ALIAS["val_auc"])
        c_pr   = find_first(df, ALIAS["val_prauc"])
        c_ll   = find_first(df, ALIAS["val_logloss"])

        # Minimal requirement: epoch + at least one metric
        if not c_epoch or not (c_auc or c_pr or c_ll):
            continue

        # Build a per-fold dataframe with suffixed metric names
        keep = {"epoch": pd.to_numeric(df[c_epoch], errors="coerce").astype("Int64")}
        if c_auc:
            keep[f"val_auc_{idx}"] = pd.to_numeric(df[c_auc], errors="coerce")
        if c_pr:
            keep[f"val_prauc_{idx}"] = pd.to_numeric(df[c_pr], errors="coerce")
        if c_ll:
            keep[f"val_logloss_{idx}"] = pd.to_numeric(df[c_ll], errors="coerce")

        frames.append(pd.DataFrame(keep).dropna(subset=["epoch"]))

    if not frames:
        raise ValueError(f"{log_dir} has logs but no recognizable columns.")

    # Outer-merge on epoch to align different fold lengths, no suffix needed now
    merged = frames[0]
    for k in range(1, len(frames)):
        merged = pd.merge(merged, frames[k], on="epoch", how="outer")
    merged = merged.sort_values("epoch").reset_index(drop=True)
    merged["epoch"] = merged["epoch"].astype(int)
    return merged


def agg_mean_std(df: pd.DataFrame, col_prefix: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate columns starting with col_prefix into mean/std by epoch.

    If a directory yields columns named exactly 'val_auc', 'val_auc_x', 'val_auc_y', ...
    we group by column pattern.
    """
    # Pick columns for this metric
    cols = [c for c in df.columns if c.startswith(col_prefix)]
    # If there's a single column equal to the prefix, still use it
    if col_prefix in df.columns and col_prefix not in cols:
        cols = [col_prefix]
    if not cols:
        return df["epoch"].values, np.full(len(df), np.nan), np.full(len(df), np.nan)

    # Compute mean/std ignoring NaN
    values = df[cols].copy()
    mean = values.mean(axis=1, skipna=True).to_numpy()
    std = values.std(axis=1, ddof=1, skipna=True).to_numpy()
    return df["epoch"].to_numpy(), mean, std


def plot_compare(model_data: Dict[str, pd.DataFrame], out: str = None) -> None:
    """Plot three panels comparing models: AUC, PR-AUC, Logloss."""
    plt.figure(figsize=(15, 6))
    gs = plt.GridSpec(1, 3, wspace=0.22)

    panels = [
        ("Validation AUC ↑", "val_auc", 0),
        ("Validation PR-AUC ↑", "val_prauc", 1),
        ("Validation Logloss ↓", "val_logloss", 2),
    ]

    # A few pleasant colors
    palette = [
        "#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    handles = []
    labels = []

    for j, (title, key, colidx) in enumerate(panels):
        ax = plt.subplot(gs[0, j])
        ax.set_title(title, fontsize=14, pad=10)

        for i, (name, df) in enumerate(model_data.items()):
            # Duplicate metric columns are suffixed by pandas; use prefix to aggregate
            # Ensure we have one base column (if no suffix was created)
            # Create consistent prefix columns:
            base_col = key
            # Rename the single base column (if exists) to f"{key}_0" so aggregation picks it up
            tmp = df.copy()
            if key in tmp.columns and not any(c.startswith(key + "_") for c in tmp.columns):
                tmp.rename(columns={key: f"{key}_0"}, inplace=True)

            # Reindex columns to make sure epoch is present
            if "epoch" not in tmp.columns:
                continue

            # Aggregate
            e, mean, std = agg_mean_std(tmp, key)
            # If all NaN, skip
            if np.all(np.isnan(mean)):
                continue

            color = palette[i % len(palette)]
            ax.plot(e, mean, color=color, lw=2)
            ax.fill_between(e, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)

            if j == 0:  # collect legend only once
                handles.append(ax.plot([], [], color=color, lw=3)[0])
                labels.append(name)

        ax.set_xlabel("Epoch")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(alpha=0.25)

    # Legend
    if handles:
        plt.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(len(labels), 4), frameon=False)

    plt.tight_layout()

    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=220, bbox_inches="tight")
        print(f"[OK] Saved to {out}")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"reports/plots/training_compare_{ts}.png"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=220, bbox_inches="tight")
        print(f"[OK] Saved to {out}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare validation curves across multiple models.")
    parser.add_argument(
        "--logdir",
        action="append",
        required=True,
        help="Log dir spec. Format: path[:label]. Can be used multiple times.",
    )
    parser.add_argument(
        "--pattern",
        default="train_log_fold*.csv",
        help="Glob pattern inside each logdir.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Default: reports/plots/training_compare_<ts>.png",
    )
    args = parser.parse_args()

    model_data = {}
    for spec in args.logdir:
        if ":" in spec:
            path, label = spec.split(":", 1)
        else:
            path, label = spec, os.path.basename(os.path.normpath(spec))
        path = path.strip()
        label = label.strip()
        df = load_one_dir(path, args.pattern)
        model_data[label] = df

    plot_compare(model_data, out=args.out)


if __name__ == "__main__":
    main()