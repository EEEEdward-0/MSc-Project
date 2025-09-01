#!/usr/bin/env python3
"""
Compare training curves across multiple models (AUC / PR-AUC / Logloss).

Usage:
  python scripts/plot_training_compare.py \
    --logdir reports/model_binary:Binary \
    --logdir reports/model_blend:Blend \
    --pattern "train_log_fold*.csv" \
    --out reports/plots/training_compare.png \
    [--allow-missing] [--debug]
"""

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---- visual style (clean + high-contrast) ----
plt.style.use("seaborn-v0_8-whitegrid")
rcParams.update({
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", action="append", required=True,
                    help="Path:Label, e.g. reports/model_binary:Binary (repeatable)")
    ap.add_argument("--pattern", default="train_log_fold*.csv",
                    help="Glob pattern for per-fold logs (default: train_log_fold*.csv)")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--allow-missing", action="store_true",
                    help="Skip missing/empty dirs instead of failing")
    ap.add_argument("--debug", action="store_true",
                    help="Print debug information for loaded data")
    return ap.parse_args()


# Map common column aliases to canonical names
ALIAS = {
    "auc": "val_auc",
    "AUC": "val_auc",
    "ap": "val_prauc",
    "average_precision": "val_prauc",
    "binary_logloss": "val_logloss",
    "logloss": "val_logloss",
}

METRICS = ["val_auc", "val_prauc", "val_logloss"]


def _rename_aliases(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: ALIAS.get(c, c) for c in df.columns}
    if any(rename[c] != c for c in df.columns):
        df = df.rename(columns=rename)
    return df


def load_one_dir(log_dir: str, pattern: str) -> Optional[pd.DataFrame]:
    """Load all per-fold CSVs from a directory and merge by epoch.

    Returns a wide dataframe with columns:
      epoch, val_auc_0..k, val_prauc_0..k, val_logloss_0..k
    or None if nothing matched.
    """
    files = sorted(glob.glob(os.path.join(log_dir, pattern)))
    if not files:
        return None

    merged: Optional[pd.DataFrame] = None
    for k, fp in enumerate(files):
        df = pd.read_csv(fp)
        df = _rename_aliases(df)
        if "epoch" not in df.columns:
            # try to synthesize epoch as 1..N
            df = df.copy()
            df.insert(0, "epoch", np.arange(1, len(df) + 1, dtype=int))

        # keep only relevant columns
        keep = ["epoch"] + [c for c in df.columns if c in METRICS]
        df = df[keep]

        # suffix by fold index
        rename = {c: f"{c}_{k}" for c in df.columns if c != "epoch"}
        df = df.rename(columns=rename)

        merged = df if merged is None else pd.merge(
            merged, df, on="epoch", how="outer", suffixes=(None, None)
        )

    # ensure epoch sorted and int
    merged = merged.sort_values("epoch").reset_index(drop=True)
    merged["epoch"] = merged["epoch"].astype(int)
    return merged


def mean_std(df_wide: pd.DataFrame, metric: str) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    cols = [c for c in df_wide.columns if c.startswith(metric + "_")]
    if not cols:
        return None, None
    return df_wide[cols].mean(axis=1), df_wide[cols].std(axis=1)


def plot_compare(models: List[Tuple[str, str, pd.DataFrame]], out_png: str) -> None:
    # Prepare figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # Fixed colors for common labels; fallback to matplotlib cycle
    palette: Dict[str, str] = {
        "Binary": "#FF7F0E",          # orange
        "Semantic/Text": "#1F77B4",   # blue
        "Blend": "#2CA02C",           # green
    }

    for (path, label, df_wide) in models:
        color = palette.get(label, None)
        epoch = df_wide["epoch"]

        panels = [
            ("Validation AUC ↑", "val_auc"),
            ("Validation PR-AUC ↑", "val_prauc"),
            ("Validation Logloss ↓", "val_logloss"),
        ]

        for ax, (title, metric) in zip(axes, panels):
            m, s = mean_std(df_wide, metric)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            # Y label
            if "AUC" in title:
                ax.set_ylabel("AUC")
            elif "PR-AUC" in title:
                ax.set_ylabel("PR-AUC")
            else:
                ax.set_ylabel("Logloss")

            if m is not None and not m.isna().all():
                y = m.values
                line_kw = dict(lw=2.2, solid_capstyle="round")
                if color:
                    ax.plot(epoch, y, label=label, color=color, **line_kw)
                else:
                    ax.plot(epoch, y, label=label, **line_kw)

                # Mark the last epoch point
                ax.plot([epoch.iloc[-1]], [y[-1]], marker="o", markersize=4,
                        color=color if color else ax.lines[-1].get_color())

                # Annotate ±std at final epoch (no shaded band) – safe against NaN/inf
                if s is not None:
                    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    final_epoch = epoch.iloc[-1]
                    final_std = float(np.nan_to_num(s.iloc[-1], nan=0.0))
                    if final_std > 0:
                        ax.text(final_epoch, y[-1], f" ±{final_std:.3f}",
                                fontsize=9, va="bottom", ha="left",
                                color=(color if color else ax.lines[-1].get_color()))
            else:
                # If this model has no data for this metric, we simply skip plotting.
                pass

            ax.grid(True, alpha=0.25)

    # Build a single legend from unique labels
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.035), ncol=len(labels), frameon=False)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] Saved to {out_png}")


def main() -> None:
    args = parse_args()

    # Parse --logdir specs
    specs: List[Tuple[str, str]] = []
    for spec in args.logdir:
        if ":" in spec:
            path, label = spec.split(":", 1)
        else:
            path, label = spec, os.path.basename(spec.rstrip("/"))
        specs.append((path, label))

    loaded: List[Tuple[str, str, pd.DataFrame]] = []
    for path, label in specs:
        df = load_one_dir(path, args.pattern)
        if df is None or df.empty:
            msg = f"[warn] No logs under {path} matching {args.pattern}"
            if args.allow_missing:
                print(msg)
                continue
            else:
                raise FileNotFoundError(msg)
        if args.debug:
            cols = df.columns.tolist()
            print(f"[debug] Loaded '{label}' from {path}")
            print(f"        rows={len(df)}, cols={cols[:8]}{'...' if len(cols)>8 else ''}")
            for k in METRICS:
                ks = [c for c in cols if c.startswith(k + "_")]
                print(f"        {k}: {len(ks)} folds")
        loaded.append((path, label, df))

    if not loaded:
        raise SystemExit("[error] No valid models to plot.")

    plot_compare(loaded, args.out)


if __name__ == "__main__":
    main()