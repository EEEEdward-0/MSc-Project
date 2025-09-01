#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Blend model utilities:
- Option A: write per-fold, single-epoch logs from a final blend summary JSON.
- Option B: run a LightGBM CV on given features to produce epoch-wise logs.

This file is intentionally simple and focused on producing logs under:
    reports/model_blend/train_log_fold{fold}.csv
"""

from pathlib import Path
import argparse
import csv
import json
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss


def load_oof(path: str, key: str, pred_name: str) -> pd.DataFrame:
    """
    Load an OOF CSV file with flexible column names for predictions and optional fold.
    Returns a DataFrame with columns: [key, 'fold' (if present), 'pred_{pred_name}'].
    Raises ValueError if key or prediction column not found.
    """
    df = pd.read_csv(path)
    if key not in df.columns:
        raise ValueError(f"Key column '{key}' not found in OOF file: {path}")
    # Find fold column if exists
    fold_col = None
    for c in df.columns:
        if c.lower() == "fold":
            fold_col = c
            break
    # Find prediction column: try common names
    pred_cols_candidates = [c for c in df.columns if c.lower() in {"pred", "prediction", "score", "proba", "probability"}]
    if not pred_cols_candidates:
        # fallback: any numeric column except key and fold
        pred_cols_candidates = [c for c in df.select_dtypes(include=[np.number]).columns if c not in {key, fold_col}]
    if not pred_cols_candidates:
        raise ValueError(f"No prediction column found in OOF file: {path}")
    pred_col = pred_cols_candidates[0]
    # Build output dataframe
    out_df = df[[key]].copy()
    if fold_col is not None:
        out_df["fold"] = df[fold_col]
    out_df[f"pred_{pred_name}"] = df[pred_col]
    return out_df


# ---- feature selection (align with train_lgbm strict mode) -------------------
LEAKY_COLS_DEFAULT = {
    # activity totals / logs
    "total_activity", "log_total_activity",
    "comments", "log_comments",
    "posts", "log_posts",
    # ratios / flags
    "comment_post_ratio", "post_comment_ratio",
    "is_post_heavy", "is_comment_heavy", "is_empty",
    # label-like username/text flags
    "username_is_highrisk", "username_risk_score",
    "text_n_comments", "text_max_prob",
}

ID_LABEL_COLS = {"user_id", "username", "label", "fold"}

def select_features_strict(df: pd.DataFrame, extra_drop: list[str] | None = None) -> list[str]:
    extra = set([c.strip() for c in (extra_drop or []) if c.strip()])
    drop = ID_LABEL_COLS | LEAKY_COLS_DEFAULT | extra
    # base candidates: numeric columns excluding drops
    feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]
    # keep order, unique
    seen = set(); feats = [c for c in feats if not (c in seen or seen.add(c))]
    return feats


# ----------------------------- I/O helpers ------------------------------------
def write_logs_from_summary(summary_path: str,
                            out_dir: str = "reports/model_blend",
                            folds: int = 5) -> bool:
    """
    Create train_log_fold*.csv from a final blend summary JSON (no-epoch blender).
    Returns True if files were written, else False.
    Expected keys (any subset): auc / val_auc / AUC, prauc / ap / average_precision,
                                logloss / binary_logloss / brier / loss
    """
    p = Path(summary_path)
    if not p.exists():
        return False

    try:
        js = json.loads(p.read_text())
    except Exception:
        return False

    auc = js.get("auc") or js.get("val_auc") or js.get("AUC")
    prauc = js.get("prauc") or js.get("ap") or js.get("average_precision") or js.get("PR_AUC")
    loss = js.get("logloss") or js.get("binary_logloss") or js.get("brier") or js.get("loss")

    if auc is None and prauc is None and loss is None:
        return False

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Canonical order: val_logloss, val_auc, val_prauc (with epoch first)
    for k in range(folds):
        with (outp / f"train_log_fold{k}.csv").open("w", newline="") as wf:
            w = csv.writer(wf)
            cols, vals = ["epoch"], [1]
            # Always order: val_logloss, val_auc, val_prauc
            if loss is not None:
                cols.append("val_logloss");  vals.append(loss)
            if auc is not None:
                cols.append("val_auc");   vals.append(auc)
            if prauc is not None:
                cols.append("val_prauc"); vals.append(prauc)
            w.writerow(cols)
            w.writerow(vals)

    print(f"[blend-log] wrote single-epoch logs from {summary_path} -> {out_dir}")
    return True


# ----------------------------- Training (optional) ----------------------------
def cv_train(dev: pd.DataFrame,
            feats: List[str],
            out_dir: str = "reports/model_blend",
            seed: int = 42,
            num_boost_round: int = 3000,
            early_stopping_rounds: int = 200) -> None:
    """
    LightGBM CV for producing epoch-wise validation logs.
    Assumes `dev` has at least: columns {fold, label} and the given `feats`.
    """
    folds = list(pd.unique(dev["fold"]))
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    params: Dict[str, object] = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss", "average_precision"],
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": seed,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "min_data_in_leaf": 20,
        "is_unbalance": True,
    }

    for f in folds:
        trn_idx = dev["fold"] != f
        val_idx = dev["fold"] == f

        Xtr = dev.loc[trn_idx, feats]
        ytr = dev.loc[trn_idx, "label"].astype(int)
        Xva = dev.loc[val_idx, feats]
        yva = dev.loc[val_idx, "label"].astype(int)

        dtr = lgb.Dataset(Xtr, label=ytr)
        dva = lgb.Dataset(Xva, label=yva)

        evals_result: Dict[str, Dict[str, List[float]]] = {}
        booster = lgb.train(
            params,
            dtr,
            num_boost_round=num_boost_round,
            valid_sets=[dva],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.record_evaluation(evals_result),
            ],
        )

        # Fold-level summary
        pred = booster.predict(Xva)
        try:
            auc = roc_auc_score(yva, pred)
        except Exception:
            auc = None
        try:
            ap = average_precision_score(yva, pred)
        except Exception:
            ap = None
        try:
            ll = log_loss(yva, pred, labels=[0, 1])
        except Exception:
            ll = None

        msg = [f"Fold {f}"]
        if auc is not None: msg.append(f"AUC={auc:.4f}")
        if ap  is not None: msg.append(f"PR-AUC={ap:.4f}")
        if ll  is not None: msg.append(f"Logloss={ll:.4f}")
        print(" ".join(msg))

        # Epoch-wise log for this fold
        valid_hist = evals_result.get("valid", {})
        epochs = max((len(v) for v in valid_hist.values()), default=0)
        # Canonical order: val_logloss, val_auc, val_prauc
        log_csv = outp / f"train_log_fold{int(f)}.csv"
        with log_csv.open("w", newline="") as wf:
            w = csv.writer(wf)
            header = ["epoch"]
            col_keys = []
            if "binary_logloss" in valid_hist:
                header.append("val_logloss")
                col_keys.append("binary_logloss")
            if "auc" in valid_hist:
                header.append("val_auc")
                col_keys.append("auc")
            if "average_precision" in valid_hist:
                header.append("val_prauc")
                col_keys.append("average_precision")
            w.writerow(header)
            for e in range(epochs):
                row = [e + 1]
                for k in col_keys:
                    row.append(valid_hist[k][e])
                w.writerow(row)
        print(f"[blend-log] wrote {log_csv}")


# ----------------------------- CLI entry --------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Blend model trainer / logger")
    ap.add_argument("--dev", default="data/processed/train.csv",
                    help="Dev CSV with folds/labels (optional).")
    ap.add_argument("--feats", default=None,
                    help="Comma-separated feature list; if omitted, auto-select numeric columns.")
    ap.add_argument("--outdir", default="reports/model_blend",
                    help="Where to write train_log_fold*.csv")
    ap.add_argument("--summary", default=None,
                    help="If provided (or auto-detected), write single-epoch logs from JSON.")
    ap.add_argument("--mode", choices=["strict", "leaky"], default="strict",
                    help="Feature filtering: strict drops suspected leakage features; leaky keeps all numeric non-ID")
    ap.add_argument("--extra-drop", default="",
                    help="Comma-separated extra columns to drop when mode=strict")
    ap.add_argument("--oof-binary", default=None,
                    help="Path to binary model OOF CSV for stacking.")
    ap.add_argument("--oof-text", default=None,
                    help="Path to text/semantic model OOF CSV for stacking.")
    ap.add_argument("--key", default="user_id",
                    help="Join key column name for OOF merges.")
    args = ap.parse_args()

    if args.oof_binary or args.oof_text:
        dev_path = Path(args.dev)
        if not dev_path.exists():
            print(f"[warn] dev CSV not found: {dev_path}. Nothing to do.")
            return
        df_dev = pd.read_csv(dev_path)
        key = args.key
        if key not in df_dev.columns:
            print(f"[error] Key column '{key}' not found in dev CSV.")
            return

        oof_frames = []
        if args.oof_binary:
            try:
                df_bin = load_oof(args.oof_binary, key, "bin")
                oof_frames.append(df_bin)
            except Exception as e:
                print(f"[error] Failed to load binary OOF: {e}")
                return
        if args.oof_text:
            try:
                df_txt = load_oof(args.oof_text, key, "txt")
                oof_frames.append(df_txt)
            except Exception as e:
                print(f"[error] Failed to load text OOF: {e}")
                return

        # Merge OOFs into dev
        merged_df = df_dev.copy()
        for oof_df in oof_frames:
            # If OOF has fold, merge fold; else keep fold from dev
            if "fold" in oof_df.columns:
                merged_df = merged_df.merge(oof_df, on=[key, "fold"], how="inner", validate="one_to_one")
            else:
                merged_df = merged_df.merge(oof_df.drop(columns=["fold"], errors="ignore"), on=key, how="inner", validate="one_to_one")

        if "label" not in merged_df.columns:
            print("[error] Merged dataframe missing 'label' column.")
            return
        if "fold" not in merged_df.columns:
            print("[error] Merged dataframe missing 'fold' column.")
            return

        pred_cols = [f for f in ["pred_bin", "pred_txt"] if f in merged_df.columns]
        if not pred_cols:
            print("[error] No prediction columns found in merged OOFs.")
            return
        print(f"[stacking] Using stacking mode with features: {', '.join(pred_cols)}")
        cv_train(merged_df, pred_cols, out_dir=args.outdir)
        return

    # Priority 1: explicit summary path
    if args.summary and write_logs_from_summary(args.summary, args.outdir):
        return

    # Priority 2: auto-detect common summary locations
    for cand in [
        "reports/final_blend_summary.json",
        "final_blend_summary.json",
        "reports/model_blend/final_blend_summary.json",
    ]:
        if write_logs_from_summary(cand, args.outdir):
            return

    # Priority 3: CV train to emit epoch-wise logs
    dev_path = Path(args.dev)
    if not dev_path.exists():
        print(f"[warn] dev CSV not found: {dev_path}. Nothing to do.")
        return

    df = pd.read_csv(dev_path)
    if not {"fold", "label"}.issubset(df.columns):
        print(f"[warn] dev CSV missing 'fold'/'label' columns: {dev_path}. Nothing to do.")
        return

    if args.feats:
        feats = [x.strip() for x in args.feats.split(",") if x.strip()]
    else:
        if args.mode == "strict":
            extra = [s for s in (args.extra_drop.split(",") if args.extra_drop else []) if s]
            feats = select_features_strict(df, extra_drop=extra)
        else:
            drop = {"fold", "label", "user_id"}
            feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]

    if not feats:
        print("[warn] No features left after filtering; try --mode leaky or adjust --extra-drop.")
        return
    print(f"[feat] mode={args.mode} using {len(feats)} features: {', '.join(feats)}")

    cv_train(df, feats, out_dir=args.outdir)


if __name__ == "__main__":
    main()