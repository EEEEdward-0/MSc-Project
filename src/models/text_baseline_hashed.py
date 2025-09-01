#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train HashingVectorizer + SGD(logistic) with L2 and early stopping.
Faster + lower memory; keep same output schema.
"""

import argparse, os, sys
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comments", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--user-col", default="username")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    # hashing + 正则 + 早停
    ap.add_argument("--n_features", type=int, default=2**20)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=1e-5)
    ap.add_argument("--early-stopping", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--n-iter-no-change", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    # 大数据可随机下采样文本（可选）
    ap.add_argument("--max_rows", type=int, default=0)  # 0=全部
    return ap.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.comments, usecols=lambda c: c in {args.user_col, args.text_col, args.label_col})
    if args.text_col not in df.columns or args.user_col not in df.columns:
        sys.exit(f"Missing columns: need [{args.user_col}, {args.text_col}]")
    df = df.dropna(subset=[args.text_col])

    # 采样（可选）
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=args.seed).reset_index(drop=True)

    has_label = (args.label_col in df.columns)
    if has_label:
        y = pd.to_numeric(df[args.label_col], errors="coerce").fillna(0).astype(int).values
        cw = "balanced" if 0 < y.mean() < 1 else None
    else:
        y, cw = None, None

    vec = HashingVectorizer(
        n_features=args.n_features,
        alternate_sign=False,
        ngram_range=(1, args.ngram_max),
        norm="l2"
    )
    clf = SGDClassifier(
        loss="log_loss", penalty="l2",
        alpha=args.alpha,
        early_stopping=args.early_stopping,
        validation_fraction=args.val_frac,
        n_iter_no_change=args.n_iter_no_change,
        max_iter=args.max_iter,
        random_state=args.seed,
        class_weight=cw
    )
    pipe = make_pipeline(vec, clf)

    Xtxt = df[args.text_col].astype(str).values
    if has_label:
        pipe.fit(Xtxt, y)
        try:
            p = pipe.predict_proba(Xtxt)[:,1]
            auc = roc_auc_score(y, p); ap = average_precision_score(y, p)
            print(f"[train] AUC={auc:.3f} PR-AUC={ap:.3f} pos={y.mean():.3f}")
        except Exception:
            pass
    else:
        pipe.fit(Xtxt, np.zeros(len(df)))

    prob = pipe.predict_proba(Xtxt)[:,1]
    out = pd.DataFrame({
        args.user_col: df[args.user_col].astype(str).values,
        "p_sensitive": prob,
        "text_len": df[args.text_col].astype(str).str.len().fillna(0).astype(int).values
    })
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[done] wrote {args.out}, rows={len(out)}")

if __name__ == "__main__":
    main()