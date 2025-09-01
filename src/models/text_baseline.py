#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train TF-IDF + SGD(logistic) with L2 and early stopping.
Output per-comment p_sensitive for later user-level agg.
"""

import argparse, os, sys
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comments", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--user-col", default="username")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")  # 可无
    # 正则化/早停
    ap.add_argument("--alpha", type=float, default=1e-5)         # L2强度
    ap.add_argument("--min-df", type=int, default=3)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--max-features", type=int, default=500000)
    ap.add_argument("--early-stopping", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--n-iter-no-change", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.comments)
    if args.text_col not in df.columns or args.user_col not in df.columns:
        sys.exit(f"Missing columns: need [{args.user_col}, {args.text_col}]")
    has_label = (args.label_col in df.columns)

    # y（可选；无则用弱标注关键词→此处退化为无监督，概率仅用于排序）
    if has_label:
        y = pd.to_numeric(df[args.label_col], errors="coerce").fillna(0).astype(int).values
        classes = np.array([0,1], dtype=int)
        # class weight
        pos_rate = float((y==1).mean())
        cw = "balanced" if (pos_rate>0 and pos_rate<1) else None
    else:
        y = None
        classes, cw = None, None

    # 模型
    vec = TfidfVectorizer(
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        max_features=args.max_features
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

    # 训练
    if has_label:
        pipe.fit(df[args.text_col].astype(str).values, y)
        # 简要评估（有标签才算）
        try:
            p = pipe.predict_proba(df[args.text_col].astype(str).values)[:,1]
            auc = roc_auc_score(y, p); ap = average_precision_score(y, p)
            print(f"[train] AUC={auc:.3f} PR-AUC={ap:.3f} pos={y.mean():.3f}")
        except Exception:
            pass
    else:
        pipe.fit(df[args.text_col].astype(str).values, np.zeros(len(df)))  # 占位

    # 预测并输出
    prob = pipe.predict_proba(df[args.text_col].astype(str).values)[:,1]
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