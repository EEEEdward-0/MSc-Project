#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train a text-only model on aggregated text features.

Inputs:
  - data/processed/splits/dev_folds_text.csv
  - data/processed/splits/test_text.csv

Outputs (reports/model_text):
  - oof_dev_text.csv, pred_test_text.csv
  - metrics_cv.csv, feature_importance.csv

Optional (--calibrate):
  - oof_dev_text_iso.csv, pred_test_text_iso.csv
  - isotonic_calibrator_text.joblib

Also prints shuffled-label AUC and adversarial AUC.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import os
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
import joblib
import csv


# === Config ===
DATA = Path("data/processed/splits")
OUT  = Path("reports/model_text")
IDCOL, LABEL, FOLD = "user_id", "label", "fold"
FEATS_BASE = ["text_mean_prob", "text_max_prob", "text_frac_sensitive", "text_n_comments"]


# === IO ===
def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev_p  = DATA / "dev_folds_text.csv"
    tst_p  = DATA / "test_text.csv"
    if not dev_p.exists():
        raise FileNotFoundError(f"missing {dev_p}")
    if not tst_p.exists():
        raise FileNotFoundError(f"missing {tst_p}")

    dev = pd.read_csv(dev_p)
    tst = pd.read_csv(tst_p)

    # basic checks
    need = [IDCOL, LABEL, FOLD] + FEATS_BASE
    for c in need:
        if c not in dev.columns:
            raise ValueError(f"dev missing col: {c}")
    if IDCOL not in tst.columns:
        raise ValueError(f"test missing col: {IDCOL}")
    for c in FEATS_BASE:
        if c not in tst.columns:
            tst[c] = 0.0

    # dtypes
    for c in FEATS_BASE:
        dev[c] = dev[c].astype("float32").fillna(0.0)
        tst[c] = tst[c].astype("float32").fillna(0.0)
    dev[LABEL] = dev[LABEL].astype(int)

    return dev, tst


# === Safety checks ===
def single_feature_scan(dev: pd.DataFrame, feats: list[str]) -> list[str]:
    """Drop features whose single-column AUC ≥ 0.95."""
    drop = []
    y = dev[LABEL].values
    for c in feats:
        x = dev[c].values
        try:
            auc = roc_auc_score(y, x)
        except Exception:
            auc = 0.5
        if auc >= 0.95:
            drop.append(c)
    return [f for f in feats if f not in drop], drop


def shuffled_label_auc(dev: pd.DataFrame, feats: list[str], seed: int) -> float:
    """Estimate AUC after shuffling labels using proper OOF CV.

    Why: training and evaluating on the same permuted labels can overfit and
    produce spuriously high AUCs. We instead compute OOF predictions by
    training on the permuted labels of the training folds and evaluating on the
    corresponding validation fold.
    """
    rng = np.random.default_rng(seed)
    y_perm = rng.permutation(dev[LABEL].to_numpy())

    # OOF container
    oof = np.zeros(len(dev), dtype=float)
    folds = sorted(dev[FOLD].dropna().unique())

    # Mild learner to avoid memorising noise
    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=31,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        min_data_in_leaf=50,
        seed=seed,
        verbose=-1,
    )

    for f in folds:
        tr_idx = dev[FOLD] != f
        va_idx = dev[FOLD] == f

        trn = lgb.Dataset(dev.loc[tr_idx, feats], label=y_perm[tr_idx])
        val = lgb.Dataset(dev.loc[va_idx, feats], label=y_perm[va_idx])

        booster = lgb.train(
            params,
            trn,
            num_boost_round=1000,
            valid_sets=[val],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof[va_idx] = booster.predict(
            dev.loc[va_idx, feats], num_iteration=booster.best_iteration
        )

    return float(roc_auc_score(y_perm, oof))


def adversarial_auc(dev: pd.DataFrame, tst: pd.DataFrame, feats: list[str], seed: int) -> float:
    """AUC of a classifier distinguishing dev vs test (domain shift)."""
    a = dev[[*feats]].copy()
    b = tst[[*feats]].copy()
    a["y"] = 0
    b["y"] = 1
    allx = pd.concat([a, b], axis=0, ignore_index=True)
    y = allx.pop("y").values
    x = allx.values
    clf = LogisticRegression(max_iter=1000, n_jobs=8, random_state=seed)
    clf.fit(x, y)
    p = clf.predict_proba(x)[:, 1]
    return float(roc_auc_score(y, p))


# === CV train ===
def train_cv(dev: pd.DataFrame, feats: list[str], seed: int) -> tuple[np.ndarray, list[tuple[int, float, float, float]], np.ndarray, int]:
    folds = sorted(dev[FOLD].dropna().unique())
    oof = np.zeros(len(dev), dtype=float)
    imp_sum = np.zeros(len(feats), dtype=float)
    logs: list[tuple[int, float, float, float]] = []
    best_iters: list[int] = []

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.03,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        min_data_in_leaf=50,
        seed=seed,
        verbose=-1,
    )

    for f in folds:
        tr = dev[dev[FOLD] != f]
        va = dev[dev[FOLD] == f]
        trn = lgb.Dataset(tr[feats], label=tr[LABEL].astype(int))
        val = lgb.Dataset(va[feats], label=va[LABEL].astype(int))

        evals_result = {}
        booster = lgb.train(
            params,
            trn,
            num_boost_round=3000,
            valid_sets=[val],
            valid_names=["valid"],
            # Removed evals_result=evals_result,
            callbacks=[
                lgb.early_stopping(200, verbose=False),
                lgb.record_evaluation(evals_result),
            ],
        )

        oof_idx = va.index.values
        oof[oof_idx] = booster.predict(va[feats], num_iteration=booster.best_iteration)

        imp_sum += booster.feature_importance(importance_type="gain")
        auc   = roc_auc_score(va[LABEL], oof[oof_idx])
        pr    = average_precision_score(va[LABEL], oof[oof_idx])
        brier = brier_score_loss(va[LABEL], oof[oof_idx])
        best_iters.append(int(booster.best_iteration or 200))
        logs.append((int(f), float(auc), float(pr), float(brier)))
        print(f"[CV] fold={f} AUC={auc:.3f} PR-AUC={pr:.3f} Brier={brier:.4f}")

        # write epoch-wise metrics for this fold (text model)
        try:
            OUT.mkdir(parents=True, exist_ok=True)
            log_csv = OUT / f"train_log_fold{int(f)}.csv"
            metrics_avail = sorted(evals_result.get("valid", {}).keys())
            epochs = 0
            for m in metrics_avail:
                epochs = max(epochs, len(evals_result["valid"].get(m, [])))
            with log_csv.open("w", newline="") as wf:
                w = csv.writer(wf)
                header = ["epoch"]
                if "binary_logloss" in metrics_avail:
                    header += ["loss"]
                for m in metrics_avail:
                    if m == "binary_logloss":
                        continue
                    header += [m]
                w.writerow(header)
                for e in range(epochs):
                    row = [e + 1]
                    if "binary_logloss" in metrics_avail:
                        row.append(evals_result["valid"]["binary_logloss"][e])
                    for m in metrics_avail:
                        if m == "binary_logloss":
                            continue
                        row.append(evals_result["valid"][m][e])
                    w.writerow(row)
            print(f"[log] wrote {log_csv}")
        except Exception as _e:
            print("[log] skip fold log:", _e)

    imp = (imp_sum / imp_sum.sum()) if imp_sum.sum() > 0 else np.zeros_like(imp_sum)
    full_rounds = max(200, int(np.clip(np.mean(best_iters) * 1.2, 200, 4000)))
    return oof, logs, imp, full_rounds


# === Final fit + save ===
def fit_full_and_predict(dev: pd.DataFrame, tst: pd.DataFrame, feats: list[str], rounds: int, seed: int) -> np.ndarray:
    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.03,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        min_data_in_leaf=50,
        seed=seed,
        verbose=-1,
    )
    trn = lgb.Dataset(dev[feats], label=dev[LABEL].astype(int))
    booster = lgb.train(params, trn, num_boost_round=rounds)
    return booster.predict(tst[feats])


# === Main ===
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibrate", action="store_true")           # isotonic on OOF
    ap.add_argument("--predict-test", action="store_true")        # always true for this script, flag kept for parity
    args = ap.parse_args()

    np.random.seed(args.seed)

    dev, tst = load_tables()

    # strict feature screening
    feats, dropped = single_feature_scan(dev, FEATS_BASE)
    if dropped:
        print("[strict] drop suspicious single-col AUC:", ", ".join(dropped))
    print("[feats]", feats)

    # CV
    oof, logs, imp_norm, n_rounds = train_cv(dev, feats, args.seed)

    # save OOF/metrics/importance
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({IDCOL: dev[IDCOL], "oof_pred_text": oof}).to_csv(OUT / "oof_dev_text.csv", index=False)

    auc   = roc_auc_score(dev[LABEL], oof)
    pr    = average_precision_score(dev[LABEL], oof)
    brier = brier_score_loss(dev[LABEL], oof)
    pd.DataFrame([{"n": len(dev), "auc": auc, "prauc": pr, "brier": brier}]).to_csv(OUT / "metrics_cv.csv", index=False)
    pd.DataFrame({"feature": feats, "importance": imp_norm}).to_csv(OUT / "feature_importance.csv", index=False)
    print(f"[CV] n={len(dev)} AUC={auc:.3f} PR-AUC={pr:.3f} Brier={brier:.4f}")

    # sanity: shuffled labels
    auc_shuf = shuffled_label_auc(dev, feats, args.seed)
    print(f"[Sanity] shuffled-label AUC={auc_shuf:.3f}")

    # sanity: adversarial (dev vs test)
    adv_auc = adversarial_auc(dev, tst, feats, args.seed)
    print(f"[Sanity] adversarial AUC (dev vs test)={adv_auc:.3f}")

    # final fit + test prediction
    ptest = fit_full_and_predict(dev, tst, feats, n_rounds, args.seed)
    pd.DataFrame({IDCOL: tst[IDCOL], "pred_text": ptest}).to_csv(OUT / "pred_test_text.csv", index=False)
    print(f"[TEST] wrote {OUT/'pred_test_text.csv'}")

    # optional calibration
    if args.calibrate:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof, dev[LABEL].astype(int).values)
        oof_cal = iso.transform(oof)
        pd.DataFrame({IDCOL: dev[IDCOL], "oof_cal": oof_cal, LABEL: dev[LABEL]}).to_csv(OUT / "oof_dev_text_iso.csv", index=False)

        ptest_cal = iso.transform(ptest)
        pd.DataFrame({IDCOL: tst[IDCOL], "pred_cal": ptest_cal}).to_csv(OUT / "pred_test_text_iso.csv", index=False)
        joblib.dump(iso, OUT / "isotonic_calibrator_text.joblib")
        print("[CAL] isotonic saved.")

    # write summary json
    summary = {
        "n_dev": int(len(dev)),
        "n_test": int(len(tst)),
        "features": feats,
        "dropped_single_auc": dropped,
        "cv": [{"fold": f, "auc": a, "prauc": p, "brier": b} for (f, a, p, b) in logs],
        "cv_overall": {"auc": float(auc), "prauc": float(pr), "brier": float(brier)},
        "sanity": {"shuffled_auc": float(auc_shuf), "adversarial_auc": float(adv_auc)},
        "rounds_full": int(n_rounds),
        "calibrated": bool(args.calibrate),
    }
    with open(OUT / "summary_text.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[done] summary ->", OUT / "summary_text.json")


if __name__ == "__main__":
    main()