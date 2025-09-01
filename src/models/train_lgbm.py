#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train username-feature LightGBM (tabular).
Inputs:
  - data/processed/splits/dev_folds.csv
  - data/processed/splits/test.csv
Outputs (dir = reports/model):
  - oof_dev.csv          # user_id,oof,label
  - pred_test.csv        # user_id,pred
  - feature_importance_name.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
import joblib
import json
from sklearn.metrics import roc_curve, f1_score
import csv

# --- util: drop duplicate-named columns safely ---
def _drop_duplicate_columns(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    dup = df.columns[df.columns.duplicated()].unique().tolist()
    if dup:
        print(f"[warn] {name} duplicated columns:", dup)
        # ensure duplicates are identical; if not, stop so we don't hide issues
        cols = pd.Index(df.columns)
        for col in cols[cols.duplicated()].unique():
            idxs = [i for i, c in enumerate(cols) if c == col]
            base = df.iloc[:, idxs[0]]
            for j in idxs[1:]:
                if not base.equals(df.iloc[:, j]):
                    raise ValueError(
                        f"Column '{col}' has multiple non-identical copies at indices {idxs}"
                    )
        # keep first occurrence
        df = df.loc[:, ~df.columns.duplicated()]
    return df

SEED = 42
DATA = Path("data/processed/splits")
OUT  = Path("reports/model_binary")
IDCOL, LABEL, FOLD = "user_id", "label", "fold"

LEAKY_COLS_DEFAULT = {
    # activity totals (strong leakage in your dataset)
    "total_activity", "log_total_activity",
    "comments", "log_comments",
    "posts", "log_posts",
    # activity ratios / heavy flags (rule-like features)
    "comment_post_ratio", "post_comment_ratio",
    "is_post_heavy", "is_comment_heavy",
    # trivial/constant or derived risk flags (avoid encoding label rules)
    "is_empty",
    "username_is_highrisk", "username_risk_score",
    "text_n_comments",
    "text_max_prob",
}

DEFAULT_DROP = {"username", "fold", "label"}  # keep user_id for joins


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict-test", action="store_true", help="only write test predictions if model exists; otherwise train then predict")
    ap.add_argument("--mode", choices=["strict", "leaky"], default="strict", help="feature filtering mode: 'strict' drops suspected leakage features; 'leaky' keeps all non-ID features")
    ap.add_argument("--extra-drop", type=str, default="", help="comma-separated extra columns to drop (in addition to strict set)")
    ap.add_argument("--calibrate", action="store_true", help="fit isotonic calibration on OOF and apply to test predictions")
    return ap.parse_args()

def load_data(mode: str, extra_drop: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    dev_path  = DATA / "dev_folds.csv"
    test_path = DATA / "test.csv"
    if not dev_path.exists() or not test_path.exists():
        raise FileNotFoundError("missing dev/test splits under data/processed/splits")
    dev  = _drop_duplicate_columns(pd.read_csv(dev_path), "dev")
    test = _drop_duplicate_columns(pd.read_csv(test_path), "test")
    # ---- Feature selection ----
    ID_LABEL_COLS = {IDCOL, "username", "label", "fold"}

    # manual leaky set by mode
    leaky_cols = set()
    if mode == "strict":
        leaky_cols = set(LEAKY_COLS_DEFAULT)
        if extra_drop:
            leaky_cols |= set(extra_drop)
    elif mode == "leaky":
        leaky_cols = set()  # keep everything except IDs/labels

    all_cols = list(dev.columns)
    base_feats = [c for c in all_cols if c not in ID_LABEL_COLS]

    # ---- auto detection (strict mode only) ----
    auto_const = set()
    auto_strong = set()
    if mode == "strict":
        y = dev["label"].astype(int).values
        scan_feats = [c for c in base_feats if c not in leaky_cols]
        for c in scan_feats:
            x = pd.to_numeric(dev[c], errors="coerce")
            if x.isna().all():
                auto_const.add(c)
                continue
            x = x.fillna(x.median())
            nunq = x.nunique()
            if nunq <= 1:
                auto_const.add(c)
                continue
            # single-feature AUC probe; if extremely high, treat as leaky
            try:
                auc1 = roc_auc_score(y, x)
            except Exception:
                continue
            if auc1 >= 0.95:
                auto_strong.add(c)

    drop_cols = set(ID_LABEL_COLS) | leaky_cols | auto_const | auto_strong
    feats = [c for c in all_cols if c not in drop_cols]
    # ensure unique feature names order-preserved
    seen = set(); feats = [c for c in feats if not (c in seen or seen.add(c))]

    print(f"[mode] {mode}")
    print("[feat] dropped (ID/label):", sorted(ID_LABEL_COLS & set(all_cols)))
    print("[feat] dropped (manual leaky):", sorted(leaky_cols & set(all_cols)) if leaky_cols else [])
    if mode == "strict":
        print("[feat] auto-drop (constant):", sorted(auto_const) if auto_const else [])
        print("[feat] auto-drop (strong AUC≥0.95):", sorted(auto_strong) if auto_strong else [])
    print(f"[feat] using {len(feats)} features: {', '.join(feats)}")

    # ensure numeric for selected features (both dev/test)
    for c in feats:
        if c in dev.columns:
            dev[c]  = pd.to_numeric(dev[c], errors="coerce").fillna(0.0)
        if c in test.columns:
            test[c] = pd.to_numeric(test[c], errors="coerce").fillna(0.0)

    print(f"[feat] final set ({len(feats)}): {', '.join(feats)}")

    if not feats:
        raise ValueError("No features left after dropping leakage/ID columns")
    return dev, test, feats

def cv_train(dev: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    folds = sorted(dev[FOLD].dropna().unique())
    oof = np.zeros(len(dev), dtype=float)
    imp_sum = np.zeros(len(feats), dtype=float)
    # Request multiple metrics so that training logs contain val_auc, val_logloss, and val_prauc.
    # The plotting script expects columns: epoch, val_logloss, val_auc, val_prauc.
    params = dict(
        objective="binary",
        metric=["auc", "binary_logloss", "average_precision"],  # request AUC, logloss, PR-AUC
        learning_rate=0.03,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        min_data_in_leaf=20,
        seed=SEED,
        verbose=-1,
        is_unbalance=True,
    )
    for f in folds:
        tr = dev[dev[FOLD] != f]
        va = dev[dev[FOLD] == f]
        trn = lgb.Dataset(tr[feats], label=tr[LABEL].astype(int))
        val = lgb.Dataset(va[feats], label=va[LABEL].astype(int))
        evals_result = {}
        booster = lgb.train(
            params, trn, num_boost_round=3000,
            valid_sets=[val], valid_names=["valid"],
            # Removed evals_result=evals_result,
            callbacks=[
                lgb.early_stopping(200, verbose=False),
                lgb.record_evaluation(evals_result),
            ],
        )
        pred = booster.predict(va[feats], num_iteration=booster.best_iteration)
        oof[va.index.values] = pred
        imp_sum += booster.feature_importance(importance_type="gain")
        auc = roc_auc_score(va[LABEL], pred)
        pr  = average_precision_score(va[LABEL], pred)
        print(f"[CV] fold={f} AUC={auc:.3f} PR-AUC={pr:.3f}")

        # Persist per-epoch validation metrics for this fold.
        # We write columns: epoch, val_auc, val_prauc, val_logloss (only if present in valid_hist, in this order).
        try:
            log_dir = Path("reports/model_binary"); log_dir.mkdir(parents=True, exist_ok=True)
            log_csv = log_dir / f"train_log_fold{int(f)}.csv"

            # evals_result structure: {"valid": {"auc": [...], "binary_logloss": [...], "average_precision": [...]}}
            valid_hist = evals_result.get("valid", {})
            epochs = max((len(v) for v in valid_hist.values()), default=0)

            # Determine which columns are present, in canonical order
            header = ["epoch"]
            keys_order = [("auc", "val_auc"), ("average_precision", "val_prauc"), ("binary_logloss", "val_logloss")]
            present = []
            for k, cname in keys_order:
                if k in valid_hist:
                    header.append(cname)
                    present.append((k, cname))

            with log_csv.open("w", newline="") as wf:
                w = csv.writer(wf)
                w.writerow(header)
                for e in range(epochs):
                    row = [e + 1]
                    for k, _ in present:
                        row.append(valid_hist[k][e])
                    w.writerow(row)
            print(f"[log] wrote {log_csv}")
        except Exception as _e:
            print("[log] skip fold log:", _e)
    return oof, imp_sum

def save_reports(dev: pd.DataFrame, oof: np.ndarray, imp_sum: np.ndarray, feats: list[str], test_pred: np.ndarray | None):
    OUT.mkdir(parents=True, exist_ok=True)
    # oof + metrics
    oof_df = pd.DataFrame({"user_id": dev[IDCOL], "oof": oof, "label": dev[LABEL].astype(int)})
    oof_df.to_csv(OUT / "oof_dev.csv", index=False)
    auc   = roc_auc_score(oof_df["label"], oof_df["oof"])
    pr    = average_precision_score(oof_df["label"], oof_df["oof"])
    brier = brier_score_loss(oof_df["label"], oof_df["oof"])
    pd.DataFrame([{"n": len(oof_df), "val_auc": auc, "val_prauc": pr, "val_logloss": brier}]).to_csv(OUT / "metrics_cv.csv", index=False)
    print(f"[CV] n={len(oof_df)} val_auc={auc:.3f} val_prauc={pr:.3f} val_logloss={brier:.4f}")
    # importance
    imp = (imp_sum / imp_sum.sum()) if imp_sum.sum() > 0 else np.zeros_like(imp_sum)
    pd.DataFrame({"feature": feats, "importance": imp}).sort_values("importance", ascending=False)\
        .to_csv(OUT / "feature_importance_name.csv", index=False)
    # test
    if test_pred is not None:
        pd.DataFrame({"user_id": test_ids, "pred": test_pred}).to_csv(OUT / "pred_test.csv", index=False)
        print(f"[TEST] wrote {OUT/'pred_test.csv'}")

# --- thresholds on OOF ---
def _compute_thresholds(oof_df: pd.DataFrame) -> dict:
    """Return conservative (FPR≈5%) and best-F1 thresholds."""
    y = oof_df["label"].astype(int).values
    p = oof_df["oof"].astype(float).values
    # Best-F1
    f1_best, thr_f1 = -1.0, 0.5
    fpr, tpr, thr = roc_curve(y, p)
    for t in thr:
        f1 = f1_score(y, (p >= t).astype(int))
        if f1 > f1_best:
            f1_best, thr_f1 = f1, t
    # FPR≈5%
    target_fpr = 0.05
    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    thr_fpr5 = float(thr[idx]) if len(thr) > 0 else 0.5
    return {"thr_f1": float(thr_f1), "f1": float(f1_best), "thr_fpr5": float(thr_fpr5), "target_fpr": target_fpr}

# --- write final artifacts ---
def _write_final_outputs(test_ids: np.ndarray,
                         test_pred: np.ndarray,
                         thresholds: dict,
                         calibrated: bool):
    """Write final CSVs/figures with conservative threshold."""
    OUT_FINAL = Path("reports/final")
    FIG_DIR = OUT_FINAL / "figs"
    OUT_FINAL.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    thr = float(thresholds.get("thr_fpr5", 0.5))
    df = pd.DataFrame({"user_id": test_ids.astype(str), "p": test_pred.astype(float)})
    df["pred_bin"] = (df["p"] >= thr).astype(int)
    def _tier(p):
        if p >= thr: return "HIGH"
        if p >= 0.7: return "MEDIUM"
        if p >= 0.5: return "LOW"
        return "VERY_LOW"
    df["risk_tier"] = df["p"].apply(_tier)

    df[["user_id","p","pred_bin","risk_tier"]].to_csv(OUT_FINAL / "test_pred_conservative.csv", index=False)

    # thresholds.json
    meta = dict(thresholds)
    meta["calibrated"] = bool(calibrated)
    meta["n_test"] = int(len(df))
    meta["n_pred_high"] = int((df["pred_bin"] == 1).sum())
    meta["share_pred_high"] = float((df["pred_bin"] == 1).mean())
    with open(OUT_FINAL / "thresholds.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Figures (best-effort)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6,4))
        plt.hist(df["p"].values, bins=50)
        plt.axvline(thr, linestyle="--")
        plt.title("TEST score distribution")
        plt.xlabel("prob"); plt.ylabel("count")
        plt.tight_layout(); plt.savefig(FIG_DIR / "test_score_hist.png"); plt.close()

        plt.figure(figsize=(5,5))
        df["risk_tier"].value_counts().plot(kind="pie", autopct="%.1f%%")
        plt.title("TEST risk tiers")
        plt.ylabel("")
        plt.tight_layout(); plt.savefig(FIG_DIR / "test_risk_tiers.png"); plt.close()
    except Exception as e:
        print("[warn] plotting failed:", e)

def main():
    args = parse_args()
    np.random.seed(SEED)
    extra = [s for s in (args.extra_drop.split(",") if args.extra_drop else []) if s]
    dev, test, feats = load_data(args.mode, extra)
    print(f"[info] rows(dev)={len(dev)} rows(test)={len(test)} feats={len(feats)} seed={SEED}")
    print(f"[cfg] feature_mode={args.mode} extra_drop={extra}")
    # CV
    oof, imp_sum = cv_train(dev, feats)
    oof_y = dev[LABEL].astype(int).values
    # Train final on all dev, predict test
    params = dict(
        objective="binary",
        metric=["auc", "binary_logloss", "average_precision"],
        learning_rate=0.03,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        min_data_in_leaf=20,
        seed=SEED,
        verbose=-1,
        is_unbalance=True,
    )
    trn = lgb.Dataset(dev[feats], label=dev[LABEL].astype(int))
    booster = lgb.train(params, trn, num_boost_round=int(1.2 * 400) or 400)
    global test_ids
    test_ids = test[IDCOL].astype(str).values
    test_pred = booster.predict(test[feats])

    # Defaults (raw)
    use_oof = oof.copy()
    use_test = test_pred.copy()
    calibrated_used = False

    # --- Optional: isotonic calibration on OOF, then apply to test ---
    if args.calibrate:
        oof_y = dev[LABEL].astype(int).values
        oof_p = oof.astype(float)

        # Fit isotonic on OOF
        iso = IsotonicRegression(out_of_bounds="clip")
        oof_cal = iso.fit_transform(oof_p, oof_y)

        # Metrics before/after (DEV)
        auc_raw   = roc_auc_score(oof_y, oof_p)
        pr_raw    = average_precision_score(oof_y, oof_p)
        brier_raw = brier_score_loss(oof_y, oof_p)

        auc_cal   = roc_auc_score(oof_y, oof_cal)
        pr_cal    = average_precision_score(oof_y, oof_cal)
        brier_cal = brier_score_loss(oof_y, oof_cal)

        print(f"[CAL] DEV raw: AUC={auc_raw:.3f} PR-AUC={pr_raw:.3f} Brier={brier_raw:.4f}")
        print(f"[CAL] DEV iso: AUC={auc_cal:.3f} PR-AUC={pr_cal:.3f} Brier={brier_cal:.4f}")

        # Apply to test
        test_pred_cal = iso.transform(test_pred)

        # Save calibrated outputs
        cal_dir = Path("reports/model_calibrated")
        cal_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"user_id": dev[IDCOL].astype(str), "oof_cal": oof_cal, "label": oof_y}).to_csv(cal_dir / "oof_dev_iso.csv", index=False)
        pd.DataFrame({"user_id": test_ids, "pred_cal": test_pred_cal}).to_csv(cal_dir / "pred_test_iso.csv", index=False)
        joblib.dump(iso, cal_dir / "isotonic_calibrator.joblib")
        print(f"[CAL] wrote calibrated files to {cal_dir}")

        use_oof = oof_cal.copy()
        use_test = test_pred_cal.copy()
        calibrated_used = True

    # Finalization: thresholds + final CSV/figures
    oof_df_for_thr = pd.DataFrame({"label": oof_y, "oof": use_oof})
    thr_dict = _compute_thresholds(oof_df_for_thr)
    _write_final_outputs(test_ids, use_test, thr_dict, calibrated_used)

    # Save
    save_reports(dev, oof, imp_sum, feats, test_pred)

if __name__ == "__main__":
    main()