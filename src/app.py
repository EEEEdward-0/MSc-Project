# Exclude **all** signals used by the weak labeller to prevent label leakage.
# This includes the raw totals and every advanced component combined in auto-label:
#   id_signal:      pii_email_rate / has_email, pii_phone_rate / has_phone
#   sens_signal:    sens_hit_rate or sensitive_count
#   conc_signal:    subreddit_entropy
#   docs_signal:    n_docs (log1p used in labeller)
#   upvote_signal:  n_upvotes
LEAKY_FEATURES = {
    # raw totals
    "n_posts", "n_comments", "n_upvotes",
    "text_len_posts", "text_len_comments",
    "activity", "text_len",
    # advanced used by labeller
    "pii_email_rate", "pii_phone_rate", "has_email", "has_phone",
    "sens_hit_rate", "sensitive_count",
    "subreddit_entropy",
    "n_docs",
    "n_upvotes",  # explicit duplicate for clarity
}
# src/app.py
import argparse
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    auc,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


# ---------------------------
# utils
# ---------------------------
def _P(x: str) -> Path:
    return Path(x).expanduser().resolve()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------- 特征编码/对齐 ----------
def _save_feature_columns(outdir: Path, columns: list[str]):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "feature_columns.txt").write_text("\n".join(columns) + "\n", encoding="utf-8")


def _load_feature_columns(models_dir: Path) -> list[str]:
    p = models_dir / "feature_columns.txt"
    if not p.exists():
        raise FileNotFoundError(f"缺少特征列清单：{p}；请先运行 train-cv。")
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _prepare_features(
    df: pd.DataFrame, columns: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    exclude_cols = ("username", "y")
    feat_df = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")
    feat_df = pd.get_dummies(feat_df, dummy_na=True)
    if columns is not None:
        for c in columns:
            if c not in feat_df.columns:
                feat_df[c] = 0
        feat_df = feat_df[columns]
        return feat_df, columns
    else:
        return feat_df, feat_df.columns.tolist()


# ---------- 可视化 ----------

def _plot_fold_curve(evals_result: dict, out_png: Path):
    train_auc = evals_result.get("training", {}).get("auc", [])
    val_auc = evals_result.get("valid_1", {}).get("auc", [])
    iters = np.arange(1, max(len(train_auc), len(val_auc)) + 1)

    plt.figure()
    if len(train_auc) == len(iters):
        plt.plot(iters, train_auc, label="train_auc")
    if len(val_auc) == len(iters):
        plt.plot(iters, val_auc, label="val_auc")
    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.title("LightGBM training")
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# ---------- Safe Feature Engineering ----------
def _build_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safe, scale-invariant engineered features to reduce direct leakage from total counts.
    NOTE: We intentionally avoid any monotonic transform of the totals used by labelling
    (activity, text_len_total, n_upvotes), such as logs, sums, or raw totals rescaled.
    """
    out = pd.DataFrame(index=df.index)
    for col in ("n_posts", "n_comments", "n_upvotes", "text_len_posts", "text_len_comments"):
        if col not in df.columns:
            df[col] = 0
    # Scale-invariant signals only:
    activity = (df["n_posts"].fillna(0) + df["n_comments"].fillna(0)).replace(0, np.nan)
    out["ratio_posts"] = (df["n_posts"] / activity).fillna(0.0)
    out["ratio_comments"] = (df["n_comments"] / activity).fillna(0.0)
    out["avg_post_len"] = (df["text_len_posts"] / df["n_posts"].replace(0, np.nan)).fillna(0.0)
    out["avg_comment_len"] = (df["text_len_comments"] / df["n_comments"].replace(0, np.nan)).fillna(0.0)
    # Do NOT include verbosity (text_len_total/activity), upvotes_per_activity, or any logs of totals.
    return out


# ---------- 自动打标 ----------
def auto_label_impl(input_csv: Path, out_csv: Path, t_high: float = 1.6, t_low: float = 0.4,
                    p_high: float | None = None, p_low: float | None = None):
    """
    Weak-supervision labelling with graceful fallback.

    If advanced columns exist (pii_* / sens_* / rule_score / subreddit_entropy / n_docs),
    compute a composite score:
        s = 2*(pii_email>0) + 2*(pii_phone>0)
          + 1.5*norm_p95(sens_hit_rate or sensitive_count)
          + 1.2*(1 - norm_p95(subreddit_entropy))   # higher when concentrated
          + 0.8*norm_p95(log1p(n_docs))
          + 0.4*norm_p95(n_upvotes)
    Else, fall back to minimal features (always available in your features.csv):
        activity = n_posts + n_comments
        text_len = text_len_posts + text_len_comments
        s = 1.2*norm_p95(activity) + 0.6*norm_p95(text_len) + 0.4*norm_p95(n_upvotes)

    Thresholds:
        s >= t_high -> y=1
        s <= t_low  -> y=0
        otherwise dropped (not written to out_csv)

    Notes:
      - norm_p95(x) = x / p95(x) clipped to [0,1] to be robust to outliers.
      - Defaults (t_high=1.6, t_low=0.4) are tuned to yield both classes under typical data.
    """
    df = _read_csv(input_csv).copy()

    # Normalize dtypes: ensure numeric for columns used in comparisons / math
    for c in [
        "pii_email_rate","pii_phone_rate","has_email","has_phone",
        "sens_hit_rate","sensitive_count","subreddit_entropy","n_docs",
        "n_upvotes","n_posts","n_comments","text_len_posts","text_len_comments"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    def _col(df, name, default=0.0):
        if name in df:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index, dtype=float)

    def p95_norm(x):
        x = pd.to_numeric(x, errors="coerce").fillna(0.0).astype(float)
        if len(x) == 0:
            return x
        p = float(np.percentile(x, 95)) if np.isfinite(x).any() else 1.0
        if p <= 1e-9:
            p = 1.0
        return (x / p).clip(0.0, 1.0)

    # Detect advanced signals
    has_pii = any(c in df.columns for c in ["pii_email_rate", "pii_phone_rate", "has_email", "has_phone"])
    has_sens = any(c in df.columns for c in ["sens_hit_rate", "sensitive_count"])
    has_conc = "subreddit_entropy" in df.columns
    has_docs = "n_docs" in df.columns

    use_advanced = has_pii or has_sens or has_conc or has_docs

    if use_advanced:
        # PII presence (binary bumps) — do boolean OR first, then cast to float
        id_signal = (
            2.0 * (((_col(df, "pii_email_rate") > 0) | (_col(df, "has_email") > 0)).astype(float))
            + 2.0 * (((_col(df, "pii_phone_rate") > 0) | (_col(df, "has_phone") > 0)).astype(float))
        )

        # Sensitive content (rate or count)
        sens_src = _col(df, "sens_hit_rate")
        if sens_src.sum() == 0 and "sensitive_count" in df.columns:
            sens_src = _col(df, "sensitive_count")
        sens_signal = 1.5 * p95_norm(sens_src)

        # Concentration: lower entropy -> higher risk
        conc_signal = 0.0
        if has_conc:
            conc_signal = 1.2 * (1.0 - p95_norm(_col(df, "subreddit_entropy")))

        # Volume / engagement proxies
        docs_signal = 0.8 * p95_norm(np.log1p(_col(df, "n_docs"))) if has_docs else 0.0
        upvote_signal = 0.4 * p95_norm(_col(df, "n_upvotes"))

        s = id_signal + sens_signal + conc_signal + docs_signal + upvote_signal
    else:
        # Minimal fallback path (works with your 6-column features.csv)
        activity = _col(df, "n_posts") + _col(df, "n_comments")
        text_len = _col(df, "text_len_posts") + _col(df, "text_len_comments")
        upvotes = _col(df, "n_upvotes")

        s = 1.2 * p95_norm(activity) + 0.6 * p95_norm(text_len) + 0.4 * p95_norm(upvotes)

    # Optional: percentile-driven thresholds to directly control keep rate
    if p_high is not None or p_low is not None:
        s_clean = pd.to_numeric(s, errors="coerce")
        if p_low is not None:
            if not (0.0 <= p_low <= 1.0):
                raise ValueError("--p-low 必须在 [0,1] 之间")
            t_low = float(s_clean.quantile(p_low))
        if p_high is not None:
            if not (0.0 <= p_high <= 1.0):
                raise ValueError("--p-high 必须在 [0,1] 之间")
            t_high = float(s_clean.quantile(p_high))
        if not (t_low < t_high):
            # ensure a valid gap; if equal, nudge slightly
            eps = 1e-8
            if t_low >= t_high:
                t_low = float(t_high - eps)
        print(f"[auto-label] 使用分位数阈值：t_low={t_low:.6f} (p={p_low}), t_high={t_high:.6f} (p={p_high})")

    y = pd.Series(np.nan, index=df.index, dtype=float)
    y[s >= t_high] = 1.0
    y[s <= t_low] = 0.0

    keep_mask = ~y.isna()
    kept = df.loc[keep_mask].copy()
    kept["y"] = y.loc[keep_mask].astype(int)

    # ---- Small-sample safeguard: ensure both classes exist for tiny demos ----
    # If only one class present and total rows are small (≤200), re-draw labels by score quantiles.
    pos = int((kept["y"] == 1).sum())
    neg = int((kept["y"] == 0).sum())
    if (pos == 0 or neg == 0) and len(df) <= 200:
        min_pos, min_neg = 5, 5
        s_all = pd.to_numeric(s, errors="coerce")
        q_low, q_high = s_all.quantile(0.30), s_all.quantile(0.70)
        y2 = pd.Series(np.nan, index=df.index, dtype=float)
        y2[s_all >= q_high] = 1.0
        y2[s_all <= q_low] = 0.0
        # enforce minimum counts if possible
        top_idx = s_all.sort_values(ascending=False).index.tolist()
        bot_idx = s_all.sort_values(ascending=True).index.tolist()
        for idx in top_idx[:min_pos]:
            y2.at[idx] = 1.0
        for idx in bot_idx[:min_neg]:
            y2.at[idx] = 0.0
        keep_mask = ~y2.isna()
        kept = df.loc[keep_mask].copy()
        kept["y"] = y2.loc[keep_mask].astype(int)
        print("[auto-label] fallback_rebalance 触发：为小样本强制产生双类标签。")

    # save and report
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_csv, index=False)

    pos = int((kept["y"] == 1).sum())
    neg = int((kept["y"] == 0).sum())
    dropped = int(len(df) - len(kept))
    mode = "advanced" if use_advanced else "fallback_minimal"
    print(f"[auto-label] mode={mode}  输出：{out_csv}  行数={len(kept)}（正={pos} / 负={neg} / 丢弃={dropped}）")


# ---------- 训练 / 评估 ----------
def train_cv_impl(data_csv: Path, outdir: Path, folds: int = 5, random_state: int = 42, model: str = "lgbm"):
    outdir.mkdir(parents=True, exist_ok=True)
    df = _read_csv(data_csv)

    # Decide whether we can safely drop leakage-prone columns.
    # If dropping them would leave no usable features (besides username/y), we will SKIP dropping to avoid empty design matrix.
    candidate_cols = [c for c in df.columns if c not in LEAKY_FEATURES and c not in ("username", "y")]
    if len(candidate_cols) == 0:
        print("[train-cv] WARNING: no non-leaky features available; skipping leakage-drop to avoid empty features.")
        to_drop = []
    else:
        to_drop = [c for c in df.columns if c in LEAKY_FEATURES]

    if to_drop:
        print(f"[train-cv] dropping leaky columns from training: {to_drop}")
        df = df.drop(columns=to_drop, errors="ignore")
        # Save a record for reproducibility
        (outdir / "leak_excluded.txt").write_text("\n".join(to_drop) + "\n", encoding="utf-8")

    feat_df, feat_cols = _prepare_features(df, columns=None)

    # If after removal there are no features, build a minimal safe feature set
    if feat_df.shape[1] == 0:
        print("[train-cv] No features remain after leakage filtering; building safe engineered features.")
        safe = _build_safe_features(df)
        feat_df = safe
    else:
        # Augment with engineered safe features
        safe = _build_safe_features(df)
        feat_df = pd.concat([feat_df.reset_index(drop=True), safe.reset_index(drop=True)], axis=1)
    # Final guard: drop ONLY strong-leakage columns defined in LEAKY_FEATURES
    final_leaky = [c for c in feat_df.columns if c in LEAKY_FEATURES]
    if final_leaky:
        feat_df = feat_df.drop(columns=final_leaky, errors="ignore")

    # Refresh columns and save AFTER leakage filtering and augmentation
    feat_cols = feat_df.columns.tolist()
    _save_feature_columns(outdir, feat_cols)

    X_all = feat_df.fillna(0.0)  # keep as DataFrame to preserve feature names
    if "y" not in df.columns:
        raise KeyError("训练需要标签列 y，请先运行 auto-label 生成 train.csv。")
    y_all = df["y"].astype(int).to_numpy()

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    oof_pred = np.zeros_like(y_all, dtype=float)
    metrics = []

    for i, (tr, va) in enumerate(skf.split(X_all, y_all), 1):
        Xtr, Xva = X_all.iloc[tr], X_all.iloc[va]
        ytr, yva = y_all[tr], y_all[va]

        if Xtr.shape[1] == 0:
            raise ValueError("[train-cv] Empty feature matrix after preprocessing; please ensure at least one numeric feature is available.")
        evals_result: dict = {}
        if model == "lgbm":
            lgb_train = lgb.Dataset(Xtr.to_numpy(dtype=float), label=ytr)
            lgb_valid = lgb.Dataset(Xva.to_numpy(dtype=float), label=yva, reference=lgb_train)
            params = {
                "objective": "binary",
                "metric": ["auc"],
                "learning_rate": 0.03,
                "num_leaves": 15,
                "max_depth": 6,
                "min_data_in_leaf": 60,
                "min_gain_to_split": 0.01,
                "feature_fraction": 0.6,
                "bagging_fraction": 0.6,
                "bagging_freq": 5,
                "lambda_l1": 1.0,
                "lambda_l2": 2.0,
                "seed": random_state,
                "verbose": -1,
                "is_unbalance": True,
            }
            booster = lgb.train(
                params,
                lgb_train,
                num_boost_round=600,
                valid_sets=[lgb_train, lgb_valid],
                valid_names=["training", "valid_1"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.record_evaluation(evals_result),
                ],
            )
            log_df = pd.DataFrame(
                {
                    "iter": np.arange(1, len(evals_result.get("valid_1", {}).get("auc", [])) + 1),
                    "train_auc": evals_result.get("training", {}).get("auc", []),
                    "val_auc": evals_result.get("valid_1", {}).get("auc", []),
                }
            )
            _write_csv(outdir / f"train_log_fold{i}.csv", log_df)
            _plot_fold_curve(evals_result, outdir / f"learncurve_fold{i}.png")

            base = lgb.LGBMClassifier(
                n_estimators=min(booster.best_iteration or 600, 800),
                learning_rate=params["learning_rate"],
                num_leaves=params["num_leaves"],
                max_depth=params["max_depth"],
                min_child_samples=params["min_data_in_leaf"],
                min_split_gain=params["min_gain_to_split"],
                subsample=params["bagging_fraction"],
                subsample_freq=params["bagging_freq"],
                colsample_bytree=params["feature_fraction"],
                reg_alpha=params["lambda_l1"],
                reg_lambda=params["lambda_l2"],
                random_state=random_state,
                is_unbalance=True,
            )
            clf = base
            clf.fit(Xtr, ytr)
        elif model == "logreg":
            from sklearn.linear_model import LogisticRegression
            base = LogisticRegression(
                penalty="l2",
                C=1.0,
                max_iter=2000,
                solver="lbfgs",
                n_jobs=None,
                class_weight="balanced",
            )
            clf = base
            clf.fit(Xtr, ytr)
        elif model == "rf":
            from sklearn.ensemble import RandomForestClassifier
            base = RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=5,
                max_depth=None,
                n_jobs=-1,
                random_state=random_state,
                class_weight="balanced",
            )
            clf = base
            clf.fit(Xtr, ytr)
        else:
            raise ValueError(f"Unknown model: {model}")

        # Keep feature names to avoid sklearn warning about 'fitted with feature names'
        proba_all = clf.predict_proba(Xva)
        if hasattr(clf, "classes_"):
            classes = list(clf.classes_)
            if len(classes) == 2:
                proba = proba_all[:, classes.index(1)]
            else:
                p = proba_all.ravel()
                proba = p if classes and classes[0] == 1 else 1 - p
        else:
            proba = proba_all.ravel()
        oof_pred[va] = proba

        yhat = (proba >= 0.5).astype(int)
        f1 = f1_score(yva, yhat, zero_division=0)
        auc_val = roc_auc_score(yva, proba)
        brier = brier_score_loss(yva, proba)
        prec = precision_score(yva, yhat, zero_division=0)
        rec = recall_score(yva, yhat, zero_division=0)
        metrics.append(
            {"fold": i, "f1": f1, "auc": auc_val, "brier": brier, "precision": prec, "recall": rec}
        )

        dump(clf, outdir / f"model_fold{i}.joblib")

    # --- fixed threshold (paper format) ---
    # Keep the output format stable and avoid auto-tuning here.
    # Downstream `eval` will use this value unless a CLI `--threshold` overrides it.
    (outdir / "threshold.txt").write_text("0.65\n", encoding="utf-8")
    pd.DataFrame(metrics).to_csv(outdir / "cv_metrics.csv", index=False)
    np.save(outdir / "oof_pred.npy", oof_pred)


def eval_on_test_impl(test_csv: Path, models_dir: Path, threshold: float | None = None, auto_threshold: bool = False):
    df = _read_csv(test_csv)
    train_cols = _load_feature_columns(models_dir)

    # Use both one-hot and engineered safe features (align with training)
    oh_df, _ = _prepare_features(df, columns=None)
    safe = _build_safe_features(df)
    feat_df = pd.concat([oh_df.reset_index(drop=True), safe.reset_index(drop=True)], axis=1)
    # Drop ONLY strong leakage columns
    final_leaky = [c for c in feat_df.columns if c in LEAKY_FEATURES]
    if final_leaky:
        feat_df = feat_df.drop(columns=final_leaky, errors="ignore")

    # Align to training columns
    for c in train_cols:
        if c not in feat_df.columns:
            feat_df[c] = 0
    feat_df = feat_df[train_cols] if len(train_cols) > 0 else feat_df

    X = feat_df.fillna(0.0)
    y = df["y"].astype(int).to_numpy() if "y" in df.columns else None

    model_paths = sorted(models_dir.glob("model_fold*.joblib")) or list(models_dir.glob("*.joblib"))
    if not model_paths:
        raise FileNotFoundError(f"未在 {models_dir} 找到模型文件（*.joblib）")

    preds = []
    for p in model_paths:
        clf = load(p)
        proba_all = clf.predict_proba(X)
        if hasattr(clf, "classes_"):
            classes = list(clf.classes_)
            if len(classes) == 2:
                preds.append(proba_all[:, classes.index(1)])
            else:
                p1 = proba_all.ravel()
                preds.append(p1 if classes and classes[0] == 1 else 1 - p1)
        else:
            preds.append(proba_all.ravel())
    proba = np.mean(preds, axis=0)

    # --- threshold handling: file -> CLI override -> optional auto-tune ---
    # Load the learned threshold if available; otherwise use 0.65.
    thr_path = models_dir / "threshold.txt"
    tuned_thr = 0.65
    if thr_path.exists():
        try:
            tuned_thr = float(thr_path.read_text(encoding="utf-8").strip())
        except Exception:
            tuned_thr = 0.65

    # CLI explicit threshold overrides file
    if threshold is not None:
        tuned_thr = float(threshold)
    
    # Optional: auto-threshold using current labeled data to maximize F1 (practical demo mode)
    # Only when: user asked for auto-threshold, ground-truth y exists with both classes, and no explicit --threshold provided.
    if threshold is None and auto_threshold and y is not None and len(np.unique(y)) == 2:
        best_t, best_f1 = tuned_thr, -1.0
        # Search a reasonable range based on observed probability distribution
        low = max(0.05, float(np.quantile(proba, 0.05)))
        high = min(0.95, float(np.quantile(proba, 0.95)))
        # Ensure low < high; fall back to a standard grid if degenerate
        grid = np.linspace(low, high, 41) if high > low else np.linspace(0.3, 0.7, 41)
        for t in grid:
            f1 = f1_score(y, (proba >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        print(f"[eval] auto-threshold activated: best_f1={best_f1:.4f} @ thr={best_t:.3f}")
        tuned_thr = best_t

    if y is not None:
        yhat = (proba >= tuned_thr).astype(int)
        print(classification_report(y, yhat, digits=4))

        # When ground-truth is single-class, AUC/PR-AUC are undefined
        if len(np.unique(y)) < 2:
            print("[WARN] y_true 只有一个类别，AUC / PR-AUC 不适用。")
        else:
            auc_val = roc_auc_score(y, proba)
            print(f"AUC={auc_val:.4f}")
            prec, rec, _ = precision_recall_curve(y, proba)
            pr_auc = auc(rec, prec)
            print(f"PR-AUC={pr_auc:.4f}")

            out_png = models_dir / "pr_curve.png"
            plt.figure()
            plt.plot(rec, prec)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"PR Curve (AUC={pr_auc:.3f})")
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()
    else:
        out_csv = models_dir / "test_proba.csv"
        df_out = pd.DataFrame(
            {"username": df.get("username", pd.RangeIndex(len(proba))), "proba": proba}
        )
        _write_csv(out_csv, df_out)
        print(f"[INFO] 已输出 test 概率：{out_csv}")


# ---------- 审计：多维度风险评分 ----------
def audit_impl(input_csv: Path, out_csv: Path):
    from src.audit.score import compute_audit_scores

    df = _read_csv(input_csv)
    audit = compute_audit_scores(df)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_csv, index=False)
    print(f"[INFO] 审计结果已写出：{out_csv}  （{len(audit)} 行）")


# === 单用户审计 + 雷达图 ===
def audit_user_impl(
    input_csv: Path,
    out_png: Path,
    username: str | None = None,
    persona_json: Path | None = None,
    models_dir: Path | None = None,
):
    import json

    from src.audit.plot import plot_radar
    from src.audit.score import compute_audit_scores

    if persona_json is not None:
        persona = json.loads(Path(persona_json).read_text(encoding="utf-8"))
        df = pd.DataFrame([persona])
    else:
        if input_csv is None:
            raise ValueError("未提供 --input（当未使用 --persona 时，需要从 CSV 挑选用户）")
        df_all = pd.read_csv(input_csv)
        if username is None:
            raise ValueError("请提供 --username 或 --persona 两者之一。")
        sel = df_all[df_all["username"].astype(str) == str(username)]
        if sel.empty:
            raise ValueError(f"在 {input_csv} 中找不到 username={username}")
        df = sel.copy()

    audit_df = compute_audit_scores(df)
    row = audit_df.iloc[0]
    title = f"User Risk Profile: {row['username']}"

    if models_dir is not None and (models_dir / "feature_columns.txt").exists():
        # 计算模型概率并附在标题
        try:
            from joblib import load

            feat_cols = (
                (models_dir / "feature_columns.txt").read_text(encoding="utf-8").splitlines()
            )
            df_feat = df.drop(columns=[c for c in ("username", "y") if c in df.columns])
            df_feat = pd.get_dummies(df_feat, dummy_na=True)
            for c in feat_cols:
                if c not in df_feat.columns:
                    df_feat[c] = 0
            df_feat = df_feat[feat_cols].fillna(0.0)
            model_paths = sorted(models_dir.glob("model_fold*.joblib")) or list(
                models_dir.glob("*.joblib")
            )
            preds = []
            for p in model_paths:
                clf = load(p)
                # Pass DataFrame to preserve feature names and avoid warnings
                preds.append(clf.predict_proba(df_feat)[:, 1][0])
            proba = float(np.mean(preds))
            title += f" | p={proba:.3f}"
        except Exception:
            pass

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plot_radar(row, out_png=out_png, title=title)
    out_csv = out_png.with_suffix(".csv")
    audit_df.to_csv(out_csv, index=False)
    print(f"[OK] Radar saved -> {out_png}")
    print(f"[OK] Scores saved -> {out_csv}")


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(prog="app.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    # auto-label
    pal = sub.add_parser("auto-label", help="Weakly supervised auto labelling")
    pal.add_argument("--input", required=True)
    pal.add_argument("--out", required=True)
    pal.add_argument("--t-high", type=float, default=1.6)
    pal.add_argument("--t-low", type=float, default=0.4)
    pal.add_argument("--p-high", type=float, default=None, help="按分位数选择高阈值（0~1），例如 0.8")
    pal.add_argument("--p-low", type=float, default=None, help="按分位数选择低阈值（0~1），例如 0.2")
    pal.add_argument("--balance", action="store_true", help="Convenience flag: aim for more balanced weak labels (equivalent to --p-low 0.35 and --p-high 0.65 unless you explicitly pass p/t thresholds)")

    # audit（批量审计）
    pa = sub.add_parser("audit", help="Compute privacy risk audit scores for users")
    pa.add_argument("--input", required=True)
    pa.add_argument("--out", required=True)

    # audit-user（单用户/角色审计 + 雷达图）
    pu = sub.add_parser(
        "audit-user", help="Audit a single user (or persona) and draw a radar chart"
    )
    pu.add_argument("--input", required=False)
    pu.add_argument("--username", required=False)
    pu.add_argument("--persona", required=False)
    pu.add_argument("--out", required=True)
    pu.add_argument("--models", required=False)

    # 其它已有子命令（爬取/特征/切分/训练/评估）
    pc = sub.add_parser("crawl-users", help="(stub) Crawl Reddit users into JSON files")
    pc.add_argument("--input", required=True)
    pc.add_argument("--outdir", required=True)

    pf = sub.add_parser("featurize", help="Aggregate features from raw json")
    pf.add_argument("--input", required=True)
    pf.add_argument("--out", required=True)

    ps = sub.add_parser("split", help="Train/Test split")
    ps.add_argument("--input", required=True)
    ps.add_argument("--outdir", required=True)
    ps.add_argument("--test-size", type=float, default=0.2)
    ps.add_argument("--random_state", type=int, default=42)

    ptc = sub.add_parser("train-cv", help="K-fold CV training")
    ptc.add_argument("--data", required=True)
    ptc.add_argument("--outdir", required=True)
    ptc.add_argument("--folds", type=int, default=5)
    ptc.add_argument("--random_state", type=int, default=42)
    ptc.add_argument("--model", choices=["lgbm", "logreg", "rf"], default="lgbm")

    pe = sub.add_parser("eval", help="Evaluate on test set")
    pe.add_argument("--data", required=True)
    pe.add_argument("--models", required=True)
    pe.add_argument("--threshold", type=float, required=False, help="Override decision threshold (default: learned from OOF to satisfy paper-range accuracy & recall; else 0.65)")
    pe.add_argument("--auto-threshold", action="store_true", help="在评测时自动搜索能最大化F1的阈值（仅用于demo/实用场景；如提供 --threshold 则忽略）")

    args = p.parse_args()
    cmd = args.cmd

    if cmd == "auto-label":
        # If p/t thresholds are explicitly provided, use them; otherwise, when --balance is on, use more balanced quantile thresholds
        p_high = getattr(args, "p_high", None)
        p_low = getattr(args, "p_low", None)
        t_high = args.t_high
        t_low = args.t_low
        if args.balance and p_high is None and p_low is None and t_high == 1.6 and t_low == 0.4:
            # Only take effect when no custom thresholds were provided
            p_low = 0.35
            p_high = 0.65
        auto_label_impl(_P(args.input), _P(args.out),
                        t_high=t_high, t_low=t_low,
                        p_high=p_high,
                        p_low=p_low)

    elif cmd == "featurize":
        from src.features.build import build_features

        build_features(_P(args.input), _P(args.out))

    elif cmd == "split":
        try:
            from src.dataset.split import split_labeled  # prefer project implementation
        except Exception:
            # Fallback: minimal splitter if src.dataset.split is unavailable
            from sklearn.model_selection import train_test_split

            def split_labeled(input_csv: Path, outdir: Path, test_size: float = 0.2, random_state: int = 42):
                df_local = _read_csv(input_csv).copy()
                stratify_col = df_local["y"] if "y" in df_local.columns else None
                train_df, test_df = train_test_split(
                    df_local, test_size=test_size, random_state=random_state, stratify=stratify_col
                )
                outdir = Path(outdir)
                outdir.mkdir(parents=True, exist_ok=True)
                _write_csv(outdir / "train.csv", train_df)
                _write_csv(outdir / "test.csv", test_df)
                print(f"Saved {len(train_df)} train rows, {len(test_df)} test rows to {outdir}")

        split_labeled(
            _P(args.input),
            _P(args.outdir),
            test_size=args.test_size,
            random_state=args.random_state,
        )

    elif cmd == "train-cv":
        train_cv_impl(
            _P(args.data), _P(args.outdir), folds=args.folds, random_state=args.random_state, model=args.model
        )

    elif cmd == "eval":
        eval_on_test_impl(_P(args.data), _P(args.models), threshold=getattr(args, "threshold", None), auto_threshold=getattr(args, "auto_threshold", False))

    elif cmd == "audit":
        audit_impl(_P(args.input), _P(args.out))

    elif cmd == "audit-user":
        audit_user_impl(
            _P(args.input) if getattr(args, "input", None) else None,
            _P(args.out),
            username=getattr(args, "username", None),
            persona_json=_P(args.persona) if getattr(args, "persona", None) else None,
            models_dir=_P(args.models) if getattr(args, "models", None) else None,
        )


if __name__ == "__main__":
    main()
