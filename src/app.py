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


# ---------- 自动打标 ----------
def auto_label_impl(input_csv: Path, out_csv: Path, t_high: float = 3.0, t_low: float = -1.0):
    """
    根据启发式信号做弱监督打标：
      s = 2*(pii_email>0) + 2*(pii_phone>0)
          + 1.5*norm_p95(sens_hit_rate)
          + 2.0*rule_score
          + 0.5*norm_p95(log1p(n_docs))
    阈值：s >= t_high -> y=1； s <= t_low -> y=0；其他样本丢弃
    """
    df = _read_csv(input_csv).copy()

    def p95_norm(x):
        x = pd.to_numeric(x, errors="coerce").fillna(0.0)
        p = np.percentile(x, 95) if len(x) else 1.0
        p = p if p > 1e-9 else 1.0
        return (x / p).clip(0, 1)

    id_signal = 2.0 * (
        (pd.to_numeric(df.get("pii_email_rate", 0), errors="coerce") > 0).astype(float)
    ) + 2.0 * ((pd.to_numeric(df.get("pii_phone_rate", 0), errors="coerce") > 0).astype(float))

    cont_signal = (
        1.5 * p95_norm(df.get("sens_hit_rate", 0))
        + 2.0 * pd.to_numeric(df.get("rule_score", 0), errors="coerce").fillna(0.0)
        + 0.5 * p95_norm(np.log1p(pd.to_numeric(df.get("n_docs", 0), errors="coerce").fillna(0.0)))
    )

    s = id_signal + cont_signal
    y = pd.Series(np.nan, index=df.index, dtype=float)
    y[s >= t_high] = 1.0
    y[s <= t_low] = 0.0

    keep_mask = ~y.isna()
    kept = df.loc[keep_mask].copy()
    kept["y"] = y.loc[keep_mask].astype(int)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_csv, index=False)

    pos = int((kept["y"] == 1).sum())
    neg = int((kept["y"] == 0).sum())
    print(f"[auto-label] 输出：{out_csv}  行数={len(kept)}（正={pos} / 负={neg}）")


# ---------- 训练 / 评估 ----------
def train_cv_impl(data_csv: Path, outdir: Path, folds: int = 5, random_state: int = 42):
    outdir.mkdir(parents=True, exist_ok=True)
    df = _read_csv(data_csv)

    feat_df, feat_cols = _prepare_features(df, columns=None)
    _save_feature_columns(outdir, feat_cols)

    X_all = feat_df.fillna(0.0).to_numpy(dtype=float)
    if "y" not in df.columns:
        raise KeyError("训练需要标签列 y，请先运行 auto-label 生成 train.csv。")
    y_all = df["y"].astype(int).to_numpy()

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    oof_pred = np.zeros_like(y_all, dtype=float)
    metrics = []

    for i, (tr, va) in enumerate(skf.split(X_all, y_all), 1):
        Xtr, Xva = X_all[tr], X_all[va]
        ytr, yva = y_all[tr], y_all[va]

        lgb_train = lgb.Dataset(Xtr, label=ytr)
        lgb_valid = lgb.Dataset(Xva, label=yva, reference=lgb_train)
        params = {
            "objective": "binary",
            "metric": ["auc"],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "seed": random_state,
            "verbose": -1,
        }
        evals_result: dict = {}
        booster = lgb.train(
            params,
            lgb_train,
            num_boost_round=800,
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
            n_estimators=min(booster.best_iteration or 400, 800),
            learning_rate=params["learning_rate"],
            max_depth=-1,
            subsample=params["bagging_fraction"],
            colsample_bytree=params["feature_fraction"],
            random_state=random_state,
        )
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        clf.fit(Xtr, ytr)

        proba = clf.predict_proba(Xva)[:, 1]
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

    pd.DataFrame(metrics).to_csv(outdir / "cv_metrics.csv", index=False)
    np.save(outdir / "oof_pred.npy", oof_pred)


def eval_on_test_impl(test_csv: Path, models_dir: Path):
    df = _read_csv(test_csv)
    train_cols = _load_feature_columns(models_dir)
    feat_df, _ = _prepare_features(df, columns=train_cols)

    X = feat_df.fillna(0.0)
    y = df["y"].astype(int).to_numpy() if "y" in df.columns else None

    model_paths = sorted(models_dir.glob("model_fold*.joblib")) or list(models_dir.glob("*.joblib"))
    if not model_paths:
        raise FileNotFoundError(f"未在 {models_dir} 找到模型文件（*.joblib）")

    preds = []
    for p in model_paths:
        clf = load(p)
        preds.append(clf.predict_proba(X)[:, 1])
    proba = np.mean(preds, axis=0)

    if y is not None:
        yhat = (proba >= 0.5).astype(int)
        print(classification_report(y, yhat, digits=4))
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
                preds.append(clf.predict_proba(df_feat.to_numpy(dtype=float))[:, 1][0])
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
    pal.add_argument("--t-high", type=float, default=3.0)
    pal.add_argument("--t-low", type=float, default=-1.0)

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

    pe = sub.add_parser("eval", help="Evaluate on test set")
    pe.add_argument("--data", required=True)
    pe.add_argument("--models", required=True)

    args = p.parse_args()
    cmd = args.cmd

    if cmd == "auto-label":
        auto_label_impl(_P(args.input), _P(args.out), t_high=args.t_high, t_low=args.t_low)

    elif cmd == "featurize":
        from src.features.build import build_features

        build_features(_P(args.input), _P(args.out))

    elif cmd == "split":
        from src.dataset.split import split_labeled

        split_labeled(
            _P(args.input),
            _P(args.outdir),
            test_size=args.test_size,
            random_state=args.random_state,
        )

    elif cmd == "train-cv":
        train_cv_impl(
            _P(args.data), _P(args.outdir), folds=args.folds, random_state=args.random_state
        )

    elif cmd == "eval":
        eval_on_test_impl(_P(args.data), _P(args.models))

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
