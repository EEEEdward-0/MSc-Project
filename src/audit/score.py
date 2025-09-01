# src/audit/score.py
import numpy as np
import pandas as pd

RISK_COLS = [
    "identity_risk",
    "sensitive_risk",
    "exposure_risk",
    "volume_risk",
    "activity_risk",
    "concentration_risk",
]


# ---------- helpers ----------
def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        s = pd.to_numeric(df[name], errors="coerce")
    else:
        s = pd.Series(default, index=df.index, dtype=float)
    return s.fillna(default).astype(float)


def _robust_norm(s: pd.Series, lo_q=0.01, hi_q=0.99) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    if len(s) == 0:
        return s
    lo = np.quantile(s, lo_q)
    hi = np.quantile(s, hi_q)
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = 1.0
    if hi <= lo + 1e-12:
        # 退化列：全常数 -> 归一化为 0
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return ((s - lo) / (hi - lo)).clip(0.0, 1.0)


def _span_days(first_ts: pd.Series, last_ts: pd.Series) -> pd.Series:
    """根据 first_active / last_active（epoch 秒）估计时间跨度（天）。"""
    f = pd.to_numeric(first_ts, errors="coerce").fillna(0.0)
    l = pd.to_numeric(last_ts, errors="coerce").fillna(0.0)
    span = (l - f) / 86400.0
    span = span.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    span = span.clip(lower=0.0)
    # 至少 1 天，避免除 0
    return span.where(span >= 1.0, 1.0)


def _nontrivial(*cols: pd.Series) -> bool:
    """是否存在至少一个有非零和的列"""
    return any((pd.to_numeric(c, errors="coerce").fillna(0.0).abs().sum() > 0) for c in cols)


# ---------- main ----------
def compute_audit_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["username"] = (
        df["username"].astype(str)
        if "username" in df.columns
        else [f"user_{i}" for i in range(len(df))]
    )

    # 基础计数
    n_posts = _col(df, "n_posts")
    n_comments = _col(df, "n_comments")
    n_docs = _col(df, "n_docs")
    active_days = _col(df, "active_days")
    activity_rate = _col(df, "activity_rate")
    night_frac = _col(df, "night_frac")
    text_len_posts = _col(df, "text_len_posts")
    text_len_comments = _col(df, "text_len_comments")

    # 分布/多样性
    sub_entropy = _col(df, "sub_entropy")
    n_subs = _col(df, "n_subs")
    hour_entropy = _col(df, "hour_entropy")
    dow_entropy = _col(df, "dow_entropy")

    # 时间戳（用于估计跨度）
    first_active = _col(df, "first_active")
    last_active = _col(df, "last_active")
    span = _span_days(first_active, last_active)

    # PII / 敏感词
    pii_email = _col(df, "pii_email_rate")
    pii_phone = _col(df, "pii_phone_rate")
    sens_hit = _col(df, "sens_hit_rate")

    # 若输入里已有风险列，做融合
    exposure_in = _col(df, "exposure_risk")
    volume_in = _col(df, "volume_risk")
    activity_in = _col(df, "activity_risk")
    concentration_in = _col(df, "concentration_risk")

    # ===== volume_risk =====
    vol_events = n_posts + n_comments
    vol_text = text_len_posts + text_len_comments
    vol_a = _robust_norm(np.log1p(vol_events))
    vol_b = _robust_norm(np.log1p(vol_text))
    # 没有 n_docs 时用 vol_events 代理
    base_docs = n_docs.where(n_docs > 0, vol_events)
    vol_c = _robust_norm(np.log1p(base_docs))
    volume_risk = 0.5 * vol_a + 0.3 * vol_b + 0.2 * vol_c
    if volume_in.abs().sum() > 0:
        volume_risk = 0.7 * volume_risk + 0.3 * _robust_norm(volume_in)
    out["volume_risk"] = volume_risk.clip(0, 1)

    # ===== activity_risk =====
    # 通过跨度估计强度：docs/day
    docs_per_day = (base_docs / span).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    act_a = _robust_norm(activity_rate)  # 若没有则为 0
    act_b = _robust_norm(docs_per_day)  # 跨度代理
    act_c = _robust_norm(night_frac)  # 夜间占比
    act_d = _robust_norm(hour_entropy)  # 小时分布
    activity_risk = 0.35 * act_a + 0.35 * act_b + 0.15 * act_c + 0.15 * act_d

    # 如果四个信号都退化（全 0），用 volume 作为弱代理，给一个下限，避免恒 0
    if not _nontrivial(act_a, act_b, act_c, act_d):
        activity_risk = np.maximum(activity_risk, 0.25 * out["volume_risk"])

    if activity_in.abs().sum() > 0:
        activity_risk = 0.7 * activity_risk + 0.3 * _robust_norm(activity_in)
    out["activity_risk"] = activity_risk.clip(0, 1)

    # ===== concentration_risk =====
    # 默认用 1 - normalized entropy，但当所有信号缺失时，给“中性”0.5，而不是 1.0
    conc_sub = 1.0 - _robust_norm(sub_entropy)
    conc_dow = 1.0 - _robust_norm(dow_entropy)
    conc_nsubs = 1.0 - _robust_norm(n_subs)  # 子版越少越集中

    concentration_risk = 0.5 * conc_sub + 0.25 * conc_dow + 0.25 * conc_nsubs
    if not _nontrivial(sub_entropy, dow_entropy, n_subs):
        concentration_risk = pd.Series(0.5, index=df.index, dtype=float)  # 中性

    if concentration_in.abs().sum() > 0:
        concentration_risk = 0.7 * concentration_risk + 0.3 * _robust_norm(concentration_in)
    out["concentration_risk"] = concentration_risk.clip(0, 1)

    # ===== exposure_risk =====
    exp_a = _robust_norm(np.log1p(base_docs))
    exp_b = _robust_norm(sub_entropy)  # 多样性高 → 曝光广
    exp_c = out["volume_risk"]
    exposure_risk = 0.4 * exp_a + 0.3 * exp_b + 0.3 * exp_c
    # 没有 sub_entropy 时，降权 b 分量
    if sub_entropy.abs().sum() == 0:
        exposure_risk = 0.5 * exp_a + 0.5 * exp_c
    if exposure_in.abs().sum() > 0:
        exposure_risk = 0.7 * exposure_risk + 0.3 * _robust_norm(exposure_in)
    out["exposure_risk"] = exposure_risk.clip(0, 1)

    # ===== identity_risk =====
    ident_raw = 0.6 * _robust_norm(pii_email) + 0.4 * _robust_norm(pii_phone)
    # 没有 PII 时，用曝光做弱代理，给一点基础分
    ident = np.where((pii_email + pii_phone) > 0, ident_raw, 0.15 * out["exposure_risk"])
    out["identity_risk"] = np.clip(ident, 0, 1)

    # ===== sensitive_risk =====
    # 没有敏感命中时，用 活跃 + 曝光 的弱代理，而不是 0
    sens = np.where(
        sens_hit > 0,
        _robust_norm(sens_hit),
        0.10 * out["activity_risk"] + 0.05 * out["exposure_risk"],
    )
    out["sensitive_risk"] = np.clip(sens, 0, 1)

    # ===== overall =====
    weights = np.array([1, 1, 1, 1, 1, 1], dtype=float)
    weights = weights / weights.sum()
    risks = out[RISK_COLS].to_numpy()
    out["overall_score"] = np.clip(risks @ weights, 0, 1)

    return out
