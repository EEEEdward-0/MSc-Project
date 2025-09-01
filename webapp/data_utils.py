# -*- coding: utf-8 -*-
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import streamlit as st

# Basic IO
@st.cache_data(show_spinner=False)
def load_csv_cached(p: str) -> Optional[pd.DataFrame]:
    path = Path(p)
    if not path.exists(): return None
    try: return pd.read_csv(path)
    except Exception: return None

@st.cache_data(show_spinner=False)
def load_json_cached(p: str) -> Optional[dict]:
    path = Path(p)
    if not path.exists(): return None
    try: return json.loads(path.read_text())
    except Exception: return None

# User options: list json filenames under data/raw/users
def load_user_options(max_n: int = 400) -> List[str]:
    raw_dir = Path("data/raw/users")
    if raw_dir.exists():
        names = sorted(
            {p.stem for p in raw_dir.glob("*.json")} |
            {p.stem for p in raw_dir.glob("*.jsonl")} |
            {p.stem for p in raw_dir.glob("*.ndjson")}
        )
        if names:
            return ["Demo"] + names[:max_n]
    return ["Demo"]

# Derive factors from processed tables
def _clip01(x: float) -> float:
    try:
        return 0.0 if x is None or (isinstance(x, float) and x != x) else max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

@st.cache_data(show_spinner=False)
def _load_processed_df() -> Optional[pd.DataFrame]:
    for fname in ["data/processed/test.csv","data/processed/train_text.csv","data/processed/train.csv"]:
        p = Path(fname)
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                continue
    return None

def _match_row(df: pd.DataFrame, key: str):
    if df is None or df.empty: return None
    key = str(key)
    for c in ["user","username","author","name"]:
        if c in df.columns:
            m = df[c].astype(str) == key
            if m.any(): return df[m].iloc[0]
    for c in ["user_id","userid","id","userhash"]:
        if c in df.columns:
            m = df[c].astype(str) == key
            if m.any(): return df[m].iloc[0]
    return None

def _derive_factors_from_row(row: pd.Series) -> Dict[str,int]:
    def pct_from(cols, default=None):
        for c in cols:
            if c in row and pd.notna(row[c]):
                return int(round(_clip01(float(row[c]))*100))
        return default

    f = {
        "identity": pct_from(["username_risk_score","username_is_highrisk","username_digit_frac"], 20),
        "sensitive": pct_from(["text_mean_prob","text_frac_sensitive","sensitive_prob"], 30),
        "exposure":  pct_from(["exposure_score"], None),
        "activity":  pct_from(["activity_score"], None),
        "volume":    None,
        "concentration": pct_from(["topic_concentration","topic_focus"], 40)
    }
    posts = float(row.get("posts",0) or 0)
    comments = float(row.get("comments",0) or 0)
    total = float(row.get("total_activity", posts+comments) or (posts+comments))

    if f["volume"] is None:
        f["volume"] = int(round(min(1.0, total/500.0)*100))
    if f["activity"] is None:
        if "log_total_activity" in row and pd.notna(row["log_total_activity"]):
            f["activity"] = int(round(min(1.0, float(row["log_total_activity"])/8.0)*100))
        else:
            f["activity"] = int(round(min(1.0, total/500.0)*100))
    if f["exposure"] is None:
        bias = 0.0
        for c in ("is_post_heavy","is_comment_heavy"):
            if c in row and pd.notna(row[c]): bias += 0.15 if bool(row[c]) else 0.0
        f["exposure"] = int(round(min(1.0, total/800.0 + bias)*100))
    for k,v in list(f.items()):
        f[k] = 0 if v is None else max(0,min(100,int(v)))
    return f

@st.cache_data(show_spinner=False)
def get_user_factors_from_processed(user_key: str) -> Optional[Dict[str,int]]:
    df = _load_processed_df()
    if df is None: return None
    row = _match_row(df, user_key)
    if row is None: return None
    try: return _derive_factors_from_row(row)
    except Exception: return None

# Fallback: derive from raw JSON
def _is_year_token(tok: str) -> bool:
    try:
        y = int(tok); return 1900 <= y <= 2100
    except Exception:
        return False

@st.cache_data(show_spinner=False)
def get_user_factors_from_raw(username: str) -> Optional[Dict[str,int]]:
    base = Path("data/raw/users")
    cand = None
    for ext in (".json",".jsonl",".ndjson"):
        p = base / f"{username}{ext}"
        if p.exists(): cand = p; break
    if cand is None: return None

    try:
        if cand.suffix == ".json":
            obj = json.loads(cand.read_text())
            items = obj if isinstance(obj, list) else obj.get("items") or obj.get("data") or [obj]
        else:
            items = [json.loads(line) for line in cand.read_text().splitlines() if line.strip()]
    except Exception:
        return None
    if not items: return None

    posts = sum(1 for it in items if str(it.get("type","")).lower() in ("post","submission","link","self","t3"))
    comments = sum(1 for it in items if str(it.get("type","")).lower() in ("comment","t1"))
    total = posts + comments

    subs = []
    for it in items:
        sr = it.get("subreddit") or it.get("sub") or it.get("community")
        if isinstance(sr, dict):
            name = sr.get("display_name") or sr.get("name")
        else:
            name = sr
        if name: subs.append(str(name))
    conc = 40
    if subs:
        from collections import Counter
        c = Counter(subs); n = sum(c.values())
        conc = int(round(100*sum((v/n)**2 for v in c.values())))

    name = str(username)
    digit_frac = sum(ch.isdigit() for ch in name)/max(1,len(name))
    has_year = any(_is_year_token(tok) for tok in re.split(r"[_\-.]+", name))

    f = {
        "identity": int(round(min(1.0, 0.5*digit_frac + (0.4 if has_year else 0.0) + (0.1 if len(name)<=5 else 0.0))*100)),
        "sensitive": 30,
        "exposure":  int(round(min(1.0, total/800.0)*100)),
        "activity":  int(round(min(1.0, total/500.0)*100)),
        "volume":    int(round(min(1.0, total/500.0)*100)),
        "concentration": max(0,min(100,int(conc))),
    }
    return f

# Estimate accuracy from metrics
def estimate_accuracy_from_metrics(df_metrics: Optional[pd.DataFrame]) -> Optional[float]:
    """Return a 0–100 "estimated accuracy".

    Priority: (1 - Brier) * 100; else PR-AUC * 100; else AUC * 100.
    This is a display heuristic and not a statistically rigorous accuracy.
    """
    if df_metrics is None or df_metrics.empty: return None
    row = df_metrics.iloc[0]
    for key in ["brier","Brier"]:
        if key in row and pd.notna(row[key]):
            try:
                b = float(row[key]); return max(0.0, min(100.0, (1.0 - b)*100.0))
            except Exception: pass
    for key in ["prauc","PR-AUC","pr-auc"]:
        if key in row and pd.notna(row[key]):
            try:
                p = float(row[key]); return max(0.0, min(100.0, p*100.0))
            except Exception: pass
    for key in ["auc","AUC"]:
        if key in row and pd.notna(row[key]):
            try:
                a = float(row[key]); return max(0.0, min(100.0, a*100.0))
            except Exception: pass
    return None