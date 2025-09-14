from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import os
from multiprocessing import Pool, cpu_count
try:
    import orjson as _orjson
except Exception:
    _orjson = None

# === Thesis-aligned regexes and helpers ===
import re, math
from collections import Counter
# === Simple regex-based PII / sensitive patterns ===
EMAIL_RE  = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE  = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})")
SENS_PAT  = re.compile(r"\b(password|ssn|social\s*security|credit\s*card|address|email|phone)\b", re.I)

def _get_text_from_post(p: Dict[str, Any]) -> str:
    txt = (p.get("selftext") or p.get("body") or "") + "\n" + (p.get("title") or "")
    return txt.strip()

def _get_text_from_comment(c: Dict[str, Any]) -> str:
    return (c.get("body") or "").strip()

def _score_of(obj: Dict[str, Any]) -> int:
    for k in ("score", "ups", "likes"):
        v = obj.get(k)
        if isinstance(v, (int, float)):
            return int(v)
    return 0

def _created(obj: Dict[str, Any]) -> float:
    for k in ("created_utc", "created"):
        v = obj.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return np.nan

def _entropy(labels: List[str]) -> float:
    if not labels:
        return 0.0
    c = Counter([str(x) for x in labels if x])
    n = sum(c.values())
    if n == 0:
        return 0.0
    ent = -sum((cnt / n) * math.log((cnt / n) + 1e-12) for cnt in c.values())
    K = len(c)
    return float(ent / math.log(K + 1e-12)) if K > 1 else 0.0

LEAKAGE_COLS = {"rule_score", "low_activity"}
BATCH_SIZE = 500  # 每批写盘大小


def _load_json_fast(path: Path) -> Dict[str, Any]:
    try:
        if _orjson is not None:
            return _orjson.loads(path.read_bytes())
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _safe_load_json(path: Path) -> Dict[str, Any]:
    return _load_json_fast(path)


def _len_of(obj: Dict[str, Any], key: str) -> int:
    v = obj.get(key, [])
    if isinstance(v, list):
        return len(v)
    return 0


def _sum_len_text(items: Any) -> int:
    if not isinstance(items, list):
        return 0
    total = 0
    for it in items:
        if isinstance(it, dict):
            txt = it.get("body") or it.get("selftext") or it.get("title") or ""
            if isinstance(txt, str):
                total += len(txt)
        elif isinstance(it, str):
            total += len(it)
    return total


def extract_features_from_user_json(json_path: Path, text_feats: bool = False, text_dims: int = 32) -> Dict[str, Any]:
    obj = _safe_load_json(json_path)
    username = obj.get("username", json_path.stem)

    username_len = len(str(username))
    has_digit_in_username = int(any(ch.isdigit() for ch in str(username)))
    has_underscore_in_username = int("_" in str(username))

    posts = obj.get("posts") or obj.get("submissions") or []
    comments = obj.get("comments") or []

    n_posts = len(posts) if isinstance(posts, list) else 0
    n_comments = len(comments) if isinstance(comments, list) else 0
    n_docs = n_posts + n_comments

    # upvotes / scores
    n_upvotes = 0
    for p in (posts if isinstance(posts, list) else []):
        n_upvotes += _score_of(p)
    for c in (comments if isinstance(comments, list) else []):
        n_upvotes += _score_of(c)

    # text lengths
    text_len_posts = _sum_len_text(posts)
    text_len_comments = _sum_len_text(comments)

    # PII / sensitive hits, subreddit list, timestamps
    email_hits = 0
    phone_hits = 0
    sens_hits = 0
    subrs: List[str] = []
    ts: List[float] = []

    for p in (posts if isinstance(posts, list) else []):
        t = _get_text_from_post(p)
        email_hits += len(EMAIL_RE.findall(t))
        phone_hits += len(PHONE_RE.findall(t))
        sens_hits += len(SENS_PAT.findall(t))
        if p.get("subreddit"):
            subrs.append(str(p["subreddit"]))
        ts.append(_created(p))

    for c in (comments if isinstance(comments, list) else []):
        t = _get_text_from_comment(c)
        email_hits += len(EMAIL_RE.findall(t))
        phone_hits += len(PHONE_RE.findall(t))
        sens_hits += len(SENS_PAT.findall(t))
        if c.get("subreddit"):
            subrs.append(str(c["subreddit"]))
        ts.append(_created(c))

    has_email = int(email_hits > 0)
    has_phone = int(phone_hits > 0)
    pii_email_rate = float(email_hits) / max(1, n_docs)
    pii_phone_rate = float(phone_hits) / max(1, n_docs)

    sensitive_count = int(sens_hits)
    sens_hit_rate = float(sens_hits) / max(1, n_docs)

    subreddit_entropy = _entropy(subrs)

    # Optional text features (non-leaky, derived from raw text only)
    text_hash: Dict[str, float] = {}
    if text_feats:
        try:
            from sklearn.feature_extraction.text import HashingVectorizer
            # Concatenate all texts for this user
            texts: list[str] = []
            for p in (posts if isinstance(posts, list) else []):
                texts.append(_get_text_from_post(p))
            for c in (comments if isinstance(comments, list) else []):
                texts.append(_get_text_from_comment(c))
            doc = "\n".join([t for t in texts if isinstance(t, str) and t])
            vec = HashingVectorizer(
                n_features=int(text_dims),
                alternate_sign=False,
                norm='l2',
                analyzer='word',
                ngram_range=(1, 2)
            )
            X = vec.transform([doc])
            arr = X.toarray().ravel()
            for i, v in enumerate(arr):
                text_hash[f"text_hash_{i}"] = float(v)
        except Exception:
            # Fail-closed: if hashing features fail, just skip them
            text_hash = {}

    return {
        "username": username,
        "n_posts": int(n_posts),
        "n_comments": int(n_comments),
        "n_upvotes": int(n_upvotes),
        "text_len_posts": int(text_len_posts),
        "text_len_comments": int(text_len_comments),
        "pii_email_rate": pii_email_rate,
        "pii_phone_rate": pii_phone_rate,
        "has_email": has_email,
        "has_phone": has_phone,
        "sens_hit_rate": sens_hit_rate,
        "sensitive_count": sensitive_count,
        "subreddit_entropy": float(subreddit_entropy),
        "n_docs": int(n_docs),
        "username_len": int(username_len),
        "has_digit_in_username": int(has_digit_in_username),
        "has_underscore_in_username": int(has_underscore_in_username),
        **text_hash,
    }


def _drop_constant_and_leakage(
    df: pd.DataFrame,
    keep_cols: Iterable[str] | None = None
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    keep = set(keep_cols or [])
    keep.update({"username", "y"})  # always keep identifiers/labels

    nunique = df.nunique(dropna=False)
    constant_cols = [
        c for c in nunique.index
        if c not in keep and nunique[c] <= 1
    ]
    leakage_cols = [c for c in df.columns if c in LEAKAGE_COLS]
    drop_cols = sorted(set(constant_cols + leakage_cols))
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df, constant_cols, leakage_cols


def _save_corr_report(df: pd.DataFrame, out_csv: Path, topk: int = 30) -> None:
    if "y" not in df.columns:
        return
    num_cols = [
        c
        for c in df.columns
        if c not in ("username", "y") and np.issubdtype(df[c].dtype, np.number)
    ]
    if not num_cols:
        return
    corrs = df[num_cols].corrwith(df["y"]).sort_values(ascending=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    corrs.head(topk).to_csv(out_csv, header=["corr_y"])


def _existing_usernames(out_csv: Path) -> set[str]:
    if out_csv.exists():
        try:
            df = pd.read_csv(out_csv, usecols=["username"])
            return set(df["username"].astype(str).tolist())
        except Exception:
            return set()
    return set()


def _append_rows(out_csv: Path, rows: List[Dict[str, Any]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not out_csv.exists():
        df.to_csv(out_csv, index=False)
    else:
        old = pd.read_csv(out_csv, nrows=1)
        all_cols = list(dict.fromkeys(list(old.columns) + list(df.columns)))
        df = df.reindex(columns=all_cols)
        df.to_csv(out_csv, mode="a", header=False, index=False)


def _iter_in_batches(items: Iterable[Path], batch_size: int) -> Iterable[List[Path]]:
    batch: List[Path] = []
    for it in items:
        batch.append(it)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_features(raw_json_dir: Path | str, out_csv: Path | str,
                   n_workers: int | None = None,
                   limit: int | None = None,
                   force: bool | None = None,
                   text_feats: bool | None = None,
                   text_dims: int | None = None) -> pd.DataFrame:
    """
    Build thesis-aligned features from raw user JSON files.

    Backward-compatible with CLI: the extra args have defaults and are optional.
    You can also control behavior via environment variables:
      - FEAT_FORCE=1         -> force full recompute (ignore existing CSV)
      - FEAT_WORKERS=8       -> parallel workers (default: cpu_count()-1)
      - FEAT_LIMIT=3000      -> only process first N files (for quick demo)
      - FEAT_TEXT=1          -> enable hashed text features (non-leaky)
      - FEAT_TEXT_DIMS=64    -> dimensions for hashed text features
    """
    raw_dir = Path(raw_json_dir).expanduser().resolve()
    out_path = Path(out_csv).expanduser().resolve()
    files: List[Path] = sorted(raw_dir.rglob("*.json"))
    if limit is None:
        try:
            limit = int(os.environ.get("FEAT_LIMIT", "0")) or None
        except Exception:
            limit = None
    if limit:
        files = files[:limit]

    if not files:
        print(f"[WARN] no json files under {raw_dir}")
        pd.DataFrame(
            columns=[
                "username",
                "n_posts", "n_comments", "n_upvotes",
                "text_len_posts", "text_len_comments",
                "pii_email_rate", "pii_phone_rate", "has_email", "has_phone",
                "sens_hit_rate", "sensitive_count",
                "subreddit_entropy", "n_docs",
                "username_len", "has_digit_in_username", "has_underscore_in_username",
            ]
        ).to_csv(out_path, index=False)
        return pd.read_csv(out_path)

    if force is None:
        force = os.environ.get("FEAT_FORCE", "") not in ("", "0", "false", "False")
    if n_workers is None:
        try:
            n_workers = int(os.environ.get("FEAT_WORKERS", "0")) or max(1, cpu_count() - 1)
        except Exception:
            n_workers = max(1, cpu_count() - 1)
    if text_feats is None:
        text_feats = os.environ.get("FEAT_TEXT", "").lower() not in ("", "0", "false")
    if text_dims is None:
        try:
            text_dims = int(os.environ.get("FEAT_TEXT_DIMS", "32"))
        except Exception:
            text_dims = 32

    thesis_cols = [
        "username",
        "n_posts", "n_comments", "n_upvotes",
        "text_len_posts", "text_len_comments",
        "pii_email_rate", "pii_phone_rate", "has_email", "has_phone",
        "sens_hit_rate", "sensitive_count",
        "subreddit_entropy", "n_docs",
        "username_len", "has_digit_in_username", "has_underscore_in_username",
    ]

    def _finalize_and_write(df: pd.DataFrame) -> pd.DataFrame:
        for c in thesis_cols:
            if c not in df.columns:
                df[c] = 0
        df = df[thesis_cols + [c for c in df.columns if c not in thesis_cols]]
        df, const_cols, leak_cols = _drop_constant_and_leakage(df, keep_cols=thesis_cols)
        _save_corr_report(df, Path("reports") / "feature_corr_top.csv", topk=30)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".tmp.csv")
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(out_path)
        if const_cols:
            print("[INFO] dropped constant cols:", const_cols)
        if leak_cols:
            print("[INFO] dropped leakage cols:", leak_cols)
        print(f"[featurize] wrote {len(df)} rows x {len(df.columns)} cols -> {out_path}")
        return df

    # Fast parallel full recompute (recommended for large datasets or when FEAT_FORCE=1)
    if force or not out_path.exists():
        print(f"[featurize] parallel build with {n_workers} workers (force={force}) on {len(files)} files")
        def _extract_with_cfg(args: Tuple[Path, bool, int]) -> Dict[str, Any]:
            p, tf, td = args
            return extract_features_from_user_json(p, text_feats=tf, text_dims=td)
        with Pool(processes=n_workers) as pool:
            rows = list(pool.imap_unordered(_extract_with_cfg, [(f, bool(text_feats), int(text_dims)) for f in files], chunksize=200))
        df_all = pd.DataFrame(rows)
        return _finalize_and_write(df_all)

    # Incremental append mode (backward-compatible)
    done_users = _existing_usernames(out_path)
    total = len(files)
    for batch in _iter_in_batches(files, BATCH_SIZE):
        rows: List[Dict[str, Any]] = []
        for f in batch:
            username = f.stem
            if username in done_users:
                continue
            try:
                rows.append(extract_features_from_user_json(f, text_feats=bool(text_feats), text_dims=int(text_dims)))
            except Exception as e:
                rows.append(
                    {
                        "username": username,
                        "n_posts": 0,
                        "n_comments": 0,
                        "n_upvotes": 0,
                        "text_len_posts": 0,
                        "text_len_comments": 0,
                        "parse_error": str(e),
                    }
                )
        if rows:
            _append_rows(out_path, rows)
            done_users.update(r["username"] for r in rows)
        print(f"[INFO] progress: {len(done_users)}/{total}")
    df_all = pd.read_csv(out_path)
    return _finalize_and_write(df_all)
