from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

LEAKAGE_COLS = {"rule_score", "low_activity"}
BATCH_SIZE = 500  # 每批写盘大小


def _safe_load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def extract_features_from_user_json(json_path: Path) -> Dict[str, Any]:
    obj = _safe_load_json(json_path)
    username = obj.get("username", json_path.stem)

    n_posts = _len_of(obj, "posts") or _len_of(obj, "submissions")
    n_comments = _len_of(obj, "comments")

    n_upvotes = 0
    for key in ("posts", "submissions"):
        if isinstance(obj.get(key), list):
            for p in obj[key]:
                try:
                    n_upvotes += int(p.get("score", 0))
                except Exception:
                    pass
    if isinstance(obj.get("comments"), list):
        for c in obj["comments"]:
            try:
                n_upvotes += int(c.get("score", 0))
            except Exception:
                pass

    text_len_posts = _sum_len_text(obj.get("posts")) + _sum_len_text(obj.get("submissions"))
    text_len_comments = _sum_len_text(obj.get("comments"))

    return {
        "username": username,
        "n_posts": int(n_posts),
        "n_comments": int(n_comments),
        "n_upvotes": int(n_upvotes),
        "text_len_posts": int(text_len_posts),
        "text_len_comments": int(text_len_comments),
    }


def _drop_constant_and_leakage(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    nunique = df.nunique(dropna=False)
    constant_cols = [c for c in nunique.index if c not in ("username", "y") and nunique[c] <= 1]
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


def build_features(raw_json_dir: Path | str, out_csv: Path | str) -> pd.DataFrame:
    raw_dir = Path(raw_json_dir).expanduser().resolve()
    out_path = Path(out_csv).expanduser().resolve()
    all_files: List[Path] = sorted(raw_dir.rglob("*.json"))  # 递归扫描

    if not all_files:
        print(f"[WARN] no json files under {raw_dir}")
        pd.DataFrame(
            columns=[
                "username",
                "n_posts",
                "n_comments",
                "n_upvotes",
                "text_len_posts",
                "text_len_comments",
            ]
        ).to_csv(out_path, index=False)
        return pd.read_csv(out_path)

    done_users = _existing_usernames(out_path)
    total = len(all_files)

    for batch in _iter_in_batches(all_files, BATCH_SIZE):
        rows: List[Dict[str, Any]] = []
        for f in batch:
            username = f.stem
            if username in done_users:
                continue
            try:
                rows.append(extract_features_from_user_json(f))
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
    df_all, const_cols, leak_cols = _drop_constant_and_leakage(df_all)
    _save_corr_report(df_all, Path("reports") / "feature_corr_top.csv", topk=30)

    tmp_path = out_path.with_suffix(".tmp.csv")
    df_all.to_csv(tmp_path, index=False)
    tmp_path.replace(out_path)

    if const_cols:
        print("[INFO] dropped constant cols:", const_cols)
    if leak_cols:
        print("[INFO] dropped leakage cols:", leak_cols)
    print(
        f"[INFO] wrote features -> {out_path} with {df_all.shape[0]} rows, {df_all.shape[1]} columns"
    )

    return df_all
