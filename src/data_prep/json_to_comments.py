#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Reddit JSON / JSONL to a 2-col CSV: username,text

Usage:
  python -m src.data_prep.json_to_comments \
    --input data/raw/users \
    --out data/raw/comments.csv \
    --keep-title-if-removed \
    --verbose

Notes:
- Accepts a directory (walk all *.json / *.jsonl) or a single file.
- Tries to read both comments and submissions.
- Output header: username,text
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import gzip
import io
from pathlib import Path
from typing import Iterable, Dict, Any, Tuple
import csv

REMOVED_TOKENS = {"[removed]", "[deleted]", None, ""}


# -------- helpers --------
def is_removed(text: str | None) -> bool:
    """Return True if text is removed/empty."""
    if text is None:
        return True
    t = str(text).strip().lower()
    return t in REMOVED_TOKENS


def norm(s: str | None) -> str:
    """Strip and collapse whitespace."""
    if s is None:
        return ""
    return " ".join(str(s).strip().split())


def open_maybe_gzip(path: Path) -> io.TextIOBase:
    """Open file, auto-handle .gz."""
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def yield_json_items(fp: io.TextIOBase) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from file (JSON or JSONL)."""
    # try JSON lines first
    first_pos = fp.tell()
    line = fp.readline()
    fp.seek(first_pos)
    if line.strip().startswith("{"):
        # Heuristic: treat as JSONL if multiple lines with braces
        is_jsonl = True
        # If the whole file is a single JSON object/array, we'll catch it below
    else:
        is_jsonl = False

    if is_jsonl:
        for ln in fp:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                yield obj
            except Exception:
                continue
        return

    # fallback: whole-file JSON (dict / list)
    try:
        data = json.load(fp)
    except Exception:
        return

    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                yield obj
    elif isinstance(data, dict):
        # common Reddit export patterns
        if "data" in data and isinstance(data["data"], list):
            for obj in data["data"]:
                if isinstance(obj, dict):
                    yield obj
        else:
            yield data


def extract_username(obj: Dict[str, Any]) -> str:
    """Pick a username field."""
    for k in ("author", "username", "user", "name"):
        v = obj.get(k)
        if v is not None:
            v = str(v).strip()
            if v and v.lower() not in {"[deleted]", "automoderator"}:
                return v
    return ""


def extract_text(obj: Dict[str, Any], keep_title_if_removed: bool) -> str:
    """
    Extract a comment/submission text.
    - Prefer 'body' for comments.
    - For submissions, combine 'title' + 'selftext'.
    - If removed body and keep_title_if_removed=True, fall back to title.
    """
    body = obj.get("body")
    title = obj.get("title")
    selftext = obj.get("selftext") or obj.get("text") or obj.get("content")

    # comment path
    if body is not None:
        if not is_removed(body):
            return norm(body)
        # fall back to title if asked
        if keep_title_if_removed and not is_removed(title):
            return norm(title)
        return ""

    # submission path
    parts = []
    if not is_removed(title):
        parts.append(norm(title))
    if not is_removed(selftext):
        parts.append(norm(selftext))
    return "\n".join([p for p in parts if p])


def parse_one_file(path: Path,
                   keep_title_if_removed: bool) -> Iterable[Tuple[str, str]]:
    """Yield (username, text) from a file.

    Supports:
      1) Plain JSON/JSONL where each object is a comment/submission.
      2) User-packed JSON: { "username": "...", "comments": [...], "submissions": [...] }.
    """
    # Try whole-file JSON first (handles user-packed single-line files)
    with open_maybe_gzip(path) as fp:
        try:
            fp.seek(0)
            obj = json.load(fp)
        except Exception:
            obj = None

    if isinstance(obj, dict) and (isinstance(obj.get("comments"), list) or isinstance(obj.get("submissions"), list)):
        # User-packed object
        uname_top = str(obj.get("username") or obj.get("user") or "").strip()

        # From comments[]: prefer 'body'
        comments = obj.get("comments") or []
        if isinstance(comments, list):
            for c in comments:
                if not isinstance(c, dict):
                    continue
                u = extract_username(c) or uname_top  # fallback to top username
                t = extract_text(c, keep_title_if_removed)
                if u and t:
                    yield u, t

        # From submissions[]: combine title/selftext
        subs = obj.get("submissions") or []
        if isinstance(subs, list):
            for s in subs:
                if not isinstance(s, dict):
                    continue
                u = extract_username(s) or uname_top
                t = extract_text(s, keep_title_if_removed)
                if u and t:
                    yield u, t
        return

    # Fallback to generic streaming (JSONL / array / dict)
    with open_maybe_gzip(path) as fp:
        for o in yield_json_items(fp):
            u = extract_username(o)
            t = extract_text(o, keep_title_if_removed)
            if u and t:
                yield u, t


# -------- main --------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Reddit JSON -> comments.csv")
    ap.add_argument("--input", required=True, help="Input dir or file")
    ap.add_argument("--out", required=True, help="Output CSV (username,text)")
    ap.add_argument("--keep-title-if-removed", action="store_true",
                    help="Use title when body is [removed]/[deleted]")
    ap.add_argument("--verbose", action="store_true", help="Print progress")
    ap.add_argument("--limit", type=int, default=0, help="Max rows (0=all)")
    return ap.parse_args()


def scan_files(inp: Path) -> Iterable[Path]:
    """Yield JSON/JSONL files under inp."""
    exts = {".json", ".jsonl", ".json.gz", ".jsonl.gz"}
    if inp.is_file():
        if any(str(inp).endswith(e) for e in exts):
            yield inp
        return
    for root, _, files in os.walk(inp):
        for fn in files:
            p = Path(root) / fn
            if any(str(p).endswith(e) for e in exts):
                yield p


def main() -> None:
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    inspected = parsed = 0
    no_author = no_text = 0

    with open(out, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["username", "text"])  # header

        for i, fpath in enumerate(scan_files(inp), 1):
            if args.verbose and i % 250 == 0:
                print(f"[json_to_comments] scanned files: {i}", file=sys.stderr)
            try:
                for user, text in parse_one_file(fpath, args.keep_title_if_removed):
                    w.writerow([user, text])
                    parsed += 1
                    if args.limit and parsed >= args.limit:
                        raise StopIteration
                inspected += 1
            except StopIteration:
                inspected += 1
                break
            except Exception:
                # best-effort: skip bad file
                inspected += 1
                continue

    # crude tallies (for transparency; not perfect per-record)
    # We can’t count per-record no_author/no_text cheaply without reading twice.
    print(f"[json_to_comments] Scanning {inspected} files from {inp}")
    print(f"[json_to_comments] parsed={parsed} (rows)  -> {out}")


if __name__ == "__main__":
    main()