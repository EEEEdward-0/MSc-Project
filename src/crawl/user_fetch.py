from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .io import dump_user_json
from .reddit_client import RedditCreds, make_client
from .throttle import Throttle


def _safe_get(obj, attr: str, default=None):
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def _safe_subreddit_name(obj) -> str:
    try:
        sub = getattr(obj, "subreddit", None)
        if sub is None:
            return ""
        # PRAW Subreddit has .display_name; fall back to str(sub)
        name = getattr(sub, "display_name", None)
        return str(name if name is not None else sub) or ""
    except Exception:
        return ""


def fetch_user_payload(
    reddit,
    username: str,
    max_submissions: int = 50,
    max_comments: int = 200,
    throttle_secs: float = 0.2,
) -> Dict[str, Any]:
    """
    抓取单个用户的最近帖子与评论，带关键字段：
    - created_utc（时间戳，秒）
    - subreddit（名字）
    - 文本/标题/得分/是否自贴/链接

    同时在顶层返回：n_posts/n_comments/n_docs/first_active/last_active，
    方便后续特征工程直接使用（无需再次遍历）。

    参数：
    - max_submissions: 最多抓取的帖子数（默认 50）
    - max_comments  : 最多抓取的评论数（默认 200）
    - throttle_secs : 每次 API 访问之间的最小等待（默认 0.2s，比之前更快）
    """
    th = Throttle(throttle_secs)

    try:
        th.wait()
        u = reddit.redditor(username)

        # -------- submissions --------
        submissions = []
        n_posts = 0
        first_ts = None
        last_ts = None
        try:
            for s in u.submissions.new(limit=max_submissions):
                ts = _safe_get(s, "created_utc", 0) or 0
                subname = _safe_subreddit_name(s)
                item = {
                    "id": _safe_get(s, "id", ""),
                    "created_utc": int(ts) if ts else 0,
                    "subreddit": subname,
                    "title": _safe_get(s, "title", ""),
                    "body": _safe_get(s, "selftext", ""),
                    "score": int(_safe_get(s, "score", 0) or 0),
                    "is_self": bool(_safe_get(s, "is_self", False)),
                    "permalink": _safe_get(s, "permalink", ""),
                    "removed_by_category": _safe_get(s, "removed_by_category", None),
                }
                submissions.append(item)
                n_posts += 1
                if ts:
                    if first_ts is None or ts < first_ts:
                        first_ts = ts
                    if last_ts is None or ts > last_ts:
                        last_ts = ts
        except Exception:
            # 保持健壮：遇到权限/隐私/被删用户等异常，继续后续流程
            pass

        # -------- comments --------
        th.wait()
        comments = []
        n_comments = 0
        try:
            for c in u.comments.new(limit=max_comments):
                ts = _safe_get(c, "created_utc", 0) or 0
                subname = _safe_subreddit_name(c)
                item = {
                    "id": _safe_get(c, "id", ""),
                    "created_utc": int(ts) if ts else 0,
                    "subreddit": subname,
                    "body": _safe_get(c, "body", ""),
                    "score": int(_safe_get(c, "score", 0) or 0),
                    "permalink": _safe_get(c, "permalink", ""),
                    "removed_by_category": _safe_get(c, "removed_by_category", None),
                }
                comments.append(item)
                n_comments += 1
                if ts:
                    if first_ts is None or ts < first_ts:
                        first_ts = ts
                    if last_ts is None or ts > last_ts:
                        last_ts = ts
        except Exception:
            pass

        n_docs = n_posts + n_comments
        payload: Dict[str, Any] = {
            "username": username,
            "n_posts": n_posts,
            "n_comments": n_comments,
            "n_docs": n_docs,
            "first_active": int(first_ts) if first_ts else 0,
            "last_active": int(last_ts) if last_ts else 0,
            "submissions": submissions,
            "comments": comments,
        }
        return payload

    except Exception as e:
        return {"username": username, "error": str(e)}


# ========== 并行抓取：单用户工作函数 ==========


def _crawl_one_worker(
    username: str,
    outdir: Path,
    max_submissions: int,
    max_comments: int,
    throttle_secs: float,
    max_retries: int = 3,
) -> Tuple[str, str, int, int, Optional[str]]:
    """
    返回: (username, status, n_posts, n_comments, error)
    status ∈ {"ok", "skip", "fail"}
    """
    try:
        out_path = outdir / f"{username}.json"
        if out_path.exists():
            return username, "skip", 0, 0, None

        # 使用 praw.ini 的 [bot]（通过传空凭证触发）
        reddit = make_client(RedditCreds(client_id="", client_secret="", user_agent=""))

        backoff = throttle_secs if throttle_secs > 0 else 0.2
        last_err: Optional[str] = None
        for _ in range(max_retries):
            try:
                payload = fetch_user_payload(
                    reddit,
                    username,
                    max_submissions=max_submissions,
                    max_comments=max_comments,
                    throttle_secs=throttle_secs,
                )
                # 原子写盘
                tmp = out_path.with_suffix(out_path.suffix + ".tmp")
                tmp.write_text(
                    __import__("json").dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
                import os as _os

                _os.replace(tmp, out_path)
                return (
                    username,
                    "ok",
                    int(payload.get("n_posts", 0)),
                    int(payload.get("n_comments", 0)),
                    None,
                )
            except Exception as e:  # 重试 + 退避
                last_err = str(e)
                __import__("time").sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)
        return username, "fail", 0, 0, (last_err or "unknown error")
    except Exception as e:
        return username, "fail", 0, 0, str(e)


# ========== CLI 入口：并行抓取 ==========
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Parallel crawl Reddit users (uses praw.ini [bot])")
    ap.add_argument("--input", required=True, help="用户名列表 txt（每行一个）")
    ap.add_argument("--outdir", required=True, help="输出 JSON 目录")
    ap.add_argument(
        "--workers", type=int, default=-1, help="并行进程数（-1=自动，使用机器最大合理并发）"
    )
    ap.add_argument("--max-submissions", type=int, default=200, help="每用户最多拉取的帖子数")
    ap.add_argument("--max-comments", type=int, default=500, help="每用户最多拉取的评论数")
    ap.add_argument("--throttle", type=float, default=0.1, help="单用户内部的最小间隔秒数")
    ap.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="每处理多少个用户打印一次带 ETA 的进度（默认 100）",
    )
    args = ap.parse_args()

    t0 = time.time()

    # 自动选择最大合理并发：CPU 核心数 x2，并保证至少 8 个
    try:
        import os

        if args.workers == -1:
            auto_workers = max(8, (os.cpu_count() or 8) * 2)
            args.workers = auto_workers
            print(f"[INFO] auto workers = {args.workers}")
    except Exception:
        if args.workers == -1:
            args.workers = 8
            print(f"[INFO] auto workers fallback = {args.workers}")

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 读取用户名（去空行）
    usernames = [ln.strip() for ln in inp.read_text(encoding="utf-8").splitlines() if ln.strip()]
    total = len(usernames)
    print(f"[INFO] loaded {total} usernames from {inp}")

    # 提前创建一个轻量 client，尽早暴露 praw.ini 配置问题（不会在子进程复用）
    try:
        _ = make_client(RedditCreds(client_id="", client_secret="", user_agent=""))
        print("[INFO] praw.ini [bot] OK")
    except Exception as e:
        print(f"[FATAL] cannot init PRAW via praw.ini [bot]: {e}")
        raise SystemExit(1)

    # 并发抓取
    submitted = 0
    ok = fail = skip = 0
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(
                    _crawl_one_worker,
                    u,
                    outdir,
                    args.max_submissions,
                    args.max_comments,
                    args.throttle,
                ): u
                for u in usernames
            }
            last_progress_print = 0
            for i, fut in enumerate(as_completed(futures), 1):
                u = futures[fut]
                try:
                    uname, status, np, nc, err = fut.result()
                    if status == "ok":
                        ok += 1
                    elif status == "skip":
                        skip += 1
                    else:
                        fail += 1
                    # 每条都有日志
                    err_msg = f" | err={err}" if err else ""
                    print(
                        f"[{i}/{total}] {uname} -> {status} (posts={np}, comments={nc}){err_msg}",
                        flush=True,
                    )
                except Exception as e:
                    fail += 1
                    print(f"[{i}/{total}] {u} -> ERROR {e}", flush=True)

                # 周期性打印 ETA 概览
                if i - last_progress_print >= max(1, args.print_every):
                    elapsed = max(1e-6, time.time() - t0)
                    rate = i / elapsed
                    eta_sec = max(0.0, (total - i) / rate) if rate > 0 else 0.0

                    def _fmt(sec: float) -> str:
                        m, s = divmod(int(sec), 60)
                        h, m = divmod(m, 60)
                        return f"{h:02d}:{m:02d}:{s:02d}"

                    print(
                        f"[PROGRESS] done={i}/{total} | ok={ok}, skip={skip}, fail={fail} | rate={rate:.2f}/s | elapsed={_fmt(elapsed)} | eta={_fmt(eta_sec)}",
                        flush=True,
                    )
                    last_progress_print = i
    except KeyboardInterrupt:
        # 友好终止：不丢进度，下次运行会跳过已存在的 JSON
        print("\n[INTERRUPT] 收到 Ctrl+C，正在请求终止子进程...", flush=True)
        try:
            ex.shutdown(cancel_futures=True)
        except Exception:
            pass

    total_elapsed = time.time() - t0

    def _fmt_total(sec: float) -> str:
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    print(
        f"[DONE] total={total}, ok={ok}, skip={skip}, fail={fail}, outdir={outdir} | elapsed={_fmt_total(total_elapsed)}"
    )
