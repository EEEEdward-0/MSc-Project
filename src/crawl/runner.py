# src/crawl/runner.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from .io import dump_user_json, load_usernames
from .reddit_client import RedditCreds, make_client
from .throttle import Throttler
from .user_fetch import fetch_user_payload


def run_crawl(
    usernames_txt: Path | str,
    outdir: Path | str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    user_agent: Optional[str] = None,
    min_interval: float = 1.0,
) -> int:
    """
    执行端到端爬取：读用户名 -> 调 Reddit API -> 写 JSON。
    返回成功写出的数量。
    """
    names = load_usernames(usernames_txt)
    throttler = Throttler(min_interval=min_interval)
    reddit = make_client(RedditCreds(client_id or "", client_secret or "", user_agent or ""))

    outdir = Path(outdir)
    ok = 0
    for u in names:
        throttler.wait()
        payload = fetch_user_payload(reddit, u)
        dump_user_json(outdir, u, payload)
        ok += 1
    return ok
