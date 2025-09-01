# src/crawl/reddit_client.py
from dataclasses import dataclass

import praw


@dataclass
class RedditCreds:
    client_id: str = ""
    client_secret: str = ""
    user_agent: str = ""


def make_client(creds: RedditCreds | None = None, site_name: str = "bot") -> praw.Reddit:
    """
    命令行传了 creds 就用显式凭证；否则从 praw.ini 的 [bot] 读取。
    """
    if creds and (creds.client_id or creds.client_secret or creds.user_agent):
        reddit = praw.Reddit(
            client_id=creds.client_id or None,
            client_secret=creds.client_secret or None,
            user_agent=creds.user_agent or None,
        )
    else:
        reddit = praw.Reddit(site_name=site_name)

    # 触发加载并尽早报错
    _ = reddit.read_only
    return reddit
