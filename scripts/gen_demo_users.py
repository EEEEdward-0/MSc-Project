#!/usr/bin/env python3
"""
生成一个极小的 Reddit 风格原始数据 users.json（本地临时文件，不提交到仓库）。
用途：让 featurize -> auto-label 能自然地产生正/负两类，验证端到端流水线可运行。
"""
from pathlib import Path
import json

OUT = Path("data/raw/users.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

users = [
    {
        "id": "demo_user_pos",
        "name": "demo_user_pos",
        "comments": [
            {
                "subreddit": "privacy",
                "body": "Contact me at test@example.com or call 123-456-7890",
                "created_utc": 1690000000,
            },
            {
                "subreddit": "privacy",
                "body": "sharing my email again: test@example.com",
                "created_utc": 1690001000,
            },
        ],
    },
    {
        "id": "demo_user_neg",
        "name": "demo_user_neg",
        "comments": [
            {"subreddit": "test", "body": "hello world", "created_utc": 1690002000},
            {"subreddit": "test", "body": "just a normal comment", "created_utc": 1690003000},
        ],
    },
]

OUT.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] wrote {OUT} (bytes={OUT.stat().st_size})")