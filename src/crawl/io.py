from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_usernames(path: Path) -> List[str]:
    p = Path(path).expanduser().resolve()
    users: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        users.append(line.split()[0])
    return users


def dump_user_json(outdir: Path, username: str, payload: Dict[str, Any]) -> None:
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = outdir / f".{username}.tmp.json"
    dst = outdir / f"{username}.json"
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(dst)
