#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录
IN = ROOT / "data" / "processed" / "train.csv"
OUTDIR = ROOT / "data" / "processed"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)
drop = [c for c in ["username"] if c in df.columns]
df = df.drop(columns=drop)
y = df["label"].astype(int)

# 80/20 分层切分（固定随机种子）
dev, test = train_test_split(df, test_size=0.2, stratify=y, random_state=42)
test.to_csv(OUTDIR / "test.csv", index=False)

# 在 dev 内做 5 折
dev = dev.copy()
dev["fold"] = -1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (_, val_idx) in enumerate(skf.split(dev, dev["label"].astype(int))):
    dev.loc[dev.iloc[val_idx].index, "fold"] = fold

dev.to_csv(OUTDIR / "dev_folds.csv", index=False)
print(f"Saved {len(dev)} dev rows with folds, {len(test)} test rows to {OUTDIR}")
