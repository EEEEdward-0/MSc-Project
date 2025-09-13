#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_labeled(
    input_csv: Path,
    outdir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    folds: int = 5
) -> None:
    """
    Split a labeled dataset into stratified train/test, add folds to train.
    - Reads input_csv
    - Drops 'username' if present
    - Ensures label column is 'y' (renames 'label' to 'y' if needed)
    - Splits into train/test stratified by 'y'
    - Writes test.csv and train.csv to outdir
    - Adds 'fold' column to train set using StratifiedKFold
    - Writes dev_folds.csv (train with folds) to outdir
    - Prints summary of counts
    """
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    if "username" in df.columns:
        df = df.drop(columns=["username"])
    if "y" not in df.columns:
        if "label" in df.columns:
            df = df.rename(columns={"label": "y"})
        else:
            raise ValueError("No label column found (expected 'y' or 'label').")
    y = df["y"].astype(int)
    train, test = train_test_split(
        df, test_size=test_size, stratify=y, random_state=random_state
    )
    test.to_csv(outdir / "test.csv", index=False)
    train = train.copy()
    train["fold"] = -1
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    for fold, (_, val_idx) in enumerate(skf.split(train, train["y"].astype(int))):
        train.loc[train.iloc[val_idx].index, "fold"] = fold
    train.to_csv(outdir / "dev_folds.csv", index=False)
    print(
        f"Saved {len(train)} train rows with folds, {len(test)} test rows to {outdir}"
    )
