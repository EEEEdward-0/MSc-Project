#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Create folders"
mkdir -p data/processed data/splits models/cv_logreg

echo "[2/6] Prepare demo features"

cp -f data/processed/features_demo.csv data/processed/features.csv

echo "[3/6] Auto-label to get train.csv (balanced-ish)"

python -m src.app auto-label \
  --input data/processed/features.csv \
  --out   data/processed/train.csv \
  --p-low 0.30 --p-high 0.70

echo "[4/6] Split train/test (20%)"
python -m src.app split \
  --input  data/processed/train.csv \
  --outdir data/splits \
  --test-size 0.2 --random_state 42

echo "[5/6] Train (logistic regression, with anti-leak rules enabled)"
python -m src.app train-cv \
  --data   data/splits/train.csv \
  --outdir models/cv_logreg \
  --folds  5 \
  --random_state 42 \
  --model logreg

echo "[6/6] Quick eval (optional)"
python -m src.app eval \
  --data   data/splits/test.csv \
  --models models/cv_logreg \
  --auto-threshold || true

echo "Done. Now run:  streamlit run webapp/app_streamlit.py"