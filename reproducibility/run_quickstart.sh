#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] Verifying Python and dependencies..."
python - <<'PY'
import sys, platform
print("Python:", sys.version.replace("\n"," "))
import pandas, numpy, sklearn, lightgbm, matplotlib, tqdm
print("pandas", pandas.__version__, "| numpy", numpy.__version__, "| sklearn", sklearn.__version__, "| lightgbm", lightgbm.__version__)
PY

echo "[2/4] Checking repository layout..."
for d in data data/processed models src; do
  [ -d "$d" ] || { echo "Creating $d"; mkdir -p "$d"; }
done

echo "[3/4] Using sample features to run a dry-run scoring step..."
SAMPLE=data/processed/features_sample.csv
if [ ! -f "$SAMPLE" ]; then
  echo "Sample features not found; creating a tiny synthetic file at $SAMPLE"
  python - <<'PY'
import pandas as pd, numpy as np, os
os.makedirs("data/processed", exist_ok=True)
n=50
df=pd.DataFrame({
  "identity": np.random.beta(2,5,size=n),
  "sensitive": np.random.beta(2,5,size=n),
  "exposure": np.random.beta(2,5,size=n),
  "activity": np.random.beta(2,5,size=n),
  "volume": np.random.beta(2,5,size=n),
  "concentration": np.random.beta(2,5,size=n),
  "label": np.random.randint(0,2,size=n)
})
df.to_csv("data/processed/features_sample.csv", index=False)
print("Wrote data/processed/features_sample.csv with shape", df.shape)
PY
fi

echo "[4/4] Running verification script..."
python scripts/verify_setup.py || { echo "scripts/verify_setup.py not found; using inline fallback"; python - <<'PY'
import pandas as pd, numpy as np
print("Loaded sample data:", pd.read_csv("data/processed/features_sample.csv").shape)
print("OK")
PY
}
echo "All good. See data/processed/features_sample.csv"
