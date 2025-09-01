from __future__ import annotations
import os, sys, importlib
import pandas as pd

def _v(mod): 
    try:
        return importlib.import_module(mod).__version__
    except Exception:
        return "n/a"

print("Python:", sys.version.replace("\n"," "))
print("pandas", _v("pandas"), "| numpy", _v("numpy"), "| sklearn", _v("sklearn"), "| lightgbm", _v("lightgbm"))
p = "data/processed/features_sample.csv"
if os.path.exists(p):
    df = pd.read_csv(p)
    print("Sample data:", df.shape, "columns:", list(df.columns))
else:
    print("Sample data not found at", p)
print("Verification complete.")
