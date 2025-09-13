import pandas as pd
import numpy as np
from utils import row_to_points33, compute_feature_vector_from_points, FEATURE_NAMES

IN_PATH  = "../pose_dataset.csv"         
OUT_PATH = "../pose_angles_dataset.csv"  


df = pd.read_csv(IN_PATH)

numeric_cols = [f"{axis}{i}" for i in range(33) for axis in ("x","y","z")]
missing = [c for c in numeric_cols + ["label"] if c not in df.columns]
if missing:
    raise ValueError(f"CSV is missing columns: {missing}")

#  invalids -> NaN then drop those rows
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
bad = df[numeric_cols].isna().any(axis=1)
if bad.any():
    print(f"⚠️ Dropping {bad.sum()} malformed rows with non-numeric landmark values.")
    df = df.loc[~bad].reset_index(drop=True)

rows = []
for _, r in df.iterrows():
    pts33 = row_to_points33(r)                     # (33,3)
    feats = compute_feature_vector_from_points(pts33)  # (23,)
    rows.append([*feats, r["label"]])

out = pd.DataFrame(rows, columns=FEATURE_NAMES + ["label"])
out.to_csv(OUT_PATH, index=False)
print(f"✅ Wrote {OUT_PATH}  ({len(out)} rows, {len(FEATURE_NAMES)} features)")


