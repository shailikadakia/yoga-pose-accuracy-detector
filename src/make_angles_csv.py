# make_angles_csv.py
"""
Convert the landmark XYZ dataset -> an angles/cues dataset that generalizes better.

What we export (columns):
- 10 joint angles (0–180°):
    L/R Elbow (11–13–15, 12–14–16) ............. straight vs bent arms (T/warrior vs sphinx/cobra)
    L/R Shoulder (13–11–23, 14–12–24) .......... arm-to-torso angle (arms out vs overhead vs down)
    L/R Hip (11–23–25, 12–24–26) ................ torso–thigh flex/extend (folds, boat, bridge/up-dog)
    L/R Knee (23–25–27, 24–26–28) ............... bent vs straight legs (warriors/lunges vs T/down-dog)
    L/R Ankle (25–27–31, 26–28–32) .............. foot articulation; helps seated lotus/butterfly vs standing

- 2 arm elevation angles vs horizontal (0–90):
    arm_elev_L/R (shoulder→wrist) ............... 0≈arms out to sides (T, Warrior II), 90≈arms overhead (Warrior I)

- Global posture & stance:
    torso_tilt (mid-hip→mid-shoulder vs vertical) 0=upright; large in boat/side plank/down-dog/folds
    pelvis_tilt (hip line vs horizontal) ......... level pelvis vs tipped (cat/cow, side-bends)
    shoulder_tilt (shoulder line vs horizontal) .. shoulder level vs tipped (reverse/extended side angle)
    shoulder_vs_hip_twist ........................ trunk rotation (extended/reverse warrior, twists)
    feet_over_shoulder ........................... stance width normalized by shoulder width (warriors/lunges/goddess)
    knee_spread_over_shoulder .................... knee width (butterfly/lotus show high spread)

- Asymmetry:
    knee_asymmetry ............................... |L_knee - R_knee| (one leg bent vs straight)
    arm_asymmetry ................................ |L_shoulder - R_shoulder| (one arm overhead/side)

- Inversion/fold booleans:
    hip_above_shoulders .......................... 1 in inverted V shapes (down-dog, sometimes plough)
    head_below_hips .............................. 1 in strong folds/inversions

- Head/neck:
    head_pitch (shoulder-mid → nose vs vertical) . head up vs down (cat/cow, forward fold vs halfway lift)

This set (~23 features) is translation/scale robust and maps well to your class list.
"""

import pandas as pd
import numpy as np
from utils import row_to_points33, compute_feature_vector_from_points, FEATURE_NAMES

IN_PATH  = "../pose_dataset.csv"         # your existing XYZ dataset
OUT_PATH = "../pose_angles_dataset.csv"  # new angles/cues dataset

def main():
    df = pd.read_csv(IN_PATH)

    # Ensure all landmark columns are numeric.
    numeric_cols = [f"{axis}{i}" for i in range(33) for axis in ("x","y","z")]
    missing = [c for c in numeric_cols + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # Coerce to numeric; invalids -> NaN then drop those rows
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

if __name__ == "__main__":
    main()
