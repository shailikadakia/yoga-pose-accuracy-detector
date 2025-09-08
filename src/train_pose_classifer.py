# train_pose_classifier.py
from utils import compute_feature_vector_from_points, row_to_points33, FEATURE_NAMES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib

def featurize_dataframe(X_df: pd.DataFrame) -> np.ndarray:
    feats = [compute_feature_vector_from_points(row_to_points33(r)) for _, r in X_df.iterrows()]
    return np.vstack(feats)

# -------- load & CLEAN the CSV ----------
data = pd.read_csv("../pose_dataset.csv")

EXPECTED = [f"{axis}{i}" for i in range(33) for axis in ("x","y","z")]
required_cols = EXPECTED + ["label"]

# 1) Check columns
missing = [c for c in required_cols if c not in data.columns]
if missing:
    # Common pitfall: the last two columns merged into 'z32' if a row is short.
    raise ValueError(f"CSV missing columns: {missing}\nGot: {list(data.columns)}")

# 2) Coerce numeric cols; anything non-numeric -> NaN
data[EXPECTED] = data[EXPECTED].apply(pd.to_numeric, errors="coerce")

# 3) Drop bad rows (where any x/y/z is NaN)
bad_mask = data[EXPECTED].isna().any(axis=1)
if bad_mask.any():
    print(f"⚠️ Dropping {bad_mask.sum()} malformed rows (non-numeric x/y/z). "
          f"E.g. rows where z32 accidentally contains the label.")
data = data.loc[~bad_mask].reset_index(drop=True)

# -------- split + pipeline (same as you had, but with featurizer) ----------
X = data[EXPECTED]         # ONLY numeric columns
le = LabelEncoder()
y = le.fit_transform(data["label"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipe = Pipeline([
    ("angles", FunctionTransformer(featurize_dataframe, validate=False)),
    ("scaler", MinMaxScaler()),
    ("knn", KNeighborsClassifier())
])

param_grid = {
    "knn__n_neighbors": range(1, 8),
    "knn__metric": ["euclidean", "manhattan", "minkowski"],
    "knn__weights": ["uniform", "distance"],
}

grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)
best_pipe = grid.best_estimator_
scaler = best_pipe.named_steps["scaler"]
knn    = best_pipe.named_steps["knn"]

bundle = {
    "scaler": scaler,
    "knn": knn,
    "label_encoder": le,
    "feature_names": FEATURE_NAMES,
}

print("Best params:", grid.best_params_)

y_pred = best_pipe.predict(X_test)
target_names = le.classes_
print(classification_report(y_test, y_pred, target_names=target_names, digits=3))

cm = confusion_matrix(y_test, y_pred, labels=range(len(target_names)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(xticks_rotation="vertical", cmap="Blues")
plt.tight_layout()
plt.show()

import joblib
joblib.dump(bundle, "pose_knn_runtime.pkl")
print("✅ Saved pose_knn_bundle.pkl")