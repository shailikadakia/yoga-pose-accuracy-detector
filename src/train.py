import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib

# -----------------------
# Load & basic checks
# -----------------------
data = pd.read_csv("../pose_dataset.csv")
print(data.info())
print("Nulls per column:\n", data.isnull().sum())

# TODO: if you need to drop bad rows, do it inside this function
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # e.g., df = df.dropna()
    return df

data = preprocess_data(data)

# -----------------------
# Features / Labels
# -----------------------
X = data.drop(columns=["label"])
le = LabelEncoder()
y = le.fit_transform(data["label"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------
# Pipeline + GridSearch
# -----------------------
pipe = Pipeline([
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
print("Best params:", grid.best_params_)

# -----------------------
# Evaluate
# -----------------------
y_pred = best_pipe.predict(X_test)

# Use the SAME encoder's classes for names
target_names = le.classes_
print(classification_report(y_test, y_pred, target_names=target_names, digits=3))

cm = confusion_matrix(y_test, y_pred, labels=range(len(target_names)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(xticks_rotation="vertical", cmap="Blues")
plt.tight_layout()
plt.show()

# -----------------------
# Save bundle (pipeline + encoder)
# -----------------------
bundle = {"pipe": best_pipe, "label_encoder": le}
joblib.dump(bundle, "pose_knn_bundle.pkl")

# (Optional) sanity prints
print("Encoder classes:", le.classes_)
print("Classifier classes:", best_pipe.named_steps["knn"].classes_)  # indices expected by the KNN
