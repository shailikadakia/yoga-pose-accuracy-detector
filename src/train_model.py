# train_pose_classifier_angles.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib

IN_PATH = "../pose_angles_dataset.csv"     
OUT_BUNDLE = "pose_knn_runtime.pkl"        

# Load Data
data = pd.read_csv(IN_PATH)
print(data.info())
print("Nulls per column:\n", data.isnull().sum())

# X = all features, y = pose label
X = data.drop(columns=["label"])
le = LabelEncoder()
y = le.fit_transform(data["label"])

feature_names = list(X.columns)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Pipeline and grid
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

# Evaluate 
y_pred = best_pipe.predict(X_test)
target_names = le.classes_
print(classification_report(y_test, y_pred, target_names=target_names, digits=3))

cm = confusion_matrix(y_test, y_pred, labels=range(len(target_names)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(xticks_rotation="vertical", cmap="Blues")
plt.tight_layout()
plt.show()

# Runtime Bundle
bundle = {
    "scaler": best_pipe.named_steps["scaler"],
    "knn": best_pipe.named_steps["knn"],
    "label_encoder": le,
    "feature_names": feature_names,
}
joblib.dump(bundle, OUT_BUNDLE)
print(f"✅ Saved {OUT_BUNDLE}")
print("Encoder classes:", le.classes_)
print("Classifier classes:", best_pipe.named_steps["knn"].classes_)
