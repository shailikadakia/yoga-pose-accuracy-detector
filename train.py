import pandas as pd
import numpy as np # take list of data and convert it into an array
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
import seaborn as sns 
import joblib

data = pd.read_csv("pose_dataset.csv")
data.info()
print(data.isnull().sum())

# Data Cleaning
# To Do: Will the data be incomplete? Need to remove those rows 
def preprocess_data(dataframe):
  return dataframe

data = preprocess_data(data)
X = data.drop(columns=["label"])
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(data["label"])
# Create Features / Target Variables (Make Flashcards)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=Y)

# ML Preprocessing
scalar = MinMaxScaler()
# don't have data to scale yet, no training and testing data yet
# X is front of the flashcard, Y is the back of the flashcard
# Computer can see train flashcards when training (X_train, Y_train)
# Then when testing the model, can only check X_test 

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

# Hyperparameter Tuning - KNN Model
# Model will drop new data into the scatterplot and try to get the nearest neighbours 
def tune_model(X_train, Y_train):
  param_grid = {
    "n_neighbors": range(1, 8), # 1  to 21 neighbors
    "metric": ["euclidean", "manhattan", "minkowski"],
    "weights": ["uniform", "distance"]
  }
  model = KNeighborsClassifier()
  grid_search = GridSearchCV(model, param_grid, cv=2, n_jobs=-1)
  grid_search.fit(X_train, Y_train)
  return grid_search.best_estimator_

# Going forward, this is the best model 
best_model = tune_model(X_train, Y_train)
scaler = scalar
label_encoder = label_encoder

joblib.dump(best_model, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# Predictions and Evaluate
def evaluate_model(model, X_test, Y_test):
  prediction = model.predict(X_test)
  Y_pred_labels = label_encoder.inverse_transform(prediction)
  Y_test_labels = label_encoder.inverse_transform(Y_test)
  report = classification_report(Y_test_labels, Y_pred_labels, zero_division=0)
  confusion = confusion_matrix(Y_test_labels, Y_pred_labels, labels=label_encoder.classes_)
  return report, confusion

prediction, confusion = evaluate_model(best_model, X_test, Y_test)
print("Prediction:", prediction)
print("Confusion", confusion)

print(data['label'].value_counts())

ConfusionMatrixDisplay.from_estimator(
    best_model,
    X_test,
    Y_test,
    display_labels=label_encoder.classes_,  # These are your pose names
    xticks_rotation='vertical'
)
plt.show()