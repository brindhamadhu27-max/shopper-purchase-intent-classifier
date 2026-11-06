"""
Train five classifiers for the Online Shoppers Purchasing Intention dataset.

Run inside BITS Virtual Lab:
    pip install -r requirements.txt
    python train_models.py

This script downloads the public UCI dataset, creates a stratified hold-out test
set, trains five pipelines, saves the fitted pipelines, and writes evaluation
results plus test_data.csv.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Revenue"

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# 1. Download dataset from the UCI Machine Learning Repository.
uci_dataset = fetch_ucirepo(id=468)
X = uci_dataset.data.features.copy()
y_frame = uci_dataset.data.targets.copy()

# Normalize target name and ensure a one-dimensional boolean target.
y = y_frame.iloc[:, 0].astype(bool)
X.columns = [str(c).strip() for c in X.columns]
data = X.copy()
data[TARGET] = y.values

# Save a local copy of the source dataset for reproducibility.
data.to_csv(ROOT / "online_shoppers_intention.csv", index=False)

# 2. Identify categorical and numerical input columns.
# Identifier-like numeric fields are treated as categorical.
categorical_candidates = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend"
]
categorical_cols = [c for c in categorical_candidates if c in X.columns]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

# 3. Split before fitting preprocessing to prevent data leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Dense one-hot output supports GaussianNB as well as the other estimators.
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# 4. Define all five required classifiers.
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE)
}

results = []
for display_name, estimator in classifiers.items():
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", estimator)
    ])
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    positive_index = list(pipeline.classes_).index(True)
    positive_probabilities = probabilities[:, positive_index]

    row = {
        "ML Model Name": display_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, positive_probabilities),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions)
    }
    results.append(row)

    filename = display_name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipeline, MODEL_DIR / filename)
    print(f"Saved {display_name}: {filename}")

# 5. Save metrics and the untouched test rows, including the target.
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

test_data = X_test.copy()
test_data[TARGET] = y_test.values
test_data.to_csv(ROOT / "test_data.csv", index=False)

print("\nDataset shape:", data.shape)
print("Input feature count:", X.shape[1])
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("\nTarget distribution in full dataset:")
print(y.value_counts())
print("\nEvaluation results:")
print(results_df.round(4).to_string(index=False))
print("\nCreated test_data.csv and all five fitted model pipelines.")
