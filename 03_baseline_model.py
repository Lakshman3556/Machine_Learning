# 03_baseline_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset from CSV (created in step 2)
df = pd.read_csv("breast_cancer_dataset.csv")

# Features & target
X = df.drop("target", axis=1)
y = df["target"]

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Baseline model: Random Forest (default hyperparameters)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
print("✅ Baseline Random Forest Accuracy:", round(acc * 100, 2), "%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
