# 04_hyperparameter_tuning.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("breast_cancer_dataset.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Pipeline + Hyperparameter tuning for RandomForest ---
rf_pipeline = Pipeline([
    ("scaler", StandardScaler()), 
    ("clf", RandomForestClassifier(random_state=42))
])

rf_params = {
    "clf__n_estimators": [50, 100, 200],
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_split": [2, 5, 10],
}

rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=5, scoring="accuracy", n_jobs=-1)
rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)

print("\n Random Forest Best Params:", rf_grid.best_params_)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))


# --- Pipeline + Hyperparameter tuning for Logistic Regression ---
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()), 
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

lr_params = {
    "clf__C": [0.1, 1, 10],
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs", "saga"],
}

lr_grid = GridSearchCV(lr_pipeline, lr_params, cv=5, scoring="accuracy", n_jobs=-1)
lr_grid.fit(X_train, y_train)

lr_best = lr_grid.best_estimator_
y_pred_lr = lr_best.predict(X_test)

print("\n Logistic Regression Best Params:", lr_grid.best_params_)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
