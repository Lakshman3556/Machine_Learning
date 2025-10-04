import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- BASELINE MODELS ---------------- #
baseline_lr = LogisticRegression(max_iter=5000)
baseline_lr.fit(X_train_scaled, y_train)
y_pred_base_lr = baseline_lr.predict(X_test_scaled)
acc_base_lr = accuracy_score(y_test, y_pred_base_lr)

baseline_rf = RandomForestClassifier(random_state=42)
baseline_rf.fit(X_train, y_train)
y_pred_base_rf = baseline_rf.predict(X_test)
acc_base_rf = accuracy_score(y_test, y_pred_base_rf)

print("\n--- Baseline Results ---")
print("Baseline Logistic Regression Accuracy:", acc_base_lr)
print("Baseline Random Forest Accuracy:", acc_base_rf)

# ---------------- TUNED MODELS ---------------- #
# Logistic Regression tuning
log_reg = LogisticRegression(max_iter=5000)
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
}
grid_lr = GridSearchCV(log_reg, param_grid_lr, cv=5, scoring='accuracy')
grid_lr.fit(X_train_scaled, y_train)

best_lr = grid_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)

print("\nBest Logistic Regression Params:", grid_lr.best_params_)
print("Tuned Logistic Regression Accuracy:", acc_lr)
print(classification_report(y_test, y_pred_lr))

# Random Forest tuning
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("\nBest Random Forest Params:", grid_rf.best_params_)
print("Tuned Random Forest Accuracy:", acc_rf)
print(classification_report(y_test, y_pred_rf))

# ---------------- PLOTS ---------------- #

# 1. Confusion Matrices (Tuned Models Only)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Logistic Regression Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap="Blues", ax=axes[0])
axes[0].set_title("Logistic Regression (Tuned) Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap="Greens", ax=axes[1])
axes[1].set_title("Random Forest (Tuned) Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# 2. Accuracy Comparison Bar Plot
models = [
    "LogReg (Baseline)", "LogReg (Tuned)",
    "RandomForest (Baseline)", "RandomForest (Tuned)"
]
accuracies = [acc_base_lr, acc_lr, acc_base_rf, acc_rf]

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=accuracies, palette="Set2")
plt.title("Baseline vs Tuned Model Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0.8, 1.0)  # since accuracies are high
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.005, f"{v:.3f}", ha='center', fontweight='bold')
plt.xticks(rotation=20)
plt.show()
