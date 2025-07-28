import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download
# Veri seti
df = pd.read_csv("data/creditcard.csv")

# Özellik ve hedef ayır
X = df.drop(["Class", "Time"], axis=1)
y = df["Class"]
X["Amount"] = StandardScaler().fit_transform(X[["Amount"]])

# Eğitim/test ayır
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


# Optuna objective
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_samples = trial.suggest_float("max_samples", 0.5, 1.0)
    contamination = trial.suggest_float("contamination", 0.001, 0.02)
    max_features = trial.suggest_float("max_features", 0.5, 1.0)

    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)

    y_pred = model.predict(X_val)
    y_pred_binary = np.where(y_pred == -1, 1, 0)
    score = roc_auc_score(y_val, y_pred_binary)
    return score


# Optuna çalıştır
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print("En iyi parametreler:")
print(study.best_params)
print(f"En iyi ROC AUC: {study.best_value:.4f}")

# En iyi model ile yeniden eğit
best_params = study.best_params
model = IsolationForest(
    **best_params,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train)

y_pred = model.predict(X_val)
y_pred_binary = np.where(y_pred == -1, 1, 0)
scores = model.decision_function(X_val)


# PCA ile görselleştirme
pca = PCA(n_components=2)
X_vis = pca.fit_transform(X_val)

plt.figure(figsize=(10,6))
plt.scatter(X_vis[y_pred_binary == 0][:, 0], X_vis[y_pred_binary == 0][:, 1], alpha=0.5, label="Normal")
plt.scatter(X_vis[y_pred_binary == 1][:, 0], X_vis[y_pred_binary == 1][:, 1], alpha=0.5, label="Anomaly", color='r')
plt.title("PCA ile 2D Isolation Forest Tahmini")
plt.legend()
plt.show()

# ROC ve PR eğrileri
fpr, tpr, _ = roc_curve(y_val, y_pred_binary)
precision, recall, _ = precision_recall_curve(y_val, y_pred_binary)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall, precision, label="PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

plt.tight_layout()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_val, y_pred_binary)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_val, y_pred_binary))