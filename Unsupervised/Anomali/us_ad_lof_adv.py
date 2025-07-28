import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.decomposition import PCA

# Veri setini yükle
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Veriyi ölçekle
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Eğitim/test ayır
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)

# LOF modelini eğit (novelty=True)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
lof.fit(X_train)

# Test verisinde tahmin yap
y_scores = lof.decision_function(X_test)
y_pred = lof.predict(X_test)
y_pred_binary = np.where(y_pred == -1, 1, 0)  # Anomali: 1, Normal: 0
# ROC & PR eğrileri
fpr, tpr, _ = roc_curve(y_test, y_pred_binary)
precision, recall, _ = precision_recall_curve(y_test, y_pred_binary)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve")
plt.legend()

plt.tight_layout()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))

# ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_binary)
print(f"\nROC AUC Skoru: {roc_auc:.4f}")

# PCA ile görselleştirme
pca = PCA(n_components=2)
X_vis = pca.fit_transform(X_test)

plt.figure(figsize=(8,6))
plt.scatter(X_vis[y_pred_binary == 0][:, 0], X_vis[y_pred_binary == 0][:, 1], label="Normal", alpha=0.5)
plt.scatter(X_vis[y_pred_binary == 1][:, 0], X_vis[y_pred_binary == 1][:, 1], label="Anomaly", alpha=0.5, color="red")
plt.title("PCA ile 2D Görselleştirme (LOF)")
plt.legend()
plt.show()