import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import seaborn as sns

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X["target"] = y
X_normal = X[X["target"] == 0].drop("target", axis=1)
X_all = X.drop("target", axis=1)
y_all = y.apply(lambda val: 1 if val == 0 else -1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_normal)
X_all_scaled = scaler.transform(X_all)

model = OneClassSVM(kernel="rbf", gamma=0.01, nu=0.05)
model.fit(X_train_scaled)
y_pred = model.predict(X_all_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_all, y_pred))
print("\nClassification Report:")
print(classification_report(y_all, y_pred))
print(f"ROC AUC: {roc_auc_score((y_all == -1).astype(int), (y_pred == -1).astype(int)):.4f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_all_scaled)
df_viz = pd.DataFrame({
    "PCA1": X_pca[:, 0],
    "PCA2": X_pca[:, 1],
    "Gerçek": y_all.map({1: "Normal", -1: "Anomali"}),
    "Tahmin": pd.Series(y_pred).map({1: "Normal", -1: "Anomali"})
})

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(data=df_viz, x="PCA1", y="PCA2", hue="Gerçek", palette="viridis")
plt.title("Gerçek Etiketler (PCA ile)")

plt.subplot(1, 2, 2)
sns.scatterplot(data=df_viz, x="PCA1", y="PCA2", hue="Tahmin", palette="coolwarm")
plt.title("Model Tahminleri (PCA ile)")

plt.tight_layout()
plt.savefig("oneclasssvm_wine_visualization.png")