
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.decomposition import PCA

#https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download
# 1. Veri Setini Yükle
df = pd.read_csv('data/creditcard.csv')


# 2. EDA
print("Veri Boyutu:", df.shape)
print("Anomali Oranı (%):", round(df['Class'].value_counts()[1] / len(df) * 100, 4))
print("\nSınıf Dağılımı:")
print(df['Class'].value_counts())

# 3. Özellikleri ve hedefi ayır
X = df.drop(['Class', 'Time'], axis=1)  # Time çıkarıldı
y_true = df['Class']

# 4. Ölçekleme (Amount hariç diğerleri zaten PCA uygulanmış)
X['Amount'] = StandardScaler().fit_transform(X[['Amount']])


# 5. Model: Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.00172, random_state=42)
model.fit(X)
y_pred = model.predict(X)  # -1 = anomaly, 1 = normal
y_pred_binary = np.where(y_pred == -1, 1, 0)


# 6. Metrikler
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_true, y_pred_binary))

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred_binary))

roc_auc = roc_auc_score(y_true, y_pred_binary)
print(f"ROC AUC: {roc_auc:.4f}")

# 7. Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_true, y_pred_binary)
pr_auc = auc(recall, precision)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 8. PCA ile 2D Görselleştirme
X_reduced = PCA(n_components=2).fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred_binary, cmap='coolwarm', alpha=0.6, edgecolors='k')
plt.title("Isolation Forest – Credit Card Fraud PCA 2D Görselleştirme")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.show()

# 9. Anomali Skorları
scores = model.decision_function(X)

plt.figure(figsize=(8, 4))
sns.histplot(scores, bins=100, kde=True)
plt.title("Anomali Skorları Dağılımı")
plt.xlabel("Skor (daha düşük = daha anomalik)")
plt.tight_layout()
plt.show()