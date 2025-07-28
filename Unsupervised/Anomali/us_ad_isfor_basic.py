
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd


# 1. Veri setini yükle
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)

# 2. Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA ile 2 boyuta indir (görselleştirme için)
X_pca = PCA(n_components=2).fit_transform(X_scaled)

# 4. Isolation Forest modeli
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_scaled)
preds = model.predict(X_scaled)  # -1: anomali, 1: normal

# 5. Görselleştir
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=preds, cmap='coolwarm', edgecolors='k', alpha=0.7)
plt.title("Isolation Forest - Breast Cancer Veri Seti")
plt.xlabel("PCA Bileşeni 1")
plt.ylabel("PCA Bileşeni 2")
plt.grid(True)
plt.tight_layout()
plt.show()