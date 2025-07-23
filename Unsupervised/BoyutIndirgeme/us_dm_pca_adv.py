import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Yüksek boyutlu veri seti: Wine

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")
print("🔹 Orijinal Veri (İlk 5 satır):")
print(X.head())
print("\n🔹 Hedef Etiketler (İlk 5):")
print(y.head())
# 2. Normalize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

df_scaled = pd.DataFrame(X_scaled, columns=data.feature_names)
print("\n🔹 Standartlaştırılmış Veri (İlk 5 satır):")
print(df_scaled.head())

# 3. PCA uygulaması
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"\n🔹 PCA ile kalan bileşen sayısı: {X_pca.shape[1]}")
print("🔹 Açıklanan Varyans Oranları:", pca.explained_variance_ratio_)

df_pca_full = pd.DataFrame(X_pca)
df_pca_full["target"] = y
print("\n🔹 PCA Sonrası Veri (İlk 5 satır):")
print(df_pca_full.head())

# 4. PCA sonrası RandomForest ile modelleme
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n✅ Doğruluk (PCA sonrası Random Forest):", accuracy_score(y_test, y_pred))

# 5. Yalnızca 2 PCA bileşeni ile görselleştirme için veri indir
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

df_2d = pd.DataFrame(X_2d, columns=["PC1", "PC2"])
print("\n🔹 Görselleştirme için 2D PCA Verisi (İlk 5 satır):")
print(df_2d.head())

# 6. KMeans ile kümeleme
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_2d)

# 7. Görselleştirme
df_vis = df_2d.copy()
df_vis["Cluster"] = clusters
df_vis["Target"] = y

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_vis,
    x="PC1", y="PC2",
    hue="Cluster", style="Target",
    palette="tab10", s=80
)
plt.title("🍷 PCA + KMeans ile Segmentasyon (Wine Veriseti)")
plt.grid(True)
plt.tight_layout()
plt.show()

