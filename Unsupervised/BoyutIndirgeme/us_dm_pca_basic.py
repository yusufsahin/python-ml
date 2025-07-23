import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Iris veri setini yükle
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 2. Orijinal veriyi göster
df_original = pd.DataFrame(X, columns=feature_names)
print("🔹 Orijinal Veri (İlk 5 Satır):")
print(df_original.head())

# 3. Veriyi standartlaştır
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Standartlaştırılmış veriyi göster
df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
print("\n🔹 StandardScaler Sonrası Veri (İlk 5 Satır):")
print(df_scaled.head())

# 5. PCA ile 2 boyuta indir
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 6. PCA sonucunu göster
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["target"] = y
print("\n🔹 PCA Sonrası Veri (İlk 5 Satır):")
print(df_pca.head())

# 7. Görselleştirme
colors = ["red", "green", "blue"]
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(target_names):
    plt.scatter(
        df_pca[df_pca["target"] == i]["PC1"],
        df_pca[df_pca["target"] == i]["PC2"],
        label=target_name,
        color=colors[i]
    )
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Iris Dataset")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
