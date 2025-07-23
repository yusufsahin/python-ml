import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# 1. Veri: Wine (13 özellik)
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

# Orijinal veriyi göster
df_original = pd.DataFrame(X, columns=feature_names)
df_original["target"] = y
print("🔹 Orijinal Veri (İlk 5 satır):")
print(df_original.head())

# 2. Normalize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
print("\n🔹 Standartlaştırılmış Veri (İlk 5 satır):")
print(df_scaled.head())

# 3. (İsteğe Bağlı) PCA ile önce boyut indir (13 → 6)
pca = PCA(n_components=6)
X_reduced = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_reduced, columns=[f"PC{i+1}" for i in range(X_reduced.shape[1])])
print(f"\n🔹 PCA ile İndirgenmiş Veri ({X_reduced.shape[1]} bileşen, İlk 5 satır):")
print(df_pca.head())


# 4. t-SNE uygula
tsne = TSNE(n_components=2, perplexity=40, learning_rate=100, max_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_reduced)

df_tsne = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])
df_tsne["target"] = y
print("\n🔹 t-SNE Sonuçları (İlk 5 satır):")
print(df_tsne.head())


# 5. Görselleştir
plt.figure(figsize=(10, 6))
for i, label in enumerate(target_names):
    subset = df_tsne[df_tsne["target"] == i]
    plt.scatter(subset["TSNE1"], subset["TSNE2"], label=label)
plt.title("t-SNE + PCA (Wine Dataset)")
plt.xlabel("TSNE1")
plt.ylabel("TSNE2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
