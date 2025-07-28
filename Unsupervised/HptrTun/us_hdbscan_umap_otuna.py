# HDBSCAN + UMAP + Optuna ile RFM Segmentasyonu
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
import optuna
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. VERİ SETİ YÜKLEME
# 1. VERİ SETİ YÜKLEME
df = pd.read_csv("data/online_retail_II.csv", encoding='ISO-8859-1')
df.columns = df.columns.str.strip()  # kolon isimlerindeki boşlukları temizle

df.dropna(inplace=True)
df = df[df['Quantity'] > 0]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# ✔️ 'Total' hesapla (Price üzerinden)
df['Total'] = df['Quantity'] * df['Price']

# 2. RFM OLUŞTURMA
snapshot = df['InvoiceDate'].max() + timedelta(days=1)
rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (snapshot - x.max()).days,
    'Invoice': 'nunique',
    'Total': 'sum'
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']



# 3. NORMALİZASYON
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm)


# 4. UMAP İLE BOYUT İNDİRME
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=5, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

# 5. OPTUNA İLE HDBSCAN TUNING
def objective(trial):
    min_cluster_size = trial.suggest_int("min_cluster_size", 5, 50)
    min_samples = trial.suggest_int("min_samples", 1, 20)
    cluster_selection_method = trial.suggest_categorical("cluster_selection_method", ["eom", "leaf"])

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        metric="euclidean"
    )

    labels = model.fit_predict(X_umap)
    if len(set(labels)) <= 1 or len(set(labels)) == len(labels):
        return -1.0
    return silhouette_score(X_umap, labels)

print("Optuna çalışıyor...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print("En iyi parametreler:", study.best_params)

# 6. EN İYİ MODELLE CLUSTER ETİKETLERİ
best_model = hdbscan.HDBSCAN(**study.best_params)
labels = best_model.fit_predict(X_umap)
rfm["Cluster"] = labels

# 7. GÖRSELLEŞTİRME
X_pca = PCA(n_components=2).fit_transform(X_umap)
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="Spectral", s=20)
plt.title("UMAP + HDBSCAN Segmentasyonu (Optuna ile)")
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.grid(True)
plt.tight_layout()
plt.show()

