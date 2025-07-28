# Gaussian Mixture Model (GMM) ile Segmentasyon + BIC/AIC seçimi
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 1. VERİ YÜKLEME
#https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
df = pd.read_csv("data/online_retail_II.csv", encoding='ISO-8859-1')
df.columns = df.columns.str.strip()  # Boşlukları temizle

df.dropna(inplace=True)
df = df[df['Quantity'] > 0]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Total'] = df['Quantity'] * df['Price']  # ✅ DÜZELTME

# 2. RFM HESAPLAMA
snapshot = df['InvoiceDate'].max() + timedelta(days=1)
rfm = df.groupby('Customer ID').agg({  # ✅ DÜZELTME
    'InvoiceDate': lambda x: (snapshot - x.max()).days,
    'Invoice': 'nunique',
    'Total': 'sum'
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# 3. NORMALİZASYON
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm)

# 4. GMM + BIC/AIC TUNING
n_components_range = range(1, 11)
bics, aics = [], []

for k in n_components_range:
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    gmm.fit(X_scaled)
    bics.append(gmm.bic(X_scaled))
    aics.append(gmm.aic(X_scaled))

# 5. GÖRSELLEŞTİRME: BIC / AIC
plt.figure(figsize=(8, 5))
plt.plot(n_components_range, bics, label='BIC', marker='o')
plt.plot(n_components_range, aics, label='AIC', marker='x')
plt.xlabel("n_components")
plt.ylabel("Skor")
plt.title("GMM - BIC / AIC Skorları")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. EN İYİ MODEL (BIC'e göre)
best_k = n_components_range[np.argmin(bics)]
print("En iyi küme sayısı (BIC):", best_k)

final_gmm = GaussianMixture(n_components=best_k, covariance_type='full', random_state=42)
rfm['Cluster'] = final_gmm.fit_predict(X_scaled)

# 7. PCA GÖRSELLEŞTİRME
X_pca = PCA(n_components=2).fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=rfm["Cluster"], cmap="Set2", s=30)
plt.title("Gaussian Mixture Segmentasyonu (BIC ile)")
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.grid(True)
plt.tight_layout()
plt.show()
