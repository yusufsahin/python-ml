
from sklearn.datasets import load_iris
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Veri setini yükle
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
X = X[["sepal length (cm)", "sepal width (cm)"]]  # sadece 2 özellik al
# LOF modeli
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = lof.fit_predict(X)
outlier_score = -lof.negative_outlier_factor_
# Görselleştir
plt.figure(figsize=(8, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, cmap="coolwarm", edgecolors="k")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("LOF ile Iris Veri Setinde Anomali Tespiti")
plt.grid(True)
plt.show()