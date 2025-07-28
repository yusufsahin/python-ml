import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# Veri seti
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # 1: benign (normal), 0: malignant (anomaly)

# Sadece benign verileri eğitim için al
X_train = X[y == 1]
X_test = X
y_test = y.apply(lambda x: 1 if x == 1 else -1)  # 1: normal, -1: anomaly


# Ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = OneClassSVM(kernel="rbf", gamma=0.01, nu=0.05)
model.fit(X_train_scaled)
# Tahmin
y_pred = model.predict(X_test_scaled)

# Sonuçlar
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))