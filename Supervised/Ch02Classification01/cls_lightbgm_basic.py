import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Veri setini yükle
iris = load_iris()
X, y = iris.data, iris.target

# 2. Eğitim ve test verilerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Tahmin yap
y_pred = model.predict(X_test)

# 5. Performans değerlendirmesi
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))