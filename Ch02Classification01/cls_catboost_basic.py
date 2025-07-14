import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Veri setini yükle
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# 2. Kullanılacak kolonları seç
df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]]

# 3. Eksik verileri düşür (opsiyonel olarak imputing yapılabilir)
df.dropna(inplace=True)

# 4. Kategorik sütunları belirt
cat_features = ["Pclass", "Sex", "Embarked"]

# 5. Özellikler ve hedefi ayır
X = df.drop("Survived", axis=1)
y = df["Survived"]

# 6. Eğitim/test bölmesi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify → sınıf dengesini korur
)

# 7. CatBoost modelini oluştur ve eğit
model = CatBoostClassifier(
    verbose=0,         # Eğitim loglarını kapatır
    random_state=42,
    cat_features=cat_features,
    eval_metric="Accuracy"
)
model.fit(X_train, y_train)

# 8. Tahmin
y_pred = model.predict(X_test)

# 9. Değerlendirme
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
