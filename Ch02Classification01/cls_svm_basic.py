from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#1.Veri setini yükle
iris=load_iris()

X,y=iris.data,iris.target
#2. Eğitim ve test setlerine ayır
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#3. Modeli oluştur - Support Vector Classifier (SVC)
model=SVC(kernel='linear',C=1.0 ,random_state=42)
model.fit(X_train,y_train)

#4. Tahmin yap
y_pred=model.predict(X_test)
#5. Sonuçları değerlendir
print("Accuracy:", accuracy_score(y_test, y_pred))
# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
