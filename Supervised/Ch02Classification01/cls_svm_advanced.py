import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import matplotlib.pyplot as plt

import seaborn as sns

#1.Veri setini yükle
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)

#2.Kolon seçimi ve temizleme
df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
df.dropna(inplace=True)

#3.Label Encoding
df["Sex"]=LabelEncoder().fit_transform(df["Sex"])
df["Embarked"]=LabelEncoder().fit_transform(df["Embarked"])

X=df.drop("Survived", axis=1)
y=df["Survived"]

#4.Eğitim/Test bölmesi
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Pipeline  : StandartScaler + SVM/SVC

pipeline=Pipeline([('scaler',StandardScaler()),('svm',SVC(probability=True))])

#Hiperparametre gridi
param_grid={
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 0.1, 1],
    'svm__kernel': ['rbf', 'linear'],
}

# GridSearchCV ile model optimizasyonu

grid=GridSearchCV(pipeline,param_grid,cv=5,scoring='accuracy',n_jobs=-1)
grid.fit(X_train,y_train)

# 8. En iyi model ve skor
print("Best Parameters:", grid.best_params_)
print("Best Cross-Validation Score:", grid.best_score_)
# 9. Test seti değerlendirmesi
y_pred=grid.predict(X_test)
y_proba=grid.predict_proba(X_test)[:, 1]

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#10.Confusion Matrix
cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
plt.title("Confusion Matrix")
plt.show()

# 11. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label="AUC = %.2f" % roc_auc_score(y_test, y_proba))
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()