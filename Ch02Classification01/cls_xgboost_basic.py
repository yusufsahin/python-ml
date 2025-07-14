from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

iris=load_iris()
X, y = iris.data, iris.target

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42)


model=xgb.XGBClassifier(use_label_encoder=False,eval_metric='mlogloss',random_state=42)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

# 5. Değerlendirme
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))