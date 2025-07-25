from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#Load the iris dataset
iris=load_iris()
X= iris.data
y= iris.target

#Split the dataset into training and testing sets

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Model training

model=GaussianNB()
model.fit(X_train,y_train)

#Prediction on the test set
y_pred=model.predict(X_test)

#Evaluate the model
print("Accuracy:",accuracy_score(y_test,y_pred))
#Confusion Matrix
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred,target_names=iris.target_names))



