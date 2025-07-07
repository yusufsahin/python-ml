import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB

from cls_logistic_adv import cv_scores

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', names=['label', 'message'])

#Encode labels
df['label_num']=df.label.map({'ham': 0, 'spam': 1})

#Feature Extraction

vectorizer= TfidfVectorizer(stop_words='english')
X=vectorizer.fit_transform(df['message'])
y=df['label_num']

# Split the dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Model training
model=MultinomialNB()
model.fit(X_train,y_train)
# Prediction on the test set

y_pred=model.predict(X_test)



#Evaluate the model
print("Accuracy:",accuracy_score(y_test,y_pred))
#Confusion Matrix
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-validated Accuracy: %.2f%%" % (cv_scores.mean() * 100))