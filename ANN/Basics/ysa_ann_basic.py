from sklearn.datasets import load_iris
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1.Veri Yükleme
iris= load_iris()
X= iris.data
y= pd.get_dummies(iris.target).values
#2. Eğitim ve Test Setlerine Ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#3. Modeli Oluşturma

model= Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#4. Modeli Eğitme
model.fit(X_train, y_train,validation_data=(X_test, y_test),  epochs=50, batch_size=8)

#5. Modeli Değerlendirme
y_pred = model.predict(X_test)
y_pred_classes =np.argmax(y_pred,axis=1)
y_true_classes =np.argmax(y_test,axis=1)
print(classification_report(y_true_classes, y_pred_classes, target_names=iris.target_names))