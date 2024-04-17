import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('Churn_Modelling.csv')
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

X = data.iloc[:,0:10].values
y = data.iloc[:,10].values

label_X_country_encoder = LabelEncoder()
X[:,1] = label_X_country_encoder.fit_transform(X[:,1])

label_X_gender_encoder = LabelEncoder()
X[:,2] = label_X_gender_encoder.fit_transform(X[:,2])

transform = ColumnTransformer([("countries", OneHotEncoder(), [1])], remainder="passthrough") # 1 is the country column
X = transform.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

print("Пассивно-агрессивный классификатор")
pac = PassiveAggressiveClassifier(max_iter=100)
pac.fit(X_train, y_train)
y_pred = pac.predict(X_test)
accuracy_ = accuracy_score(y_test, y_pred)
print(f'Accuracy Score of Passive Aggresive Scassifier: {round(accuracy_*100,2)}%')

mlp_clf = MLPClassifier(hidden_layer_sizes=(5,2),
                        max_iter = 300,activation = 'relu',
                        solver = 'adam')
mlp_clf = MLPClassifier(hidden_layer_sizes=(150,100,50),
                        max_iter = 300,activation = 'relu',
                        solver = 'adam')

mlp_clf.fit(X_train, y_train)
ypred = mlp_clf.predict(X_test)

accuracy = accuracy_score(y_test, ypred)
print(f'Accuracy : {round(accuracy*100,2)}%')
