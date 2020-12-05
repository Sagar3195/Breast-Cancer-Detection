
## Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

##Loading dataset
df = pd.read_csv("breast_cancer_wisconsin.csv")

print(df.head())

print(df.shape)

##Let's check misssing values in dataset
print(df.isnull().sum())

df.drop(columns='Unnamed: 32',axis= 1 ,inplace= True)

print(df.columns)

print(df.info())

print(df.describe())

df['diagnosis'].value_counts()

df['diagnosis'] = df['diagnosis'].replace({"B":0, "M": 1})

df['diagnosis'].value_counts(normalize = True)

print(df.dtypes)

plt.figure(figsize = (25, 25))
sns.heatmap(df.corr(), annot = True)

### Now splitting dataset into independent variable & dependent variable
X = df.iloc[:, 2:]
y = df.iloc[:,1]

print(X.head())


##Now splitting dataset into training data and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1234)

print(x_train.shape, x_test.shape)

print(y_train.shape, y_test.shape)


##Now we do feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

##Let's use the RandomForestClassifier algorithm

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 10)

classifier.fit(x_train, y_train)

prediction = classifier.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, prediction)
print("Accuracy of the model: ", accuracy)

print(classification_report(y_test, prediction))

print(confusion_matrix(y_test, prediction))

print(X.columns)

cols = ['radius_mean', 'area_mean','compactness_mean','concavity_mean', 'concave points_mean']

X_data = X[cols]

print(X_data.head())

X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size = 0.2, random_state = 1234)

print(X_train.shape, X_test.shape)


model = RandomForestClassifier()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print(accuracy_score(y_test, y_predict))

print(classification_report(y_test, y_predict))

print(confusion_matrix(y_test, y_predict))

import joblib

joblib.dump(model, "cancer_model.pkl")
