import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
from sklearn import tree

#!Random Forest for Regession

# Reading the information from the csv file into the dataset
datasets = pd.read_csv('petrol_consumption.csv')
datasets.head()

# Prepearing the Data for training
#* Data must be divided into attributes and labels
X = datasets.iloc[:,0:4].values
Y = datasets.iloc[:,4].values

#Dividing the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state = 0)

# Feature scaling
#* We need to do this because our values in the dataset are in different scales, some in tens others in thousands
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#Training Phase using Random Forests
#* n_estimators is the number of trees in the random forest
regressor = RandomForestRegressor(n_estimators=20,random_state = 0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

#Evaluating the algorithm
print("--------------------------------Evaluating Regression--------------------------------")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Root Mean Squared Error:', r2_score(y_test, y_pred))

# #!Random Forest for Classification

# Reading the information from the csv file into the dataset
datasets = pd.read_csv('bill_authentication.csv')
datasets.head()

# Prepearing the Data for training
#* Data must be divided into attributes and labels
X = datasets.iloc[:,0:4].values
Y = datasets.iloc[:,4].values

#Dividing the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state = 0)

# Feature scaling
#* We need to do this because our values in the dataset are in different scales, some in tens others in thousands
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#Training Phase using Random Forests
#* n_estimators is the number of trees in the random forest

count = 10
svm_accuracy = 0
y_pred_acc = 0

for i in range(count): 
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

    #SVM
    model = svm.SVC()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    svm_accuracy += accuracy_score(y_test, prediction)

    #Random Forest Model
    classifier = RandomForestClassifier(n_estimators=20,random_state = 0)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    y_pred_acc += accuracy_score(y_test, y_pred)

y_pred_acc = y_pred_acc/count
svm_accuracy = svm_accuracy/count


print("--------------------------------Evaluating SVM--------------------------------")
#print('Prediction: ', prediction)
#print('Actual:     ', y_test)
print('Accuracy: ', svm_accuracy)

print("--------------------------------Evaluating Random Forest Classification--------------------------------")
#print('Prediction: ', y_pred)
#print('Actual:     ', y_test)
print('Accuracy: ',y_pred_acc)