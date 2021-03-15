import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # pylint: disable=import-error
from sklearn.ensemble import RandomForestRegressor # pylint: disable=import-error
from sklearn.ensemble import RandomForestClassifier # pylint: disable=import-error
from sklearn.model_selection import train_test_split # pylint: disable=import-error
from sklearn.preprocessing import StandardScaler # pylint: disable=import-error
from sklearn import metrics # pylint: disable=import-error
from sklearn import svm# pylint: disable=import-error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score# pylint: disable=import-error
from sklearn.utils import shuffle# pylint: disable=import-error
import pandas as pd
import numpy as np
from sklearn import tree# pylint: disable=import-error

""" #!Random Forest for Regession

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
print('Root Mean Squared Error:', r2_score(y_test, y_pred)) """

# #!Random Forest for Classification

# Reading the information from the csv file into the dataset
datasets = pd.read_csv('bill_authentication.csv')
#print(datasets.head())

# Prepearing the Data for training
#* Data must be divided into attributes and labels
X = datasets.iloc[:,0:4].values
Y = datasets.iloc[:,4].values

dataSizer = 100
svm_acc_array = []
rf_acc_array = []
data_point_axis = []

for i in range(5, dataSizer, 1): #code crashes at 5 due to lack of data
    
    X, Y = shuffle(X, Y)
    X = X[:dataSizer,:]
    Y = Y[:dataSizer]

    #Dividing the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)#, random_state = 0)

    # Feature scaling
    #* We need to do this because our values in the dataset are in different scales, some in tens others in thousands
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)

    #Training Phase using Random Forests
    #* n_estimators is the number of trees in the random forest

    count = 10
    svm_accuracy = 0
    rf_accuracy = 0


    for j in range(count): 
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

        #SVM
        model = svm.SVC()
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        svm_accuracy += accuracy_score(y_test, prediction)

        #Random Forest Model
        classifier = RandomForestClassifier(n_estimators=20)#,random_state = 0)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        rf_accuracy += accuracy_score(y_test, y_pred)

    rf_accuracy = rf_accuracy/count
    svm_accuracy = svm_accuracy/count

    #print(f"\n\nData Size: {dataSizer}")
    #print("--------------------------------Evaluating SVM--------------------------------")
    #print('Prediction: ', prediction)
    #print('Actual:     ', y_test)
    #print('Accuracy: ', svm_accuracy)
    svm_acc_array.append(svm_accuracy*100)

    #print("--------------------------------Evaluating Random Forest Classification--------------------------------")
    #print('Prediction: ', y_pred)
    #print('Actual:     ', y_test)
    #print('Accuracy: ',rf_accuracy)
    rf_acc_array.append(rf_accuracy*100)
    
    data_point_axis.append(i)

""" print(data_point_axis)
print(rf_acc_array)
print(svm_acc_array) """

plt.plot(data_point_axis, svm_acc_array, color='#adad3b', label = 'SVM')

plt.plot(data_point_axis, rf_acc_array, color='#5a7d9a', label = 'RF')

plt.xlabel("Amount of Data values")
plt.ylabel("Accuracy %")
plt.title("Acc variability over 100 Dataset sizes")

plt.legend()

plt.show()