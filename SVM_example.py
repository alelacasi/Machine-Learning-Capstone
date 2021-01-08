from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm
import numpy as np
import pandas as pd

iris = datasets.load_iris()
X = iris.data
print(iris.data)
Y = iris.target
print(iris.target)

classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

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
print('Prediction: ', prediction)
print('Actual:     ', y_test)
print('Accuracy: ', svm_accuracy)

print("--------------------------------Evaluating Random Forest Classification--------------------------------")
print('Prediction: ', y_pred)
print('Actual:     ', y_test)
print('Accuracy: ',y_pred_acc)

# for i in range(len(prediction)):
#     if(prediction[i] != y_test[i]):
#         print('Index: ', i+1)
#         print('Prediction: ', classes[prediction[i]])
#         print('Actual: ', classes[y_test[i]])