from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np
import pandas as pd

iris = datasets.load_iris()
X = iris.data
Y = iris.target

classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

model = svm.SVC()
model.fit(x_train, y_train)

print(model)

prediction = model.predict(x_test)
accuracy = accuracy_score(y_test, prediction)

print('Prediction: ', prediction)
print('Actual:     ', y_test)
print('Accuracy: ', accuracy)

for i in range(len(prediction)):
    if(prediction[i] != y_test[i]):
        print('Index: ', i+1)
        print('Prediction: ', classes[prediction[i]])
        print('Actual: ', classes[y_test[i]])