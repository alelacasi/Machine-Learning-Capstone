from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

data = pd.read_csv('car.data')
#print(data.head())

#We define the important labels of the data that will be used
X = data [[
    'buying',
    'maint',
    'safety'
]].values
Y = data[['class']]

#Convert the strings in the data to values with LabelEncoder
le = LabelEncoder()
for i in range(len(X[0])):
    X[:,i] = le.fit_transform(X[:,i])
#Convert the strings in the data to values with Mapping
label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3,
}
Y['class'] = Y['class'].map(label_mapping)
Y = np.array(Y)

#Creating our KNN model

knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

knn.fit(x_train, y_train)

prediction = knn.predict(x_test)

accuracy = metrics.accuracy_score(y_test, prediction)
#print('Prediction: ', prediction)
#print('Actual: ', y_test)
print('Accuracy: ', accuracy)

# for i in range(10):
#     print('Prediction: ', prediction[i])
#     print('Actual: ', y_test[i])

