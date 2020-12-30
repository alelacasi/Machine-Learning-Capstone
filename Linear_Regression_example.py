from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
import pandas as pd

boston = datasets.load_boston()

X = boston.data
Y = boston.target

linear_regressor = linear_model.LinearRegression()

plt.scatter(X.T[5], Y)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

#training data
model = linear_regressor.fit(x_train, y_train)
prediction = model.predict(x_test)

print("Prediction: ", prediction)
print("R^2 value: ", linear_regressor.score(X,Y))
print("Coefficient: ", linear_regressor.coef_)
print("Intercept: ", linear_regressor.intercept_)


