import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree

iris = load_iris()
X = iris.data
Y = iris.target
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0)

clf = DecisionTreeClassifier(max_depth = 2, random_state = 0)
clf.fit(x_train, y_train)
clf.predict(x_test)

#Visualize with sklearn
tree.plot_tree(clf)
plt.show()

#Visualize with matplotlib
# Feature Names and Class Names
feat_names  = ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cls_names = ['setosa', 'versicolor', 'virginica']

fig2, ax2 = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=100)

tree.plot_tree(clf,feature_names = feat_names, class_names = cls_names, filled = True)

fig2.savefig('Decision_Tree.png')