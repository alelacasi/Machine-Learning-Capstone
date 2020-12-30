from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test