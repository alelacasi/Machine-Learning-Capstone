import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn import neighbors
import pandas as pd
import numpy as np
from sklearn import tree

#! This example shows all the steps that must be taken in any ML problem
#*The question & required data: Predict max temp in a certain city. Data: Seattle

# Data Acquisition
#* Note: About 80% of the time spent in data analysis is cleaning and retrieving data 
features = pd.read_csv('temperature_data.csv')
features.head(5)
print('The shape of our features is:', features.shape)
# Data Preparation
#* Descriptive statistics for each column
features.describe()

# One Hot Encoding
#* Turns Week days (Mon Tue etc.) into numerical values the mac

features = pd.get_dummies(features)
features.iloc[:,5:].head(5)
# Convert the data into features and targets
#* Labels are the values we want to predict
labels = np.array(features['actual'])
#* Remove the labels from the features
#* axis 1 refers to the columns
features= features.drop('actual', axis = 1)
#* Saving feature names for later use
feature_list = list(features.columns)
#* Convert to numpy array
features = np.array(features)

# Training and testing
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 0)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Establish a baseline for our model to beat
#* We will use historical max temperature averages as baseline
baseline_preds = test_features[:, feature_list.index('average')]
#* Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2) , 'degrees.')

# Here we finally use the Sci-kit learn model
#* Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 0)
#* Train the model on training data
rf.fit(train_features, train_labels)

# Analyze our model prediction relative to the baseline
#* Use the forest's predict method on the test data
predictions = rf.predict(test_features)
#* Calculate the absolute errors
errors = abs(predictions - test_labels)
#* Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

#Determine our model's accuracy
#* Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
#* Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Finally our last goal is to adjust the parameters to make the most accurate results
# We can do this by running and comparing multiple possibilities

# We are basically done, now the rest is more for visualizing and understanding what is going on

# First we will visualize a tree
