#!Random Forest for Classification

# # Reading the information from the csv file into the dataset
# datasets = pd.read_csv('bill_authentication.csv')
# datasets.head()

# # Prepearing the Data for training
# #* Data must be divided into attributes and labels
# X = datasets.iloc[:,0:4].values
# Y = datasets.iloc[:,4].values

# #Dividing the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state = 0)

# # Feature scaling
# #* We need to do this because our values in the dataset are in different scales, some in tens others in thousands
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.fit_transform(x_test)

# #Training Phase using Random Forests
# #* n_estimators is the number of trees in the random forest
# classifier = RandomForestClassifier(n_estimators=20,random_state = 0)
# classifier.fit(x_train, y_train)
# y_pred = classifier.predict(x_test)

# #Evaluating the algorithm
# print("--------------------------------Evaluating Classification--------------------------------")
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# print('Accuracy: ',accuracy_score(y_test, y_pred))