#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import  OneHotEncoder, LabelEncoder
from sklearn.compose import  ColumnTransformer
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler

#load dataset
dataset = pd.read_csv('Data/data_preprocessing.csv')

#Independent variable
X = dataset.iloc[:, :-1].values

#dependent variable
y = dataset.iloc[:, 3].values

#print(X)
#Taking care of missing data

#using simple mean method
#imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
#imputer = imputer.fit(X[:, 1:3])
#X[:, 1:3] = imputer.transform(X[:, 1:3])

#print('after')
#print(X)

# Using Nearest neighbour method
#print('knn')

knnimputer = KNNImputer(n_neighbors=2, weights="uniform")
knnimputer = knnimputer.fit(X[:, 1:3])
X[:, 1:3] = knnimputer.transform(X[:, 1:3])
#print(X)

columntransformer = ColumnTransformer([('country',OneHotEncoder(),[0])],remainder='passthrough')
X = columntransformer.fit_transform(X)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting dataset into training set and test test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#Feature scaling
standarscaler = StandardScaler()
X_train[:,3:5] = standarscaler.fit_transform(X_train[:,3:5])
X_test[:,3:5] = standarscaler.fit_transform(X_test[:,3:5])

