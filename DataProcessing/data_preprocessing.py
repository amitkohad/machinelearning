#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import  OneHotEncoder, LabelEncoder
from sklearn.compose import  ColumnTransformer

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

print (y)
