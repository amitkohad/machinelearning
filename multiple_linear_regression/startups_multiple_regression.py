#Multiple regression on Startups

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Data/startups.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#Encoding Independent categorical variables
ct = ColumnTransformer(transformers=[('state',OneHotEncoder(),[3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


#split data into training and test data
X_train,X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#building multi linear regression
mlr = LinearRegression()
mlr.fit(X_train,y_train)

#predict test set results
y_pred = mlr.predict(X_test)

# set_printoptions keep decimal pints to precision
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
