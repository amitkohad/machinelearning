import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('Data/Position_Salaries.csv')

X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# Training on Linear Regression
lr = LinearRegression()
lr.fit(X, y)


#Creating power features for polynomial regression
pf = PolynomialFeatures(degree = 4)
X_poly = pf.fit_transform(X)

#Creating Linear regression for polynomial feature
pr = LinearRegression()
pr.fit(X_poly, y)

#Visualising Linear regression model
plt.scatter(X, y, color = 'blue')
plt.plot(X, lr.predict(X), color='green')
plt.title('Linear regression model')
plt.xlabel('position')
plt.ylabel('Salary')
plt.show()


#Visualising Polynomial Linear regression model
plt.scatter(X, y, color = 'blue')
plt.plot(X, pr.predict(X_poly), color='green')
plt.title('Polynomial Linear regression model')
plt.xlabel('position')
plt.ylabel('Salary')
plt.show()


#Predicting salary with Linear Regression salary
lr_pred = lr.predict([[6.5]])
print(lr_pred)

#Predicting salary with Polynomial regression

pr_pred = pr.predict(pf.fit_transform([[6.5]]))
print(pr_pred)