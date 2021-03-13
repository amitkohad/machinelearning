import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Read dataset
dataset = pd.read_csv('Data/Salary_Data.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 1]

#Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8)

# Create linear regression model
lr = LinearRegression()
lr.fit(X_train,y_train)

# Predict using linear regression

y_pred = lr.predict(X_test)

# Visualizing the linear regression
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.title('Salary Prediction')
plt.xlabel = 'Year of experience'
plt.ylabel = 'Salary'
plt.show()

plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, lr.predict(X_train), color = 'orange')
plt.title('Salary Prediction')
plt.xlabel = 'Year of experience'
plt.ylabel = 'Salary'
plt.show()