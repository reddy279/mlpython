import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("/Users/shashi_sudi/Shashi/MachineLearning/Simple_Linear_Regression/Salary_Data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

print('Idependent Variables: ', X, 'Dependent Variables: ', y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=1/3, random_state=0)

print(X_train, X_test,y_train,y_test)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)

prediction = regressor.predict(X_test)
print(prediction)
# plt.interactive(True)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Experiance (Training Set)')
plt.xlabel('Salary')
plt.ylabel('Years of Experiance')
plt.show