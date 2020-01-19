import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('annual_temp.csv')
a = dataset.iloc[:, 1:2].values
b = dataset.iloc[:, 2:3].values

from  sklearn.model_selection import train_test_split
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.2, random_state = 0)


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(a, b)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(a, b)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
a_poly = poly_reg.fit_transform(a)
poly_reg.fit(a_poly, b)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(a_poly, b)

# Visualising the Linear Regression results
plt.scatter(a, b, color = 'red')
plt.plot(a, lin_reg.predict(a), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Year')
plt.ylabel('Mean Temp')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(a, b, color = 'red')
plt.plot(a, lin_reg_2.predict(poly_reg.fit_transform(a)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Year')
plt.ylabel('Mean Temp')
plt.show()

print('Mean Temperature In 2016 Will Be')
print(regressor.predict([[2016]]))

print('Mean Temperature In 2017 Will Be')
print(regressor.predict([[2017]]))