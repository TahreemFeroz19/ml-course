
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('global_co2.csv')
a = dataset.iloc[:, :1].values
b = dataset.iloc[:, 1:2]

from sklearn.model_selection import train_test_split
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.2, random_state = 0)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
a_poly = poly_reg.fit_transform(a)
poly_reg.fit(a_poly, b)
lin_reg.fit(a_poly, b)

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
a_grid = np.arange(min(a), max(a), 0.1)
a_grid = a_grid.reshape((len(a_grid), 1))
plt.scatter(a, b, color = 'red')
plt.plot(a_grid, lin_reg.predict(poly_reg.fit_transform(a_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Year')
plt.ylabel('Total CO2 Produced')
plt.show()

print('CO2 Prodeced In The Year 2011 Will Be')
print(lin_reg.predict(poly_reg.fit_transform([[2011]])))

print('CO2 Prodeced In The Year 2012 Will Be')
print(lin_reg.predict(poly_reg.fit_transform([[2012]])))

print('CO2 Prodeced In The Year 2013 Will Be')
print(lin_reg.predict(poly_reg.fit_transform([[2013]])))