# Random forest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
# چون نمیخوایم ستون صفرو بگیریم
y = dataset.iloc[:,-1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(random_state=0, n_estimators=100)
# n_estimators = تعداد درختان
regressor.fit(X,y)

y_pred = regressor.predict([[6.5]])
print(y_pred)


x_grid = np.arange(min(X) , max(X) , 0.1)
x_grid = x_grid.reshape(len(x_grid), 1) 
plt.scatter(X , y , color = 'red')
plt.plot(x_grid , regressor.predict(x_grid) , color = 'blue')
plt.title('Truth or Bluff (Random forest regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
