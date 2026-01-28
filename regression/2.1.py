# SVR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
# چون نمیخوایم ستون صفرو بگیریم
y = dataset.iloc[:,-1].values
print(f'X : {X}')
print(f'y : {y}')

# ایکس دو بعدیه اما وای نه . باید وای رو دو بعدی کنیم چون برای مقیاس بندی ارایه دو بعدی باید داد
y = y.reshape(len(y),1)
# row = len(y), column = 1

# مقیاس بندی
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X) 
y = sc_y.fit_transform(y) 
print(f'X : {X}')
print(f'y : {y}')


from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X , y)

print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1 , 1)))

# sc_y.inverse_transform(...)
# «این عدد اسکیل‌شده را برگردان به واحد واقعی حقوق.»

# مثلاً:
# 0.42  →  170000

plt.scatter(sc_x.inverse_transform(X) , sc_y.inverse_transform(y) , color = 'red')
plt.plot(sc_x.inverse_transform(X) , sc_y.inverse_transform(regressor.predict(X).reshape(-1 , 1)) , color = 'blue')
# inverse_transform
# پس برای نمایش انسانی و قابل فهم، باید برگردانیم به مقیاس واقعی.
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# SVR خروجی را یک‌بعدی می‌خواهد
# ولی StandardScaler دو‌بعدی
# برای همین این reshape‌ها لازم‌اند.