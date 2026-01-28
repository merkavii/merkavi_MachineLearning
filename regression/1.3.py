# رگرسیون چند جمله ای

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
# چون نمیخوایم ستون صفرو بگیریم
y = dataset.iloc[:,-1].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X , y)
# آموزش مدل خطی روی کل داده‌ها (نیازی به تقسیم نیست چون داده کم است)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 4)
# ۱) مدل خطی فقط یک خط صاف می‌کشد
# یعنی فقط می‌تواند رابطهٔ خطی بسازد:
# مثل y = a + bX
# اما در دیتاست Position_Salaries، رابطه خطی نیست؛
# حقوق یک‌دفعه در سطح‌های بالاتر خیلی زیاد می‌شود.
# پس یک خط ساده اصلاً نمی‌تواند این رفتار را درست توضیح دهد.

# این مدل کاری می‌کند که مدل به جای یک خط صاف،
# بتواند یک منحنی نرم و منعطف بکشد.
# y = a + bX + cX² + dX³ + eX⁴

# پس این دستور (degree=4) یعنی:

# برای ستون ایکس، ۴ نسخه جدید بساز:

# X¹ (خود مقدار اصلی)
# X² (مربع مقدار)
# X³ (توان سوم)
# X⁴ (توان چهارم)
# این یعنی مدل به جای ۱ ورودی، ۴ ورودی مربوط به powers دریافت می‌کند.
# با این کار مدل می‌تواند یک منحنی بسیار دقیق رسم کند.

# اگر degree خیلی کم باشد (مثلاً 2)، مدل ساده می‌شود و شکل واقعی داده را نمی‌گیرد.
# اگر degree خیلی زیاد باشد (مثلاً 10)، مدل زیادی پیچیده می‌شود و overfitting ایجاد می‌کند.

X_poly = poly_reg.fit_transform(X)
# تبدیل X به نسخه چندجمله‌ای
lin_reg_2 = LinearRegression()
# ساخت مدل خطی جدید که روی داده‌های چندجمله‌ای آموزش می‌بیند
lin_reg_2.fit(X_poly , y)


plt.scatter(X , y , color = 'red')
plt.plot(X , lin_reg.predict(X) , color = 'blue')
plt.title('Truth or Bluff (Linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


plt.scatter(X , y , color = 'red')
plt.plot(X , lin_reg_2.predict(X_poly) , color = 'blue')
plt.title('Truth or Bluff (Polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


print('Linear regression predict : ',lin_reg.predict([[6.5]]))
print('Polynomial regression predict : ',lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))