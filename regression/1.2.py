# رگرسیون خطی چند گانه
# هدف مدل = پیش‌بینی Profit (سود شرکت)
# تمام ورودی‌ها (ستون‌های دیگر) فقط ویژگی هستند که به مدل کمک می‌کنند سود را بهتر حدس بزند:
# R&D Spend → چقدر خرج تحقیق و توسعه شده
# Administration → هزینه‌های اداری
# Marketing Spend → هزینهٔ بازاریابی
# State → ایالت (اثر مکانی / بازاری)
# Profit → خروجی / هدف / آنچه باید پیش‌بینی شود


# کدام بخش هزینه‌ها بیشترین تأثیر را روی سود دارد؟

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values


# تبدیل ستون کشور به عدد
ct = ColumnTransformer(transformers= [('encoder', OneHotEncoder() , [3] )], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=0)

# نیازی به مقیاس بندی نیست


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , y_train)

# یه نمودار درست میکنیم با دو بردار
# خب 20 درصد پنجاه میشه 10 همون تست مدل
# یه بردار قراره اون سود واقعی اون 10 تا دیتارو نشون بده
# یکیش پیش بینی مدل

y_pred = regressor.predict(X_test)
# بردار پیش بینی مدل
np.set_printoptions(precision= 2)
# همرو تا دو رقم اعشار نمایش میده
# خب حالا از فانکشن کانکتونیت برای هماهنگی دو بردار یا حتی آرایه میشه استفاده کرد
print(np.concatenate((y_pred.reshape(len(y_pred),1) ,  y_test.reshape(len(y_test),1)),1))
# اول دو بردار را تبدیل به ستون می‌کنیم (reshape)
# بعد با concatenate این دو ستون را از سمتِ پهلو به هم می‌چسبانیم (axis=1)
# ستون اول سود پیش‌بینی‌شده است و ستون دوم سود واقعی
# با چاپ این دو ستون کنار هم می‌توانیم میزان دقت مدل را ببینیم
