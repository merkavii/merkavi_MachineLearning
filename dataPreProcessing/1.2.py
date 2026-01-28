import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values
imputer = SimpleImputer(missing_values=np.nan , strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# خب ما میخوایم داده ی کشور رو به سه ستون تبدیل کنیم.چون سه  نوع کشور داریم اینجا.و همچنین ستون آخر رو به 0 یا 1 تبدیل کنیم
# ستون کشور رو میخوایم با رمزنگاری داغ اینطوریش کنیم
# مثلا آلمان : 001 فرانسه :010 اسپانیا : 100
 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers= [('encoder', OneHotEncoder() , [0] )], remainder='passthrough')
# transformers : نوع تحول که همون درحال رمز گزاریست .چه نوع رمزگزاری که ما نوع یک یا همون هات.ایندکس ستون که برای ما ستون کشور
# remainder='passthrough' یعنی بقیه ستون ها دست نخورده باقی بمانند
X = np.array(ct.fit_transform(X))
print(X)
# الان ستون کشور تبدیل به سه ستون عددی شد
print('****************************************************')

# حالا ستون آخر
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# دیگه نمیخواد آرایه نامپای بشه چون این بردار متغیر وابستس
print(f'y variable : {y}')
print('****************************************************')


# حالا تمرین و تست داده ها
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=1)
# یعنی 20 درصدش برای تست بره و همچنین داده هارو رندوم انتخاب کنه


# حالا مقیاس بندی.یعنی هم سطح کردن دیتا ها یا همون نرمالایز کردن
# نرمالایز یعنی بین 0 و 1 و استانداردزیشن یعنی بین -3 و 3
#   البته اینجا بعضی از دیتا های ما بین این رنج هستند و فقط بدتر میشه اگه استاندارد کنیمش اینجا فقط سن و حقوق رو میخوایم استاندارد کنیم
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[: ,3:] = sc.fit_transform(X_train[: ,3:])  # یادگیری + تبدیل
X_test[: ,3:] = sc.transform(X_test[: ,3:]) # فقط تبدیل با همان قواعد
# روی داده‌های آموزش فیت می‌کنیم تا الگو یاد گرفته شود
# روی داده‌های آزمایش فقط ترنسفورم می‌کنیم تا اطلاعاتِ تست وارد آموزش نشود
print(X_train)
print('****************************************************')
print(X_test)
