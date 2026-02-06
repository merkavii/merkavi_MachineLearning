import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split



dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:-1].values
# ستون های ای دی و نام خانوادگی و شناسه رو نمیگیریم چون بدرد نمیخورن
y = dataset.iloc[:,-1].values

# تبدیل ستون جنسیت به اعداد
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])
# ستون ایندکس دو همون جنسیت 

# حالا هات کدینگ برای ستون کشور
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers= [('encoder', OneHotEncoder() , [1] )], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.25 , random_state=0)

# مقیاس بندی(کاملا اجباری در ساخت شبکه های عصبی)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)

# building ANN
ann = tf.keras.models.Sequential()
# → یه شبکه عصبی خالی ساختیم (Sequential = لایه‌ها پشت سر هم میان)
# adding input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units = 6 ,activation = 'relu'))
# ۶ نورون (units=6)
# تابع فعال‌سازی relu (خیلی رایج و خوب)
# adding second hidden layer
ann.add(tf.keras.layers.Dense(units = 6 ,activation = 'relu'))
# adding output layer
ann.add(tf.keras.layers.Dense(units = 1 ,activation = 'sigmoid'))
# فقط ۱ نورون (چون پیش‌بینی ۰ یا ۱ داریم)
# sigmoid → خروجی رو بین ۰ تا ۱ می‌بره (احتمال رفتن مشتری)


ann.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'] )
# optimizer = 'adam' → بهترین انتخاب برای بیشتر مسائل (خودش سرعت یادگیری رو تنظیم می‌کنه)
# loss = 'binary_crossentropy' → معیار خطا برای مسائل دودویی (۰ یا ۱)
# metrics=['accuracy'] → درصد درست پیش‌بینی کردن رو نشون بده

ann.fit(X_train,y_train,batch_size = 32,epochs = 100)
# batch_size=32 → هر بار ۳۲ تا نمونه رو با هم می‌فرسته داخل شبکه
# epochs=100 → کل داده‌های آموزشی رو ۱۰۰ بار کامل می‌بینه و یاد می‌گیره

new_data = [[1, 0, 0, 1, 600, 40, 3, 60000, 2, 1, 1, 50000]]
y_prediction = ann.predict(sc.transform(new_data))
print(y_prediction > 0.5) #0 =  تو بانک میمونه
# 0.5: یعنی بالای نیم رو یک و زیرش رو 0 درنظر بگیر

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5) # تبدیل به باینری
comparing_preds =  np.concatenate((y_pred.reshape(len(y_pred),1) ,  y_test.reshape(len(y_test),1)),1)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test , y_pred)
print(cm)
# confusion matrix به شکل زیر خوانده می‌شود:
# [[TN  FP]
#  [FN  TP]]
# [[65  3]
#  [ 8 24]]
# TN = 65 → نخریده و مدل هم درست گفته نخریده
# FP = 3  → نخریده ولی مدل اشتباه گفته خریده
# FN = 8  → خریده ولی مدل اشتباه گفته نخریده
# TP = 24 → خریده و مدل هم درست گفته خریده

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)