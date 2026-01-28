import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=0)

# خب حالا مدل رگرسیون خطیمونو بسازیم توضیحاشم تو دفتر هست
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train , y_train)

# پیش بینی
y_pred = regression.predict(X_test)

# خب میدونیم که ایکس میشه سن و اون متغیر وابسته یا همون چیزی که مدل باید حدس بزنه درآمده
# حالا میخوایم با نقاط قرمز مقدار درآمد اصلی و با نقاط آبی پیش بینی مدل رو ببینیم
plt.scatter(X_train,y_train,color = 'red')
# درآمد اصلی در اون سن از تجربه
plt.plot(X_train , regression.predict(X_train) , color = 'blue')
# اول نقاط واقعی را قرمز رسم می‌کنیم تا معلوم شود داده‌ها واقعاً چه شکلی‌اند
# بعد خط آبی را رسم می‌کنیم تا ببینیم مدل چه خطی را از روی این داده‌ها یاد گرفته
plt.title('Salary VS Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# حالا بخش تست اینجا حقوق واقعی پیش بینی شده است

# در اینجا نقاط قرمز حقوقِ واقعیِ بخش تست هستند
# خط آبی همان خطی است که مدل از روی داده‌های آموزش یاد گرفته
# نزدیک بودن نقاط قرمز به خط آبی یعنی پیش‌بینیِ دقیق‌تر

plt.scatter(X_test,y_test,color = 'red')
plt.plot(X_train , regression.predict(X_train) , color = 'blue')
plt.title('Salary VS Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# حالا این چون ارتباط اجزای دیتاست خطی بود انقد با دقت پیش بینی کرد