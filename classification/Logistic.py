# Logistic
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap



dataset = pd.read_csv('Social_Network_Ads.csv')
dataset = dataset.drop(columns=['User ID', 'Gender'])
# ستون‌های User ID و Gender را به‌طور کامل از دیتاست حذف کن.
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.25 , random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train , y_train)

print(classifier.predict(sc.transform([[30,87000]])))

# predict the test set result
y_pred = classifier.predict(X_test)
comparing_preds =  np.concatenate((y_pred.reshape(len(y_pred),1) ,  y_test.reshape(len(y_test),1)),1)
# print(comparing_preds)
# سمت چپ پیش بینی سمت راست تصمیمات واقعی

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


# برگرداندن داده‌ها به مقیاس واقعی برای نمایش
X_set, y_set = sc.inverse_transform(X_train), y_train

# ساخت شبکه‌ای از نقاط برای رسم مرز تصمیم
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 10,
              stop=X_set[:, 0].max() + 10,
              step=0.25),
    np.arange(start=X_set[:, 1].min() - 1000,
              stop=X_set[:, 1].max() + 1000,
              step=0.25)
)

# رسم ناحیه‌های تصمیم (Decision Regions)
plt.contourf(
    X1,
    X2,
    classifier.predict(
        sc.transform(
            np.array([X1.ravel(), X2.ravel()]).T
        )
    ).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)

# تنظیم محدوده محورها
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# رسم نقاط واقعی داده‌ها
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(('red', 'green'))(i),
        label=j
    )

# تنظیمات نهایی نمودار
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
# این نمودار نشان می‌دهد مدل روی داده‌هایی که با آن‌ها آموزش دیده چه تصمیمی می‌گیرد




X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 10,
              stop=X_set[:, 0].max() + 10,
              step=0.25),
    np.arange(start=X_set[:, 1].min() - 1000,
              stop=X_set[:, 1].max() + 1000,
              step=0.25)
)
plt.contourf(
    X1,
    X2,
    classifier.predict(
        sc.transform(
            np.array([X1.ravel(), X2.ravel()]).T
        )
    ).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(('red', 'green'))(i),
        label=j
    )

plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
# این نمودار نشان می‌دهد همان مدل روی داده‌هایی که ندیده چطور عمل می‌کند
