# SVR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap



dataset = pd.read_csv('Social_Network_Ads.csv')
dataset = dataset.drop(columns=['User ID', 'Gender'])
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.25 , random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC 
classifier = SVC(kernel='linear',random_state=0)
# kernel='linear' اگر دیتا رو بشه به صورت خطی جدا کرد 
classifier.fit(X_train , y_train)

print(classifier.predict(sc.transform([[30,87000]])))

y_pred = classifier.predict(X_test)
comparing_preds =  np.concatenate((y_pred.reshape(len(y_pred),1) ,  y_test.reshape(len(y_test),1)),1)


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test , y_pred)
print(cm)

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)


X_set, y_set = sc.inverse_transform(X_train), y_train

X1, X2 = np.meshgrid(
    np.arange(X_set[:, 0].min() - 2, X_set[:, 0].max() + 2, 0.5),      
    np.arange(X_set[:, 1].min() - 2000, X_set[:, 1].max() + 2000, 250) 
)

plt.contourf(
    X1, X2,
    classifier.predict(sc.transform(np.c_[X1.ravel(), X2.ravel()])).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('SVR (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()




X_set, y_set = sc.inverse_transform(X_test), y_test

X1, X2 = np.meshgrid(
    np.arange(X_set[:, 0].min() - 2, X_set[:, 0].max() + 2, 0.5),
    np.arange(X_set[:, 1].min() - 2000, X_set[:, 1].max() + 2000, 250)
)

plt.contourf(
    X1, X2,
    classifier.predict(sc.transform(np.c_[X1.ravel(), X2.ravel()])).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('SVR (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
# این نمودار نشان می‌دهد همان مدل روی داده‌هایی که ندیده چطور عمل می‌کند
