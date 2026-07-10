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
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train , y_train)

print(classifier.predict(sc.transform([[30,87000]])))
y_pred = classifier.predict(X_test)
comparing_preds =  np.concatenate((y_pred.reshape(len(y_pred),1) ,  y_test.reshape(len(y_test),1)),1)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test , y_pred)
print(cm)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

# applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10 )
print(accuracies)
avg = accuracies.mean()
# print(avg)
variance = accuracies.std()
print('Accuracy : {:.2f}%'.format(avg * 100))
print('Standard Deviation : {:.2f}%'.format(variance * 100))


# applying grid search
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.25,0.5,0.75,1],'kernel': ['linear']},
              {'C': [0.25,0.5,0.75,1],'kernel': ['rbf'],'gamma':[0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9]}]

grid_search = GridSearchCV(estimator= classifier,param_grid=parameters,scoring= 'accuracy',cv=10,n_jobs=-1)
grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print('Best Accuracy : {:.2f}%'.format(best_accuracy * 100))
print(f'Best Parameters : {best_parameters}')


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
