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

# applying the xgboost on training set
from xgboost import XGBClassifier
classifier = XGBClassifier(    n_estimators=100,  # تعداد درخت‌ها
    max_depth=3,       # عمق درخت
    learning_rate=0.1, # نرخ یادگیری
    random_state=0
)
classifier.fit(X_train,y_train)


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
print(avg)
variance = accuracies.std()
print(variance)


