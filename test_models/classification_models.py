import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score




dataset = pd.read_csv('Social_Network_Ads.csv')
dataset = dataset.drop(columns=['User ID', 'Gender'])
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.25 , random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def Logistic_cl():
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train , y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test , y_pred)
    errors = cm[0,1] + cm[1,0]
    # cm[واقعی, پیش‌بینی]
    # [[65  3]
    #  [ 8 24]]
    accuracy = accuracy_score(y_test,y_pred)
    return 'Logistic_cl',accuracy,errors

def KNN_cl():
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train , y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test , y_pred)
    errors = cm[0,1] + cm[1,0]
    accuracy = accuracy_score(y_test,y_pred)
    return 'KNN_cl',accuracy,errors

def SVR_linear_cl():
    from sklearn.svm import SVC 
    classifier = SVC(kernel='linear',random_state=0)
    classifier.fit(X_train , y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test , y_pred)
    errors = cm[0,1] + cm[1,0]
    accuracy = accuracy_score(y_test,y_pred)
    return 'SVR_linear_cl',accuracy,errors
    
def SVR_rbf_cl():
    from sklearn.svm import SVC 
    classifier = SVC(kernel='rbf',random_state=0)
    classifier.fit(X_train , y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test , y_pred)
    errors = cm[0,1] + cm[1,0]
    accuracy = accuracy_score(y_test,y_pred)
    return 'SVR_rbf_cl',accuracy,errors
    
def Naive_Bayes_cl():
    from sklearn.naive_bayes import GaussianNB 
    classifier = GaussianNB()
    classifier.fit(X_train , y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test , y_pred)
    errors = cm[0,1] + cm[1,0]
    accuracy = accuracy_score(y_test,y_pred)
    return 'Naive_Bayes_cl',accuracy,errors
    
def Decision_Tree_cl():
    from sklearn.tree import DecisionTreeClassifier 
    classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
    classifier.fit(X_train , y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test , y_pred)
    errors = cm[0,1] + cm[1,0]
    accuracy = accuracy_score(y_test,y_pred)
    return 'Decision_Tree_cl',accuracy,errors
    
def Random_Forest_cl():
    from sklearn.ensemble import RandomForestClassifier 
    classifier = RandomForestClassifier(criterion='entropy',random_state=0,n_estimators=100)
    classifier.fit(X_train , y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test , y_pred)
    errors = cm[0,1] + cm[1,0]
    accuracy = accuracy_score(y_test,y_pred)
    return 'Random_Forest_cl',accuracy,errors

Logistic_res = Logistic_cl()
KNN_res = KNN_cl()
SVR_linear_res = SVR_linear_cl()
SVR_rbf_res = SVR_rbf_cl()
Naive_Bayes_res = Naive_Bayes_cl()
Decision_Tree_res = Decision_Tree_cl()
Random_Forest_res = Random_Forest_cl()

result_list = [Logistic_res, KNN_res, SVR_linear_res, SVR_rbf_res,
               Naive_Bayes_res, Decision_Tree_res, Random_Forest_res]

highest_name = ''
highest_acc = -1
highest_mis = 0
lowest_name = ''
lowest_acc = 101
lowest_mis = 0
print('*********************************************************')
for name, acc, err in result_list:
    print(f'{name} accuracy : {acc} , mistakes : {err}')
    if acc > highest_acc:
        highest_acc = acc
        highest_name = name
        highest_mis = err
    if acc < lowest_acc :
        lowest_acc = acc
        lowest_name = name
        lowest_mis = err
multiple_highest = []
for name, acc, err in result_list:
    if acc == highest_acc and err == highest_mis:
        multiple_highest.append(name)
if len(multiple_highest) > 1 :
    highest_name = ''
    for i in multiple_highest :
        highest_name += f' {i} |'
print(f'Highest accuracy {highest_name} : {highest_acc} , mistakes : {highest_mis}')
print(f'Lowest accuracy {lowest_name} : {lowest_acc} , mistakes : {lowest_mis}')
    

