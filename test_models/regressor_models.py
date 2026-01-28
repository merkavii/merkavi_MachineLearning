import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# dataset = pd.read_csv('Bigdata.csv',header=1)# یعنی سطر دوم (AT,V,...) را هدر بگیر
dataset = pd.read_csv('medium_numeric_dataset.csv')# یک دیتا ست جدید
X = dataset.iloc[:, 1:-1].values # ستون اول ایدیه پس نباید بگیریم
y = dataset.iloc[:,-1].values
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=0)


def Multiple_Linear_Regression():
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train , y_train)
    y_pred = regressor.predict(X_test)
    # np.set_printoptions(precision= 2)
    # print( np.concatenate((y_pred.reshape(len(y_pred),1) ,  y_test.reshape(len(y_test),1)),1))
    return r2_score(y_test,y_pred)

def Polynomial_Regression():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree= 2)
    X_poly = poly_reg.fit_transform(X_train)
    regressor = LinearRegression()
    regressor.fit(X_poly , y_train)
    y_pred = regressor.predict(poly_reg.transform(X_test))
    # np.set_printoptions(precision= 2)
    # print( np.concatenate((y_pred.reshape(len(y_pred),1) ,  y_test.reshape(len(y_test),1)),1))
    return r2_score(y_test,y_pred)

def SVR():
    # در سناریوی شما، بهترین الگو این است که X_train, X_test, y_train, y_test را به تابع‌ها پاس بدهی. این هم امن است، هم تمیز، هم قابل مقایسه.
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=0)
    y_2d = y_train.reshape(-1, 1)
    # «پایتون، خودت تعداد ردیف‌ها را حساب کن»
    # «من فقط می‌خواهم 1 ستون داشته باشم»
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_train_scaled  = sc_x.fit_transform(X_train) 
    X_test_scaled  = sc_x.transform(X_test)
    y_train_scaled = sc_y.fit_transform(y_2d).ravel()
    regressor = SVR(kernel='rbf')
    regressor.fit(X_train_scaled , y_train_scaled)
    y_pred = sc_y.inverse_transform(regressor.predict(X_test_scaled).reshape(-1 , 1))
    # np.set_printoptions(precision= 2)
    # print( np.concatenate((y_pred.reshape(len(y_pred),1) ,  y_test.reshape(len(y_test),1)),1))
    return r2_score(y_test,y_pred)

def Decision_Tree_Regression():
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    # np.set_printoptions(precision= 2)
    # print( np.concatenate((y_pred.reshape(len(y_pred),1) ,  y_test.reshape(len(y_test),1)),1))
    return r2_score(y_test,y_pred)

def Random_Forest_Regression():
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(random_state=0, n_estimators=100)
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    # np.set_printoptions(precision= 2)
    # print( np.concatenate((y_pred.reshape(len(y_pred),1) ,  y_test.reshape(len(y_test),1)),1))
    return r2_score(y_test,y_pred)
    
#هر کدوم به یک نزدیک باشه مدل بهتره 
Multiple_Linear_Regression_result = Multiple_Linear_Regression()
Polynomial_Regression_result = Polynomial_Regression()
SVR_result = SVR()
Decision_Tree_Regression_result = Decision_Tree_Regression()
Random_Forest_Regression_result = Random_Forest_Regression()
result_list = [Multiple_Linear_Regression_result,Polynomial_Regression_result,SVR_result,Decision_Tree_Regression_result,Random_Forest_Regression_result]
highest = ''
if max(result_list) == Multiple_Linear_Regression_result :
    highest = 'Multiple_Linear_Regression'
elif max(result_list) == Polynomial_Regression_result :
    highest = 'Polynomial_Regression'
elif max(result_list) == SVR_result :
    highest = 'SVR'
elif max(result_list) == Random_Forest_Regression_result :
    highest = 'Random_Forest_Regression'
elif max(result_list) == Decision_Tree_Regression_result :
    highest = 'Decision_Tree_Regression'
else :
    highest = 'nobody'
lowest = ''
if min(result_list) == Multiple_Linear_Regression_result :
    lowest = 'Multiple_Linear_Regression'
elif min(result_list) == Polynomial_Regression_result :
    lowest = 'Polynomial_Regression'
elif min(result_list) == SVR_result :
    lowest = 'SVR'
elif min(result_list) == Random_Forest_Regression_result :
    lowest = 'Random_Forest_Regression'
elif min(result_list) == Decision_Tree_Regression_result :
    lowest = 'Decision_Tree_Regression'
else :
    lowest = 'nobody'
print('************************************************************************************')
print(f'Multiple_Linear_Regression : {Multiple_Linear_Regression_result}')  
print(f'Polynomial_Regression : {Polynomial_Regression_result}')
print(f'SVR : {SVR_result}')
print(f'Decision_Tree_Regression : {Decision_Tree_Regression_result}')
print(f'Random_Forest_Regression : {Random_Forest_Regression_result}')
print(f'Highest R² score : {highest}')
print(f'Lowest R² score : {lowest}')
print('************************************************************************************')
# Accuracy برای پیش‌بینی برچسب است (Classification)
# R² Score برای پیش‌بینی عدد است (Regression)

