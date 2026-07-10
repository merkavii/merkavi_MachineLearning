import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy import stats

# --- ۱. آماده‌سازی داده‌ها (همانند مراحل قبل) ---
housing = pd.read_csv("D:\machine_learning\professional_sklearn\housing.csv")
df = pd.DataFrame(housing)

# جدا کردن ویژگی‌ها از برچسب هدف
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=0)

num_attribs = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']
cat_attribs = ['ocean_proximity']

# گام اول (imputer): پر کردن مقادیر خالی با «میانه» هر ستون
# گام دوم (std_scaler): استانداردسازی داده‌ها (میانگین صفر و انحراف معیار یک)
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

# بخش اول (num): پایپ‌لاین عددی را روی ستون‌های لیست num_attribs اعمال می‌کند
# بخش دوم (cat): ابزار OneHotEncoder را روی ستون‌های متنی لیست cat_attribs اعمال می‌کند
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

# --- ۲. ساخت پایپ‌لاین نهایی همراه با مدل ---
# ابتدا مدل را با تنظیمات پیش‌فرض تعریف می‌کنیم
forest_reg = RandomForestRegressor(random_state=42)

# ساخت پایپ‌لاین نهایی: داده ابتدا وارد فرآیند پیش‌پردازش شده و سپس مستقیماً به مدل تزریق می‌شود
final_pipeline = Pipeline([
    ("preprocessing", full_pipeline),
    ("model", forest_reg)
])


# --- ۳. تعریف شبکه هایپرپارامترها (Hyperparameter Grid) ---
# نکته بسیار مهم: چون مدل داخل پایپ‌لاین است، باید از ساختار [نام_گام__نام پارامتر] استفاده کنیم (با دو آندراسکور)
param_grid = [
    {
        'model__n_estimators': [10, 30], 
        'model__max_features': [2, 4, 6]
    },
    {
        'model__bootstrap': [False], 
        'model__n_estimators': [3, 10], 
        'model__max_features': [2, 3]
    },
]

# --- ۴. پیکربندی و اجرای GridSearchCV ---
# cv=5: برای هر ترکیب پارامتر، داده‌ها به ۵ بخش تقسیم شده و ۵ بار ارزیابی متقابل (Cross-Validation) انجام می‌شود
# scoring: معیار ارزیابی منفی میانگین مربعات خطا است تا Scikit-Learn بتواند آن را بیشینه‌سازی کند
# return_train_score=True: علاوه بر امتیاز داده‌های تست، امتیاز روی داده‌های آموزشی را هم ذخیره می‌کند تا بیش‌برازش را چک کنیم
grid_search = GridSearchCV(
    final_pipeline, 
    param_grid, 
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True
)

print("Starting Grid Search... Please wait.")
grid_search.fit(X_train, y_train)
# شروع فرآیند آموزش: در این مرحله کل ۱۰ ترکیب پارامتر، هر کدام ۵ بار (مجموعاً ۵۰ بار آموزش مدل) اجرا می‌شوند
print("Grid Search completed successfully!\n")

# --- ۵. استخراج و نمایش بهترین نتایج ---
print("--- Best Hyperparameters ---")
print(grid_search.best_params_)

print("\n--- Best Estimator (Pipeline) ---")
print(grid_search.best_estimator_)
# result: --- Best Hyperparameters ---
# {'model__max_features': 6, 'model__n_estimators': 30

# نمایش امتیاز تمام ترکیب‌ها برای بررسی دقیق‌تر
print("\n--- All Evaluation Scores ---")
cv_res = grid_search.cv_results_
for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
    rmse_score = np.sqrt(-mean_score)
    print(f"RMSE: {rmse_score:.4f} for parameters: {params}")
    
    
    
    
    
    
# ۱. استخراج بهترین مدل پایپ‌لاین به همراه تمام تنظیمات بهینه شده
final_model = grid_search.best_estimator_

# ۲. فرض کنید داده‌های تست (X_test و y_test) را از قبل جدا کرده‌ایم
# نکته: پایپ‌لاین خودش متد transform را روی پیش‌پردازش اعمال می‌کند و سپس predict می‌زند
final_predictions = final_model.predict(X_test)

# ۳. محاسبه RMSE نهایی روی داده‌های تست
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print("--- Final Model Performance on Test Set ---")
print(f"Final RMSE: {final_rmse:.4f}")

# --- ۴. محاسبه بازه اطمینان ۹۵ درصد (95% Confidence Interval) ---
# این کار به ما می‌گوید که خطای واقعی مدل با احتمال ۹۵٪ در چه بازه‌ای قرار دارد
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2

# استفاده از توزیع t-student برای محاسبه بازه اطمینان خطای مربعات
confidence_interval = np.sqrt(stats.t.interval(
    confidence, 
    len(squared_errors) - 1,
    loc=squared_errors.mean(),
    scale=stats.sem(squared_errors)
))

print("\n--- 95% Confidence Interval for RMSE ---")
print(f"The true RMSE is likely between: {confidence_interval}")