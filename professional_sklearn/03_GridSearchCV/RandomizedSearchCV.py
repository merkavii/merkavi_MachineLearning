# اگر بخواهیم ۱۰ هایپرپارامتر مختلف را بررسی کنیم و برای هر کدام ۱۰ مقدار کاندید داشته باشیم، GridSearchCV باید 10^10 (۱۰ میلیارد!) ترکیب را تست کند که عملاً غیرممکن است.

# در Randomized Search: شما یک بازه یا توزیع آماری به آن می‌دهید و مشخص می‌کنید که مثلاً ۱۰۰ ترکیب را به صورت تصادفی انتخاب کند و بسنجد. این روش به شما اجازه می‌دهد فضای بسیار بزرگتری را با هزینه محاسباتی بسیار کمتر جستجو کنید.





import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- ۱. آماده‌سازی سریع داده‌ها ---
housing = pd.read_csv("D:\machine_learning\professional_sklearn\housing.csv")
df = pd.DataFrame(housing)

# جدا کردن ویژگی‌ها از برچسب هدف
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

num_attribs = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

final_pipeline = Pipeline([
    ("preprocessing", full_pipeline),
    ("model", RandomForestRegressor(random_state=42))
])

# --- ۲. تعریف توزیع احتمالی هایپرپارامترها ---
# به جای لیست ثابت، از randint استفاده می‌کنیم تا در هر تکرار یک عدد تصادفی انتخاب کند
param_distribs = {
    'model__n_estimators': randint(low=10, high=200), # انتخاب تصادفی تعداد درخت‌ها بین ۱۰ تا ۲۰۰
    'model__max_features': randint(low=2, high=6),    # انتخاب تصادفی تعداد ویژگی‌ها بین ۲ تا ۶
}

# --- ۳. پیکربندی و اجرای RandomizedSearchCV ---
# n_iter=10: یعنی فقط ۱۰ ترکیب تصادفی از کل فضای پارامترها انتخاب و تست خواهند شد
# cv=3: برای هر کدام از آن ۱۰ ترکیب، ۳ فولد کراس‌ولیدیشن اجرا می‌شود (مجموعاً ۳۰ بار آموزش)
rnd_search = RandomizedSearchCV(
    final_pipeline, 
    param_distributions=param_distribs, 
    n_iter=10, 
    cv=3,
    scoring='neg_mean_squared_error', 
    random_state=42
)

print("Starting Randomized Search... Please wait.")
rnd_search.fit(X, y)
print("Randomized Search completed successfully!\n")

# --- ۴. نمایش نتایج بهینه‌سازی ---
print("--- Best Hyperparameters Found ---")
print(rnd_search.best_params_)

print("\n--- Best RMSE Score ---")
negative_mse = rnd_search.best_score_
print(f"RMSE: {np.sqrt(-negative_mse):.4f}")