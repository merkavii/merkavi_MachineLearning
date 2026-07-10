# Feature Importance:
# مشخص می‌کند هر Feature چقدر روی تصمیم مدل تاثیر داشته است.
# برای حذف فیچر کم‌اهمیت، تحلیل مدل و Feature Selection استفاده می‌شود.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- ۱. آماده‌سازی داده‌ها (همانند مراحل قبل) ---
housing = pd.read_csv(r"D:\machine_learning\professional_sklearn\housing.csv")
df = pd.DataFrame(housing)

# جدا کردن ویژگی‌ها از برچسب هدف
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]


num_attribs = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']
cat_attribs = ['ocean_proximity']

# --- ۲. ساخت پایپ‌لاین پیش‌پردازش و مدل ---
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

final_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor(n_estimators=50, random_state=42))
])

# --- ۳. آموزش مدل ---
final_pipeline.fit(X, y)

# --- ۴. استخراج نام جدید ستون‌ها پس از پیش‌پردازش ---
# این متد نام ستون‌های عددی و ستون‌های جدید حاصل از OneHotEncoder را خروجی می‌دهد
feature_names = final_pipeline.named_steps["preprocessing"].get_feature_names_out()
# final_pipeline -> داخلش دو Step داریم: preprocessing, model -> named_steps["preprocessing"] -> داریم به همون ColumnTransformer دسترسی پیدا می‌کنیم.
# .get_feature_names_out() -> بعد از تمام Transformهایی که انجام دادی، اسم فیچر های جدید رو بهم بده. 
print(f'Feature names: {feature_names}')

# --- ۵. استخراج میزان اهمیت ویژگی‌ها از مدل آموزش دیده ---
importances = final_pipeline.named_steps["model"].feature_importances_
# مدل RandomForest بعد از ترین شدن یک اتریبیوت دارد: feature_importances_
# هنگام تصمیم‌گیری، این Featureها این مقدار اهمیت داشته‌اند.

# --- ۶. ترکیب نام‌ها و امتیازها در یک DataFrame برای نمایش زیباتر ---
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False) # مرتب‌سازی از بیشترین به کمترین

print("--- Top Feature Importances ---")
print(feature_importances.to_string(index=False))