# ما توی دیتاست انواع داده ها داریم که مثلا میحوایم هرکدومشون یه نوع پردازش خاصی روشون انجام بشه
# این کار با استفاده ار پیپ لاین در کالمن ترنسفورمر انجام میشه.مثلا ما میخوایم ستون رشته ایمونو اول تبدیل به عدد شه 
# بعدش اسکیل شه و ...

# سینتکس اصلیش یک لیست از سه‌تایی‌هاست:
# ColumnTransformer([
#     ('اسم دلخواه', transformer_یا_pipeline, لیست_ستون‌ها),
#     ...
# ])


import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ۱. بارگذاری دیتاسات (برای شبیه‌سازی دقیق کتاب، یک ستون متنی فرضی اضافه می‌کنیم)
# housing_data = fetch_california_housing(as_frame=True)
# df = housing_data.frame
housing = pd.read_csv("D:\machine_learning\professional_sklearn\housing.csv")
df = pd.DataFrame(housing)

# جدا کردن ویژگی‌ها از برچسب هدف
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# ۲. تفکیک نام ستون‌ها بر اساس نوع آن‌ها
num_attribs = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']
cat_attribs = ['ocean_proximity']

# ۳. ساخت خط لوله (Pipeline) برای بخش عددی
# ابتدا داده‌های گم‌شده را با میانگین پر می‌کنیم، سپس داده‌ها را مقیاس‌دهی (Standardize) می‌کنیم
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")), # داده هایی که نیستنو و خالین رو با میانگین اون ستون پر میکنه
    ('std_scaler', StandardScaler()),
])

# ۴. بخش اصلی: تعریف ColumnTransformer
# این ابزار تعیین می‌کند که روی ستون‌های عددی 'num_pipeline' اعمال شود و روی ستون‌های دسته‌ای 'OneHotEncoder'
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

# ۵. اعمال تبدیل روی کل داده‌ها
X_prepared = full_pipeline.fit_transform(X)

print("Shape of data before: ", X.shape)
print("Shape of data after ColumnTransformer:", X_prepared.shape)
print(type(X_prepared))

# نکته
# ColumnTransformer فقط
# Transformer قبول میکنه
# نه Estimator.
# 🔴👇🏻 غلطه
# ColumnTransformer([
#     ("model", LogisticRegression(), cols)
# ])










import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# ۱. تعریف مدل رگرسیون خطی
reg_model = SVR(kernel='rbf')

# ۲. ایجاد یک پایپ‌لاین کامل: ابتدا دیتای خام ترانسفورم می‌شود، سپس به مدل داده می‌شود
model_pipeline = Pipeline([
    ("preprocessing", full_pipeline),  # همان ColumnTransformer گام قبل
    ("model", reg_model)
])

# ۳. اجرای K-Fold Cross-Validation (مثلاً با 5 فولد)
# معیار ارزیابی را منفیِ میانگین مربعات خطا (neg_mean_squared_error) قرار می‌دهیم
scores = cross_val_score(model_pipeline, X, y, scoring="neg_mean_squared_error", cv=5)


# cross_val_score(
#     estimator=model,   # مدل یا Pipeline
#     X=X_train,         # ویژگی‌ها
#     y=y_train,         # برچسب‌ها
#     cv=5,              # تعداد Foldها
#     scoring="accuracy" # معیار ارزیابی
# )

# ۴. محاسبه معیار RMSE (ریشه میانگین مربعات خطا)
rmse_scores = np.sqrt(-scores)

# ۵. نمایش نتایج
print("RMSE score in 5 Fold: ", np.round(rmse_scores, 3))
print("Model's mean error :", round(rmse_scores.mean(), 3))
print("Standard deviation of the error (model stability): ", round(rmse_scores.std(), 3))