# feature_importances می‌گوید کدام ویژگی در کل دیتاسات چقدر مهم است. اما این ابزار دو ضعف بزرگ دارد:
# به ما نمی‌گوید که ویژگی‌ها اثر مثبت دارند یا منفی؟ (مثلاً بالا رفتن درآمد، قیمت خانه را بالا می‌برد یا پایین؟)
# به ما نمی‌گوید که مدل برای یک خانه خاص یا یک مشتری مشخص چطور تصمیم گرفته است؟

# برای حل این مشکل، دیتاساینتیست‌ها از پکیج‌های پیشرفته‌ای برای تفسیرپذیری مدل (Model Explainability) استفاده می‌کنند که
# معروف‌ترین و قدرتمندترین آن‌ها SHAP (SHapley Additive exPlanations) است.


import numpy as np
import pandas as pd
import shap
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# --- ۱. آماده‌سازی داده‌ها (ساده‌شده برای تمرکز روی SHAP) ---
housing = pd.read_csv(r"D:\machine_learning\professional_sklearn\housing.csv")
df = pd.DataFrame(housing)

# جدا کردن ویژگی‌ها از برچسب هدف
X = df.drop(['median_house_value','ocean_proximity'], axis=1)
y = df["median_house_value"]


# تعریف یک پیش‌پردازش ساده عددی
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler())
    ]), X.columns)
])

# ساخت و آموزش خط‌لوله نهایی
final_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor(n_estimators=50, random_state=42))
])
final_pipeline.fit(X, y)

# --- ۲. آماده‌سازی ابزار SHAP ---
# داده‌ها باید دقیقاً همان Preprocessing زمان آموزش را طی کنند،
# چون مدل روی داده‌ی خام آموزش ندیده است.
X_transformed = final_pipeline.named_steps["preprocessing"].transform(X)

# خود مدل RandomForest را از Pipeline جدا می‌کنیم
model = final_pipeline.named_steps["model"]

# TreeExplainer مخصوص مدل‌های درختی (Decision Tree، RandomForest، XGBoost و ...)
# است و می‌تواند سهم هر فیچر را در پیش‌بینی مدل محاسبه کند
explainer = shap.TreeExplainer(model)

# برای سرعت بیشتر فقط 500 نمونه اول را تحلیل می‌کنیم.
# خروجی SHAP شامل میزان تاثیر هر فیچر روی هر نمونه است.
shap_values = explainer(X_transformed[:500])

# --- ۳. تفسیر پیش‌بینی برای اولین خانه دیتاسات (نمونه فردی) ---
print("--- SHAP Analysis for the First House Prediction ---")

# مقدار پایه (Base Value):
# اگر هیچ اطلاعاتی از خانه نداشته باشیم،
# مدل به طور میانگین این مقدار را پیش‌بینی می‌کند.
print(f"Base Value (Average Prediction): {explainer.expected_value[0]:.4f}")

# پیش‌بینی دقیق مدل برای این خانه خاص
# پیش‌بینی نهایی مدل برای اولین خانه
predicted_value = model.predict(X_transformed[0:1])[0]
print(f"Actual Model Prediction for this house: {predicted_value:.4f}")

# نمایش سهم هر ویژگی در پیش‌بینی این خانه خاص
# مقادیر مثبت یعنی ویژگی قیمت را بالا برده و مقادیر منفی یعنی قیمت را پایین آورده است
for i, feature in enumerate(X.columns):
    feature_impact = shap_values.values[0, i]
    print(f"Feature '{feature}' impact: {feature_impact:+.4f}")