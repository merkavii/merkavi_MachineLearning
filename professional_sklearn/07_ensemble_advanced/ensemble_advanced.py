import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error

# وارد کردن مدل‌های آنسامبل پیشرفته
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# --- ۱. خواندن داده‌ها دقیقاً با ساختار درخواستی شما ---
housing = pd.read_csv(r"D:\machine_learning\professional_sklearn\housing.csv")
df = pd.DataFrame(housing)

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

num_attribs = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']
cat_attribs = ['ocean_proximity']

# تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- ۲. ساخت پایپ‌لاین پیش‌پردازش یکسان ---
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs),
])

# --- ۳. پیاده‌سازی و تست مدل اول: XGBoost ---
xgb_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model_xgb", XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42))
])

print("Training XGBoost Model...")
xgb_pipeline.fit(X_train, y_train)
xgb_preds = xgb_pipeline.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
print(f"XGBoost Final RMSE: {xgb_rmse:.4f}\n")

# --- ۴. پیاده‌سازی و تست مدل دوم: LightGBM ---
lgb_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model_lgb", LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1))
])

print("Training LightGBM Model...")
lgb_pipeline.fit(X_train, y_train)
lgb_preds = lgb_pipeline.predict(X_test)
lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_preds))
print(f"LightGBM Final RMSE: {lgb_rmse:.4f}")