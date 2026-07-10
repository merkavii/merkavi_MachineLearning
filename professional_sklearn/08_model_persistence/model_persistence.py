# تا این مرحله، ما زمان و انرژی زیادی صرف کردیم تا داده‌ها را پیش‌پرازیش کنیم، ویژگی‌های جدید بسازیم و بهترین هایپرپارامترها
# ا برای یک مدل پیشرفته (مثل XGBoost) پیدا کنیم. اما وقتی اسکریپت یا وی‌اس‌کد را ببندیم، تمام این محاسبات و مدل آموزش‌دیده
# از حافظه RAM پاک می‌شوند!

# در پروژه‌های واقعی، ما باید این مدل نهایی (که شامل کل پایپ‌لاین پیش‌پردازش + مدل اصلی است) را به صورت یک فایل روی
# هارد ذخیره کنیم تا بعداً بتوانیم در یک وب‌سایت، اپلیکیشن موبایل یا سیستم‌های دیگر بدون نیاز به آموزش مجدد، از آن برای پیش‌بینی
# داده‌های جدید استفاده کنیم.

# برای ذخیره اشیاء پایتون معمولاً از کتابخانه استاندارد pickle استفاده می‌شود. اما
# برای مدل‌های ماشین‌لرنینگ که حاوی آرایه‌های بزرگ نامپای هستند، کتابخانه joblib فوق‌العاده سریع‌تر و بهینه‌تر عمل می‌کند.


import numpy as np
import pandas as pd
import joblib # کتابخانه اصلی برای ذخیره مدل
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor

# --- ۱. خواندن داده‌ها با ساختار درخواستی شما ---
housing = pd.read_csv(r"D:\machine_learning\professional_sklearn\housing.csv")
df = pd.DataFrame(housing)

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

num_attribs = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']
cat_attribs = ['ocean_proximity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- ۲. ساخت و آموزش پایپ‌لاین کامل ---
preprocessor = ColumnTransformer([
    ("num", Pipeline([('imputer', SimpleImputer(strategy="median")), ('scaler', StandardScaler())]), num_attribs),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs),
])

trained_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", XGBRegressor(n_estimators=50, random_state=42))
])

print("Training the final pipeline...")
trained_pipeline.fit(X_train, y_train)

# --- ۳. ذخیره کردن کل پایپ‌لاین روی هارد (Save) ---
# این فایل شامل تمام ساختار پیش‌پردازش (مثل میانه‌ها و مقیاس‌ها) و وزن‌های مدل XGBoost است
model_filename = "final_housing_pipeline.pkl"
joblib.dump(trained_pipeline, model_filename)
print(f"Pipeline saved successfully as '{model_filename}'!\n")

# ==========================================
# شبیه‌سازی یک سیستم کاملاً مجزا (مثلاً سرور سایت)
# ==========================================

# --- ۴. بارگذاری مجدد پایپ‌لاین از روی فایل (Load) ---
print("Loading the saved pipeline on a production server...")
loaded_pipeline = joblib.load(model_filename)

# --- ۵. پیش‌بینی روی داده‌های جدید بدون نیاز به fit مجدد ---
# فرض کنید یک خانه جدید با ویژگی‌های زیر ثبت شده است
sample_new_data = pd.DataFrame([{
    'longitude': -122.23,
    'latitude': 37.88,
    'housing_median_age': 41.0,
    'total_rooms': 880.0,
    'total_bedrooms': 129.0,
    'population': 322.0,
    'households': 126.0,
    'median_income': 8.3252,
    'ocean_proximity': 'NEAR BAY'
}])

# پایپ‌لاین لود شده خودش داده خام را پیش‌پردازش کرده و قیمت را پیش‌بینی می‌کند
predicted_price = loaded_pipeline.predict(sample_new_data)
print(f"Predicted House Value for new data: ${predicted_price[0]:,.2f}")