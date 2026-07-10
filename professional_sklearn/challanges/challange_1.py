import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# --- ۱. آماده‌سازی داده‌ها ---
housing = pd.read_csv(r"D:\machine_learning\professional_sklearn\housing.csv")
df = pd.DataFrame(housing)

np.random.seed(42)
df.iloc[np.random.choice(df.index, 50), df.columns.get_loc('total_rooms')] = np.nan
df.iloc[np.random.choice(df.index, 50), df.columns.get_loc('median_income')] = np.nan

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# جداسازی دقیق ستون‌ها بدون تداخل
num_attribs_median = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households']
num_attribs_mean = ['median_income']
cat_attribs = ['ocean_proximity']

# --- ۲. ساخت کِلاس سفارشی استاندارد برای پایپ‌لاین ---
class MixingColumns(BaseEstimator, TransformerMixin):
    def __init__(self, first_col_idx=3, second_col_idx=5): # ایندکس‌های total_rooms و population در دیتاسات
        self.first_col_idx = first_col_idx
        self.second_col_idx = second_col_idx
        
    def fit(self, X, y=None):
        return self # متد fit در ترانسفورمرهای بدون یادگیری فقط خود شیء را برمی‌گرداند
        
    def transform(self, X):
        # تبدیل به آرایه نامپای برای سرعت بالاتر و سازگاری با سایکیت‌لرن
        X_numpy = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        # محاسبه ستون جدید: تقسیم اتاق‌ها بر جمعیت
        rooms_per_person = X_numpy[:, self.first_col_idx] / X_numpy[:, self.second_col_idx]
        # اضافه کردن ستون جدید به انتهای آرایه داده‌ها
        return np.c_[X_numpy, rooms_per_person]

# --- ۳. طراحی پله‌پله پیش‌پردازش داده‌ها ---

# بخش عددی اول (پر کردن با میانه + ساخت ویژگی سفارشی + استانداردسازی)
# نکته چالش: ویژگی جدید باید قبل از StandardScaler ساخته می‌شد!
median_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('feature_mixer', MixingColumns(first_col_idx=3, second_col_idx=5)), # ساخت ویژگی داخل پایپ‌لاین
    ('std_scaler', StandardScaler()),
])

# بخش عددی دوم (پر کردن با میانگین + استانداردسازی)
mean_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('std_scaler', StandardScaler())
])

# ادغام همه‌چیز در ColumnTransformer
full_pipeline = ColumnTransformer([
    ('total_rooms_pipe', median_pipeline, num_attribs_median),
    ('median_income_pipe', mean_pipeline, num_attribs_mean),
    ('cat_pipe', OneHotEncoder(handle_unknown='ignore'), cat_attribs)
])

# --- ۴. اتصال به مدل و تنظیم GridSearchCV ---
forest_reg = RandomForestRegressor(random_state=42)

final_pipeline = Pipeline([
    ("preprocessing", full_pipeline),
    ("model", forest_reg)
])

# اصلاح ساختار پارامترها و نحوه آدرس‌دهی دقیق لایه‌های درونی
param_grid = [
    {
        'preprocessing__total_rooms_pipe__imputer__strategy': ['median', 'most_frequent'],
        'model__n_estimators': [10, 30], 
        'model__max_features': [2, 4]
    }
]

grid_search = GridSearchCV(
    final_pipeline, 
    param_grid, 
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True
)

print("Starting Grid Search... Please wait.")
grid_search.fit(X, y)
print("Grid Search completed successfully!\n")

print("--- Best Hyperparameters ---")
print(grid_search.best_params_)