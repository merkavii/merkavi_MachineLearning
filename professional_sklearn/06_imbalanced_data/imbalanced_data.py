# در مسائل دسته‌بندی ، خیلی وقت‌ها تعداد نمونه‌های یک کلاس بسیار بیشتر از کلاس دیگر است.

# ثال: فرض کن می‌خواهیم تراکنش‌های مشکوک بانکی (کلاهبرداری) را پیدا کنیم. از بین ۱ میلیون تراکنش، شاید فقط ۱۰۰ مورد کلاهبرداری باشد (کمتر از 0.1%).
# مشکل: اگر یک مدل ساده بسازیم که همیشه بگوید «تراکنش امن است»، دقت (Accuracy) مدل 99.9% خواهد بود! اما این مدل عملاً به درد بانک نمی‌خورد چون حتی یک کلاهبرداری را هم پیدا نکرده است.


# Oversampling (بیش‌نمونه‌گیری): تولید داده‌های ساختگی برای کلاس اقلیت (مثلاً با الگوریتم معروف SMOTE).

# Undersampling (کم‌نمونه‌گیری): حذف تصادفی داده‌ها از کلاس اکثریت برای هم‌تراز شدن با کلاس اقلیت.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
# توجه: از پایپ‌لاین و اس‌ام‌او‌تیِ مخصوص داده‌های نامتوازن استفاده می‌کنیم
from imblearn.pipeline import Pipeline 
from imblearn.over_sampling import SMOTE

# --- ۱. خواندن داده‌ها دقیقاً با ساختار درخواستی شما ---
housing = pd.read_csv(r"D:\machine_learning\professional_sklearn\housing.csv")
df = pd.DataFrame(housing)

# تبدیل متغیر هدف به یک مسئله نامتوازن باینری 
# این کار یک کلاس اقلیت شدید (حدود ۵٪ داده‌ها) ایجاد می‌کند
threshold = df["median_house_value"].quantile(0.95)

df["is_expensive"] = (
    df["median_house_value"] >= threshold
).astype(int)

# جدا کردن ویژگی‌ها از برچسب هدف جدید
X = df.drop(["median_house_value", "is_expensive"], axis=1)
y = df["is_expensive"]

# ستون‌های دقیق شما
num_attribs = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']
cat_attribs = ['ocean_proximity']

# --- ۲. تقسیم داده‌ها به آموزش و تست قبل از هر کاری ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# stratify=y -> همون نسبتی که توی دیتاست اصلی هست، توی Train و Test هم حفظ کن.

print(f"Original class distribution in Train Set:\n{y_train.value_counts()}\n")

# --- ۳. طراحی پایپ‌لاین پیش‌پردازش ---
# نکته بسیار حیاتی این است که برای جلوگیری از نشت داده ، باید به جای پایپ لاین معمولی سایکیت-لرن، از پایپ لاین مخصوص کتابخانه ایم بی لرن استفاده کنیم تا عملیات نمونه‌گیری (SMOTE) فقط و فقط روی داده‌های آموزشی اعمال شود، نه داده‌های تست!
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs),
])

# --- ۴. ادغام پیش‌پردازش، الگوریتم SMOTE و مدل در یک پایپ‌لاین هوشمند ---
final_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("smote", SMOTE(random_state=42)), # کلاس اقلیت را به صورت خودکار و بدون نشت داده تکثیر می‌کند
    ("model", RandomForestClassifier(random_state=42))
])

# --- ۵. آموزش مدل ---
# در طول این فیت، SMOTE فقط فولد‌های آموزشی را بالانس می‌کند
final_pipeline.fit(X_train, y_train)
print("Model trained successfully with SMOTE balancing!\n")

# --- ۶. ارزیابی نهایی روی داده‌های تست ---
predictions = final_pipeline.predict(X_test)

print("--- Evaluation Report (Focus on Minority Class 1) ---")
# گزارش جامع شامل معیارهای Precision، Recall و F1-Score
print(classification_report(y_test, predictions))



#       precision    recall  f1-score   support

#    0       0.99      0.98      0.98      3922
#    1       0.65      0.73      0.69       206
   
# کلاس 0 = خانه ارزان
# کلاس 1 = خانه گران     

# Precision for class 1: 0.65 ->  مدل هر وقت گفت "این خانه گرانه" ۶۵٪ مواقع درست گفته.
# Recall for class 1: 0.73 ->  از تمام خانه‌های گران موجود در دیتاست مدل ۷۳ درصدشان را پیدا کرده.
# F1 Score for class 1: 0.69 ->  این یعنی تعادل بین Precision و Recall.
# Support for class 1: 206 -> فقط تعداد نمونه‌های واقعی آن کلاس است.


#     accuracy                           0.97      4128
#    macro avg       0.82      0.86      0.83      4128
# weighted avg       0.97      0.97      0.97      4128

# accuracy : 0.97 -> از کل 4128 نمونه 97درصد را درست پیش بینی کرده اما چون دیتاست نامتوازنه معیار خوبی نیست
