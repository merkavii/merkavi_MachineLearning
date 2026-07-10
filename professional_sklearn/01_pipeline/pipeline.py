# خب این میائ دنبالهای از ترنسفورم و کلا هرچیزی که بهش میدیم  رو پشت سر هم اجرا میکنه مثل nn.Sequential
# عملا یه خط لوله اس که هربار به ازای هر مقادیری که بهش میدیم اون دستورات رو دنباله ای اجرا میکنه
# و حتی هایپرپارامتر ها هم با دستور __ میتونه عوض بشه
# جلوگیری از Data Leakage
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import set_config
from sklearn.pipeline import make_pipeline

X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=0)
pipe = Pipeline([('scaler', StandardScaler()), ('pca',PCA(n_components=2)),('logisticRegression', LogisticRegression())])
# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
print(pipe.fit(X_train, y_train).score(X_test, y_test)) # اگر اخرش استیمیتور داشتیم مثل اس وی سی یا لینیر رگرشن یا لاجیستیک رگرشن و ... اونموقع دستور فیت میزنیم
# An estimator's parameter can be set using '__' syntax
print(pipe.set_params(logisticRegression__random_state=0).fit(X_train, y_train).score(X_test, y_test))


# وقتی اسم مرحله‌ها برات مهم نیست (و حوصله نداری دستی اسم بدی)، می‌تونی از make_pipeline استفاده کنی که خودش به‌صورت خودکار اسم می‌ذاره (اسم کلاس به حروف کوچک):
pipe = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression())
# اسم مرحله‌ها میشه: standardscaler, pca, logisticregression
print(pipe.named_steps)


# یه پایپ لاینو توی یه پایپ لاین دیگ هم میشه استفاده کرد