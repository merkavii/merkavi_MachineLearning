import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Restaurant_Reviews.tsv"
dataset = pd.read_csv(DATA_PATH , delimiter='\t',quoting = 3)
# چون فایل تی اس ویسه اینجوری میگیم که فایل تی اس ویه و با سی اس وی اشتباه نکنه : delimiter='\t'
# چون کوتیشن های زیادی توی دیتاست هست برای اینکه
# موقع خوندن و اینا قاطی نکنه و مثلا فقط داخل کوتیشنو متن حساب نکنه اینو میزاریم و خیلی هم مهمه:quoting = 3

# cleaning text
import re
import nltk
# nltk.download('stopwords')
# یبار دان کردیم کافیه
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# برای تبدیل مثلا loved به love 
# برای فضای کمتر چون خب جفتشون به معنی نظر مثبت هستند پس دیگه دوتا فضا اشغال نشه
corpus = []
# لیستی که قراره داده های تمیز بیاد توش
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]' , ' ', dataset['Review'][i])
    # این متغیر قراره مرحله به مرحله تمیز بشه
    # [[^a-zA-Z]' , ' ' : یعنی هرچیزی بجز حروف کوچک ای تا زد و حروف بزرگ ای تا زد بود با اسپیس جدا کن
    # ^ = not
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    # چون نات رو ی کلمه غیر ضروری میدونست پس ما حذفش کردی چون نات جمله رو منفی نشون میده
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)] 
    # میخوایم تو کلمات ریویو بگردیم بدون استفاده هارو پاک کنیم
    review = ' '.join(review)
    corpus.append(review)
# print(f'corpus : {corpus}')
    
#creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
# متن‌ها رو می‌گیره
# می‌شکنه به کلمه
# می‌شماره هر کلمه چند بار اومده
# و در نهایت متن رو تبدیل می‌کنه به اعداد
# for example : [0,2,0,0,1,0,0,0,4,....]
cv = CountVectorizer(max_features = 1500)
# فقط ۱۵۰۰ تا کلمه مهم‌تر رو نگه دار
X = cv.fit_transform(corpus).toarray()
# fit:
# کلمات corpus رو بررسی می‌کنه
# دیکشنری ۱۵۰۰تایی می‌سازه

# transform: هر review رو تبدیل می‌کنه به یه بردار عددی

# toarray:
# خروجی رو از حالت sparse matrix
# تبدیل می‌کنه به آرایه‌ی NumPy

# X :
# | review | good | bad | love | food | not |
# | ------ | ---- | --- | ---- | ---- | --- |
# | r1     | 1    | 0   | 1    | 1    | 0   |
# | r2     | 0    | 1   | 0    | 1    | 1   |
# | r3     | 1    | 0   | 0    | 0    | 0   |

y = dataset.iloc[:,-1].values

# from sklearn.model_selection import train_test_split
# X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.25 , random_state=0)
# from sklearn.naive_bayes import GaussianNB 
# classifier = GaussianNB()
# classifier.fit(X_train , y_train)

# y_pred = classifier.predict(X_test)
# comparing_preds =  np.concatenate((y_pred.reshape(len(y_pred),1) ,  y_test.reshape(len(y_test),1)),1)

# from sklearn.metrics import confusion_matrix,accuracy_score
# cm = confusion_matrix(y_test , y_pred)
# print(cm)

# accuracy = accuracy_score(y_test,y_pred)
# print(accuracy)
from test_models.classification_models import main
main(X,y)



# python -m NLP.bag_of_words
