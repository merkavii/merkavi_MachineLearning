import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Restaurant_Rewievs.tsv' , delimiter='\t',quoting = 3)
# چون فایل تی اس ویسه اینجوری میگیم که فایل تی اس ویه و با سی اس وی اشتباه نکنه : delimiter='\t'
# چون کوتیشن های زیادی توی دیتاست هست برای اینکه
# موقع خوندن و اینا قاطی نکنه و مثلا فقط داخل کوتیشنو متن حساب نکنه اینو میزاریم و خیلی هم مهمه:quoting = 3

# cleaning text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# برای تبدیل مثلا loved به love 
# برای فضای کمتر چون خب جفتشون به معنی نظر مثبت هستند پس دیگه دوتا فضا اشغال نشه
corpus = []
# لیستی که قراره داده های تمیز بیاد توش
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]' , ' ', dataset['Reviews'][i])
    # این متغیر قراره مرحله به مرحله تمیز بشه
    # [[^a-zA-Z]' , ' ' : یعنی هرچیزی بجز حروف کوچک ای تا زد و حروف بزرگ ای تا زد بود با اسپیس جدا کن
    # ^ = not
    review = review.lower()
    review = review.split()
    