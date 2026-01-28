import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# UCB می‌گه:
# «برای هر تبلیغ، یک حد بالا (Upper Bound) حساب کن
# تبلیغی که بیشترین حد بالا رو داره رو نشون بده»
# ین حد بالا از دو بخش ساخته می‌شه:
# میانگین پاداش قبلی (Exploitation)
# میزان عدم قطعیت (Exploration)
dataset = pd.read_csv('ads_CTR_optimisation.csv')
# قدم به قدم مراحل بهینه سازی یو سی بی رو جلو میریم
import math
N_users = 10000
# تعداد کل کاربران 
# هر کاربر = یک تصمیم
#  در هر دور، فقط یک تبلیغ انتخاب می‌شه
ads = 10
# تعداد تبلیغ‌ها 
# 10 تا گزینه داریم
#  نمی‌دونیم کدوم بهتره
ads_selected = []
# لیستی برای ذخیره اینکه در هر دور کدوم تبلیغ انتخاب شده
numbers_of_selections = [0] * ads
# تعداد دفعاتی که هر تبلیغ تا الان انتخاب شده
# در ابتدا هیچ تبلیغی انتخاب نشده
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

sums_of_rewards = [0] * ads
# اگر تبلیغ 3:
# 5 بار نمایش داده شده
# 2 بار کلیک خورده
# → sum = 2

total_reward = 0
# الگوریتم چقدر خوب عمل کرده؟

for n in range(0,N_users):
    # انتخاب تبلیغ
    # گرفتن پاداش
    # آپدیت اطلاعات
    ad = 0
    # ad → تبلیغ منتخب این دور
    max_upper_bound = 0
    # max_upper_bound → بیشترین حد بالا بین تبلیغ‌ها
    for i in range(0,ads):
        # برای هر تبلیغ:
        # محاسبه Upper Bound
        # مقایسه با بقیه  
        if numbers_of_selections[i] > 0 :
            # اگر تبلیغ قبلاً انتخاب شده باشد
            # تبلیغی که هنوز ندیدیم، باید شانس تست داشته باشه
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            # میانگین پاداش (Exploitation)
            # این تبلیغ تا الان چقدر خوب بوده؟
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            # بخش اکتشاف (Exploration)
            # اگر تبلیغ کم انتخاب شده → delta بزرگ
            # اگر تبلیغ زیاد انتخاب شده → delta کوچیک
            upper_bound = average_reward + delta_i
            # UCB = exploitation + exploration
        else :
            # اگر تبلیغ هنوز انتخاب نشده باشد
            upper_bound = 1e400
            # «این تبلیغ رو حتماً حداقل یک‌بار امتحان کن»
        if upper_bound > max_upper_bound :
            # انتخاب بهترین تبلیغ
            max_upper_bound = upper_bound
            ad = i
    # ثبت انتخاب
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n , ad]
    # میاد بهنرین تبلیغو به عنوان پاداش توی این متغیر ذخیره میکنه
    sums_of_rewards[ad] += reward
    total_reward += reward
    # الگوریتم یاد گرفت
    # دور بعد تصمیم بهتری می‌گیره
    
plt.hist(ads_selected)
plt.title('histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()