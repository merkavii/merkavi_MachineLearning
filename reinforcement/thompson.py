import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('ads_CTR_optimisation.csv')

import random
N_users = 10000
ads = 10
ads_selected = []
numberOfReward_1 = [0] * 10
# numberOfReward_1[i] → تعداد کلیک‌هایی که تبلیغ i گرفت
# numberOfReward_0[i] → تعداد بارهایی که تبلیغ i دیده شد ولی کلیک نگرفت
numberOfReward_0 = [0] * 10
total_reward = 0
for n in range(0,N_users):
    ad = 0
    max_random = 0
    for i in range(0,ads):
        random_beta = random.betavariate(numberOfReward_1[i] + 1 , numberOfReward_0[i] + 1)
        if random_beta > max_random :
            max_random = random_beta
            ad = i
            # تبلیغ‌هایی که کلیک بیشتری دارند، عدد بزرگتری می‌گیرند
            # تبلیغ‌هایی که کمتر دیده شده‌اند، اعداد تصادفی بیشتری دارند → فرصت امتحان دوباره
            # تبلیغ با بزرگترین random_beta انتخاب می‌شود (ad)
    ads_selected.append(ad)
    reward = dataset.values[n , ad]
    # یجور حالت تست مدله
    # میگه اگر به این کاربر تبلیغ ad نشون داده بشه، کلیک می‌کنه یا نه
    # الگوریتم فقط با این پاداش واقعی یاد می‌گیرد کدام تبلیغ بهتر است.
    if reward == 1 :
        numberOfReward_1[ad] += 1
    elif reward == 0 :
        numberOfReward_0[ad] += 1
    total_reward += reward
        
            


plt.hist(ads_selected)
plt.title('histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()