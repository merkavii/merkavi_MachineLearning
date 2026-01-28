import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_customers.csv')
X = dataset.iloc[:, [3,4]].values
# ایندکس 3 و چهار دیتاست

# حالا با استفاده از الگوریتم البو میخوایم تعداد متغیر وابسته رو بدست بیاریم.یا همون بفهمیم در داده مون افراد به چند گروه تقسیم میشوند
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    # adding WCSS of the kmeans object into the list
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('The Number Of Clusters')
plt.ylabel('WCSS')
plt.show()
# بهترین تعداد خوشه ها جاییه که دبلیو سی اس اس کند میشه اینجا حدودا پنج میشه
# پس ما متغیر وابسته مون میشه بین 0 تا 4 یعنی اون شخص یا توی گروه 0 قرار میگیره یا 1 و ... یا گروه 4

kmeans = KMeans(n_clusters=5,init='k-means++',random_state=42)
ykmeans = kmeans.fit_predict(X)
print(ykmeans)

plt.scatter(X[ykmeans == 0,0],X[ykmeans == 0,1],s=100,c='red',label='cluster 0')
# ستون 0و1 اونایی که توی گروه 0 قرار دارند
plt.scatter(X[ykmeans == 1,0],X[ykmeans == 1,1],s=100,c='purple',label='cluster 1')
plt.scatter(X[ykmeans == 2,0],X[ykmeans == 2,1],s=100,c='blue',label='cluster 2')
plt.scatter(X[ykmeans == 3,0],X[ykmeans == 3,1],s=100,c='green',label='cluster 3')
plt.scatter(X[ykmeans == 4,0],X[ykmeans == 4,1],s=100,c='orange',label='cluster 4')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s = 300 , c='yellow',label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annule Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()