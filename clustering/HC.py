# سلسله مراتبی
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_customers.csv')
X = dataset.iloc[:, [3,4]].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5 , linkage='ward')
# hc = AgglomerativeClustering(n_clusters=3 , linkage='ward')
y_hc = hc.fit_predict(X)
print(y_hc)
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1],s=100,c='red',label='cluster 0')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1],s=100,c='purple',label='cluster 1')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1],s=100,c='blue',label='cluster 2')
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1],s=100,c='green',label='cluster 3')
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1],s=100,c='orange',label='cluster 4')
plt.title('Clusters of customers')
plt.xlabel('Annule Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()