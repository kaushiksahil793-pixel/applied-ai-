import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

X2 = np.random.rand(200, 3)

db = DBSCAN(eps=0.3, min_samples=5).fit(X2)
labels_db = db.labels_

plt.scatter(X2[:,0], X2[:,1], c=labels_db)
plt.title('DBSCAN Clusters (Noise = -1)')
plt.show()

# KMeans comparison
kmeans = KMeans(n_clusters=3)
labels_km = kmeans.fit_predict(X2)

plt.scatter(X2[:,0], X2[:,1], c=labels_km)
plt.title('KMeans Clusters')
plt.show()