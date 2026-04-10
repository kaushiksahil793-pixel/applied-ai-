import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Sample dataset (replace with CSV)
data = pd.DataFrame({
    'Income': np.random.randint(20, 100, 100),
    'Spending': np.random.randint(1, 100, 100),
    'Age': np.random.randint(18, 60, 100)
})

X = data[['Income', 'Spending', 'Age']]

# Elbow Method
inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.show()

# Silhouette Score
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    print(f"k={k}, Silhouette={silhouette_score(X, labels)}")

# Final clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

plt.scatter(X['Income'], X['Spending'], c=labels)
plt.xlabel('Income')
plt.ylabel('Spending')
plt.title('Customer Segments')
plt.show()
