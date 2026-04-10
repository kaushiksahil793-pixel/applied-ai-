import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
iris = load_iris()
X4 = iris.data
y4 = iris.target

# Without PCA
kmeans = KMeans(n_clusters=3)
labels1 = kmeans.fit_predict(X4)
print("ARI without PCA:", adjusted_rand_score(y4, labels1))

# With PCA
pipeline = Pipeline([
    ('pca', PCA(n_components=2)),
    ('kmeans', KMeans(n_clusters=3))
])
labels2 = pipeline.fit_predict(X4)
print("ARI with PCA:", adjusted_rand_score(y4, labels2))