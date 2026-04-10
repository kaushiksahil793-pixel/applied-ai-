import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


digits = load_digits()
X3 = digits.data
y3 = digits.target

pca_2 = PCA(n_components=2)
X2D = pca_2.fit_transform(X3)

plt.scatter(X2D[:,0], X2D[:,1], c=y3)
plt.title('2D PCA Projection')
plt.show()

# Variance
pca_full = PCA().fit(X3)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

plt.plot(cum_var)
plt.axhline(y=0.95)
plt.title('Cumulative Variance')
plt.show()