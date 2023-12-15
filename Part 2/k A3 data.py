import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('A3-data.txt')

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data[['x', 'y', 'z', 't']])

# Perform KMeans clustering
for k in range(2, 6):  # the range is from 2 to 5
    kmeans = KMeans(n_init=10, n_clusters=k)
    data['cluster'] = kmeans.fit_predict(data[['x', 'y', 'z', 't']])

    # Plot the data
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=data['cluster'], palette=sns.color_palette('dark', k))
    plt.title(f'KMeans Clustering with k={k}')
    plt.show()
