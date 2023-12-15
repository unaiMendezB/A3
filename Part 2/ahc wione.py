import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv('A3-wine/wine-data.txt', sep='\t')

# Separate features and labels
X = data.drop('quality', axis=1)
y = data['quality']

# Compute the Euclidean distances
distances = pdist(X.values, metric='euclidean')

# Perform AHC with UPGMA and CL methods
linkage_matrix_upgma = linkage(distances, method='average')
linkage_matrix_cl = linkage(distances, method='complete')

# Plot the dendrograms
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
dendrogram(linkage_matrix_upgma, labels=y.values, leaf_rotation=90)
plt.title('UPGMA')

plt.subplot(1, 2, 2)
dendrogram(linkage_matrix_cl, labels=y.values, leaf_rotation=90)
plt.title('CL')

plt.show()
