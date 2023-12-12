import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('A3-wine/wine-data.txt', sep='\t')

# Separate out the features and the target
features = df.columns[:-1]
x = df.loc[:, features].values
y = df.loc[:,['quality']].values

# Perform PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

# Convert to DataFrame
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['quality']]], axis = 1)

# Plotting the 2D projection
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

targets = df['quality'].unique()
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['quality'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
ax.legend(targets)
ax.grid()

# Scree plot
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Accumulated variance')
plt.show()

