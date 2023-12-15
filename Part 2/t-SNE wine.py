import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Read the data
data = pd.read_csv('A3-wine/wine-data.txt', sep='\t')

# Separate features and labels
X = data.drop('quality', axis=1)
y = data['quality']

# Create a t-SNE object
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)

# Definir los parámetros para los gráficos
parameters = [
    {'perplexity': 10, 'learning_rate': 100},
    {'perplexity': 20, 'learning_rate': 200},
    {'perplexity': 30, 'learning_rate': 300},
    {'perplexity': 40, 'learning_rate': 400},
    {'perplexity': 50, 'learning_rate': 500},
    {'perplexity': 20, 'learning_rate': 1000},
    {'perplexity': 20, 'learning_rate': 500},
]

# Generar y dibujar los gráficos
for i, params in enumerate(parameters):
    tsne = TSNE(n_components=2, perplexity=params['perplexity'], learning_rate=params['learning_rate'])
    X_embedded = tsne.fit_transform(X)

    plt.figure(i, figsize=(6, 6))
    plt.title(f'Attempt {i+1}: Perplexity={params["perplexity"]}, Learning Rate={params["learning_rate"]}')
    for i, label in enumerate(np.unique(y)):
        plt.scatter(X_embedded[y == label, 0], X_embedded[y == label, 1], label=label)
    plt.legend()
plt.show()