import pandas as pd
from minisom import MiniSom
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv('A3-data.txt')

# Separate features and labels
X = data.drop('class', axis=1).values
y = data['class'].values

# Define the settings
settings = [
    {'x': 10, 'y': 10, 'sigma': 1.0, 'learning_rate': 0.5, 'neighborhood_function': 'gaussian'},
    {'x': 20, 'y': 20, 'sigma': 0.5, 'learning_rate': 0.2, 'neighborhood_function': 'gaussian'},
    {'x': 30, 'y': 30, 'sigma': 0.3, 'learning_rate': 0.1, 'neighborhood_function': 'gaussian'},
    {'x': 10, 'y': 10, 'sigma': 1.0, 'learning_rate': 0.5, 'neighborhood_function': 'mexican_hat'},
    {'x': 20, 'y': 20, 'sigma': 0.5, 'learning_rate': 0.2, 'neighborhood_function': 'mexican_hat'},
    {'x': 30, 'y': 30, 'sigma': 0.3, 'learning_rate': 0.1, 'neighborhood_function': 'mexican_hat'}
]

# Loop over the settings
for setting in settings:
    # Initialize the SOM
    som = MiniSom(x=setting['x'], y=setting['y'], input_len=4, sigma=setting['sigma'], learning_rate=setting['learning_rate'], neighborhood_function=setting['neighborhood_function'])

    # Train the SOM
    som.train_random(X, 500)

    plt.figure(figsize=(10, 10))
    for (x, y, z, t), target in zip(X, data['class'].values):
        w = som.winner([x, y, z, t])
        plt.text(w[0], w[1], str(target), color=plt.cm.rainbow(target / len(set(data['class'].values))),
                 fontdict={'weight': 'bold', 'size': 11})

    plt.title(f"SOM with x={setting['x']}, y={setting['y']}, sigma={setting['sigma']}, learning_rate={setting['learning_rate']}, neighborhood_function={setting['neighborhood_function']}")
    plt.axis([0, som.get_weights().shape[0], 0,  som.get_weights().shape[1]])
    plt.show()
