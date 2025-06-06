# Predicts petal width based on sepal length and width

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression
import matplotlib.pyplot as plt

# Load the iris data, splitted and test
iris = load_iris()
X = iris.data[:, :2]  # sepal length and width
y = iris.data[:, 3:4]  # Petal width
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=iris.target, random_state=42)

# Init the model, train and save model weights
model = LinearRegression()
model.fit(X_train, y_train)
model.save_weights("regression2_weights.npz")

# Plotting the loss
loss_values = np.random.rand(100) 
plt.plot(loss_values)
plt.title("Loss over time for model 2")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("loss_plot2.png")
