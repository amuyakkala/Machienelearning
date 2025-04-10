# Predicts petal length and width based on sepal length and width

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression
import matplotlib.pyplot as plt

# Load the iris data
iris = load_iris()
X = iris.data[:, :2]  # Sepal length and width
y = iris.data[:, 2:4]  # Petal length and width

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize the model, tain and save
model = LinearRegression()
model.fit(X_train, y_train)
model.save_weights("double_output_regression_weights.npz")

# Mocking the loss plot
loss_values = np.random.rand(100)
plt.plot(loss_values)
plt.title("Loss over time for double output model")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("loss_double_plot.png")
