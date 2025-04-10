# predicts petal length based on all features

import numpy as np
from sklearn.datasets import load_iris
from LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data  # All features
y = iris.data[:, 2:3]  # Petal length

# Split the data
_, X_test, _, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = LinearRegression()
model.load_weights("regression3_weights.npz")

mse = model.score(X_test, y_test)
print(f"Mean Squared Error for Model 3 based on all features: {mse}")

