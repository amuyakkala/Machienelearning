import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """Activation function used to map any real value between 0 and 1."""
        return 1 / (1 + np.exp(-z))

    def _compute_gradients(self, X, y, predictions):
        """Compute gradients for weights and bias."""
        n_samples = X.shape[0]
        dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
        db = (1 / n_samples) * np.sum(predictions - y)
        return dw, db

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent to update weights and bias
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_model)
            
            dw, db = self._compute_gradients(X, y, predictions)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(linear_model)
        return np.where(predictions > 0.5, 1, 0)

    def save_weights(self, path):
        """Save weights and bias to the given path."""
        np.savez(path, weights=self.weights, bias=self.bias)

    def load_weights(self, path):
        """Load weights and bias from the given path."""
        data = np.load(path)
        self.weights = data['weights']
        self.bias = data['bias']

