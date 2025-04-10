import numpy as np

class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def _initialize_weights(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.zeros((1, output_dim))

    def _update_weights(self, gradients_w, gradients_b):
        self.weights -= 0.01 * gradients_w
        self.bias -= 0.01 * gradients_b

    def _split_data(self, X, y):
        train_size = int(0.9 * X.shape[0])
        return X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    def fit(self, X, y, batch_size=None, regularization=None, max_epochs=None, patience=None):
        batch_size = batch_size or self.batch_size
        regularization = regularization or self.regularization
        max_epochs = max_epochs or self.max_epochs
        patience = patience or self.patience

        X_train, X_val, y_train, y_val = self._split_data(X, y)
        self._initialize_weights(X.shape[1], y.shape[1])

        best_weights = np.copy(self.weights)
        best_bias = np.copy(self.bias)
        best_val_loss = float('inf')
        increasing_epochs = 0

        for epoch in range(max_epochs):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                predictions = self.predict(X_batch)
                
                gradients_w = (-2*np.dot(X_batch.T, y_batch - predictions) + 
                               2*regularization*self.weights) / batch_size
                gradients_b = -2*np.sum(y_batch - predictions, axis=0) / batch_size
                
                self._update_weights(gradients_w, gradients_b)

            val_loss = self.score(X_val, y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = np.copy(self.weights)
                best_bias = np.copy(self.bias)
                increasing_epochs = 0
            else:
                increasing_epochs += 1

            if increasing_epochs >= patience:
                break

        self.weights = best_weights
        self.bias = best_bias

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        predictions = self.predict(X)
        n, m = y.shape
        mse = (1 / (n * m)) * np.sum((y - predictions) ** 2)
        return mse

    def save_weights(self, file_path):
        np.savez(file_path, weights=self.weights, bias=self.bias)

    def load_weights(self, file_path):
        data = np.load(file_path)
        self.weights = data['weights']
        self.bias = data['bias']