import numpy as np

class LDA:
    def __init__(self):
        self.weights = None
        self.mean_0 = None
        self.mean_1 = None

    def fit(self, X, y):
        # Segregating data based on class labels
        class_0 = X[y == 0]
        class_1 = X[y == 1]
        
        # Calculating the means here
        self.mean_0 = np.mean(class_0, axis=0)
        self.mean_1 = np.mean(class_1, axis=0)
        
        # Compute covariance matrices
        cov_0 = np.cov(class_0, rowvar=False, bias=True)
        cov_1 = np.cov(class_1, rowvar=False, bias=True)
        
        # Weighted average of covariance matrices
        shared_cov = (len(class_0) * cov_0 + len(class_1) * cov_1) / len(X)
        
        # Compute weights for LDA
        self.weights = np.linalg.inv(shared_cov).dot(self.mean_1 - self.mean_0)

    def predict(self, X):
        projections = np.dot(X, self.weights)
        return np.where(projections > 0, 1, 0)

    def save_weights(self, path):
        """Save weights, mean_0, and mean_1 to the given path."""
        np.savez(path, mean_0=self.mean_0, mean_1=self.mean_1, weights=self.weights)

    def load_weights(self, path):
        """Load weights, mean_0, and mean_1 from the given path."""
        data = np.load(path)
        self.mean_0 = data['mean_0']
        self.mean_1 = data['mean_1']
        self.weights = data['weights']
