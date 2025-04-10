import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Logistic_regression import LogisticRegression
from Linear_Discriminant_Analysis import LDA

iris = load_iris()
X = iris.data  # All features
y = iris.target
y[y == 2] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# For Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_model.save_weights('logistic_all_weights3.npz')
logistic_model.load_weights('logistic_all_weights3.npz')
y_pred_logistic = logistic_model.predict(X_test)
print(f"Accuracy of Logistic Regression using all features: {accuracy_score(y_test, y_pred_logistic)}")

# For LDA
lda_model = LDA()
lda_model.fit(X_train, y_train)
lda_model.save_weights('lda_all_weights3.npz')
lda_model.load_weights('lda_all_weights3.npz')
y_pred_lda = lda_model.predict(X_test)
print(f"Accuracy of LDA using all features: {accuracy_score(y_test, y_pred_lda)}")
