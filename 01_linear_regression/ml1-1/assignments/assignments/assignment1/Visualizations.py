import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from mlxtend.plotting import plot_decision_regions
from Logistic_regression import LogisticRegression
from Linear_Discriminant_Analysis import LDA

dataset = datasets.load_iris()

def visualize_model(data_x, data_y, model, model_name):
    plot_decision_regions(data_x, data_y, clf=model)
    plt.title(f"{model_name} - Petal Dimensions")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.show()

# Using petal dimensions
features = dataset.data[:, 2:4]
labels = dataset.target
labels[labels == 2] = 1

# Splitting dataset into training and testing
train_x, test_x, train_y, test_y = tts(features, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

# Logistic Regression model
log_reg_model = LogisticRegression()
log_reg_model.fit(train_x, train_y)
visualize_model(test_x, test_y, log_reg_model, "Logistic Regression")

# LDA
lda_clf = LDA()
lda_clf.fit(train_x, train_y)
visualize_model(test_x, test_y, lda_clf, "Linear Discriminant Analysis")
