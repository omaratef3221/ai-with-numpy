import numpy as np


# DatasetLink: https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression
# Original Function y = mx+b
class CustomLinearRegressor:
    def __init__(self, learning_rate = 0.01, iterations = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, Y):
        self.weights = np.random.uniform(low=0, high=1, size=(X.shape[0]))
        self.bias = np.random.uniform(low=0, high=1, size=(X.shape[0]))
        for _ in range(self.iterations):
            y_pred = self.weights* X + self.bias
            loss = mean_square_error(Y, y_pred)
            self.weights, self.bias = sgd_optimizer(self.weights, self.bias)

    def predict(self, X):
        return self.weights* X + self.bias


def mean_square_error(y_true, y_pred):
    return np.sum(y_true - y_pred) ** 2 / len(y_true)

def sgd_optimizer(l):
    pass