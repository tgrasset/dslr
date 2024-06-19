import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import argv
from typing import List


class SimpleImputer:
    def __init__(self, numerical_columns, categorical_columns):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.means = None
        self.modes = None

    def __repr__(self):
        return f"--- SimpleImputer ---\nMeans: {self.means}\nModes: {self.modes}\n"

    def fit(self, X):
        self.means = {column: X[column].mean() for column in self.numerical_columns}
        self.modes = {column: X[column].mode()[0] for column in self.categorical_columns}

    def transform(self, X):
        copy = X.copy()
        for column in self.numerical_columns:
            copy[column] = copy[column].fillna(self.means[column])
        for column in self.categorical_columns:
            copy[column] = copy[column].fillna(self.modes[column])
        return copy


class StandardScaler:
    def __init__(self, numerical_columns):
        self.numerical_columns = numerical_columns
        self.means = None
        self.stds = None

    def __repr__(self):
        return f"--- StandardScaler ---\nMeans: {self.means}\nStandard Deviations: {self.stds}\n"

    def fit(self, X):
        self.means = {column: X[column].mean() for column in self.numerical_columns}
        self.stds = {column: X[column].std() for column in self.numerical_columns}

    def transform(self, X):
        copy = X.copy()
        for column in self.numerical_columns:
            copy[column] = (copy[column] - self.means[column]) / self.stds[column]
        return copy


class OneHotEncoder:
    def __init__(self, categorical_columns, drop_last=True):
        self.categorical_columns = categorical_columns
        self.drop_last = drop_last
        self.new_columns = {}

    def __repr__(self):
        return f"--- OneHotEncoder ---\nColumns mapping: {self.new_columns}\nDrop Last: {self.drop_last}\n"

    def fit(self, X):
        for column in self.categorical_columns:
            self.new_columns[column] = []
            values = np.unique(X[column]).astype(str)
            if self.drop_last:
                values = values[:-1]
            for value in values:
                self.new_columns[column].append(column+'_'+value)

    def transform(self, X):
        copy = X.copy()
        for column in self.categorical_columns:
            for new_column in self.new_columns[column]:
                value = new_column.split('_')[-1]
                copy[new_column] = (copy[column].astype(str) == str(value)).astype(int)
        copy = copy.drop(columns=self.categorical_columns)
        return copy


class PreprocessorPipeline:
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def __repr__(self):
        repr = ""
        for preprocessor in self.preprocessors:
            repr = repr + preprocessor.__repr__() + '\n'
        return repr

    def fit(self, X):
        for preprocessor in self.preprocessors:
            preprocessor.fit(X)

    def transform(self, X):
        X_preprocessed = X.copy()
        for preprocessor in self.preprocessors:
            X_preprocessed = preprocessor.transform(X_preprocessed)
        return X_preprocessed


class SortingHat:
    def __init__(self, n_features, lr):
        self.logreg_s = LogReg(n_features, 'House Slytherin', lr)
        self.logreg_h = LogReg(n_features, 'House Hufflepuff', lr)
        self.logreg_g = LogReg(n_features, 'House Gryffindor', lr)
        self.logreg_r = LogReg(n_features, 'House Ravenclaw', lr)
        self.learning_rate = lr
        self.losses = []

    def __call__(self, X):
        # print(np.array([self.logreg_s(X), self.logreg_h(X), self.logreg_g(X), self.logreg_r(X)]))
        return np.array([self.logreg_s(X), self.logreg_h(X), self.logreg_g(X), self.logreg_r(X)])

    def predict(self, X):
        return np.argmax(self.__call__(X), axis=0)

    def train_step(self, X_train, Y_train, X_test, Y_test, lr=None):
        if lr == None:
            lr = self.learning_rate
        self.logreg_s.train_step(X_train, Y_train, X_test, Y_test, lr)
        self.logreg_h.train_step(X_train, Y_train, X_test, Y_test, lr)
        self.logreg_g.train_step(X_train, Y_train, X_test, Y_test, lr)
        self.logreg_r.train_step(X_train, Y_train, X_test, Y_test, lr)
        self.losses.append({
            'step': len(self.losses) + 1,
            'train_loss':
                self.logreg_s.losses[-1]['train_loss'] +
                self.logreg_h.losses[-1]['train_loss'] +
                self.logreg_g.losses[-1]['train_loss'] +
                self.logreg_r.losses[-1]['train_loss'],
            'test_loss':
                self.logreg_s.losses[-1]['test_loss'] +
                self.logreg_h.losses[-1]['test_loss'] +
                self.logreg_g.losses[-1]['test_loss'] +
                self.logreg_r.losses[-1]['test_loss']
        })


class LogReg:
    def __init__(self, n_features, target_column, lr):
        self.target_column = target_column
        self.weights = np.array([0 for i in range(n_features)])
        self.bias = 0
        self.learning_rate = lr
        self.losses = []

    def __call__(self, X):
        linreg_result = self.bias + X @ self.weights
        return 1 / (1 + np.exp(-linreg_result))

    def predict(self, X):
        proba = self.__call__(X)
        return (proba >= 0.5).astype(int)

    def train_step(self, X_train, Y_train, X_test, Y_test, lr=None):
        if lr == None:
            lr = self.learning_rate
        error = self.__call__(X_train) - Y_train[self.target_column]
        self.weights = self.weights - lr * (1 / len(X_train)) * (X_train.T @ error)
        self.bias = self.bias - lr * (np.sum(error))
        self.losses.append({
            'step': len(self.losses) + 1,
            'train_loss': self.loss(X_train, Y_train[self.target_column]),
            'test_loss': self.loss(X_test, Y_test[self.target_column])
        })

    def loss(self, X, Y):
        Y_pred = self.__call__(X)
        loss =  - (1 / len(X)) * np.sum(Y * np.log(Y_pred) + (1 - Y) * np.log(1 - Y_pred))
        return loss


# def updateParams(estimate, km, prices, tmp0, tmp1, learningRate):
#     """
#     theta0 and theta1 are updated according to the formulas given in the subject
#     according to the gradient descent rule. The new values are returned.
#     """
#     m = len(prices)
#     tmp0 = tmp0 - (learningRate * (1 / m) * np.sum(estimate - prices))
#     tmp1 = tmp1 - (learningRate * (1 / m) * np.sum((estimate - prices) * km))
#     return tmp0, tmp1


# def linearRegression(normKm, prices, kmMean, kmStd):
#     """
#     This function implements the gradient descent algorithm to find the best
#     parameters for our model, updating the two arguments given to it many times
#     """
#     tmp0 = 0
#     tmp1 = 0
#     epochs = 100
#     learningRate = 0.1
#     x = np.linspace(min(normKm), max(normKm), 100)

#     for i in range(epochs):
#         estimate = tryParams(normKm, tmp0, tmp1)
#         # if i % 5 == 0:
#         #     deNormX = x * kmStd + kmMean
#         #     y = tmp0 + tmp1 * x
#         #     plt.plot(deNormX, y, label=f'Epoch {i}')
#         tmp0, tmp1 = updateParams(estimate, normKm, prices, tmp0, tmp1, learningRate)
#     deNormX = x * kmStd + kmMean
#     y = tmp0 + tmp1 * x
#     plt.plot(deNormX, y, 'r', label='Final Regression Line')
#     plt.legend()
#     plt.title('Cars selling price vs Mileage')
#     plt.xlabel('km')
#     plt.ylabel('price')
#     return tmp0, tmp1

# def readData(filePath):
#     """
#     Gets data from the file and returns two arrays corresponding to mileage and price
#     """
#     data = pd.read_csv(filePath)
#     assert data is not None, "The data set is not a proper csv file"
#     assert data.shape[1] == 2, "The data set must contain exactly two columns"
#     x = data.iloc[:, 0]
#     y = data.iloc[:, 1]
#     return x, y

# def normalizeData(x):
#     """
#     Since kilometers have such a big scale, we normalize the data, reducing values to
#     small values around 0 to minimize the error during gradient descent math
#     """
#     mean = np.mean(x)
#     std = np.std(x)
#     return (x - mean) / std, mean, std


# def main():
#     try:
#         assert len(argv) == 2, "The script needs a data file as argument"
#         km, prices = readData(argv[1])
#         normKm, kmMean, kmStd = normalizeData(km)
#         normTheta0, normTheta1= linearRegression(normKm, prices, kmMean, kmStd)
#         # following two lines convert theta1 and theta0 to their un-normalized values
#         theta1 = normTheta1 / kmStd
#         theta0 = normTheta0 - (theta1 * kmMean)
#         print("Training complete !")
#         print(f"Final values ---> theta 0 : {theta0}, theta1 : {theta1}")
#         df = pd.DataFrame({
#             "theta0": [theta0],
#             "theta1": [theta1]
#         })
#         df.to_csv("thetas.csv", index=False)
#         print("Values saved in thetas.csv")
#         plt.scatter(km, prices)
#         plt.show()
#         return 0

#     except (AssertionError, Exception) as err:
#         print("Error: ", err)
#         return 1

# if __name__ == "__main__":
#     main()
