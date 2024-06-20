import numpy as np
import pandas as pd
import json


TARGET_CLASSES = ['Slytherin', 'Hufflepuff', 'Gryffindor', 'Ravenclaw']

class LogReg:
    def __init__(self, n_features, target_column, lr):
        self.target_column = target_column
        self.weights = [0 for i in range(n_features)]
        self.bias = 0
        self.learning_rate = lr
        self.losses = []

    def __call__(self, X):
        linreg_result = self.bias + X @ self.weights
        return 1 / (1 + np.exp(-linreg_result))

    def predict(self, X):
        proba = self.__call__(X)
        return (proba >= 0.5).astype(int)

    def train_step(self, X_train, Y_train, X_test=None, Y_test=None, lr=None):
        if lr == None:
            lr = self.learning_rate
        error = self.__call__(X_train) - Y_train[self.target_column]
        self.weights = (self.weights - lr * (1 / len(X_train)) * (X_train.T @ error)).values.tolist()
        self.bias = self.bias - lr * (np.sum(error))
        loss = {
            'step': len(self.losses) + 1,
            'train_loss': self.loss(X_train, Y_train[self.target_column])
        }
        if X_test is not None and Y_test is not None:
            loss['test_loss'] = self.loss(X_test, Y_test[self.target_column])
        self.losses.append(loss)

    def loss(self, X, Y):
        Y_pred = self.__call__(X)
        loss =  - (1 / len(X)) * np.sum(Y * np.log(Y_pred) + (1 - Y) * np.log(1 - Y_pred))
        return loss


class SortingHat:
    def __init__(self, n_features, lr):

        self.logregs = {target_class: LogReg(n_features, target_class, lr) for target_class in TARGET_CLASSES}
        self.learning_rate = lr
        self.parameters = {
            logreg_item[0]: [logreg_item[1].bias] + logreg_item[1].weights for logreg_item in self.logregs.items()
        }
        self.losses = []

    def __call__(self, X):
        return {target_class: self.logregs[target_class].__call__(X) for target_class in TARGET_CLASSES}

    def predict(self, X):
        return pd.DataFrame(self.__call__(X)).idxmax(axis=1)

    def train_step(self, X_train, Y_train, X_test=None, Y_test=None, lr=None):
        if lr == None:
            lr = self.learning_rate
        for logreg in self.logregs.values():
            logreg.train_step(X_train, Y_train, X_test, Y_test, lr)
        loss = {
            'step': len(self.losses) + 1,
            'train_loss': sum([logreg.losses[-1]['train_loss'] for logreg in self.logregs.values()])
        }
        if X_test is not None and Y_test is not None:
            loss['test_loss'] = sum([logreg.losses[-1]['test_loss'] for logreg in self.logregs.values()])
        self.losses.append(loss)
        self.parameters = {
            logreg_item[0]: [logreg_item[1].bias] + logreg_item[1].weights for logreg_item in self.logregs.items()
        }

    def save_model(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.parameters, f, ensure_ascii=False, indent=4)

    def load_model(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.parameters = json.load(f)
        for logreg in self.logregs.values():
            logreg_parameters = self.parameters[logreg.target_column]
            logreg.weights = logreg_parameters[1:]
            logreg.bias = logreg_parameters[0]
